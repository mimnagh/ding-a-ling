"""Thermal conductivity extraction from open-system simulation results."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from ..core.chain import Chain
from ..simulation.engine import SimulationResult
from ..simulation.flux import FluxMeter


@dataclass
class TransportResult:
    """Thermal transport observables extracted from a simulation run."""

    kappa: float
    kappa_error: float
    flux: float
    flux_error: float
    gradient: float
    gradient_error: float
    n_particles: int
    positions: List[float]
    temperatures: List[float]


class TransportAnalyzer:
    """
    Extract thermal conductivity κ from open-system simulation results.

    Pipeline:
    1. Time-average the temperature profile over a steady-state window.
    2. Exclude boundary-layer particles from the gradient fit.
    3. Fit a linear temperature gradient T(x) = a·x + b via OLS.
    4. Obtain the steady-state heat flux from the FluxMeter.
    5. Compute κ = −J / (dT/dx) with bootstrap uncertainty.
    """

    def __init__(
        self,
        n_boundary_exclude: int = 2,
        t_start: Optional[float] = None,
        n_bootstrap: int = 200,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            n_boundary_exclude: Number of particles to drop from each end
                before fitting the temperature gradient.
            t_start: Start of the averaging window for both temperature and
                flux. If None, defaults to halfway through the simulation.
            n_bootstrap: Number of bootstrap resamples used for κ error.
            rng: Optional random-number generator for reproducibility.
        """
        if n_boundary_exclude < 0:
            raise ValueError(f"n_boundary_exclude must be >= 0, got {n_boundary_exclude}")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

        self.n_boundary_exclude = n_boundary_exclude
        self.t_start = t_start
        self.n_bootstrap = n_bootstrap
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public helpers (tasks 9.2 – 9.5)
    # ------------------------------------------------------------------

    def exclude_boundary_layers(
        self,
        positions: List[float],
        temperatures: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drop the outermost n_boundary_exclude particles from each end.

        Args:
            positions:    Equilibrium positions of all particles.
            temperatures: Time-averaged temperatures of all particles.

        Returns:
            (interior_positions, interior_temperatures) as numpy arrays.

        Raises:
            ValueError: If excluding boundary layers leaves fewer than 2 points.
        """
        n = len(positions)
        lo = self.n_boundary_exclude
        hi = n - self.n_boundary_exclude

        if hi - lo < 2:
            raise ValueError(
                f"Excluding {self.n_boundary_exclude} particles from each end "
                f"of a chain of length {n} leaves fewer than 2 interior points."
            )

        x = np.asarray(positions[lo:hi], dtype=float)
        T = np.asarray(temperatures[lo:hi], dtype=float)
        return x, T

    def fit_gradient(
        self,
        positions: List[float],
        temperatures: List[float],
    ) -> Tuple[float, float, float]:
        """
        Fit a linear temperature profile T(x) = slope·x + intercept.

        Boundary layers are excluded according to self.n_boundary_exclude.
        Uses scipy.stats.linregress (ordinary least squares).

        Args:
            positions:    Equilibrium positions of all particles.
            temperatures: Time-averaged temperatures of all particles.

        Returns:
            (slope, intercept, slope_stderr)
            where slope is dT/dx and slope_stderr is its standard error.
        """
        x, T = self.exclude_boundary_layers(positions, temperatures)
        result = stats.linregress(x, T)
        return float(result.slope), float(result.intercept), float(result.stderr)

    def calculate_conductivity(self, flux: float, gradient: float) -> float:
        """
        Fourier's law: κ = −J / (dT/dx).

        Args:
            flux:     Steady-state heat flux J (positive = leftward to rightward).
            gradient: Temperature gradient dT/dx (negative in a hot-left system).

        Returns:
            Thermal conductivity κ (always non-negative for physical systems).

        Raises:
            ValueError: If gradient is zero (undefined conductivity).
        """
        if gradient == 0.0:
            raise ValueError("Temperature gradient is zero; κ is undefined.")
        return -flux / gradient

    def estimate_kappa_error(
        self,
        flux_meter: FluxMeter,
        t_start: float,
        t_end: float,
        positions: List[float],
        temperature_snapshots: List[Tuple[float, List[float]]],
    ) -> float:
        """
        Bootstrap estimate of the standard error in κ.

        Resamples flux crossings (with replacement) to get a distribution
        of flux values, and uses the gradient standard error from the linear
        fit to propagate uncertainty.

        Args:
            flux_meter:            FluxMeter from the simulation result.
            t_start:               Start of the steady-state averaging window.
            t_end:                 End of the averaging window.
            positions:             Equilibrium positions of all particles.
            temperature_snapshots: All (time, temps) entries from the result.

        Returns:
            Bootstrap standard error on κ.
        """
        # --- flux bootstrap ---
        crossings = flux_meter.crossings_in_window(t_start, t_end)
        dt = t_end - t_start

        if len(crossings) < 2 or dt <= 0:
            return float("nan")

        contributions = np.array([r.flux_contribution for r in crossings])

        bootstrap_kappas = []
        for _ in range(self.n_bootstrap):
            sample = self._rng.choice(contributions, size=len(contributions), replace=True)
            J_boot = float(sample.sum() / dt)

            # Re-fit gradient on temperature snapshots in the window
            window_temps = [
                temps for t, temps in temperature_snapshots
                if t_start <= t <= t_end
            ]
            if not window_temps:
                continue

            n_particles = len(window_temps[0])
            avg_T = [
                sum(s[i] for s in window_temps) / len(window_temps)
                for i in range(n_particles)
            ]

            try:
                slope, _, _ = self.fit_gradient(positions, avg_T)
                if slope == 0.0:
                    continue
                bootstrap_kappas.append(-J_boot / slope)
            except (ValueError, Exception):
                continue

        if len(bootstrap_kappas) < 2:
            return float("nan")

        return float(np.std(bootstrap_kappas, ddof=1))

    # ------------------------------------------------------------------
    # Main analysis entry point (task 9.1)
    # ------------------------------------------------------------------

    def analyze(self, result: SimulationResult, chain: Chain) -> TransportResult:
        """
        Extract thermal conductivity from a completed simulation.

        Args:
            result: Output of EventDrivenSimulator.run().
            chain:  The chain used in the simulation.

        Returns:
            TransportResult with κ, uncertainties, and diagnostic data.

        Raises:
            ValueError: If the result has no flux meter (closed system).
        """
        if result.flux_meter is None:
            raise ValueError(
                "TransportAnalyzer requires an open-system SimulationResult "
                "with a FluxMeter. Run with BoundaryCondition.OPEN."
            )

        t_start = self.t_start if self.t_start is not None else result.time / 2.0
        t_end = result.time

        # --- temperature profile ---
        positions = result.particle_positions(chain)
        avg_temperatures = result.time_averaged_temperature(t_start=t_start, t_end=t_end)

        # --- gradient fit ---
        slope, intercept, slope_stderr = self.fit_gradient(positions, avg_temperatures)

        # --- flux ---
        dt = t_end - t_start
        if dt <= 0:
            raise ValueError(
                f"Averaging window has zero duration (t_start={t_start}, t_end={t_end})."
            )
        flux = result.flux_meter.time_averaged_flux(t_start, t_end)

        # Flux uncertainty from Poisson counting noise: σ_J ≈ std(contributions)/sqrt(N)/dt
        crossings = result.flux_meter.crossings_in_window(t_start, t_end)
        if len(crossings) >= 2:
            contribs = np.array([r.flux_contribution for r in crossings])
            flux_error = float(np.std(contribs, ddof=1) / (np.sqrt(len(contribs)) * dt))
        else:
            flux_error = float("nan")

        # --- conductivity ---
        kappa = self.calculate_conductivity(flux, slope)

        # --- κ error via bootstrap ---
        kappa_error = self.estimate_kappa_error(
            result.flux_meter,
            t_start,
            t_end,
            positions,
            result.temperature_profile,
        )

        return TransportResult(
            kappa=kappa,
            kappa_error=kappa_error,
            flux=flux,
            flux_error=flux_error,
            gradient=slope,
            gradient_error=slope_stderr,
            n_particles=len(chain),
            positions=positions,
            temperatures=avg_temperatures,
        )
