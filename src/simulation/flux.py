"""Heat flux measurement for the ding-a-ling thermal transport model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..core.particle import Particle, ParticleType


@dataclass
class CrossingRecord:
    """Record of a single particle crossing the reference plane."""

    time: float
    particle_index: int
    direction: int  # +1 for rightward, -1 for leftward
    energy: float   # Total energy of particle at crossing time

    @property
    def flux_contribution(self) -> float:
        """Signed energy flux: direction * energy."""
        return self.direction * self.energy


class FluxMeter:
    """
    Measures energy flow across a fixed reference plane.

    Tracks particle crossings of position x_ref and computes the
    time-averaged energy flux J = (1/Δt) * Σ_i (direction_i * energy_i).

    Positive flux means net energy flows to the right (+x direction).
    """

    def __init__(self, position: float, name: str = ""):
        """
        Args:
            position: x-coordinate of the reference plane.
            name: Optional label (e.g. "left", "center") for diagnostics.
        """
        self.x_ref = position
        self.name = name
        self._crossings: List[CrossingRecord] = []

    @property
    def flux_history(self) -> List[CrossingRecord]:
        """All recorded crossing events, in chronological order."""
        return self._crossings

    def record_crossing(
        self, particle: Particle, time: float, direction: int
    ) -> CrossingRecord:
        """
        Record that a particle has crossed the reference plane.

        Args:
            particle: Particle that crossed (at its state at crossing time).
            time: Simulation time of the crossing.
            direction: +1 for rightward, -1 for leftward.

        Returns:
            The CrossingRecord that was appended.
        """
        record = CrossingRecord(
            time=time,
            particle_index=particle.index,
            direction=direction,
            energy=particle.total_energy(),
        )
        self._crossings.append(record)
        return record

    def check_crossing(
        self,
        particle: Particle,
        t_start: float,
        t_end: float,
    ) -> Optional[Tuple[float, int]]:
        """
        Check whether a particle crosses x_ref during (t_start, t_end].

        The particle is assumed to be at its current state at t_start.
        For free particles the crossing time is solved analytically.
        For harmonic particles Brent's method finds the first root.

        Args:
            particle: Particle at its state corresponding to t_start.
            t_start: Start of the interval (exclusive).
            t_end: End of the interval (inclusive).

        Returns:
            (absolute_crossing_time, direction) if a crossing occurs,
            None otherwise. direction is +1 (right) or -1 (left).
        """
        if particle.particle_type == ParticleType.FREE:
            return self._check_crossing_free(particle, t_start, t_end)
        else:
            return self._check_crossing_harmonic(particle, t_start, t_end)

    def _check_crossing_free(
        self,
        particle: Particle,
        t_start: float,
        t_end: float,
    ) -> Optional[Tuple[float, int]]:
        """Crossing detection for a ballistically moving free particle."""
        v = particle.velocity
        if abs(v) < 1e-15:
            return None

        # x(dt) = position + v * dt  =>  dt_cross = (x_ref - position) / v
        dt_cross = (self.x_ref - particle.position) / v
        if dt_cross <= 0 or dt_cross > (t_end - t_start):
            return None

        direction = 1 if v > 0 else -1
        return (t_start + dt_cross, direction)

    def _check_crossing_harmonic(
        self,
        particle: Particle,
        t_start: float,
        t_end: float,
    ) -> Optional[Tuple[float, int]]:
        """
        Crossing detection for a harmonically bound particle.

        Finds the first root of x(dt) - x_ref = 0 in (0, t_end - t_start]
        using Brent's method after identifying a sign change.
        """
        from scipy.optimize import brentq

        omega = np.sqrt(particle.spring_constant / particle.mass)
        x0 = particle.position - particle.equilibrium_pos
        v0 = particle.velocity
        x_eq = particle.equilibrium_pos

        def x_relative(dt: float) -> float:
            """Position relative to x_ref at elapsed time dt."""
            return (
                x_eq
                + x0 * np.cos(omega * dt)
                + (v0 / omega) * np.sin(omega * dt)
                - self.x_ref
            )

        dt_max = t_end - t_start

        # Sample densely enough to catch oscillations (≥4 points per half-period)
        n_samples = max(20, int(dt_max * omega / np.pi) * 4 + 4)
        dt_vals = np.linspace(0.0, dt_max, n_samples + 1)
        f_vals = np.array([x_relative(dt) for dt in dt_vals])

        # Find first sign change; skip dt=0 (particle is at t_start, not yet crossing)
        for i in range(1, len(dt_vals) - 1):
            if f_vals[i] * f_vals[i + 1] < 0:
                dt_cross = brentq(
                    x_relative, dt_vals[i], dt_vals[i + 1], xtol=1e-12
                )
                v_cross = (
                    -x0 * omega * np.sin(omega * dt_cross)
                    + v0 * np.cos(omega * dt_cross)
                )
                direction = 1 if v_cross > 0 else -1
                return (t_start + dt_cross, direction)

        return None

    def time_averaged_flux(self, t_start: float, t_end: float) -> float:
        """
        Compute the time-averaged energy flux over [t_start, t_end].

        J = (1/Δt) * Σ_i (direction_i * energy_i)

        where the sum is over all crossings with t_start <= time <= t_end.

        Args:
            t_start: Start of the averaging window.
            t_end: End of the averaging window.

        Returns:
            Time-averaged flux. Positive = net rightward energy flow.
            Returns 0.0 for an empty window or zero duration.
        """
        dt = t_end - t_start
        if dt <= 0:
            return 0.0

        total = sum(
            r.flux_contribution
            for r in self._crossings
            if t_start <= r.time <= t_end
        )
        return total / dt

    def cumulative_flux(self, t: float) -> float:
        """
        Return cumulative (time-integrated) energy flux up to time t.

        Σ_i (direction_i * energy_i)  for all crossings with time <= t.
        """
        return sum(r.flux_contribution for r in self._crossings if r.time <= t)

    def crossings_in_window(
        self, t_start: float, t_end: float
    ) -> List[CrossingRecord]:
        """Return crossings that occurred within [t_start, t_end]."""
        return [r for r in self._crossings if t_start <= r.time <= t_end]

    def reset(self) -> None:
        """Clear all recorded crossings."""
        self._crossings.clear()

    def __repr__(self) -> str:
        label = f" ({self.name})" if self.name else ""
        return (
            f"FluxMeter{label}(x_ref={self.x_ref:.3f}, "
            f"n_crossings={len(self._crossings)})"
        )
