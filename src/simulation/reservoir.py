"""Thermal reservoir (heat bath) for the ding-a-ling open system."""

from typing import List, Optional, Tuple

import numpy as np

from ..core.particle import Particle
from ..core.chain import Chain


class HeatBath:
    """
    Stochastic thermostat coupling a boundary particle to a thermal reservoir.

    Models the reservoir as a Poisson process with rate gamma: in a time
    interval dt the particle's velocity is resampled from the Maxwell-Boltzmann
    distribution with probability 1 - exp(-gamma * dt).

    Energy bookkeeping tracks the cumulative energy injected into (positive)
    or extracted from (negative) the particle, enabling reservoir power
    calculation as an alternative flux measure.
    """

    def __init__(
        self,
        temperature: float,
        coupling_rate: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            temperature:   Bath temperature T (in units where k_B = 1).
            coupling_rate: Poisson rate gamma (thermostat events per unit time).
            rng:           Optional numpy Generator for reproducibility.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if coupling_rate <= 0:
            raise ValueError(f"coupling_rate must be positive, got {coupling_rate}")

        self.temperature = temperature
        self.gamma = coupling_rate
        self._rng = rng if rng is not None else np.random.default_rng()

        self._energy_exchanged: float = 0.0
        self._n_events: int = 0

    # ------------------------------------------------------------------
    # Maxwell-Boltzmann sampling
    # ------------------------------------------------------------------

    def sample_velocity(self, mass: float) -> float:
        """
        Draw a velocity from the Maxwell-Boltzmann distribution.

        P(v) ∝ exp(-m v² / 2T),  i.e.  v ~ N(0, sqrt(T/m)).

        Args:
            mass: Particle mass.

        Returns:
            Sampled velocity.
        """
        sigma = np.sqrt(self.temperature / mass)
        return float(self._rng.normal(0.0, sigma))

    # ------------------------------------------------------------------
    # Stochastic thermostat
    # ------------------------------------------------------------------

    def apply_thermostat(self, particle: Particle, dt: float) -> bool:
        """
        Possibly resample the particle's velocity (Poisson thermostat).

        The resampling probability over interval dt is
            p = 1 - exp(-gamma * dt).

        When resampling occurs the particle's velocity is replaced with a
        fresh Maxwell-Boltzmann draw and the energy change is accumulated.

        Args:
            particle: Particle to thermostat (mutated in-place if fired).
            dt:       Elapsed time since last thermostat check.

        Returns:
            True if a resampling event occurred, False otherwise.
        """
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt}")

        prob = 1.0 - np.exp(-self.gamma * dt)
        if self._rng.random() < prob:
            e_before = particle.kinetic_energy()
            particle.velocity = self.sample_velocity(particle.mass)
            e_after = particle.kinetic_energy()
            self._energy_exchanged += e_after - e_before
            self._n_events += 1
            return True
        return False

    # ------------------------------------------------------------------
    # Event scheduling
    # ------------------------------------------------------------------

    def next_event_time(self, current_time: float) -> float:
        """
        Sample the absolute time of the next thermostat event.

        Waiting time is exponentially distributed: tau ~ Exp(gamma).

        Args:
            current_time: Current simulation time.

        Returns:
            Absolute time of the next thermostat event.
        """
        waiting = -np.log(self._rng.random()) / self.gamma
        return current_time + waiting

    # ------------------------------------------------------------------
    # Energy / statistics
    # ------------------------------------------------------------------

    @property
    def energy_exchanged(self) -> float:
        """Cumulative energy given to particles (negative = extracted)."""
        return self._energy_exchanged

    @property
    def n_events(self) -> int:
        """Total number of thermostat resampling events that have fired."""
        return self._n_events

    @property
    def power(self) -> float:
        """
        Instantaneous reservoir power estimate: energy_exchanged / n_events.

        Returns 0.0 if no events have occurred.
        """
        return self._energy_exchanged / self._n_events if self._n_events > 0 else 0.0

    def reset_statistics(self) -> None:
        """Zero the energy exchange counter and event count."""
        self._energy_exchanged = 0.0
        self._n_events = 0

    def __repr__(self) -> str:
        return (
            f"HeatBath(T={self.temperature}, gamma={self.gamma}, "
            f"n_events={self._n_events}, E_exchanged={self._energy_exchanged:.4f})"
        )


# ---------------------------------------------------------------------------
# Boundary particle identification (task 8.2)
# ---------------------------------------------------------------------------

def identify_boundary_particles(
    chain: Chain, n_boundary: int = 1
) -> Tuple[List[int], List[int]]:
    """
    Return the indices of left- and right-boundary particles.

    In an open system the outermost particles couple to the hot and cold
    heat baths respectively.

    Args:
        chain:      The particle chain.
        n_boundary: Number of particles on each end to treat as boundary.

    Returns:
        (left_indices, right_indices) — lists of particle indices.

    Raises:
        ValueError: If n_boundary is not positive or the chain is too short.
    """
    n = len(chain)
    if n_boundary < 1:
        raise ValueError(f"n_boundary must be >= 1, got {n_boundary}")
    if 2 * n_boundary > n:
        raise ValueError(
            f"Chain length {n} too short for n_boundary={n_boundary} on each end"
        )

    left = list(range(n_boundary))
    right = list(range(n - n_boundary, n))
    return left, right
