"""Thermal reservoir (heat bath) for the ding-a-ling open system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq

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
# Thermostat event scheduling (task 6.4)
# ---------------------------------------------------------------------------


@dataclass(order=True)
class ThermostatEvent:
    """
    Scheduled thermostat event in the priority queue.

    Attributes:
        time:           Absolute simulation time of the event.
        particle_index: Index of the particle to thermostat.
        bath_index:     Which bath (0 = hot/left, 1 = cold/right).
    """
    time: float
    particle_index: int = -1
    bath_index: int = 0


class ThermostatScheduler:
    """
    Manages a priority queue of thermostat events for boundary particles.

    Each boundary particle has exactly one pending thermostat event at any
    time.  When an event is processed the particle's velocity is resampled
    via the appropriate HeatBath and the next event for that particle is
    immediately scheduled.
    """

    def __init__(
        self,
        hot_bath: HeatBath,
        cold_bath: HeatBath,
        left_indices: List[int],
        right_indices: List[int],
    ):
        """
        Args:
            hot_bath:      HeatBath for left-boundary particles.
            cold_bath:     HeatBath for right-boundary particles.
            left_indices:  Particle indices coupled to the hot bath.
            right_indices: Particle indices coupled to the cold bath.
        """
        self.baths: List[HeatBath] = [hot_bath, cold_bath]
        self.left_indices = list(left_indices)
        self.right_indices = list(right_indices)

        # Map each thermostated particle index to its bath index
        self._particle_bath: Dict[int, int] = {}
        for idx in self.left_indices:
            self._particle_bath[idx] = 0
        for idx in self.right_indices:
            self._particle_bath[idx] = 1

        self._event_queue: List[ThermostatEvent] = []
        self.current_time: float = 0.0

    def build_event_queue(self, current_time: float) -> None:
        """Schedule the first thermostat event for every boundary particle."""
        self.current_time = current_time
        self._event_queue = []

        for particle_idx, bath_idx in self._particle_bath.items():
            bath = self.baths[bath_idx]
            t_event = bath.next_event_time(current_time)
            heapq.heappush(
                self._event_queue,
                ThermostatEvent(time=t_event, particle_index=particle_idx,
                                bath_index=bath_idx),
            )

    def get_next_event(self) -> Optional[ThermostatEvent]:
        """Pop and return the earliest thermostat event, or None if empty."""
        if not self._event_queue:
            return None
        return heapq.heappop(self._event_queue)

    def peek_next_time(self) -> float:
        """Return the time of the next event without consuming it.

        Returns ``inf`` if the queue is empty.
        """
        if not self._event_queue:
            return float("inf")
        return self._event_queue[0].time

    def process_event(self, event: ThermostatEvent, chain: Chain) -> bool:
        """
        Apply the thermostat for *event* and schedule the next event.

        The particle's velocity is unconditionally resampled from the
        Maxwell-Boltzmann distribution of the corresponding bath (the
        Poisson waiting time already encodes the stochastic acceptance).

        Args:
            event: The thermostat event to process.
            chain: The particle chain (mutated in-place).

        Returns:
            True (the resampling always fires at a scheduled event time).
        """
        bath = self.baths[event.bath_index]
        particle = chain[event.particle_index]

        # Record energy exchange
        e_before = particle.kinetic_energy()
        particle.velocity = bath.sample_velocity(particle.mass)
        e_after = particle.kinetic_energy()
        bath._energy_exchanged += e_after - e_before
        bath._n_events += 1

        self.current_time = event.time

        # Schedule the next event for this particle
        t_next = bath.next_event_time(event.time)
        heapq.heappush(
            self._event_queue,
            ThermostatEvent(time=t_next, particle_index=event.particle_index,
                            bath_index=event.bath_index),
        )
        return True

    def reschedule_particle(self, particle_index: int, current_time: float) -> None:
        """
        Discard and reschedule the pending event for *particle_index*.

        Call this when a collision changes a boundary particle's state,
        invalidating its previously scheduled thermostat time.
        """
        if particle_index not in self._particle_bath:
            return

        # Remove the old event for this particle
        self._event_queue = [
            e for e in self._event_queue
            if e.particle_index != particle_index
        ]
        heapq.heapify(self._event_queue)

        # Push a fresh event
        bath_idx = self._particle_bath[particle_index]
        bath = self.baths[bath_idx]
        t_event = bath.next_event_time(current_time)
        heapq.heappush(
            self._event_queue,
            ThermostatEvent(time=t_event, particle_index=particle_index,
                            bath_index=bath_idx),
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
