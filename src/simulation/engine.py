"""Event-driven simulation engine for the ding-a-ling model."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..core.chain import Chain, BoundaryCondition
from ..core.collision import CollisionDetector, resolve_collision
from ..core.particle import ParticleType
from .flux import FluxMeter
from .reservoir import HeatBath, ThermostatScheduler, identify_boundary_particles
from .steady_state import SteadyStateDetector


@dataclass
class SimulationConfig:
    """Configuration for an event-driven simulation run."""

    total_time: float = 100.0

    # Open-system parameters (used only when chain boundary is OPEN)
    T_hot: float = 2.0
    T_cold: float = 0.5
    gamma: float = 1.0
    n_boundary: int = 1

    # Observable recording
    measurement_interval: float = 1.0
    flux_position: Optional[float] = None  # default: centre of chain

    # Reproducibility
    seed: Optional[int] = None

    # Steady-state detection (open systems only)
    ss_window_size: int = 100
    ss_threshold: float = 0.01


@dataclass
class SimulationResult:
    """Collected observables from a simulation run."""

    time: float
    n_collisions: int
    n_thermostat_events: int
    energy_history: List[Tuple[float, float]]
    temperature_profile: List[Tuple[float, List[float]]]
    config: SimulationConfig

    # Detailed data carriers (None for closed systems)
    flux_meter: Optional[FluxMeter] = None
    hot_bath: Optional[HeatBath] = None
    cold_bath: Optional[HeatBath] = None

    # Steady-state detection
    is_steady_state: bool = False
    steady_state_time: Optional[float] = None

    def time_averaged_temperature(
        self, t_start: float = 0.0, t_end: Optional[float] = None,
    ) -> List[float]:
        """Average temperature profile over a time window.

        Args:
            t_start: Start of averaging window (inclusive).
            t_end:   End of averaging window (inclusive). Defaults to last snapshot.

        Returns:
            List of time-averaged temperatures, one per particle.
        """
        if t_end is None:
            t_end = self.time

        snapshots = [
            temps for t, temps in self.temperature_profile
            if t_start <= t <= t_end
        ]
        if not snapshots:
            return []

        n_particles = len(snapshots[0])
        return [
            sum(s[i] for s in snapshots) / len(snapshots)
            for i in range(n_particles)
        ]

    def particle_positions(self, chain: 'Chain') -> List[float]:
        """Return equilibrium positions of all particles.

        Useful for plotting temperature vs spatial position.

        Args:
            chain: The chain used in the simulation.

        Returns:
            List of equilibrium positions, one per particle.
        """
        return [p.equilibrium_pos for p in chain.particles]


class EventDrivenSimulator:
    """
    Unified simulation engine for closed and open ding-a-ling chains.

    For closed (PERIODIC) systems the engine processes only collision events.
    For open (OPEN) systems it additionally schedules thermostat events from
    hot/cold heat baths on the boundary particles and measures heat flux
    across a reference plane.
    """

    def __init__(self, chain: Chain, config: SimulationConfig):
        self.chain = chain
        self.config = config
        self.time: float = 0.0

        self._is_open = chain.config.boundary == BoundaryCondition.OPEN

        # Collision detection (always needed)
        self._collision_detector = CollisionDetector(chain)

        # Open-system components
        self._thermostat_scheduler: Optional[ThermostatScheduler] = None
        self._flux_meter: Optional[FluxMeter] = None
        self.hot_bath: Optional[HeatBath] = None
        self.cold_bath: Optional[HeatBath] = None

        if self._is_open:
            rng = np.random.default_rng(config.seed)
            # Create two independent child RNGs for each bath
            seeds = rng.integers(0, 2**63, size=2)
            self.hot_bath = HeatBath(
                config.T_hot, config.gamma,
                rng=np.random.default_rng(seeds[0]),
            )
            self.cold_bath = HeatBath(
                config.T_cold, config.gamma,
                rng=np.random.default_rng(seeds[1]),
            )

            left, right = identify_boundary_particles(chain, config.n_boundary)
            self._thermostat_scheduler = ThermostatScheduler(
                self.hot_bath, self.cold_bath, left, right,
            )
            self._boundary_indices = set(left + right)

            flux_pos = config.flux_position
            if flux_pos is None:
                flux_pos = (chain[0].position + chain[-1].position) / 2.0
            self._flux_meter = FluxMeter(position=flux_pos, name="center")

        # Steady-state detector (open systems only)
        self._ss_detector: Optional[SteadyStateDetector] = None
        if self._is_open:
            self._ss_detector = SteadyStateDetector(
                window_size=config.ss_window_size,
                threshold=config.ss_threshold,
            )

        # Observable accumulators
        self._energy_history: List[Tuple[float, float]] = []
        self._temperature_profile: List[Tuple[float, List[float]]] = []
        self._n_collisions = 0
        self._n_thermostat_events = 0
        self._last_measurement_time = -np.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute the simulation and return collected observables."""
        # Build collision heap
        self._collision_detector.build_event_queue()

        # Thermostat queue (if open system)
        if self._thermostat_scheduler is not None:
            self._thermostat_scheduler.build_event_queue(self.time)

        # Record initial state
        self._measure_observables()

        while self.time < self.config.total_time:
            # --- peek at next event times (without consuming) ---
            cq = self._collision_detector.event_queue
            t_collision = cq[0].time if cq else np.inf

            t_thermostat = np.inf
            if self._thermostat_scheduler is not None:
                t_thermostat = self._thermostat_scheduler.peek_next_time()

            t_event = min(t_collision, t_thermostat)

            # No more events possible
            if t_event == np.inf:
                dt = self.config.total_time - self.time
                if dt > 0:
                    self._check_flux_crossings(self.time, self.time + dt)
                    self._evolve_all(dt)
                    self.time = self.config.total_time
                    self._collision_detector.current_time = self.time
                break

            # Clamp to total_time
            if t_event > self.config.total_time:
                dt = self.config.total_time - self.time
                self._check_flux_crossings(self.time, self.time + dt)
                self._evolve_all(dt)
                self.time = self.config.total_time
                self._collision_detector.current_time = self.time
                break

            dt = t_event - self.time

            # --- flux crossings (before evolution, particles at t_start) ---
            self._check_flux_crossings(self.time, self.time + dt)

            # --- evolve all particles to event time ---
            self._evolve_all(dt)
            self.time = t_event
            self._collision_detector.current_time = t_event

            # --- resolve event ---
            is_collision = (t_collision <= t_thermostat)

            if is_collision:
                # NOW pop the collision from the heap
                collision_event = self._collision_detector.get_next_event()
                p_i = self.chain[collision_event.particle_i]
                p_j = self.chain[collision_event.particle_j]
                resolve_collision(p_i, p_j)
                self._collision_detector.update_events_for_particles(
                    [collision_event.particle_i, collision_event.particle_j]
                )
                self._n_collisions += 1

                # If a boundary particle was involved, reschedule thermostat
                if self._thermostat_scheduler is not None:
                    for idx in (collision_event.particle_i, collision_event.particle_j):
                        if idx in self._boundary_indices:
                            self._thermostat_scheduler.reschedule_particle(
                                idx, self.time
                            )
            else:
                # Thermostat event — collision heap untouched
                thermo_event = self._thermostat_scheduler.get_next_event()
                if thermo_event is not None:
                    self._thermostat_scheduler.process_event(
                        thermo_event, self.chain
                    )
                    self._n_thermostat_events += 1
                    # Velocity changed — update collision heap for this particle
                    self._collision_detector.update_events_for_particles(
                        [thermo_event.particle_index]
                    )

            # --- periodic observable measurement ---
            if self.time - self._last_measurement_time >= self.config.measurement_interval:
                self._measure_observables()

        # Final measurement
        self._measure_observables()

        ss_reached = self._ss_detector.is_steady_state if self._ss_detector else False
        ss_time = self._ss_detector.steady_state_time if self._ss_detector else None

        return SimulationResult(
            time=self.time,
            n_collisions=self._n_collisions,
            n_thermostat_events=self._n_thermostat_events,
            energy_history=self._energy_history,
            temperature_profile=self._temperature_profile,
            config=self.config,
            flux_meter=self._flux_meter,
            hot_bath=self.hot_bath,
            cold_bath=self.cold_bath,
            is_steady_state=ss_reached,
            steady_state_time=ss_time,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evolve_all(self, dt: float) -> None:
        """Evolve every particle by *dt* using its analytic solution."""
        if dt <= 0:
            return
        for p in self.chain.particles:
            if p.particle_type == ParticleType.FREE:
                p.evolve_free(dt)
            else:
                p.evolve_harmonic(dt)

    def _check_flux_crossings(self, t_start: float, t_end: float) -> None:
        """Record any particle crossings of the flux meter reference plane."""
        if self._flux_meter is None:
            return
        for p in self.chain.particles:
            result = self._flux_meter.check_crossing(p, t_start, t_end)
            if result is not None:
                crossing_time, direction = result
                self._flux_meter.record_crossing(p, crossing_time, direction)

    def _measure_observables(self) -> None:
        """Append current energy and temperature profile to history."""
        self._energy_history.append((self.time, self.chain.total_energy()))
        temps = [
            self.chain.local_temperature(i)
            for i in range(len(self.chain))
        ]
        self._temperature_profile.append((self.time, temps))

        # Feed flux to steady-state detector
        if self._ss_detector is not None and self._flux_meter is not None:
            t_start = self._last_measurement_time if self._last_measurement_time >= 0 else 0.0
            flux = self._flux_meter.time_averaged_flux(t_start, self.time)
            self._ss_detector.add_sample(flux, self.time)

        self._last_measurement_time = self.time
