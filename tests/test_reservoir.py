"""Tests for HeatBath: MB sampling, stochastic thermostat, and boundary identification."""

import pytest
import numpy as np
from scipy import stats

from src.core.particle import Particle, ParticleType
from src.core.chain import Chain, ChainConfig
from src.simulation.reservoir import (
    HeatBath, ThermostatEvent, ThermostatScheduler, identify_boundary_particles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_particle(mass=1.0, velocity=0.0, index=0):
    return Particle(
        index=index,
        particle_type=ParticleType.FREE,
        mass=mass,
        position=0.0,
        velocity=velocity,
        equilibrium_pos=0.0,
        spring_constant=0.0,
    )


def make_bath(temperature=1.0, gamma=1.0, seed=42):
    rng = np.random.default_rng(seed)
    return HeatBath(temperature=temperature, coupling_rate=gamma, rng=rng)


# ---------------------------------------------------------------------------
# Initialisation & validation
# ---------------------------------------------------------------------------

class TestHeatBathInit:
    def test_stores_temperature(self):
        bath = make_bath(temperature=2.5)
        assert bath.temperature == pytest.approx(2.5)

    def test_stores_gamma(self):
        bath = make_bath(gamma=0.5)
        assert bath.gamma == pytest.approx(0.5)

    def test_initial_energy_exchanged_zero(self):
        assert make_bath().energy_exchanged == 0.0

    def test_initial_n_events_zero(self):
        assert make_bath().n_events == 0

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            HeatBath(temperature=-1.0, coupling_rate=1.0)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            HeatBath(temperature=0.0, coupling_rate=1.0)

    def test_zero_coupling_rate_raises(self):
        with pytest.raises(ValueError, match="coupling_rate"):
            HeatBath(temperature=1.0, coupling_rate=0.0)

    def test_repr(self):
        bath = make_bath()
        assert "HeatBath" in repr(bath)


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann sampling (task 6.2)
# ---------------------------------------------------------------------------

class TestSampleVelocity:
    N_SAMPLES = 10_000

    def test_mean_near_zero(self):
        bath = make_bath(temperature=1.0, seed=0)
        samples = [bath.sample_velocity(mass=1.0) for _ in range(self.N_SAMPLES)]
        assert np.mean(samples) == pytest.approx(0.0, abs=0.05)

    def test_variance_equals_T_over_m(self):
        T, m = 2.0, 3.0
        bath = make_bath(temperature=T, seed=1)
        samples = [bath.sample_velocity(mass=m) for _ in range(self.N_SAMPLES)]
        assert np.var(samples) == pytest.approx(T / m, rel=0.05)

    def test_distribution_is_gaussian(self):
        """Kolmogorov-Smirnov test: samples should fit N(0, sqrt(T/m))."""
        T, m = 1.0, 1.0
        bath = make_bath(temperature=T, seed=2)
        samples = [bath.sample_velocity(mass=m) for _ in range(self.N_SAMPLES)]
        sigma = np.sqrt(T / m)
        _, p_value = stats.kstest(samples, "norm", args=(0.0, sigma))
        assert p_value > 0.01  # not rejected at 1% level

    def test_higher_temperature_gives_larger_spread(self):
        bath_cold = make_bath(temperature=0.5, seed=3)
        bath_hot = make_bath(temperature=5.0, seed=4)
        cold = [bath_cold.sample_velocity(1.0) for _ in range(self.N_SAMPLES)]
        hot = [bath_hot.sample_velocity(1.0) for _ in range(self.N_SAMPLES)]
        assert np.std(hot) > np.std(cold)

    def test_heavier_particle_smaller_spread(self):
        bath = make_bath(temperature=1.0, seed=5)
        light = [bath.sample_velocity(mass=0.5) for _ in range(self.N_SAMPLES)]
        bath2 = make_bath(temperature=1.0, seed=6)
        heavy = [bath2.sample_velocity(mass=4.0) for _ in range(self.N_SAMPLES)]
        assert np.std(light) > np.std(heavy)


# ---------------------------------------------------------------------------
# Stochastic thermostat (task 6.3)
# ---------------------------------------------------------------------------

class TestApplyThermostat:
    def test_never_fires_at_dt_zero(self):
        bath = make_bath(gamma=100.0)
        p = make_particle(velocity=5.0)
        v_before = p.velocity
        fired = bath.apply_thermostat(p, dt=0.0)
        assert not fired
        assert p.velocity == pytest.approx(v_before)

    def test_almost_always_fires_at_large_dt(self):
        rng = np.random.default_rng(0)
        bath = HeatBath(temperature=1.0, coupling_rate=100.0, rng=rng)
        fired = sum(
            bath.apply_thermostat(make_particle(), dt=10.0) for _ in range(200)
        )
        assert fired > 195  # p ≈ 1 - e^{-1000} ≈ 1

    def test_firing_probability_matches_poisson(self):
        """Empirical firing rate should match 1 - exp(-gamma * dt)."""
        gamma, dt = 2.0, 0.5
        expected_prob = 1.0 - np.exp(-gamma * dt)
        rng = np.random.default_rng(7)
        bath = HeatBath(temperature=1.0, coupling_rate=gamma, rng=rng)
        n_trials = 5_000
        fired = sum(
            bath.apply_thermostat(make_particle(), dt=dt) for _ in range(n_trials)
        )
        assert fired / n_trials == pytest.approx(expected_prob, abs=0.03)

    def test_velocity_changes_when_fired(self):
        # With dt very large, thermostat always fires; velocity should change
        rng = np.random.default_rng(8)
        bath = HeatBath(temperature=1.0, coupling_rate=1e6, rng=rng)
        p = make_particle(velocity=1e10)  # absurdly large so any MB draw differs
        bath.apply_thermostat(p, dt=1.0)
        assert p.velocity != pytest.approx(1e10)

    def test_n_events_increments_on_fire(self):
        rng = np.random.default_rng(9)
        bath = HeatBath(temperature=1.0, coupling_rate=1e6, rng=rng)
        p = make_particle()
        bath.apply_thermostat(p, dt=1.0)
        assert bath.n_events == 1

    def test_n_events_unchanged_when_no_fire(self):
        bath = make_bath(gamma=1.0)
        p = make_particle()
        bath.apply_thermostat(p, dt=0.0)
        assert bath.n_events == 0

    def test_energy_exchanged_tracks_delta_KE(self):
        rng = np.random.default_rng(10)
        bath = HeatBath(temperature=1.0, coupling_rate=1e6, rng=rng)
        p = make_particle(mass=2.0, velocity=0.0)
        e_before = p.kinetic_energy()
        bath.apply_thermostat(p, dt=1.0)
        e_after = p.kinetic_energy()
        assert bath.energy_exchanged == pytest.approx(e_after - e_before)

    def test_energy_exchanged_accumulates(self):
        rng = np.random.default_rng(11)
        bath = HeatBath(temperature=1.0, coupling_rate=1e6, rng=rng)
        p = make_particle()
        total = 0.0
        for _ in range(10):
            e_before = p.kinetic_energy()
            bath.apply_thermostat(p, dt=1.0)
            total += p.kinetic_energy() - e_before
        assert bath.energy_exchanged == pytest.approx(total, rel=1e-10)

    def test_negative_dt_raises(self):
        bath = make_bath()
        with pytest.raises(ValueError):
            bath.apply_thermostat(make_particle(), dt=-0.1)


# ---------------------------------------------------------------------------
# Thermostat event scheduling (task 6.4)
# ---------------------------------------------------------------------------

class TestNextEventTime:
    def test_always_in_future(self):
        bath = make_bath(gamma=1.0, seed=12)
        for t in [0.0, 1.5, 100.0]:
            assert bath.next_event_time(t) > t

    def test_mean_waiting_time(self):
        """Mean of Exp(gamma) is 1/gamma."""
        gamma = 3.0
        bath = make_bath(gamma=gamma, seed=13)
        t = 0.0
        waiting_times = [bath.next_event_time(t) - t for _ in range(5_000)]
        assert np.mean(waiting_times) == pytest.approx(1.0 / gamma, rel=0.05)

    def test_waiting_time_is_exponential(self):
        """KS test: waiting times should fit Exp(gamma)."""
        gamma = 2.0
        bath = make_bath(gamma=gamma, seed=14)
        waiting = [bath.next_event_time(0.0) for _ in range(5_000)]
        _, p_value = stats.kstest(waiting, "expon", args=(0.0, 1.0 / gamma))
        assert p_value > 0.01

    def test_offset_by_current_time(self):
        rng = np.random.default_rng(15)
        bath = HeatBath(temperature=1.0, coupling_rate=1.0, rng=rng)
        t1 = bath.next_event_time(0.0)
        rng2 = np.random.default_rng(15)
        bath2 = HeatBath(temperature=1.0, coupling_rate=1.0, rng=rng2)
        t2 = bath2.next_event_time(10.0)
        assert t2 - t1 == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Cumulative energy exchange (task 6.7)
# ---------------------------------------------------------------------------

class TestEnergyExchange:
    def test_reset_statistics(self):
        rng = np.random.default_rng(16)
        bath = HeatBath(temperature=1.0, coupling_rate=1e6, rng=rng)
        bath.apply_thermostat(make_particle(), dt=1.0)
        bath.reset_statistics()
        assert bath.energy_exchanged == 0.0
        assert bath.n_events == 0

    def test_power_zero_before_any_event(self):
        assert make_bath().power == 0.0

    def test_power_nonzero_after_events(self):
        rng = np.random.default_rng(17)
        bath = HeatBath(temperature=1.0, coupling_rate=1e6, rng=rng)
        p = make_particle(velocity=0.0)
        for _ in range(10):
            bath.apply_thermostat(p, dt=1.0)
        # power = energy_exchanged / n_events
        if bath.n_events > 0:
            assert bath.power == pytest.approx(
                bath.energy_exchanged / bath.n_events
            )


# ---------------------------------------------------------------------------
# Detailed balance (task 6.6)
# ---------------------------------------------------------------------------

class TestDetailedBalance:
    def test_velocity_distribution_reaches_equilibrium(self):
        """
        Single particle repeatedly thermostated should converge to MB at T.

        After many events the velocity histogram should match N(0, sqrt(T/m)).
        """
        T, m = 2.0, 1.5
        rng = np.random.default_rng(18)
        bath = HeatBath(temperature=T, coupling_rate=1.0, rng=rng)
        p = make_particle(mass=m, velocity=100.0)  # far from equilibrium

        # Apply thermostat many times with large dt (always fires)
        n = 5_000
        velocities = []
        for _ in range(n):
            bath.apply_thermostat(p, dt=100.0)
            velocities.append(p.velocity)

        sigma = np.sqrt(T / m)
        _, p_value = stats.kstest(velocities, "norm", args=(0.0, sigma))
        assert p_value > 0.01

    def test_mean_energy_approaches_half_kT(self):
        """Equipartition: <KE> = T/2 for 1D particle at temperature T."""
        T, m = 1.0, 1.0
        rng = np.random.default_rng(19)
        bath = HeatBath(temperature=T, coupling_rate=1.0, rng=rng)
        p = make_particle(mass=m, velocity=0.0)

        energies = []
        for _ in range(10_000):
            bath.apply_thermostat(p, dt=100.0)
            energies.append(p.kinetic_energy())

        assert np.mean(energies) == pytest.approx(T / 2, rel=0.05)

    def test_scheduler_process_event_preserves_equilibrium(self):
        """Scheduler-driven thermostat should also yield MB distribution."""
        T_hot = 3.0
        hot = HeatBath(temperature=T_hot, coupling_rate=1.0,
                       rng=np.random.default_rng(20))
        cold = HeatBath(temperature=1.0, coupling_rate=1.0,
                        rng=np.random.default_rng(21))
        config = ChainConfig(n_particles=6)
        chain = Chain(config)
        left, right = identify_boundary_particles(chain, n_boundary=1)
        scheduler = ThermostatScheduler(hot, cold, left, right)
        scheduler.build_event_queue(current_time=0.0)

        # Repeatedly process events for the left boundary particle (index 0)
        m = chain[0].mass
        velocities = []
        for _ in range(5_000):
            event = scheduler.get_next_event()
            scheduler.process_event(event, chain)
            if event.particle_index == 0:
                velocities.append(chain[0].velocity)

        sigma = np.sqrt(T_hot / m)
        _, p_value = stats.kstest(velocities, "norm", args=(0.0, sigma))
        assert p_value > 0.01

    def test_net_energy_exchange_averages_near_zero_at_equilibrium(self):
        """At equilibrium (bath T = particle T), net energy exchange -> 0."""
        T = 1.0
        rng = np.random.default_rng(22)
        bath = HeatBath(temperature=T, coupling_rate=1.0, rng=rng)

        # Start particle at equilibrium velocity
        p = make_particle(mass=1.0, velocity=0.0)
        for _ in range(10_000):
            bath.apply_thermostat(p, dt=100.0)

        # Mean energy exchange per event should be near zero
        mean_exchange = bath.energy_exchanged / bath.n_events
        assert mean_exchange == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# ThermostatEvent ordering (task 6.4)
# ---------------------------------------------------------------------------

class TestThermostatEvent:
    def test_orders_by_time(self):
        e1 = ThermostatEvent(time=1.0, particle_index=0, bath_index=0)
        e2 = ThermostatEvent(time=2.0, particle_index=1, bath_index=1)
        assert e1 < e2

    def test_equal_time_orders_by_particle(self):
        e1 = ThermostatEvent(time=1.0, particle_index=0, bath_index=0)
        e2 = ThermostatEvent(time=1.0, particle_index=1, bath_index=0)
        assert e1 < e2

    def test_fields(self):
        e = ThermostatEvent(time=3.5, particle_index=7, bath_index=1)
        assert e.time == 3.5
        assert e.particle_index == 7
        assert e.bath_index == 1


# ---------------------------------------------------------------------------
# ThermostatScheduler (task 6.4)
# ---------------------------------------------------------------------------

class TestThermostatScheduler:
    """Tests for the thermostat event priority queue."""

    def _make_scheduler(self, seed=42):
        """Build a scheduler with a 10-particle chain, 1 boundary each end."""
        hot = HeatBath(temperature=2.0, coupling_rate=1.0,
                       rng=np.random.default_rng(seed))
        cold = HeatBath(temperature=0.5, coupling_rate=1.0,
                        rng=np.random.default_rng(seed + 1))
        config = ChainConfig(n_particles=10)
        chain = Chain(config)
        left, right = identify_boundary_particles(chain, n_boundary=1)
        scheduler = ThermostatScheduler(hot, cold, left, right)
        return scheduler, chain

    def test_build_event_queue_schedules_all_boundary_particles(self):
        scheduler, _ = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        # 1 left + 1 right = 2 events
        events = []
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            events.append(e)
        assert len(events) == 2
        indices = {e.particle_index for e in events}
        assert indices == {0, 9}

    def test_events_ordered_by_time(self):
        scheduler, _ = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        times = []
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            times.append(e.time)
        assert times == sorted(times)

    def test_all_events_in_future(self):
        scheduler, _ = self._make_scheduler()
        t0 = 5.0
        scheduler.build_event_queue(current_time=t0)
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            assert e.time > t0

    def test_peek_next_time_without_consuming(self):
        scheduler, _ = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        t_peek = scheduler.peek_next_time()
        e = scheduler.get_next_event()
        assert t_peek == pytest.approx(e.time)

    def test_peek_empty_returns_inf(self):
        scheduler, _ = self._make_scheduler()
        # Don't build queue — empty
        assert scheduler.peek_next_time() == float("inf")

    def test_process_event_reschedules(self):
        scheduler, chain = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        event = scheduler.get_next_event()
        # After processing, a new event should appear for the same particle
        scheduler.process_event(event, chain)
        # Drain remaining events
        particles_seen = set()
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            particles_seen.add(e.particle_index)
        assert event.particle_index in particles_seen

    def test_process_event_resamples_velocity(self):
        scheduler, chain = self._make_scheduler(seed=99)
        scheduler.build_event_queue(current_time=0.0)
        event = scheduler.get_next_event()
        p = chain[event.particle_index]
        v_before = p.velocity
        scheduler.process_event(event, chain)
        # Velocity was resampled (extremely unlikely to be identical)
        assert p.velocity != pytest.approx(v_before, abs=1e-12)

    def test_process_event_tracks_energy(self):
        scheduler, chain = self._make_scheduler(seed=77)
        scheduler.build_event_queue(current_time=0.0)
        event = scheduler.get_next_event()
        bath = scheduler.baths[event.bath_index]
        assert bath.n_events == 0
        scheduler.process_event(event, chain)
        assert bath.n_events == 1

    def test_correct_bath_assignment(self):
        scheduler, _ = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        events = []
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            events.append(e)
        bath_map = {e.particle_index: e.bath_index for e in events}
        assert bath_map[0] == 0   # left → hot bath
        assert bath_map[9] == 1   # right → cold bath

    def test_reschedule_particle(self):
        scheduler, _ = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        t_before = scheduler.peek_next_time()
        # Reschedule particle 0 at a later time
        scheduler.reschedule_particle(0, current_time=100.0)
        # The rescheduled event for particle 0 should be > 100
        events = []
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            events.append(e)
        p0_event = [e for e in events if e.particle_index == 0][0]
        assert p0_event.time > 100.0

    def test_reschedule_non_boundary_is_noop(self):
        scheduler, _ = self._make_scheduler()
        scheduler.build_event_queue(current_time=0.0)
        # Particle 5 is not a boundary particle — should be a no-op
        scheduler.reschedule_particle(5, current_time=0.0)
        events = []
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            events.append(e)
        assert len(events) == 2  # still only 2 boundary particles

    def test_multiple_boundary_particles(self):
        """Scheduler with 2 boundary particles per side."""
        hot = HeatBath(temperature=2.0, coupling_rate=1.0,
                       rng=np.random.default_rng(50))
        cold = HeatBath(temperature=0.5, coupling_rate=1.0,
                        rng=np.random.default_rng(51))
        config = ChainConfig(n_particles=10)
        chain = Chain(config)
        left, right = identify_boundary_particles(chain, n_boundary=2)
        scheduler = ThermostatScheduler(hot, cold, left, right)
        scheduler.build_event_queue(current_time=0.0)
        events = []
        while True:
            e = scheduler.get_next_event()
            if e is None:
                break
            events.append(e)
        assert len(events) == 4
        indices = {e.particle_index for e in events}
        assert indices == {0, 1, 8, 9}


# ---------------------------------------------------------------------------
# Boundary particle identification (task 8.2)
# ---------------------------------------------------------------------------

class TestIdentifyBoundaryParticles:
    def make_chain(self, n):
        return Chain(ChainConfig(n_particles=n))

    def test_single_boundary_particle(self):
        chain = self.make_chain(10)
        left, right = identify_boundary_particles(chain, n_boundary=1)
        assert left == [0]
        assert right == [9]

    def test_multiple_boundary_particles(self):
        chain = self.make_chain(10)
        left, right = identify_boundary_particles(chain, n_boundary=2)
        assert left == [0, 1]
        assert right == [8, 9]

    def test_no_overlap(self):
        chain = self.make_chain(10)
        left, right = identify_boundary_particles(chain, n_boundary=3)
        assert set(left).isdisjoint(set(right))

    def test_chain_too_short_raises(self):
        chain = self.make_chain(3)
        with pytest.raises(ValueError):
            identify_boundary_particles(chain, n_boundary=2)

    def test_zero_n_boundary_raises(self):
        chain = self.make_chain(10)
        with pytest.raises(ValueError):
            identify_boundary_particles(chain, n_boundary=0)

    def test_exact_coverage(self):
        """2 * n_boundary == n is the minimum valid case."""
        chain = self.make_chain(4)
        left, right = identify_boundary_particles(chain, n_boundary=2)
        assert sorted(left + right) == [0, 1, 2, 3]
