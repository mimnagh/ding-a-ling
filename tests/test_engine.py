"""Tests for the EventDrivenSimulator engine."""

import pytest
import numpy as np

from src.core.chain import Chain, ChainConfig, BoundaryCondition
from src.simulation.engine import (
    EventDrivenSimulator, SimulationConfig, SimulationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_closed_chain(n=10, temperature=2.0, seed=42, spacing=0.5):
    """Create a PERIODIC chain with thermal velocities.

    Higher temperature and tighter spacing ensure collisions happen.
    """
    np.random.seed(seed)
    config = ChainConfig(
        n_particles=n,
        spacing=spacing,
        boundary=BoundaryCondition.PERIODIC,
        temperature=temperature,
    )
    return Chain(config)


def make_open_chain(n=10, temperature=2.0, seed=42, spacing=0.5):
    """Create an OPEN chain with thermal velocities."""
    np.random.seed(seed)
    config = ChainConfig(
        n_particles=n,
        spacing=spacing,
        boundary=BoundaryCondition.OPEN,
        temperature=temperature,
    )
    return Chain(config)


# ---------------------------------------------------------------------------
# Closed system tests
# ---------------------------------------------------------------------------

class TestClosedSystem:
    def test_energy_conservation(self):
        """Closed system should conserve energy to machine precision."""
        chain = make_closed_chain(n=10)
        initial_energy = chain.total_energy()

        sim = EventDrivenSimulator(chain, SimulationConfig(total_time=50.0))
        result = sim.run()

        final_energy = chain.total_energy()
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        assert relative_error < 1e-8

    def test_collisions_occur(self):
        chain = make_closed_chain(n=10)
        sim = EventDrivenSimulator(chain, SimulationConfig(total_time=50.0))
        result = sim.run()
        assert result.n_collisions > 0

    def test_no_thermostat_events(self):
        """Closed system should have zero thermostat events."""
        chain = make_closed_chain(n=10)
        sim = EventDrivenSimulator(chain, SimulationConfig(total_time=10.0))
        result = sim.run()
        assert result.n_thermostat_events == 0

    def test_no_flux_meter(self):
        chain = make_closed_chain(n=10)
        sim = EventDrivenSimulator(chain, SimulationConfig(total_time=10.0))
        result = sim.run()
        assert result.flux_meter is None
        assert result.hot_bath is None
        assert result.cold_bath is None

    def test_energy_history_recorded(self):
        chain = make_closed_chain(n=10)
        sim = EventDrivenSimulator(
            chain, SimulationConfig(total_time=10.0, measurement_interval=2.0)
        )
        result = sim.run()
        assert len(result.energy_history) >= 2
        # Times should be monotonically increasing
        times = [t for t, _ in result.energy_history]
        assert times == sorted(times)

    def test_temperature_profile_recorded(self):
        chain = make_closed_chain(n=10)
        sim = EventDrivenSimulator(
            chain, SimulationConfig(total_time=10.0, measurement_interval=2.0)
        )
        result = sim.run()
        assert len(result.temperature_profile) >= 2
        # Each profile should have one entry per particle
        for _, temps in result.temperature_profile:
            assert len(temps) == 10

    def test_reaches_total_time(self):
        chain = make_closed_chain(n=10)
        total = 20.0
        sim = EventDrivenSimulator(chain, SimulationConfig(total_time=total))
        result = sim.run()
        assert result.time == pytest.approx(total)


# ---------------------------------------------------------------------------
# Open system tests
# ---------------------------------------------------------------------------

class TestOpenSystem:
    def test_runs_without_error(self):
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=50.0, T_hot=2.0, T_cold=0.5, gamma=1.0, seed=123,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_thermostat_events_fire(self):
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=50.0, T_hot=2.0, T_cold=0.5, gamma=2.0, seed=456,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        assert result.n_thermostat_events > 0

    def test_collisions_also_occur(self):
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=50.0, T_hot=2.0, T_cold=0.5, gamma=1.0, seed=789,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        assert result.n_collisions > 0

    def test_flux_meter_records_crossings(self):
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=100.0, T_hot=2.0, T_cold=0.5, gamma=1.0, seed=101,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        assert result.flux_meter is not None
        assert len(result.flux_meter.flux_history) > 0

    def test_bath_energy_exchange(self):
        """Both baths should have exchanged energy after sufficient time."""
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=100.0, T_hot=2.0, T_cold=0.5, gamma=2.0, seed=202,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        assert result.hot_bath is not None
        assert result.cold_bath is not None
        assert result.hot_bath.n_events > 0
        assert result.cold_bath.n_events > 0

    def test_energy_history_recorded(self):
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=20.0, T_hot=2.0, T_cold=0.5, gamma=1.0,
            measurement_interval=5.0, seed=303,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        assert len(result.energy_history) >= 2

    def test_temperature_profile_has_correct_length(self):
        n = 10
        chain = make_open_chain(n=n)
        config = SimulationConfig(
            total_time=10.0, T_hot=2.0, T_cold=0.5, gamma=1.0, seed=404,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        for _, temps in result.temperature_profile:
            assert len(temps) == n

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        def run_once():
            chain = make_open_chain(n=10, seed=55)
            config = SimulationConfig(
                total_time=20.0, T_hot=2.0, T_cold=0.5, gamma=1.0, seed=999,
            )
            sim = EventDrivenSimulator(chain, config)
            return sim.run()

        r1 = run_once()
        r2 = run_once()
        assert r1.n_collisions == r2.n_collisions
        assert r1.n_thermostat_events == r2.n_thermostat_events
        assert r1.time == pytest.approx(r2.time)


# ---------------------------------------------------------------------------
# Temperature profile measurement tests (task 8.4)
# ---------------------------------------------------------------------------

class TestTemperatureProfile:
    def test_time_averaged_single_snapshot(self):
        """With one snapshot, average equals that snapshot."""
        chain = make_closed_chain(n=5)
        sim = EventDrivenSimulator(
            chain, SimulationConfig(total_time=0.0, measurement_interval=100.0),
        )
        result = sim.run()
        # Only the initial measurement should exist
        assert len(result.temperature_profile) >= 1
        avg = result.time_averaged_temperature()
        _, first_temps = result.temperature_profile[0]
        for a, t in zip(avg, first_temps):
            assert a == pytest.approx(t)

    def test_time_averaged_multiple_snapshots(self):
        """Average over multiple snapshots is the arithmetic mean."""
        chain = make_closed_chain(n=5)
        sim = EventDrivenSimulator(
            chain, SimulationConfig(total_time=10.0, measurement_interval=2.0),
        )
        result = sim.run()
        assert len(result.temperature_profile) >= 3
        avg = result.time_averaged_temperature()
        assert len(avg) == 5
        # Manually verify first particle's average
        all_t0 = [temps[0] for _, temps in result.temperature_profile]
        assert avg[0] == pytest.approx(sum(all_t0) / len(all_t0))

    def test_time_averaged_window_filtering(self):
        """t_start and t_end filter snapshots correctly."""
        chain = make_closed_chain(n=5)
        sim = EventDrivenSimulator(
            chain, SimulationConfig(total_time=10.0, measurement_interval=2.0),
        )
        result = sim.run()
        # Average over just the second half
        avg_full = result.time_averaged_temperature()
        avg_late = result.time_averaged_temperature(t_start=6.0)
        # They should generally differ (different time windows)
        assert len(avg_late) == 5
        # Verify only late snapshots included
        late_snapshots = [
            temps for t, temps in result.temperature_profile if t >= 6.0
        ]
        expected_0 = sum(s[0] for s in late_snapshots) / len(late_snapshots)
        assert avg_late[0] == pytest.approx(expected_0)

    def test_time_averaged_empty_window(self):
        """Window with no snapshots returns empty list."""
        chain = make_closed_chain(n=5)
        sim = EventDrivenSimulator(
            chain, SimulationConfig(total_time=10.0, measurement_interval=2.0),
        )
        result = sim.run()
        avg = result.time_averaged_temperature(t_start=999.0)
        assert avg == []

    def test_open_system_temperature_gradient(self):
        """Hot end should be warmer than cold end in steady state."""
        chain = make_open_chain(n=10)
        config = SimulationConfig(
            total_time=200.0, T_hot=4.0, T_cold=0.5, gamma=2.0,
            measurement_interval=5.0, seed=42,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()
        # Average over second half (after transient)
        avg = result.time_averaged_temperature(t_start=100.0)
        assert len(avg) == 10
        # Left end (hot bath) should be warmer than right end (cold bath)
        assert avg[0] > avg[-1]

    def test_particle_positions(self):
        """particle_positions returns equilibrium positions."""
        chain = make_closed_chain(n=5, spacing=0.5)
        sim = EventDrivenSimulator(chain, SimulationConfig(total_time=0.0))
        result = sim.run()
        positions = result.particle_positions(chain)
        assert len(positions) == 5
        for i, pos in enumerate(positions):
            assert pos == pytest.approx(i * 0.5)
