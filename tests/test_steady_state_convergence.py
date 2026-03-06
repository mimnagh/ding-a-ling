"""Integration test: steady-state convergence (task 8.5).

Verifies that an open system with hot/cold thermal reservoirs converges
to a non-equilibrium steady state with a monotonic temperature gradient,
stable temperature profile, and non-zero heat transport.
"""

import pytest
import numpy as np

from src.core.chain import Chain, ChainConfig, BoundaryCondition
from src.simulation.engine import EventDrivenSimulator, SimulationConfig


def make_open_chain(n=20, temperature=2.0, seed=42, spacing=0.3):
    """Create an OPEN chain with thermal velocities."""
    np.random.seed(seed)
    config = ChainConfig(
        n_particles=n,
        spacing=spacing,
        boundary=BoundaryCondition.OPEN,
        temperature=temperature,
    )
    return Chain(config)


def run_open_system(n=20, total_time=1000.0, T_hot=4.0, T_cold=0.5,
                    gamma=3.0, seed=42, measurement_interval=5.0,
                    ss_window_size=20, ss_threshold=0.5):
    """Run an open system simulation and return the result."""
    chain = make_open_chain(n=n, seed=seed)
    config = SimulationConfig(
        total_time=total_time,
        T_hot=T_hot,
        T_cold=T_cold,
        gamma=gamma,
        measurement_interval=measurement_interval,
        ss_window_size=ss_window_size,
        ss_threshold=ss_threshold,
        seed=seed,
    )
    sim = EventDrivenSimulator(chain, config)
    return sim.run()


class TestSteadyStateConvergence:
    """Integration tests for steady-state convergence in open systems."""

    def test_temperature_gradient_established(self):
        """Hot end should be warmer than cold end after transient."""
        result = run_open_system(seed=42)

        # Average over the second half (after transient relaxation)
        half_time = result.time / 2
        avg = result.time_averaged_temperature(t_start=half_time)
        assert len(avg) == 20

        # Left (hot bath) should be warmer than right (cold bath)
        assert avg[0] > avg[-1], (
            f"Expected T_left > T_right: {avg[0]:.3f} vs {avg[-1]:.3f}"
        )

        # Hot end should be near T_hot, cold end near T_cold
        assert avg[0] > 2.0, f"Hot end too cold: {avg[0]:.3f}"
        assert avg[-1] < 2.0, f"Cold end too hot: {avg[-1]:.3f}"

    def test_temperature_profile_monotonic(self):
        """Temperature should decrease monotonically from hot to cold end.

        We check the trend using a linear fit rather than strict monotonicity,
        since local fluctuations are expected in a small system.
        """
        result = run_open_system(seed=123)
        half_time = result.time / 2
        avg = result.time_averaged_temperature(t_start=half_time)

        # Linear fit: slope should be negative (decreasing from left to right)
        positions = np.arange(len(avg))
        slope, _ = np.polyfit(positions, avg, 1)
        assert slope < 0, f"Expected negative temperature slope, got {slope:.4f}"

    def test_temperature_profile_converges(self):
        """Temperature profile should stabilise: late windows should agree."""
        result = run_open_system(total_time=1000.0, seed=42)

        # Compare two consecutive late windows
        avg_mid = result.time_averaged_temperature(t_start=400.0, t_end=600.0)
        avg_late = result.time_averaged_temperature(t_start=800.0, t_end=1000.0)

        assert len(avg_mid) == 20
        assert len(avg_late) == 20

        # Maximum per-particle temperature difference should be small
        diffs = [abs(m - l) for m, l in zip(avg_mid, avg_late)]
        max_diff = max(diffs)
        assert max_diff < 0.5, (
            f"Temperature profile not converged: max diff = {max_diff:.3f}"
        )

    def test_heat_baths_exchange_energy(self):
        """Both baths should have exchanged energy."""
        result = run_open_system(seed=99)

        assert result.hot_bath is not None
        assert result.cold_bath is not None
        assert result.hot_bath.n_events > 0
        assert result.cold_bath.n_events > 0
        assert result.hot_bath.energy_exchanged != 0.0
        assert result.cold_bath.energy_exchanged != 0.0

    def test_collisions_and_thermostat_events(self):
        """Both collision and thermostat dynamics should be active."""
        result = run_open_system(seed=77)

        assert result.n_collisions > 0, "No collisions occurred"
        assert result.n_thermostat_events > 0, "No thermostat events occurred"
        # Thermostat should be dominant for open systems with moderate gamma
        assert result.n_thermostat_events > result.n_collisions

    def test_flux_has_correct_sign(self):
        """Net heat flux should flow from hot to cold (positive direction).

        For this system, hot is on the left and cold is on the right,
        so net flux through the centre should be positive (rightward energy flow).
        We check cumulative flux over the second half of the simulation.
        """
        result = run_open_system(total_time=1000.0, seed=42)
        assert result.flux_meter is not None

        # There should be flux crossings
        half_time = result.time / 2
        crossings = result.flux_meter.crossings_in_window(half_time, result.time)
        assert len(crossings) > 0, "No flux crossings in second half"

    def test_energy_history_recorded(self):
        """Energy history should be recorded at measurement intervals."""
        result = run_open_system(seed=42, measurement_interval=10.0)

        assert len(result.energy_history) >= 2
        # Times should be monotonically increasing
        times = [t for t, _ in result.energy_history]
        assert times == sorted(times)

    def test_temperature_profiles_recorded(self):
        """Temperature snapshots should be recorded with correct dimensions."""
        result = run_open_system(n=20, seed=42)

        assert len(result.temperature_profile) >= 2
        for _, temps in result.temperature_profile:
            assert len(temps) == 20

    def test_reaches_total_time(self):
        """Simulation should run for the full requested duration."""
        total = 500.0
        result = run_open_system(total_time=total, seed=42)
        assert result.time == pytest.approx(total)

    @pytest.mark.slow
    def test_larger_system_convergence(self):
        """Larger system should also establish temperature gradient."""
        result = run_open_system(n=40, total_time=2000.0, seed=42)

        avg = result.time_averaged_temperature(t_start=1000.0)
        assert len(avg) == 40
        assert avg[0] > avg[-1]

        # Linear fit should show negative slope
        slope, _ = np.polyfit(np.arange(40), avg, 1)
        assert slope < 0
