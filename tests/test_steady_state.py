"""Tests for steady-state detection."""

import pytest
import numpy as np

from src.core.chain import Chain, ChainConfig, BoundaryCondition
from src.simulation.engine import EventDrivenSimulator, SimulationConfig
from src.simulation.steady_state import SteadyStateDetector


# ---------------------------------------------------------------------------
# Unit tests for SteadyStateDetector
# ---------------------------------------------------------------------------

class TestSteadyStateDetector:
    def test_not_triggered_insufficient_samples(self):
        """Detector should not trigger before collecting window_size samples."""
        det = SteadyStateDetector(window_size=10, threshold=0.01)
        for i in range(9):
            assert det.add_sample(1.0, float(i)) is False
        assert det.is_steady_state is False
        assert det.steady_state_time is None

    def test_triggered_constant_flux(self):
        """Constant flux has zero CV — should trigger immediately at window_size."""
        det = SteadyStateDetector(window_size=10, threshold=0.01)
        for i in range(10):
            result = det.add_sample(5.0, float(i))
        assert result is True
        assert det.is_steady_state is True
        assert det.steady_state_time == 9.0

    def test_not_triggered_high_variance(self):
        """Alternating flux should have high CV and not trigger."""
        det = SteadyStateDetector(window_size=10, threshold=0.01)
        for i in range(100):
            flux = 1.0 if i % 2 == 0 else 10.0
            det.add_sample(flux, float(i))
        assert det.is_steady_state is False

    def test_latching_behavior(self):
        """Once triggered, steady state stays True even with noisy samples."""
        det = SteadyStateDetector(window_size=5, threshold=0.05)
        # Constant flux triggers detection
        for i in range(5):
            det.add_sample(3.0, float(i))
        assert det.is_steady_state is True
        t_first = det.steady_state_time

        # Add noisy samples — should still report True
        for i in range(5, 20):
            flux = 3.0 + (-1) ** i * 100.0
            assert det.add_sample(flux, float(i)) is True
        assert det.steady_state_time == t_first

    def test_zero_mean_not_triggered(self):
        """If mean flux ≈ 0, CV is undefined — should not trigger."""
        det = SteadyStateDetector(window_size=10, threshold=0.5)
        for i in range(20):
            # Symmetric flux cancels to zero mean
            flux = 1.0 if i % 2 == 0 else -1.0
            det.add_sample(flux, float(i))
        assert det.is_steady_state is False

    def test_coefficient_of_variation_none_before_window(self):
        """CV should be None before enough samples are collected."""
        det = SteadyStateDetector(window_size=10, threshold=0.01)
        assert det.coefficient_of_variation() is None
        for i in range(5):
            det.add_sample(1.0, float(i))
        assert det.coefficient_of_variation() is None

    def test_coefficient_of_variation_value(self):
        """CV should reflect the actual standard deviation / mean."""
        det = SteadyStateDetector(window_size=4, threshold=1.0)
        samples = [2.0, 4.0, 2.0, 4.0]
        for i, s in enumerate(samples):
            det.add_sample(s, float(i))
        cv = det.coefficient_of_variation()
        expected_cv = np.std(samples) / abs(np.mean(samples))
        assert cv == pytest.approx(expected_cv, rel=1e-10)

    def test_gradual_convergence(self):
        """Detector triggers when flux stabilizes after initial transient."""
        det = SteadyStateDetector(window_size=10, threshold=0.02)
        rng = np.random.default_rng(42)

        # Transient phase: noisy flux
        for i in range(50):
            det.add_sample(5.0 + rng.normal(0, 2.0), float(i))

        # Steady phase: nearly constant flux
        for i in range(50, 100):
            det.add_sample(5.0 + rng.normal(0, 0.01), float(i))

        assert det.is_steady_state is True
        assert det.steady_state_time is not None
        assert det.steady_state_time >= 50.0


# ---------------------------------------------------------------------------
# Integration test with simulator
# ---------------------------------------------------------------------------

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


class TestSteadyStateIntegration:
    def test_result_has_steady_state_fields(self):
        """SimulationResult should contain steady-state detection fields."""
        chain = make_open_chain(n=6)
        config = SimulationConfig(
            total_time=50.0,
            T_hot=2.0,
            T_cold=0.5,
            gamma=2.0,
            measurement_interval=0.5,
            ss_window_size=10,
            ss_threshold=0.5,  # generous threshold for short sim
            seed=123,
        )
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()

        # Fields should exist and be the right types
        assert isinstance(result.is_steady_state, bool)
        assert result.steady_state_time is None or isinstance(
            result.steady_state_time, float
        )

    def test_closed_system_no_steady_state(self):
        """Closed systems should always report is_steady_state=False."""
        np.random.seed(42)
        config_chain = ChainConfig(
            n_particles=6,
            spacing=0.5,
            boundary=BoundaryCondition.PERIODIC,
            temperature=2.0,
        )
        chain = Chain(config_chain)
        config = SimulationConfig(total_time=10.0, seed=42)
        sim = EventDrivenSimulator(chain, config)
        result = sim.run()

        assert result.is_steady_state is False
        assert result.steady_state_time is None
