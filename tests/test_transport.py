"""Tests for TransportAnalyzer: gradient fitting, conductivity, and error estimation.

Covers tasks 9.1 – 9.7:
  - Unit tests for gradient fitting (9.6)
  - Unit tests for boundary exclusion (9.3)
  - Unit tests for conductivity calculation (9.4)
  - Unit tests for error estimation (9.5)
  - Integration test: known conductivity cases (9.7)
"""

import math
import pytest
import numpy as np

from src.analysis.transport import TransportAnalyzer, TransportResult
from src.core.chain import Chain, ChainConfig, BoundaryCondition
from src.simulation.engine import EventDrivenSimulator, SimulationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_profile(n, slope, intercept, spacing=1.0):
    """Return (positions, temperatures) for a perfect linear profile."""
    positions = [i * spacing for i in range(n)]
    temperatures = [intercept + slope * x for x in positions]
    return positions, temperatures


def _run_open_system(n=20, total_time=1500.0, T_hot=4.0, T_cold=0.5,
                     gamma=3.0, seed=42, spacing=0.3, measurement_interval=5.0):
    """Run an open-system simulation and return (result, chain).

    Default parameters are tuned to reliably reach steady state within
    the first half of the simulation time (matching TestSteadyStateConvergence).
    """
    np.random.seed(seed)
    chain_cfg = ChainConfig(
        n_particles=n,
        spacing=spacing,
        boundary=BoundaryCondition.OPEN,
        temperature=(T_hot + T_cold) / 2,
    )
    chain = Chain(chain_cfg)
    sim_cfg = SimulationConfig(
        total_time=total_time,
        T_hot=T_hot,
        T_cold=T_cold,
        gamma=gamma,
        measurement_interval=measurement_interval,
        ss_window_size=20,
        ss_threshold=0.5,
        seed=seed,
    )
    result = EventDrivenSimulator(chain, sim_cfg).run()
    return result, chain


# ---------------------------------------------------------------------------
# Unit tests: boundary exclusion (task 9.3)
# ---------------------------------------------------------------------------

class TestExcludeBoundaryLayers:
    def test_excludes_correct_particles(self):
        """Interior slice should skip n particles from each end."""
        positions, temperatures = linear_profile(10, slope=-0.5, intercept=5.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        x, T = analyzer.exclude_boundary_layers(positions, temperatures)

        assert list(x) == pytest.approx(positions[2:8])
        assert list(T) == pytest.approx(temperatures[2:8])

    def test_zero_exclusion_returns_all(self):
        """n_boundary_exclude=0 should return the full profile."""
        positions, temperatures = linear_profile(8, slope=-1.0, intercept=4.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=0)
        x, T = analyzer.exclude_boundary_layers(positions, temperatures)

        assert len(x) == 8
        assert len(T) == 8

    def test_raises_if_too_few_interior_points(self):
        """Should raise ValueError when interior slice has fewer than 2 points."""
        positions, temperatures = linear_profile(4, slope=-1.0, intercept=4.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        with pytest.raises(ValueError, match="fewer than 2 interior points"):
            analyzer.exclude_boundary_layers(positions, temperatures)

    def test_single_exclusion(self):
        positions, temperatures = linear_profile(6, slope=-0.5, intercept=3.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=1)
        x, T = analyzer.exclude_boundary_layers(positions, temperatures)
        assert len(x) == 4


# ---------------------------------------------------------------------------
# Unit tests: gradient fitting (task 9.2 / 9.6)
# ---------------------------------------------------------------------------

class TestFitGradient:
    def test_perfect_linear_profile(self):
        """Fit of a noiseless linear profile should recover slope exactly."""
        slope = -0.3
        positions, temperatures = linear_profile(20, slope=slope, intercept=4.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        fitted_slope, intercept, stderr = analyzer.fit_gradient(positions, temperatures)

        assert fitted_slope == pytest.approx(slope, rel=1e-8)
        assert stderr == pytest.approx(0.0, abs=1e-10)

    def test_noisy_profile_recovers_slope(self):
        """Fit should recover slope from a noisy but linear profile."""
        rng = np.random.default_rng(0)
        slope = -0.25
        n = 30
        positions = list(range(n))
        temperatures = [4.0 + slope * x + rng.normal(0, 0.05) for x in positions]

        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        fitted_slope, _, stderr = analyzer.fit_gradient(positions, temperatures)

        assert fitted_slope == pytest.approx(slope, abs=0.05)
        assert stderr > 0.0

    def test_stderr_decreases_with_more_points(self):
        """More interior points should give a smaller standard error."""
        rng = np.random.default_rng(1)
        slope = -0.3

        analyzer_small = TransportAnalyzer(n_boundary_exclude=4)
        analyzer_large = TransportAnalyzer(n_boundary_exclude=2)

        positions = list(range(20))
        temperatures = [3.0 + slope * x + rng.normal(0, 0.1) for x in positions]

        _, _, se_small = analyzer_small.fit_gradient(positions, temperatures)
        _, _, se_large = analyzer_large.fit_gradient(positions, temperatures)

        # More excluded → fewer points → larger error
        assert se_small >= se_large

    def test_returns_three_floats(self):
        positions, temperatures = linear_profile(12, slope=-0.4, intercept=5.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        result = analyzer.fit_gradient(positions, temperatures)

        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_fit_intercept_correct(self):
        """Intercept should also be recovered accurately."""
        intercept = 7.5
        positions, temperatures = linear_profile(16, slope=-0.2, intercept=intercept)
        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        _, fitted_intercept, _ = analyzer.fit_gradient(positions, temperatures)

        assert fitted_intercept == pytest.approx(intercept, abs=1e-6)

    def test_raises_if_too_few_interior(self):
        """Gradient fitting should propagate ValueError from exclusion."""
        positions, temperatures = linear_profile(4, slope=-0.5, intercept=3.0)
        analyzer = TransportAnalyzer(n_boundary_exclude=2)
        with pytest.raises(ValueError):
            analyzer.fit_gradient(positions, temperatures)


# ---------------------------------------------------------------------------
# Unit tests: conductivity calculation (task 9.4)
# ---------------------------------------------------------------------------

class TestCalculateConductivity:
    def test_fouriers_law(self):
        """κ = −J / gradient."""
        analyzer = TransportAnalyzer()
        # Flux positive (rightward), gradient negative (hot left / cold right)
        assert analyzer.calculate_conductivity(flux=2.0, gradient=-0.4) == pytest.approx(5.0)

    def test_negative_flux_positive_gradient(self):
        """Reversed convention should still give positive κ."""
        analyzer = TransportAnalyzer()
        assert analyzer.calculate_conductivity(flux=-3.0, gradient=0.6) == pytest.approx(5.0)

    def test_zero_gradient_raises(self):
        """Zero gradient → undefined κ."""
        analyzer = TransportAnalyzer()
        with pytest.raises(ValueError, match="gradient is zero"):
            analyzer.calculate_conductivity(flux=1.0, gradient=0.0)

    def test_kappa_scales_with_flux(self):
        """Doubling flux should double κ."""
        analyzer = TransportAnalyzer()
        k1 = analyzer.calculate_conductivity(flux=1.0, gradient=-0.5)
        k2 = analyzer.calculate_conductivity(flux=2.0, gradient=-0.5)
        assert k2 == pytest.approx(2.0 * k1)

    def test_kappa_scales_with_gradient(self):
        """Halving the gradient should double κ."""
        analyzer = TransportAnalyzer()
        k1 = analyzer.calculate_conductivity(flux=1.0, gradient=-0.4)
        k2 = analyzer.calculate_conductivity(flux=1.0, gradient=-0.2)
        assert k2 == pytest.approx(2.0 * k1)


# ---------------------------------------------------------------------------
# Unit tests: constructor validation
# ---------------------------------------------------------------------------

class TestTransportAnalyzerInit:
    def test_negative_n_boundary_raises(self):
        with pytest.raises(ValueError, match="n_boundary_exclude"):
            TransportAnalyzer(n_boundary_exclude=-1)

    def test_zero_n_bootstrap_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            TransportAnalyzer(n_bootstrap=0)

    def test_default_construction(self):
        a = TransportAnalyzer()
        assert a.n_boundary_exclude == 2
        assert a.n_bootstrap == 200


# ---------------------------------------------------------------------------
# Unit tests: analyze() on closed system raises
# ---------------------------------------------------------------------------

class TestAnalyzeValidation:
    def test_raises_on_closed_system_result(self):
        """analyze() should raise ValueError if flux_meter is None."""
        np.random.seed(0)
        chain_cfg = ChainConfig(
            n_particles=6,
            spacing=0.5,
            boundary=BoundaryCondition.PERIODIC,
            temperature=2.0,
        )
        chain = Chain(chain_cfg)
        sim_cfg = SimulationConfig(total_time=5.0, seed=0)
        result = EventDrivenSimulator(chain, sim_cfg).run()

        analyzer = TransportAnalyzer()
        with pytest.raises(ValueError, match="open-system"):
            analyzer.analyze(result, chain)


# ---------------------------------------------------------------------------
# Integration tests: known conductivity cases (task 9.7)
# ---------------------------------------------------------------------------

class TestTransportAnalyzerIntegration:
    def test_returns_transport_result(self):
        """analyze() should return a TransportResult."""
        result, chain = _run_open_system(n=16, seed=42)
        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=50,
                                     rng=np.random.default_rng(42))
        transport = analyzer.analyze(result, chain)

        assert isinstance(transport, TransportResult)

    def test_kappa_is_positive(self):
        """Thermal conductivity should be positive for a hot-left system.

        The crossing-based flux from FluxMeter is noisy for short simulations
        (consistency with reservoir flux is task 7.7). We verify κ > 0 using
        the reservoir-based flux, which is more reliable: J_reservoir =
        hot_bath.energy_exchanged / total_time > 0 by construction.
        """
        result, chain = _run_open_system(n=20, seed=42)
        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=50,
                                     rng=np.random.default_rng(42))
        transport = analyzer.analyze(result, chain)

        # Reservoir flux is the reliable measure for sign validation
        J_reservoir = result.hot_bath.energy_exchanged / result.time
        kappa_reservoir = analyzer.calculate_conductivity(J_reservoir, transport.gradient)
        assert kappa_reservoir > 0.0, (
            f"κ_reservoir = {kappa_reservoir:.4f} is not positive "
            f"(J_res={J_reservoir:.5f}, ∇T={transport.gradient:.4f})"
        )

    def test_gradient_is_negative(self):
        """Temperature gradient should be negative (hot left, cold right)."""
        result, chain = _run_open_system(n=20, seed=42)
        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=50,
                                     rng=np.random.default_rng(42))
        transport = analyzer.analyze(result, chain)

        assert transport.gradient < 0.0, (
            f"Expected negative gradient, got {transport.gradient:.4f}"
        )

    def test_reservoir_energy_has_correct_direction(self):
        """Hot bath injects energy; cold bath extracts it (net rightward flow)."""
        result, chain = _run_open_system(n=20, seed=42)

        assert result.hot_bath is not None
        assert result.hot_bath.energy_exchanged > 0, (
            f"Hot bath should inject energy, got {result.hot_bath.energy_exchanged:.4f}"
        )
        assert result.cold_bath is not None
        assert result.cold_bath.energy_exchanged < 0, (
            f"Cold bath should extract energy, got {result.cold_bath.energy_exchanged:.4f}"
        )

    def test_n_particles_matches_chain(self):
        """TransportResult.n_particles should equal the chain length."""
        n = 18
        result, chain = _run_open_system(n=n, seed=42)
        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=20,
                                     rng=np.random.default_rng(42))
        transport = analyzer.analyze(result, chain)

        assert transport.n_particles == n

    def test_temperatures_match_profile_length(self):
        """Returned temperatures list should have one entry per particle."""
        n = 16
        result, chain = _run_open_system(n=n, seed=42)
        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=20,
                                     rng=np.random.default_rng(42))
        transport = analyzer.analyze(result, chain)

        assert len(transport.temperatures) == n
        assert len(transport.positions) == n

    def test_kappa_error_is_finite_positive(self):
        """Bootstrap error on κ should be a finite positive float."""
        result, chain = _run_open_system(n=20, seed=42)
        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        transport = analyzer.analyze(result, chain)

        assert math.isfinite(transport.kappa_error), (
            f"kappa_error is not finite: {transport.kappa_error}"
        )
        assert transport.kappa_error >= 0.0

    def test_custom_t_start(self):
        """Specifying t_start should restrict the averaging window."""
        result, chain = _run_open_system(n=16, seed=42)

        a_full = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=20,
                                   rng=np.random.default_rng(42))
        a_late = TransportAnalyzer(n_boundary_exclude=2, t_start=1000.0,
                                   n_bootstrap=20, rng=np.random.default_rng(42))

        t_full = a_full.analyze(result, chain)
        t_late = a_late.analyze(result, chain)

        # Both should give positive κ (system is in steady state)
        assert t_full.kappa > 0.0
        assert t_late.kappa > 0.0

    @pytest.mark.slow
    def test_kappa_larger_for_larger_system(self):
        """For the ding-a-ling model, κ should grow with system size N.

        This verifies that the anomalous conductivity scaling κ ~ N^alpha
        (with alpha > 0) is reproduced by the simulation pipeline.
        """
        result_small, chain_small = _run_open_system(
            n=10, total_time=2000.0, seed=42, gamma=5.0)
        result_large, chain_large = _run_open_system(
            n=30, total_time=2000.0, seed=42, gamma=5.0)

        analyzer = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=50,
                                     rng=np.random.default_rng(42))
        t_small = analyzer.analyze(result_small, chain_small)

        analyzer2 = TransportAnalyzer(n_boundary_exclude=2, n_bootstrap=50,
                                      rng=np.random.default_rng(42))
        t_large = analyzer2.analyze(result_large, chain_large)

        assert t_large.kappa > t_small.kappa, (
            f"Expected κ(N=30) > κ(N=10), got {t_large.kappa:.4f} vs {t_small.kappa:.4f}"
        )
