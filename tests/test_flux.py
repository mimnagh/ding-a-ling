"""Tests for FluxMeter: crossing detection, recording, and flux calculation."""

import pytest
import numpy as np

from src.core.particle import Particle, ParticleType
from src.simulation.flux import CrossingRecord, FluxMeter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_free(index=0, position=0.0, velocity=1.0):
    return Particle(
        index=index,
        particle_type=ParticleType.FREE,
        mass=1.0,
        position=position,
        velocity=velocity,
        equilibrium_pos=position,
        spring_constant=0.0,
    )


def make_harmonic(index=0, position=0.0, velocity=0.0, eq_pos=0.0, k=1.0, mass=1.0):
    return Particle(
        index=index,
        particle_type=ParticleType.HARMONIC,
        mass=mass,
        position=position,
        velocity=velocity,
        equilibrium_pos=eq_pos,
        spring_constant=k,
    )


# ---------------------------------------------------------------------------
# CrossingRecord
# ---------------------------------------------------------------------------

class TestCrossingRecord:
    def test_flux_contribution_rightward(self):
        rec = CrossingRecord(time=1.0, particle_index=0, direction=1, energy=2.5)
        assert rec.flux_contribution == pytest.approx(2.5)

    def test_flux_contribution_leftward(self):
        rec = CrossingRecord(time=1.0, particle_index=0, direction=-1, energy=2.5)
        assert rec.flux_contribution == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestFluxMeterInit:
    def test_position_stored(self):
        fm = FluxMeter(position=3.5)
        assert fm.x_ref == pytest.approx(3.5)

    def test_name_stored(self):
        fm = FluxMeter(position=0.0, name="center")
        assert fm.name == "center"

    def test_history_starts_empty(self):
        fm = FluxMeter(position=0.0)
        assert fm.flux_history == []

    def test_repr(self):
        fm = FluxMeter(position=1.0, name="left")
        assert "FluxMeter" in repr(fm)
        assert "left" in repr(fm)


# ---------------------------------------------------------------------------
# record_crossing
# ---------------------------------------------------------------------------

class TestRecordCrossing:
    def test_appends_record(self):
        fm = FluxMeter(position=1.0)
        p = make_free(position=1.0, velocity=2.0)
        fm.record_crossing(p, time=0.5, direction=1)
        assert len(fm.flux_history) == 1

    def test_record_fields(self):
        fm = FluxMeter(position=1.0)
        p = make_free(index=3, position=1.0, velocity=2.0)
        rec = fm.record_crossing(p, time=0.5, direction=1)
        assert rec.time == pytest.approx(0.5)
        assert rec.particle_index == 3
        assert rec.direction == 1
        assert rec.energy == pytest.approx(p.total_energy())

    def test_multiple_crossings_ordered(self):
        fm = FluxMeter(position=0.0)
        p = make_free()
        fm.record_crossing(p, time=1.0, direction=1)
        fm.record_crossing(p, time=2.0, direction=-1)
        fm.record_crossing(p, time=3.0, direction=1)
        assert [r.time for r in fm.flux_history] == [1.0, 2.0, 3.0]

    def test_returns_crossing_record(self):
        fm = FluxMeter(position=0.0)
        p = make_free()
        rec = fm.record_crossing(p, time=1.0, direction=1)
        assert isinstance(rec, CrossingRecord)


# ---------------------------------------------------------------------------
# check_crossing – free particles
# ---------------------------------------------------------------------------

class TestCheckCrossingFree:
    def test_rightward_crossing(self):
        # particle at x=0, v=+2, x_ref=1 → crosses at dt=0.5
        fm = FluxMeter(position=1.0)
        p = make_free(position=0.0, velocity=2.0)
        result = fm.check_crossing(p, t_start=0.0, t_end=1.0)
        assert result is not None
        t_cross, direction = result
        assert t_cross == pytest.approx(0.5)
        assert direction == 1

    def test_leftward_crossing(self):
        # particle at x=2, v=-2, x_ref=1 → crosses at dt=0.5
        fm = FluxMeter(position=1.0)
        p = make_free(position=2.0, velocity=-2.0)
        result = fm.check_crossing(p, t_start=0.0, t_end=1.0)
        assert result is not None
        t_cross, direction = result
        assert t_cross == pytest.approx(0.5)
        assert direction == -1

    def test_no_crossing_moving_away(self):
        # particle at x=2, v=+1 moving away from x_ref=1
        fm = FluxMeter(position=1.0)
        p = make_free(position=2.0, velocity=1.0)
        assert fm.check_crossing(p, t_start=0.0, t_end=10.0) is None

    def test_no_crossing_outside_window(self):
        # particle at x=0, v=1, x_ref=5 → crosses at dt=5, but window is [0,3]
        fm = FluxMeter(position=5.0)
        p = make_free(position=0.0, velocity=1.0)
        assert fm.check_crossing(p, t_start=0.0, t_end=3.0) is None

    def test_no_crossing_stationary(self):
        fm = FluxMeter(position=1.0)
        p = make_free(position=0.0, velocity=0.0)
        assert fm.check_crossing(p, t_start=0.0, t_end=10.0) is None

    def test_crossing_at_boundary_of_window(self):
        # exactly at t_end — should be included
        fm = FluxMeter(position=1.0)
        p = make_free(position=0.0, velocity=1.0)
        result = fm.check_crossing(p, t_start=0.0, t_end=1.0)
        assert result is not None

    def test_t_start_offset(self):
        # Particle state is at t=5; v=1, position=0, x_ref=2 → crosses at dt=2 → t=7
        fm = FluxMeter(position=2.0)
        p = make_free(position=0.0, velocity=1.0)
        result = fm.check_crossing(p, t_start=5.0, t_end=8.0)
        assert result is not None
        t_cross, _ = result
        assert t_cross == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# check_crossing – harmonic particles
# ---------------------------------------------------------------------------

class TestCheckCrossingHarmonic:
    def test_harmonic_crosses_x_ref(self):
        # Harmonic at eq=0, starts at x=−1 with v=0, k=1, m=1 → ω=1
        # x(t) = -cos(t); crosses x_ref=0 at t=π/2
        fm = FluxMeter(position=0.0)
        p = make_harmonic(position=-1.0, velocity=0.0, eq_pos=0.0, k=1.0, mass=1.0)
        result = fm.check_crossing(p, t_start=0.0, t_end=np.pi)
        assert result is not None
        t_cross, direction = result
        assert t_cross == pytest.approx(np.pi / 2, abs=1e-6)
        assert direction == 1  # moving rightward at π/2 (sin is positive)

    def test_harmonic_no_crossing_amplitude_too_small(self):
        # Oscillates around eq=0 with amplitude 0.1; x_ref=1.0 is out of range
        fm = FluxMeter(position=1.0)
        p = make_harmonic(position=0.1, velocity=0.0, eq_pos=0.0, k=1.0, mass=1.0)
        assert fm.check_crossing(p, t_start=0.0, t_end=10.0) is None

    def test_harmonic_leftward_crossing(self):
        # Starts at x=+1, v=0 → x(t) = cos(t); first crosses 0 at t=π/2 moving left
        fm = FluxMeter(position=0.0)
        p = make_harmonic(position=1.0, velocity=0.0, eq_pos=0.0, k=1.0, mass=1.0)
        result = fm.check_crossing(p, t_start=0.0, t_end=np.pi)
        assert result is not None
        _, direction = result
        assert direction == -1


# ---------------------------------------------------------------------------
# time_averaged_flux
# ---------------------------------------------------------------------------

class TestTimeAveragedFlux:
    def test_single_rightward_crossing(self):
        fm = FluxMeter(position=0.0)
        # Particle with KE = 0.5*1*2^2 = 2.0, direction +1
        p = make_free(position=0.0, velocity=2.0)
        fm.record_crossing(p, time=5.0, direction=1)
        # J = 2.0 / 10.0 = 0.2
        assert fm.time_averaged_flux(0.0, 10.0) == pytest.approx(0.2)

    def test_equal_and_opposite_crossings_cancel(self):
        fm = FluxMeter(position=0.0)
        p = make_free(position=0.0, velocity=2.0)
        energy = p.total_energy()
        fm.record_crossing(p, time=2.0, direction=1)
        fm.record_crossing(p, time=4.0, direction=-1)
        assert fm.time_averaged_flux(0.0, 10.0) == pytest.approx(0.0)

    def test_zero_duration_returns_zero(self):
        fm = FluxMeter(position=0.0)
        p = make_free()
        fm.record_crossing(p, time=1.0, direction=1)
        assert fm.time_averaged_flux(1.0, 1.0) == 0.0

    def test_empty_history_returns_zero(self):
        fm = FluxMeter(position=0.0)
        assert fm.time_averaged_flux(0.0, 10.0) == 0.0

    def test_excludes_crossings_outside_window(self):
        fm = FluxMeter(position=0.0)
        p = make_free(velocity=2.0)
        fm.record_crossing(p, time=1.0, direction=1)   # inside
        fm.record_crossing(p, time=20.0, direction=1)  # outside
        J = fm.time_averaged_flux(0.0, 10.0)
        assert J == pytest.approx(p.total_energy() / 10.0)

    def test_net_flux_direction(self):
        # Three rightward, one leftward → positive net flux
        fm = FluxMeter(position=0.0)
        p = make_free(velocity=1.0)
        for t in [1.0, 2.0, 3.0]:
            fm.record_crossing(p, time=t, direction=1)
        fm.record_crossing(p, time=4.0, direction=-1)
        assert fm.time_averaged_flux(0.0, 10.0) > 0.0


# ---------------------------------------------------------------------------
# cumulative_flux
# ---------------------------------------------------------------------------

class TestCumulativeFlux:
    def test_no_crossings(self):
        fm = FluxMeter(position=0.0)
        assert fm.cumulative_flux(10.0) == 0.0

    def test_cumulative_increases_rightward(self):
        fm = FluxMeter(position=0.0)
        p = make_free(velocity=2.0)
        fm.record_crossing(p, time=1.0, direction=1)
        fm.record_crossing(p, time=2.0, direction=1)
        c1 = fm.cumulative_flux(1.5)
        c2 = fm.cumulative_flux(3.0)
        assert c2 > c1

    def test_excludes_future_crossings(self):
        fm = FluxMeter(position=0.0)
        p = make_free(velocity=2.0)
        fm.record_crossing(p, time=5.0, direction=1)
        assert fm.cumulative_flux(4.0) == 0.0


# ---------------------------------------------------------------------------
# crossings_in_window
# ---------------------------------------------------------------------------

class TestCrossingsInWindow:
    def test_returns_correct_subset(self):
        fm = FluxMeter(position=0.0)
        p = make_free()
        fm.record_crossing(p, time=1.0, direction=1)
        fm.record_crossing(p, time=5.0, direction=1)
        fm.record_crossing(p, time=9.0, direction=1)
        window = fm.crossings_in_window(2.0, 8.0)
        assert len(window) == 1
        assert window[0].time == pytest.approx(5.0)

    def test_empty_for_no_match(self):
        fm = FluxMeter(position=0.0)
        p = make_free()
        fm.record_crossing(p, time=10.0, direction=1)
        assert fm.crossings_in_window(0.0, 5.0) == []


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_clears_history(self):
        fm = FluxMeter(position=0.0)
        p = make_free()
        fm.record_crossing(p, time=1.0, direction=1)
        fm.reset()
        assert fm.flux_history == []

    def test_flux_after_reset_is_zero(self):
        fm = FluxMeter(position=0.0)
        p = make_free(velocity=2.0)
        fm.record_crossing(p, time=1.0, direction=1)
        fm.reset()
        assert fm.time_averaged_flux(0.0, 10.0) == 0.0


# ---------------------------------------------------------------------------
# Integration: check_crossing → record_crossing → time_averaged_flux
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_free_particle_flux_pipeline(self):
        """Particle crosses plane, gets recorded, flux is nonzero."""
        fm = FluxMeter(position=1.0)
        p = make_free(position=0.0, velocity=2.0)

        result = fm.check_crossing(p, t_start=0.0, t_end=2.0)
        assert result is not None
        t_cross, direction = result

        # Advance particle to crossing time and record
        p.evolve_free(t_cross)
        fm.record_crossing(p, time=t_cross, direction=direction)

        J = fm.time_averaged_flux(0.0, 2.0)
        assert J > 0.0  # rightward energy flow

    def test_balanced_flux_gives_zero(self):
        """Equal rightward and leftward crossings → J ≈ 0."""
        fm = FluxMeter(position=0.0)
        p = make_free(velocity=1.0)
        energy = p.total_energy()

        fm.record_crossing(p, time=1.0, direction=1)
        fm.record_crossing(p, time=2.0, direction=-1)

        assert fm.time_averaged_flux(0.0, 10.0) == pytest.approx(0.0)
