"""Unit tests for Particle class."""

import pytest
import numpy as np
from src.core.particle import Particle, ParticleType


class TestFreeParticleEvolution:
    """Tests for free particle ballistic motion."""
    
    def test_free_particle_evolution(self):
        """Free particle moves ballistically with constant velocity."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=0.0,
            velocity=2.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        dt = 1.5
        initial_pos = particle.position
        initial_vel = particle.velocity
        
        particle.evolve_free(dt)
        
        # Position should increase by velocity * time
        assert np.isclose(particle.position, initial_pos + initial_vel * dt)
        # Velocity should remain constant
        assert np.isclose(particle.velocity, initial_vel)
    
    def test_free_particle_negative_velocity(self):
        """Free particle with negative velocity moves backward."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=10.0,
            velocity=-3.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        dt = 2.0
        particle.evolve_free(dt)
        
        assert np.isclose(particle.position, 10.0 - 3.0 * 2.0)
        assert np.isclose(particle.velocity, -3.0)


class TestHarmonicParticleEvolution:
    """Tests for harmonic particle oscillation."""
    
    def test_harmonic_particle_evolution(self):
        """Harmonic particle oscillates according to analytic solution."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.HARMONIC,
            mass=1.0,
            position=1.0,
            velocity=0.0,
            equilibrium_pos=0.0,
            spring_constant=1.0
        )
        
        # For k=m=1, omega=1, period T=2π
        # Starting at x=1, v=0 (maximum displacement)
        # After quarter period (π/2), should be at x=0, v=-1
        dt = np.pi / 2
        particle.evolve_harmonic(dt)
        
        assert np.isclose(particle.position, 0.0, atol=1e-10)
        assert np.isclose(particle.velocity, -1.0, atol=1e-10)
    
    def test_harmonic_particle_full_period(self):
        """Harmonic particle returns to initial state after full period."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.HARMONIC,
            mass=2.0,
            position=0.5,
            velocity=1.0,
            equilibrium_pos=0.0,
            spring_constant=8.0
        )
        
        # omega = sqrt(k/m) = sqrt(8/2) = 2
        # period T = 2π/ω = π
        omega = np.sqrt(particle.spring_constant / particle.mass)
        period = 2 * np.pi / omega
        
        initial_pos = particle.position
        initial_vel = particle.velocity
        
        particle.evolve_harmonic(period)
        
        assert np.isclose(particle.position, initial_pos, atol=1e-10)
        assert np.isclose(particle.velocity, initial_vel, atol=1e-10)
    
    def test_harmonic_particle_with_equilibrium_offset(self):
        """Harmonic particle oscillates around non-zero equilibrium."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.HARMONIC,
            mass=1.0,
            position=5.0,
            velocity=0.0,
            equilibrium_pos=5.0,
            spring_constant=4.0
        )
        
        # Starting at equilibrium with zero velocity
        # Should remain at equilibrium
        dt = 1.0
        particle.evolve_harmonic(dt)
        
        assert np.isclose(particle.position, 5.0, atol=1e-10)
        assert np.isclose(particle.velocity, 0.0, atol=1e-10)


class TestHarmonicEnergyConservation:
    """Tests for energy conservation in harmonic oscillator."""
    
    def test_harmonic_energy_conservation(self):
        """Energy is conserved to machine precision: |ΔE/E| < 10^-10."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.HARMONIC,
            mass=1.0,
            position=2.0,
            velocity=1.5,
            equilibrium_pos=0.0,
            spring_constant=3.0
        )
        
        initial_energy = particle.total_energy()
        
        # Evolve for multiple time steps
        for _ in range(100):
            particle.evolve_harmonic(0.1)
            current_energy = particle.total_energy()
            relative_error = abs(current_energy - initial_energy) / initial_energy
            assert relative_error < 1e-10, f"Energy conservation violated: ΔE/E = {relative_error}"
    
    def test_free_particle_energy_conservation(self):
        """Free particle kinetic energy is conserved."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=2.0,
            position=0.0,
            velocity=3.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        initial_energy = particle.kinetic_energy()
        
        for _ in range(100):
            particle.evolve_free(0.1)
            current_energy = particle.kinetic_energy()
            assert np.isclose(current_energy, initial_energy, atol=1e-10)


class TestCollisionTimePrediction:
    """Tests for collision time calculation."""
    
    def test_collision_time_free_free_approaching(self):
        """Two free particles approaching each other collide."""
        p1 = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=0.0,
            velocity=1.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        p2 = Particle(
            index=1,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=10.0,
            velocity=-1.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        t_collision = p1.time_to_collision(p2)
        
        # They should meet at x=5 after t=5
        assert np.isclose(t_collision, 5.0)
    
    def test_collision_time_free_free_receding(self):
        """Two free particles moving apart never collide."""
        p1 = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=0.0,
            velocity=-1.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        p2 = Particle(
            index=1,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=10.0,
            velocity=1.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        t_collision = p1.time_to_collision(p2)
        
        assert np.isinf(t_collision)
    
    def test_collision_time_free_free_same_velocity(self):
        """Particles with same velocity never collide."""
        p1 = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=0.0,
            velocity=2.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        p2 = Particle(
            index=1,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=5.0,
            velocity=2.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        t_collision = p1.time_to_collision(p2)
        
        assert np.isinf(t_collision)


class TestEnergyCalculations:
    """Tests for energy calculation methods."""
    
    def test_kinetic_energy(self):
        """Kinetic energy calculation: KE = (1/2) * m * v^2."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=2.0,
            position=0.0,
            velocity=3.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        expected_ke = 0.5 * 2.0 * 3.0**2
        assert np.isclose(particle.kinetic_energy(), expected_ke)
    
    def test_potential_energy_free(self):
        """Free particle has zero potential energy."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=100.0,
            velocity=5.0,
            equilibrium_pos=0.0,
            spring_constant=0.0
        )
        
        assert particle.potential_energy() == 0.0
    
    def test_potential_energy_harmonic(self):
        """Harmonic particle potential energy: PE = (1/2) * k * (x - x_eq)^2."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.HARMONIC,
            mass=1.0,
            position=3.0,
            velocity=0.0,
            equilibrium_pos=1.0,
            spring_constant=2.0
        )
        
        displacement = 3.0 - 1.0
        expected_pe = 0.5 * 2.0 * displacement**2
        assert np.isclose(particle.potential_energy(), expected_pe)
    
    def test_total_energy(self):
        """Total energy is sum of kinetic and potential."""
        particle = Particle(
            index=0,
            particle_type=ParticleType.HARMONIC,
            mass=1.0,
            position=2.0,
            velocity=3.0,
            equilibrium_pos=0.0,
            spring_constant=4.0
        )
        
        ke = particle.kinetic_energy()
        pe = particle.potential_energy()
        total = particle.total_energy()
        
        assert np.isclose(total, ke + pe)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
