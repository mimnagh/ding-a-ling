#!/usr/bin/env python3
"""Simple test runner to validate particle tests."""

import sys
import numpy as np
from src.core.particle import Particle, ParticleType


def test_free_particle():
    """Test free particle evolution."""
    print("Testing free particle evolution...")
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
    
    assert np.isclose(particle.position, initial_pos + initial_vel * dt)
    assert np.isclose(particle.velocity, initial_vel)
    print("✓ Free particle test passed")


def test_harmonic_energy_conservation():
    """Test energy conservation in harmonic oscillator."""
    print("Testing harmonic energy conservation...")
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
    for i in range(100):
        particle.evolve_harmonic(0.1)
        current_energy = particle.total_energy()
        relative_error = abs(current_energy - initial_energy) / initial_energy
        assert relative_error < 1e-10, f"Energy conservation violated at step {i}: ΔE/E = {relative_error}"
    
    print(f"✓ Energy conservation test passed (|ΔE/E| < 10^-10)")


def test_collision_time():
    """Test collision time calculation."""
    print("Testing collision time prediction...")
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
    assert np.isclose(t_collision, 5.0)
    print("✓ Collision time test passed")


if __name__ == "__main__":
    try:
        test_free_particle()
        test_harmonic_energy_conservation()
        test_collision_time()
        print("\n✅ All basic tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
