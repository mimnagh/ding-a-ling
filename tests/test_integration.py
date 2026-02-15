"""Integration tests for full system energy conservation."""

import pytest
import numpy as np
import time
from src.core import (
    Chain, ChainConfig, BoundaryCondition, ParticleType,
    CollisionDetector, resolve_collision
)


def run_simulation(chain: Chain, n_collisions: int, max_time: float = 1e6):
    """
    Run event-driven simulation for specified number of collisions.
    
    Args:
        chain: Chain to simulate
        n_collisions: Number of collisions to process
        max_time: Maximum simulation time
        
    Returns:
        Tuple of (final_time, energy_history)
    """
    detector = CollisionDetector(chain)
    detector.build_event_queue()
    
    energy_history = [chain.total_energy()]
    collision_count = 0
    
    while collision_count < n_collisions and detector.current_time < max_time:
        # Get next event
        event = detector.get_next_event()
        
        if event is None:
            # No more collisions
            break
        
        # Evolve all particles to collision time
        dt = event.time - detector.current_time
        for particle in chain.particles:
            if particle.particle_type == ParticleType.FREE:
                particle.evolve_free(dt)
            else:
                particle.evolve_harmonic(dt)
        
        # Update current time
        detector.current_time = event.time
        
        # Resolve collision
        p1 = chain[event.particle_i]
        p2 = chain[event.particle_j]
        resolve_collision(p1, p2)
        
        # Update event queue
        detector.update_events_for_particles([event.particle_i, event.particle_j])
        
        # Track energy
        collision_count += 1
        if collision_count % 1000 == 0:
            energy_history.append(chain.total_energy())
    
    return detector.current_time, energy_history


def test_closed_system_energy_conservation_small():
    """Test energy conservation in a small closed system."""
    # Small test with N=10 for quick validation
    config = ChainConfig(
        n_particles=10,
        mass_free=1.0,
        mass_harmonic=1.0,
        spring_constant=1.0,
        spacing=1.0,
        boundary=BoundaryCondition.PERIODIC,
        temperature=1.0
    )
    chain = Chain(config)
    
    initial_energy = chain.total_energy()
    
    # Run for 1000 collisions
    final_time, energy_history = run_simulation(chain, n_collisions=1000)
    
    final_energy = chain.total_energy()
    
    # Check energy conservation
    relative_error = abs(final_energy - initial_energy) / initial_energy
    assert relative_error < 1e-8, f"Energy not conserved: ΔE/E = {relative_error}"
    
    # Check all intermediate energies
    for energy in energy_history:
        rel_err = abs(energy - initial_energy) / initial_energy
        assert rel_err < 1e-8, f"Energy drift detected: ΔE/E = {rel_err}"


@pytest.mark.slow
def test_closed_system_energy_conservation_large():
    """Test energy conservation in larger system with many collisions."""
    # N=100 chain as specified in requirements
    config = ChainConfig(
        n_particles=100,
        mass_free=1.0,
        mass_harmonic=1.0,
        spring_constant=1.0,
        spacing=1.0,
        boundary=BoundaryCondition.PERIODIC,
        temperature=1.0
    )
    chain = Chain(config)
    
    initial_energy = chain.total_energy()
    
    # Run for 10^6 collisions (this will take time)
    start_time = time.time()
    final_time, energy_history = run_simulation(chain, n_collisions=1_000_000)
    elapsed_time = time.time() - start_time
    
    final_energy = chain.total_energy()
    
    # Check energy conservation
    relative_error = abs(final_energy - initial_energy) / initial_energy
    assert relative_error < 1e-10, f"Energy not conserved: ΔE/E = {relative_error}"
    
    # Check all intermediate energies
    max_error = 0.0
    for energy in energy_history:
        rel_err = abs(energy - initial_energy) / initial_energy
        max_error = max(max_error, rel_err)
        assert rel_err < 1e-10, f"Energy drift detected: ΔE/E = {rel_err}"
    
    print(f"\nPerformance: {elapsed_time:.2f}s for 10^6 collisions")
    print(f"Maximum energy error: {max_error:.2e}")


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
def test_energy_conservation_various_epsilon(epsilon):
    """Test energy conservation with various spring constant values."""
    config = ChainConfig(
        n_particles=20,
        mass_free=1.0,
        mass_harmonic=1.0,
        spring_constant=epsilon,
        spacing=1.0,
        boundary=BoundaryCondition.PERIODIC,
        temperature=1.0
    )
    chain = Chain(config)
    
    initial_energy = chain.total_energy()
    
    # Run for 10000 collisions
    final_time, energy_history = run_simulation(chain, n_collisions=10_000)
    
    final_energy = chain.total_energy()
    
    # Check energy conservation
    relative_error = abs(final_energy - initial_energy) / initial_energy
    assert relative_error < 1e-9, f"Energy not conserved for ε={epsilon}: ΔE/E = {relative_error}"


def test_momentum_conservation_periodic():
    """Test that momentum is conserved in periodic boundary conditions with free particles only."""
    # Note: Momentum is only strictly conserved for free particles
    # Harmonic potentials can act as momentum sources/sinks
    config = ChainConfig(
        n_particles=10,
        mass_free=1.0,
        mass_harmonic=1.0,
        spring_constant=1.0,
        spacing=1.0,
        boundary=BoundaryCondition.PERIODIC,
        temperature=1.0
    )
    chain = Chain(config)
    
    # Replace all with free particles for this test
    from src.core import Particle, ParticleType
    for i in range(len(chain)):
        chain.particles[i] = Particle(
            index=i,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=i * 1.0,
            velocity=np.random.normal(0, 1.0),
            equilibrium_pos=i * 1.0,
            spring_constant=0.0
        )
    
    initial_momentum = chain.total_momentum()
    
    # Run simulation
    final_time, energy_history = run_simulation(chain, n_collisions=1000)
    
    final_momentum = chain.total_momentum()
    
    # Check momentum conservation (should be exact for free particles)
    assert abs(final_momentum - initial_momentum) < 1e-8
