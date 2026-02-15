"""Tests for Chain class."""

import pytest
import numpy as np
from src.core import Chain, ChainConfig, BoundaryCondition, ParticleType


def test_chain_initialization():
    """Test basic chain initialization."""
    config = ChainConfig(n_particles=10, spacing=1.0)
    chain = Chain(config)
    
    assert len(chain) == 10
    assert chain.config.n_particles == 10
    
    # Check alternating particle types
    for i in range(10):
        if i % 2 == 0:
            assert chain[i].particle_type == ParticleType.HARMONIC
        else:
            assert chain[i].particle_type == ParticleType.FREE


def test_chain_energy():
    """Test energy calculations."""
    config = ChainConfig(n_particles=4, temperature=None)
    chain = Chain(config)
    
    # Zero velocity -> only potential energy from harmonic particles
    initial_energy = chain.total_energy()
    assert initial_energy >= 0
    
    # Set velocities
    chain.set_thermal_velocities(temperature=1.0)
    energy_with_velocity = chain.total_energy()
    assert energy_with_velocity > initial_energy


def test_chain_momentum():
    """Test momentum calculations."""
    config = ChainConfig(n_particles=4, temperature=None)
    chain = Chain(config)
    
    # Zero velocity -> zero momentum
    assert chain.total_momentum() == 0.0
    
    # Set custom velocities
    chain.set_custom_state(
        positions=[0.0, 1.0, 2.0, 3.0],
        velocities=[1.0, -1.0, 1.0, -1.0]
    )
    # Momentum should be zero (symmetric velocities)
    assert abs(chain.total_momentum()) < 1e-10


def test_chain_neighbors_periodic():
    """Test neighbor finding with periodic boundaries."""
    config = ChainConfig(n_particles=5, boundary=BoundaryCondition.PERIODIC)
    chain = Chain(config)
    
    # Middle particle
    left, right = chain.get_neighbors(2)
    assert left == 1
    assert right == 3
    
    # Edge particles wrap around
    left, right = chain.get_neighbors(0)
    assert left == 4
    assert right == 1
    
    left, right = chain.get_neighbors(4)
    assert left == 3
    assert right == 0


def test_chain_neighbors_open():
    """Test neighbor finding with open boundaries."""
    config = ChainConfig(n_particles=5, boundary=BoundaryCondition.OPEN)
    chain = Chain(config)
    
    # Middle particle
    left, right = chain.get_neighbors(2)
    assert left == 1
    assert right == 3
    
    # Edge particles have None neighbors
    left, right = chain.get_neighbors(0)
    assert left is None
    assert right == 1
    
    left, right = chain.get_neighbors(4)
    assert left == 3
    assert right is None


def test_local_temperature():
    """Test local temperature calculation."""
    config = ChainConfig(n_particles=10, temperature=1.0)
    chain = Chain(config)
    
    # Temperature should be roughly 1.0 (with statistical variation)
    temp = chain.local_temperature(5, window=5)
    assert 0.1 < temp < 10.0  # Broad range due to randomness


def test_custom_state():
    """Test setting custom positions and velocities."""
    config = ChainConfig(n_particles=3)
    chain = Chain(config)
    
    positions = [0.5, 1.5, 2.5]
    velocities = [0.1, 0.2, 0.3]
    
    chain.set_custom_state(positions, velocities)
    
    for i in range(3):
        assert chain[i].position == positions[i]
        assert chain[i].velocity == velocities[i]


def test_custom_state_invalid_length():
    """Test that custom state rejects wrong-length lists."""
    config = ChainConfig(n_particles=3)
    chain = Chain(config)
    
    with pytest.raises(ValueError):
        chain.set_custom_state([0.0, 1.0], [0.0, 1.0, 2.0])
