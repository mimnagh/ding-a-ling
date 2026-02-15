"""Chain configuration and container for the ding-a-ling model."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from .particle import Particle, ParticleType


class BoundaryCondition(Enum):
    """Boundary condition for the chain."""
    OPEN = "open"
    PERIODIC = "periodic"


@dataclass
class ChainConfig:
    """
    Configuration parameters for initializing a chain.
    
    Attributes:
        n_particles: Number of particles in the chain
        mass_free: Mass of free particles
        mass_harmonic: Mass of harmonic particles
        spring_constant: Spring constant k for harmonic particles
        spacing: Initial spacing between particles
        boundary: Boundary condition (OPEN or PERIODIC)
        temperature: Initial temperature for thermal initialization (optional)
    """
    n_particles: int
    mass_free: float = 1.0
    mass_harmonic: float = 1.0
    spring_constant: float = 1.0
    spacing: float = 1.0
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC
    temperature: Optional[float] = None


class Chain:
    """
    Container for a chain of alternating free and harmonic particles.
    
    The chain alternates between free particles (odd indices) and 
    harmonically bound particles (even indices).
    """
    
    def __init__(self, config: ChainConfig):
        """
        Initialize chain with alternating particle types.
        
        Args:
            config: Chain configuration parameters
        """
        self.config = config
        self.particles: List[Particle] = []
        
        # Initialize particles with alternating types
        for i in range(config.n_particles):
            # Alternate: even indices are harmonic, odd indices are free
            particle_type = ParticleType.HARMONIC if i % 2 == 0 else ParticleType.FREE
            mass = config.mass_harmonic if particle_type == ParticleType.HARMONIC else config.mass_free
            
            # Initial position: evenly spaced
            position = i * config.spacing
            equilibrium_pos = position
            
            # Initial velocity: zero or thermal
            velocity = 0.0
            if config.temperature is not None:
                # Maxwell-Boltzmann distribution: v ~ N(0, sqrt(kT/m))
                velocity = np.random.normal(0, np.sqrt(config.temperature / mass))
            
            particle = Particle(
                index=i,
                particle_type=particle_type,
                mass=mass,
                position=position,
                velocity=velocity,
                equilibrium_pos=equilibrium_pos,
                spring_constant=config.spring_constant if particle_type == ParticleType.HARMONIC else 0.0
            )
            
            self.particles.append(particle)
    
    def total_energy(self) -> float:
        """
        Calculate total energy of the chain.
        
        Returns:
            Sum of kinetic and potential energies of all particles
        """
        return sum(p.total_energy() for p in self.particles)
    
    def total_momentum(self) -> float:
        """
        Calculate total momentum of the chain.
        
        Returns:
            Sum of momenta (m*v) of all particles
        """
        return sum(p.mass * p.velocity for p in self.particles)
    
    def local_temperature(self, index: int, window: int = 5) -> float:
        """
        Calculate local temperature around a particle.
        
        Temperature is estimated from kinetic energy of nearby particles:
        T = <KE> / (k_B / 2) where we set k_B = 1
        
        Args:
            index: Particle index
            window: Number of neighbors on each side to include
            
        Returns:
            Local temperature estimate
        """
        # Determine window bounds
        start = max(0, index - window)
        end = min(len(self.particles), index + window + 1)
        
        # Calculate average kinetic energy in window
        ke_sum = sum(self.particles[i].kinetic_energy() for i in range(start, end))
        n_particles = end - start
        
        # Temperature from equipartition: <KE> = (1/2) k_B T
        # With k_B = 1: T = 2 * <KE>
        avg_ke = ke_sum / n_particles if n_particles > 0 else 0.0
        return 2.0 * avg_ke
    
    def get_neighbors(self, index: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Get indices of neighboring particles.
        
        Args:
            index: Particle index
            
        Returns:
            Tuple of (left_neighbor_index, right_neighbor_index)
            None if no neighbor exists (open boundaries)
        """
        n = len(self.particles)
        
        if self.config.boundary == BoundaryCondition.OPEN:
            left = index - 1 if index > 0 else None
            right = index + 1 if index < n - 1 else None
        else:  # PERIODIC
            left = (index - 1) % n
            right = (index + 1) % n
        
        return (left, right)
    
    def set_thermal_velocities(self, temperature: float) -> None:
        """
        Set particle velocities from Maxwell-Boltzmann distribution.
        
        Args:
            temperature: Target temperature
        """
        for particle in self.particles:
            sigma = np.sqrt(temperature / particle.mass)
            particle.velocity = np.random.normal(0, sigma)
    
    def set_custom_state(self, positions: List[float], velocities: List[float]) -> None:
        """
        Set custom positions and velocities for all particles.
        
        Args:
            positions: List of particle positions
            velocities: List of particle velocities
            
        Raises:
            ValueError: If list lengths don't match number of particles
        """
        if len(positions) != len(self.particles) or len(velocities) != len(self.particles):
            raise ValueError(
                f"Position/velocity lists must have length {len(self.particles)}, "
                f"got {len(positions)} and {len(velocities)}"
            )
        
        for i, particle in enumerate(self.particles):
            particle.position = positions[i]
            particle.velocity = velocities[i]
    
    def __len__(self) -> int:
        """Return number of particles in chain."""
        return len(self.particles)
    
    def __getitem__(self, index: int) -> Particle:
        """Access particle by index."""
        return self.particles[index]
    
    def __repr__(self) -> str:
        """String representation of chain."""
        return (
            f"Chain(n={len(self.particles)}, "
            f"boundary={self.config.boundary.value}, "
            f"E={self.total_energy():.4f}, "
            f"p={self.total_momentum():.4f})"
        )
