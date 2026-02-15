"""Particle dynamics for the ding-a-ling model."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class ParticleType(Enum):
    """Type of particle in the chain."""
    FREE = "free"
    HARMONIC = "harmonic"


@dataclass
class Particle:
    """
    Single particle with position, velocity, and type-specific properties.
    
    Attributes:
        index: Particle index in the chain
        particle_type: FREE or HARMONIC
        mass: Particle mass
        position: Current position (1D scalar)
        velocity: Current velocity (1D scalar)
        equilibrium_pos: Equilibrium position for harmonic particles
        spring_constant: Spring constant k for harmonic particles (ignored for free)
    """
    index: int
    particle_type: ParticleType
    mass: float
    position: float
    velocity: float
    equilibrium_pos: float
    spring_constant: float = 0.0
    
    def evolve_free(self, dt: float) -> None:
        """
        Evolve free particle ballistically for time dt.
        
        Free particles move with constant velocity between collisions.
        
        Args:
            dt: Time step
        """
        self.position += self.velocity * dt
    
    def evolve_harmonic(self, dt: float) -> None:
        """
        Evolve harmonic particle using exact analytic solution.
        
        Harmonic oscillator solution:
            x(t) = x_eq + (x0 - x_eq) * cos(ω*t) + (v0/ω) * sin(ω*t)
            v(t) = -(x0 - x_eq) * ω * sin(ω*t) + v0 * cos(ω*t)
        
        where ω = sqrt(k/m), x0 and v0 are initial conditions.
        
        Args:
            dt: Time step
        """
        omega = np.sqrt(self.spring_constant / self.mass)
        
        # Initial displacement from equilibrium
        x0 = self.position - self.equilibrium_pos
        v0 = self.velocity
        
        # Compute trig functions
        cos_wt = np.cos(omega * dt)
        sin_wt = np.sin(omega * dt)
        
        # Update position and velocity
        self.position = self.equilibrium_pos + x0 * cos_wt + (v0 / omega) * sin_wt
        self.velocity = -x0 * omega * sin_wt + v0 * cos_wt
    
    def time_to_collision(self, other: 'Particle') -> float:
        """
        Calculate time until collision with another particle.
        
        For free-free collisions: solve x1(t) = x2(t)
        For free-harmonic or harmonic-harmonic: requires numerical root finding
        
        Args:
            other: Other particle
            
        Returns:
            Time until collision, or np.inf if no collision
        """
        if self.particle_type == ParticleType.FREE and other.particle_type == ParticleType.FREE:
            return self._time_to_collision_free_free(other)
        else:
            # For harmonic particles, collision detection is more complex
            # This is a simplified version - full implementation would use root finding
            return self._time_to_collision_general(other)
    
    def _time_to_collision_free_free(self, other: 'Particle') -> float:
        """
        Collision time for two free particles.
        
        Solve: x1 + v1*t = x2 + v2*t
        => t = (x2 - x1) / (v1 - v2)
        
        Args:
            other: Other free particle
            
        Returns:
            Time until collision, or np.inf if no collision
        """
        relative_velocity = self.velocity - other.velocity
        
        # No collision if moving apart or parallel
        if abs(relative_velocity) < 1e-10:
            return np.inf
        
        relative_position = other.position - self.position
        t_collision = relative_position / relative_velocity
        
        # Only return positive times (future collisions)
        return t_collision if t_collision > 0 else np.inf
    
    def _time_to_collision_general(self, other: 'Particle', dt: float = 0.01, 
                                   max_time: float = 1000.0) -> float:
        """
        General collision detection using time-stepping.
        
        This is a simplified implementation. A production version would use
        proper root-finding algorithms for harmonic particles.
        
        Args:
            other: Other particle
            dt: Time step for checking
            max_time: Maximum time to check
            
        Returns:
            Approximate time until collision, or np.inf if no collision found
        """
        # Create temporary copies to evolve
        p1_pos, p1_vel = self.position, self.velocity
        p2_pos, p2_vel = other.position, other.velocity
        
        prev_distance = abs(p2_pos - p1_pos)
        
        for step in range(int(max_time / dt)):
            t = step * dt
            
            # Evolve particles
            if self.particle_type == ParticleType.FREE:
                p1_pos = self.position + self.velocity * t
            else:
                omega = np.sqrt(self.spring_constant / self.mass)
                x0 = self.position - self.equilibrium_pos
                p1_pos = self.equilibrium_pos + x0 * np.cos(omega * t) + \
                        (self.velocity / omega) * np.sin(omega * t)
            
            if other.particle_type == ParticleType.FREE:
                p2_pos = other.position + other.velocity * t
            else:
                omega = np.sqrt(other.spring_constant / other.mass)
                x0 = other.position - other.equilibrium_pos
                p2_pos = other.equilibrium_pos + x0 * np.cos(omega * t) + \
                        (other.velocity / omega) * np.sin(omega * t)
            
            distance = abs(p2_pos - p1_pos)
            
            # Check if particles crossed (distance decreased then increased)
            if distance < 1e-6:  # Collision threshold
                return t
            
            prev_distance = distance
        
        return np.inf
    
    def kinetic_energy(self) -> float:
        """
        Calculate kinetic energy: KE = (1/2) * m * v^2
        
        Returns:
            Kinetic energy
        """
        return 0.5 * self.mass * self.velocity ** 2
    
    def potential_energy(self) -> float:
        """
        Calculate potential energy.
        
        For free particles: PE = 0
        For harmonic particles: PE = (1/2) * k * (x - x_eq)^2
        
        Returns:
            Potential energy
        """
        if self.particle_type == ParticleType.FREE:
            return 0.0
        else:
            displacement = self.position - self.equilibrium_pos
            return 0.5 * self.spring_constant * displacement ** 2
    
    def total_energy(self) -> float:
        """
        Calculate total energy: E = KE + PE
        
        Returns:
            Total energy
        """
        return self.kinetic_energy() + self.potential_energy()
