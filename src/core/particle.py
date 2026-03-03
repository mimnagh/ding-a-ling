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
    
    def _time_to_collision_general(self, other: 'Particle',
                                   max_time: float = 100.0) -> float:
        """
        General collision detection using vectorized sign-change detection.

        Computes the signed gap (other.pos - self.pos) at many time points
        using numpy, finds the first sign change, then refines with bisection.

        Args:
            other: Other particle
            max_time: Maximum time to check

        Returns:
            Time until collision, or np.inf if no collision found
        """
        def _gap_scalar(t):
            """Signed gap at scalar time t."""
            if self.particle_type == ParticleType.FREE:
                p1 = self.position + self.velocity * t
            else:
                omega1 = np.sqrt(self.spring_constant / self.mass)
                x01 = self.position - self.equilibrium_pos
                p1 = (self.equilibrium_pos + x01 * np.cos(omega1 * t)
                      + (self.velocity / omega1) * np.sin(omega1 * t))
            if other.particle_type == ParticleType.FREE:
                p2 = other.position + other.velocity * t
            else:
                omega2 = np.sqrt(other.spring_constant / other.mass)
                x02 = other.position - other.equilibrium_pos
                p2 = (other.equilibrium_pos + x02 * np.cos(omega2 * t)
                      + (other.velocity / omega2) * np.sin(omega2 * t))
            return p2 - p1

        # Vectorized gap over array of times.
        # Start at dt (not 0) to skip post-collision states where gap ≈ 0
        # but particles are separating.
        dt = 0.05
        ts = np.arange(dt, max_time + dt, dt)
        # Build position arrays
        if self.particle_type == ParticleType.FREE:
            p1 = self.position + self.velocity * ts
        else:
            omega1 = np.sqrt(self.spring_constant / self.mass)
            x01 = self.position - self.equilibrium_pos
            p1 = (self.equilibrium_pos + x01 * np.cos(omega1 * ts)
                  + (self.velocity / omega1) * np.sin(omega1 * ts))
        if other.particle_type == ParticleType.FREE:
            p2 = other.position + other.velocity * ts
        else:
            omega2 = np.sqrt(other.spring_constant / other.mass)
            x02 = other.position - other.equilibrium_pos
            p2 = (other.equilibrium_pos + x02 * np.cos(omega2 * ts)
                  + (other.velocity / omega2) * np.sin(omega2 * ts))

        gaps = p2 - p1

        # --- Detect sign-change crossings ---
        signs = np.sign(gaps)
        crossings = np.where(signs[1:] * signs[:-1] < 0)[0]

        # --- Detect tangential touches (gap approaches 0 without crossing) ---
        # Find local minima where |gap| is very small
        tangent_indices = np.array([], dtype=int)
        if len(gaps) > 2:
            local_min = (gaps[1:-1] <= gaps[:-2]) & (gaps[1:-1] <= gaps[2:])
            near_zero = np.abs(gaps[1:-1]) < dt  # loose pre-filter
            tangent_candidates = np.where(local_min & near_zero)[0] + 1
            # Refine each candidate: check if true minimum is ≈ 0
            verified = []
            for k in int(tangent_candidates) if tangent_candidates.ndim == 0 else tangent_candidates:
                t_lo = ts[max(k - 1, 0)]
                t_hi = ts[min(k + 1, len(ts) - 1)]
                # Golden-section-like refinement
                for _ in range(30):
                    t_m1 = t_lo + (t_hi - t_lo) * 0.382
                    t_m2 = t_lo + (t_hi - t_lo) * 0.618
                    if abs(_gap_scalar(t_m1)) < abs(_gap_scalar(t_m2)):
                        t_hi = t_m2
                    else:
                        t_lo = t_m1
                t_min = (t_lo + t_hi) / 2
                if abs(_gap_scalar(t_min)) < 1e-8:
                    verified.append(k)
            tangent_indices = np.array(verified, dtype=int)

        # Combine candidates and return earliest collision time
        all_candidates = np.concatenate([crossings, tangent_indices])
        if len(all_candidates) == 0:
            return np.inf

        idx = int(all_candidates.min())

        # For crossings: bisect to find exact root
        if idx in crossings:
            t_lo, t_hi = ts[idx], ts[idx + 1]
            g_lo = float(gaps[idx])
            for _ in range(50):
                t_mid = (t_lo + t_hi) / 2
                g_mid = _gap_scalar(t_mid)
                if abs(g_mid) < 1e-12:
                    return float(t_mid)
                if g_mid * g_lo < 0:
                    t_hi = t_mid
                else:
                    t_lo = t_mid
                    g_lo = g_mid
            return float((t_lo + t_hi) / 2)
        else:
            # Tangential touch: find minimum via golden section
            t_lo = ts[max(idx - 1, 0)]
            t_hi = ts[min(idx + 1, len(ts) - 1)]
            for _ in range(50):
                t_m1 = t_lo + (t_hi - t_lo) * 0.382
                t_m2 = t_lo + (t_hi - t_lo) * 0.618
                if abs(_gap_scalar(t_m1)) < abs(_gap_scalar(t_m2)):
                    t_hi = t_m2
                else:
                    t_lo = t_m1
            return float((t_lo + t_hi) / 2)
    
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
