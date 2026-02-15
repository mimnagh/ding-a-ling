"""Collision detection and resolution for the ding-a-ling model."""

from dataclasses import dataclass
from typing import List, Optional
import heapq
import numpy as np

from .particle import Particle
from .chain import Chain


@dataclass(order=True)
class CollisionEvent:
    """
    Represents a collision event in the priority queue.
    
    Attributes:
        time: Time when collision occurs
        particle_i: Index of first particle
        particle_j: Index of second particle
        event_type: Type of collision ('particle-particle', 'wall', etc.)
    """
    time: float
    particle_i: int = -1
    particle_j: int = -1
    event_type: str = "particle-particle"
    
    def __post_init__(self):
        """Ensure proper ordering in priority queue."""
        # Make sure particle_i < particle_j for consistency
        if self.particle_i > self.particle_j and self.particle_j >= 0:
            self.particle_i, self.particle_j = self.particle_j, self.particle_i


class CollisionDetector:
    """
    Event-driven collision detection using priority queue.
    
    Maintains a heap of upcoming collision events and efficiently
    updates the queue as particles evolve.
    """
    
    def __init__(self, chain: Chain):
        """
        Initialize collision detector for a chain.
        
        Args:
            chain: Chain of particles to monitor
        """
        self.chain = chain
        self.event_queue: List[CollisionEvent] = []
        self.current_time = 0.0
        
    def find_next_collision(self) -> Optional[CollisionEvent]:
        """
        Find the next collision by scanning all particle pairs.
        
        Returns:
            CollisionEvent with earliest collision time, or None if no collisions
        """
        min_time = np.inf
        min_event = None
        
        n = len(self.chain)
        
        # Check all neighboring pairs
        for i in range(n):
            left, right = self.chain.get_neighbors(i)
            
            # Check collision with right neighbor
            if right is not None:
                t_collision = self.chain[i].time_to_collision(self.chain[right])
                if t_collision < min_time:
                    min_time = t_collision
                    min_event = CollisionEvent(
                        time=self.current_time + t_collision,
                        particle_i=i,
                        particle_j=right,
                        event_type="particle-particle"
                    )
        
        return min_event
    
    def build_event_queue(self) -> None:
        """
        Build initial event queue by finding all upcoming collisions.
        
        This scans all particle pairs and adds collision events to the heap.
        """
        self.event_queue = []
        
        n = len(self.chain)
        
        # Check all neighboring pairs
        for i in range(n):
            left, right = self.chain.get_neighbors(i)
            
            # Check collision with right neighbor (avoid double-counting)
            if right is not None and right > i:
                t_collision = self.chain[i].time_to_collision(self.chain[right])
                if t_collision < np.inf:
                    event = CollisionEvent(
                        time=self.current_time + t_collision,
                        particle_i=i,
                        particle_j=right,
                        event_type="particle-particle"
                    )
                    heapq.heappush(self.event_queue, event)
    
    def update_events_for_particles(self, particle_indices: List[int]) -> None:
        """
        Update collision events involving specified particles.
        
        After a collision, we need to recalculate collision times for
        the involved particles and their neighbors.
        
        Args:
            particle_indices: Indices of particles whose events need updating
        """
        # Remove old events involving these particles
        # (In practice, we'd mark them invalid; here we rebuild for simplicity)
        affected_particles = set(particle_indices)
        
        # Add neighbors to affected set
        for idx in particle_indices:
            left, right = self.chain.get_neighbors(idx)
            if left is not None:
                affected_particles.add(left)
            if right is not None:
                affected_particles.add(right)
        
        # Remove events involving affected particles
        self.event_queue = [
            event for event in self.event_queue
            if event.particle_i not in affected_particles 
            and event.particle_j not in affected_particles
        ]
        heapq.heapify(self.event_queue)
        
        # Add new events for affected particles
        for idx in affected_particles:
            left, right = self.chain.get_neighbors(idx)
            
            # Check collision with right neighbor
            if right is not None and right > idx:
                t_collision = self.chain[idx].time_to_collision(self.chain[right])
                if t_collision < np.inf:
                    event = CollisionEvent(
                        time=self.current_time + t_collision,
                        particle_i=idx,
                        particle_j=right,
                        event_type="particle-particle"
                    )
                    heapq.heappush(self.event_queue, event)
    
    def get_next_event(self) -> Optional[CollisionEvent]:
        """
        Get next event from priority queue.
        
        Returns:
            Next collision event, or None if queue is empty
        """
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)


def resolve_collision(particle_i: Particle, particle_j: Particle) -> None:
    """
    Resolve elastic collision between two particles.
    
    Uses conservation of momentum and energy for 1D elastic collision:
        v1' = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2' = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    
    Args:
        particle_i: First particle
        particle_j: Second particle
    """
    m1 = particle_i.mass
    m2 = particle_j.mass
    v1 = particle_i.velocity
    v2 = particle_j.velocity
    
    # Calculate new velocities using elastic collision formulas
    total_mass = m1 + m2
    
    v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / total_mass
    v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / total_mass
    
    # Update velocities
    particle_i.velocity = v1_new
    particle_j.velocity = v2_new
