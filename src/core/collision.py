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
    
    def __init__(
        self,
        chain: Chain,
        left_wall: Optional[float] = None,
        right_wall: Optional[float] = None,
    ):
        """
        Initialize collision detector for a chain.

        Args:
            chain: Chain of particles to monitor
            left_wall: Position of a hard elastic wall left of particle 0,
                or None for no wall.
            right_wall: Position of a hard elastic wall right of the last
                particle, or None for no wall.
        """
        self.chain = chain
        self.left_wall = left_wall
        self.right_wall = right_wall
        self.event_queue: List[CollisionEvent] = []
        self.current_time = 0.0

    def _time_to_wall(self, idx: int) -> float:
        """
        Time until the end particle *idx* reaches its wall, or np.inf.

        Only particle 0 (left wall) and particle n-1 (right wall) can hit
        walls; chain ordering keeps interior particles away from them.
        """
        n = len(self.chain)
        if idx == 0 and self.left_wall is not None:
            wall = self.left_wall
        elif idx == n - 1 and self.right_wall is not None:
            wall = self.right_wall
        else:
            return np.inf

        p = self.chain[idx]

        # Outward = toward this wall.  Only motion INTO the wall counts as
        # a hit: after a reflection float error can leave the particle a
        # hair past the wall plane, and predicting the "crossing" back
        # through it would schedule a zero-advance event loop.
        outward = -1.0 if idx == 0 else 1.0

        from .particle import ParticleType
        if p.particle_type == ParticleType.FREE:
            if p.velocity * outward <= 1e-15:
                return np.inf
            dt = (wall - p.position) / p.velocity
            return dt if dt > 1e-12 else np.inf

        # Harmonic particle: hits the wall only if its amplitude reaches it.
        omega = np.sqrt(p.spring_constant / p.mass)
        x0 = p.position - p.equilibrium_pos
        v0 = p.velocity
        amplitude = np.sqrt(x0 ** 2 + (v0 / omega) ** 2)
        if abs(wall - p.equilibrium_pos) > amplitude:
            return np.inf

        def offset(t):
            return (
                p.equilibrium_pos
                + x0 * np.cos(omega * t)
                + (v0 / omega) * np.sin(omega * t)
                - wall
            )

        # One full period suffices: the motion is periodic.
        period = 2 * np.pi / omega
        eps = 1e-9
        ts = np.linspace(eps, period, 257)
        fs = offset(ts)
        sign_changes = np.where(fs[:-1] * fs[1:] < 0)[0]
        from scipy.optimize import brentq
        for k in sign_changes:
            t_root = float(brentq(offset, ts[k], ts[k + 1], xtol=1e-12))
            v_root = (
                -x0 * omega * np.sin(omega * t_root)
                + v0 * np.cos(omega * t_root)
            )
            # Accept only roots where the particle moves INTO the wall;
            # crossings back through the plane (float error can leave it
            # marginally outside after a reflection) are not hits.
            if v_root * outward > 1e-15:
                return t_root
        return np.inf

    def _push_wall_event(self, idx: int) -> None:
        """Queue a wall-reflection event for end particle *idx*, if any."""
        t_wall = self._time_to_wall(idx)
        if t_wall < np.inf:
            heapq.heappush(
                self.event_queue,
                CollisionEvent(
                    time=self.current_time + t_wall,
                    particle_i=idx,
                    particle_j=-1,
                    event_type="wall",
                ),
            )
        
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
        
        # Check all neighboring pairs (each pair checked once via right neighbor)
        for i in range(n):
            left, right = self.chain.get_neighbors(i)

            if right is not None:
                t_collision = self.chain[i].time_to_collision(self.chain[right])
                if t_collision < np.inf:
                    event = CollisionEvent(
                        time=self.current_time + t_collision,
                        particle_i=i,
                        particle_j=right,
                        event_type="particle-particle"
                    )
                    heapq.heappush(self.event_queue, event)

        # Wall events for the end particles
        self._push_wall_event(0)
        self._push_wall_event(n - 1)

    def update_events_for_particles(self, particle_indices: List[int]) -> None:
        """
        Update collision events involving specified particles.
        
        After a collision, we need to recalculate collision times for
        the involved particles and their neighbors.
        
        Args:
            particle_indices: Indices of particles whose events need updating
        """
        # Only events involving particles whose state (velocity) changed are
        # stale.  Events between untouched neighbours are still valid and
        # MUST stay queued: removing them without re-adding silently loses
        # collisions and lets particles pass through each other.
        changed = set(particle_indices)

        self.event_queue = [
            event for event in self.event_queue
            if event.particle_i not in changed
            and event.particle_j not in changed
        ]
        heapq.heapify(self.event_queue)

        # Re-predict both neighbour pairs of every changed particle.
        pairs = set()
        for idx in changed:
            left, right = self.chain.get_neighbors(idx)
            if left is not None:
                pairs.add((min(left, idx), max(left, idx)))
            if right is not None:
                pairs.add((min(idx, right), max(idx, right)))

        for i, j in pairs:
            t_collision = self.chain[i].time_to_collision(self.chain[j])
            if t_collision < np.inf:
                event = CollisionEvent(
                    time=self.current_time + t_collision,
                    particle_i=i,
                    particle_j=j,
                    event_type="particle-particle"
                )
                heapq.heappush(self.event_queue, event)

        # Re-predict wall events for any changed end particle
        n = len(self.chain)
        if 0 in changed:
            self._push_wall_event(0)
        if (n - 1) in changed:
            self._push_wall_event(n - 1)
    
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
