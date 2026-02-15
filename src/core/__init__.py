"""Core physics module for ding-a-ling thermal transport simulator."""

from .particle import Particle, ParticleType
from .chain import Chain, ChainConfig, BoundaryCondition
from .collision import CollisionEvent, CollisionDetector, resolve_collision

__all__ = [
    'Particle', 'ParticleType', 
    'Chain', 'ChainConfig', 'BoundaryCondition',
    'CollisionEvent', 'CollisionDetector', 'resolve_collision'
]
