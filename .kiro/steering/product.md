# Product Overview

This is a Python-based physics simulation package for studying thermal conductivity in 1D chains of alternating free and harmonically-bound particles (the "ding-a-ling" model).

The system simulates how heat flows through a chain where particles alternate between:
- Free particles that move ballistically between collisions
- Harmonically bound particles that oscillate in quadratic potentials

The primary research goal is to reproduce and extend results from a physics paper on thermal transport, studying the transition between ballistic and diffusive transport regimes, and understanding how chaos affects thermal conductivity.

Key capabilities:
- Event-driven collision dynamics with exact energy conservation
- Open systems with thermal reservoirs for conductivity measurements
- Closed periodic systems for Lyapunov exponent (chaos) calculations
- Finite-size scaling analysis to identify thermodynamic limits
- Future extension to 2D lattices

Target hardware: Mac mini M2 Pro with performance optimizations via Numba JIT compilation.
