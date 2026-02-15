# Project Structure

## Directory Organization

```
ding-a-ling/
├── src/
│   ├── core/           # Core physics: particles, chains, collisions
│   ├── simulation/     # Event-driven simulation engine, reservoirs
│   ├── analysis/       # Transport metrics, Lyapunov exponents
│   └── visualization/  # Plotting and animation tools
├── tests/              # Unit, integration, and property-based tests
├── examples/           # Jupyter notebooks and example scripts
├── docs/               # Documentation and paper references
└── .kiro/
    ├── specs/          # Feature specifications
    └── steering/       # Project guidelines (this file)
```

## Core Layer (`src/core/`)
Contains fundamental physics implementations:
- `particle.py`: Particle class with analytic evolution methods
- `chain.py`: Chain container with boundary conditions
- `collision.py`: Collision detection and resolution

## Simulation Layer (`src/simulation/`)
Event-driven dynamics engine:
- `engine.py`: Main simulation loop with event scheduling
- `reservoir.py`: Thermal baths and flux measurement

## Analysis Layer (`src/analysis/`)
Post-processing and diagnostics:
- `transport.py`: Conductivity extraction, regime classification
- `lyapunov.py`: Chaos metrics via tangent space evolution
- `scaling.py`: Finite-size scaling studies

## Visualization Layer (`src/visualization/`)
Plotting utilities for results presentation

## Architecture Pattern
Layered architecture: User Interface → Analysis → Simulation → Core Physics

Each layer depends only on layers below it. Core physics has no dependencies on simulation or analysis layers.
