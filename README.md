# Ding-a-Ling Model: Thermal Conductivity Simulation

A Python implementation of the alternating free-harmonic particle chain model for studying thermal conductivity and the transition between ballistic and diffusive transport regimes.

## Overview

This project simulates a 1D chain of particles alternating between:
- **Free particles**: move ballistically between collisions
- **Harmonically bound particles**: oscillate in quadratic potentials

The system exhibits rich physics including:
- Fourier's law emergence in the thermodynamic limit
- Ballistic ↔ diffusive transport crossover
- Chaos-controlled finite-size effects
- Temperature-dependent thermal conductivity

## Key Features

- **Event-driven dynamics**: exact collision detection and handling
- **Dual configurations**: 
  - Open chains with thermal reservoirs (conductivity measurements)
  - Periodic rings (Lyapunov exponent calculations)
- **Transport diagnostics**: heat flux, temperature profiles, conductivity scaling
- **Chaos metrics**: maximum Lyapunov exponent computation

## Project Structure

```
ding-a-ling/
├── src/
│   ├── core/           # Core physics and particle dynamics
│   ├── simulation/     # Event-driven simulation engine
│   ├── analysis/       # Transport and chaos diagnostics
│   └── visualization/  # Plotting and animation tools
├── tests/              # Unit and integration tests
├── examples/           # Example simulations and notebooks
└── docs/               # Documentation and paper reference
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.simulation import DingALingSimulator
from src.core import ChainConfig

# Configure system
config = ChainConfig(
    N=100,              # chain length
    epsilon=1.0,        # energy parameter
    T_hot=1.5,          # hot reservoir temperature
    T_cold=0.5          # cold reservoir temperature
)

# Run simulation
sim = DingALingSimulator(config)
results = sim.run(total_time=1000.0)

# Analyze
print(f"Thermal conductivity: {results.conductivity:.4f}")
```

## Physics Parameters

- **ε (epsilon)**: Dimensionless energy parameter = E/(v²ℓ₀²)
  - Low ε: stiff springs, strong chaos, diffusive transport
  - High ε: weak springs, near-integrable, ballistic transport
  - Critical crossover: ε_c ≈ 1-10 (system-dependent)

## References

Based on: "Thermal conductivity in a chain of alternating masses with on-site potentials"
(See `docs/` for full paper)
# ding-a-ling
