# Technology Stack

## Language & Core Libraries
- Python 3.x
- NumPy for numerical computations
- Numba for JIT compilation and performance optimization (ARM NEON instructions)
- SciPy for scientific computing utilities

## Testing
- pytest for unit and integration tests
- Hypothesis for property-based testing (physics invariants)

## Visualization
- Matplotlib for plotting and animation
- Jupyter notebooks for interactive analysis

## Build System
- pyproject.toml for package configuration
- pip for dependency management

## Common Commands

Currently the project is in early setup phase. Standard commands will include:

```bash
# Installation
pip install -e .

# Run tests
pytest tests/

# Run property-based tests
pytest tests/ -k property

# Launch Jupyter
jupyter notebook examples/
```

## Performance Targets
- N=1000 chain, 10^6 collisions in < 60 seconds
- Energy conservation: |ΔE/E| < 10^-10
- Leverage Numba JIT for hot loops (collision detection, particle evolution)
