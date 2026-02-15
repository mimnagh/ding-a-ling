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
- uv for fast dependency management and virtual environments

## Common Commands

```bash
# Setup virtual environment (first time)
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install package with dependencies
uv pip install -e ".[test]"

# Run tests
pytest tests/

# Run property-based tests
pytest tests/ -k property

# Run quick validation
python run_tests.py

# Launch Jupyter (after installing viz extras)
uv pip install -e ".[viz]"
jupyter notebook examples/
```

## Performance Targets
- N=1000 chain, 10^6 collisions in < 60 seconds
- Energy conservation: |ΔE/E| < 10^-10
- Leverage Numba JIT for hot loops (collision detection, particle evolution)
