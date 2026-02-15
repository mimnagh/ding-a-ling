# API Documentation

## Installation

### Package Installation

```bash
# Basic installation
pip install -e .

# Development installation (includes pytest, hypothesis, matplotlib, jupyter, black, ruff)
pip install -e ".[dev]"

# Testing only
pip install -e ".[test]"

# Visualization only
pip install -e ".[viz]"
```

### Requirements

- Python >= 3.9
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Numba >= 0.57.0

## Core Module (`src.core`)

### Particle Class

The `Particle` class represents a single particle in the ding-a-ling chain with position, velocity, and type-specific properties.

#### Constructor

```python
Particle(
    index: int,
    particle_type: ParticleType,
    mass: float,
    position: float,
    velocity: float,
    equilibrium_pos: float,
    spring_constant: float = 0.0
)
```

**Parameters:**
- `index`: Particle index in the chain
- `particle_type`: Either `ParticleType.FREE` or `ParticleType.HARMONIC`
- `mass`: Particle mass (typically 1.0)
- `position`: Current position (1D scalar)
- `velocity`: Current velocity (1D scalar)
- `equilibrium_pos`: Equilibrium position for harmonic particles
- `spring_constant`: Spring constant k for harmonic particles (ignored for free particles)

#### Methods

##### `evolve_free(dt: float) -> None`

Evolve free particle ballistically for time `dt`. Free particles move with constant velocity between collisions.

**Example:**
```python
particle = Particle(0, ParticleType.FREE, 1.0, 0.0, 1.0, 0.0)
particle.evolve_free(dt=0.5)
# particle.position is now 0.5
```

##### `evolve_harmonic(dt: float) -> None`

Evolve harmonic particle using exact analytic solution. Uses the harmonic oscillator equations:
- x(t) = x_eq + (x0 - x_eq) * cos(ω*t) + (v0/ω) * sin(ω*t)
- v(t) = -(x0 - x_eq) * ω * sin(ω*t) + v0 * cos(ω*t)

where ω = sqrt(k/m).

**Example:**
```python
particle = Particle(0, ParticleType.HARMONIC, 1.0, 1.0, 0.0, 1.0, 1.0)
particle.evolve_harmonic(dt=np.pi/2)  # Quarter period
```

##### `time_to_collision(other: Particle) -> float`

Calculate time until collision with another particle.

**Returns:** Time until collision, or `np.inf` if no collision will occur.

**Note:** For free-free collisions, uses exact analytic solution. For collisions involving harmonic particles, uses numerical approximation.

**Example:**
```python
p1 = Particle(0, ParticleType.FREE, 1.0, 0.0, 1.0, 0.0)
p2 = Particle(1, ParticleType.FREE, 1.0, 2.0, -0.5, 2.0)
t_collision = p1.time_to_collision(p2)
```

##### `kinetic_energy() -> float`

Calculate kinetic energy: KE = (1/2) * m * v²

**Returns:** Kinetic energy

##### `potential_energy() -> float`

Calculate potential energy:
- For free particles: PE = 0
- For harmonic particles: PE = (1/2) * k * (x - x_eq)²

**Returns:** Potential energy

##### `total_energy() -> float`

Calculate total energy: E = KE + PE

**Returns:** Total energy

### ParticleType Enum

```python
class ParticleType(Enum):
    FREE = "free"
    HARMONIC = "harmonic"
```

Defines the two types of particles in the ding-a-ling model:
- `FREE`: Particles that move ballistically between collisions
- `HARMONIC`: Particles bound by harmonic springs to equilibrium positions

## Usage Examples

### Creating an Alternating Chain

```python
from src.core import Particle, ParticleType

particles = []
for i in range(10):
    if i % 2 == 0:
        # Free particle
        p = Particle(
            index=i,
            particle_type=ParticleType.FREE,
            mass=1.0,
            position=float(i),
            velocity=0.0,
            equilibrium_pos=float(i),
            spring_constant=0.0
        )
    else:
        # Harmonic particle
        p = Particle(
            index=i,
            particle_type=ParticleType.HARMONIC,
            mass=1.0,
            position=float(i),
            velocity=0.0,
            equilibrium_pos=float(i),
            spring_constant=1.0
        )
    particles.append(p)
```

### Energy Conservation Check

```python
# Create harmonic particle
p = Particle(0, ParticleType.HARMONIC, 1.0, 0.5, 1.0, 0.0, 1.0)

# Record initial energy
E_initial = p.total_energy()

# Evolve for one period
period = 2 * np.pi / np.sqrt(1.0 / 1.0)
p.evolve_harmonic(period)

# Check energy conservation
E_final = p.total_energy()
assert abs(E_final - E_initial) < 1e-10
```

### Collision Detection

```python
# Two free particles approaching each other
p1 = Particle(0, ParticleType.FREE, 1.0, 0.0, 1.0, 0.0)
p2 = Particle(1, ParticleType.FREE, 1.0, 5.0, -2.0, 5.0)

# Find collision time
t_collision = p1.time_to_collision(p2)
print(f"Particles will collide in {t_collision:.2f} time units")

# Evolve to collision
p1.evolve_free(t_collision)
p2.evolve_free(t_collision)

# Verify they're at same position
assert abs(p1.position - p2.position) < 1e-6
```

## Testing

### Test Suite (`tests/test_particle.py`)

Comprehensive unit tests verify all particle functionality:

#### TestFreeParticleEvolution
- `test_free_particle_evolution`: Verifies ballistic motion with constant velocity
- `test_free_particle_negative_velocity`: Tests backward motion

#### TestHarmonicParticleEvolution
- `test_harmonic_particle_evolution`: Validates analytic solution at quarter period
- `test_harmonic_particle_full_period`: Confirms return to initial state after full period
- `test_harmonic_particle_with_equilibrium_offset`: Tests oscillation around non-zero equilibrium

#### TestHarmonicEnergyConservation
- `test_harmonic_energy_conservation`: Verifies |ΔE/E| < 10⁻¹⁰ over 100 time steps
- `test_free_particle_energy_conservation`: Confirms kinetic energy conservation for free particles

#### TestCollisionTimePrediction
- `test_collision_time_free_free_approaching`: Validates collision time for approaching particles
- `test_collision_time_free_free_receding`: Confirms no collision for receding particles
- `test_collision_time_free_free_same_velocity`: Tests parallel motion case

#### TestEnergyCalculations
- `test_kinetic_energy`: Verifies KE = (1/2) * m * v²
- `test_potential_energy_free`: Confirms PE = 0 for free particles
- `test_potential_energy_harmonic`: Validates PE = (1/2) * k * (x - x_eq)²
- `test_total_energy`: Tests E = KE + PE

**Running Tests:**
```bash
# Run all particle tests
pytest tests/test_particle.py -v

# Run specific test class
pytest tests/test_particle.py::TestHarmonicEnergyConservation -v

# Run with coverage
pytest tests/test_particle.py --cov=src.core.particle

# Run property-based tests (when available)
pytest -k property -v
```

**Test Configuration:**

The project uses pytest with the following configuration (from `pyproject.toml`):
- Test discovery: `tests/test_*.py`
- Verbose output enabled by default
- Strict marker mode

**Code Quality:**

```bash
# Format code with Black (line length: 100)
black src/ tests/

# Lint with Ruff
ruff check src/ tests/
```

## Implementation Notes

### Numerical Precision

- Free particle evolution is exact (no discretization error)
- Harmonic particle evolution uses analytic solution (machine precision)
- Energy conservation: |ΔE/E| < 10^-10 for harmonic particles (verified by tests)
- All energy conservation tests pass with tolerance < 10^-10

### Performance Considerations

- All methods are designed for Numba JIT compilation
- Collision detection for harmonic particles uses simplified time-stepping (will be optimized in future versions)
- For production use, consider vectorizing operations across multiple particles

### Future API Extensions

The following modules are planned:
- `src.core.chain`: Chain container with boundary conditions
- `src.core.collision`: Collision detection and resolution
- `src.simulation.engine`: Event-driven simulation loop
- `src.simulation.reservoir`: Thermal baths and flux measurement
- `src.analysis.transport`: Conductivity extraction
- `src.analysis.lyapunov`: Chaos diagnostics
