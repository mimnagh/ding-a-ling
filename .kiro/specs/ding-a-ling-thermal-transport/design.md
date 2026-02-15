# Design: Ding-a-Ling Thermal Transport Simulator

## Architecture Overview

The system follows a layered architecture optimized for scientific computing:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│              (Jupyter Notebooks, Scripts)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Analysis & Visualization                  │
│     (Transport metrics, Lyapunov, Plotting, Regime ID)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Engine                         │
│        (Event scheduler, Time evolution, Observables)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Physics Layer                        │
│     (Particles, Collisions, Chain, Reservoirs)              │
└─────────────────────────────────────────────────────────────┘
```

## Core Physics Layer

### 1. Particle System

**File**: `src/core/particle.py`

```python
@dataclass
class Particle:
    """Single particle with position, velocity, and type-specific properties."""
    index: int
    type: ParticleType  # FREE or HARMONIC
    mass: float
    position: float  # 1D: scalar, 2D: np.array([x, y])
    velocity: float  # 1D: scalar, 2D: np.array([vx, vy])
    equilibrium_pos: float
    spring_constant: float
    
    # Core methods
    def evolve_free(self, dt: float) -> None
    def evolve_harmonic(self, dt: float) -> None
    def time_to_collision(self, other: Particle) -> float
    def kinetic_energy(self) -> float
    def potential_energy(self) -> float
```

**Key Design Decisions**:
- **Analytic Evolution**: Use exact solutions for harmonic oscillators (no discretization error)
- **Collision Prediction**: Solve transcendental equations for harmonic particles
- **Memory Layout**: Consider struct-of-arrays for large N (better cache locality)

**Numba Optimization**:
```python
@njit
def evolve_harmonic_numba(pos, vel, eq_pos, k, m, dt):
    """JIT-compiled harmonic evolution for performance."""
    omega = np.sqrt(k / m)
    x0 = pos - eq_pos
    v0 = vel
    cos_wt = np.cos(omega * dt)
    sin_wt = np.sin(omega * dt)
    new_pos = eq_pos + x0 * cos_wt + (v0 / omega) * sin_wt
    new_vel = -x0 * omega * sin_wt + v0 * cos_wt
    return new_pos, new_vel
```

### 2. Chain Configuration

**File**: `src/core/chain.py`

```python
class Chain:
    """Container for N particles with boundary conditions."""
    
    def __init__(self, config: ChainConfig):
        self.N = config.N
        self.particles = self._initialize_particles(config)
        self.boundary = config.boundary  # OPEN or PERIODIC
        self.length = config.length
        
    def total_energy(self) -> float
    def total_momentum(self) -> float
    def local_temperature(self, region: slice) -> float
    def get_neighbors(self, index: int) -> List[int]
```

**ChainConfig**:
```python
@dataclass
class ChainConfig:
    N: int  # Number of particles
    epsilon: float  # Dimensionless energy parameter
    mass_free: float = 1.0
    mass_harmonic: float = 1.0
    spring_constant: float = 1.0
    length: float = None  # Total chain length (auto if None)
    boundary: BoundaryType = BoundaryType.OPEN
    initial_condition: InitialCondition = InitialCondition.THERMAL
    temperature: float = 1.0  # For thermal IC
```

**Initialization Strategies**:
- **Thermal**: Sample from Maxwell-Boltzmann at given T
- **Soliton**: Localized energy pulse for ballistic transport tests
- **Uniform**: Equal energy per particle
- **Custom**: User-provided positions/velocities

### 3. Collision Detection & Resolution

**File**: `src/core/collision.py`

```python
@dataclass
class CollisionEvent:
    """Represents a collision between two particles."""
    time: float
    particle_i: int
    particle_j: int
    type: CollisionType  # PARTICLE_PARTICLE, PARTICLE_WALL
    
class CollisionDetector:
    """Finds next collision in the system."""
    
    def __init__(self, chain: Chain):
        self.chain = chain
        self.event_queue = []  # Min-heap of CollisionEvents
        
    def find_next_collision(self) -> CollisionEvent:
        """Scan all particle pairs, return soonest collision."""
        
    def resolve_collision(self, event: CollisionEvent) -> None:
        """Update velocities via elastic collision."""
        
    def update_queue_after_collision(self, affected_indices: List[int]) -> None:
        """Recompute collision times for affected particles."""
```

**Collision Resolution** (Elastic, 1D):
```python
def resolve_elastic_collision_1d(p1: Particle, p2: Particle) -> None:
    """
    Elastic collision in 1D (head-on).
    
    Conservation laws:
        m1*v1 + m2*v2 = m1*v1' + m2*v2'  (momentum)
        m1*v1² + m2*v2² = m1*v1'² + m2*v2'²  (energy)
    
    Solution:
        v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)
        v2' = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)
    """
    m1, m2 = p1.mass, p2.mass
    v1, v2 = p1.velocity, p2.velocity
    
    v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    
    p1.velocity = v1_new
    p2.velocity = v2_new
```

**Performance Optimization**:
- **Spatial Hashing**: For large N, divide chain into cells (only check nearby particles)
- **Event Invalidation**: Mark events as stale after collision, lazy removal from queue
- **Vectorized Prediction**: Compute all collision times in parallel (Numba)

### 4. Thermal Reservoirs

**File**: `src/simulation/reservoir.py`

```python
class HeatBath:
    """Stochastic thermostat for boundary particles."""
    
    def __init__(self, temperature: float, coupling_rate: float):
        self.T = temperature
        self.gamma = coupling_rate  # Collision rate with bath
        
    def apply_thermostat(self, particle: Particle, dt: float) -> bool:
        """
        Stochastically resample velocity from Maxwell-Boltzmann.
        
        Returns True if resampling occurred (Poisson process).
        """
        prob = 1 - np.exp(-self.gamma * dt)
        if np.random.rand() < prob:
            particle.velocity = self.sample_velocity(particle.mass)
            return True
        return False
        
    def sample_velocity(self, mass: float) -> float:
        """Sample from Maxwell-Boltzmann: P(v) ∝ exp(-mv²/2kT)."""
        sigma = np.sqrt(self.T / mass)
        return np.random.normal(0, sigma)
```

**Energy Flux Measurement**:
```python
class FluxMeter:
    """Measures energy flow across a reference plane."""
    
    def __init__(self, position: float):
        self.x_ref = position
        self.flux_history = []
        
    def record_crossing(self, particle: Particle, direction: int):
        """Record energy carried by particle crossing plane."""
        energy = particle.kinetic_energy() + particle.potential_energy()
        flux = direction * energy  # +1 for right, -1 for left
        self.flux_history.append((time, flux))
        
    def time_averaged_flux(self, t_start: float, t_end: float) -> float:
        """Compute <J> over time window."""
        relevant = [(t, J) for t, J in self.flux_history if t_start <= t <= t_end]
        return np.mean([J for _, J in relevant])
```

## Simulation Engine

**File**: `src/simulation/engine.py`

```python
class EventDrivenSimulator:
    """Main simulation loop using event-driven dynamics."""
    
    def __init__(self, chain: Chain, config: SimulationConfig):
        self.chain = chain
        self.collision_detector = CollisionDetector(chain)
        self.time = 0.0
        self.config = config
        
        # Observables
        self.energy_history = []
        self.flux_meters = []
        self.temperature_profile = []
        
        # Reservoirs (if open system)
        if config.boundary == BoundaryType.OPEN:
            self.hot_bath = HeatBath(config.T_hot, config.gamma)
            self.cold_bath = HeatBath(config.T_cold, config.gamma)
            
    def run(self, total_time: float) -> SimulationResult:
        """
        Main event loop.
        
        1. Find next event (collision or thermostat)
        2. Evolve all particles to event time
        3. Resolve event
        4. Update collision queue
        5. Record observables
        6. Repeat until total_time reached
        """
        while self.time < total_time:
            # Find next collision
            next_collision = self.collision_detector.find_next_collision()
            
            # Find next thermostat event (if open system)
            next_thermostat = self._next_thermostat_event()
            
            # Take whichever comes first
            next_event = min(next_collision, next_thermostat, key=lambda e: e.time)
            dt = next_event.time - self.time
            
            # Evolve all particles
            self._evolve_all(dt)
            self.time = next_event.time
            
            # Resolve event
            if isinstance(next_event, CollisionEvent):
                self.collision_detector.resolve_collision(next_event)
            else:
                self._apply_thermostat(next_event)
                
            # Record observables
            if self.time % self.config.measurement_interval < dt:
                self._measure_observables()
                
        return self._compile_results()
        
    def _evolve_all(self, dt: float):
        """Evolve all particles for time dt."""
        for p in self.chain.particles:
            if p.type == ParticleType.FREE:
                p.evolve_free(dt)
            else:
                p.evolve_harmonic(dt)
                
    def _measure_observables(self):
        """Record energy, temperature, flux, etc."""
        self.energy_history.append((self.time, self.chain.total_energy()))
        # ... other measurements
```

**Steady-State Detection**:
```python
def detect_steady_state(flux_history: List[float], 
                        window_size: int = 1000,
                        threshold: float = 0.01) -> bool:
    """
    Check if flux variance is below threshold.
    
    Steady state when: std(J) / mean(J) < threshold
    """
    if len(flux_history) < window_size:
        return False
    recent = flux_history[-window_size:]
    return np.std(recent) / np.abs(np.mean(recent)) < threshold
```

## Analysis Layer

### 1. Lyapunov Exponent Calculation

**File**: `src/analysis/lyapunov.py`

```python
class LyapunovCalculator:
    """Compute Lyapunov exponents via tangent space evolution."""
    
    def __init__(self, chain: Chain):
        self.chain = chain
        self.dim = 2 * chain.N  # Phase space dimension (x, v for each particle)
        self.tangent_vectors = self._initialize_tangent_space()
        
    def _initialize_tangent_space(self) -> np.ndarray:
        """Start with orthonormal basis in phase space."""
        return np.eye(self.dim)
        
    def evolve_tangent_space(self, dt: float):
        """
        Evolve tangent vectors alongside system trajectory.
        
        For free particles: δx' = δx + δv * dt, δv' = δv
        For harmonic: linearize equations of motion
        """
        # Compute Jacobian of flow map
        J = self._compute_jacobian(dt)
        
        # Evolve tangent vectors
        self.tangent_vectors = J @ self.tangent_vectors
        
        # Gram-Schmidt orthonormalization
        self.tangent_vectors, norms = self._gram_schmidt(self.tangent_vectors)
        
        return norms  # Growth rates
        
    def compute_lyapunov_spectrum(self, total_time: float, 
                                   n_steps: int) -> np.ndarray:
        """
        Compute all Lyapunov exponents.
        
        λ_i = lim_{t→∞} (1/t) * log(||δv_i(t)||)
        """
        dt = total_time / n_steps
        log_growth = np.zeros(self.dim)
        
        for _ in range(n_steps):
            # Evolve system
            self.chain.evolve(dt)
            
            # Evolve tangent space
            norms = self.evolve_tangent_space(dt)
            log_growth += np.log(norms)
            
        return log_growth / total_time
        
    def _compute_jacobian(self, dt: float) -> np.ndarray:
        """
        Jacobian of time-dt flow map.
        
        For harmonic oscillator:
            J = [[cos(ωt), sin(ωt)/ω],
                 [-ω*sin(ωt), cos(ωt)]]
        """
        # Implementation depends on particle types
        pass
```

**Validation**: For Hamiltonian systems, Σλ_i = 0 (phase space volume conservation).

### 2. Transport Analysis

**File**: `src/analysis/transport.py`

```python
class TransportAnalyzer:
    """Extract thermal conductivity and classify transport regime."""
    
    def __init__(self, simulation_result: SimulationResult):
        self.result = simulation_result
        
    def compute_conductivity(self) -> float:
        """
        Extract κ from Fourier's law: J = -κ * ∇T
        
        Steps:
        1. Measure steady-state flux <J>
        2. Fit temperature profile T(x)
        3. Compute gradient ∇T (exclude boundary layers)
        4. κ = -<J> / ∇T
        """
        J = self.result.mean_flux()
        grad_T = self._fit_temperature_gradient()
        return -J / grad_T
        
    def _fit_temperature_gradient(self) -> float:
        """
        Fit T(x) and extract gradient.
        
        For nonlinear profiles: use local gradient at midpoint.
        """
        x = self.result.positions
        T = self.result.temperature_profile
        
        # Exclude boundary layers (10% on each side)
        mask = (x > 0.1 * x.max()) & (x < 0.9 * x.max())
        x_bulk = x[mask]
        T_bulk = T[mask]
        
        # Fit polynomial (quadratic for curved profiles)
        coeffs = np.polyfit(x_bulk, T_bulk, deg=2)
        
        # Gradient at midpoint
        x_mid = 0.5 * x.max()
        grad_T = coeffs[1] + 2 * coeffs[0] * x_mid
        
        return grad_T
        
    def classify_regime(self) -> TransportRegime:
        """
        Determine if transport is ballistic, diffusive, or crossover.
        
        Criteria:
        - Ballistic: κ ~ N (scales with system size)
        - Diffusive: κ → const (size-independent)
        - Crossover: intermediate behavior
        """
        # Compute Knudsen number
        lambda_mfp = self._mean_free_path()
        L = self.result.chain_length
        Kn = lambda_mfp / L
        
        if Kn > 1:
            return TransportRegime.BALLISTIC
        elif Kn < 0.1:
            return TransportRegime.DIFFUSIVE
        else:
            return TransportRegime.CROSSOVER
            
    def _mean_free_path(self) -> float:
        """Estimate from collision rate: λ = v_avg / collision_rate."""
        v_avg = np.mean([abs(p.velocity) for p in self.result.chain.particles])
        collision_rate = len(self.result.collision_history) / self.result.total_time
        return v_avg / collision_rate
```

### 3. Finite-Size Scaling

**File**: `src/analysis/scaling.py`

```python
class FiniteSizeScaling:
    """Analyze how conductivity converges with system size."""
    
    def run_scaling_study(self, 
                          epsilon: float,
                          N_values: List[int],
                          n_runs: int = 5) -> ScalingResult:
        """
        Measure κ(N) for various chain lengths.
        
        Returns:
            N_values, κ_mean, κ_std, regime_classification
        """
        results = []
        
        for N in N_values:
            kappa_samples = []
            
            for _ in range(n_runs):
                config = ChainConfig(N=N, epsilon=epsilon)
                sim = EventDrivenSimulator(Chain(config), SimulationConfig())
                result = sim.run(total_time=10000.0)
                
                analyzer = TransportAnalyzer(result)
                kappa = analyzer.compute_conductivity()
                kappa_samples.append(kappa)
                
            results.append({
                'N': N,
                'kappa_mean': np.mean(kappa_samples),
                'kappa_std': np.std(kappa_samples)
            })
            
        return ScalingResult(results)
        
    def fit_scaling_law(self, scaling_result: ScalingResult) -> Dict:
        """
        Fit κ(N) to power law: κ ~ N^α
        
        - α = 1: ballistic
        - α = 0: diffusive
        - 0 < α < 1: anomalous
        """
        N = np.array([r['N'] for r in scaling_result.data])
        kappa = np.array([r['kappa_mean'] for r in scaling_result.data])
        
        # Log-log fit
        log_N = np.log(N)
        log_kappa = np.log(kappa)
        alpha, intercept = np.polyfit(log_N, log_kappa, deg=1)
        
        return {'alpha': alpha, 'intercept': intercept}
```

## 2D Extension Design

### Challenges

1. **Collision Detection**: In 2D, particles can collide at arbitrary angles
   - Need to solve for collision time in 2D trajectories
   - Harmonic particles trace ellipses → complex geometry

2. **Topology**: Checkerboard pattern for alternating types
   ```
   F H F H F
   H F H F H
   F H F H F
   ```

3. **Heat Reservoirs**: Apply to edges (e.g., left/right walls)

4. **Conductivity Tensor**: κ_xx, κ_yy may differ if lattice is anisotropic

### Implementation Strategy

**Option 1: Hard-Core Collisions** (like 1D)
- Particles are hard spheres
- Collision detection: solve for intersection of trajectories
- Pro: Event-driven, exact
- Con: Complex geometry for harmonic particles

**Option 2: Soft Potentials** (continuous forces)
- Replace hard-core with soft repulsion (e.g., Lennard-Jones)
- Time-stepping required (no longer event-driven)
- Pro: Easier to implement
- Con: Discretization error, slower

**Recommendation**: Start with Option 2 for 2D, use adaptive time-stepping to maintain accuracy.

## Performance Benchmarks

### Target Performance (M2 Pro)

| System Size | Collisions | Target Time | Strategy |
|-------------|-----------|-------------|----------|
| N = 100     | 10^6      | < 10s       | Pure Python |
| N = 1000    | 10^6      | < 60s       | Numba JIT |
| N = 10000   | 10^6      | < 600s      | Numba + vectorization |

### Optimization Checklist

- [ ] Numba JIT for collision detection
- [ ] Numba JIT for particle evolution
- [ ] Struct-of-arrays memory layout
- [ ] Spatial hashing for large N
- [ ] Parallel parameter sweeps (multiprocessing)
- [ ] Lazy observable computation
- [ ] Checkpoint/resume for long runs

## Testing Strategy

### Unit Tests

```python
def test_free_particle_evolution():
    """Free particle moves ballistically."""
    p = Particle(type=ParticleType.FREE, position=0, velocity=1)
    p.evolve_free(dt=2.0)
    assert p.position == 2.0
    assert p.velocity == 1.0

def test_harmonic_particle_energy_conservation():
    """Harmonic oscillator conserves energy."""
    p = Particle(type=ParticleType.HARMONIC, k=1.0, m=1.0, ...)
    E0 = p.total_energy()
    p.evolve_harmonic(dt=10.0)
    assert abs(p.total_energy() - E0) < 1e-10

def test_elastic_collision_conservation():
    """Collision conserves momentum and energy."""
    p1 = Particle(m=1, v=1)
    p2 = Particle(m=1, v=-1)
    E0 = p1.kinetic_energy() + p2.kinetic_energy()
    p0 = p1.mass * p1.velocity + p2.mass * p2.velocity
    
    resolve_elastic_collision_1d(p1, p2)
    
    E1 = p1.kinetic_energy() + p2.kinetic_energy()
    p1_final = p1.mass * p1.velocity + p2.mass * p2.velocity
    
    assert abs(E1 - E0) < 1e-10
    assert abs(p1_final - p0) < 1e-10
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(
    m1=st.floats(min_value=0.1, max_value=10),
    m2=st.floats(min_value=0.1, max_value=10),
    v1=st.floats(min_value=-10, max_value=10),
    v2=st.floats(min_value=-10, max_value=10)
)
def test_collision_always_conserves_energy(m1, m2, v1, v2):
    """Energy conservation holds for any masses and velocities."""
    p1 = Particle(mass=m1, velocity=v1)
    p2 = Particle(mass=m2, velocity=v2)
    
    E_before = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    resolve_elastic_collision_1d(p1, p2)
    E_after = 0.5 * m1 * p1.velocity**2 + 0.5 * m2 * p2.velocity**2
    
    assert abs(E_after - E_before) < 1e-10

@given(
    N=st.integers(min_value=10, max_value=100),
    epsilon=st.floats(min_value=0.1, max_value=10)
)
def test_closed_system_conserves_energy(N, epsilon):
    """Closed system (periodic BC) conserves total energy."""
    config = ChainConfig(N=N, epsilon=epsilon, boundary=BoundaryType.PERIODIC)
    chain = Chain(config)
    sim = EventDrivenSimulator(chain, SimulationConfig())
    
    E0 = chain.total_energy()
    sim.run(total_time=100.0)
    E1 = chain.total_energy()
    
    assert abs(E1 - E0) / E0 < 1e-8
```

### Integration Tests

```python
def test_equilibrium_thermalization():
    """System coupled to single reservoir reaches equilibrium."""
    config = ChainConfig(N=100, epsilon=1.0)
    chain = Chain(config)
    
    # Couple all particles to same reservoir
    bath = HeatBath(temperature=1.0, coupling_rate=0.1)
    
    # Run until equilibrium
    sim = EventDrivenSimulator(chain, SimulationConfig())
    result = sim.run(total_time=10000.0)
    
    # Check: temperature = bath temperature
    T_final = chain.local_temperature(slice(None))
    assert abs(T_final - 1.0) < 0.05

def test_reproduce_paper_figure_2():
    """Reproduce conductivity vs N curve from paper."""
    epsilon = 1.0
    N_values = [50, 100, 200, 500, 1000]
    
    scaling = FiniteSizeScaling()
    result = scaling.run_scaling_study(epsilon, N_values)
    
    # Load reference data from paper
    reference = load_paper_data("figure_2_epsilon_1.csv")
    
    # Compare (allow 10% error due to finite sampling)
    for N, kappa_measured in zip(result.N, result.kappa_mean):
        kappa_expected = reference.interpolate(N)
        assert abs(kappa_measured - kappa_expected) / kappa_expected < 0.1
```

## Visualization Design

### Real-Time Animation

```python
class ParticleAnimator:
    """Matplotlib animation of particle motion."""
    
    def __init__(self, chain: Chain):
        self.chain = chain
        self.fig, self.ax = plt.subplots()
        
    def animate(self, total_time: float, fps: int = 30):
        """
        Animate particle positions over time.
        
        - Free particles: blue circles
        - Harmonic particles: red circles with spring
        - Size proportional to kinetic energy
        """
        def update(frame):
            # Evolve system
            self.chain.evolve(dt=1/fps)
            
            # Update plot
            self.ax.clear()
            for p in self.chain.particles:
                color = 'blue' if p.type == ParticleType.FREE else 'red'
                size = 100 * p.kinetic_energy()
                self.ax.scatter(p.position, 0, c=color, s=size)
                
        anim = FuncAnimation(self.fig, update, frames=int(total_time * fps))
        return anim
```

### Diagnostic Plots

1. **Temperature Profile**: T(x) with fitted curve
2. **Energy History**: E(t) to verify conservation
3. **Flux History**: J(t) with steady-state window
4. **Scaling Curve**: κ vs N with power-law fit
5. **Lyapunov Spectrum**: λ_i vs index
6. **Regime Diagram**: ε vs N heatmap (color = regime)

## Open Questions & Future Work

1. **Optimal Thermostat Coupling**: How does γ affect convergence time?
2. **Boundary Layer Width**: How far do reservoir effects penetrate?
3. **2D Lattice Geometry**: Square vs hexagonal vs triangular?
4. **Anharmonic Potentials**: Quartic, Morse, etc.
5. **Quantum Extension**: Can we add quantum effects (phonons)?

## References

- Allen & Tildesley, "Computer Simulation of Liquids" (event-driven MD)
- Rapaport, "The Art of Molecular Dynamics Simulation"
- Benettin et al., "Lyapunov Characteristic Exponents for smooth dynamical systems"
- Lepri, Livi, Politi, "Thermal conduction in classical low-dimensional lattices"
