# Requirements: Ding-a-Ling Thermal Transport Simulator

## Project Overview

Build a high-performance physics simulation package to reproduce and extend results from the paper on thermal conductivity in 1D chains of alternating free and harmonically-bound particles. The system will serve as a platform for studying the emergence of Fourier's law, ballistic-to-diffusive transport crossover, and chaos-driven finite-size effects.

## Target Hardware

- **Platform**: Mac mini M2 Pro (ARM64)
- **Memory**: 32GB unified memory
- **Performance Goal**: Simulate chains up to N=10,000 particles for extended time periods (hours to days)
- **Optimization**: Leverage ARM NEON instructions via Numba, efficient memory layout

## User Stories

### 1. Paper Reproduction
**As a researcher**, I want to reproduce the key figures and results from the original paper so that I can validate the implementation and understand the physics.

**Acceptance Criteria**:
- Reproduce thermal conductivity vs chain length curves for various ε values
- Match temperature profile shapes (including nonlinear curvature)
- Reproduce Lyapunov exponent scaling behavior
- Verify the stiff-spring vs weak-spring regime transition
- Generate equivalent plots to paper's Figures 1-8

### 2. Event-Driven Simulation Engine
**As a physicist**, I want an accurate event-driven simulator that handles collisions exactly so that energy conservation and long-time dynamics are reliable.

**Acceptance Criteria**:
- Collision detection accurate to machine precision
- Analytic evolution between collisions (no time-stepping errors)
- Energy conservation: |ΔE/E| < 10^-10 over 10^6 collisions
- Efficient priority queue for next-event scheduling
- Support for both hard-sphere and point-particle collision models

### 3. Open System with Thermal Reservoirs
**As a user**, I want to simulate chains coupled to heat baths at different temperatures so that I can measure steady-state thermal conductivity.

**Acceptance Criteria**:
- Stochastic velocity resampling at boundaries (Maxwell-Boltzmann distribution)
- Configurable hot/cold reservoir temperatures
- Steady-state detection (flux variance < 1% over measurement window)
- Local temperature measurement via kinetic energy averaging
- Heat flux calculation: J = Σ(energy flow across reference plane)
- Conductivity extraction: κ = -J / ∇T with proper gradient estimation

### 4. Closed System for Chaos Diagnostics
**As a researcher**, I want to compute Lyapunov exponents in periodic systems so that I can correlate chaos strength with transport properties.

**Acceptance Criteria**:
- Periodic boundary conditions (ring topology)
- Tangent space evolution for Lyapunov calculation
- Maximum Lyapunov exponent λ_max with error estimates
- Lyapunov spectrum (all exponents) for phase space analysis
- Verification: Σλ_i = 0 for Hamiltonian systems (within numerical precision)

### 5. Finite-Size Scaling Analysis
**As a researcher**, I want to systematically vary chain length N and measure how conductivity converges so that I can identify the thermodynamic limit.

**Acceptance Criteria**:
- Automated parameter sweeps over N = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
- Parallel execution of independent simulations
- Convergence detection: identify when κ(N) plateaus
- Scaling law fitting: κ(N) ~ N^α for ballistic regime
- Export data in standard format (CSV, HDF5) for external analysis

### 6. Transport Regime Classification
**As a user**, I want the system to automatically identify whether transport is ballistic, diffusive, or in a crossover regime so that I can understand the physics.

**Acceptance Criteria**:
- Compute mean free path λ_mfp from collision rate
- Calculate Knudsen number: Kn = λ_mfp / L
- Classify regime:
  - Ballistic: Kn > 1, κ ~ N
  - Diffusive: Kn << 1, κ → const
  - Crossover: Kn ~ 1
- Correlation with Lyapunov exponent magnitude
- Visual indicators in output (regime labels on plots)

### 7. Performance Optimization
**As a user**, I want simulations to run efficiently on my M2 Pro hardware so that I can explore large parameter spaces in reasonable time.

**Acceptance Criteria**:
- Numba JIT compilation for collision detection and evolution
- Memory-efficient particle storage (< 1KB per particle)
- Vectorized operations where possible
- Benchmark: N=1000 chain, 10^6 collisions in < 60 seconds
- Parallel parameter sweeps using multiprocessing
- Progress bars and time-remaining estimates

### 8. Jupyter Notebook Interface
**As a researcher**, I want interactive notebooks that guide me through standard analyses so that I can quickly explore the system.

**Acceptance Criteria**:
- Example notebooks for:
  - Basic simulation setup and visualization
  - Conductivity measurement workflow
  - Lyapunov exponent calculation
  - Finite-size scaling study
  - Regime diagram generation (ε vs N phase plot)
- Live animation of particle motion (optional, for small N)
- Interactive parameter widgets (ipywidgets)
- Clear documentation and physics explanations in markdown cells

### 9. Extension to 2D Lattices
**As a researcher**, I want to extend the model to 2D square lattices so that I can study dimensionality effects on thermal transport.

**Acceptance Criteria**:
- 2D lattice topology with alternating particle types (checkerboard pattern)
- Collision detection in 2D (more complex geometry)
- Heat reservoirs on opposite edges
- Temperature field visualization (heatmaps)
- Conductivity tensor measurement (κ_xx, κ_yy)
- Comparison with 1D results: how does κ scale with dimensionality?

### 10. Physical Invariant Testing
**As a developer**, I want property-based tests that verify fundamental physics laws so that I can catch bugs that violate conservation laws or thermodynamics.

**Acceptance Criteria**:
- Energy conservation in closed systems (all collision types)
- Momentum conservation in collisions
- Detailed balance: equilibrium distributions match canonical ensemble
- Second law: entropy production ≥ 0 in open systems
- Time-reversal symmetry in Hamiltonian evolution
- Ergodicity tests: long-time averages = ensemble averages (for chaotic regime)

## Technical Architecture

### Core Components

1. **Particle System** (`src/core/particle.py`)
   - `Particle` class: position, velocity, type, mass, spring constant
   - `ParticleType` enum: FREE, HARMONIC
   - Analytic evolution methods: `evolve_free()`, `evolve_harmonic()`
   - Collision time prediction: `time_to_collision(other)`

2. **Chain Configuration** (`src/core/chain.py`)
   - `Chain` class: container for N particles
   - Alternating particle type initialization
   - Boundary condition modes: OPEN, PERIODIC
   - Energy/momentum calculation methods
   - Spatial indexing for efficient neighbor lookup

3. **Collision Engine** (`src/core/collision.py`)
   - `CollisionEvent` dataclass: time, particle indices, type
   - `CollisionDetector`: find next collision in system
   - Priority queue (heapq) for event scheduling
   - Elastic collision resolution (velocity updates)
   - Collision rate tracking for diagnostics

4. **Thermal Reservoirs** (`src/simulation/reservoir.py`)
   - `HeatBath` class: temperature, coupling strength
   - Maxwell-Boltzmann velocity sampling
   - Stochastic thermostat (Andersen-like)
   - Energy flux measurement at boundaries

5. **Simulation Engine** (`src/simulation/engine.py`)
   - `EventDrivenSimulator`: main simulation loop
   - Time evolution: advance to next collision, resolve, repeat
   - Observable measurement and logging
   - Checkpoint/resume functionality for long runs

6. **Lyapunov Calculator** (`src/analysis/lyapunov.py`)
   - Tangent space vector evolution
   - Gram-Schmidt orthonormalization
   - Lyapunov exponent extraction from growth rates
   - Full spectrum calculation

7. **Transport Analysis** (`src/analysis/transport.py`)
   - Temperature profile fitting
   - Heat flux calculation (time-averaged)
   - Thermal conductivity extraction
   - Finite-size scaling analysis
   - Regime classification

8. **Visualization** (`src/visualization/`)
   - Real-time particle animation (Matplotlib)
   - Phase space plots (x-v diagrams)
   - Temperature profile plots
   - Scaling curves (κ vs N, λ vs ε)
   - Regime diagrams (heatmaps)

### Data Flow

```
Configuration → Chain Initialization → Simulation Loop → Analysis → Visualization
                                            ↓
                                    [Collision Detection]
                                            ↓
                                    [Event Resolution]
                                            ↓
                                    [Observable Logging]
                                            ↓
                                    [Steady-State Check]
```

### Performance Strategy

1. **Numba JIT**: Hot loops (collision detection, evolution, Lyapunov)
2. **Vectorization**: Batch operations on particle arrays
3. **Memory Layout**: Struct-of-arrays for cache efficiency
4. **Lazy Evaluation**: Compute observables only when needed
5. **Parallel Sweeps**: Multiprocessing for independent parameter points

## Testing Strategy

### Unit Tests
- Particle evolution (free and harmonic) against analytic solutions
- Collision time prediction accuracy
- Collision resolution (momentum/energy conservation)
- Reservoir sampling (distribution verification)
- Temperature measurement (known equilibrium states)

### Integration Tests
- Full simulation: equilibrium thermalization
- Open system: steady-state convergence
- Closed system: energy conservation over long times
- Lyapunov calculation: match known chaotic systems (e.g., hard-sphere gas)

### Property-Based Tests (Hypothesis)
- **Energy conservation**: ∀ collision, E_before = E_after
- **Momentum conservation**: ∀ collision, p_before = p_after
- **Detailed balance**: equilibrium → no net flux
- **Entropy production**: open system → dS/dt ≥ 0
- **Time reversibility**: reverse velocities → retrace trajectory
- **Ergodicity**: chaotic regime → time avg = ensemble avg

### Validation Tests
- Reproduce paper's Figure 2: κ vs N for ε = 0.5, 1.0, 2.0
- Reproduce paper's Figure 5: λ_max vs ε
- Reproduce paper's Figure 7: temperature profiles
- Free particle limit: κ → ∞ (ballistic)
- Harmonic crystal limit: κ → finite (diffusive)

## Deliverables

### Phase 1: Core Engine (Weeks 1-2)
- [ ] Particle and chain classes
- [ ] Event-driven collision engine
- [ ] Basic unit tests
- [ ] Energy conservation verification

### Phase 2: Open System (Weeks 3-4)
- [ ] Thermal reservoir implementation
- [ ] Steady-state detection
- [ ] Conductivity measurement
- [ ] Reproduce paper's conductivity curves

### Phase 3: Chaos Diagnostics (Week 5)
- [ ] Periodic boundary conditions
- [ ] Lyapunov exponent calculation
- [ ] Reproduce paper's chaos-transport correlation

### Phase 4: Analysis Tools (Week 6)
- [ ] Finite-size scaling automation
- [ ] Regime classification
- [ ] Transport analysis suite
- [ ] Property-based tests

### Phase 5: 2D Extension (Weeks 7-8)
- [ ] 2D lattice topology
- [ ] 2D collision detection
- [ ] Conductivity tensor measurement
- [ ] Dimensionality comparison study

### Phase 6: Documentation & Polish (Week 9)
- [ ] Jupyter notebook examples
- [ ] API documentation
- [ ] Performance benchmarks
- [ ] Publication-quality plots

## Success Metrics

1. **Accuracy**: Reproduce paper's key results within 5% error
2. **Performance**: N=1000 chain, 10^6 collisions in < 60s
3. **Scalability**: Successfully simulate N=10,000 chains
4. **Extensibility**: 2D lattice implementation working
5. **Reliability**: All tests passing, energy conservation < 10^-10
6. **Usability**: Clear notebooks, < 10 lines to run basic simulation

## Open Questions

1. **2D Collision Geometry**: How to handle particle-particle collisions in 2D with mixed free/harmonic types? May need soft potentials instead of hard-core.

2. **Lyapunov in 2D**: Tangent space dimension scales as 4N - computational cost?

3. **Finite-Size Effects in 2D**: Does the crossover length scale differently? Need theoretical guidance.

4. **Visualization**: Real-time 2D animation feasible for N ~ 1000?

5. **Alternative Models**: Should we support other on-site potentials (quartic, Morse) for generalization?

## References

- Original paper: "Thermal conductivity in a chain of alternating masses with on-site potentials"
- Casati et al. (1984): Earlier ding-a-ling results
- Lepri, Livi, Politi (2003): Review of heat transport in low-dimensional systems
- Dhar (2008): Heat transport in harmonic lattices
