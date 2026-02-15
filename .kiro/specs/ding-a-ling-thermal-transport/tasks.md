# Tasks: Ding-a-Ling Thermal Transport Simulator

## Phase 1: Core Physics Engine (Weeks 1-2)

### 1. Project Setup
- [ ] 1.1 Initialize Python package structure
- [ ] 1.2 Configure pyproject.toml with dependencies
- [ ] 1.3 Set up pytest configuration
- [ ] 1.4 Create .gitignore for Python projects
- [ ] 1.5 Set up pre-commit hooks (black, ruff)

### 2. Particle System Implementation
- [ ] 2.1 Implement `ParticleType` enum (FREE, HARMONIC)
- [ ] 2.2 Implement `Particle` dataclass with core attributes
- [ ] 2.3 Implement `evolve_free()` method
- [ ] 2.4 Implement `evolve_harmonic()` method with analytic solution
- [ ] 2.5 Implement `time_to_collision()` for free-free pairs
- [ ] 2.6 Implement `time_to_collision()` for free-harmonic pairs
- [ ] 2.7 Implement `time_to_collision()` for harmonic-harmonic pairs
- [ ] 2.8 Implement energy calculation methods
- [ ] 2.9 Write unit tests for particle evolution
- [ ] 2.10 Write unit tests for collision time prediction

### 3. Chain Configuration
- [ ] 3.1 Implement `ChainConfig` dataclass
- [ ] 3.2 Implement `Chain` class with particle initialization
- [ ] 3.3 Implement alternating particle type assignment
- [ ] 3.4 Implement `total_energy()` method
- [ ] 3.5 Implement `total_momentum()` method
- [ ] 3.6 Implement `local_temperature()` method
- [ ] 3.7 Implement neighbor lookup for boundary conditions
- [ ] 3.8 Write unit tests for chain initialization
- [ ] 3.9 Write unit tests for energy/momentum calculations

### 4. Collision Detection & Resolution
- [ ] 4.1 Implement `CollisionEvent` dataclass
- [ ] 4.2 Implement `CollisionDetector` class
- [ ] 4.3 Implement priority queue for event scheduling
- [ ] 4.4 Implement `find_next_collision()` method
- [ ] 4.5 Implement elastic collision resolution (1D)
- [ ] 4.6 Implement collision queue update logic
- [ ] 4.7 Add collision rate tracking
- [ ] 4.8 Write unit tests for collision detection
- [ ] 4.9 Write unit tests for collision resolution
- [ ] 4.10 Write property-based tests for conservation laws

### 5. Event-Driven Simulator (Basic)
- [ ] 5.1 Implement `SimulationConfig` dataclass
- [ ] 5.2 Implement `EventDrivenSimulator` class skeleton
- [ ] 5.3 Implement main event loop
- [ ] 5.4 Implement `_evolve_all()` method
- [ ] 5.5 Implement basic observable recording
- [ ] 5.6 Implement `SimulationResult` dataclass
- [ ] 5.7 Write integration test: closed system energy conservation
- [ ] 5.8 Write integration test: equilibrium thermalization

## Phase 2: Open System with Thermal Reservoirs (Weeks 3-4)

### 6. Thermal Reservoir Implementation
- [ ] 6.1 Implement `HeatBath` class
- [ ] 6.2 Implement Maxwell-Boltzmann velocity sampling
- [ ] 6.3 Implement stochastic thermostat application
- [ ] 6.4 Implement thermostat event scheduling
- [ ] 6.5 Write unit tests for velocity distribution
- [ ] 6.6 Write unit tests for detailed balance

### 7. Heat Flux Measurement
- [ ] 7.1 Implement `FluxMeter` class
- [ ] 7.2 Implement crossing detection logic
- [ ] 7.3 Implement time-averaged flux calculation
- [ ] 7.4 Implement flux history storage
- [ ] 7.5 Write unit tests for flux measurement

### 8. Open System Simulation
- [ ] 8.1 Integrate heat baths into simulator
- [ ] 8.2 Implement boundary particle identification
- [ ] 8.3 Implement steady-state detection
- [ ] 8.4 Implement temperature profile measurement
- [ ] 8.5 Write integration test: steady-state convergence
- [ ] 8.6 Write integration test: flux measurement accuracy

### 9. Thermal Conductivity Extraction
- [ ] 9.1 Implement `TransportAnalyzer` class
- [ ] 9.2 Implement temperature gradient fitting
- [ ] 9.3 Implement boundary layer exclusion logic
- [ ] 9.4 Implement conductivity calculation (κ = -J/∇T)
- [ ] 9.5 Implement error estimation for κ
- [ ] 9.6 Write unit tests for gradient fitting
- [ ] 9.7 Write integration test: known conductivity cases

### 10. Paper Reproduction: Conductivity Curves
- [ ] 10.1 Implement parameter sweep for N values
- [ ] 10.2 Run simulations for ε = 0.5, 1.0, 2.0
- [ ] 10.3 Generate κ vs N plots
- [ ] 10.4 Compare with paper's Figure 2
- [ ] 10.5 Document any discrepancies

## Phase 3: Chaos Diagnostics (Week 5)

### 11. Periodic Boundary Conditions
- [ ] 11.1 Implement periodic neighbor lookup
- [ ] 11.2 Implement periodic collision detection
- [ ] 11.3 Modify simulator for periodic mode
- [ ] 11.4 Write unit tests for periodic boundaries
- [ ] 11.5 Verify energy conservation in periodic system

### 12. Lyapunov Exponent Calculation
- [ ] 12.1 Implement `LyapunovCalculator` class
- [ ] 12.2 Implement tangent space initialization
- [ ] 12.3 Implement Jacobian computation for free particles
- [ ] 12.4 Implement Jacobian computation for harmonic particles
- [ ] 12.5 Implement Gram-Schmidt orthonormalization
- [ ] 12.6 Implement tangent space evolution
- [ ] 12.7 Implement Lyapunov spectrum calculation
- [ ] 12.8 Write unit tests for Jacobian accuracy
- [ ] 12.9 Write integration test: Σλ_i = 0 for Hamiltonian systems
- [ ] 12.10 Write validation test: match known chaotic systems

### 13. Paper Reproduction: Lyapunov Scaling
- [ ] 13.1 Run Lyapunov calculations for various ε
- [ ] 13.2 Generate λ_max vs ε plot
- [ ] 13.3 Compare with paper's Figure 5
- [ ] 13.4 Identify crossover ε_c
- [ ] 13.5 Document chaos-transport correlation

## Phase 4: Analysis Tools & Validation (Week 6)

### 14. Transport Regime Classification
- [ ] 14.1 Implement mean free path calculation
- [ ] 14.2 Implement Knudsen number calculation
- [ ] 14.3 Implement regime classification logic
- [ ] 14.4 Implement regime visualization
- [ ] 14.5 Write unit tests for regime identification

### 15. Finite-Size Scaling Analysis
- [ ] 15.1 Implement `FiniteSizeScaling` class
- [ ] 15.2 Implement automated parameter sweeps
- [ ] 15.3 Implement parallel execution (multiprocessing)
- [ ] 15.4 Implement power-law fitting (κ ~ N^α)
- [ ] 15.5 Implement convergence detection
- [ ] 15.6 Write integration test: scaling law extraction

### 16. Property-Based Testing
- [ ] 16.1 Set up Hypothesis framework
- [ ] 16.2 Write PBT: energy conservation in collisions
- [ ] 16.3 Write PBT: momentum conservation in collisions
- [ ] 16.4 Write PBT: detailed balance in equilibrium
- [ ] 16.5 Write PBT: entropy production ≥ 0 in open systems
- [ ] 16.6 Write PBT: time-reversal symmetry
- [ ] 16.7 Write PBT: ergodicity in chaotic regime

### 17. Performance Optimization
- [ ] 17.1 Profile code to identify bottlenecks
- [ ] 17.2 Apply Numba JIT to collision detection
- [ ] 17.3 Apply Numba JIT to particle evolution
- [ ] 17.4 Implement struct-of-arrays memory layout
- [ ] 17.5 Implement spatial hashing for large N
- [ ] 17.6 Benchmark: N=1000, 10^6 collisions < 60s
- [ ] 17.7 Optimize memory usage

### 18. Visualization Tools
- [ ] 18.1 Implement temperature profile plotting
- [ ] 18.2 Implement energy history plotting
- [ ] 18.3 Implement flux history plotting
- [ ] 18.4 Implement scaling curve plotting
- [ ] 18.5 Implement Lyapunov spectrum plotting
- [ ] 18.6 Implement regime diagram (ε vs N heatmap)
- [ ] 18.7 Implement particle animation (optional)

## Phase 5: 2D Extension (Weeks 7-8)

### 19. 2D Lattice Topology
- [ ] 19.1 Extend `Particle` to support 2D positions/velocities
- [ ] 19.2 Implement 2D checkerboard initialization
- [ ] 19.3 Implement 2D neighbor lookup
- [ ] 19.4 Implement 2D periodic boundaries
- [ ] 19.5 Write unit tests for 2D topology

### 20. 2D Collision Detection
- [ ] 20.1 Research soft potential approach (Lennard-Jones)
- [ ] 20.2 Implement force calculation for soft potentials
- [ ] 20.3 Implement adaptive time-stepping integrator
- [ ] 20.4 Implement 2D collision detection (if hard-core)
- [ ] 20.5 Write unit tests for 2D collisions
- [ ] 20.6 Verify energy conservation in 2D

### 21. 2D Heat Transport
- [ ] 21.1 Implement 2D heat reservoirs (edge coupling)
- [ ] 21.2 Implement 2D flux measurement
- [ ] 21.3 Implement 2D temperature field calculation
- [ ] 21.4 Implement conductivity tensor extraction (κ_xx, κ_yy)
- [ ] 21.5 Write integration test: 2D steady-state

### 22. 2D Visualization
- [ ] 22.1 Implement 2D particle position plotting
- [ ] 22.2 Implement 2D temperature field heatmap
- [ ] 22.3 Implement 2D flux vector field
- [ ] 22.4 Implement 2D animation (optional)

### 23. Dimensionality Comparison Study
- [ ] 23.1 Run 1D simulations for reference
- [ ] 23.2 Run 2D simulations with equivalent parameters
- [ ] 23.3 Compare conductivity scaling (1D vs 2D)
- [ ] 23.4 Analyze dimensionality effects on transport
- [ ] 23.5 Document findings

## Phase 6: Documentation & Polish (Week 9)

### 24. Jupyter Notebook Examples
- [ ] 24.1 Create notebook: Basic simulation setup
- [ ] 24.2 Create notebook: Conductivity measurement workflow
- [ ] 24.3 Create notebook: Lyapunov exponent calculation
- [ ] 24.4 Create notebook: Finite-size scaling study
- [ ] 24.5 Create notebook: Regime diagram generation
- [ ] 24.6 Create notebook: 2D lattice exploration
- [ ] 24.7 Add interactive widgets (ipywidgets)

### 25. API Documentation
- [ ] 25.1 Write docstrings for all public classes
- [ ] 25.2 Write docstrings for all public methods
- [ ] 25.3 Generate Sphinx documentation
- [ ] 25.4 Create API reference guide
- [ ] 25.5 Create user guide with examples

### 26. Performance Benchmarks
- [ ] 26.1 Benchmark N=100, 1000, 10000 chains
- [ ] 26.2 Benchmark collision detection scaling
- [ ] 26.3 Benchmark Lyapunov calculation overhead
- [ ] 26.4 Document performance characteristics
- [ ] 26.5 Create performance tuning guide

### 27. Publication-Quality Plots
- [ ] 27.1 Reproduce all paper figures with high DPI
- [ ] 27.2 Create comparison plots (simulation vs paper)
- [ ] 27.3 Create 2D extension figures
- [ ] 27.4 Export plots in vector format (PDF, SVG)
- [ ] 27.5 Create figure generation scripts

### 28. Final Validation & Testing
- [ ] 28.1 Run full test suite
- [ ] 28.2 Verify all paper results reproduced
- [ ] 28.3 Check code coverage (target: >80%)
- [ ] 28.4 Perform code review
- [ ] 28.5 Fix any remaining bugs

### 29. Project Cleanup
- [ ] 29.1 Remove debug code and print statements
- [ ] 29.2 Organize file structure
- [ ] 29.3 Update README with final instructions
- [ ] 29.4 Create CHANGELOG
- [ ] 29.5 Tag v1.0.0 release

## Optional Extensions (Future Work)

### 30. Advanced Features
- [ ]* 30.1 Implement checkpoint/resume for long runs
- [ ]* 30.2 Implement GPU acceleration (CuPy/JAX)
- [ ]* 30.3 Implement alternative on-site potentials (quartic, Morse)
- [ ]* 30.4 Implement 3D lattice extension
- [ ]* 30.5 Implement quantum corrections (phonon model)

### 31. User Interface Enhancements
- [ ]* 31.1 Create command-line interface (Click)
- [ ]* 31.2 Create configuration file support (YAML)
- [ ]* 31.3 Create web-based dashboard (Streamlit)
- [ ]* 31.4 Create real-time monitoring tools

### 32. Scientific Extensions
- [ ]* 32.1 Study temperature-dependent conductivity κ(T)
- [ ]* 32.2 Study mass ratio effects
- [ ]* 32.3 Study disorder effects (random masses/springs)
- [ ]* 32.4 Study anharmonic effects
- [ ]* 32.5 Prepare manuscript for publication

---

## Task Dependencies

```
Phase 1 (Core Engine)
  ├─> Phase 2 (Open System)
  │     └─> Phase 4 (Analysis & Validation)
  │           └─> Phase 6 (Documentation)
  └─> Phase 3 (Chaos Diagnostics)
        └─> Phase 4 (Analysis & Validation)
              └─> Phase 6 (Documentation)

Phase 5 (2D Extension) can start after Phase 4
```

## Estimated Timeline

- **Phase 1**: 2 weeks (40 hours)
- **Phase 2**: 2 weeks (40 hours)
- **Phase 3**: 1 week (20 hours)
- **Phase 4**: 1 week (20 hours)
- **Phase 5**: 2 weeks (40 hours)
- **Phase 6**: 1 week (20 hours)

**Total**: 9 weeks (180 hours)

## Success Criteria

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] All property-based tests passing
- [ ] Paper's Figure 2 reproduced (κ vs N)
- [ ] Paper's Figure 5 reproduced (λ vs ε)
- [ ] Energy conservation < 10^-10 in closed systems
- [ ] N=1000 chain, 10^6 collisions in < 60s
- [ ] 2D lattice simulations working
- [ ] Complete Jupyter notebook examples
- [ ] API documentation complete
