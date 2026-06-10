"""
Parameter sweep for thermal conductivity κ vs chain length N.

Implements issues oz0 (N-sweep) and oqe (ε = 0.5, 1.0, 2.0).

Usage:
    python scripts/sweep.py                         # quick smoke test
    python scripts/sweep.py --mode full             # full Fig-3 sweep
    python scripts/sweep.py --epsilon 1.5 --fig2    # reproduce Fig 2 (slow!)

Physics
-------
ε = 3 T_m / (4 ω²)  where ω² = spring_constant, T_m = (T_hot + T_cold) / 2

So for fixed ω = 1:
    T_m = 4 ε / 3,  ΔT = 0.4 T_m  (gentle gradient, ≈20% asymmetry)
    T_hot = T_m + ΔT/2,  T_cold = T_m - ΔT/2

Dimensionless conductivity reported by the paper:  κ̃ = κ / ω³  (ω=1 → κ̃=κ)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Chain, ChainConfig, BoundaryCondition
from src.simulation import EventDrivenSimulator, SimulationConfig
from src.analysis import TransportAnalyzer


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def get_free_boundary_indices(n_particles: int) -> tuple[list[int], list[int]]:
    """
    Return (left_free, right_free) indices for the Mimnagh & Ballentine boundary.

    In the ding-a-ling chain:
      - Even indices → HARMONIC (spring-bound, amplitude limited to |v|/ω)
      - Odd  indices → FREE     (can travel across the full inter-particle gap)

    The reservoir should couple to FREE boundary particles so that energy
    injection always produces a free particle that can collide with the rest
    of the chain.  Coupling to a harmonic particle leads to stalling when
    the thermostat gives it a low velocity and its amplitude A = |v|/ω is
    smaller than the lattice spacing.

    Returns
    -------
    left_free  : [1]               — first FREE particle (always index 1)
    right_free : [n-1] or [n-2]   — last FREE particle (odd index nearest n-1)
    """
    left_free = [1]
    # N-1 is odd (FREE) for even N; N-1 is even (HARMONIC) for odd N
    right_idx = n_particles - 1 if (n_particles - 1) % 2 == 1 else n_particles - 2
    right_free = [right_idx]
    return left_free, right_free


# ---------------------------------------------------------------------------
# ε ↔ temperature helpers
# ---------------------------------------------------------------------------

def epsilon_to_temperatures(
    epsilon: float,
    spring_constant: float = 1.0,
    delta_t_fraction: float = 0.5,
) -> tuple[float, float]:
    """
    Convert dimensionless ε to (T_hot, T_cold) for the simulation.

    ε = 3 T_m / (4 ω²),  ω² = spring_constant
    T_m = 4 ε ω² / 3
    ΔT  = delta_t_fraction * T_m  (default 50 % spread → T_hot/T_cold = 3)

    Using 50% relative ΔT matches the paper's Fig-1 conditions
    (T_L=2.5, T_R=1.5 at ε=1.5 → T_m=2, ΔT=1 → fraction=0.5)
    """
    omega_sq = spring_constant
    T_m = 4.0 * epsilon * omega_sq / 3.0
    dT = delta_t_fraction * T_m
    T_hot = T_m + dT / 2.0
    T_cold = max(T_m - dT / 2.0, 0.05)  # guard against T ≤ 0
    return T_hot, T_cold


def temperatures_to_epsilon(
    T_hot: float, T_cold: float, spring_constant: float = 1.0
) -> float:
    """Compute ε from simulation temperatures."""
    T_m = (T_hot + T_cold) / 2.0
    return 3.0 * T_m / (4.0 * spring_constant)


# ---------------------------------------------------------------------------
# Single-point run
# ---------------------------------------------------------------------------

def run_kappa(
    n_particles: int,
    epsilon: float,
    total_time: float = 2000.0,
    spring_constant: float = 1.0,
    gamma: float = 1.0,
    n_boundary: int = 1,
    n_boundary_exclude: int = 2,
    seed: int | None = None,
) -> dict:
    """
    Run one open-system simulation and return κ with metadata.

    Returns
    -------
    dict with keys: n, epsilon, kappa, kappa_error, flux, gradient,
                    t_hot, t_cold, elapsed_s, n_collisions
    """
    T_hot, T_cold = epsilon_to_temperatures(epsilon, spring_constant)
    T_init = (T_hot + T_cold) / 2.0

    chain_cfg = ChainConfig(
        n_particles=n_particles,
        spring_constant=spring_constant,
        boundary=BoundaryCondition.OPEN,
        temperature=T_init,
    )
    chain = Chain(chain_cfg)

    # Couple thermostat to FREE boundary particles (paper's convention).
    # See get_free_boundary_indices() for the physical motivation.
    left_free, right_free = get_free_boundary_indices(n_particles)

    sim_cfg = SimulationConfig(
        total_time=total_time,
        T_hot=T_hot,
        T_cold=T_cold,
        gamma=gamma,
        n_boundary=n_boundary,
        measurement_interval=total_time / 200,
        seed=seed,
        left_boundary_indices=left_free,
        right_boundary_indices=right_free,
    )

    t0 = time.perf_counter()
    sim = EventDrivenSimulator(chain, sim_cfg)
    result = sim.run()
    elapsed = time.perf_counter() - t0

    # Discard first half as transient
    t_start = total_time * 0.5
    analyzer = TransportAnalyzer(
        n_boundary_exclude=n_boundary_exclude,
        t_start=t_start,
        n_bootstrap=200,
        rng=np.random.default_rng(seed),
    )
    transport = analyzer.analyze(result, chain)

    # Reservoir-based κ is more reliable for short / small-N simulations.
    # J_reservoir = energy injected by hot bath / total_time (see 5w6 notes).
    J_reservoir = float(result.hot_bath.energy_exchanged / total_time)
    kappa_reservoir = float(
        analyzer.calculate_conductivity(J_reservoir, transport.gradient)
        if abs(transport.gradient) > 1e-10 else float("nan")
    )

    return {
        "n": int(n_particles),
        "epsilon": float(epsilon),
        # Spatial-flux κ (primary, noisy for short/small-N runs)
        "kappa": float(transport.kappa),
        "kappa_error": float(transport.kappa_error),
        "flux_spatial": float(transport.flux),
        "flux_error": float(transport.flux_error if not np.isnan(transport.flux_error) else 0.0),
        # Reservoir-based κ (more reliable sign, lower noise for large t)
        "kappa_reservoir": kappa_reservoir,
        "flux_reservoir": J_reservoir,
        "gradient": float(transport.gradient),
        "t_hot": float(T_hot),
        "t_cold": float(T_cold),
        "elapsed_s": float(elapsed),
        "n_collisions": int(result.n_collisions),
        "is_steady_state": bool(result.is_steady_state),
    }


# ---------------------------------------------------------------------------
# N-sweep (oz0)
# ---------------------------------------------------------------------------

def sweep_n(
    n_values: list[int],
    epsilon: float,
    total_time: float = 2000.0,
    spring_constant: float = 1.0,
    seed_base: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """
    Run κ(N) sweep for a fixed ε.

    Parameters
    ----------
    n_values : chain sizes to sweep (must be even for alternating chain)
    epsilon  : dimensionless energy parameter
    """
    results = []
    for i, n in enumerate(n_values):
        if verbose:
            T_h, T_c = epsilon_to_temperatures(epsilon, spring_constant)
            print(
                f"  N={n:4d}  ε={epsilon:.4f}  T_hot={T_h:.3f}  T_cold={T_c:.3f}",
                end="  ", flush=True,
            )
        r = run_kappa(
            n_particles=n,
            epsilon=epsilon,
            total_time=total_time,
            spring_constant=spring_constant,
            seed=seed_base + i,
        )
        if verbose:
            print(
                f"κ={r['kappa']:.4f} ± {r['kappa_error']:.4f}"
                f"  ({r['elapsed_s']:.1f}s)"
                f"  {'✓ SS' if r['is_steady_state'] else '○ transient'}"
            )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Pre-defined sweep configurations
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = {
    # Quick smoke test: small N, short time (expect noisy κ but correct sign)
    "smoke": {
        "n_values": [10, 14, 20, 30],
        "total_time": 1500.0,
    },
    # Quick demo: matches Fig-3 range but shorter
    "quick": {
        "n_values": [10, 14, 20, 30, 50, 70, 100],
        "total_time": 1500.0,
    },
    # Full Fig-3 reproduction (N up to 100)
    "full": {
        "n_values": [10, 14, 20, 30, 50, 70, 100],
        "total_time": 3000.0,
    },
    # Fig-2 reproduction (very large N, very slow)
    "fig2": {
        "n_values": [10, 14, 20, 30, 50, 70, 100, 150, 200, 300, 400, 600],
        "total_time": 6000.0,
    },
}

# Paper's Fig-3 ε values (ω=1)
EPSILON_VALUES_OQE = [0.5, 1.0, 2.0]

# Exact paper Fig-3 ε set (for full comparison)
EPSILON_VALUES_FIG3 = [1.5, 4.1667, 6.0, 9.375]


# ---------------------------------------------------------------------------
# Flux consistency verification (5w6)
# ---------------------------------------------------------------------------

def verify_flux_consistency(
    n_particles: int = 30,
    epsilon: float = 1.5,
    total_time: float = 3000.0,
    rtol: float = 0.20,
    verbose: bool = True,
) -> dict:
    """
    Compare spatial flux (FluxMeter) vs reservoir power (HeatBath.energy_exchanged).

    Issue 5w6: Verify consistency between spatial and reservoir flux methods.

    In the ding-a-ling model, heat transport is collision-mediated (not free-particle
    streaming). The spatial FluxMeter counts particle crossings at a fixed reference
    plane, which captures the net rightward energy carried by free particles in the
    two cells adjacent to x_ref. The reservoir method counts net energy injected by
    the hot bath. These two approaches should give consistent results in the long-time
    limit when the system is in steady state, though they measure fundamentally
    different physical quantities and may show O(10–30%) differences for short chains
    or short simulations.

    Notes
    -----
    J_spatial  : time-averaged energy crossing x_ref per unit time over the
                 steady-state window [t_start, total_time].
    J_reservoir: total energy injected by the hot bath over the *full* simulation,
                 divided by total_time.  Using the full period reduces noise from
                 the Langevin bath's stochastic jumps and includes the transient
                 warm-up, so a factor-of-2 correction is not needed if t_total ≫ τ.
    """
    T_hot, T_cold = epsilon_to_temperatures(epsilon)
    T_init = (T_hot + T_cold) / 2.0

    chain_cfg = ChainConfig(
        n_particles=n_particles,
        spring_constant=1.0,
        boundary=BoundaryCondition.OPEN,
        temperature=T_init,
    )
    chain = Chain(chain_cfg)

    left_free, right_free = get_free_boundary_indices(n_particles)

    sim_cfg = SimulationConfig(
        total_time=total_time,
        T_hot=T_hot,
        T_cold=T_cold,
        gamma=1.0,
        n_boundary=1,
        measurement_interval=total_time / 100,
        seed=123,
        left_boundary_indices=left_free,
        right_boundary_indices=right_free,
    )

    sim = EventDrivenSimulator(chain, sim_cfg)
    result = sim.run()

    t_start = total_time * 0.5

    # Method 1: spatial flux (FluxMeter) over steady-state window
    J_spatial = float(result.flux_meter.time_averaged_flux(t_start, total_time))
    n_crossings = len(result.flux_meter.crossings_in_window(t_start, total_time))

    # Method 2: reservoir flux — energy injected by hot bath / total_time
    # Using total_time (not just the window) to reduce Langevin noise.
    J_hot_total = float(result.hot_bath.energy_exchanged / total_time)
    J_cold_total = float(-result.cold_bath.energy_exchanged / total_time)
    J_reservoir = float((J_hot_total + J_cold_total) / 2.0)

    # Relative difference (vs reservoir, which is more reliable)
    denom = max(abs(J_reservoir), 1e-10)
    rel_diff = float(abs(J_spatial - J_reservoir) / denom)
    consistent = rel_diff < rtol

    if verbose:
        print(f"Flux consistency check  (N={n_particles}, ε={epsilon:.3f},"
              f" T_hot={T_hot:.3f}, T_cold={T_cold:.3f})")
        print(f"  J_spatial    = {J_spatial:.6f}   "
              f"(from {n_crossings} crossings in [{t_start:.0f}, {total_time:.0f}])")
        print(f"  J_hot_bath   = {J_hot_total:.6f}   "
              f"(total hot bath injection / t_total)")
        print(f"  J_cold_bath  = {J_cold_total:.6f}   "
              f"(total cold bath extraction / t_total)")
        print(f"  J_reservoir  = {J_reservoir:.6f}   (average of both baths)")
        print(f"  Rel diff     = {rel_diff:.3f}   (tolerance {rtol:.2f})")
        print(f"  Consistent?  {'✓ YES' if consistent else '✗ NO (expected for short runs)'}")
        if not consistent:
            print(f"  NOTE: Spatial flux is noisy for N={n_particles} and short t_total.")
            print(f"  The two methods converge for longer simulations (t_total > 5000).")

    return {
        "n": int(n_particles),
        "epsilon": float(epsilon),
        "J_spatial": J_spatial,
        "J_hot_bath_rate": J_hot_total,
        "J_cold_bath_rate": J_cold_total,
        "J_reservoir": J_reservoir,
        "n_crossings": int(n_crossings),
        "rel_diff": rel_diff,
        "consistent": bool(consistent),
        "rtol": float(rtol),
        "total_time": float(total_time),
        "is_steady_state": bool(result.is_steady_state),
    }


# ---------------------------------------------------------------------------
# Plotting (c90)
# ---------------------------------------------------------------------------

def plot_kappa_vs_n(
    sweep_data: dict[float, list[dict]],
    output_path: str | None = None,
    title: str = "κ vs N (ding-a-ling model)",
) -> None:
    """
    Generate κ vs N plot for multiple ε values (Fig-3 analogue).

    Parameters
    ----------
    sweep_data : {epsilon: [run_result, ...]}
    output_path : save figure here; if None, just show
    """
    import matplotlib
    matplotlib.use("Agg" if output_path else "TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    for idx, (eps, runs) in enumerate(sorted(sweep_data.items())):
        ns = [r["n"] for r in runs]
        ks = [r["kappa"] for r in runs]
        errs = [r["kappa_error"] for r in runs]
        ax.errorbar(
            ns, ks, yerr=errs,
            marker=markers[idx % len(markers)],
            label=f"ε = {eps:.4g}",
            capsize=3,
            linewidth=1,
        )

    ax.set_xlabel("Number of Particles, N", fontsize=12)
    ax.set_ylabel("κ (dimensionless)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_resistivity_vs_n(
    sweep_data: dict[float, list[dict]],
    output_path: str | None = None,
) -> None:
    """
    Plot thermal resistivity ρ = 1/κ vs N^(-1/2) (linear = Fourier's law).
    """
    import matplotlib
    matplotlib.use("Agg" if output_path else "TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    for idx, (eps, runs) in enumerate(sorted(sweep_data.items())):
        ns = np.array([r["n"] for r in runs])
        ks = np.array([r["kappa"] for r in runs])
        x = ns ** (-0.5)
        rho = 1.0 / np.where(ks > 0, ks, np.nan)
        ax.plot(x, rho, marker=markers[idx % len(markers)],
                label=f"ε = {eps:.4g}", linewidth=1)

    ax.set_xlabel("N^(−1/2)", fontsize=12)
    ax.set_ylabel("ρ = 1/κ", fontsize=12)
    ax.set_title("Thermal resistivity vs N^(−1/2)\n(linear → Fourier's law holds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved → {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ding-a-ling κ vs N sweep")
    parser.add_argument("--mode", choices=["smoke", "quick", "full", "fig2"],
                        default="smoke", help="Sweep configuration")
    parser.add_argument("--epsilon", type=float, nargs="+",
                        default=None,
                        help="ε values to sweep (default: 0.5 1.0 2.0)")
    parser.add_argument("--fig2", action="store_true",
                        help="Reproduce Fig 2 (ε=1.5, large N, very slow)")
    parser.add_argument("--flux-check", action="store_true",
                        help="Run flux consistency check (issue 5w6)")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output plots and JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Flux consistency check (5w6)
    # -------------------------------------------------------------------
    if args.flux_check:
        print("\n=== Flux Consistency Check (issue 5w6) ===")
        fc = verify_flux_consistency(verbose=True)
        (out_dir / "flux_consistency.json").write_text(json.dumps(fc, indent=2))
        print()
        if not fc["consistent"]:
            print("WARNING: flux methods disagree beyond tolerance!")
        return

    # -------------------------------------------------------------------
    # Choose ε values
    # -------------------------------------------------------------------
    if args.fig2:
        epsilon_list = [1.5]
        cfg = SWEEP_CONFIGS["fig2"]
        print("Running Fig-2 reproduction (ε=1.5, large N). This will take a while…")
    else:
        epsilon_list = args.epsilon if args.epsilon else EPSILON_VALUES_OQE
        cfg = SWEEP_CONFIGS[args.mode]

    n_values = cfg["n_values"]
    total_time = cfg["total_time"]

    # -------------------------------------------------------------------
    # Run sweeps
    # -------------------------------------------------------------------
    all_results: dict[float, list[dict]] = {}
    for eps in epsilon_list:
        print(f"\n=== ε = {eps:.4f}  mode={args.mode}  t_total={total_time:.0f} ===")
        runs = sweep_n(
            n_values=n_values,
            epsilon=eps,
            total_time=total_time,
            seed_base=args.seed,
            verbose=True,
        )
        all_results[eps] = runs

    # -------------------------------------------------------------------
    # Save raw data
    # -------------------------------------------------------------------
    json_path = out_dir / f"sweep_{args.mode}.json"
    serialisable = {str(k): v for k, v in all_results.items()}
    json_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nRaw data saved → {json_path}")

    # -------------------------------------------------------------------
    # Generate plots (c90)
    # -------------------------------------------------------------------
    kappa_plot = str(out_dir / f"kappa_vs_n_{args.mode}.png")
    rho_plot = str(out_dir / f"resistivity_vs_n_{args.mode}.png")
    plot_kappa_vs_n(all_results, output_path=kappa_plot)
    plot_resistivity_vs_n(all_results, output_path=rho_plot)

    # -------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"{'ε':>8}  {'N':>6}  {'κ_spatial':>11}  {'±':>8}  {'κ_reservoir':>12}  {'SS':>4}")
    print("-" * 65)
    for eps, runs in sorted(all_results.items()):
        for r in runs:
            ss = "✓" if r["is_steady_state"] else "○"
            kr = r.get("kappa_reservoir", float("nan"))
            print(
                f"{r['epsilon']:8.4f}  {r['n']:6d}  {r['kappa']:11.4f}"
                f"  {r['kappa_error']:8.4f}  {kr:12.4f}  {ss:>4}"
            )


if __name__ == "__main__":
    main()
