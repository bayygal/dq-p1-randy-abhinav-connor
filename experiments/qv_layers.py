"""
Experiment: how does the NUMBER OF QUANTUM VOLUME (QV) LAYERS affect CLOPS?

Sweeps D (QV layers) while holding qubits, templates, updates and shots fixed.
Also measures quality (heavy output probability) on a noisy simulator, to test
the hypothesis that more layers = more noise = less accurate results.
See clops_harness.py for the CLOPS definition (Wack et al. 2021).
"""
import numpy as np
import matplotlib.pyplot as plt
from clops_harness import (
    make_simulator, make_noisy_simulator, measure_clops, heavy_output_probability,
)
from paths import FIGURES

print("Starting QV layers CLOPS simulation...", flush=True)

NUM_QUBITS = 4               # held constant (even, so no qubit sits idle)
M_TEMPLATES = 20             # held constant (paper uses 100; reduced for speed)
K_UPDATES = 10               # held constant (paper value)
SHOTS = 100                  # held constant (paper value)
REPEATS = 3                  # repeats per setting, median is reported
LAYER_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]   # <-- the variable


def main():
    simulator = make_simulator()
    noisy_sim = make_noisy_simulator()

    # Warm up simulator
    measure_clops(NUM_QUBITS, 1, SHOTS, K_UPDATES, M_TEMPLATES, simulator, seed=0)

    clops_values, times, hops = [], [], []

    print(f"\n{'layers':>6} | {'time (s)':>9} | {'CLOPS':>10} | {'heavy output prob':>17}")
    for d in LAYER_COUNTS:
        trials = [measure_clops(NUM_QUBITS, d, SHOTS, K_UPDATES, M_TEMPLATES,
                                simulator, seed=100 * d + r)
                  for r in range(REPEATS)]
        clops = float(np.median([t[0] for t in trials]))
        elapsed = float(np.median([t[1] for t in trials]))
        hop = heavy_output_probability(NUM_QUBITS, d, noisy_sim, seed=d)

        clops_values.append(clops)
        times.append(elapsed)
        hops.append(hop)
        print(f"{d:>6} | {elapsed:>9.3f} | {clops:>10,.0f} | {hop:>17.3f}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(LAYER_COUNTS, clops_values, marker="o")
    axes[0].set_xlabel("Number of QV Layers (D)")
    axes[0].set_ylabel("CLOPS")
    axes[0].set_title("Effect of QV Layers on CLOPS")

    axes[1].plot(LAYER_COUNTS, times, marker="s", color="tab:orange")
    axes[1].set_xlabel("Number of QV Layers (D)")
    axes[1].set_ylabel("Total Execution Time (seconds)")
    axes[1].set_title(f"Time for {M_TEMPLATES}×{K_UPDATES} circuits × {SHOTS} shots")

    axes[2].plot(LAYER_COUNTS, hops, marker="^", color="tab:green")
    axes[2].axhline(2 / 3, linestyle="--", color="gray", label="QV pass threshold (2/3)")
    axes[2].axhline(0.5, linestyle=":", color="red", label="Pure noise (0.5)")
    axes[2].set_xlabel("Number of QV Layers (D)")
    axes[2].set_ylabel("Heavy Output Probability")
    axes[2].set_title("Effect of QV Layers on Quality (noisy)")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "clops_vs_qv_layers.png", dpi=300)

    print("\nSimulation complete! Graph saved to clops_vs_qv_layers.png", flush=True)


if __name__ == "__main__":
    main()
