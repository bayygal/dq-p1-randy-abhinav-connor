"""
Full-factorial experiment: effects of QV layers (D), shots (S) and parameter
updates (K) on CLOPS, including their interactions.

Every combination of D x S x K is run REPEATS times in a shuffled order, so
slow drift in machine speed cannot line up with any one factor. Quality
(heavy output probability) is measured separately, since it depends on the
circuit (D and qubit count), not on how many shots or updates are run.

Outputs:
    factorial_results.csv   one row per (D, S, K, repeat) with CLOPS and time
    quality_results.csv     HOP vs D at fixed qubit count
    qubit_results.csv       CLOPS and HOP vs qubit count, with D = qubits
Then runs analyze_clops.py to make the plots and statistics.
"""
import itertools
import random
import pandas as pd
from clops_harness import (
    make_simulator, make_noisy_simulator, measure_clops, heavy_output_probability,
)
import analyze_clops
from paths import DATA

NUM_QUBITS = 4               # held constant (even, so no qubit sits idle)
M_TEMPLATES = 10             # held constant
REPEATS = 3
SEED = 2026

LAYER_COUNTS = [1, 2, 4, 8, 16]          # D
SHOT_COUNTS = [100, 500, 1000, 5000]     # S
UPDATE_COUNTS = [1, 5, 10, 20]           # K

QUBIT_COUNTS = [2, 4, 6, 8]              # qubit sweep, D = qubits (square QV)
QUBIT_SWEEP_SHOTS = 100
QUBIT_SWEEP_UPDATES = 10


def run_factorial(simulator):
    runs = [(d, s, k, r) for d, s, k in itertools.product(LAYER_COUNTS, SHOT_COUNTS, UPDATE_COUNTS)
            for r in range(REPEATS)]
    random.Random(SEED).shuffle(runs)

    rows = []
    for i, (d, s, k, r) in enumerate(runs, 1):
        clops, elapsed = measure_clops(NUM_QUBITS, d, s, k, M_TEMPLATES, simulator,
                                       seed=SEED + 1000 * i)
        rows.append({"D": d, "S": s, "K": k, "M": M_TEMPLATES, "repeat": r, "run_order": i,
                     "time_s": elapsed, "clops": clops})
        if i % 20 == 0 or i == len(runs):
            print(f"  factorial run {i}/{len(runs)}", flush=True)
    return pd.DataFrame(rows).sort_values(["D", "S", "K", "repeat"])


def run_quality(noisy_sim):
    rows = []
    for d in LAYER_COUNTS:
        hop = heavy_output_probability(NUM_QUBITS, d, noisy_sim, seed=SEED + d)
        rows.append({"D": d, "qubits": NUM_QUBITS, "hop": hop})
        print(f"  quality D={d:>2}: HOP = {hop:.3f}", flush=True)
    return pd.DataFrame(rows)


def run_qubit_sweep(simulator, noisy_sim):
    rows = []
    for n in QUBIT_COUNTS:
        for r in range(REPEATS):
            clops, elapsed = measure_clops(n, n, QUBIT_SWEEP_SHOTS, QUBIT_SWEEP_UPDATES,
                                           M_TEMPLATES, simulator, seed=SEED + 10 * n + r)
            rows.append({"qubits": n, "D": n, "repeat": r, "time_s": elapsed,
                         "clops": clops, "hop": None})
        hop = heavy_output_probability(n, n, noisy_sim, seed=SEED + n)
        for row in rows[-REPEATS:]:
            row["hop"] = hop
        print(f"  qubits={n}: HOP = {hop:.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    simulator = make_simulator()
    noisy_sim = make_noisy_simulator()

    # Warm up simulator
    measure_clops(NUM_QUBITS, 1, 100, 1, M_TEMPLATES, simulator, seed=0)

    n_cells = len(LAYER_COUNTS) * len(SHOT_COUNTS) * len(UPDATE_COUNTS)
    print(f"Running factorial: {n_cells} cells x {REPEATS} repeats...", flush=True)
    run_factorial(simulator).to_csv(DATA / "factorial_results.csv", index=False)

    print("Measuring quality vs QV layers...", flush=True)
    run_quality(noisy_sim).to_csv(DATA / "quality_results.csv", index=False)

    print("Running qubit-count sweep...", flush=True)
    run_qubit_sweep(simulator, noisy_sim).to_csv(DATA / "qubit_results.csv", index=False)

    print("Data saved. Running analysis...\n", flush=True)
    analyze_clops.main()


if __name__ == "__main__":
    main()
