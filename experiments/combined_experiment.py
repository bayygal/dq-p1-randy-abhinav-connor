"""
All three variables together, measuring both speed (CLOPS) and accuracy.

Same task as ParameterUpdatesTest.py: train a noisy 2-qubit circuit to output
'11', using the parameter-shift rule. Here the circuit has D layers, so all
three variables act on the same circuit:

    D = QV layers           (more layers = more gates = more noise)
    S = shots per circuit   (fewer shots = noisier training signal)
    K = parameter updates   (training steps)

Speed:    CLOPS = circuits run * S * D / training time
Accuracy: fraction of 1000 shots that give '11' after training (1.0 = perfect)
"""
import itertools
import math
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from paths import DATA

P_1Q = 0.001                 # same noise as ParameterUpdatesTest.py
P_2Q = 0.02
LEARNING_RATE = 0.5
EVAL_SHOTS = 1000
TRIALS = 10

LAYER_COUNTS = [1, 2, 4, 8]
SHOT_COUNTS = [100, 1000]
UPDATE_COUNTS = [2, 5, 10, 20]


def make_backend():
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(P_1Q, 1), ["ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(P_2Q, 2), ["cx"])
    return AerSimulator(noise_model=noise, max_parallel_threads=1)


def make_template(num_layers, backend):
    """ry on both qubits, then D x (cx, ry, ry). D = 1 is the original circuit."""
    params = ParameterVector("θ", 2 * (num_layers + 1))
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    for layer in range(num_layers):
        qc.cx(0, 1)
        qc.ry(params[2 * layer + 2], 0)
        qc.ry(params[2 * layer + 3], 1)
    qc.measure_all()
    return transpile(qc, backend, optimization_level=0)


def prob_11(template, values, backend, shots):
    counts = backend.run(template.assign_parameters(values), shots=shots).result().get_counts()
    return counts.get("11", 0) / shots


def train(template, backend, shots, updates, rng):
    """Returns (final parameters, circuits run, seconds)."""
    n = template.num_parameters
    values = rng.uniform(0, 2 * math.pi, n)
    circuits = 0
    start = time.perf_counter()
    for _ in range(updates):
        grads = np.zeros(n)
        for i in range(n):
            plus, minus = values.copy(), values.copy()
            plus[i] += math.pi / 2
            minus[i] -= math.pi / 2
            grads[i] = (prob_11(template, plus, backend, shots)
                        - prob_11(template, minus, backend, shots)) / 2
            circuits += 2
        values = values + LEARNING_RATE * grads
    return values, circuits, time.perf_counter() - start


def main():
    backend = make_backend()
    templates = {d: make_template(d, backend) for d in LAYER_COUNTS}
    prob_11(templates[1], np.zeros(4), backend, 100)  # warm up

    rows = []
    for d, s, k in itertools.product(LAYER_COUNTS, SHOT_COUNTS, UPDATE_COUNTS):
        for trial in range(TRIALS):
            rng = np.random.default_rng(10_000 * d + 100 * k + trial)
            values, circuits, elapsed = train(templates[d], backend, s, k, rng)
            accuracy = prob_11(templates[d], values, backend, EVAL_SHOTS)
            rows.append({"D": d, "S": s, "K": k, "trial": trial, "circuits": circuits,
                         "time_s": elapsed, "clops": circuits * s * d / elapsed,
                         "accuracy": accuracy})
        cell = pd.DataFrame(rows[-TRIALS:])
        print(f"layers {d:>2} | shots {s:>4} | updates {k:>2} | "
              f"CLOPS {cell['clops'].mean():>10,.0f} | accuracy {cell['accuracy'].mean():.3f}",
              flush=True)

    pd.DataFrame(rows).to_csv(DATA / "combined_results.csv", index=False)
    print("Saved combined_results.csv")


if __name__ == "__main__":
    main()
