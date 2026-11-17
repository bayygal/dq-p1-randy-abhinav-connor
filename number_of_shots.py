

import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

print("Starting CLOPS simulation...", flush=True)

NUM_QUBITS = 5
NUM_LAYERS = 10
REPEATS = 5
SHOT_COUNTS = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000]

def make_circuit(num_qubits, num_layers):
    qc = QuantumCircuit(num_qubits)

    for layer in range(num_layers):
        for q in range(num_qubits):
            qc.ry(0.3 * (layer + 1), q)
            qc.rz(0.2 * (q + 1), q)

        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    qc.measure_all()
    return qc

circuit = make_circuit(NUM_QUBITS, NUM_LAYERS)

simulator = AerSimulator(
    method="statevector",
    max_parallel_threads=1
)

compiled = transpile(
    circuit,
    simulator,
    optimization_level=0
)

depth = compiled.remove_final_measurements(
    inplace=False
).depth()

# Warm up simulator
simulator.run(compiled, shots=100).result()

times = []
clops_values = []

for shots in SHOT_COUNTS:
    print(f"Testing {shots} shots...", flush=True)

    trial_times = []

    for repeat in range(REPEATS):
        start = time.perf_counter()

        simulator.run(compiled, shots=shots).result()

        elapsed = time.perf_counter() - start
        trial_times.append(elapsed)

    median_time = np.median(trial_times)

    clops_est = shots * depth / median_time

    times.append(median_time)
    clops_values.append(clops_est)

    print(
        f"Shots: {shots:>6} | "
        f"Time: {median_time:.5f} s | "
        f"CLOPS estimate: {clops_est:,.0f}",
        flush=True
    )

# Plot estimated CLOPS
plt.figure(figsize=(9, 5))
plt.plot(SHOT_COUNTS, clops_values, marker="o")
plt.xlabel("Number of Shots")
plt.ylabel("Estimated Circuit Layers per Second")
plt.title("Effect of Shot Count on CLOPS")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clops_vs_shots.png", dpi=300)

# Plot execution time
plt.figure(figsize=(9, 5))
plt.plot(SHOT_COUNTS, times, marker="s")
plt.xlabel("Number of Shots")
plt.ylabel("Execution Time (seconds)")
plt.title("Effect of Shot Count on Simulation Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("execution_time_vs_shots.png", dpi=300)

print("Simulation complete! Graphs saved.", flush=True)
