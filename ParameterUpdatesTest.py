"""
Simple experiment: does the NUMBER OF PARAMETER UPDATES change the QUALITY of a
(noisy) quantum circuit?

Task: train a 2-qubit circuit so that it outputs the state |11>.
Quality = fraction of shots that measure '11'  (1.0 = perfect, higher = better).

Hypothesis: more updates = more circuits run = more chances for noise to
compromise accuracy, so more updates might be counter-productive.

Needs only: qiskit, qiskit-aer   (everything else is Python's standard library)
"""
import importlib
import math
import random

try:
    qiskit_module = importlib.import_module("qiskit")
    QuantumCircuit = qiskit_module.QuantumCircuit
    transpile = qiskit_module.transpile
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "qiskit is required for this experiment. "
        "Install it with: pip install qiskit"
    ) from exc

try:
    aer_module = importlib.import_module("qiskit_aer")
    noise_module = importlib.import_module("qiskit_aer.noise")
    AerSimulator = aer_module.AerSimulator
    NoiseModel = noise_module.NoiseModel
    depolarizing_error = noise_module.depolarizing_error
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "qiskit-aer is required for this experiment. "
        "Install it with: pip install qiskit-aer"
    ) from exc

# ---------------- settings ----------------
SHOTS = 1000                 # held constant
LEARNING_RATE = 0.5          # held constant
P_1Q = 0.001                 # 1-qubit gate error
P_2Q = 0.02                  # 2-qubit gate (CX) error
DRIFT = 0.0005               # extra CX error added per update (0 = constant noise)
UPDATE_COUNTS = [0, 5, 10, 20, 40, 80]    # <-- the variable being tested
TRIALS = 5                   # repeats per setting, to average out randomness


def make_backend(p1, p2):
    """A simulator whose gates are noisy (depolarizing errors)."""
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return AerSimulator(noise_model=noise)


def prob_11(params, backend):
    """Run the circuit and return the fraction of shots that gave '11'."""
    a, b, c, d = params
    qc = QuantumCircuit(2)
    qc.ry(a, 0)
    qc.ry(b, 1)
    qc.cx(0, 1)
    qc.ry(c, 0)
    qc.ry(d, 1)
    qc.measure_all()
    qc = transpile(qc, backend, optimization_level=0)
    counts = backend.run(qc, shots=SHOTS).result().get_counts()
    return counts.get("11", 0) / SHOTS


def train(n_updates):
    """Gradient ascent on prob_11. Returns the final parameters."""
    params = [random.uniform(0, 2 * math.pi) for _ in range(4)]
    for k in range(n_updates):
        backend = make_backend(P_1Q, min(P_2Q + DRIFT * k, 0.5))
        grads = []
        for i in range(4):
            plus = params.copy();  plus[i] += math.pi / 2
            minus = params.copy(); minus[i] -= math.pi / 2
            # parameter-shift rule: exact gradient formula for rotation gates
            grads.append((prob_11(plus, backend) - prob_11(minus, backend)) / 2)
        params = [p + LEARNING_RATE * g for p, g in zip(params, grads)]  # the update
    return params


def main():
    final_backend = make_backend(P_1Q, P_2Q)    # same noise for every final test
    print("updates | circuits run per trial | average quality")
    for n in UPDATE_COUNTS:
        qualities = []
        for trial in range(TRIALS):
            random.seed(1000 * n + trial)
            params = train(n)
            qualities.append(prob_11(params, final_backend))
        avg = sum(qualities) / len(qualities)
        print(f"{n:7d} | {n * 8:22d} | {avg:.3f}")


if __name__ == "__main__":
    main()


# This script tests the effect of different numbers of parameter updates on the quality of the final parameters in a noisy quantum circuit simulation.
# It uses gradient ascent with the parameter-shift rule to optimize the parameters and evaluates the resulting quality under a fixed noise model.
# The script prints the number of updates, the number of circuits run per trial, and the average quality for each setting.
    

