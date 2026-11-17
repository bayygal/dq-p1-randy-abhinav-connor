"""
Shared CLOPS and quality measurements, following Wack et al. 2021
(arXiv:2110.14108).

    CLOPS = M * K * S * D / total_time

    M = number of circuit templates
    K = number of parameter updates per template
    S = number of shots per circuit
    D = number of QV layers per circuit

A QV layer is a random permutation of the qubits followed by a random
two-qubit (SU(4)) gate on each pair of qubits. Parameter values for each
update are derived from the previous run's measurement results, like the
paper's benchmark does.

Quality is the heavy output probability (HOP) on a noisy simulator: the
fraction of shots landing on outputs whose ideal probability is above the
median. HOP > 2/3 is the paper's passing threshold; 0.5 is pure noise.
"""
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

P_1Q = 0.001                 # 1-qubit gate error for the quality test
P_2Q = 0.01                  # 2-qubit gate (CX) error for the quality test


def make_simulator():
    return AerSimulator(method="statevector", max_parallel_threads=1)


def make_noisy_simulator(p1=P_1Q, p2=P_2Q):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u", "rz", "ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return AerSimulator(noise_model=noise, max_parallel_threads=1)


def add_su4_block(qc, params, a, b):
    """General 2-qubit gate (15 parameters, 3 CX) on qubits a and b."""
    qc.u(params[0], params[1], params[2], a)
    qc.u(params[3], params[4], params[5], b)
    qc.cx(b, a)
    qc.rz(params[6], a)
    qc.ry(params[7], b)
    qc.cx(a, b)
    qc.ry(params[8], b)
    qc.cx(b, a)
    qc.u(params[9], params[10], params[11], a)
    qc.u(params[12], params[13], params[14], b)


def make_qv_template(num_qubits, num_layers, rng):
    """Parameterized QV circuit with `num_layers` QV layers."""
    pairs_per_layer = num_qubits // 2
    params = ParameterVector("θ", 15 * pairs_per_layer * num_layers)
    qc = QuantumCircuit(num_qubits)
    i = 0
    for _ in range(num_layers):
        perm = rng.permutation(num_qubits)
        for p in range(pairs_per_layer):
            add_su4_block(qc, params[i:i + 15], int(perm[2 * p]), int(perm[2 * p + 1]))
            i += 15
    qc.measure_all()
    return qc


def next_parameter_values(counts, num_params):
    """Pick the next parameters from the last run's results."""
    seed = int(max(counts, key=counts.get), 2)
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, num_params)


def measure_clops(num_qubits, num_layers, shots, updates, templates, simulator, seed):
    """Run M templates x K updates x S shots and return (CLOPS, seconds)."""
    rng = np.random.default_rng(seed)
    compiled = [
        transpile(make_qv_template(num_qubits, num_layers, rng), simulator,
                  optimization_level=0)
        for _ in range(templates)
    ]

    start = time.perf_counter()
    for template in compiled:
        n_params = template.num_parameters
        values = rng.uniform(0, 2 * np.pi, n_params)
        for _ in range(updates):
            bound = template.assign_parameters(values)
            counts = simulator.run(bound, shots=shots).result().get_counts()
            values = next_parameter_values(counts, n_params)
    elapsed = time.perf_counter() - start

    clops = templates * updates * shots * num_layers / elapsed
    return clops, elapsed


def heavy_output_probability(num_qubits, num_layers, noisy_sim, seed,
                             circuits=20, shots=1000):
    """Average fraction of noisy shots that land on 'heavy' outputs."""
    rng = np.random.default_rng(seed)
    hops = []
    for _ in range(circuits):
        template = make_qv_template(num_qubits, num_layers, rng)
        qc = template.assign_parameters(
            rng.uniform(0, 2 * np.pi, template.num_parameters))

        ideal = Statevector(qc.remove_final_measurements(inplace=False))
        probs = ideal.probabilities_dict()
        median = np.median([probs.get(format(i, f"0{num_qubits}b"), 0.0)
                            for i in range(2 ** num_qubits)])
        heavy = {b for b, p in probs.items() if p > median}

        compiled = transpile(qc, noisy_sim, optimization_level=0)
        counts = noisy_sim.run(compiled, shots=shots).result().get_counts()
        hops.append(sum(c for b, c in counts.items() if b in heavy) / shots)
    return float(np.mean(hops))
