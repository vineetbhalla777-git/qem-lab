"""
executors.py
============
Thin wrappers around Qiskit Aer that turn "a circuit + a noise model" into
the two things mitigation techniques need:
    1. An expectation-value executor  (circuit) -> float
    2. A raw-counts executor          (circuit) -> Dict[bitstring, count]

Mitiq's API expects executors of the first kind; the benchmarking /
distribution-fidelity code in analysis/ uses the second kind.
"""

from typing import Callable, Dict, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp, Statevector


def make_counts_executor(
    noise_model: Optional[NoiseModel],
    shots: int = 4096,
    seed: Optional[int] = None,
) -> Callable[[QuantumCircuit], Dict[str, int]]:
    """Returns a function: measured QuantumCircuit -> counts dict.
    The circuit passed in MUST already contain measurements."""
    backend = AerSimulator(noise_model=noise_model)

    def executor(circuit: QuantumCircuit) -> Dict[str, int]:
        tqc = transpile(circuit, backend, optimization_level=1)
        result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
        return result.get_counts()

    return executor


def make_expectation_executor(
    noise_model: Optional[NoiseModel],
    observable: SparsePauliOp,
    shots: int = 4096,
    seed: Optional[int] = None,
) -> Callable[[QuantumCircuit], float]:
    """Returns a Mitiq-compatible executor: (unmeasured) QuantumCircuit -> float.
    Computes <observable> by measuring in the appropriate Pauli bases and
    combining shot statistics -- this models a real, sampling-noise-limited
    expectation value estimate (not a noiseless statevector shortcut)."""
    backend = AerSimulator(noise_model=noise_model)

    def executor(circuit: QuantumCircuit) -> float:
        total = 0.0
        for pauli, coeff in zip(observable.paulis, observable.coeffs):
            label = pauli.to_label()
            if set(label) == {"I"}:
                total += float(np.real(coeff))
                continue
            meas_qc = circuit.copy()
            for q, p in enumerate(reversed(label)):
                if p == "X":
                    meas_qc.h(q)
                elif p == "Y":
                    meas_qc.sdg(q)
                    meas_qc.h(q)
            meas_qc.measure_all()
            tqc = transpile(meas_qc, backend, optimization_level=1)
            result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
            counts = result.get_counts()
            n_shots = sum(counts.values())
            exp_val = 0.0
            active_qubits = [i for i, p in enumerate(reversed(label)) if p != "I"]
            for bitstring, cnt in counts.items():
                bits = bitstring.replace(" ", "")[::-1]  # little-endian -> index order
                parity = 1
                for q in active_qubits:
                    if bits[q] == "1":
                        parity *= -1
                exp_val += parity * cnt
            exp_val /= n_shots
            total += float(np.real(coeff)) * exp_val
        return total

    return executor


def noiseless_expectation(circuit: QuantumCircuit, observable: SparsePauliOp) -> float:
    """Exact statevector expectation value (ground truth, no sampling noise)."""
    sv = Statevector.from_instruction(circuit)
    return float(np.real(sv.expectation_value(observable)))
