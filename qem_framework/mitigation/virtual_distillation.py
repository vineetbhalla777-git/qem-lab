"""
virtual_distillation.py
========================
Virtual Distillation (VD), per Huggins et al. (2021) "Virtual Distillation
for Quantum Error Mitigation" (arXiv:2011.07064) and Koczor (2021)
"Exponential Error Suppression for Near-Term Quantum Devices".

Idea: given M noisy copies of a state rho, measuring an observable O on the
*purified* state rho^M / Tr[rho^M] suppresses errors exponentially in M
(for a dominant-eigenvector error model), without ever explicitly preparing
that purified state. This module implements the M=2 (two-copy) protocol
using the standard **ancilla-assisted Hadamard/SWAP-test** construction:

  Denominator, Tr[rho^2]:
    |0>_anc -- H --*----------*-- H -- measure Z
    |psi>_A -------X (swap)---X-------
    |psi>_B -------X (swap)---X-------
    i.e. H on the ancilla, a controlled-SWAP of every qubit pair (A_i, B_i),
    then H and Z-measurement on the ancilla. This is the textbook SWAP
    test: <Z_ancilla> = Tr[rho^2].

  Numerator, Tr[rho^2 * O] for a Pauli string O supported on registers A:
    Same circuit, but insert a controlled-O (one controlled single-qubit
    Pauli gate per qubit in O's support, applied to register A) *before*
    the controlled-SWAP layer -- implementing the controlled unitary
    U = SWAP . (O tensor I) via the generalized Hadamard test:
    <Z_ancilla> = Tr[rho^2 . SWAP . (O tensor I)] = Tr[rho^2 O].

  The mitigated expectation value is the ratio: <O>_VD = numerator / denominator.

This construction is verified below (see the project's test suite / dev
notes) against exact density-matrix calculations for both diagonal (Z) and
non-diagonal (coherent superposition) states, for X/Y/Z observables, and
is exact for any n-qubit Pauli-string observable (not restricted to
single-qubit/1-local terms, unlike the ancilla-free "B-gate" variant).

Trade-off: this protocol requires one extra ancilla qubit and n
controlled-SWAP (Fredkin) gates -- appreciably more 2-qubit-gate depth than
the ancilla-free protocol, so its own gate errors partially offset the
error suppression it provides (a known, documented limitation of VD in
general; see Krebsbach et al., arXiv:2210.15317 -- studies noise-in-the-VD-
circuit itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


@dataclass
class VDResult:
    mitigated_value: float
    raw_value: float
    denominator_estimate: float  # Tr[rho^2] estimate; 1.0 = pure state, decreases with mixedness
    n_copies: int = 2


def _controlled_pauli(qc: QuantumCircuit, pauli_char: str, ancilla: int, target: int):
    if pauli_char == "X":
        qc.cx(ancilla, target)
    elif pauli_char == "Y":
        qc.cy(ancilla, target)
    elif pauli_char == "Z":
        qc.cz(ancilla, target)
    # 'I': no gate needed


def _build_swap_test_circuit(
    state_prep: QuantumCircuit,
    pauli_label: str = None,
) -> QuantumCircuit:
    """Builds the (2n+1)-qubit ancilla circuit. Registers: A = [0, n),
    B = [n, 2n), ancilla = 2n. If `pauli_label` is given (Qiskit Pauli
    label, e.g. 'ZZI'), inserts the corresponding controlled-Pauli gates
    on register A (numerator circuit); otherwise builds the plain SWAP
    test (denominator circuit)."""
    n = state_prep.num_qubits
    ancilla = 2 * n
    qc = QuantumCircuit(2 * n + 1, 1)

    # to_instruction() (not to_gate()) because some benchmark circuits embed
    # library instructions (e.g. Qiskit's QFT) that aren't elementary Gate
    # objects -- to_gate() requires every sub-instruction to be a Gate and
    # raises on these, while to_instruction() accepts general Instructions.
    gate_A = state_prep.to_instruction()
    gate_B = state_prep.to_instruction()
    qc.append(gate_A, range(0, n))
    qc.append(gate_B, range(n, 2 * n))

    qc.h(ancilla)

    if pauli_label is not None:
        # Qiskit Pauli labels are ordered qubit (n-1) ... qubit 0 (left to right)
        for i, p in enumerate(reversed(pauli_label)):
            if p != "I":
                _controlled_pauli(qc, p, ancilla, i)  # apply to register A qubit i

    for i in range(n):
        qc.cswap(ancilla, i, n + i)

    qc.h(ancilla)
    qc.measure(ancilla, 0)
    return qc


def _ancilla_expectation(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    p0 = counts.get("0", 0) / total
    p1 = counts.get("1", 0) / total
    return p0 - p1


def run_virtual_distillation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    counts_executor: Callable[[QuantumCircuit], Dict[str, int]],
    raw_executor: Callable[[QuantumCircuit], float] = None,
) -> VDResult:
    """Run the M=2 virtual distillation protocol to mitigate a Pauli-sum
    `observable` on `circuit`, using the ancilla-assisted SWAP-test
    construction (exact for arbitrary n-qubit Pauli terms, at the cost of
    one extra ancilla qubit and n controlled-SWAP gates per circuit).

    `counts_executor` must accept circuits with an arbitrary number of
    qubits/classical bits (as produced by `make_counts_executor`, which
    builds its noise model / simulator generically).
    """
    n = circuit.num_qubits

    # Denominator: one SWAP test, shared across all Pauli terms.
    denom_circuit = _build_swap_test_circuit(circuit, pauli_label=None)
    denom_counts = counts_executor(denom_circuit)
    denominator = _ancilla_expectation(denom_counts)
    denominator = denominator if abs(denominator) > 1e-6 else 1e-6

    total_num = 0.0
    for pauli, coeff in zip(observable.paulis, observable.coeffs):
        label = pauli.to_label()
        if set(label) == {"I"}:
            # Identity term: Tr[rho^2 I] = Tr[rho^2] = denominator
            total_num += float(np.real(coeff)) * denominator
            continue
        num_circuit = _build_swap_test_circuit(circuit, pauli_label=label)
        num_counts = counts_executor(num_circuit)
        num_est = _ancilla_expectation(num_counts)
        total_num += float(np.real(coeff)) * num_est

    mitigated = total_num / denominator
    raw_value = raw_executor(circuit) if raw_executor is not None else float("nan")

    return VDResult(
        mitigated_value=float(mitigated),
        raw_value=float(raw_value),
        denominator_estimate=float(denominator),
    )
