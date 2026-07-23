"""
circuits.py
===========
Benchmark quantum circuits used across the project (Phase III):
    - Bell state preparation
    - GHZ state generation
    - Quantum Fourier Transform (QFT)
    - Variational Quantum Eigensolver (VQE) ansatz (with a simple Ising
      Hamiltonian and exact-diagonalization ground truth for scoring)
    - Quantum Approximate Optimization Algorithm (QAOA) for MaxCut

Every builder returns a `BenchmarkCircuit` with the circuit itself plus
enough metadata (ideal expectation value / ideal bitstring distribution)
to score mitigation performance against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, SparsePauliOp


@dataclass
class BenchmarkCircuit:
    name: str
    circuit: QuantumCircuit               # circuit WITHOUT measurement
    n_qubits: int
    depth: int = field(init=False)
    ideal_statevector: Optional[Statevector] = None
    observable: Optional[SparsePauliOp] = None      # for expectation-value benchmarks
    ideal_expectation: Optional[float] = None
    ideal_distribution: Optional[Dict[str, float]] = None  # for sampling benchmarks
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.depth = self.circuit.depth()

    def measured_circuit(self) -> QuantumCircuit:
        qc = self.circuit.copy()
        qc.measure_all()
        return qc


# --------------------------------------------------------------------------
# Bell state
# --------------------------------------------------------------------------
def bell_state() -> BenchmarkCircuit:
    qc = QuantumCircuit(2, name="bell_state")
    qc.h(0)
    qc.cx(0, 1)
    sv = Statevector.from_instruction(qc)
    ideal_dist = {"00": 0.5, "11": 0.5}
    return BenchmarkCircuit(
        name="Bell State",
        circuit=qc,
        n_qubits=2,
        ideal_statevector=sv,
        ideal_distribution=ideal_dist,
        metadata={"description": "Maximally entangled 2-qubit Bell pair |00>+|11>"},
    )


# --------------------------------------------------------------------------
# GHZ state
# --------------------------------------------------------------------------
def ghz_state(n_qubits: int = 4) -> BenchmarkCircuit:
    qc = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}")
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    sv = Statevector.from_instruction(qc)
    ideal_dist = {"0" * n_qubits: 0.5, "1" * n_qubits: 0.5}
    return BenchmarkCircuit(
        name=f"GHZ State ({n_qubits}q)",
        circuit=qc,
        n_qubits=n_qubits,
        ideal_statevector=sv,
        ideal_distribution=ideal_dist,
        metadata={"description": f"{n_qubits}-qubit GHZ state, tests multi-qubit coherence"},
    )


# --------------------------------------------------------------------------
# Quantum Fourier Transform
# --------------------------------------------------------------------------
def qft_circuit(n_qubits: int = 4, input_state: Optional[str] = None) -> BenchmarkCircuit:
    """Applies QFT to a computational basis input state (default: |0..01>)."""
    if input_state is None:
        input_state = "0" * (n_qubits - 1) + "1"
    assert len(input_state) == n_qubits

    qc = QuantumCircuit(n_qubits, name=f"qft_{n_qubits}")
    for i, bit in enumerate(reversed(input_state)):
        if bit == "1":
            qc.x(i)
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))

    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    ideal_dist = {k: v for k, v in probs.items() if v > 1e-9}

    return BenchmarkCircuit(
        name=f"QFT ({n_qubits}q)",
        circuit=qc,
        n_qubits=n_qubits,
        ideal_statevector=sv,
        ideal_distribution=ideal_dist,
        metadata={"description": f"{n_qubits}-qubit QFT on input |{input_state}>", "input_state": input_state},
    )


# --------------------------------------------------------------------------
# VQE ansatz for a transverse-field Ising Hamiltonian
# --------------------------------------------------------------------------
def _tfim_hamiltonian(n_qubits: int, j: float = 1.0, h: float = 1.0) -> SparsePauliOp:
    """H = -J * sum ZZ - h * sum X  (1D chain, open boundary)."""
    terms: List[Tuple[str, float]] = []
    for i in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[i] = "Z"
        label[i + 1] = "Z"
        terms.append(("".join(reversed(label)), -j))
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[i] = "X"
        terms.append(("".join(reversed(label)), -h))
    return SparsePauliOp.from_list(terms)


def vqe_ansatz(n_qubits: int = 4, layers: int = 2, params: Optional[np.ndarray] = None,
               seed: int = 7) -> BenchmarkCircuit:
    """Hardware-efficient VQE ansatz (RY + CX ladder) evaluated against a
    transverse-field Ising model. Parameters default to a fixed pseudo-random
    (but reproducible) point rather than a fully optimized VQE run, since the
    benchmark's purpose is to test mitigation, not the classical optimizer."""
    rng = np.random.default_rng(seed)
    n_params = n_qubits * (layers + 1)
    if params is None:
        params = rng.uniform(0, 2 * np.pi, size=n_params)

    qc = QuantumCircuit(n_qubits, name=f"vqe_ansatz_{n_qubits}q_{layers}L")
    idx = 0
    for q in range(n_qubits):
        qc.ry(params[idx], q)
        idx += 1
    for _ in range(layers):
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1

    hamiltonian = _tfim_hamiltonian(n_qubits)
    sv = Statevector.from_instruction(qc)
    ideal_expectation = float(np.real(sv.expectation_value(hamiltonian)))

    return BenchmarkCircuit(
        name=f"VQE Ansatz ({n_qubits}q, {layers} layers)",
        circuit=qc,
        n_qubits=n_qubits,
        ideal_statevector=sv,
        observable=hamiltonian,
        ideal_expectation=ideal_expectation,
        metadata={"description": "Hardware-efficient ansatz, TFIM Hamiltonian <H>", "params": params.tolist()},
    )


# --------------------------------------------------------------------------
# QAOA for MaxCut
# --------------------------------------------------------------------------
def _maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    n = graph.number_of_nodes()
    terms: List[Tuple[str, float]] = []
    for (i, j) in graph.edges():
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        terms.append(("".join(reversed(label)), 0.5))
    terms.append(("I" * n, -0.5 * graph.number_of_edges()))
    return SparsePauliOp.from_list(terms)


def qaoa_maxcut(n_qubits: int = 4, p: int = 1, graph: Optional[nx.Graph] = None,
                 seed: int = 11) -> BenchmarkCircuit:
    """Single/multi-layer QAOA circuit for MaxCut on a ring graph (default)."""
    if graph is None:
        graph = nx.cycle_graph(n_qubits)

    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, np.pi, size=p)
    betas = rng.uniform(0, np.pi / 2, size=p)

    qc = QuantumCircuit(n_qubits, name=f"qaoa_maxcut_{n_qubits}q_p{p}")
    qc.h(range(n_qubits))
    for layer in range(p):
        for (i, j) in graph.edges():
            qc.cx(i, j)
            qc.rz(2 * gammas[layer], j)
            qc.cx(i, j)
        for q in range(n_qubits):
            qc.rx(2 * betas[layer], q)

    cost_h = _maxcut_hamiltonian(graph)
    sv = Statevector.from_instruction(qc)
    ideal_expectation = float(np.real(sv.expectation_value(cost_h)))
    probs = sv.probabilities_dict()
    ideal_dist = {k: v for k, v in probs.items() if v > 1e-9}

    return BenchmarkCircuit(
        name=f"QAOA MaxCut ({n_qubits}q, p={p})",
        circuit=qc,
        n_qubits=n_qubits,
        ideal_statevector=sv,
        observable=cost_h,
        ideal_expectation=ideal_expectation,
        ideal_distribution=ideal_dist,
        metadata={"description": f"QAOA p={p} for MaxCut on a {n_qubits}-node ring graph",
                  "graph_edges": list(graph.edges())},
    )


BENCHMARK_REGISTRY: Dict[str, Callable[..., BenchmarkCircuit]] = {
    "bell": bell_state,
    "ghz": ghz_state,
    "qft": qft_circuit,
    "vqe": vqe_ansatz,
    "qaoa": qaoa_maxcut,
}
