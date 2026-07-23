"""
mem.py
======
Measurement (readout) Error Mitigation.

Implements the standard calibration-matrix approach (as used by e.g.
Qiskit's deprecated `CompleteMeasFitter` and IBM's `mthree`, in spirit):

  1. Prepare each computational basis state |i> and measure it `shots` times
     to build a confusion / calibration matrix A, where
         A[j, i] = P(measure j | prepared i)
  2. Given noisy counts vector c_noisy, solve  A @ c_ideal = c_noisy
     for c_ideal, i.e. c_ideal = A^{-1} @ c_noisy (least-squares if A is
     ill-conditioned), then clip to a valid probability simplex.

For n_qubits <= ~10 a full 2^n x 2^n calibration matrix is tractable; for
larger circuits this module falls back to a tensored single-qubit model
(assumes uncorrelated readout errors per qubit), which scales linearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel


@dataclass
class MEMResult:
    mitigated_counts: Dict[str, float]   # quasi-probabilities (can be slightly negative pre-clip; clipped+renormalized here)
    raw_counts: Dict[str, int]
    calibration_matrix: np.ndarray
    method: str


def _full_calibration_matrix(
    n_qubits: int,
    counts_executor: Callable[[QuantumCircuit], Dict[str, int]],
    shots: int,
) -> np.ndarray:
    dim = 2 ** n_qubits
    A = np.zeros((dim, dim))
    for i in range(dim):
        prep = format(i, f"0{n_qubits}b")
        qc = QuantumCircuit(n_qubits, n_qubits)
        for q, bit in enumerate(reversed(prep)):
            if bit == "1":
                qc.x(q)
        qc.measure(range(n_qubits), range(n_qubits))
        counts = counts_executor(qc)
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            j = int(bitstring.replace(" ", ""), 2)
            A[j, i] += cnt / total
    return A


def _per_qubit_calibration_matrices(
    n_qubits: int,
    counts_executor: Callable[[QuantumCircuit], Dict[str, int]],
    shots: int,
) -> List[np.ndarray]:
    mats = []
    for q in range(n_qubits):
        A = np.zeros((2, 2))
        for i in (0, 1):
            qc = QuantumCircuit(n_qubits, n_qubits)
            if i == 1:
                qc.x(q)
            qc.measure(range(n_qubits), range(n_qubits))
            counts = counts_executor(qc)
            total = sum(counts.values())
            for bitstring, cnt in counts.items():
                bit = bitstring.replace(" ", "")[::-1][q]  # index by qubit position
                j = int(bit)
                A[j, i] += cnt / total
        mats.append(A)
    return mats


def build_calibration_matrix(
    n_qubits: int,
    noise_model: Optional[NoiseModel],
    shots: int = 8192,
    method: str = "auto",
):
    """Build calibration data. Returns either a full 2^n x 2^n matrix
    ('full') or a list of n per-qubit 2x2 matrices ('tensored')."""
    from qem_framework.utils.executors import make_counts_executor

    counts_executor = make_counts_executor(noise_model, shots=shots)
    if method == "auto":
        method = "full" if n_qubits <= 6 else "tensored"

    if method == "full":
        return _full_calibration_matrix(n_qubits, counts_executor, shots), "full"
    else:
        return _per_qubit_calibration_matrices(n_qubits, counts_executor, shots), "tensored"


def _invert_full(A: np.ndarray, counts_vec: np.ndarray) -> np.ndarray:
    try:
        A_inv = np.linalg.pinv(A)
        corrected = A_inv @ counts_vec
    except np.linalg.LinAlgError:
        corrected = counts_vec.copy()
    corrected = np.clip(corrected, 0, None)
    if corrected.sum() > 0:
        corrected /= corrected.sum()
    return corrected


def _invert_tensored(mats: List[np.ndarray], counts: Dict[str, int], n_qubits: int) -> Dict[str, float]:
    total = sum(counts.values())
    dim = 2 ** n_qubits
    counts_vec = np.zeros(dim)
    for bitstring, cnt in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        counts_vec[idx] = cnt / total

    # Build full inverse as a tensor product of per-qubit inverses (uncorrelated model)
    inv_mats = [np.linalg.pinv(m) for m in mats]
    A_inv_full = inv_mats[0]
    for m in inv_mats[1:]:
        A_inv_full = np.kron(m, A_inv_full)
    corrected = A_inv_full @ counts_vec
    corrected = np.clip(corrected, 0, None)
    if corrected.sum() > 0:
        corrected /= corrected.sum()
    return {format(i, f"0{n_qubits}b"): float(p) for i, p in enumerate(corrected) if p > 1e-12}


def apply_mem(
    raw_counts: Dict[str, int],
    n_qubits: int,
    calibration_data,
    calibration_method: str,
) -> MEMResult:
    """Apply measurement error mitigation to a raw counts dict."""
    if calibration_method == "full":
        A = calibration_data
        total = sum(raw_counts.values())
        counts_vec = np.zeros(2 ** n_qubits)
        for bitstring, cnt in raw_counts.items():
            idx = int(bitstring.replace(" ", ""), 2)
            counts_vec[idx] = cnt / total
        corrected = _invert_full(A, counts_vec)
        mitigated = {format(i, f"0{n_qubits}b"): float(p) for i, p in enumerate(corrected) if p > 1e-12}
    else:
        mitigated = _invert_tensored(calibration_data, raw_counts, n_qubits)

    return MEMResult(
        mitigated_counts=mitigated,
        raw_counts=raw_counts,
        calibration_matrix=calibration_data if calibration_method == "full" else np.array(calibration_data),
        method=calibration_method,
    )
