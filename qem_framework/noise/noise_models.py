"""
noise_models.py
================
Noise model generation for NISQ device simulation.

Implements the noise mechanisms called out in Phase I of the project scope:
    - Depolarizing noise
    - Amplitude damping noise
    - Phase damping noise
    - Readout (measurement) errors
    - Coherent gate errors (systematic over/under rotation)

Each function returns a Qiskit Aer `NoiseModel` that can be handed straight
to an `AerSimulator`. A `NoiseProfile` dataclass bundles the parameters so
experiments can sweep over noise "strength" with a single scalar knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
    thermal_relaxation_error,
)
from qiskit.circuit.library import RZGate, RXGate, RYGate


@dataclass
class NoiseProfile:
    """A single scalar 'noise strength' fans out into physically-motivated
    per-gate and per-qubit error parameters. This lets benchmarking code
    sweep one number (`strength`) and get a self-consistent noise model.

    Parameters
    ----------
    strength : float
        Dimensionless knob in [0, 1]. 0 = noiseless, 1 = heavily noisy.
    n_qubits : int
        Number of qubits the model needs readout/relaxation errors for.
    include_readout : bool
        Whether to attach measurement (readout) error.
    include_coherent : bool
        Whether to attach a coherent (systematic rotation) gate error on
        top of the stochastic depolarizing/amplitude/phase channels.
    seed : Optional[int]
        RNG seed for reproducibility of coherent-error angles.
    """

    strength: float = 0.05
    n_qubits: int = 5
    include_readout: bool = True
    include_coherent: bool = False
    seed: Optional[int] = 42

    # Derived physical parameters (populated in __post_init__)
    p1_depol: float = field(init=False)
    p2_depol: float = field(init=False)
    t1_us: float = field(init=False)
    t2_us: float = field(init=False)
    gate_time_1q_ns: float = field(init=False)
    gate_time_2q_ns: float = field(init=False)
    readout_p01: float = field(init=False)
    readout_p10: float = field(init=False)
    coherent_angle: float = field(init=False)

    def __post_init__(self):
        s = float(np.clip(self.strength, 0.0, 1.0))
        # 1- and 2-qubit depolarizing probabilities scale with strength;
        # 2-qubit gates are noisier, matching typical hardware ratios (~10x).
        self.p1_depol = 0.001 + 0.02 * s
        self.p2_depol = 0.01 + 0.15 * s

        # T1/T2 shrink as strength grows (worse coherence times).
        self.t1_us = 100.0 * (1.0 - 0.8 * s) + 5.0
        self.t2_us = 80.0 * (1.0 - 0.8 * s) + 4.0
        self.gate_time_1q_ns = 35.0
        self.gate_time_2q_ns = 300.0

        # Readout error grows with strength (asymmetric, as on real hardware:
        # |1> -> |0> misassignment is typically worse than |0> -> |1>).
        self.readout_p01 = 0.005 + 0.05 * s
        self.readout_p10 = 0.01 + 0.09 * s

        # Coherent (systematic) over-rotation angle in radians.
        self.coherent_angle = 0.02 + 0.15 * s


def _relaxation_error_1q(profile: NoiseProfile, gate_time_ns: float):
    t1 = profile.t1_us * 1000.0  # ns
    t2 = min(profile.t2_us * 1000.0, 2 * t1 * 0.999)
    return thermal_relaxation_error(t1, t2, gate_time_ns)


def build_noise_model(profile: NoiseProfile) -> NoiseModel:
    """Compose a full Aer NoiseModel from a NoiseProfile.

    Combines (in order): depolarizing error -> thermal relaxation
    (amplitude+phase damping via T1/T2) -> optional coherent rotation error,
    then attaches readout error per qubit.
    """
    nm = NoiseModel()

    # --- 1-qubit gate errors: depolarizing + thermal relaxation composed ---
    depol_1q = depolarizing_error(profile.p1_depol, 1)
    relax_1q = _relaxation_error_1q(profile, profile.gate_time_1q_ns)
    error_1q = depol_1q.compose(relax_1q)

    one_qubit_gates = ["id", "x", "y", "z", "h", "s", "sdg", "sx", "rx", "ry", "rz"]
    nm.add_all_qubit_quantum_error(error_1q, one_qubit_gates)

    # --- 2-qubit gate errors: depolarizing (dominant on real hardware) ---
    depol_2q = depolarizing_error(profile.p2_depol, 2)
    nm.add_all_qubit_quantum_error(depol_2q, ["cx", "cz", "swap"])

    # --- Readout error ---
    if profile.include_readout:
        p01, p10 = profile.readout_p01, profile.readout_p10
        ro_error = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        for q in range(profile.n_qubits):
            nm.add_readout_error(ro_error, [q])

    return nm


def depolarizing_only_model(strength: float, n_qubits: int = 5) -> NoiseModel:
    """Pure depolarizing-channel noise model (no readout/relaxation) —
    useful as a clean baseline when isolating one noise mechanism."""
    profile = NoiseProfile(strength=strength, n_qubits=n_qubits, include_readout=False)
    nm = NoiseModel()
    depol_1q = depolarizing_error(profile.p1_depol, 1)
    depol_2q = depolarizing_error(profile.p2_depol, 2)
    nm.add_all_qubit_quantum_error(depol_1q, ["id", "x", "y", "z", "h", "s", "sdg", "sx", "rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(depol_2q, ["cx", "cz", "swap"])
    return nm


def amplitude_damping_model(strength: float, n_qubits: int = 5) -> NoiseModel:
    """Pure amplitude-damping (T1 relaxation / energy loss) noise model."""
    gamma = 0.01 + 0.3 * float(np.clip(strength, 0, 1))
    nm = NoiseModel()
    error_1q = amplitude_damping_error(gamma)
    error_2q = error_1q.tensor(error_1q)
    nm.add_all_qubit_quantum_error(error_1q, ["id", "x", "y", "z", "h", "s", "sdg", "sx", "rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])
    return nm


def phase_damping_model(strength: float, n_qubits: int = 5) -> NoiseModel:
    """Pure phase-damping (T2 dephasing) noise model."""
    lam = 0.01 + 0.3 * float(np.clip(strength, 0, 1))
    nm = NoiseModel()
    error_1q = phase_damping_error(lam)
    error_2q = error_1q.tensor(error_1q)
    nm.add_all_qubit_quantum_error(error_1q, ["id", "x", "y", "z", "h", "s", "sdg", "sx", "rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])
    return nm


def readout_only_model(strength: float, n_qubits: int) -> NoiseModel:
    """Pure measurement (readout) error model — no gate errors at all.
    Used to isolate the effect of Measurement Error Mitigation (MEM)."""
    s = float(np.clip(strength, 0, 1))
    p01 = 0.005 + 0.05 * s
    p10 = 0.01 + 0.09 * s
    nm = NoiseModel()
    ro_error = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    for q in range(n_qubits):
        nm.add_readout_error(ro_error, [q])
    return nm


NOISE_MECHANISMS = {
    "depolarizing": depolarizing_only_model,
    "amplitude_damping": amplitude_damping_model,
    "phase_damping": phase_damping_model,
    "readout": readout_only_model,
    "composite": lambda strength, n_qubits: build_noise_model(
        NoiseProfile(strength=strength, n_qubits=n_qubits)
    ),
}
