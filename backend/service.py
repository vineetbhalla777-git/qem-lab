"""
service.py
==========
Thin service layer between the FastAPI app and qem_framework. Keeps circuit
sizes / shot counts small enough that every technique responds in roughly
1-3 seconds, suitable for a live "Run Experiment" button in the UI.

Benchmark circuit and qubit count are user-selectable (within per-benchmark
bounds) rather than hardcoded, while each technique still pairs itself with
the noise model it's actually designed for (e.g. MEM always tests against
readout-only noise, PEC against local depolarizing noise) -- letting a
technique run against a noise model it isn't built to address would produce
a misleading demo, not a more flexible one.
"""

import base64
import io
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from qiskit.quantum_info import SparsePauliOp

from qem_framework.noise.noise_models import (
    NoiseProfile, build_noise_model, depolarizing_only_model, readout_only_model,
)
from qem_framework.benchmarks.circuits import BENCHMARK_REGISTRY, BenchmarkCircuit
from qem_framework.utils.executors import make_counts_executor, make_expectation_executor, noiseless_expectation
from qem_framework.mitigation.zne import run_zne
from qem_framework.mitigation.mem import build_calibration_matrix, apply_mem
from qem_framework.mitigation.pec import run_pec
from qem_framework.mitigation.cdr import run_cdr
from qem_framework.mitigation.dynamical_decoupling import run_ddd
from qem_framework.mitigation.virtual_distillation import run_virtual_distillation
from qem_framework.analysis.metrics import (
    classical_fidelity, total_variation_distance, counts_to_probs, error_reduction_percent,
)

TECHNIQUES = {
    "zne": {
        "name": "Zero-Noise Extrapolation",
        "short": "ZNE",
        "description": (
            "Deliberately amplifies circuit noise at several scale factors, then "
            "extrapolates the measured expectation values back to the zero-noise limit."
        ),
        "benchmark": "ghz",
        "metric": "expectation",
        "overhead": "3x circuit evaluations (one per scale factor)",
        "noise_type": "composite (depolarizing + T1/T2 + readout)",
    },
    "mem": {
        "name": "Measurement Error Mitigation",
        "short": "MEM",
        "description": (
            "Builds a calibration matrix by preparing and measuring each computational "
            "basis state, then inverts it to correct readout errors in raw counts."
        ),
        "benchmark": "ghz",
        "metric": "distribution",
        "overhead": "2^n calibration circuits",
        "noise_type": "readout-only",
    },
    "pec": {
        "name": "Probabilistic Error Cancellation",
        "short": "PEC",
        "description": (
            "Represents ideal gates as a quasi-probability mixture of noisy, "
            "implementable operations; Monte-Carlo sampling cancels the noise in expectation."
        ),
        "benchmark": "bell",
        "metric": "expectation",
        "overhead": "~40-200x sampled circuits",
        "noise_type": "local depolarizing-only",
    },
    "cdr": {
        "name": "Clifford Data Regression",
        "short": "CDR",
        "description": (
            "Learns a noisy-to-ideal correction by training a regression model on "
            "near-Clifford circuits that are cheap to simulate exactly."
        ),
        "benchmark": "vqe",
        "metric": "expectation",
        "overhead": "~6-10 training circuits",
        "noise_type": "composite (depolarizing + T1/T2 + readout)",
    },
    "ddd": {
        "name": "Dynamical Decoupling",
        "short": "DDD",
        "description": (
            "Inserts pulse sequences (e.g. XYXY) into idle windows to average out "
            "low-frequency dephasing noise, at no extra circuit-evaluation cost."
        ),
        "benchmark": "ghz",
        "metric": "expectation",
        "overhead": "1x (no extra circuits)",
        "noise_type": "composite (depolarizing + T1/T2 + readout)",
    },
    "vd": {
        "name": "Virtual Distillation",
        "short": "VD",
        "description": (
            "Entangles two copies of the state via an ancilla-controlled SWAP test, "
            "virtually projecting onto the dominant (least mixed) eigenvector of rho."
        ),
        "benchmark": "vqe",
        "metric": "expectation",
        "overhead": "2n+1 qubits, n controlled-SWAP gates",
        "noise_type": "composite (depolarizing + T1/T2 + readout)",
    },
}

# Per-benchmark qubit-count bounds. Bell state is definitionally a 2-qubit
# state (min == max == 2); the others can scale, capped to keep live
# response times low (VD in particular simulates 2n+1 qubits).
BENCHMARKS = {
    "bell": {"name": "Bell State", "n_qubits": 2, "min_qubits": 2, "max_qubits": 2,
              "description": "Maximally entangled 2-qubit pair."},
    "ghz": {"name": "GHZ State", "n_qubits": 3, "min_qubits": 2, "max_qubits": 6,
             "description": "n-qubit Greenberger-Horne-Zeilinger state."},
    "qft": {"name": "Quantum Fourier Transform", "n_qubits": 3, "min_qubits": 2, "max_qubits": 6,
             "description": "QFT applied to a basis state."},
    "vqe": {"name": "VQE Ansatz", "n_qubits": 2, "min_qubits": 2, "max_qubits": 5,
             "description": "Hardware-efficient ansatz, TFIM Hamiltonian."},
    "qaoa": {"name": "QAOA MaxCut", "n_qubits": 4, "min_qubits": 3, "max_qubits": 6,
              "description": "QAOA p=1 for MaxCut on a ring graph."},
}

# Benchmarks made entirely of Clifford gates (H, CX only) -- CDR's near-Clifford
# training circuits degenerate to the original circuit itself in this case,
# so its correction is less meaningful than on genuinely non-Clifford circuits.
_FULLY_CLIFFORD_BENCHMARKS = {"bell", "ghz"}

# Known technique x benchmark incompatibilities, found by actually running every
# combination (see dev notes) rather than assumed. Both are library limitations,
# not modeling choices: Mitiq's PEC representation step only supports 1- and
# 2-qubit gates and can't decompose the QFT library instruction's controlled-phase
# gates automatically; Mitiq's CDR near-Clifford replacement crashes on the SWAP
# gates this project's QFT circuit uses (do_swaps=True in qiskit.circuit.library.QFT).
_INCOMPATIBLE = {
    ("pec", "qft"): "Mitiq's PEC representation step only supports 1- and 2-qubit gates; this project's QFT circuit isn't pre-decomposed to that form.",
    ("cdr", "qft"): "Mitiq's CDR training-circuit generator doesn't support the SWAP gates this project's QFT circuit uses internally.",
}


def _build_benchmark(benchmark: str, n_qubits: Optional[int]) -> BenchmarkCircuit:
    if benchmark not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    bounds = BENCHMARKS[benchmark]
    n = n_qubits if n_qubits is not None else bounds["n_qubits"]
    n = max(bounds["min_qubits"], min(bounds["max_qubits"], n))

    if benchmark == "bell":
        return BENCHMARK_REGISTRY["bell"]()
    if benchmark == "qaoa":
        return BENCHMARK_REGISTRY["qaoa"](n, p=1)
    if benchmark == "vqe":
        return BENCHMARK_REGISTRY["vqe"](n, layers=1)
    return BENCHMARK_REGISTRY[benchmark](n)


def _default_observable(bc: BenchmarkCircuit, benchmark: str) -> SparsePauliOp:
    """Uses the benchmark's own Hamiltonian if it has one (VQE, QAOA);
    otherwise picks a per-benchmark-family fallback observable that's
    verified to give a robustly nonzero ideal value across qubit counts.

    A single one-size-fits-all fallback doesn't work here: the full n-qubit
    Z parity string gives ideal = 0 for odd-sized GHZ states (verified:
    n=3 -> 0.0, n=5 -> 0.0), and *every* Z-type observable gives ideal = 0
    for this project's QFT benchmark (its output is a uniform-magnitude
    superposition, so Z-basis correlations cancel by symmetry) -- both
    would reintroduce the "percent-error blowup near a near-zero ideal
    value" problem documented elsewhere in this project (see REPORT.md,
    Virtual Distillation section). Verified alternatives used below:
    Z0*Z1 gives ideal = +1 for a GHZ state of any size (n=2..6 checked);
    X on the highest-index qubit gives ideal = -1 for this project's QFT
    benchmark at any size (n=2..6 checked, thanks to its phase structure).
    """
    if bc.observable is not None:
        return bc.observable

    n = bc.n_qubits
    if benchmark == "qft":
        label = ["I"] * n
        label[n - 1] = "X"
        return SparsePauliOp.from_list([("".join(reversed(label)), 1.0)])

    # bell / ghz (and any future Z-correlation-based benchmark)
    label = ["I"] * n
    label[0] = "Z"
    label[min(1, n - 1)] = "Z"
    return SparsePauliOp.from_list([("".join(reversed(label)), 1.0)])


def _default_distribution(bc: BenchmarkCircuit) -> dict:
    if bc.ideal_distribution is not None:
        return bc.ideal_distribution
    probs = bc.ideal_statevector.probabilities_dict()
    return {k: v for k, v in probs.items() if v > 1e-9}


def _notes_for(technique: str, benchmark: str) -> list:
    notes = []
    if technique == "cdr" and benchmark in _FULLY_CLIFFORD_BENCHMARKS:
        notes.append(
            "This benchmark's circuit is made entirely of Clifford gates (H, CX). "
            "CDR's near-Clifford training circuits degenerate to the circuit itself in "
            "this case, so its correction is less meaningful here than on a genuinely "
            "non-Clifford circuit like VQE or QFT."
        )
    return notes


def _gate_counts(circuit) -> dict:
    """Elementary gate counts, fully decomposing any nested library
    instructions (e.g. Qiskit's QFT) so counts reflect the actual
    operations noise gets applied to, not one opaque named block."""
    decomposed = circuit
    for _ in range(6):
        counts = dict(decomposed.count_ops())
        # A gate name in all-caps that isn't a standard 1-2 letter gate is
        # typically a composite library instruction (e.g. "QFT") -- keep
        # decomposing until none remain or we hit the rep limit.
        if not any(name.isupper() and len(name) > 2 for name in counts):
            break
        decomposed = decomposed.decompose()
    return dict(decomposed.count_ops())


def _circuit_diagram_base64(circuit) -> Optional[str]:
    """Renders the circuit as a dark-themed PNG (matching the site's
    palette) and returns it as a base64 data URI, or None if rendering
    fails for any reason (drawing is a nice-to-have, not load-bearing)."""
    try:
        import matplotlib.pyplot as plt
        fig = circuit.draw(output="mpl", style="iqp-dark", fold=-1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _run_zne(noise_strength, shots, benchmark, n_qubits):
    bc = _build_benchmark(benchmark, n_qubits)
    obs = _default_observable(bc, benchmark)
    ideal = noiseless_expectation(bc.circuit, obs)
    nm = build_noise_model(NoiseProfile(strength=noise_strength, n_qubits=bc.n_qubits))
    executor = make_expectation_executor(nm, obs, shots=shots)
    res = run_zne(bc.circuit, executor, scale_factors=[1.0, 2.0, 3.0], extrapolation="richardson")
    return {
        "benchmark": f"{BENCHMARKS[benchmark]['name']} ({bc.n_qubits}q)",
        "observable": f"<{obs.paulis[0].to_label()}>",
        "ideal": ideal, "raw": res.raw_value, "mitigated": res.mitigated_value,
        "detail": {"scale_factors": res.scale_factors, "scaled_values": res.scaled_values},
    }


def _run_mem(noise_strength, shots, benchmark, n_qubits):
    bc = _build_benchmark(benchmark, n_qubits)
    ideal_dist = _default_distribution(bc)
    nm = readout_only_model(noise_strength, bc.n_qubits)
    counts_exec = make_counts_executor(nm, shots=shots)
    raw_counts = counts_exec(bc.measured_circuit())
    raw_probs = counts_to_probs(raw_counts)
    cal, method = build_calibration_matrix(bc.n_qubits, nm, shots=shots)
    mem_res = apply_mem(raw_counts, bc.n_qubits, cal, method)
    f_raw = classical_fidelity(ideal_dist, raw_probs)
    f_mit = classical_fidelity(ideal_dist, mem_res.mitigated_counts)
    return {
        "benchmark": f"{BENCHMARKS[benchmark]['name']} ({bc.n_qubits}q)",
        "observable": "bitstring distribution",
        "ideal": 1.0, "raw": f_raw, "mitigated": f_mit,
        "detail": {
            "ideal_distribution": ideal_dist, "raw_distribution": raw_probs,
            "mitigated_distribution": mem_res.mitigated_counts,
            "metric_name": "classical fidelity",
        },
    }


def _run_pec(noise_strength, shots, benchmark, n_qubits):
    bc = _build_benchmark(benchmark, n_qubits)
    obs = _default_observable(bc, benchmark)
    ideal = noiseless_expectation(bc.circuit, obs)
    nm = depolarizing_only_model(noise_strength, bc.n_qubits)
    executor = make_expectation_executor(nm, obs, shots=shots)
    assumed_strength = 0.01 + 0.15 * noise_strength
    res = run_pec(bc.circuit, executor, assumed_noise_strength=assumed_strength, num_samples=40)
    return {
        "benchmark": f"{BENCHMARKS[benchmark]['name']} ({bc.n_qubits}q)",
        "observable": f"<{obs.paulis[0].to_label()}>",
        "ideal": ideal, "raw": res.raw_value, "mitigated": res.mitigated_value,
        "detail": {"assumed_noise_strength": assumed_strength, "num_samples": res.num_samples},
    }


def _run_cdr(noise_strength, shots, benchmark, n_qubits):
    bc = _build_benchmark(benchmark, n_qubits)
    obs = _default_observable(bc, benchmark)
    ideal = bc.ideal_expectation if bc.ideal_expectation is not None else noiseless_expectation(bc.circuit, obs)
    nm = build_noise_model(NoiseProfile(strength=noise_strength, n_qubits=bc.n_qubits))
    executor = make_expectation_executor(nm, obs, shots=shots)
    res = run_cdr(bc.circuit, executor, obs, num_training_circuits=6, fraction_non_clifford=0.2)
    return {
        "benchmark": f"{BENCHMARKS[benchmark]['name']} ({bc.n_qubits}q)",
        "observable": f"<{obs.paulis[0].to_label()}>" if bc.observable is None else "<H>",
        "ideal": ideal, "raw": res.raw_value, "mitigated": res.mitigated_value,
        "detail": {"num_training_circuits": res.num_training_circuits},
    }


def _run_ddd(noise_strength, shots, benchmark, n_qubits):
    bc = _build_benchmark(benchmark, n_qubits)
    obs = _default_observable(bc, benchmark)
    ideal = noiseless_expectation(bc.circuit, obs)
    nm = build_noise_model(NoiseProfile(strength=noise_strength, n_qubits=bc.n_qubits))
    executor = make_expectation_executor(nm, obs, shots=shots)
    res = run_ddd(bc.circuit, executor, rule="xyxy")
    return {
        "benchmark": f"{BENCHMARKS[benchmark]['name']} ({bc.n_qubits}q)",
        "observable": f"<{obs.paulis[0].to_label()}>",
        "ideal": ideal, "raw": res.raw_value, "mitigated": res.mitigated_value,
        "detail": {"rule": res.rule_name},
    }


def _run_vd(noise_strength, shots, benchmark, n_qubits):
    bc = _build_benchmark(benchmark, n_qubits)
    obs = _default_observable(bc, benchmark)
    ideal = bc.ideal_expectation if bc.ideal_expectation is not None else noiseless_expectation(bc.circuit, obs)
    n = bc.n_qubits
    nm_vd = build_noise_model(NoiseProfile(strength=noise_strength, n_qubits=2 * n + 1))
    nm_raw = build_noise_model(NoiseProfile(strength=noise_strength, n_qubits=n))
    counts_exec = make_counts_executor(nm_vd, shots=shots)
    exp_exec = make_expectation_executor(nm_raw, obs, shots=shots)
    res = run_virtual_distillation(bc.circuit, obs, counts_exec, raw_executor=exp_exec)
    return {
        "benchmark": f"{BENCHMARKS[benchmark]['name']} ({bc.n_qubits}q)",
        "observable": f"<{obs.paulis[0].to_label()}>" if bc.observable is None else "<H>",
        "ideal": ideal, "raw": res.raw_value, "mitigated": res.mitigated_value,
        "detail": {"denominator_tr_rho2": res.denominator_estimate},
    }


_RUNNERS = {
    "zne": _run_zne, "mem": _run_mem, "pec": _run_pec,
    "cdr": _run_cdr, "ddd": _run_ddd, "vd": _run_vd,
}

# Roughly caps total simulated qubits (accounting for VD's 2n+1 blowup) so a
# live request never gets uncomfortably slow.
_MAX_EFFECTIVE_QUBITS = 13


def get_techniques_with_compatibility() -> dict:
    """Returns TECHNIQUES enriched with each technique's list of benchmarks
    known NOT to work (see _INCOMPATIBLE), so the frontend can disable those
    combinations in the UI rather than let a user hit a 400 after the fact."""
    enriched = {}
    for key, meta in TECHNIQUES.items():
        incompatible = [b for (t, b) in _INCOMPATIBLE if t == key]
        enriched[key] = {**meta, "incompatible_benchmarks": incompatible}
    return enriched


def run_technique(
    technique: str,
    noise_strength: float,
    shots: int = 1500,
    benchmark: Optional[str] = None,
    n_qubits: Optional[int] = None,
) -> dict:
    if technique not in _RUNNERS:
        raise ValueError(f"Unknown technique: {technique}")
    noise_strength = max(0.0, min(1.0, noise_strength))
    shots = max(200, min(4000, shots))

    benchmark = benchmark or TECHNIQUES[technique]["benchmark"]
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(BENCHMARKS)}")
    if (technique, benchmark) in _INCOMPATIBLE:
        raise ValueError(
            f"{TECHNIQUES[technique]['name']} doesn't currently work with the "
            f"{BENCHMARKS[benchmark]['name']} benchmark: {_INCOMPATIBLE[(technique, benchmark)]}"
        )

    bounds = BENCHMARKS[benchmark]
    requested_n = n_qubits if n_qubits is not None else bounds["n_qubits"]
    clamped_n = max(bounds["min_qubits"], min(bounds["max_qubits"], requested_n))

    effective_n = 2 * clamped_n + 1 if technique == "vd" else clamped_n
    if effective_n > _MAX_EFFECTIVE_QUBITS:
        raise ValueError(
            f"{clamped_n} qubits is too many for {TECHNIQUES[technique]['name']} to run live "
            f"(simulates {effective_n} effective qubits, max {_MAX_EFFECTIVE_QUBITS}). Try fewer qubits."
        )

    t0 = time.time()
    result = _RUNNERS[technique](noise_strength, shots, benchmark, clamped_n)
    wall_time = time.time() - t0

    # Rebuild the same benchmark circuit (cheap, deterministic) purely for
    # display purposes -- the diagram/gate-counts describe the *benchmark*
    # circuit itself, not any technique-specific wrapping (e.g. VD's ancilla
    # circuit, ZNE's folded copies), which is what "benchmark circuit" means
    # to someone reading the UI.
    display_bc = _build_benchmark(benchmark, clamped_n)
    gate_counts = _gate_counts(display_bc.circuit)
    circuit_diagram = _circuit_diagram_base64(display_bc.circuit)

    ideal, raw, mitigated = result["ideal"], result["raw"], result["mitigated"]
    err_reduction = error_reduction_percent(ideal, raw, mitigated)

    return {
        "technique": technique,
        "technique_name": TECHNIQUES[technique]["name"],
        "noise_strength": noise_strength,
        "shots": shots,
        "wall_time_s": round(wall_time, 3),
        "benchmark": result["benchmark"],
        "benchmark_key": benchmark,
        "n_qubits": clamped_n,
        "observable": result["observable"],
        "ideal": ideal,
        "raw": raw,
        "mitigated": mitigated,
        "error_reduction_pct": err_reduction,
        "raw_error": abs(raw - ideal),
        "mitigated_error": abs(mitigated - ideal),
        "detail": result.get("detail", {}),
        "notes": _notes_for(technique, benchmark),
        "gate_counts": gate_counts,
        "circuit_depth": display_bc.circuit.depth(),
        "circuit_diagram": circuit_diagram,
    }
