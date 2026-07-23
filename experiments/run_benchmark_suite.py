"""
run_benchmark_suite.py
=======================
Automated benchmarking script (Phase III + Phase IV of the project scope).

Runs every implemented QEM technique against its natural benchmark
circuit(s), across a sweep of noise strengths, and produces:
    - results/data/benchmark_results.csv   (raw numeric results)
    - results/figures/*.png                (accuracy-vs-noise, bar charts,
                                              distribution comparisons,
                                              overhead-vs-accuracy, scalability)
    - A console summary table

Run with:  python3 experiments/run_benchmark_suite.py
(from the qem_project/ root, or anywhere with qem_project/ on PYTHONPATH)
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qem_framework.noise.noise_models import (
    NoiseProfile, build_noise_model, depolarizing_only_model, readout_only_model,
)
from qem_framework.benchmarks.circuits import bell_state, ghz_state, vqe_ansatz, qaoa_maxcut
from qem_framework.utils.executors import (
    make_counts_executor, make_expectation_executor, noiseless_expectation,
)
from qem_framework.mitigation.zne import run_zne
from qem_framework.mitigation.mem import build_calibration_matrix, apply_mem
from qem_framework.mitigation.pec import run_pec
from qem_framework.mitigation.cdr import run_cdr
from qem_framework.mitigation.dynamical_decoupling import run_ddd
from qem_framework.mitigation.virtual_distillation import run_virtual_distillation
from qem_framework.analysis.metrics import (
    classical_fidelity, total_variation_distance, counts_to_probs,
    error_reduction_percent, OverheadRecord,
)
from qem_framework.analysis import visualization as viz

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SHOTS = 3000
NOISE_STRENGTHS = [0.05, 0.15, 0.25, 0.35]

all_rows = []  # accumulate CSV rows across all experiments


def log_row(experiment, technique, benchmark, noise_strength, ideal, raw, mitigated,
            error_reduction, overhead_factor, wall_time, extra=""):
    row = dict(
        experiment=experiment, technique=technique, benchmark=benchmark,
        noise_strength=noise_strength, ideal=ideal, raw=raw, mitigated=mitigated,
        error_reduction_pct=error_reduction, overhead_factor=overhead_factor,
        wall_time_s=wall_time, extra=extra,
    )
    all_rows.append(row)
    return row


# ---------------------------------------------------------------------------
# Experiment A: Measurement Error Mitigation (MEM) on Bell & GHZ states
# ---------------------------------------------------------------------------
def experiment_mem():
    print("\n=== Experiment A: Measurement Error Mitigation (MEM) ===")
    fidelities_raw, fidelities_mit, strengths_used = [], [], []
    last_ideal_dist = last_raw_dist = last_mit_dist = None

    bc = ghz_state(4)
    ideal_dist = bc.ideal_distribution

    for s in NOISE_STRENGTHS:
        t0 = time.time()
        nm = readout_only_model(s, bc.n_qubits)
        counts_exec = make_counts_executor(nm, shots=SHOTS)
        raw_counts = counts_exec(bc.measured_circuit())
        raw_probs = counts_to_probs(raw_counts)

        cal, method = build_calibration_matrix(bc.n_qubits, nm, shots=SHOTS)
        mem_res = apply_mem(raw_counts, bc.n_qubits, cal, method)
        mit_probs = mem_res.mitigated_counts

        f_raw = classical_fidelity(ideal_dist, raw_probs)
        f_mit = classical_fidelity(ideal_dist, mit_probs)
        tvd_raw = total_variation_distance(ideal_dist, raw_probs)
        tvd_mit = total_variation_distance(ideal_dist, mit_probs)
        err_reduction = error_reduction_percent(0.0, tvd_raw, tvd_mit)  # TVD: 0 is ideal
        wall_time = time.time() - t0

        # calibration circuits = 2^n, plus 1 for the actual measurement
        overhead = (2 ** bc.n_qubits + 1)
        log_row("A_MEM", "MEM", bc.name, s, 1.0, f_raw, f_mit, err_reduction, overhead, wall_time,
                 extra=f"fidelity based; tvd_raw={tvd_raw:.4f}, tvd_mit={tvd_mit:.4f}")

        fidelities_raw.append(f_raw)
        fidelities_mit.append(f_mit)
        strengths_used.append(s)
        last_ideal_dist, last_raw_dist, last_mit_dist = ideal_dist, raw_probs, mit_probs
        print(f"  strength={s:.2f}  fidelity raw={f_raw:.4f}  mitigated={f_mit:.4f}  "
              f"TVD raw={tvd_raw:.4f} mitigated={tvd_mit:.4f}")

    viz.plot_accuracy_vs_noise(
        strengths_used, [1 - f for f in fidelities_raw], [1 - f for f in fidelities_mit],
        title="MEM: Infidelity (1-F) vs Readout Noise Strength (GHZ-4)",
        save_path=os.path.join(FIG_DIR, "A_mem_fidelity_vs_noise.png"),
        ylabel="1 - classical fidelity",
    )
    viz.plot_distribution_comparison(
        last_ideal_dist, last_raw_dist, last_mit_dist,
        title=f"GHZ-4 Distribution: Ideal vs Raw vs MEM-Mitigated (strength={NOISE_STRENGTHS[-1]})",
        save_path=os.path.join(FIG_DIR, "A_mem_distribution_comparison.png"),
    )


# ---------------------------------------------------------------------------
# Experiment B: Zero Noise Extrapolation (ZNE) on GHZ state (Z0 Z1 Z2 observable)
# ---------------------------------------------------------------------------
def experiment_zne():
    print("\n=== Experiment B: Zero Noise Extrapolation (ZNE) ===")
    bc = ghz_state(3)
    obs = SparsePauliOp.from_list([("ZZI", 1.0)])
    ideal = noiseless_expectation(bc.circuit, obs)

    raw_errors, mit_errors, strengths_used, reductions = [], [], [], []
    for s in NOISE_STRENGTHS:
        t0 = time.time()
        nm = build_noise_model(NoiseProfile(strength=s, n_qubits=bc.n_qubits))
        executor = make_expectation_executor(nm, obs, shots=SHOTS)
        res = run_zne(bc.circuit, executor, scale_factors=[1.0, 2.0, 3.0], extrapolation="richardson")
        wall_time = time.time() - t0

        err_reduction = error_reduction_percent(ideal, res.raw_value, res.mitigated_value)
        log_row("B_ZNE", "ZNE (Richardson)", bc.name, s, ideal, res.raw_value, res.mitigated_value,
                 err_reduction, overhead_factor=len(res.scale_factors), wall_time=wall_time)

        raw_errors.append(abs(res.raw_value - ideal))
        mit_errors.append(abs(res.mitigated_value - ideal))
        strengths_used.append(s)
        reductions.append(err_reduction)
        print(f"  strength={s:.2f}  raw={res.raw_value:.4f}  mitigated={res.mitigated_value:.4f}  "
              f"ideal={ideal:.4f}  error_reduction={err_reduction:.1f}%")

    viz.plot_accuracy_vs_noise(
        strengths_used, raw_errors, mit_errors,
        title="ZNE: |<ZZI> error| vs Noise Strength (GHZ-3)",
        save_path=os.path.join(FIG_DIR, "B_zne_error_vs_noise.png"),
    )


# ---------------------------------------------------------------------------
# Experiment C: Dynamical Decoupling (DDD) on GHZ state
# ---------------------------------------------------------------------------
def experiment_ddd():
    print("\n=== Experiment C: Dynamical Decoupling (DDD) ===")
    bc = ghz_state(3)
    obs = SparsePauliOp.from_list([("ZZI", 1.0)])
    ideal = noiseless_expectation(bc.circuit, obs)

    raw_errors, mit_errors, strengths_used, reductions = [], [], [], []
    for s in NOISE_STRENGTHS:
        t0 = time.time()
        nm = build_noise_model(NoiseProfile(strength=s, n_qubits=bc.n_qubits))
        executor = make_expectation_executor(nm, obs, shots=SHOTS)
        res = run_ddd(bc.circuit, executor, rule="xyxy")
        wall_time = time.time() - t0

        err_reduction = error_reduction_percent(ideal, res.raw_value, res.mitigated_value)
        log_row("C_DDD", "Dynamical Decoupling (XYXY)", bc.name, s, ideal, res.raw_value,
                 res.mitigated_value, err_reduction, overhead_factor=1.0, wall_time=wall_time)

        raw_errors.append(abs(res.raw_value - ideal))
        mit_errors.append(abs(res.mitigated_value - ideal))
        strengths_used.append(s)
        reductions.append(err_reduction)
        print(f"  strength={s:.2f}  raw={res.raw_value:.4f}  mitigated={res.mitigated_value:.4f}  "
              f"ideal={ideal:.4f}  error_reduction={err_reduction:.1f}%")

    viz.plot_accuracy_vs_noise(
        strengths_used, raw_errors, mit_errors,
        title="Dynamical Decoupling: |<ZZI> error| vs Noise Strength (GHZ-3)",
        save_path=os.path.join(FIG_DIR, "C_ddd_error_vs_noise.png"),
    )


# ---------------------------------------------------------------------------
# Experiment D: Clifford Data Regression (CDR) on VQE ansatz
# ---------------------------------------------------------------------------
def experiment_cdr():
    print("\n=== Experiment D: Clifford Data Regression (CDR) ===")
    bc = vqe_ansatz(3, layers=2)
    ideal = bc.ideal_expectation

    raw_errors, mit_errors, strengths_used, reductions = [], [], [], []
    for s in NOISE_STRENGTHS:
        t0 = time.time()
        nm = build_noise_model(NoiseProfile(strength=s, n_qubits=bc.n_qubits))
        executor = make_expectation_executor(nm, bc.observable, shots=SHOTS)
        res = run_cdr(bc.circuit, executor, bc.observable,
                       num_training_circuits=8, fraction_non_clifford=0.2)
        wall_time = time.time() - t0

        err_reduction = error_reduction_percent(ideal, res.raw_value, res.mitigated_value)
        log_row("D_CDR", "Clifford Data Regression", bc.name, s, ideal, res.raw_value,
                 res.mitigated_value, err_reduction, overhead_factor=res.num_training_circuits,
                 wall_time=wall_time)

        raw_errors.append(abs(res.raw_value - ideal))
        mit_errors.append(abs(res.mitigated_value - ideal))
        strengths_used.append(s)
        reductions.append(err_reduction)
        print(f"  strength={s:.2f}  raw={res.raw_value:.4f}  mitigated={res.mitigated_value:.4f}  "
              f"ideal={ideal:.4f}  error_reduction={err_reduction:.1f}%")

    viz.plot_accuracy_vs_noise(
        strengths_used, raw_errors, mit_errors,
        title="CDR: |<H_TFIM> error| vs Noise Strength (VQE ansatz, 3q)",
        save_path=os.path.join(FIG_DIR, "D_cdr_error_vs_noise.png"),
    )


# ---------------------------------------------------------------------------
# Experiment E: Probabilistic Error Cancellation (PEC) on Bell state
# ---------------------------------------------------------------------------
def experiment_pec():
    print("\n=== Experiment E: Probabilistic Error Cancellation (PEC) ===")
    bc = bell_state()
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    ideal = noiseless_expectation(bc.circuit, obs)

    raw_errors, mit_errors, strengths_used, reductions = [], [], [], []
    for s in NOISE_STRENGTHS:
        t0 = time.time()
        nm = depolarizing_only_model(s, bc.n_qubits)
        executor = make_expectation_executor(nm, obs, shots=SHOTS)
        assumed_strength = 0.01 + 0.15 * s  # matches NoiseProfile.p2_depol scaling
        res = run_pec(bc.circuit, executor, assumed_noise_strength=assumed_strength, num_samples=80)
        wall_time = time.time() - t0

        err_reduction = error_reduction_percent(ideal, res.raw_value, res.mitigated_value)
        log_row("E_PEC", "PEC", bc.name, s, ideal, res.raw_value, res.mitigated_value,
                 err_reduction, overhead_factor=res.num_samples, wall_time=wall_time,
                 extra=f"assumed_noise={assumed_strength:.4f}")

        raw_errors.append(abs(res.raw_value - ideal))
        mit_errors.append(abs(res.mitigated_value - ideal))
        strengths_used.append(s)
        reductions.append(err_reduction)
        print(f"  strength={s:.2f}  raw={res.raw_value:.4f}  mitigated={res.mitigated_value:.4f}  "
              f"ideal={ideal:.4f}  error_reduction={err_reduction:.1f}%")

    viz.plot_accuracy_vs_noise(
        strengths_used, raw_errors, mit_errors,
        title="PEC: |<ZZ> error| vs Noise Strength (Bell state)",
        save_path=os.path.join(FIG_DIR, "E_pec_error_vs_noise.png"),
    )


# ---------------------------------------------------------------------------
# Experiment F: Virtual Distillation (M=2) on GHZ state, single-qubit Z0
# ---------------------------------------------------------------------------
def experiment_vd():
    print("\n=== Experiment F: Virtual Distillation (M=2, ancilla-based) ===")
    bc = vqe_ansatz(2, layers=1)
    obs = SparsePauliOp.from_list([("IZ", 1.0)])  # single-qubit Z on qubit 0
    ideal = noiseless_expectation(bc.circuit, obs)
    n = bc.n_qubits
    vd_total_qubits = 2 * n + 1  # registers A, B, + 1 ancilla

    raw_errors, mit_errors, strengths_used, reductions = [], [], [], []
    for s in NOISE_STRENGTHS:
        t0 = time.time()
        nm_vd = build_noise_model(NoiseProfile(strength=s, n_qubits=vd_total_qubits))
        nm_raw = build_noise_model(NoiseProfile(strength=s, n_qubits=n))
        counts_exec = make_counts_executor(nm_vd, shots=SHOTS)
        exp_exec = make_expectation_executor(nm_raw, obs, shots=SHOTS)
        res = run_virtual_distillation(bc.circuit, obs, counts_exec, raw_executor=exp_exec)
        wall_time = time.time() - t0

        err_reduction = error_reduction_percent(ideal, res.raw_value, res.mitigated_value)
        log_row("F_VD", "Virtual Distillation (M=2)", bc.name, s, ideal, res.raw_value,
                 res.mitigated_value, err_reduction, overhead_factor=2.0, wall_time=wall_time,
                 extra=f"denominator(Tr rho^2)={res.denominator_estimate:.4f}")

        raw_errors.append(abs(res.raw_value - ideal))
        mit_errors.append(abs(res.mitigated_value - ideal))
        strengths_used.append(s)
        reductions.append(err_reduction)
        print(f"  strength={s:.2f}  raw={res.raw_value:.4f}  mitigated={res.mitigated_value:.4f}  "
              f"ideal={ideal:.4f}  error_reduction={err_reduction:.1f}%  denom={res.denominator_estimate:.3f}")

    viz.plot_accuracy_vs_noise(
        strengths_used, raw_errors, mit_errors,
        title="Virtual Distillation: |<Z0> error| vs Noise Strength (VQE ansatz, 2q)",
        save_path=os.path.join(FIG_DIR, "F_vd_error_vs_noise.png"),
    )


# ---------------------------------------------------------------------------
# Experiment G: QAOA MaxCut cross-technique comparison at fixed noise strength
# ---------------------------------------------------------------------------
def experiment_qaoa_comparison():
    print("\n=== Experiment G: Cross-Technique Comparison on QAOA MaxCut ===")
    bc = qaoa_maxcut(4, p=1)
    ideal = bc.ideal_expectation
    s = 0.2
    nm = build_noise_model(NoiseProfile(strength=s, n_qubits=bc.n_qubits))
    executor = make_expectation_executor(nm, bc.observable, shots=SHOTS)
    raw_value = executor(bc.circuit)

    names, reductions, overheads = [], [], []

    zne_res = run_zne(bc.circuit, executor, scale_factors=[1.0, 2.0, 3.0], extrapolation="richardson")
    er = error_reduction_percent(ideal, raw_value, zne_res.mitigated_value)
    names.append("ZNE"); reductions.append(er); overheads.append(3.0)
    log_row("G_QAOA_compare", "ZNE", bc.name, s, ideal, raw_value, zne_res.mitigated_value, er, 3.0, None)

    ddd_res = run_ddd(bc.circuit, executor, rule="xyxy")
    er = error_reduction_percent(ideal, raw_value, ddd_res.mitigated_value)
    names.append("DDD"); reductions.append(er); overheads.append(1.0)
    log_row("G_QAOA_compare", "DDD", bc.name, s, ideal, raw_value, ddd_res.mitigated_value, er, 1.0, None)

    cdr_res = run_cdr(bc.circuit, executor, bc.observable, num_training_circuits=8, fraction_non_clifford=0.2)
    er = error_reduction_percent(ideal, raw_value, cdr_res.mitigated_value)
    names.append("CDR"); reductions.append(er); overheads.append(8.0)
    log_row("G_QAOA_compare", "CDR", bc.name, s, ideal, raw_value, cdr_res.mitigated_value, er, 8.0, None)

    print(f"  ideal={ideal:.4f}  raw={raw_value:.4f}")
    for n, r in zip(names, reductions):
        print(f"  {n:6s} error_reduction={r:.1f}%")

    viz.plot_error_reduction_bar(
        names, reductions,
        title=f"QAOA MaxCut (4q, p=1): Error Reduction by Technique (noise={s})",
        save_path=os.path.join(FIG_DIR, "G_qaoa_technique_comparison.png"),
    )
    viz.plot_overhead_vs_accuracy(
        names, overheads, reductions,
        title="Accuracy Improvement vs. Circuit-Evaluation Overhead (QAOA)",
        save_path=os.path.join(FIG_DIR, "G_qaoa_overhead_vs_accuracy.png"),
    )


# ---------------------------------------------------------------------------
# Experiment H: Scalability (wall-clock runtime vs. qubit count)
# ---------------------------------------------------------------------------
def experiment_scalability():
    print("\n=== Experiment H: Scalability (runtime vs. qubit count) ===")
    qubit_counts = [2, 3, 4, 5]
    zne_times, mem_times = [], []

    for n in qubit_counts:
        bc = ghz_state(n)
        obs = SparsePauliOp.from_list([("Z" * n, 1.0)])
        nm = build_noise_model(NoiseProfile(strength=0.15, n_qubits=n))

        t0 = time.time()
        executor = make_expectation_executor(nm, obs, shots=1500)
        run_zne(bc.circuit, executor, scale_factors=[1.0, 2.0, 3.0])
        zne_times.append(time.time() - t0)

        t0 = time.time()
        counts_exec = make_counts_executor(readout_only_model(0.15, n), shots=1500)
        raw_counts = counts_exec(bc.measured_circuit())
        cal, method = build_calibration_matrix(n, readout_only_model(0.15, n), shots=1500)
        apply_mem(raw_counts, n, cal, method)
        mem_times.append(time.time() - t0)

        log_row("H_scalability", "ZNE", bc.name, 0.15, None, None, None, None, None, zne_times[-1],
                 extra=f"n_qubits={n}")
        log_row("H_scalability", "MEM", bc.name, 0.15, None, None, None, None, None, mem_times[-1],
                 extra=f"n_qubits={n}")
        print(f"  n_qubits={n}  ZNE time={zne_times[-1]:.2f}s  MEM time={mem_times[-1]:.2f}s "
              f"(MEM calibration circuits = 2^{n} = {2**n})")

    viz.plot_scalability(
        qubit_counts, {"ZNE": zne_times, "MEM (full calibration)": mem_times},
        title="Wall-clock Runtime vs. Qubit Count",
        save_path=os.path.join(FIG_DIR, "H_scalability.png"),
    )


def write_csv():
    path = os.path.join(DATA_DIR, "benchmark_results.csv")
    fieldnames = ["experiment", "technique", "benchmark", "noise_strength", "ideal", "raw",
                  "mitigated", "error_reduction_pct", "overhead_factor", "wall_time_s", "extra"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nSaved {len(all_rows)} result rows to {path}")


if __name__ == "__main__":
    t_start = time.time()
    experiment_mem()
    experiment_zne()
    experiment_ddd()
    experiment_cdr()
    experiment_pec()
    experiment_vd()
    experiment_qaoa_comparison()
    experiment_scalability()
    write_csv()
    print(f"\nTotal benchmark suite runtime: {time.time() - t_start:.1f}s")
    print(f"Figures saved to: {FIG_DIR}")
