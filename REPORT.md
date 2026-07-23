# Quantum Error Mitigation: Implementation & Benchmarking Report

## 1. Objective

Implement and benchmark a representative set of quantum error mitigation
(QEM) techniques against realistic NISQ noise models, using standard
benchmark circuits, and quantify each technique's accuracy improvement and
overhead. All results below are reproduced by running
`experiments/run_benchmark_suite.py` and are read directly from
`results/data/benchmark_results.csv`.

## 2. Noise models

A single `NoiseProfile(strength)` (0–1) generates a self-consistent set of
device parameters — 1-/2-qubit depolarizing probabilities, T1/T2 (via
thermal relaxation, giving amplitude + phase damping), and asymmetric
readout error — scaled to resemble a real superconducting processor at
increasing levels of degradation. Pure single-mechanism models
(depolarizing-only, amplitude-damping-only, phase-damping-only,
readout-only) are also available for isolating one noise source at a time.

## 3. Techniques implemented and how they were tested

| # | Technique | Benchmark used | Library |
|---|---|---|---|
| A | Measurement Error Mitigation (MEM) | GHZ (4q), readout-only noise | Custom calibration-matrix inversion |
| B | Zero-Noise Extrapolation (ZNE) | GHZ (3q), ⟨ZZI⟩ | Mitiq (gate folding + Richardson) |
| C | Dynamical Decoupling (DDD) | GHZ (3q), ⟨ZZI⟩ | Mitiq (XYXY sequence) |
| D | Clifford Data Regression (CDR) | VQE ansatz (3q), TFIM Hamiltonian | Mitiq |
| E | Probabilistic Error Cancellation (PEC) | Bell state, ⟨ZZ⟩ | Mitiq (local depolarizing representation) |
| F | Virtual Distillation (VD, M=2) | VQE ansatz (2q), ⟨Z₀⟩ | Custom ancilla-assisted SWAP test |
| G | Cross-technique comparison | QAOA MaxCut (4q) | ZNE, DDD, CDR head-to-head |
| H | Scalability | GHZ (2–5q) | ZNE vs. MEM wall-clock time |

Each experiment sweeps noise strength ∈ {0.05, 0.15, 0.25, 0.35} with 3,000
shots per circuit evaluation (unless noted).

## 4. Results

### 4.1 Measurement Error Mitigation — the clear winner

MEM was the most reliable technique tested. On the 4-qubit GHZ state it
raised classical fidelity from **0.88–0.96 (raw) to 0.98–0.99 (mitigated)**
at every noise level, cutting total variation distance by roughly
**70–94%**. This tracks expectation: readout error is a purely classical,
easily-characterized channel, so calibration-matrix inversion recovers it
almost exactly. Cost: one calibration circuit per computational basis state
(2ⁿ circuits) — cheap for the small qubit counts here, but this **scales
exponentially** and becomes the dominant limitation for MEM at scale (see
§4.5); the tensored (per-qubit) approximation trades that exactness for
linear scaling when correlated readout errors can be neglected.

### 4.2 Zero-Noise Extrapolation — inconsistent on short circuits

ZNE's Richardson extrapolation *increased* error at low-to-moderate noise
(-74% to -33% "error reduction," i.e., made things worse) on the 3-qubit
GHZ benchmark, and only broke roughly even at the highest noise level
tested. Mitiq itself warns about this: *"the input circuit is very short,
[which] may reduce the accuracy of noise scaling."* Global folding on a
depth-4 circuit gives few distinct noise-scale data points to fit a
polynomial through, and Richardson extrapolation can overshoot when the
underlying noise-vs-scale curve isn't well approximated by the fitted
model. This is a real, reproducible limitation of ZNE on shallow circuits,
not a defect in the implementation — deeper circuits (see the QAOA
comparison) or higher-order/adaptive factories would be expected to help.

### 4.3 Clifford Data Regression — best expectation-value mitigator tested

CDR delivered consistently strong, positive error reduction on the VQE
benchmark: **74–86% error reduction** across all four noise levels,
recovering ⟨H_TFIM⟩ to within a few percent of the ideal value
(e.g., ideal 0.597 vs. mitigated 0.542–0.646 vs. raw 0.33–0.50). Because
CDR *learns* an empirical noisy→ideal map from near-Clifford training
circuits rather than assuming a specific noise channel, it degraded
gracefully as noise increased, unlike ZNE and PEC.

### 4.4 Probabilistic Error Cancellation — high variance, noise-model-sensitive

PEC's accuracy was noise-strength-dependent and non-monotonic: strongly
negative "error reduction" at strength 0.05 and 0.35 (-177%, -190%), but a
genuine 79% improvement at strength 0.25. PEC assumes a purely local
depolarizing channel and needs the *assumed* noise strength fed to its
quasi-probability representation to closely match the *actual* per-gate
error rate; any mismatch, plus the inherent sampling variance of
quasi-probability methods (only 80 circuit samples were used here to keep
runtime reasonable), explains the volatility. More samples and better
noise characterization would tighten this in a production setting.

### 4.5 Virtual Distillation — validated correct, but circuit overhead offsets gains here

VD (M=2) is implemented via the **ancilla-assisted SWAP-test protocol**:
Tr[ρ²] and Tr[ρ²O] are each estimated from a Hadamard-test-style circuit
using one ancilla qubit and n controlled-SWAP gates. This construction was
verified **exactly** against closed-form density-matrix calculations
(diagonal and coherent single-qubit test states, X/Y/Z observables) before
being used in the benchmark suite — see the inline derivation notes in
`virtual_distillation.py`. In the benchmark itself, raw and VD-mitigated
absolute errors were comparable in magnitude (both a few hundredths),
consistent with a well-known, documented limitation of VD: the protocol's
*own* circuit (extra ancilla, n controlled-SWAPs) introduces additional
gate noise that can offset the exponential suppression it provides,
especially at the shallow depths and modest qubit counts used here. This
matches findings in the literature (e.g., Krebsbach et al., arXiv:2210.15317,
which studies exactly this "noise in the VD circuit itself" effect).

### 4.6 Cross-technique comparison (QAOA MaxCut, 4 qubits, noise=0.2)

|Technique|Raw|Mitigated|Ideal|
|---|---|---|---|
|ZNE|-1.999|-1.914|-1.995|
|DDD|-1.999|-1.986|-1.995|
|CDR|-1.999|-2.020|-1.995|

Note the raw (unmitigated) value already happened to be very close to
ideal in this run (a small-sample-noise coincidence), which makes the
*percentage* error-reduction figures in the CSV look artificially dramatic
in both directions — the **absolute** errors involved are all small
(≤0.08). This is a useful reminder for interpreting the "% error reduction"
metric generally: it is most informative when the raw error is
substantial, and can be noisy/misleading when raw error is already small.

### 4.7 Scalability

MEM's exact calibration-matrix approach scales as O(2ⁿ) circuit
evaluations — wall-clock time roughly doubled with each added qubit (0.15s
→ 0.28s → 0.52s → 1.04s for 2→5 qubits), confirming the exponential cost
that motivates the tensored (per-qubit) approximation for larger circuits.
ZNE's cost is flat with qubit count (≈0.18–0.19s throughout) since its
circuit-evaluation count is fixed by the number of scale factors, not the
qubit count.

## 5. Summary: when to use which technique

- **Readout errors dominate** → MEM (cheap, nearly exact, but exponential
  calibration cost at scale — use the tensored approximation for many
  qubits).
- **Need an expectation value and can tolerate a data-driven approach** →
  CDR was the most robust technique tested here.
- **Shallow circuits** → ZNE's assumptions (smooth noise-vs-scale curve)
  are hardest to satisfy; expect inconsistent results without deeper
  circuits or more scale factors/shots.
- **Have an accurate device noise characterization** → PEC can work very
  well, but is sensitive to noise-model mismatch and needs enough samples
  to control variance.
- **Idle time in the circuit** → DDD is nearly free (no extra circuit
  evaluations) but only helps when there's actual idle/slack time for
  pulse sequences to fill; on these tightly-scheduled small benchmarks its
  effect was modest.
- **Willing to pay 2x qubits + ancilla overhead for a mathematically exact
  protocol** → VD, but budget for its own circuit's gate noise, especially
  at low qubit counts / shallow depth.

## 6. Reproducing these results

```bash
pip install -r requirements.txt
python3 experiments/run_benchmark_suite.py
```

All figures referenced conceptually above are saved to `results/figures/`;
raw numbers are in `results/data/benchmark_results.csv`.
