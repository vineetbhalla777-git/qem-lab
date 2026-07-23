# The Six Mitigation Techniques Implemented in This Project

## Zero-Noise Extrapolation (ZNE)

Deliberately amplifies a circuit's noise at several scale factors (e.g. by
"folding" gates -- inserting a gate and its inverse to triple its
effective duration without changing the logical operation), measures the
expectation value at each scale, then extrapolates back to the zero-noise
limit using a fitted curve (linear, polynomial, or Richardson
extrapolation). Implemented here via Mitiq. In this project's own
benchmark data, ZNE performed inconsistently on a shallow 3-qubit GHZ
circuit -- sometimes making the estimate worse -- because short circuits
give the extrapolation too few, too-similar data points to fit reliably.
Mitiq itself warns about this exact scenario.

## Measurement Error Mitigation (MEM)

Targets only the classical, easily-characterized error that occurs when
reading out a qubit's final state. Works by preparing every computational
basis state, measuring it many times to build a "confusion matrix"
(probability of reading state j given state i was prepared), then
inverting that matrix to correct raw measurement counts. In this
project's benchmarks, MEM was the most reliable technique tested,
consistently recovering near-ideal fidelity because readout error doesn't
depend on circuit structure the way gate errors do.

## Probabilistic Error Cancellation (PEC)

Represents each ideal (noiseless) gate as a quasi-probability-weighted
combination of *actually implementable* noisy operations. Sampling
circuits according to this quasi-probability distribution and averaging
results (with sign flips from negative coefficients) gives an unbiased
estimator of the noiseless expectation value -- at the cost of
significant sampling overhead that grows with assumed noise strength.
Accuracy depends heavily on how well the *assumed* noise model matches the
*actual* device noise; a mismatch (or too few samples) can make PEC
perform worse than doing nothing.

## Clifford Data Regression (CDR)

A learning-based approach: generate near-Clifford training circuits
(cheap to simulate exactly on a classical computer since Clifford circuits
don't require full quantum simulation), run each on both the noisy device
and the exact classical simulator, then fit a regression model mapping
noisy results to ideal results. Apply that fitted map to the real circuit
of interest. Because it learns an empirical correction rather than
assuming a specific noise channel, CDR degraded more gracefully than ZNE
or PEC as noise increased in this project's tests.

## Dynamical Decoupling (DDD)

Inserts pulse sequences (such as XX or XYXY) into idle windows in a
circuit -- periods where a qubit is doing nothing while other qubits are
being operated on. These sequences compose to the identity in the
noiseless case but average out slow-drifting dephasing noise during the
idle time. It's nearly free (no extra circuit evaluations needed) but only
helps when there's meaningful idle time to fill; on tightly-scheduled
small benchmark circuits its effect is modest.

## Virtual Distillation (VD)

Given M copies of a noisy state, measuring an observable on the
*virtually purified* state (proportional to rho^M) suppresses error
exponentially in M, for noise models with one dominant "true" eigenvector.
This project implements the M=2 case via an ancilla-assisted SWAP-test
circuit (one ancilla qubit, controlled-SWAP gates between the two
copies), verified exactly against closed-form density-matrix calculations.
The catch: the protocol's own extra circuitry (ancilla + controlled-SWAPs)
introduces additional gate noise, which can partly offset the exponential
suppression it's designed to provide -- especially at shallow depths and
low qubit counts, a limitation also documented in the wider literature
(e.g. Krebsbach et al., arXiv:2210.15317).

---
*This article describes this project's own implementations; see
`qem_framework/mitigation/` in the codebase and `REPORT.md` for the full
methodology and measured results.*
