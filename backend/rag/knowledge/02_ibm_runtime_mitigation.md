# IBM Quantum Runtime: Error Mitigation & Suppression in Practice

IBM Quantum's cloud service (accessed through Qiskit Runtime) bundles
several error mitigation and suppression techniques behind its Estimator
and Sampler primitives, so users don't have to hand-implement them. This
is the production, hardware-facing counterpart to the techniques this
project simulates and studies from first principles.

## Resilience levels

Rather than asking users to configure each technique individually, Qiskit
Runtime's Estimator primitive exposes a single "resilience level" knob
(0-3) that dials in a cost/accuracy trade-off: higher levels apply more
(and more expensive) mitigation for better accuracy.

- **Level 0** -- no error mitigation; fastest, most affected by noise.
- **Level 1** -- turns on measurement-error mitigation via **TREX**
  (Twirled Readout Error eXtinction), which uses randomized calibration
  circuits to correct readout bias. This is the runtime's counterpart to
  the Measurement Error Mitigation (MEM) technique implemented in this
  project, though TREX uses randomization rather than a fixed calibration
  matrix.
- **Level 2** -- adds **Zero-Noise Extrapolation (ZNE)**: circuits are
  run at amplified noise levels and the results extrapolated back toward
  the zero-noise limit -- the same idea implemented in this project's
  `zne.py`, though IBM's production version can amplify noise by
  stretching pulse schedules or digital gate folding rather than folding
  alone.
- **Level 3** -- adds **Probabilistic Error Cancellation (PEC)**, the
  most powerful (and most expensive) option offered, capable of producing
  unbiased expectation values when the device noise model is
  well-characterized -- directly analogous to this project's `pec.py`.

## Error suppression (applied before mitigation, at compile time)

Separately from resilience levels, Qiskit Runtime automatically applies
**dynamical decoupling** to idle qubits at optimization level 1 and above
-- inserting pulse sequences (such as "XpXm") during idle windows to
average out low-frequency dephasing noise, the same concept as this
project's `dynamical_decoupling.py`. **Gate twirling** is another
suppression technique available: it randomizes gate implementations
across many equivalent circuit variants so that coherent errors average
out into simpler, more mitigable stochastic noise.

## The IBM Circuit function

For users who don't want to manage primitives directly, IBM also offers a
higher-level "Circuit function" with its own `optimization_level` and
`mitigation_level` options, which can combine gate twirling with ZNE (via
probabilistic error amplification) under one setting.

---
*Sources consulted and paraphrased: IBM Quantum Documentation -- "Configure
error mitigation," "Error mitigation and suppression techniques," "Combine
error mitigation options with the Estimator primitive," "Error mitigation
with the IBM Circuit function" (docs.quantum.ibm.com / quantum.cloud.ibm.com,
accessed 2026); IBM Research blog, "Introducing new Qiskit Runtime
capabilities" (2022).*
