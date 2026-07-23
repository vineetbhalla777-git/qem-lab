"""
pec.py
======
Probabilistic Error Cancellation (PEC), per Temme, Bravyi & Gambetta (2017)
and Endo, Benjamin & Li (2018).

Idea: represent the ideal (noiseless) gate as a quasi-probability
combination of implementable noisy operations. Sampling circuits from this
quasi-probability distribution and averaging the results (weighted by the
sign of each sampled coefficient) gives an unbiased estimator of the
noiseless expectation value -- at the cost of increased sampling variance
(the "sampling overhead", which grows with the noise strength assumed).

This module assumes a (locally) depolarizing noise model, which is the
standard, tractable case supported natively by Mitiq's PEC representations.
The noise strength fed to the representation should match (or approximate)
the real per-gate error rate for the cancellation to be accurate -- this
mismatch is itself an interesting experimental knob (see experiments/).
"""


from dataclasses import dataclass
from typing import Callable, List, Optional

from qiskit import QuantumCircuit
from mitiq import pec
from mitiq.pec.representations import represent_operations_in_circuit_with_local_depolarizing_noise


@dataclass
class PECResult:
    mitigated_value: float
    raw_value: float
    assumed_noise_strength: float
    num_samples: int
    sampling_overhead: float  # Mitiq's estimated 1-norm overhead, if available


def run_pec(
    circuit: QuantumCircuit,
    executor: Callable[[QuantumCircuit], float],
    assumed_noise_strength: float,
    num_samples: int = 200,
) -> PECResult:
    """Run PEC assuming a local depolarizing noise channel of the given
    strength on every gate in `circuit`.

    Parameters
    ----------
    assumed_noise_strength : the per-gate depolarizing probability PEC's
        quasi-probability representation is built for. In a real deployment
        this comes from device characterization; here it should match the
        NoiseProfile used to generate the noisy `executor`.
    num_samples : number of Monte-Carlo circuit samples drawn from the
        quasi-probability representation (more samples -> lower variance,
        higher sampling overhead / runtime).
    """
    representations = represent_operations_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=circuit,
        noise_level=assumed_noise_strength,
    )

    raw_value = executor(circuit)

    mitigated = pec.execute_with_pec(
        circuit,
        executor,
        representations=representations,
        num_samples=num_samples,
        force_run_all=False,
    )

    # Rough one-norm-based overhead estimate: sum of |eta| over representations
    try:
        overhead = sum(r.norm for r in representations) / max(len(representations), 1)
    except Exception:
        overhead = float("nan")

    return PECResult(
        mitigated_value=float(mitigated),
        raw_value=float(raw_value),
        assumed_noise_strength=assumed_noise_strength,
        num_samples=num_samples,
        sampling_overhead=float(overhead),
    )
