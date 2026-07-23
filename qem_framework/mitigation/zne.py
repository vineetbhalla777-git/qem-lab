"""
zne.py
======
Zero Noise Extrapolation (ZNE), per Temme, Bravyi & Gambetta (2017).

Idea: deliberately amplify the circuit's noise by a set of scale factors
(via gate folding), measure the expectation value at each scale, then
extrapolate back to the zero-noise limit using a fitted curve.

This module wraps Mitiq's `zne` sub-package (gate-folding scaling +
Richardson / linear / polynomial extrapolation) behind a small,
project-specific API so the rest of the framework doesn't need to know
Mitiq's internals.
"""


from dataclasses import dataclass
from typing import Callable, List, Optional

from qiskit import QuantumCircuit
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory, LinearFactory, PolyFactory


@dataclass
class ZNEResult:
    mitigated_value: float
    raw_value: float                 # value at scale factor 1 (no folding)
    scale_factors: List[float]
    scaled_values: List[float]
    extrapolation_method: str


_FACTORY_MAP = {
    "richardson": RichardsonFactory,
    "linear": LinearFactory,
    "poly2": lambda scale_factors: PolyFactory(scale_factors, order=2),
}


def run_zne(
    circuit: QuantumCircuit,
    executor: Callable[[QuantumCircuit], float],
    scale_factors: Optional[List[float]] = None,
    extrapolation: str = "richardson",
) -> ZNEResult:
    """Run ZNE on `circuit` using `executor` (circuit -> expectation value).

    Parameters
    ----------
    scale_factors : noise-scaling factors, e.g. [1, 2, 3]. Must include 1.0
        (the unfolded circuit) so we can also report the raw value.
    extrapolation : one of 'richardson', 'linear', 'poly2'.
    """
    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.0]
    if extrapolation not in _FACTORY_MAP:
        raise ValueError(f"Unknown extrapolation method: {extrapolation}")

    factory_ctor = _FACTORY_MAP[extrapolation]
    factory = factory_ctor(scale_factors)

    mitigated = zne.execute_with_zne(
        circuit,
        executor,
        factory=factory,
        scale_noise=fold_global,
    )

    # Recover the individual scaled expectation values Mitiq already computed
    scaled_values = [r for r in factory.get_expectation_values()]
    raw_value = scaled_values[0] if scaled_values else executor(circuit)

    return ZNEResult(
        mitigated_value=float(mitigated),
        raw_value=float(raw_value),
        scale_factors=list(scale_factors),
        scaled_values=[float(v) for v in scaled_values],
        extrapolation_method=extrapolation,
    )
