"""
dynamical_decoupling.py
========================
Digital Dynamical Decoupling (DD), used to suppress dephasing/decoherence
error during idle ("slack") periods in a circuit by inserting pulse
sequences (e.g. XX, XYXY) that average out low-frequency noise while
composing to identity in the noiseless case.

Wraps Mitiq's `ddd` sub-package, which inserts these sequences into idle
windows automatically based on circuit scheduling.
"""


from dataclasses import dataclass
from typing import Callable, Optional

from qiskit import QuantumCircuit
from mitiq import ddd


@dataclass
class DDDResult:
    mitigated_value: float
    raw_value: float
    rule_name: str


_RULES = {
    "xx": ddd.rules.xx,
    "yy": ddd.rules.yy,
    "xyxy": ddd.rules.xyxy,
}


def run_ddd(
    circuit: QuantumCircuit,
    executor: Callable[[QuantumCircuit], float],
    rule: str = "xyxy",
) -> DDDResult:
    """Apply dynamical decoupling sequence `rule` to idle windows in
    `circuit`, then evaluate with `executor`."""
    if rule not in _RULES:
        raise ValueError(f"Unknown DD rule: {rule}. Choose from {list(_RULES)}")

    raw_value = executor(circuit)
    rule_fn = _RULES[rule]

    mitigated = ddd.execute_with_ddd(
        circuit,
        executor,
        rule=rule_fn,
    )

    return DDDResult(
        mitigated_value=float(mitigated),
        raw_value=float(raw_value),
        rule_name=rule,
    )
