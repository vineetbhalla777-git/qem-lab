"""
cdr.py
======
Clifford Data Regression (CDR), per Czarnik et al. (2021) / used widely
alongside Endo et al. as a learning-based mitigation technique.

Idea: generate a set of "training" circuits that are near-Clifford
(and hence classically simulable exactly), run each on both the noisy
device/simulator and the noiseless simulator, then fit a simple regression
model (e.g. linear) mapping noisy expectation values -> ideal expectation
values. Apply that fitted map to the actual noisy result of the circuit
of interest.

Trades classical simulability of the training set for the ability to
learn/correct a noise model implicitly, without needing an explicit noise
representation (unlike PEC).
"""


from dataclasses import dataclass
from typing import Callable, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from mitiq import cdr


@dataclass
class CDRResult:
    mitigated_value: float
    raw_value: float
    num_training_circuits: int
    fraction_non_clifford: float


def _make_simulator_executor(observable: SparsePauliOp):
    """Noiseless statevector executor used to generate training labels."""
    def simulator(circuit: QuantumCircuit) -> float:
        sv = Statevector.from_instruction(circuit)
        return float(sv.expectation_value(observable).real)
    return simulator


def run_cdr(
    circuit: QuantumCircuit,
    executor: Callable[[QuantumCircuit], float],
    observable: SparsePauliOp,
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
) -> CDRResult:
    """Run CDR to mitigate the expectation value of `observable` on `circuit`.

    `executor` should be the NOISY expectation-value executor. The
    noiseless "simulator" needed to generate training labels is built
    internally from a statevector simulation of each near-Clifford
    training circuit (these are cheap since they're near-Clifford).
    """
    simulator = _make_simulator_executor(observable)
    raw_value = executor(circuit)

    mitigated = cdr.execute_with_cdr(
        circuit,
        executor,
        simulator=simulator,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
    )

    return CDRResult(
        mitigated_value=float(mitigated),
        raw_value=float(raw_value),
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
    )
