"""
metrics.py
==========
Performance-analysis utilities (Phase IV of the project scope):
    - Fidelity improvement (classical fidelity between distributions)
    - Expectation-value accuracy (absolute / relative error)
    - Error reduction percentage
    - Sampling & runtime overhead bookkeeping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def classical_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Classical (Bhattacharyya) fidelity between two probability
    distributions given as bitstring -> probability dicts:
        F(p, q) = ( sum_x sqrt(p_x * q_x) )^2
    Returns a value in [0, 1]; 1 = identical distributions."""
    keys = set(p) | set(q)
    total = sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)
    return float(np.clip(total ** 2, 0.0, 1.0))


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """TVD(p, q) = 0.5 * sum_x |p_x - q_x|. 0 = identical, 1 = disjoint supports."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def counts_to_probs(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


def expectation_error(ideal: float, estimate: float) -> Dict[str, float]:
    abs_err = abs(estimate - ideal)
    denom = abs(ideal) if abs(ideal) > 1e-9 else 1.0
    rel_err = abs_err / denom
    return {"absolute_error": abs_err, "relative_error": rel_err}


def error_reduction_percent(ideal: float, raw: float, mitigated: float) -> float:
    """Percentage reduction in absolute error achieved by mitigation, relative
    to the raw (unmitigated) error. Positive = improvement, negative = the
    mitigation made things worse."""
    raw_err = abs(raw - ideal)
    mit_err = abs(mitigated - ideal)
    if raw_err < 1e-12:
        return 0.0
    return 100.0 * (raw_err - mit_err) / raw_err


@dataclass
class OverheadRecord:
    """Bookkeeping for the accuracy/overhead trade-off analysis (Phase IV)."""
    technique: str
    circuit_evaluations: int          # number of distinct circuits executed
    total_shots: int                  # total measurement shots consumed
    wall_time_seconds: Optional[float] = None
    extra: Dict = field(default_factory=dict)

    def shots_overhead_factor(self, baseline_shots: int) -> float:
        return self.total_shots / baseline_shots if baseline_shots > 0 else float("nan")
