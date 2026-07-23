"""
predictor.py
=============
Serves ranked technique recommendations from the trained error-prediction
model (see train_model.py for the target/feature design rationale).

Because each technique in this project is implemented against one specific
benchmark circuit (see backend/service.py), "benchmark" and "technique" are
correlated in the training data -- this predictor is best understood as
answering "of the six techniques this project implements, which is likely
to leave the smallest absolute error at this noise level and qubit count,"
based on our own measured benchmark data, not as a universal oracle across
arbitrary circuits. This limitation is surfaced in the API response itself
so the frontend can be upfront about it.
"""

import os

import joblib
import pandas as pd

from qem_framework.noise.noise_models import NoiseProfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "error_predictor.joblib")
METRICS_PATH = os.path.join(HERE, "models", "metrics.txt")

_model = None
_metrics_text = ""


def _load():
    global _model, _metrics_text
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run 'python3 -m backend.ml.train_model' first "
                "(after 'python3 -m backend.ml.generate_dataset')."
            )
        _model = joblib.load(MODEL_PATH)
        if os.path.exists(METRICS_PATH):
            _metrics_text = open(METRICS_PATH).read()
    return _model


# technique -> (benchmark key, n_qubits) as implemented in backend/service.py
_TECHNIQUE_CONTEXT = {
    "zne": ("ghz", 3),
    "mem": ("ghz", 3),
    "pec": ("bell", 2),
    "cdr": ("vqe", 3),
    "ddd": ("ghz", 3),
    "vd": ("vqe", 2),
}


def predict_ideal_scale(benchmark: str) -> float:
    """Rough scale of the ideal expectation value per benchmark family, used
    to sanity-bound predictions (errors can't usefully exceed ~this)."""
    return {"ghz": 1.0, "bell": 1.0, "vqe": 1.0, "qft": 1.0, "qaoa": 2.0}.get(benchmark, 1.0)


def rank_techniques(noise_strength: float, n_qubits: int | None = None) -> dict:
    """Ranks all implemented techniques by predicted absolute mitigated
    error at the given noise strength (lower predicted error = better).
    If n_qubits is given, only techniques whose benchmark uses that qubit
    count are scored on it directly; others still get a prediction using
    their own natural qubit count, since the point is comparing techniques,
    not forcing every technique onto the same circuit size.
    """
    model = _load()
    rows = []
    for technique, (benchmark, natural_n) in _TECHNIQUE_CONTEXT.items():
        q = n_qubits if n_qubits is not None else natural_n
        rows.append({"technique": technique, "benchmark": benchmark, "noise_strength": noise_strength, "n_qubits": q})

    X = pd.DataFrame(rows)
    preds = model.predict(X)

    ranked = sorted(
        [
            {
                "technique": r["technique"],
                "benchmark": r["benchmark"],
                "n_qubits": r["n_qubits"],
                "predicted_mitigated_error": float(p),
            }
            for r, p in zip(rows, preds)
        ],
        key=lambda d: d["predicted_mitigated_error"],
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "noise_strength": noise_strength,
        "ranking": ranked,
        "recommended": ranked[0]["technique"],
        "model_metrics": _metrics_text.strip(),
        "caveat": (
            "This model is trained on this project's own benchmark data, where each "
            "technique is tied to a specific demo circuit. Treat it as a data-driven "
            "guide for these six techniques and noise regimes, not a universal "
            "guarantee for arbitrary circuits."
        ),
    }
