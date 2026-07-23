"""
train_model.py
===============
Trains a regression model that predicts a technique's *absolute mitigated
error* given (technique, noise strength, qubit count, benchmark family).

Why predict absolute error instead of "% error reduction": the percentage
metric divides by the raw (unmitigated) error, which can be tiny -- and
when it is, small absolute differences produce huge, unstable percentages
(this is documented in REPORT.md as a real limitation of that metric, not
just a modeling inconvenience). Predicting the absolute mitigated error
directly avoids that instability, and recommendation is done by comparing
predicted absolute errors across techniques (lower = better) rather than
by comparing volatile percentages.

Run with:
    python3 -m backend.ml.train_model
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "training_data.csv")
MODEL_PATH = os.path.join(HERE, "models", "error_predictor.joblib")
METRICS_PATH = os.path.join(HERE, "models", "metrics.txt")

FEATURES = ["technique", "benchmark", "noise_strength", "n_qubits"]
TARGET = "mitigated_error"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["technique", "benchmark"]),
        ],
        remainder="passthrough",
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Also report a naive baseline (predict the training mean) for context --
    # a model is only useful if it clearly beats this.
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    report = (
        f"Training rows: {len(X_train)}   Test rows: {len(X_test)}\n"
        f"Model MAE:     {mae:.5f}\n"
        f"Baseline MAE (predict mean): {baseline_mae:.5f}\n"
        f"Model R^2:     {r2:.4f}\n"
    )
    print(report)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        f.write(report)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
