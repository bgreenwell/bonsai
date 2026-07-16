#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["catboost", "numpy", "pandas", "scikit-learn"]
# ///
"""Generate CatBoost regression test case (numeric features only)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 500
N_TEST = 50
N_TREES = 20
DEPTH = 4
N_NUMERIC = 5


def main():
    print("=" * 70)
    print("CatBoost - Regression (Numeric Features Only)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Generating synthetic data...")
    X, y = make_synthetic_data(N_SAMPLES, N_NUMERIC, task="regression", seed=SEED)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED
    )

    print("\n[2/4] Training CatBoost regressor...")
    model = CatBoostRegressor(
        iterations=N_TREES,
        depth=DEPTH,
        learning_rate=0.3,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)

    print("\n[3/4] Exporting JSON model...")
    model.save_model(str(output_dir / "model.json"), format="json")

    print("\n[4/4] Generating ground truth predictions...")
    ground_truth = model.predict(X_test)

    header = [f"feat_{i}" for i in range(N_NUMERIC)] + ["ground_truth"]
    with open(output_dir / "test_data.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for row, gt in zip(X_test, ground_truth):
            f.write(",".join(repr(float(v)) for v in row) + f",{float(gt)!r}\n")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "format": "catboost",
                "task": "regression",
                "n_trees": N_TREES,
                "depth": DEPTH,
                "n_numeric_features": N_NUMERIC,
                "categorical_features": [],
                "seed": SEED,
                "loss_function": "RMSE",
            },
            f,
            indent=2,
        )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()
