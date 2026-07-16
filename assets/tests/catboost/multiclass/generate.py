#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["catboost", "numpy", "pandas", "scikit-learn"]
# ///
"""Generate CatBoost multiclass classification test case."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))

from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 600
N_TEST = 50
N_TREES = 10
DEPTH = 3
N_NUMERIC = 5
N_CLASSES = 3


def main():
    print("=" * 70)
    print("CatBoost - Multiclass Classification")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Generating synthetic data...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_NUMERIC,
        n_informative=4,
        n_redundant=0,
        n_classes=N_CLASSES,
        random_state=SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED
    )

    print("\n[2/4] Training CatBoost multiclass classifier...")
    model = CatBoostClassifier(
        iterations=N_TREES,
        depth=DEPTH,
        learning_rate=0.3,
        loss_function="MultiClass",
        classes_count=N_CLASSES,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)

    print("\n[3/4] Exporting JSON model...")
    model.save_model(str(output_dir / "model.json"), format="json")

    print("\n[4/4] Generating ground truth probabilities...")
    proba = model.predict_proba(X_test)

    header = (
        [f"feat_{i}" for i in range(N_NUMERIC)]
        + ["target"]
        + [f"ground_truth_proba_{c}" for c in range(N_CLASSES)]
    )
    with open(output_dir / "test_data.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for row, target, probs in zip(X_test, y_test, proba):
            cols = [repr(float(v)) for v in row]
            cols.append(str(int(target)))
            cols.extend(repr(float(p)) for p in probs)
            f.write(",".join(cols) + "\n")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "format": "catboost",
                "task": "multiclass",
                "n_trees": N_TREES,
                "depth": DEPTH,
                "n_numeric_features": N_NUMERIC,
                "n_classes": N_CLASSES,
                "categorical_features": [],
                "seed": SEED,
                "loss_function": "MultiClass",
            },
            f,
            indent=2,
        )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()
