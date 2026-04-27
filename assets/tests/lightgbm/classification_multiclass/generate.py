#!/usr/bin/env python3
"""Generate LightGBM multiclass classification test case (numeric features only)."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import inject_nans, save_test_data_multiclass

import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_ROUNDS = 20            # num_boost_round; LightGBM trains N_ROUNDS * N_CLASSES trees
N_CLASSES = 3
MAX_DEPTH = 4
N_NUMERIC = 8

def main():
    print("=" * 70)
    print("LightGBM - Multiclass Classification (3 classes, Numeric Features)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Generating synthetic data...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_NUMERIC,
        n_informative=6,
        n_redundant=1,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=SEED,
    )
    X = inject_nans(X, 0.03, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples, {N_NUMERIC} features, {N_CLASSES} classes")

    print("\n[2/4] Training LightGBM multiclass classifier...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "max_depth": MAX_DEPTH,
        "num_leaves": 2 ** MAX_DEPTH,
        "seed": SEED,
        "verbose": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS)
    total_trees = N_ROUNDS * N_CLASSES
    print(f"   Model trained: {total_trees} trees ({N_ROUNDS} rounds × {N_CLASSES} classes)")

    print("\n[3/4] Exporting JSON model...")
    model_path = output_dir / "model.json"
    model_dict = model.dump_model()
    with open(model_path, "w") as f:
        json.dump(model_dict, f)
    print(f"   ✓ Saved: {model_path}")

    print("\n[4/4] Generating ground truth predictions (per-class probabilities)...")
    # predict() with objective=multiclass returns shape (n_samples, n_classes)
    proba_matrix = model.predict(X_test)
    save_test_data_multiclass(X_test, y_test, proba_matrix, output_dir / "test_data.csv")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "format": "lightgbm",
            "task": "multiclass",
            "n_classes": N_CLASSES,
            "n_trees": total_trees,
            "n_numeric_features": N_NUMERIC,
            "n_categorical_features": 0,
            "seed": SEED,
            "objective": "multiclass",
        }, f, indent=2)

    print("\n✓ SUCCESS!")

if __name__ == "__main__":
    main()
