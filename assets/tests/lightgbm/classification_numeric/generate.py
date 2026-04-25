#!/usr/bin/env python3
"""Generate LightGBM binary classification test case (numeric features only)."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data, inject_nans, save_test_data

import lightgbm as lgb
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_TREES = 50
MAX_DEPTH = 5
N_NUMERIC = 10
NAN_PCT = 0.05

def main():
    print("=" * 70)
    print("LightGBM - Binary Classification (Numeric Features Only)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Generating synthetic data...")
    X, y = make_synthetic_data(N_SAMPLES, N_NUMERIC, task="classification", seed=SEED)
    X = inject_nans(X, NAN_PCT, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples, {N_NUMERIC} features")

    print("\n[2/4] Training LightGBM classifier...")
    # init_score=0 (raw/margin space) so base_score in IR stays 0.0
    dtrain = lgb.Dataset(X_train, label=y_train,
                         init_score=np.zeros(len(y_train)))
    params = {
        "objective": "binary",
        "max_depth": MAX_DEPTH,
        "num_leaves": 2 ** MAX_DEPTH,
        "n_estimators": N_TREES,
        "seed": SEED,
        "verbose": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=N_TREES)
    print(f"   Model trained: {N_TREES} trees")

    print("\n[3/4] Exporting JSON model...")
    model_path = output_dir / "model.json"
    model_dict = model.dump_model()
    with open(model_path, "w") as f:
        json.dump(model_dict, f)
    print(f"   ✓ Saved: {model_path}")

    print("\n[4/4] Generating ground truth predictions (probabilities)...")
    ground_truth = model.predict(X_test)   # already probabilities for binary
    save_test_data(X_test, y_test, ground_truth, None, None,
                   output_dir / "test_data.csv", "classification")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "format": "lightgbm", "task": "classification",
            "n_trees": N_TREES, "n_numeric_features": N_NUMERIC,
            "n_categorical_features": 0, "seed": SEED,
            "objective": "binary",
        }, f, indent=2)

    print("\n✓ SUCCESS!")

if __name__ == "__main__":
    main()
