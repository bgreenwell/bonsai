#!/usr/bin/env python3
"""Generate sklearn ONNX regression test case WITH categorical features."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data, add_categorical_features, inject_nans, save_test_data

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_TREES = 50
MAX_DEPTH = 5
N_NUMERIC = 7
NAN_PCT = 0.05

CATEGORICAL_CONFIGS = [
    {'levels': ['blue', 'green', 'red', 'yellow'], 'effect': 40.0},
    {'levels': ['large', 'medium', 'small'], 'effect': 30.0},
    {'levels': ['A', 'B'], 'effect': 25.0},
]

def main():
    print("=" * 70)
    print("sklearn ONNX - Regression (WITH Categorical Features)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    # Generate data
    print("\n[1/5] Generating data with categoricals...")
    X, y = make_synthetic_data(N_SAMPLES, N_NUMERIC, task="regression", seed=SEED)
    categorical_arrays, y = add_categorical_features(X, y, CATEGORICAL_CONFIGS, seed=SEED)
    cat_a, cat_b, cat_c = categorical_arrays

    X = inject_nans(X, NAN_PCT, seed=SEED)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=N_TEST, random_state=SEED)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    cat_a_train, cat_a_test = cat_a[train_idx], cat_a[test_idx]
    cat_b_train, cat_b_test = cat_b[train_idx], cat_b[test_idx]
    cat_c_train, cat_c_test = cat_c[train_idx], cat_c[test_idx]

    # Label encode categorical features
    print("\n[2/5] Label encoding categorical features...")
    le_a, le_b, le_c = LabelEncoder(), LabelEncoder(), LabelEncoder()
    cat_a_train_enc = le_a.fit_transform(cat_a_train).reshape(-1, 1)
    cat_b_train_enc = le_b.fit_transform(cat_b_train).reshape(-1, 1)
    cat_c_train_enc = le_c.fit_transform(cat_c_train).reshape(-1, 1)

    cat_a_test_enc = le_a.transform(cat_a_test).reshape(-1, 1)
    cat_b_test_enc = le_b.transform(cat_b_test).reshape(-1, 1)
    cat_c_test_enc = le_c.transform(cat_c_test).reshape(-1, 1)

    X_train_combined = np.hstack([X_train, cat_a_train_enc, cat_b_train_enc, cat_c_train_enc])
    X_test_combined = np.hstack([X_test, cat_a_test_enc, cat_b_test_enc, cat_c_test_enc])

    # Train sklearn model
    print("\n[3/5] Training sklearn HistGradientBoostingRegressor...")
    reg = HistGradientBoostingRegressor(
        max_iter=N_TREES,
        max_depth=MAX_DEPTH,
        random_state=SEED,
    )
    reg.fit(X_train_combined, y_train)
    print(f"   Model trained: {len(reg._predictors[0])} trees")

    # Export ONNX
    print("\n[4/5] Exporting to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, N_NUMERIC + 3]))]
    onnx_model = convert_sklearn(reg, initial_types=initial_type, target_opset=14)
    onnx_path = output_dir / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   ✓ ONNX saved: {onnx_path}")
    print("   NOTE: Categorical features are label-encoded as numeric (not bitsets)")

    # Save test data
    print("\n[5/5] Generating ground truth...")
    ground_truth = reg.predict(X_test_combined)
    save_test_data(X_test, y_test, ground_truth, [cat_a_test, cat_b_test, cat_c_test],
                   ['cat_a', 'cat_b', 'cat_c'], output_dir / "test_data.csv", "regression")

    # Metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "format": "sklearn_onnx", "task": "regression",
            "n_trees": N_TREES, "n_numeric_features": N_NUMERIC,
            "n_categorical_features": 3, "seed": SEED,
            "categorical_encoding": "label_encoded"
        }, f, indent=2)

    print("\n✓ SUCCESS!")

if __name__ == "__main__":
    main()
