#!/usr/bin/env python3
"""Generate sklearn ONNX classification test case (numeric features only)."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data, save_test_data

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_TREES = 50
MAX_DEPTH = 5
N_NUMERIC = 10

def main():
    print("=" * 70)
    print("sklearn ONNX - Classification (Numeric Features Only)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/5] Generating synthetic data...")
    X, y = make_synthetic_data(N_SAMPLES, N_NUMERIC, task="classification", seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples, {N_NUMERIC} features")

    print("\n[2/5] Training sklearn GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    print(f"   Model trained: {len(clf.estimators_)} trees")

    print("\n[3/5] Exporting to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, N_NUMERIC]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=14)
    onnx_path = output_dir / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   ✓ ONNX saved: {onnx_path}")

    print("\n[4/5] Generating ground truth...")
    ground_truth = clf.predict_proba(X_test)[:, 1]
    save_test_data(X_test, y_test, ground_truth, None, None,
                   output_dir / "test_data.csv", "classification")

    print("\n[5/5] Saving metadata...")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "format": "sklearn_onnx", "task": "classification",
            "n_trees": N_TREES, "n_numeric_features": N_NUMERIC,
            "n_categorical_features": 0, "seed": SEED,
        }, f, indent=2)

    print("\n✓ SUCCESS!")

if __name__ == "__main__":
    main()
