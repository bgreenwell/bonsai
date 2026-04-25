#!/usr/bin/env python3
"""Generate H2O MOJO regression test case (numeric features only)."""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data, inject_nans, save_test_data

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from sklearn.model_selection import train_test_split

# Configuration
SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_TREES = 50
MAX_DEPTH = 5
N_NUMERIC = 10
NAN_PCT = 0.05

def main():
    print("=" * 70)
    print("H2O MOJO - Regression (Numeric Features Only)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    # 1. Generate data
    print("\n[1/5] Generating synthetic data...")
    X, y = make_synthetic_data(N_SAMPLES, N_NUMERIC, task="regression", seed=SEED)
    X = inject_nans(X, NAN_PCT, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=N_TEST, random_state=SEED)
    print(f"   Train: {X_train.shape[0]} samples, {N_NUMERIC} features")

    # 2. Train H2O model
    print("\n[2/5] Training H2O GBM...")
    h2o.init()
    train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    train_df["target"] = y_train
    h2o_train = h2o.H2OFrame(train_df)

    test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    test_df["target"] = y_test
    h2o_test = h2o.H2OFrame(test_df)

    gbm = H2OGradientBoostingEstimator(
        ntrees=N_TREES, max_depth=MAX_DEPTH, distribution="gaussian", seed=SEED
    )
    gbm.train(x=[f"feat_{i}" for i in range(N_NUMERIC)], y="target", training_frame=h2o_train)
    print(f"   Model trained: {gbm.ntrees} trees")

    # 3. Export MOJO
    print("\n[3/5] Exporting MOJO...")
    mojo_path = gbm.download_mojo(path=str(output_dir), get_genmodel_jar=False)
    os.rename(mojo_path, output_dir / "model.zip")
    print(f"   ✓ MOJO saved: {output_dir / 'model.zip'}")

    # 4. Save test data
    print("\n[4/5] Generating ground truth...")
    ground_truth = gbm.predict(h2o_test).as_data_frame().values.flatten()
    save_test_data(X_test, y_test, ground_truth, None, None, output_dir / "test_data.csv", "regression")

    # 5. Save metadata
    print("\n[5/5] Saving metadata...")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "format": "h2o_mojo", "task": "regression",
            "n_trees": N_TREES, "n_numeric_features": N_NUMERIC,
            "n_categorical_features": 0, "seed": SEED
        }, f, indent=2)

    h2o.cluster().shutdown(prompt=False)
    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
