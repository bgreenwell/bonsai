#!/usr/bin/env python3
"""
Generate H2O MOJO binary classification test case (numeric features only).

Creates:
- generated/model.zip (H2O MOJO model)
- generated/test_data.csv (test features + ground truth predictions)
- generated/metadata.json (model metadata)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add common utilities to path
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
    print("H2O MOJO - Binary Classification (Numeric Features Only)")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    # 1. Generate synthetic data
    print("\n[1/5] Generating synthetic data...")
    X, y = make_synthetic_data(
        n_samples=N_SAMPLES,
        n_numeric=N_NUMERIC,
        n_informative=8,
        task="classification",
        seed=SEED
    )

    # Inject NaN values
    X = inject_nans(X, NAN_PCT, seed=SEED)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED, stratify=y
    )

    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Classes: {np.bincount(y_train)} (train), {np.bincount(y_test)} (test)")
    print(f"   NaN cells: {np.isnan(X_train).sum()} train, {np.isnan(X_test).sum()} test")

    # 2. Start H2O and train model
    print("\n[2/5] Training H2O GBM...")
    h2o.init()

    # Convert to H2O frames
    train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    train_df["target"] = y_train
    h2o_train = h2o.H2OFrame(train_df)
    h2o_train["target"] = h2o_train["target"].asfactor()

    test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    test_df["target"] = y_test
    h2o_test = h2o.H2OFrame(test_df)
    h2o_test["target"] = h2o_test["target"].asfactor()

    # Train GBM
    gbm = H2OGradientBoostingEstimator(
        ntrees=N_TREES,
        max_depth=MAX_DEPTH,
        distribution="bernoulli",
        seed=SEED,
        model_id="h2o_gbm_classification_numeric",
    )

    feature_cols = [f"feat_{i}" for i in range(N_NUMERIC)]
    gbm.train(x=feature_cols, y="target", training_frame=h2o_train)

    print(f"   Model trained: {gbm.ntrees} trees, max_depth {gbm.max_depth}")
    print(f"   AUC (train): {gbm.auc(train=True):.4f}")

    # 3. Export MOJO
    print("\n[3/5] Exporting MOJO...")
    mojo_path = gbm.download_mojo(path=str(output_dir), get_genmodel_jar=False)
    # Rename to standard name
    os.rename(mojo_path, output_dir / "model.zip")
    print(f"   ✓ MOJO saved: {output_dir / 'model.zip'}")

    # 4. Generate predictions and save test data
    print("\n[4/5] Generating ground truth predictions...")
    h2o_preds = gbm.predict(h2o_test)
    proba_col = h2o_preds.columns[2]  # Usually 'p1' for positive class
    ground_truth = h2o_preds[proba_col].as_data_frame().values.flatten()

    save_test_data(
        X_test=X_test,
        y_test=y_test,
        predictions=ground_truth,
        categorical_arrays=None,
        categorical_names=None,
        output_path=output_dir / "test_data.csv",
        task="classification"
    )

    # 5. Save metadata
    print("\n[5/5] Saving metadata...")
    metadata = {
        "format": "h2o_mojo",
        "task": "classification",
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_trees": N_TREES,
        "max_depth": MAX_DEPTH,
        "n_numeric_features": N_NUMERIC,
        "n_categorical_features": 0,
        "categorical_features": [],
        "nan_pct": NAN_PCT,
        "seed": SEED,
        "distribution": "bernoulli",
        "link_function": "logit",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Metadata saved: {output_dir / 'metadata.json'}")

    # Cleanup
    h2o.cluster().shutdown(prompt=False)

    print("\n" + "=" * 70)
    print("✓ SUCCESS! Generated files:")
    print(f"   - {output_dir / 'model.zip'}")
    print(f"   - {output_dir / 'test_data.csv'}")
    print(f"   - {output_dir / 'metadata.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
