#!/usr/bin/env python3
"""
Generate H2O MOJO regression test case WITH categorical features.

This is the most important test case as it validates:
- Categorical bitset encoding/decoding
- Correct branch direction (value IN bitset → RIGHT child)
- Mixed numeric + categorical features

Creates:
- generated/model.zip (H2O MOJO model with categorical features)
- generated/test_data.csv (test features + ground truth predictions)
- generated/metadata.json (model metadata including categorical domains)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add common utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data, add_categorical_features, inject_nans

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from sklearn.model_selection import train_test_split

# Configuration
SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_TREES = 50
MAX_DEPTH = 5
N_NUMERIC = 7  # Reduced to make room for categoricals
NAN_PCT = 0.05

# Categorical feature configurations
CATEGORICAL_CONFIGS = [
    {'levels': ['blue', 'green', 'red', 'yellow'], 'effect': 40.0},  # cat_a: 4 levels
    {'levels': ['large', 'medium', 'small'], 'effect': 30.0},        # cat_b: 3 levels
    {'levels': ['A', 'B'], 'effect': 25.0},                          # cat_c: 2 levels
]

def main():
    print("=" * 70)
    print("H2O MOJO - Regression (WITH Categorical Features)")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    # 1. Generate synthetic data
    print("\n[1/5] Generating synthetic data with categorical features...")
    X, y = make_synthetic_data(
        n_samples=N_SAMPLES,
        n_numeric=N_NUMERIC,
        n_informative=5,
        task="regression",
        seed=SEED
    )

    # Add categorical features
    categorical_arrays, y = add_categorical_features(X, y, CATEGORICAL_CONFIGS, seed=SEED)
    cat_a, cat_b, cat_c = categorical_arrays

    # Inject NaN values in numeric features only
    X = inject_nans(X, NAN_PCT, seed=SEED)

    # Split data (index-based to maintain alignment)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=N_TEST, random_state=SEED)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    cat_a_train, cat_a_test = cat_a[train_idx], cat_a[test_idx]
    cat_b_train, cat_b_test = cat_b[train_idx], cat_b[test_idx]
    cat_c_train, cat_c_test = cat_c[train_idx], cat_c[test_idx]

    print(f"   Train: {X_train.shape[0]} samples, {N_NUMERIC} numeric + 3 categorical features")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Categorical: cat_a (4 levels), cat_b (3 levels), cat_c (2 levels)")
    print(f"   NaN cells: {np.isnan(X_train).sum()} train, {np.isnan(X_test).sum()} test")

    # 2. Start H2O and train model
    print("\n[2/5] Training H2O GBM with categorical features...")
    h2o.init()

    # Create DataFrames with numeric + categorical features
    train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    train_df['cat_a'] = cat_a_train
    train_df['cat_b'] = cat_b_train
    train_df['cat_c'] = cat_c_train
    train_df['target'] = y_train

    test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    test_df['cat_a'] = cat_a_test
    test_df['cat_b'] = cat_b_test
    test_df['cat_c'] = cat_c_test
    test_df['target'] = y_test

    # Convert to H2O frames and mark categoricals as factors
    h2o_train = h2o.H2OFrame(train_df)
    h2o_train['cat_a'] = h2o_train['cat_a'].asfactor()
    h2o_train['cat_b'] = h2o_train['cat_b'].asfactor()
    h2o_train['cat_c'] = h2o_train['cat_c'].asfactor()

    h2o_test = h2o.H2OFrame(test_df)
    h2o_test['cat_a'] = h2o_test['cat_a'].asfactor()
    h2o_test['cat_b'] = h2o_test['cat_b'].asfactor()
    h2o_test['cat_c'] = h2o_test['cat_c'].asfactor()

    # Train GBM
    gbm = H2OGradientBoostingEstimator(
        ntrees=N_TREES,
        max_depth=MAX_DEPTH,
        distribution="gaussian",
        seed=SEED,
        model_id="h2o_gbm_regression_categorical",
    )

    feature_cols = [f"feat_{i}" for i in range(N_NUMERIC)] + ['cat_a', 'cat_b', 'cat_c']
    gbm.train(x=feature_cols, y="target", training_frame=h2o_train)

    print(f"   Model trained: {gbm.ntrees} trees, max_depth {gbm.max_depth}")
    print(f"   RMSE (train): {gbm.rmse(train=True):.4f}")

    # 3. Export MOJO
    print("\n[3/5] Exporting MOJO with categorical features...")
    mojo_path = gbm.download_mojo(path=str(output_dir), get_genmodel_jar=False)
    # Rename to standard name
    os.rename(mojo_path, output_dir / "model.zip")
    print(f"   ✓ MOJO saved: {output_dir / 'model.zip'}")
    print(f"   NOTE: This MOJO contains categorical bitset splits")

    # 4. Generate predictions and save test data
    print("\n[4/5] Generating ground truth predictions...")
    h2o_preds = gbm.predict(h2o_test)
    ground_truth = h2o_preds.as_data_frame().values.flatten()

    # Save test data with categorical features
    from generators import save_test_data
    save_test_data(
        X_test=X_test,
        y_test=y_test,
        predictions=ground_truth,
        categorical_arrays=[cat_a_test, cat_b_test, cat_c_test],
        categorical_names=['cat_a', 'cat_b', 'cat_c'],
        output_path=output_dir / "test_data.csv",
        task="regression"
    )

    # 5. Save metadata
    print("\n[5/5] Saving metadata...")
    metadata = {
        "format": "h2o_mojo",
        "task": "regression",
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_trees": N_TREES,
        "max_depth": MAX_DEPTH,
        "n_numeric_features": N_NUMERIC,
        "n_categorical_features": 3,
        "categorical_features": [
            {"name": "cat_a", "levels": sorted(CATEGORICAL_CONFIGS[0]['levels'])},
            {"name": "cat_b", "levels": sorted(CATEGORICAL_CONFIGS[1]['levels'])},
            {"name": "cat_c", "levels": sorted(CATEGORICAL_CONFIGS[2]['levels'])},
        ],
        "nan_pct": NAN_PCT,
        "seed": SEED,
        "distribution": "gaussian",
        "link_function": "identity",
        "notes": "This model tests categorical bitset splits. Branch direction: value IN bitset → RIGHT child"
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Metadata saved: {output_dir / 'metadata.json'}")

    # Cleanup
    h2o.cluster().shutdown(prompt=False)

    print("\n" + "=" * 70)
    print("✓ SUCCESS! Generated files:")
    print(f"   - {output_dir / 'model.zip'} (with categorical bitsets)")
    print(f"   - {output_dir / 'test_data.csv'}")
    print(f"   - {output_dir / 'metadata.json'}")
    print("\nIMPORTANT: This model tests the categorical branch direction fix:")
    print("   Value IN bitset → RIGHT child (NOT left)")
    print("=" * 70)


if __name__ == "__main__":
    main()
