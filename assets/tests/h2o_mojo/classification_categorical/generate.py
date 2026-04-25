#!/usr/bin/env python3
"""Generate H2O MOJO classification test case WITH categorical features."""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
from generators import make_synthetic_data, add_categorical_features, inject_nans, save_test_data

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 1000
N_TEST = 100
N_TREES = 50
MAX_DEPTH = 5
N_NUMERIC = 7
NAN_PCT = 0.05

CATEGORICAL_CONFIGS = [
    {'levels': ['blue', 'green', 'red', 'yellow'], 'effect': 0.3},
    {'levels': ['large', 'medium', 'small'], 'effect': 0.2},
    {'levels': ['A', 'B'], 'effect': 0.15},
]

def main():
    print("=" * 70)
    print("H2O MOJO - Classification (WITH Categorical Features)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    # Generate data
    print("\n[1/5] Generating data with categoricals...")
    X, y = make_synthetic_data(N_SAMPLES, N_NUMERIC, task="classification", seed=SEED)
    categorical_arrays, y = add_categorical_features(X, y, CATEGORICAL_CONFIGS, seed=SEED)
    cat_a, cat_b, cat_c = categorical_arrays

    X = inject_nans(X, NAN_PCT, seed=SEED)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=N_TEST, random_state=SEED, stratify=y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    cat_a_train, cat_a_test = cat_a[train_idx], cat_a[test_idx]
    cat_b_train, cat_b_test = cat_b[train_idx], cat_b[test_idx]
    cat_c_train, cat_c_test = cat_c[train_idx], cat_c[test_idx]

    # Train H2O
    print("\n[2/5] Training H2O GBM...")
    h2o.init()

    train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    train_df['cat_a'] = cat_a_train
    train_df['cat_b'] = cat_b_train
    train_df['cat_c'] = cat_c_train
    train_df['target'] = y_train
    h2o_train = h2o.H2OFrame(train_df)
    h2o_train['cat_a'] = h2o_train['cat_a'].asfactor()
    h2o_train['cat_b'] = h2o_train['cat_b'].asfactor()
    h2o_train['cat_c'] = h2o_train['cat_c'].asfactor()
    h2o_train['target'] = h2o_train['target'].asfactor()

    test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(N_NUMERIC)])
    test_df['cat_a'] = cat_a_test
    test_df['cat_b'] = cat_b_test
    test_df['cat_c'] = cat_c_test
    test_df['target'] = y_test
    h2o_test = h2o.H2OFrame(test_df)
    h2o_test['cat_a'] = h2o_test['cat_a'].asfactor()
    h2o_test['cat_b'] = h2o_test['cat_b'].asfactor()
    h2o_test['cat_c'] = h2o_test['cat_c'].asfactor()
    h2o_test['target'] = h2o_test['target'].asfactor()

    gbm = H2OGradientBoostingEstimator(
        ntrees=N_TREES, max_depth=MAX_DEPTH, distribution="bernoulli", seed=SEED
    )
    feature_cols = [f"feat_{i}" for i in range(N_NUMERIC)] + ['cat_a', 'cat_b', 'cat_c']
    gbm.train(x=feature_cols, y="target", training_frame=h2o_train)

    # Export
    print("\n[3/5] Exporting MOJO...")
    mojo_path = gbm.download_mojo(path=str(output_dir), get_genmodel_jar=False)
    os.rename(mojo_path, output_dir / "model.zip")

    # Save test data
    print("\n[4/5] Generating ground truth...")
    h2o_preds = gbm.predict(h2o_test)
    ground_truth = h2o_preds[h2o_preds.columns[2]].as_data_frame().values.flatten()
    save_test_data(X_test, y_test, ground_truth, [cat_a_test, cat_b_test, cat_c_test],
                   ['cat_a', 'cat_b', 'cat_c'], output_dir / "test_data.csv", "classification")

    # Metadata
    print("\n[5/5] Saving metadata...")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "format": "h2o_mojo", "task": "classification",
            "n_trees": N_TREES, "n_numeric_features": N_NUMERIC,
            "n_categorical_features": 3, "seed": SEED
        }, f, indent=2)

    h2o.cluster().shutdown(prompt=False)
    print("\n✓ SUCCESS!")

if __name__ == "__main__":
    main()
