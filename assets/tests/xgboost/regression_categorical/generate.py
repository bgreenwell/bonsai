#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["xgboost", "numpy", "pandas", "scikit-learn"]
# ///
"""Generate XGBoost regression test case with native categorical splits.

Trains with enable_categorical=True so the JSON contains split_type=1
nodes with category bitsets. The test CSV stores the category code as a
plain float feature, which is how the generated Rust code consumes it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 42
N_SAMPLES = 600
N_TEST = 50
N_TREES = 15
MAX_DEPTH = 4
N_LEVELS = 6


def main():
    print("=" * 70)
    print("XGBoost - Regression (Native Categorical Splits)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Generating synthetic data...")
    rng = np.random.default_rng(SEED)
    cat = rng.integers(0, N_LEVELS, N_SAMPLES)
    num = rng.normal(size=N_SAMPLES)
    y = (cat % 2) * 3.0 + (cat == 4) * 1.5 + 0.5 * num + rng.normal(0, 0.1, N_SAMPLES)

    df = pd.DataFrame(
        {
            "feat_0": pd.Categorical(cat, categories=range(N_LEVELS)),
            "feat_1": num,
        }
    )
    train_df, test_df = df.iloc[N_TEST:], df.iloc[:N_TEST]
    y_train = y[N_TEST:]

    print("\n[2/4] Training XGBoost regressor with enable_categorical...")
    dtrain = xgb.DMatrix(train_df, label=y_train, enable_categorical=True)
    model = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": MAX_DEPTH,
            "tree_method": "hist",
            "seed": SEED,
        },
        dtrain,
        num_boost_round=N_TREES,
    )

    print("\n[3/4] Exporting JSON model...")
    model.save_model(str(output_dir / "model.json"))

    print("\n[4/4] Generating ground truth predictions...")
    ground_truth = model.predict(xgb.DMatrix(test_df, enable_categorical=True))

    with open(output_dir / "test_data.csv", "w") as f:
        f.write("feat_0,feat_1,ground_truth\n")
        for row, gt in zip(test_df.itertuples(), ground_truth):
            f.write(f"{float(row.feat_0)!r},{float(row.feat_1)!r},{float(gt)!r}\n")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "format": "xgboost",
                "task": "regression",
                "n_trees": N_TREES,
                "max_depth": MAX_DEPTH,
                "n_numeric_features": 1,
                "n_categorical_features": 1,
                "categorical_levels": N_LEVELS,
                "seed": SEED,
                "objective": "reg:squarederror",
                "enable_categorical": True,
            },
            f,
            indent=2,
        )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()
