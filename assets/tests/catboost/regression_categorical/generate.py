#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["catboost", "numpy", "pandas", "scikit-learn"]
# ///
"""Generate CatBoost regression test case with a native categorical feature.

The integration test harness (`run_catboost_cat_test_case`) expects exactly
one float feature column `feature_0`, one categorical column `cat_feature`,
and a `ground_truth` column. `one_hot_max_size=0` forces CatBoost to use
Online CTR splits, which is the path bonsai implements.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool

SEED = 42
N_SAMPLES = 300
N_TEST = 30
N_TREES = 10
DEPTH = 3
LEVELS = ["A", "B", "C"]


def main():
    print("=" * 70)
    print("CatBoost - Regression (Native Categorical via Online CTR)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/4] Generating synthetic data...")
    rng = np.random.default_rng(SEED)
    x_num = rng.uniform(0.0, 1.0, N_SAMPLES)
    x_cat = rng.choice(LEVELS, N_SAMPLES)
    level_effect = {"A": 1.0, "B": 2.0, "C": 3.0}
    y = (
        np.array([level_effect[c] for c in x_cat])
        + 0.5 * x_num
        + rng.normal(0.0, 0.1, N_SAMPLES)
    )

    df = pd.DataFrame({"feature_0": x_num, "cat_feature": x_cat})
    train_df, test_df = df.iloc[N_TEST:], df.iloc[:N_TEST]
    y_train = y[N_TEST:]

    print("\n[2/4] Training CatBoost regressor with categorical feature...")
    train_pool = Pool(train_df, y_train, cat_features=["cat_feature"])
    model = CatBoostRegressor(
        iterations=N_TREES,
        depth=DEPTH,
        learning_rate=0.3,
        loss_function="RMSE",
        random_seed=SEED,
        one_hot_max_size=0,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(train_pool)

    print("\n[3/4] Exporting JSON model...")
    model.save_model(str(output_dir / "model.json"), format="json")

    print("\n[4/4] Generating ground truth predictions...")
    test_pool = Pool(test_df, cat_features=["cat_feature"])
    ground_truth = model.predict(test_pool)

    out = test_df.copy()
    out["ground_truth"] = ground_truth
    out.to_csv(output_dir / "test_data.csv", index=False)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "format": "catboost",
                "task": "regression",
                "n_trees": N_TREES,
                "depth": DEPTH,
                "categorical_features": [
                    {"name": "cat_feature", "levels": LEVELS}
                ],
                "seed": SEED,
                "loss_function": "RMSE",
                "one_hot_max_size": 0,
            },
            f,
            indent=2,
        )

    print("\nSUCCESS")


if __name__ == "__main__":
    main()
