"""
Shared utilities for generating synthetic test data and models.

Provides consistent data generation across all test cases.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def make_synthetic_data(
    n_samples: int,
    n_numeric: int,
    n_informative: int = None,
    noise: float = 10.0,
    task: str = "regression",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing.

    Args:
        n_samples: Number of samples to generate
        n_numeric: Number of numeric features
        n_informative: Number of informative features (default: n_numeric - 2)
        noise: Noise level for data generation
        task: "regression" or "classification"
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if n_informative is None:
        n_informative = max(1, n_numeric - 2)

    np.random.seed(seed)

    if task == "regression":
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_numeric,
            n_informative=n_informative,
            noise=noise,
            random_state=seed,
            shuffle=True,
        )
    else:  # classification
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_numeric,
            n_informative=n_informative,
            n_redundant=max(0, n_numeric - n_informative - 2),
            n_clusters_per_class=2,
            flip_y=0.01,
            random_state=seed,
            shuffle=True,
        )

    return X, y


def add_categorical_features(
    X: np.ndarray,
    y: np.ndarray,
    categorical_configs: List[Dict],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Add categorical features to the dataset.

    Args:
        X: Existing numeric features
        y: Target variable
        categorical_configs: List of dicts with 'levels' and 'effect' keys
            Example: [{'levels': ['A', 'B', 'C'], 'effect': 10.0}, ...]
        seed: Random seed

    Returns:
        Tuple of (cat_features, adjusted_y, categorical_arrays)
        where categorical_arrays is list of numpy arrays with categorical values
    """
    np.random.seed(seed)
    n_samples = len(X)

    categorical_arrays = []

    for i, config in enumerate(categorical_configs):
        levels = config['levels']
        effect = config.get('effect', 0.0)

        # Generate categorical feature
        # Make one category sparse (90% one value) to test edge cases
        if i == 0 and len(levels) > 2:
            # First categorical: 90% first level, 10% others
            cat_feature = np.random.choice(
                levels,
                size=n_samples,
                p=[0.9] + [0.1 / (len(levels) - 1)] * (len(levels) - 1)
            )
        else:
            # Other categoricals: uniform distribution
            cat_feature = np.random.choice(levels, size=n_samples)

        categorical_arrays.append(cat_feature)

        # Add effect to target (cast to float to support both regression and classification)
        y = y.astype(float)
        for j, level in enumerate(levels):
            mask = cat_feature == level
            if j == 0:
                y[mask] += effect
            elif j == len(levels) - 1:
                y[mask] -= effect

    return categorical_arrays, y


def inject_nans(X: np.ndarray, pct: float, seed: int = 42) -> np.ndarray:
    """
    Inject NaN values into numeric features.

    Args:
        X: Feature array
        pct: Percentage of values to make NaN (0.0 to 1.0)
        seed: Random seed

    Returns:
        Modified feature array with NaNs
    """
    np.random.seed(seed)
    X_copy = X.copy()
    mask = np.random.random(X_copy.shape) < pct
    X_copy[mask] = np.nan
    return X_copy


def save_test_data(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    categorical_arrays: Optional[List[np.ndarray]],
    categorical_names: Optional[List[str]],
    output_path: Path,
    task: str = "regression",
) -> None:
    """
    Save test data with ground truth predictions to CSV.

    Args:
        X_test: Test features (numeric only)
        y_test: Test targets
        predictions: Ground truth predictions from the framework
        categorical_arrays: List of categorical feature arrays (for test set)
        categorical_names: Names of categorical features
        output_path: Path to save CSV file
        task: "regression" or "classification"
    """
    # Create DataFrame with numeric features
    n_numeric = X_test.shape[1]
    df = pd.DataFrame(
        X_test,
        columns=[f'feat_{i}' for i in range(n_numeric)]
    )

    # Add categorical features if present
    if categorical_arrays and categorical_names:
        for cat_arr, cat_name in zip(categorical_arrays, categorical_names):
            df[cat_name] = cat_arr

    # Add target and predictions
    df['target'] = y_test

    if task == "classification":
        df['ground_truth_proba'] = predictions
    else:
        df['ground_truth'] = predictions

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved test data to {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Numeric features: {n_numeric}")
    if categorical_arrays:
        print(f"  Categorical features: {len(categorical_arrays)}")
    print(f"  NaN values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.1f}%)")


def save_test_data_multiclass(
    X_test: np.ndarray,
    y_test: np.ndarray,
    proba_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save test data for multiclass softmax models.

    Args:
        X_test: Test features (numeric only)
        y_test: Integer class labels
        proba_matrix: Shape (n_samples, n_classes) — per-class probabilities
        output_path: Path to save CSV file
    """
    n_numeric = X_test.shape[1]
    n_classes = proba_matrix.shape[1]

    df = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(n_numeric)])
    df['target'] = y_test

    for c in range(n_classes):
        df[f'ground_truth_proba_{c}'] = proba_matrix[:, c]

    df.to_csv(output_path, index=False)
    print(f"✓ Saved multiclass test data to {output_path}")
    print(f"  Samples: {len(df)}, Features: {n_numeric}, Classes: {n_classes}")
