"""
Shared utilities for validating bonsai transpilation and predictions.

Provides validation functions for test cases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_test_data(csv_path: Path, task: str = "regression") -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load test data from CSV.

    Args:
        csv_path: Path to test_data.csv
        task: "regression" or "classification"

    Returns:
        Tuple of (features_df, ground_truth)
    """
    df = pd.DataFrame(pd.read_csv(csv_path))

    # Separate ground truth from features
    if task == "classification":
        ground_truth = df['ground_truth_proba'].values
        features_df = df.drop(columns=['target', 'ground_truth_proba'])
    else:
        ground_truth = df['ground_truth'].values
        features_df = df.drop(columns=['target', 'ground_truth'])

    return features_df, ground_truth


def validate_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    tolerance: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    Validate that predictions match ground truth within tolerance.

    Args:
        predictions: Predictions from bonsai-generated code
        ground_truth: Ground truth predictions from framework
        tolerance: Maximum allowed absolute difference
        verbose: Print detailed results

    Returns:
        True if all predictions match within tolerance
    """
    # Handle NaN values in ground truth
    valid_mask = ~np.isnan(ground_truth)
    predictions_valid = predictions[valid_mask]
    ground_truth_valid = ground_truth[valid_mask]

    # Calculate errors
    abs_errors = np.abs(predictions_valid - ground_truth_valid)
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)
    mismatches = np.sum(abs_errors > tolerance)

    if verbose:
        print("=" * 70)
        print("Prediction Validation Results")
        print("=" * 70)
        print(f"Samples tested:  {len(predictions_valid)}")
        print(f"Max error:       {max_error:.2e}")
        print(f"Mean error:      {mean_error:.2e}")
        print(f"Mismatches:      {mismatches} / {len(predictions_valid)} (tolerance={tolerance:.0e})")
        print()

        if mismatches == 0:
            print("✓ SUCCESS: All predictions match within tolerance!")
        else:
            print("✗ FAILURE: Predictions do not match!")
            print("\nFirst 5 mismatches:")
            mismatch_indices = np.where(abs_errors > tolerance)[0]
            for i in mismatch_indices[:5]:
                print(f"  Row {i}: predicted={predictions_valid[i]:.6f}, "
                      f"expected={ground_truth_valid[i]:.6f}, "
                      f"error={abs_errors[i]:.2e}")

        print("=" * 70)

    return mismatches == 0


def validate_model_structure(
    rust_code: str,
    expected_trees: int,
    has_categorical: bool,
    task: str = "regression",
    verbose: bool = True,
) -> bool:
    """
    Validate structure of generated Rust code.

    Args:
        rust_code: Generated Rust source code
        expected_trees: Expected number of trees
        has_categorical: Whether categorical features are present
        task: "regression" or "classification"
        verbose: Print detailed results

    Returns:
        True if code structure is valid
    """
    issues = []

    # Count tree functions
    tree_count = rust_code.count("fn tree_")
    if tree_count != expected_trees:
        issues.append(f"Expected {expected_trees} trees, found {tree_count}")

    # Check bitset_contains helper
    has_bitset_helper = "fn bitset_contains" in rust_code
    if has_categorical and not has_bitset_helper:
        issues.append("Expected bitset_contains() helper for categorical features, but not found")
    if not has_categorical and has_bitset_helper:
        issues.append("Found bitset_contains() helper but no categorical features")

    # Check post-transform
    if task == "classification":
        has_logit = "1.0f64 / (1.0f64 + (-raw).exp())" in rust_code or "1.0 / (1.0 + (-raw).exp())" in rust_code
        if not has_logit:
            issues.append("Expected logit post-transform for classification")
    else:
        has_identity = "raw as f32" in rust_code
        if not has_identity:
            issues.append("Expected identity post-transform for regression")

    # Check Model struct
    if "pub struct Model" not in rust_code:
        issues.append("Missing 'pub struct Model'")

    if "pub fn predict(&self, features: &[f32]) -> f32" not in rust_code:
        issues.append("Missing 'pub fn predict()' method")

    if verbose:
        print("=" * 70)
        print("Code Structure Validation")
        print("=" * 70)
        print(f"Tree functions: {tree_count} (expected {expected_trees})")
        print(f"Bitset helper:  {has_bitset_helper} (expected {has_categorical})")
        print(f"Task type:      {task}")
        print()

        if len(issues) == 0:
            print("✓ SUCCESS: Code structure is valid!")
        else:
            print("✗ FAILURE: Code structure issues:")
            for issue in issues:
                print(f"  - {issue}")

        print("=" * 70)

    return len(issues) == 0


def validate_edge_cases(
    features_df: pd.DataFrame,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    verbose: bool = True,
) -> bool:
    """
    Validate edge case handling (NaN values, categorical edge cases).

    Args:
        features_df: Test features DataFrame
        predictions: Predictions from bonsai
        ground_truth: Ground truth predictions
        verbose: Print detailed results

    Returns:
        True if edge cases handled correctly
    """
    issues = []

    # Find rows with all NaN numeric features
    numeric_cols = [col for col in features_df.columns if col.startswith('feat_')]
    all_nan_mask = features_df[numeric_cols].isnull().all(axis=1)

    if all_nan_mask.any():
        all_nan_indices = np.where(all_nan_mask)[0]
        all_nan_errors = np.abs(predictions[all_nan_indices] - ground_truth[all_nan_indices])

        if np.any(all_nan_errors > 1e-5):
            issues.append(f"Rows with all NaN features have prediction errors: max={np.max(all_nan_errors):.2e}")

    # Find rows with many NaN values
    nan_counts = features_df[numeric_cols].isnull().sum(axis=1)
    high_nan_mask = nan_counts >= len(numeric_cols) // 2

    if high_nan_mask.any():
        high_nan_indices = np.where(high_nan_mask)[0]
        high_nan_errors = np.abs(predictions[high_nan_indices] - ground_truth[high_nan_indices])

        if np.any(high_nan_errors > 1e-5):
            issues.append(f"Rows with high NaN count have prediction errors: max={np.max(high_nan_errors):.2e}")

    if verbose:
        print("=" * 70)
        print("Edge Case Validation")
        print("=" * 70)
        print(f"Rows with all NaN: {all_nan_mask.sum()}")
        print(f"Rows with >50% NaN: {high_nan_mask.sum()}")
        print()

        if len(issues) == 0:
            print("✓ SUCCESS: Edge cases handled correctly!")
        else:
            print("✗ FAILURE: Edge case issues:")
            for issue in issues:
                print(f"  - {issue}")

        print("=" * 70)

    return len(issues) == 0
