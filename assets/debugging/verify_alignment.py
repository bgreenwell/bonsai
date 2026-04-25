#!/usr/bin/env python3
"""
Verify that categorical values align with numeric features in test data.
"""

import pandas as pd
import numpy as np

# Reproduce the data generation EXACTLY as in train_and_export.py
SEED = 42
np.random.seed(SEED)

from sklearn.datasets import make_regression
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=10.0,
    random_state=SEED,
    shuffle=True,
)

# Inject NaN values
mask = np.random.random(X.shape) < 0.05
X[mask] = np.nan

# Add categorical features
np.random.seed(SEED)
cat_a = np.random.choice(['red', 'green', 'blue', 'yellow'], size=1000)
cat_b = np.random.choice(['small', 'medium', 'large'], size=1000)
cat_c = np.random.choice(['A', 'B'], size=1000)

# Adjust target
y = y + (cat_a == 'red') * 50.0 - (cat_a == 'blue') * 30.0
y = y + (cat_b == 'large') * 40.0
y = y + (cat_c == 'B') * 25.0

# Split using indices
indices = np.arange(len(X))
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(indices, test_size=100, random_state=SEED)

X_test = X[test_idx]
y_test = y[test_idx]
cat_a_test = cat_a[test_idx]
cat_b_test = cat_b[test_idx]
cat_c_test = cat_c[test_idx]

# Load the actual test data
test_df = pd.read_csv("test_data.csv")

print("Verifying first 5 rows alignment:")
print("=" * 80)
for i in range(5):
    print(f"\nRow {i}:")
    print(f"  Expected target: {y_test[i]:.4f}")
    print(f"  Actual target:   {test_df.iloc[i]['target']:.4f}")
    print(f"  Match: {abs(y_test[i] - test_df.iloc[i]['target']) < 0.001}")

    print(f"  Expected cat_a: {cat_a_test[i]}")
    print(f"  Actual cat_a:   {test_df.iloc[i]['cat_a']}")
    print(f"  Match: {cat_a_test[i] == test_df.iloc[i]['cat_a']}")

    print(f"  Expected feat_0: {X_test[i, 0]:.6f}" + (" (NaN)" if np.isnan(X_test[i, 0]) else ""))
    print(f"  Actual feat_0:   {test_df.iloc[i]['feat_0']:.6f}" + (" (NaN)" if pd.isna(test_df.iloc[i]['feat_0']) else ""))

print("\n" + "=" * 80)
if np.allclose(y_test, test_df['target'].values, rtol=1e-5, equal_nan=True):
    print("✓ All target values match!")
else:
    print("✗ Target values don't match")

if np.all(cat_a_test == test_df['cat_a'].values):
    print("✓ All cat_a values match!")
else:
    print("✗ cat_a values don't match")
