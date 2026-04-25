#!/usr/bin/env python3
"""
Compare H2O vs bonsai prediction on a single row in detail.
"""

import pandas as pd
import h2o
import subprocess
import numpy as np

h2o.init()

# Load test data
test_df = pd.read_csv("test_data.csv")

# Get first row
row = test_df.iloc[0]

print("=" * 80)
print("Comparing predictions for Row 0:")
print("=" * 80)
print("\nInput features:")
for i in range(10):
    val = row[f'feat_{i}']
    print(f"  feat_{i}: {val:10.6f}" + (" (NaN)" if pd.isna(val) else ""))
print(f"  cat_a: {row['cat_a']} (encoded as 0)")
print(f"  cat_b: {row['cat_b']} (encoded as 0)")
print(f"  cat_c: {row['cat_c']} (encoded as 1)")

# H2O prediction
from h2o.estimators import H2OGenericEstimator
model = H2OGenericEstimator.from_file("h2o_gbm_regression.zip")

# Create H2O frame
input_cols = [f'feat_{i}' for i in range(10)] + ['cat_a', 'cat_b', 'cat_c']
h2o_frame = h2o.H2OFrame(test_df.iloc[[0]][input_cols])
h2o_frame["cat_a"] = h2o_frame["cat_a"].asfactor()
h2o_frame["cat_b"] = h2o_frame["cat_b"].asfactor()
h2o_frame["cat_c"] = h2o_frame["cat_c"].asfactor()

h2o_pred = model.predict(h2o_frame)[0, 0]

# Bonsai prediction
# Build input line: numeric features + encoded categoricals
numeric_feats = [str(row[f'feat_{i}']) for i in range(10)]
cat_encoded = ['0.0', '0.0', '1.0']  # blue, large, B
input_line = ','.join(numeric_feats + cat_encoded)

result = subprocess.run(
    ['/tmp/test_categorical'],
    input=input_line,
    capture_output=True,
    text=True
)
bonsai_pred = float(result.stdout.strip())

print("\n" + "=" * 80)
print("Predictions:")
print("=" * 80)
print(f"  H2O (from MOJO):        {h2o_pred:.10f}")
print(f"  H2O (stored in CSV):    {row['h2o_prediction']:.10f}")
print(f"  Bonsai (from Rust):     {bonsai_pred:.10f}")
print(f"\n  Difference (bonsai - h2o): {bonsai_pred - h2o_pred:.10f}")
print("=" * 80)

h2o.cluster().shutdown(prompt=False)
