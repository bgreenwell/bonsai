#!/bin/bash
# Master script to generate all test fixtures for bonsai.
# Requires Python 3 with h2o, scikit-learn, skl2onnx, xgboost, lightgbm.

# Base directory for tests
BASE_DIR="assets/tests"

# Find all generate.py files and run them
# We use a subshell to avoid changing the current directory permanently
find "$BASE_DIR" -name "generate.py" | while read -r script; do
    echo "--------------------------------------------------------"
    echo "🚀 Running: $script"
    dir=$(dirname "$script")
    (cd "$dir" && python3 generate.py)
done

echo "--------------------------------------------------------"
echo "✅ All fixtures generated successfully!"
