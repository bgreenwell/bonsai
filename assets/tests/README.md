# Bonsai Test Suite

Comprehensive integration tests for validating bonsai model transpilation across different formats and feature types.

## Structure

```
tests/
├── common/                     # Shared utilities
│   ├── generators.py          # Data generation functions
│   ├── validators.py          # Prediction validation functions
│   └── test_harness.rs.template  # Rust test harness template
│
├── h2o_mojo/                  # H2O MOJO format tests
│   ├── classification_numeric/
│   ├── classification_categorical/
│   ├── regression_numeric/
│   └── regression_categorical/  # IMPORTANT: Tests categorical bitset fix
│
└── sklearn_onnx/              # sklearn → ONNX tests
    ├── classification_numeric/
    ├── classification_categorical/
    ├── regression_numeric/
    └── regression_categorical/
```

## Test Matrix

| Format | Task | Features | Status |
|--------|------|----------|--------|
| H2O MOJO | Classification | Numeric only | ✅ Implemented |
| H2O MOJO | Classification | + Categorical | 🔲 Template ready |
| H2O MOJO | Regression | Numeric only | 🔲 Template ready |
| H2O MOJO | Regression | + Categorical | ✅ Implemented |
| sklearn ONNX | Classification | Numeric only | 🔲 Template ready |
| sklearn ONNX | Classification | + Categorical | 🔲 Template ready |
| sklearn ONNX | Regression | Numeric only | 🔲 Template ready |
| sklearn ONNX | Regression | + Categorical | 🔲 Template ready |
| XGBoost | Classification | Numeric only | ✅ Implemented |
| XGBoost | Regression | Numeric only | ✅ Implemented |
| LightGBM | Classification | Numeric only | ✅ Implemented |
| LightGBM | Regression | Numeric only | ✅ Implemented |

## Running Tests

### Generate Test Models and Data

Each test directory has a `generate.py` script that:
1. Generates synthetic data with specific characteristics
2. Trains the model (H2O or sklearn)
3. Exports to the target format (MOJO or ONNX)
4. Creates test data with ground truth predictions
5. Saves metadata about the test case

```bash
# Example: Generate H2O MOJO regression test with categorical features
cd h2o_mojo/regression_categorical
python generate.py

# This creates:
#   generated/model.zip         - H2O MOJO model
#   generated/test_data.csv    - Test features + ground truth
#   generated/metadata.json    - Model metadata
```

### Run Cargo Integration Tests

```bash
# From project root
cargo test --test integration_test -- --ignored
```

## Test Configuration

All tests use consistent configuration for reproducibility:

- **Samples:** 1000 total (900 train, 100 test)
- **Trees:** 50
- **Max Depth:** 5
- **Random Seed:** 42
- **NaN Percentage:** 5% of numeric cells
- **Categorical Levels:**
  - cat_a: 4 levels (blue, green, red, yellow)
  - cat_b: 3 levels (large, medium, small)
  - cat_c: 2 levels (A, B)

## Validation Criteria

Each test validates:

1. **Prediction Accuracy**
   - bonsai predictions match framework ground truth within 1e-5 tolerance
   - All 100 test samples must pass

2. **Code Structure**
   - Correct number of tree functions generated
   - `bitset_contains()` helper present ONLY when categorical features exist
   - Proper post-transform (logit for classification, identity for regression)

3. **Edge Cases**
   - Rows with all NaN values
   - Rows with high NaN percentage (>50%)
   - Sparse categorical distributions (one level dominates)

## Important Notes

### Categorical Features in MOJO vs ONNX

**H2O MOJO:**
- Categorical features stored natively as bitsets
- Uses `SplitKind::Categorical` in IR
- Generates `bitset_contains()` helper function
- **Critical:** Value IN bitset → follow RIGHT child (not left!)

**sklearn ONNX:**
- Categorical features label-encoded to numeric (blue=0, green=1, etc.)
- Uses standard `SplitKind::Numeric` threshold comparisons
- No special categorical handling needed

### Debugging Failed Tests

If a test fails:

1. Check `generated/metadata.json` for test configuration
2. Compare first few predictions manually
3. Use debugging utilities in `../debugging/`
4. Verify MOJO/ONNX model was generated correctly
5. Check that bonsai version includes categorical fix (Feb 2026)

## Adding New Test Cases

To add a new test case:

1. Create directory: `format/task_features/`
2. Copy `generate.py` from similar test
3. Update configuration (features, categorical levels, etc.)
4. Run `python generate.py` to verify it works
5. Add test to `tests/integration_test.rs`
6. Update this README's test matrix

## Requirements

Each format has its own `requirements.txt`:

- `h2o_mojo/requirements.txt` - h2o, numpy, pandas, scikit-learn
- `sklearn_onnx/requirements.txt` - scikit-learn, skl2onnx, numpy, pandas

Install with:
```bash
pip install -r h2o_mojo/requirements.txt
```

## Common Utilities

The `common/` directory provides shared functions:

- `generators.py`:
  - `make_synthetic_data()` - Generate classification/regression data
  - `add_categorical_features()` - Add categorical features with effects on target
  - `inject_nans()` - Randomly inject missing values
  - `save_test_data()` - Save test CSV with ground truth

- `validators.py`:
  - `validate_predictions()` - Compare bonsai vs ground truth
  - `validate_model_structure()` - Check generated code structure
  - `validate_edge_cases()` - Test NaN handling, edge cases

- `test_harness.rs.template`:
  - Template for Rust test harnesses that load generated models
  - Handles CSV input, NaN values, prediction output
V input, NaN values, prediction output
