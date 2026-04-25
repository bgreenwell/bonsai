# Debugging Utilities

Tools for debugging bonsai transpilation issues. These are development utilities, not part of the automated test suite.

## Available Tools

### compare_single_row.py
Compare H2O vs bonsai prediction on a single row in detail.

**Usage:**
```bash
python compare_single_row.py
```

Shows:
- Input feature values
- H2O prediction (from MOJO)
- H2O prediction (stored in CSV)
- Bonsai prediction (from Rust code)
- Difference between predictions

**When to use:** When you see systematic prediction mismatches and want to debug a specific row.

### verify_alignment.py
Verify that test data is correctly aligned with training data.

**Usage:**
```bash
python verify_alignment.py
```

Checks:
- Target values match expected
- Categorical values align with numeric features
- No data shuffling occurred during train/test split

**When to use:** When you suspect data alignment issues causing prediction mismatches.

### categorical_bitset_inspector.py
(Planned) Tool to inspect MOJO bitset encoding for categorical features.

**Planned features:**
- Extract bitset bytes from MOJO file
- Show which categorical levels are included in each bitset
- Visualize bit patterns
- Verify alphabetical ordering of levels

**When to use:** When debugging categorical split logic.

## Notes

- These scripts were used to debug the categorical branch direction bug (Feb 2026)
- They expect to be run from a test directory with `generated/` subdirectory
- Not all scripts are maintained - some may need updates to work with current code

## Related Issues

### Categorical Branch Direction Bug (Fixed Feb 2026)

**Symptom:** Predictions didn't match for models with categorical features

**Root cause:** Values IN bitset were going LEFT instead of RIGHT

**How these tools helped:**
- `compare_single_row.py` identified the specific prediction mismatch
- `verify_alignment.py` ruled out data alignment issues
- Manual inspection of bitsets revealed branch direction problem

**Fix:** `src/backends/rust.rs:168-185` - negated `bitset_contains()` for left branch
