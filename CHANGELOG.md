# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **XGBoost JSON frontend** (`src/frontends/xgboost.rs`)
  - Parses XGBoost native JSON models saved via `model.save_model("*.json")`
  - Supports `reg:squarederror` (Identity), `binary:logistic` / `reg:logistic` (Logit), and `multi:softmax` / `multi:softprob` (Softmax) objectives
  - Handles NaN routing via `default_left` parallel arrays
  - Applies logit transform to `base_score` for pre-1.6 models that stored it in probability space
  - Auto-detected from `.json` extension by checking for top-level `"learner"` key
  - 5 unit tests with inline JSON fixtures
  - Integration test stubs: `test_xgboost_regression_numeric`, `test_xgboost_classification_numeric` (run with `--include-ignored` after `generate.py`)
- **LightGBM JSON frontend** (`src/frontends/lightgbm.rs`)
  - Parses LightGBM models exported via `Booster.dump_model()` (JSON format)
  - Supports `regression` (Identity), `binary` (Logit), and `multiclass` / `softmax` (Softmax) objectives
  - Handles the recursive `tree_structure` format; accepts threshold as JSON string or number
  - NaN routing via `default_left` boolean per node
  - Auto-detected from `.json` extension by checking for top-level `"tree_info"` key
  - 5 unit tests with inline JSON fixtures
  - Integration test stubs: `test_lightgbm_regression_numeric`, `test_lightgbm_classification_numeric`
- **Multiclass classification (softmax)** — new `PostTransform::Softmax { n_classes }` IR variant
  - Backend generates `predict_proba(&self, features: &[f32]) -> Vec<f32>` for multiclass models
  - Trees are round-robin assigned to classes: tree `i` → class `i % n_classes`
  - Numerically-stable softmax (subtract max before exp)
  - ONNX frontend now detects `post_transform = "SOFTMAX"` / `"SOFTMAX_ZERO"` with `n_classes > 2` and extracts per-class leaf weights correctly
  - 5 new unit tests in `src/backends/rust.rs`
- **GitHub Actions CI** (`.github/workflows/ci.yml`)
  - `test` job: `cargo build --release` + `cargo test` (unit tests)
  - `lint` job: `cargo fmt --check` + `cargo clippy -D warnings`
  - Runs on push/PR to `main`; integration tests remain `#[ignore]`

### Changed
- **Integration test validation is now real** — `tests/integration_test.rs` previously skipped numeric comparison entirely (placeholder stub); it now compiles the generated `model.rs` with `rustc` at test time, pipes feature rows through stdin, and asserts predictions match ground truth within 1e-5
  - Categorical string values in `test_data.csv` are converted to integer indices using level mappings from `metadata.json`
  - Uses `env!("CARGO_BIN_EXE_bonsai")` instead of `cargo run` for speed
  - sklearn ONNX `generate.py` scripts for categorical tests now write `categorical_features` levels to `metadata.json`
  - Removed stale `tests/h2o3_integration.rs` (referenced old `assets/examples/` paths)
- Fixed `assets/tests/common/test_harness.rs.template`: `{{:.10}}` → `{:.10}` (Python-style escaping was producing invalid Rust format string)
- Added `serde_json` to dev-dependencies for metadata parsing in integration tests

### Documentation
- `AGENTS.md` updated to reflect actual test state: unit test counts per file (`ir.rs` ~19, `tree_parser.rs` ~18, `rust.rs` 5), correct integration test paths (`assets/tests/` not `assets/examples/`), and a note that prediction validation was a stub (now fixed)

### Fixed
- **[CRITICAL]** Categorical split branch direction in H2O MOJO models (2025-02-26)
  - **Issue:** Predictions from bonsai-generated code didn't match H2O ground truth for models with categorical features (errors of 50+ units)
  - **Root Cause:** MOJO categorical splits use bitset membership tests with inverted branch semantics. When a categorical value is IN the bitset, H2O follows the RIGHT child (not left). The generated Rust code was sending values in the bitset to the LEFT child, causing systematic prediction errors.
  - **Fix Location:** `src/backends/rust.rs:168-185` - Negated `bitset_contains()` test for the left branch condition
  - **Impact:** All categorical predictions now match H2O ground truth within f32 precision (1e-5). Affects only MOJO models with categorical features; ONNX models use label encoding (numeric) and were unaffected.
  - **Files Changed:**
    - `src/backends/rust.rs`
  - **Testing:** Added comprehensive regression test with 3 categorical features (4, 3, and 2 levels) in `assets/tests/h2o_mojo/regression_categorical/`

- **[CRITICAL]** MOJO parser misalignment on NaVsRest nodes (2025-02-06)
  - **Issue:** Parser failed on binomial GBM models with "failed to fill whole buffer" error
  - **Root Cause:** NaVsRest split nodes (which split solely on NaN-ness) don't store a numeric threshold in MOJO binary format. Parser was incorrectly reading 4 bytes for a split_value that doesn't exist, causing buffer misalignment.
  - **Fix Location 1:** `src/parsers/tree_parser.rs:193-196` - Skip reading split_value for NaVsRest nodes, use f32::NAN placeholder
  - **Fix Location 2:** `src/backends/rust.rs:140-144` - Generate `!val.is_nan()` condition without threshold comparison for NaVsRest nodes
  - **Impact:** MOJO parser now correctly handles binomial GBM models. Predictions match H2O-3 ground truth within 1e-7 (f32 precision)
  - **Files Changed:**
    - `src/parsers/tree_parser.rs`
    - `src/backends/rust.rs`

## [0.1.0] - 2025-02-05

### Added
- Initial project structure merging h2o-poet and transmute
- Single-crate architecture with frontends/backends/parsers
- CLI interface with `--input` and `--output` flags
- Integration test suite for H2O-3 models

[Unreleased]: https://github.com/yourusername/bonsai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/bonsai/releases/tag/v0.1.0
