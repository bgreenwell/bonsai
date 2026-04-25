# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- H2O-3 MOJO parser for GBM models (binary classification and regression)
- ONNX TreeEnsemble parser for generic tree models
- Rust code generator backend with zero runtime dependencies
- Support for numeric and categorical splits
- Missing value handling (NaLeft, NaRight, NaVsRest)
- Binary classification with logit post-transform
- Regression with identity and log post-transforms
- H2O-3 example with validation against ground truth
- Batch scoring binary using Polars + Rayon

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
