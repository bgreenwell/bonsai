# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-27

### Added
- Initial release of the bonsai tree ensemble transpiler.
- Support for H2O-3 MOJO, ONNX, XGBoost, LightGBM, and CatBoost models.
- Native categorical support for CatBoost (CTR) and H2O/ONNX (Bitsets).
- Oblivious tree optimization for branchless inference.
- High-throughput Batch API (`predict_batch`).
- Parallel batch scoring CLI (`polars_score`) with multiclass support.
- Model inspection and structural analysis tool (`bonsai inspect`).
- Array code layout (`--layout`) for large forests, auto-selected above 10k nodes.
- Reproducible CatBoost fixture generation scripts (`assets/tests/catboost/*/generate.py`).
- CI job that regenerates fixtures and runs XGBoost, LightGBM, CatBoost, and sklearn ONNX integration tests.
- XGBoost DART booster support via weight_drop tree scaling.
- Log-link objectives (Poisson, gamma, Tweedie) for XGBoost and LightGBM.
- Ranking objectives (XGBoost rank:*, LightGBM lambdarank/xendcg) emit raw scores.
- Clear errors for unsupported models: XGBoost gblinear, native categorical splits, multi-output; LightGBM multiclassova and cross_entropy_lambda.

### Fixed
- Oblivious-tree fast path returned depth-reversed leaves for trees deeper than one level.
- Transpiler panicked on non-finite split thresholds and leaf values.
- CatBoost multiclass leaf values were read in the wrong order for trees deeper than one level.
- CatBoost loss function was not detected in newer JSON exports.
- LightGBM Poisson/gamma/Tweedie models were missing the exp() output transform.
- XGBoost DART models were scored without weight_drop scaling.
