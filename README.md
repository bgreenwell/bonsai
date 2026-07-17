# 🌱 bonsai

[![CI](https://img.shields.io/github/actions/workflow/status/bgreenwell/bonsai/ci.yml?style=for-the-badge)](https://github.com/bgreenwell/bonsai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-%232196F3.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-%23D34516.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)

Transpile trained tree ensemble models into standalone, dependency-free Rust code.

bonsai converts gradient boosting and random forest models from common training
frameworks into plain Rust source. The generated module has no runtime
dependencies and no model-loading step, which makes it a good fit for servers,
edge devices, and WASM targets where shipping an ML runtime is impractical.

## Supported formats

- H2O-3 MOJO (`.zip`)
- ONNX tree ensembles (`.onnx`, `.pb`)
- XGBoost native JSON (`booster.save_model("model.json")`)
- LightGBM JSON dump (`booster.dump_model()`)
- CatBoost JSON (`model.save_model("model.json", format="json")`), including
  native categorical (CTR) support and a branchless fast path for oblivious
  trees

## Installation

Not yet published to crates.io; build from source:

```bash
git clone https://github.com/bgreenwell/bonsai
cd bonsai
cargo install --path .
```

## Quick start

```bash
# Convert a model to Rust code
bonsai transpile --input model.json --output model.rs

# Verify the transpiled model against reference predictions:
# transpiles, compiles with rustc, scores the CSV, and diffs
bonsai verify --input model.json --data test.csv --tolerance 1e-5

# Same check without compiling, via the built-in IR interpreter
# (bit-identical to compiled output)
bonsai verify --input model.json --data test.csv --engine interpret

# Inspect a model's structure and statistics
bonsai inspect --input model.json
```

The verify CSV holds the feature columns plus a `ground_truth` column
(scalar models) or `ground_truth_proba_<c>` columns (multiclass).

To ship the model as a library instead of a bare source file, emit a full
cargo crate; with `--data` the reference predictions are baked in as a
`cargo test`-runnable golden test:

```bash
bonsai transpile --input model.json --output scorer/ --emit crate --data test.csv
```

Very large forests (above ~10k nodes) automatically use a flattened array
layout that keeps rustc compile times low; override with
`--layout {auto,ifelse,array}`.

For embedded targets, `--no-std` emits core-only code: softmax models expose
`predict_proba_into(features, &mut out)` instead of a `Vec`-returning
`predict_proba`, and models with a sigmoid/exp output transform call
`libm::exp`, so add `libm` to the consuming crate. CatBoost CTR models are
not supported in this mode.

The generated `model.rs` exposes a `Model` struct with `predict` (scalar),
`predict_batch` (high-throughput batch), `predict_proba` (classification
probabilities), and, for CatBoost models with categorical features,
`predict_cat`.

For an end-to-end walkthrough (train an XGBoost model in Python, transpile
it, and benchmark the result), see [`demo.ipynb`](demo.ipynb).

### Batch scoring CLI

An optional batch scorer built on Polars and Rayon:

```bash
cargo build --release --features scorer --bin polars_score
./target/release/polars_score --input data.parquet --output predictions.parquet
```

## Performance

Representative numbers from the included Criterion benchmarks (`benches/`),
using an XGBoost binary classifier:

- Single-row latency: ~137 ns per prediction, versus ~3.5 us for ONNX Runtime
  on the same model.
- Batch throughput: ~7.5M rows/sec; CatBoost oblivious trees evaluate
  branchlessly in batch mode.
- The generated inference code performs no heap allocations.

Results vary with model size and hardware; run `cargo bench --features bench-models` to reproduce (after generating the example model).

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - release history
- [AGENTS.md](AGENTS.md) - architecture map and development notes

## Development

```bash
# Quality gate
cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo test

# Integration tests require Python-generated model fixtures;
# see assets/tests/README.md
cargo test -- --include-ignored
```

## License

MIT
