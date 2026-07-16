# bonsai

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

# Inspect a model's structure and statistics
bonsai inspect --input model.json
```

Very large forests (above ~10k nodes) automatically use a flattened array
layout that keeps rustc compile times low; override with
`--layout {auto,ifelse,array}`.

The generated `model.rs` exposes a `Model` struct with `predict` (scalar),
`predict_batch` (high-throughput batch), `predict_proba` (classification
probabilities), and, for CatBoost models with categorical features,
`predict_cat`.

For an end-to-end walkthrough — train an XGBoost model in Python, transpile
it, and benchmark the result — see [`demo.ipynb`](demo.ipynb).

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

Results vary with model size and hardware; run `cargo bench` to reproduce.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — release history
- [PLAN.md](PLAN.md) — architecture notes and roadmap

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
