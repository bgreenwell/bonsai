# bonsai

Transpile machine learning tree ensemble models to standalone Rust code.

## Overview

**bonsai** converts trained tree-based models (Random Forests, Gradient Boosting Machines, etc.) from various ML frameworks into efficient, dependency-free Rust code. The generated code can be embedded directly into your application without requiring the original ML runtime.

## Supported Formats

- **H2O-3**: MOJO (native binary format, `.zip`)
- **ONNX**: Generic tree ensemble models from any framework (`.onnx`)
- **XGBoost**: Native JSON format (`booster.save_model("model.json")`)
- **LightGBM**: JSON dump format (`booster.dump_model()`)

## Usage

### Transpile a Model

```bash
# Convert a model to Rust code
bonsai transpile --input model.zip --output model.rs

# Or using cargo
cargo run -- transpile --input model.zip --output model.rs

# Use the generated model
rustc model.rs -o predictor
./predictor < test_data.csv
```

### Inspect a Model

```bash
# View model structure and statistics
bonsai inspect model.zip

# Show detailed tree structures (first 3 trees)
bonsai inspect model.zip --trees

# Show more trees
bonsai inspect model.zip --trees --num-trees 10
```

The inspect command shows:
- Model metadata (trees, features, task type, aggregation)
- Tree statistics (depth, nodes, split types, missing value handling)
- Feature usage analysis (most-used features, unused features)
- Categorical feature details (bitsets, encoding)
- Tree structure visualization (with --trees flag)
- Validation warnings (unused features, unusual values)

### Generate Test Models and Data

```bash
# Generate all test fixtures (requires Python with h2o/scikit-learn)
./scripts/generate_all_fixtures.sh
# or
python3 scripts/generate_all_fixtures.py

# Run cargo integration tests
cargo test --test integration_test -- --include-ignored
```

See [`assets/tests/README.md`](assets/tests/README.md) for details on individual test scenarios.

## Development

```bash
# Run unit tests
cargo test

# Run all integration tests
cargo test -- --ignored
```

See [`CHANGELOG.md`](CHANGELOG.md) for recent changes and fixes.

## Benchmarks

See [`examples/xgboost_benchmark/`](examples/xgboost_benchmark/) for a full comparison of bonsai vs ONNX Runtime (Python + Rust) on a 100-tree XGBoost model. On Apple M-series hardware, bonsai scores a single row in **~137 ns** vs ~3.5 µs for the `ort` Rust crate.

## Future Roadmap

See [`PLAN.md`](PLAN.md) for planned features including:
- CatBoost JSON support
- SIMD batch optimization (`predict_batch`)
- Python bindings (PyO3)
- WebAssembly (WASM) target

## License

See LICENSE file for details.
