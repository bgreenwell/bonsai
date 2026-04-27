# bonsai

Transpile machine learning tree ensemble models to standalone Rust code.

## Overview

**bonsai** converts trained tree-based models (Random Forests, Gradient Boosting Machines, etc.) from various ML frameworks into efficient, dependency-free Rust code. The generated code can be embedded directly into your application without requiring the original ML runtime.

## Supported Formats

- **H2O-3**: MOJO (native binary format) and ONNX (via onnxmltools)
- **ONNX**: Generic tree ensemble models from multiple frameworks

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
cargo test --test integration_test -- --ignored
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

## Future Roadmap

See [`PLAN.md`](PLAN.md) for planned features including:
- CatBoost JSON support
- Distributed Random Forest (DRF)
- Batch scoring optimization
- WebAssembly (WASM) target

## License

See LICENSE file for details.
n
- Distributed Random Forest (DRF)
- XGBoost, LightGBM, CatBoost integration

## License

See LICENSE file for details.
