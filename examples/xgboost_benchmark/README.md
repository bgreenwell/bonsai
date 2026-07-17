# XGBoost Benchmark

Compares the common ways to score an XGBoost binary-classification model:

| Method | Language | Description |
|--------|----------|-------------|
| XGBoost native | Python | `booster.predict(DMatrix(X))` |
| ONNX Runtime | Python | `onnxruntime.InferenceSession` |
| ONNX Runtime (`ort`) | Rust | `ort::Session::run` |
| bonsai | Rust | transpiled, inlined decision trees |

## Prerequisites

- Rust toolchain (stable)
- [uv](https://github.com/astral-sh/uv) Python package manager

## Steps

### 1. Generate model assets and run Python benchmarks

```bash
cd examples/xgboost_benchmark
uv venv
uv pip install -r requirements.txt
uv run python generate.py
```

This trains a 100-tree XGBoost binary classifier (10 numeric features), exports it as
`generated/model.json` (bonsai) and `generated/model.onnx` (ort / onnxruntime), times
the Python inference methods, and prints a summary table.

### 2. Transpile the model with bonsai

```bash
# From the repo root:
cargo run -- transpile \
    --input  examples/xgboost_benchmark/generated/model.json \
    --output examples/xgboost_benchmark/generated/model.rs
```

### 3. Run Rust benchmarks

```bash
cargo bench --bench xgboost
```

Criterion results are written to `target/criterion/xgboost_binary_classification/`.

## Typical results (Apple M-series, 100 trees, depth 5)

### Python side (`generate.py`)

| Method | Single row | Per-row @ batch 1000 |
|--------|-----------|----------------------|
| XGBoost native | ~46 µs | ~0.05 µs |
| ONNX Runtime (Python) | ~5 µs | ~0.55 µs |

> XGBoost native has high fixed overhead per call (DMatrix allocation) but
> amortises almost perfectly over large batches. ONNX Runtime is faster
> for single-row serving but scales linearly (no SIMD batching in this model).

### Rust side (`cargo bench --bench xgboost`)

| Method | Single row | Throughput (batch 1000) |
|--------|-----------|-------------------------|
| bonsai | ~137 ns | ~7.1 Melem/s |
| ort (Rust) | ~3.5 µs | ~2.4 Melem/s |

> bonsai is ~25x faster than ort for single-row inference: no FFI, no tensor
> allocation, just inlined Rust decision-tree code.
> ort catches up somewhat at large batches (ONNX Runtime uses internal SIMD);
> bonsai Phase 2 (oblivious SIMD codegen) is planned to close this gap.

## What's next

- **Phase 1** (`predict_batch`): scalar loop over rows, enables auto-vectorisation
  of the tree-score accumulation by LLVM.
- **Phase 2** (oblivious SIMD): branchless tree evaluation using `std::simd`,
  processing 8–16 rows per SIMD lane - expected to match or beat ort batch throughput.
