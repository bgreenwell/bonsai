# Bonsai: Tree Ensemble Transpiler

**Mission:** Convert trained tree-based ML models (Random Forests, GBMs) into standalone, zero-dependency Rust code for ultra-low latency inference.

---

## Current Architecture

Bonsai is a **single-crate transpiler** that follows a three-stage compiler pipeline:

```
Input Format → Frontend → IR → Backend → Output Code
 (JSON/ONNX/   (Parser)  (Forest/  (Codegen)  (Rust .rs)
  MOJO/.zip)             Tree/Node)
```

### Directory Structure

```
bonsai/
├── Cargo.toml              # Single-crate project
├── build.rs                # Protobuf build (ONNX)
├── src/
│   ├── main.rs             # CLI: parse args, orchestrate pipeline
│   ├── ir.rs               # Intermediate representation (Forest, Node, etc.)
│   ├── inspector.rs        # `bonsai inspect` — human-readable model summary
│   │
│   ├── frontends/          # Format-specific ingest + parse
│   │   ├── mod.rs          # Frontend trait definition
│   │   ├── mojo.rs         # H2O MOJO (.zip) → IR
│   │   ├── onnx.rs         # ONNX (.onnx) → IR
│   │   ├── xgboost.rs      # XGBoost JSON (.json) → IR
│   │   └── lightgbm.rs     # LightGBM JSON dump (.json) → IR
│   │
│   ├── parsers/            # Low-level binary parsers (H2O MOJO only)
│   │   ├── mod.rs
│   │   ├── tree_parser.rs  # H2O MOJO tree binary format (version 1.40)
│   │   └── ini.rs          # MOJO model.ini metadata
│   │
│   ├── backends/           # Code generation
│   │   ├── mod.rs
│   │   └── rust.rs         # IR → Rust source code
│   │
│   ├── bin/
│   │   └── polars_score.rs # Batch scoring CLI (separate binary)
│   │
│   └── proto/              # ONNX protobuf definitions
│       └── onnx.proto
│
├── assets/tests/           # Integration test fixtures
│   ├── common/             # Shared Python data generation utilities
│   ├── xgboost/            # XGBoost regression + classification
│   ├── lightgbm/           # LightGBM regression + classification
│   ├── sklearn_onnx/       # sklearn GBM via ONNX (4 variants)
│   └── h2o_mojo/           # H2O MOJO (requires Java to generate)
│
└── tests/
    └── integration_test.rs # End-to-end transpile → compile → score tests
```

---

## Implemented Features

### Supported Input Formats

| Format | Notes |
|--------|-------|
| ✅ **H2O MOJO** (v1.40+) | GBM, DRF; Binomial + Gaussian distributions |
| ✅ **ONNX** | `TreeEnsembleRegressor` + `TreeEnsembleClassifier` operators |
| ✅ **XGBoost JSON** | `booster.save_model("model.json")`; handles XGBoost 3.x bracket format for `base_score` |
| ✅ **LightGBM JSON** | `booster.dump_model()`; handles both string and array objective formats |

### Supported ML Tasks

| Task | Post-Transform |
|------|---------------|
| ✅ Regression | `Identity` or `Log` |
| ✅ Binary classification | `Logit` (sigmoid) |
| ✅ Multiclass classification | `Softmax { n_classes }` |

### Supported Tree Features

- ✅ **Numeric splits**: Threshold-based (`<`, `<=`, `>`, `>=`, `==`, `!=`)
- ✅ **Categorical splits**: Bitset-encoded set membership (H2O MOJO / ONNX)
- ✅ **Missing value handling**: `Left`, `Right`, `NaVsRest`, `None`
- ✅ **Aggregation**: `Sum` (GBM) and `Average` (DRF / RF)
- ✅ **Base score / intercept**: Preserved in f64

### Code Generation

- ✅ **Zero dependencies**: Generated Rust has no runtime deps
- ✅ **Inline tree functions**: `#[inline(always)] fn tree_N(features: &[f32]) -> f64`
- ✅ **f64 aggregation, f32 output**: `predict` returns `f32`; internal accumulation in `f64`
- ✅ **Conditional helpers**: `bitset_contains()` emitted only when categoricals are present
- ✅ **Multiclass output**: `predict_proba` returns `Vec<f32>` for softmax models

---

## IR Reference

```rust
pub struct Forest {
    pub trees:          Vec<Tree>,
    pub base_score:     f64,           // bias/intercept (Sum only)
    pub aggregation:    AggregationKind,
    pub post_transform: PostTransform,
}

pub enum AggregationKind { Sum, Average }

pub enum PostTransform {
    Identity,
    Logit,                             // 1 / (1 + exp(-raw))
    Log,                               // exp(raw)
    Softmax { n_classes: usize },      // round-robin tree→class assignment
}

pub enum Node {
    Split {
        feature_idx:       usize,
        split:             SplitKind,
        left_child:        Box<Node>,
        right_child:       Box<Node>,
        missing_direction: MissingDirection,
    },
    Leaf { value: f64 },               // f64 preserves LightGBM's native precision
}

pub enum SplitKind {
    Numeric     { threshold: f32, operator: Operator },
    Categorical { bitoff: u16, nbits: u32, data: Vec<u8> },
}

pub enum MissingDirection {
    Left,      // NaN always goes left
    Right,     // NaN always goes right
    NaVsRest,  // Split solely on NaN-ness (non-NaN→left, NaN→right); threshold ignored
    None,      // Format did not specify; backend defaults to Right
}
```

---

## Pipeline Details

### Frontend (Ingest + Parse)

**`frontends/mojo.rs`** — H2O MOJO (`.zip`):
- Parses `model.ini` for metadata (algo, distribution, link function)
- Extracts tree binaries (`trees/t00_XXX.bin` + `_aux.bin`)
- Calls `parsers/tree_parser.rs` for each tree

**`frontends/onnx.rs`** — ONNX (`.onnx`):
- Decodes protobuf via `prost`
- Extracts `TreeEnsembleRegressor` / `TreeEnsembleClassifier`
- Handles per-node `missing_value_tracks_true` flag
- Populates leaves from `target_weights` (regressor) or `class_weights` (classifier)

**`frontends/xgboost.rs`** — XGBoost JSON (`.json`):
- Strips XGBoost 3.x bracket notation from `base_score` (e.g., `"[0E0]"`)
- Applies logit to `base_score` when it is in probability space (logistic objectives)
- Flat parallel array format: `left_children`, `right_children`, `split_indices`, etc.

**`frontends/lightgbm.rs`** — LightGBM JSON (`.json`):
- Handles both string (`"binary sigmoid:1"`) and array (`["binary", "crossentropy"]`) objective formats
- Recursive `tree_structure` node format with `leaf_value` / `split_feature`
- Threshold encoded as string or number depending on LightGBM version

### Backend (Code Generation)

**`backends/rust.rs`:**
- Emits one `tree_N(features: &[f32]) -> f64` function per tree
- Recursively compiles `Node` → nested `if/else` blocks
- `NaVsRest` emits `!val.is_nan()` with no threshold comparison
- Aggregation for `Sum`: `base_score + Σ(Self::tree_N(features) * weight)`
- Aggregation for `Average`: `Σ(Self::tree_N(features)) / n`
- Post-transforms: `raw as f32` / `(1/(1+exp(-raw))) as f32` / `raw.exp() as f32`
- Softmax: generates `predict_proba(&self, features: &[f32]) -> Vec<f32>` instead of `predict`

---

## Testing

### Unit Tests (53 passing)

- `ir.rs`: node traversal, aggregation, post-transforms, missing direction, categoricals
- `backends/rust.rs`: code generation for identity, logit, softmax; categorical helper inclusion
- `frontends/xgboost.rs`: tree structure, NaN routing, base_score logit conversion, multiclass
- `frontends/lightgbm.rs`: tree structure, NaN routing, multiclass, numeric threshold as JSON number
- `parsers/tree_parser.rs`: H2O MOJO binary format parsing

### Integration Tests (9/13 passing)

Each test: transpile model → compile with rustc → score CSV → compare against Python ground truth.

| Test | Status | Tolerance |
|------|--------|-----------|
| `xgboost/regression_numeric` | ✅ | 1e-4 (f32 leaf accumulation) |
| `xgboost/classification_numeric` | ✅ | 1e-5 |
| `lightgbm/regression_numeric` | ✅ | 1e-5 |
| `lightgbm/classification_numeric` | ✅ | 1e-5 |
| `sklearn_onnx/regression_numeric` | ✅ | 1e-5 |
| `sklearn_onnx/regression_categorical` | ✅ | 1e-5 |
| `sklearn_onnx/classification_numeric` | ✅ | 1e-5 |
| `sklearn_onnx/classification_categorical` | ✅ | 1e-5 |
| `h2o_mojo/*` (4 tests) | ⏭ | Require Java to generate model files |

Run with: `cargo test --test integration_test -- --include-ignored`

---

## Known Limitations

- **XGBoost regression leaf precision**: XGBoost stores leaves internally as `f32`, so accumulated error over many trees can reach ~4×10⁻⁵. Regression tolerance is set to 1e-4 accordingly.
- **LightGBM categorical splits**: `decision_type == "=="` is not yet supported; will return an error.
- **H2O MOJO generation**: Requires a Java runtime. The four H2O tests are `#[ignore]` and skip if model files are absent.

---

## Roadmap

### Near-Term
- [ ] CatBoost JSON support (oblivious tree layout)
- [ ] H2O MOJO: generate test fixtures in CI without a local Java install
- [ ] LightGBM categorical splits (`decision_type == "=="`)
- [ ] `bonsai inspect` output: add tree count, depth, feature count summary
- [ ] Benchmark suite (latency vs. native XGBoost/LightGBM predict)

### Mid-Term
- [ ] SHAP value computation (feature contributions)
- [ ] Batch scoring optimization (SIMD, Rayon parallelism)
- [ ] WASM target (browser inference)
- [ ] Python bindings (PyO3) for PySpark `pandas_udf`

### Long-Term
- [ ] Binary `.bon` model format (bincode/rkyv) for fast cold-start loading
- [ ] C backend (`--backend=c`) for embedded / FFI targets
- [ ] `bonsai pipe` streaming mode: stdin CSV → stdout scores (Spark `RDD.pipe()`)

---

## Dependencies

**Core:**
- `anyhow` — error handling
- `clap` — CLI argument parsing
- `zip` — MOJO archive extraction
- `byteorder` — little-endian binary parsing
- `serde_json` — XGBoost / LightGBM JSON parsing

**Code Generation:**
- `proc-macro2`, `quote` — Rust token stream construction

**ONNX:**
- `prost` — protobuf decoding (generated at build time via `build.rs`)

**Batch Scoring Binary (`polars_score`):**
- `polars` — DataFrame I/O
- `rayon` — parallel scoring

---

## Development Workflow

```bash
# Build
cargo build --release

# Transpile a model
./target/release/bonsai transpile --input model.json --output model.rs

# Inspect a model
./target/release/bonsai inspect --input model.json

# Unit tests
cargo test

# Integration tests (requires generated model fixtures)
cargo test --test integration_test -- --include-ignored
```

### Adding a New Frontend

1. Create `src/frontends/newformat.rs`, implement `Frontend` trait (`parse(&self, path: &Path) -> Result<Forest>`)
2. Add `pub mod newformat;` in `src/frontends/mod.rs`
3. Wire detection in `src/main.rs` (`detect_and_parse_*` or the JSON probe)
4. Add generate script under `assets/tests/newformat/*/generate.py`
5. Add `#[ignore]` integration test in `tests/integration_test.rs`

### Adding a New Backend

1. Create `src/backends/newlang.rs`, implement codegen over `Forest`
2. Add `--backend` flag to CLI in `src/main.rs`
3. Add unit tests verifying generated code structure

---

## Philosophy

**Bonsai trims tree models for production:**
- **No runtime dependencies**: Deploy anywhere (servers, edge, WASM)
- **Explicit over implicit**: IR makes every transformation auditable
- **Performance by default**: Inline everything; f64 accumulation, f32 output
- **Framework-agnostic**: H2O, ONNX, XGBoost, LightGBM — same IR, same backend

**Train in Python/R/Java. Deploy as pure Rust.**
