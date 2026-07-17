# Bonsai: Tree Ensemble Transpiler

**Mission:** Convert trained tree-based ML models (Random Forests, GBMs) into standalone, zero-dependency Rust code for low-latency inference.

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
│   ├── inspector.rs        # `bonsai inspect` - human-readable model summary
│   │
│   ├── frontends/          # Format-specific ingest + parse
│   │   ├── mod.rs          # Frontend trait definition
│   │   ├── catboost.rs     # CatBoost JSON → IR
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
│   ├── catboost/           # CatBoost regression
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
| **H2O MOJO** (v1.40+) | GBM, DRF; Binomial + Gaussian distributions |
| **ONNX** | `TreeEnsembleRegressor` + `TreeEnsembleClassifier` operators |
| **XGBoost JSON** | `booster.save_model("model.json")` |
| **LightGBM JSON** | `booster.dump_model()` |
| **CatBoost JSON** | `model.save_model("model.json", format="json")`; oblivious trees |

### Supported ML Tasks

| Task | Post-Transform |
|------|---------------|
| Regression | `Identity` or `Log` |
| Binary classification | `Logit` (sigmoid) |
| Multiclass classification | `Softmax { n_classes }` |

### Supported Tree Features

- **Numeric splits**: Threshold-based (`<`, `<=`, `>`, `>=`, `==`, `!=`)
- **Categorical splits**: Bitset-encoded set membership (H2O MOJO / ONNX)
- **Missing value handling**: `Left`, `Right`, `NaVsRest`, `None`
- **Aggregation**: `Sum` (GBM) and `Average` (DRF / RF)
- **Base score / intercept**: Preserved in f64
- **Oblivious Trees**: Symmetric structure optimization (CatBoost)

### Code Generation

- **Zero dependencies**: Generated Rust has no runtime deps
- **Inline tree functions**: `#[inline(always)] fn tree_N(features: &[f32]) -> f64`
- **f64 aggregation, f32 output**: `predict` returns `f32`; internal accumulation in `f64`
- **Batch API**: `predict_batch` and `predict_proba_batch` for high-throughput scoring
- **Conditional helpers**: `bitset_contains()` emitted only when categoricals are present
- **Multiclass output**: `predict_proba(&self, features: &[f32]) -> Vec<f32>`

---

## IR Reference

```rust
pub struct Forest {
    pub trees:          Vec<Tree>,
    pub base_score:     f64,           // bias/intercept (Sum only)
    pub base_scores:    Vec<f64>,      // per-class biases (XGBoost multiclass)
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

impl Node {
    // Detects symmetric structure for branchless fast-path
    pub fn get_oblivious_splits(&self) -> Option<Vec<(usize, SplitKind, MissingDirection)>>;
    // Collects leaves in bitmasked order
    pub fn collect_leaves(&self, out: &mut Vec<f64>);
}

pub enum SplitKind {
    Numeric     { threshold: f32, operator: Operator },
    Categorical { bitoff: u16, nbits: u32, data: Vec<u8> },
}
```

---

## Pipeline Details

### Frontend (Ingest + Parse)

**`frontends/catboost.rs`** - CatBoost JSON:
- Parses `oblivious_trees` with symmetric structure
- Maps leaf values to a flat array for bitmasked indexing
- Supports `Logloss` (Binary) and `RMSE` (Regression)

**`frontends/mojo.rs`** - H2O MOJO (`.zip`):
- Parses `model.ini` for metadata (algo, distribution, link function)
- Extracts tree binaries (`trees/t00_XXX.bin` + `_aux.bin`)
- Calls `parsers/tree_parser.rs` for each tree

**`frontends/onnx.rs`** - ONNX (`.onnx`):
- Decodes protobuf via `prost`
- Extracts `TreeEnsembleRegressor` / `TreeEnsembleClassifier`
- Handles per-node `missing_value_tracks_true` flag
- Populates leaves from `target_weights` (regressor) or `class_weights` (classifier)

**`frontends/xgboost.rs`** - XGBoost JSON (`.json`):
- Strips XGBoost 3.x bracket notation from `base_score`
- Applies logit to `base_score` when it is in probability space
- Flat parallel array format: `left_children`, `right_children`, `split_indices`, etc.

**`frontends/lightgbm.rs`** - LightGBM JSON (`.json`):
- Handles both string and array objective formats
- Recursive `tree_structure` node format
- Threshold encoded as string or number depending on version

### Backend (Code Generation)

**`backends/rust.rs`:**
- Emits one `tree_N(features: &[f32]) -> f64` function per tree
- **Oblivious Optimization**: Generates branchless bitmasking for symmetric trees
- `NaVsRest` emits `!val.is_nan()` with no threshold comparison
- **Batch API**: `predict_batch` chunks input into rows for auto-vectorization
- Aggregation for `Sum`: `base_score + Σ(Self::tree_N(features) * weight)`
- Aggregation for `Average`: `Σ(Self::tree_N(features)) / n`
- Post-transforms: `raw as f32` / `(1/(1+exp(-raw))) as f32` / `raw.exp() as f32`
- Softmax: generates `predict_proba(&self, features: &[f32]) -> Vec<f32>` and `predict_proba_batch`

---

## Testing

### Unit Tests (100+ passing)

- `ir.rs`: node traversal, aggregation, post-transforms, missing direction, categoricals, **oblivious detection**
- `backends/rust.rs`: code generation for identity, logit, softmax, **oblivious fast-path, batch API**
- `backends/rust_array.rs`: array-layout flattening, layout resolution, no_std output, plus rustc-compiled differential tests requiring bit-identical output across layouts
- `interpreter.rs`: split/missing-direction semantics, plus seeded fuzz tests comparing the interpreter against both compiled layouts bit-for-bit
- `frontends/xgboost.rs`: tree structure, NaN routing, base_score logit conversion, multiclass, DART weight_drop, unsupported-model rejection
- `frontends/lightgbm.rs`: tree structure, NaN routing, multiclass, objective mapping (log-link, ranking, rejected transforms)
- `frontends/catboost.rs`: minimal/CTR model parsing, malformed-input rejection
- `parsers/tree_parser.rs`: H2O MOJO binary format parsing
- `verify.rs` / `emit_crate.rs`: CSV contract parsing, crate-name sanitization; end-to-end tests in `tests/` drive the actual binary

### Integration Tests (13 non-H2O scenarios passing in CI; H2O MOJO needs a local Java runtime)

Each test: transpile model → compile with rustc → score CSV → compare against Python ground truth.

| Test | Status | Tolerance |
|------|--------|-----------|
| `catboost/regression` | | 1e-5 |
| `xgboost/regression_numeric` | | 1e-4 |
| `xgboost/classification_numeric` | | 1e-5 |
| `xgboost/classification_multiclass` | | 1e-4 |
| `lightgbm/regression_numeric` | | 1e-5 |
| `lightgbm/classification_numeric` | | 1e-5 |
| `lightgbm/classification_multiclass` | | 1e-5 |
| `sklearn_onnx/regression_numeric` | | 1e-5 |
| `sklearn_onnx/regression_categorical` | | 1e-5 |
| `sklearn_onnx/classification_numeric` | | 1e-5 |
| `sklearn_onnx/classification_categorical` | | 1e-5 |
| `h2o_mojo/*` (4 tests) | ⏭ | Require Java to generate model files |

Run with: `cargo test --test integration_test -- --include-ignored`

---

## Known Limitations

- **XGBoost regression leaf precision**: XGBoost stores leaves internally as `f32`, so accumulated error over many trees can reach ~4×10⁻⁵. Regression tolerance is set to 1e-4 accordingly.
- **H2O MOJO generation**: Requires a Java runtime.
- **ONNX true categorical nodes**: `nodes_categorical_attributes` is not supported.

---

## Roadmap

### Near-Term
- [x] **Benchmarking Harness**: bonsai ~137 ns/row vs ort ~3.5 µs.
- [x] **CatBoost JSON support**: Support oblivious tree structures.
- [x] **SIMD Optimization - Phase 1**: `predict_batch` scalar loop; enables LLVM auto-vectorization.
- [x] **CI Integration**: Integration tests run in GitHub Actions; fixtures regenerated via pip-installed frameworks.
- [x] **Array code layout**: `--layout array` keeps rustc practical on very large forests (auto above 10k nodes).
- [x] **`bonsai verify`**: transpile → compile → score → diff against reference predictions, or `--engine interpret` without rustc.
- [x] **`--emit crate`**: full cargo crate output with baked-in golden tests.
- [x] **`--no-std`**: core-only generated code for embedded targets.
- [ ] **crates.io release**: publish 0.1.0.

### Mid-Term
- [ ] **Python Bindings (PyO3)**: Generate Python-loadable modules for easy validation.
- [ ] **SHAP value computation**: Feature contributions (TreeSHAP).
- [ ] **SIMD Optimization - Phase 2**: Oblivious tree evaluation via `std::simd` - evaluate all nodes branchlessly, `select` the leaf. Process 8–16 rows per SIMD lane. Targets trees ≤ depth 8.
- [ ] **Batch scoring optimization**: Rayon/SIMD integration in `polars_score`.

### Long-Term
- [ ] **WASM target**: Browser and edge-runtime inference.
- [ ] **Binary `.bon` format**: Fast cold-start loading via rkyv/bincode.
- [ ] **C backend**: For embedded and legacy system FFI.
- [ ] **`bonsai pipe`**: Streaming stdin CSV → stdout scores.

---

## Dependencies

**Core:**
- `anyhow` - error handling
- `clap` - CLI argument parsing
- `zip` - MOJO archive extraction
- `byteorder` - little-endian binary parsing
- `serde_json` - XGBoost / LightGBM / CatBoost JSON parsing

**Code Generation:**
- `proc-macro2`, `quote` - Rust token stream construction

**ONNX:**
- `prost` - protobuf decoding (generated at build time via `build.rs`)

**Batch Scoring Binary (`polars_score`):**
- `polars` - DataFrame I/O
- `rayon` - parallel scoring

---

## Philosophy

**Bonsai trims tree models for production:**
- **No runtime dependencies**: Deploy anywhere (servers, edge, WASM)
- **Explicit over implicit**: IR makes every transformation auditable
- **Performance by default**: Inline everything; f64 accumulation, f32 output; oblivious fast-path
- **Framework-agnostic**: H2O, ONNX, XGBoost, LightGBM, CatBoost - same IR, same backend

**Train in Python/R/Java. Deploy as pure Rust.**
