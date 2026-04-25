# Bonsai: Tree Ensemble Transpiler

**Mission:** Convert trained tree-based ML models (Random Forests, GBMs) into standalone, zero-dependency Rust code for ultra-low latency inference.

---

## Current Architecture

Bonsai is a **single-crate transpiler** (not a workspace) that follows a three-stage compiler pipeline:

```
Input Format → Frontend → IR → Backend → Output Code
  (MOJO/ONNX)   (Parser)  (Tree) (Codegen)  (Rust .rs)
```

### Directory Structure

```
bonsai/
├── Cargo.toml              # Single-crate project
├── build.rs                # Protobuf build (ONNX)
├── src/
│   ├── main.rs             # CLI: parse args, orchestrate pipeline
│   ├── ir.rs               # Intermediate representation (Forest, Node, etc.)
│   │
│   ├── frontends/          # Format-specific ingest + parse
│   │   ├── mojo.rs         # H2O MOJO (.zip) → IR
│   │   └── onnx.rs         # ONNX (.onnx) → IR
│   │
│   ├── parsers/            # Low-level binary parsers
│   │   ├── tree_parser.rs  # H2O MOJO tree format (version 1.40)
│   │   └── ini.rs          # MOJO model.ini metadata
│   │
│   ├── backends/           # Code generation
│   │   └── rust.rs         # IR → Rust source code
│   │
│   ├── bin/
│   │   └── polars_score.rs # Batch scoring CLI (separate binary)
│   │
│   └── proto/              # ONNX protobuf definitions
│
├── assets/examples/        # Integration tests + examples
│   └── h2o3/               # H2O-3 GBM example (MOJO + ONNX)
│
└── archive/                # Previous implementations (h2o-poet, transmute)
```

---

## Implemented Features

### Supported Formats
- ✅ **H2O MOJO** (version 1.40+): Native binary format
  - GBM (Gradient Boosting Machine)
  - Binomial, Gaussian distributions
  - Logit, Identity, Log link functions
- ✅ **ONNX**: TreeEnsemble operators
  - Via sklearn `HistGradientBoostingClassifier`
  - Generic tree ensemble support

### Supported Features
- ✅ **Numeric splits**: Threshold-based decision nodes
- ✅ **Categorical splits**: Bitset-encoded set membership
- ✅ **Missing value handling**: `NaVsRest`, `NaLeft`, `NaRight`, `Left`, `Right`
- ✅ **Aggregation**: Sum (GBM), Average (Random Forest)
- ✅ **Post-transforms**: Identity, Logit (sigmoid), Log (exp)
- ✅ **Binary classification**: Bernoulli + logit link
- ✅ **Regression**: Gaussian + identity link

### Code Generation
- ✅ **Zero dependencies**: Generated Rust code has no runtime deps
- ✅ **Inline tree functions**: `#[inline(always)]` for each tree
- ✅ **Conditional helpers**: `bitset_contains()` only if categoricals present
- ✅ **f64 aggregation, f32 output**: Precision where it matters

---

## Pipeline Details

### 1. Frontend (Ingest + Parse)

**`frontends/mojo.rs`:**
- Opens `.zip` archive (MOJO format)
- Parses `model.ini` for metadata (algo, n_trees, distribution, link_function)
- Extracts tree files: `trees/t00_XXX.bin`, `trees/t00_XXX_aux.bin`
- Calls `parsers/tree_parser.rs` for each tree
- Builds `ir::Forest`

**`frontends/onnx.rs`:**
- Loads `.onnx` file via protobuf (prost)
- Extracts `TreeEnsembleClassifier` or `TreeEnsembleRegressor` operators
- Converts ONNX nodes/splits → `ir::Node`
- Builds `ir::Forest`

### 2. Intermediate Representation (IR)

**`ir.rs`:**
```rust
pub struct Forest {
    pub trees: Vec<Tree>,
    pub aggregation: AggregationKind,  // Sum | Average
    pub post_transform: PostTransform, // Identity | Logit | Log
    pub base_score: f64,
}

pub enum Node {
    Leaf { value: f32 },
    Split {
        feature_idx: usize,
        split: SplitKind,
        left_child: Box<Node>,
        right_child: Box<Node>,
        missing_direction: MissingDirection,
    },
}

pub enum SplitKind {
    Numeric { threshold: f32, operator: Operator },
    Categorical { bitoff: u16, nbits: u32, data: Vec<u8> },
}

pub enum MissingDirection {
    Left,      // NaN always goes left
    Right,     // NaN always goes right
    None,      // No special handling
    NaVsRest,  // Split solely on NaN-ness (NaN→right, non-NaN→left)
}
```

### 3. Backend (Code Generation)

**`backends/rust.rs`:**
- Generates `pub struct Model;` with `pub fn predict(&self, features: &[f32]) -> f32`
- Creates one `tree_N()` function per tree
- Recursively compiles `Node` → nested if/else blocks
- Handles `NaVsRest` by emitting `!val.is_nan()` without threshold comparison
- Outputs aggregation expression: `base_score + tree_0(...) * w0 + tree_1(...) * w1 + ...`
- Applies post-transform: `(1.0 / (1.0 + (-raw).exp())) as f32` for logit

**Generated code example:**
```rust
pub struct Model;

impl Model {
    pub fn predict(&self, features: &[f32]) -> f32 {
        let raw: f64 = 0.0 + Self::tree_0(features) as f64 * 1.0
                           + Self::tree_1(features) as f64 * 1.0;
        (1.0f64 / (1.0f64 + (-raw).exp())) as f32
    }

    #[inline(always)]
    fn tree_0(features: &[f32]) -> f32 {
        {
            let val = features[3];
            if !val.is_nan() && (val < 0.5) {
                0.123
            } else {
                -0.456
            }
        }
    }
}
```

---

## Recent Fixes (Feb 2025)

### NaVsRest Node Bug

**Problem:** MOJO parser failed on binomial GBM models with "failed to fill whole buffer" error.

**Root Cause:** `NaVsRest` split nodes don't store a numeric threshold in the MOJO binary format (they split solely on NaN-ness). The parser was incorrectly reading 4 bytes for a split_value that doesn't exist, causing misalignment.

**Fixes:**
1. **Parser** (`parsers/tree_parser.rs:193-196`):
   ```rust
   } else if na_vs_rest {
       // FIX: NaVsRest splits solely on NaN-ness, no threshold value stored
       Split::Numeric { split_value: f32::NAN }
   }
   ```

2. **Codegen** (`backends/rust.rs:140-144`):
   ```rust
   let left_cond = match (missing_direction, split) {
       (MissingDirection::NaVsRest, _) => {
           quote! { !val.is_nan() }
       }
       // ... handle numeric/categorical normally
   }
   ```

**Result:** MOJO parser now handles binomial GBM models correctly. Predictions match H2O-3 ground truth within 1e-7 (f32 precision).

---

## Project Audit (Feb 2026)

### Audit Summary
- **Architecture**: The 3-stage pipeline is well-implemented and provides a solid foundation for multi-format support.
- **Reliability**: Key semantic issues (categorical routing, NaN handling) are correctly addressed in the IR and Backend.
- **Testing Gap**: **Critical deficiency** in automated testing. There are currently no unit tests in the `src/` directory, and integration tests are ignored by default and require external Python-based setup.
- **Documentation**: `AGENTS.md` is outdated and incorrectly describes the testing state.

### Audit Recommendations
- **Testing**: Prioritize a unit test suite for core transpiler logic (`ir.rs`, `tree_parser.rs`, `rust.rs`).
- **Automation**: Integrate validation into a standard Rust CI workflow with committed mock assets.
- **Documentation**: Synchronize `AGENTS.md` with the actual codebase structure and testing procedures.

---

## Ideas for Deployment: Scaling to 300M+ Records

To truly scale in enterprise environments like Databricks/Unity Catalog, `bonsai` should move from generating code to generating **deployable artifacts**.

### Approach A: The "Speed Demon" (Native Rust/Polars)
*   **Workflow**: Compile model into the `polars_score` binary. Run as a standalone Databricks Task.
*   **Performance**: Absolute highest throughput. Limited only by I/O.
*   **Pros**: Simplest architecture; zero overhead from Spark/Python; utilizes Polars Lazy API for memory-mapped I/O.

### Approach B: The "Seamless" (Rust-backed Pandas UDF)
*   **Workflow**: Use PyO3 to wrap the model in a Python library. Call via `pandas_udf` in PySpark.
*   **Performance**: High. Leverages Spark for distribution while using Rust/Arrow for zero-copy scoring.
*   **Pros**: Easiest integration into existing Spark pipelines; respects Unity Catalog permissions out-of-the-box.

### Approach C: The "Enterprise" (Native Polars Plugin)
*   **Workflow**: Compile model as a Polars Expression Plugin. Use `pl.read_delta()` from Unity Catalog.
*   **Performance**: Ultra-High. Provides a high-level Python API with raw Rust/SIMD performance.
*   **Pros**: Cleanest API for Data Scientists; bypasses Spark's "runtime tax."

---

## Roadmap

### Near-Term (Audit Fixes & Next Sprint)
- [ ] **Core Unit Test Suite**: Implement tests for `ir.rs`, `tree_parser.rs`, and `rust.rs`.
- [ ] **Sync Documentation**: Update `AGENTS.md` and `README.md` to reflect actual testing state.
- [ ] **CI Integration**: Setup GitHub Actions for `cargo test` and automated integration validation.
- [ ] Multi-class classification (softmax post-transform)
- [ ] Distributed Random Forest (DRF) validation
- [ ] XGBoost JSON format support
- [ ] LightGBM JSON format support
- [ ] Benchmark suite (latency, throughput vs H2O/sklearn)

### Mid-Term
- [ ] CatBoost support (oblivious trees)
- [ ] SHAP value computation (feature contributions)
- [ ] Batch scoring optimization (SIMD, Rayon)
- [ ] WASM target (browser inference)

### Long-Term
- [ ] Spark integration (RDD.pipe() example)
- [ ] Python bindings (PyO3) for PySpark
- [ ] Binary model format (.bon) for fast loading
- [ ] Model inspection CLI (`bonsai inspect model.zip`)

---

## Dependencies

**Core:**
- `anyhow` - Error handling
- `clap` - CLI argument parsing
- `zip` - MOJO archive extraction
- `byteorder` - Little-endian binary parsing

**Code Generation:**
- `proc-macro2`, `quote` - Rust code generation

**ONNX:**
- `prost` - Protobuf parsing (build-time via `build.rs`)

**Batch Scoring Binary:**
- `polars` - DataFrame operations
- `rayon` - Parallel scoring

---

## Development Workflow

### Build and Test
```bash
# Build the transpiler
cargo build --release

# Convert a model
./target/release/bonsai --input model.zip --output model.rs

# Run tests
cargo test

# Run integration tests (requires external setup)
cargo test h2o3_integration -- --ignored
```

### Adding a New Format

1. **Create frontend:** `src/frontends/newformat.rs`
   - Parse format-specific metadata
   - Extract tree structures
   - Convert to `ir::Forest`

2. **Add CLI flag:** Update `src/main.rs` to detect format

3. **Add tests:** Create `assets/examples/newformat/` with validation

4. **Update docs:** Add to README.md and this PLAN.md

### Adding a New Backend

1. **Create backend:** `src/backends/python.rs` (or `c.rs`, `java.rs`, etc.)
   - Implement `compile_node()` recursion
   - Handle aggregation + post-transform
   - Generate target language code

2. **Add CLI flag:** `--backend=rust|python|c`

3. **Add tests:** Validate generated code correctness

---

## Philosophy

**Bonsai trims tree models for production:**
- **No runtime dependencies**: Deploy anywhere (servers, edge, WASM)
- **Explicit over implicit**: IR makes transformations auditable
- **Performance by default**: Inline everything, f64 where needed
- **Framework-agnostic**: MOJO, ONNX, XGBoost, LightGBM, CatBoost

**Train in Python/R/Java. Deploy as pure Rust.**
