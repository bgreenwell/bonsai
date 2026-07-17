# AGENTS.md

@../CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**bonsai** is a tree ensemble model transpiler that converts trained ML models (Random Forests, GBMs) from XGBoost, LightGBM, CatBoost, H2O-3 MOJO, and ONNX into standalone, zero-dependency Rust code for low-latency inference.

**Key principle:** Train in Python/R/Java. Deploy as pure Rust.

## Build and Test Commands

```bash
# Build the transpiler
cargo build --release

# Convert a model to Rust code
./target/release/bonsai transpile --input model.zip --output model.rs
# Or during development:
cargo run -- transpile --input model.zip --output model.rs

# Verify a model end to end (transpile, compile, score CSV, diff)
./target/release/bonsai verify --input model.json --data test.csv

# Run all unit tests
cargo test

# Run a specific test
cargo test test_name

# Run integration tests (requires Python environment and model generation)
cargo test --test integration_test -- --ignored

# Build the optional polars_score batch scoring binary
cargo build --release --features scorer --bin polars_score
```

## Architecture: Three-Stage Compiler Pipeline

```
Input Format → Frontend → IR → Backend → Output Code
 (JSON/ONNX/    (Parser)  (Tree) (Codegen)  (Rust .rs)
  MOJO/.zip)
```

### 1. Frontends (`src/frontends/`)
Parse format-specific models into a universal intermediate representation:

- **`mojo.rs`**: H2O MOJO format (.zip archives)
  - Opens zip, parses `model.ini` metadata (algorithm, n_trees, distribution, link function)
  - Extracts binary tree files: `trees/t00_XXX.bin` and `trees/t00_XXX_aux.bin`
  - Calls low-level parsers in `src/parsers/tree_parser.rs`
  - Builds `ir::Forest`

- **`onnx.rs`**: ONNX format (.onnx, .pb)
  - Loads protobuf via prost (generated from `src/proto/onnx.proto` during build)
  - Extracts `TreeEnsembleClassifier` or `TreeEnsembleRegressor` operators
  - Converts ONNX parallel arrays (nodes_treeids, nodes_featureids, etc.) to recursive tree structure
  - Builds `ir::Forest`

- **`xgboost.rs`**: XGBoost format (.json)
  - Parses JSON model dumped via `booster.save_model()`
  - Handles parallel array format and logit transformation for `base_score`
  - Builds `ir::Forest`

- **`lightgbm.rs`**: LightGBM format (.json)
  - Parses JSON model dumped via `booster.dump_model()`
  - Handles recursive tree structure and various objective types
  - Builds `ir::Forest`

- **`catboost.rs`**: CatBoost format (.json)
  - Parses JSON model saved via `model.save_model(path, format="json")` (top-level `oblivious_trees` key)
  - Expands oblivious (symmetric) trees into IR trees; backend detects symmetry and emits a branchless fast path
  - Parses CTR metadata (`features_info.ctrs`) and CTR value tables (`ctr_data`) for native categorical features; generated code hashes categories with CityHash64 and looks up CTR values via binary search
  - Builds `ir::Forest` with `catboost_metadata` attached when categoricals are present

**File extension detection in `main.rs`:** `.zip` → MOJO, `.onnx`/`.pb` → ONNX, `.json` → XGBoost/LightGBM/CatBoost (auto-detected via JSON keys; CatBoost via `oblivious_trees`)

### 2. Intermediate Representation (`src/ir.rs`)

Universal tree ensemble representation that's format-agnostic:

```rust
pub struct Forest {
    pub trees: Vec<Tree>,
    pub aggregation: AggregationKind,  // Sum (GBM) | Average (RF)
    pub post_transform: PostTransform, // Identity | Logit | Log | Softmax
    pub base_score: f64,
    pub base_scores: Vec<f64>,         // per-class biases (XGBoost multiclass)
}

pub struct Tree {
    pub root: Node,
    pub weight: f64,
}

pub enum Node {
    Leaf { value: f64 },
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

### 3. Backend (`src/backends/rust.rs`, `src/backends/rust_array.rs`)

Two code layouts, selected via `--layout {auto,ifelse,array}` (default auto):

- **if/else** (`rust.rs`): one nested if/else function per tree. Fastest at small
  model sizes; source grows with every node.
- **array** (`rust_array.rs`): the forest flattened into static parallel arrays
  (`FEAT`/`THR`/`FLAGS`/`LEFT`/`RIGHT`/`LEAVES`/`ROOTS`/`WEIGHTS`) walked by a
  loop. Keeps rustc compile time practical for very large forests (a 158k-node
  model compiles in under a second vs 10+ minutes as if/else). Numeric splits
  only; auto mode falls back to if/else for categorical/CTR models and switches
  to array above `ARRAY_LAYOUT_NODE_THRESHOLD` (10k nodes). Both layouts produce
  bit-identical predictions (enforced by differential tests that compile both
  with rustc and compare outputs).

The if/else backend generates standalone Rust code from IR:

- Creates `pub struct Model;` with `pub fn predict(&self, features: &[f32]) -> f32`
- Generates one `#[inline(always)] fn tree_N()` per tree
- Recursively compiles `Node` enum into nested if/else blocks using `quote!` macro
- Emits aggregation: `base_score + tree_0(...) * w0 + tree_1(...) * w1 + ...`
- Applies post-transform: `(1.0 / (1.0 + (-raw).exp())) as f32` for logit
- **Conditionally** generates `bitset_contains()` helper only if categorical splits present

## Categorical Features: CRITICAL Implementation Details

**H2O MOJO vs ONNX handle categoricals completely differently:**

### MOJO (Bitset-based, native categorical splits)
- **Parser** (`src/parsers/tree_parser.rs`): Reads bitset from binary format (bitoff, nbits, data bytes)
- **IR**: `SplitKind::Categorical { bitoff, nbits, data }`
- **Codegen** (`src/backends/rust.rs:168-185`): Generates `bitset_contains()` calls
- **Branch semantics**: If value is **IN bitset** → follow **RIGHT child** (NOT left!)
  - Code uses `!bitset_contains(...)` for the left condition to match H2O semantics
  - This was a critical bug fixed in Feb 2025 - see git history

### ONNX (Label-encoded, numeric splits)
- H2O **automatically label-encodes** categoricals when exporting to ONNX
- Example: `blue=0, green=1, red=2, yellow=3` become numeric features
- Uses standard numeric threshold comparisons: `val <= 1.5`
- No special categorical handling needed - works with `SplitKind::Numeric`

**When debugging categorical predictions:**
1. Verify the format: MOJO uses bitsets, ONNX uses label encoding
2. Check branch direction: MOJO categorical splits go RIGHT when in bitset
3. Inspect bitset bytes in hex to understand membership encoding

## Special Cases and Edge Cases

### NaVsRest Splits
- **Purpose:** Split solely on whether value is NaN (not a numeric threshold)
- **MOJO format:** No split_value stored in binary (only 1 byte for split_col)
- **Parser:** Sets `split_value: f32::NAN` as sentinel
- **Codegen:** Emits `!val.is_nan()` (no threshold comparison)
- This was a critical parser bug fixed in Feb 2025

### Missing Value Routing
- Each split node has a `MissingDirection` enum
- **Left/Right:** NaN always goes to specified child
- **None:** No special NaN handling (follows normal comparison)
- **NaVsRest:** Split is ONLY about NaN vs non-NaN (non-NaN→left, NaN→right)

## Testing Strategy

### Unit Tests
- Located in same file as code under test (`#[cfg(test)] mod tests`)
- **`src/ir.rs`**: ~19 tests covering Forest/Tree/Node construction, categorical semantics, weight handling
- **`src/parsers/tree_parser.rs`**: ~18 tests covering binary parsing, NaVsRest nodes, categorical bitsets
- **`src/backends/rust.rs`**: 5 tests covering identity, logit, softmax codegen, categorical helper, odd-tree-count rejection
- Run with: `cargo test`

### Integration Tests
- Located in `tests/integration_test.rs` and `assets/tests/`
- Cover 17 scenarios across XGBoost, LightGBM, CatBoost, sklearn ONNX, and H2O MOJO
- All are `#[ignore]` - require Python environment and pre-generated model assets
- Regenerate all fixtures at once with `python scripts/generate_all_fixtures.py`, optionally filtered by framework (`... xgboost lightgbm catboost sklearn_onnx`)
- CI runs the non-H2O integration tests on every push: it pip-installs the training frameworks, regenerates fixtures, and runs `cargo test --test integration_test -- --ignored --skip test_h2o`
- Model generation scripts: `assets/tests/<format>/<scenario>/generate.py`
- **Note:** Prediction validation is now real - it compiles the generated `model.rs` with `rustc` at test time, pipes feature rows through stdin, and asserts predictions match ground truth within a specified tolerance.

### Running Integration Tests
```bash
# Generate model assets first (requires Python + h2o or sklearn)
cd assets/tests/xgboost/regression_numeric
uv run python generate.py

# Run integration tests
cargo test --test integration_test -- --include-ignored
```

## Code Generation with quote! and proc-macro2

The Rust backend uses the `quote!` macro to generate code as token streams:

```rust
use quote::quote;
use proc_macro2::TokenStream;

// Example: generating a comparison
let threshold = 0.5f32;
let code = quote! {
    if val < #threshold {
        left_branch
    } else {
        right_branch
    }
};
```

**Key points:**
- `#var` interpolates a variable into the token stream
- Generated code is returned as `String` via `.to_string()`
- Use `#(#vec),*` to expand iterators (e.g., bitset data bytes)

## Dependencies and Build System

### Core Dependencies
- `clap` - CLI argument parsing
- `anyhow` - Error handling
- `zip` - MOJO archive extraction
- `byteorder` - Little-endian binary parsing
- `quote`, `proc-macro2` - Rust code generation

### ONNX Protobuf Handling
- `prost` (runtime), `prost-build` (build-time)
- `build.rs` compiles `src/proto/onnx.proto` → `$OUT_DIR/onnx.rs`
- Accessed via `pub mod onnx { include!(...) }` in `main.rs`

### Optional Features
- `scorer` feature enables `polars_score` binary with polars + rayon for batch scoring
- Default build is minimal (just the transpiler)

## Adding Support for New Model Formats

1. **Create frontend:** `src/frontends/newformat.rs`
   - Implement `Frontend` trait: `fn parse(&self, path: &Path) -> Result<Forest>`
   - Parse format-specific metadata
   - Extract tree structures and convert to `ir::Node` recursively
   - Return `ir::Forest`

2. **Update main.rs:**
   - Add file extension detection (e.g., `.json` for XGBoost)
   - Instantiate frontend and call `parse()`

3. **Add integration test:**
   - Create `assets/tests/newformat/` directory
   - Include training script, test data, validation scripts
   - Organize outputs in `generated/` subdirectory

4. **Update documentation:**
   - Add to README.md supported formats
   - Update the roadmap in `.agents/PLAN.md` (local planning notes, not version controlled)

## Philosophy and Design Decisions

- **Zero runtime dependencies:** Generated Rust code has no external crates
- **Explicit over implicit:** All transformations visible in IR
- **f64 aggregation, f32 output:** Precision where needed, compact output
- **Inline everything:** `#[inline(always)]` for tree functions
- **Format-agnostic IR:** Easy to add new frontends and backends
- **Production-first:** Built for deployment (servers, edge, WASM)

## Key Files to Understand

- `src/main.rs` - CLI entry point, orchestrates pipeline
- `src/ir.rs` - Universal IR definitions (Forest, Node, SplitKind)
- `src/frontends/mojo.rs` - H2O MOJO parser
- `src/parsers/tree_parser.rs` - Low-level MOJO binary tree parsing
- `src/backends/rust.rs` - Rust code generation (especially `compile_node()`)
- `.agents/PLAN.md` - Detailed architecture and roadmap (local planning notes, gitignored)
