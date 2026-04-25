# 🌳 Bonsai: High-Performance Tree Inference in Pure Rust

**Project Codename:** `bonsai`
**Tagline:** "Trim your tree models for production."
**Core Philosophy:** Zero-dependency, static-binary inference for tree ensembles. Train in Python/R, deploy anywhere (Servers, Spark, WASM, Edge).

---

## 1. Supported Scope

Bonsai is strictly an **inference-only** engine. It does not support training.

### Primary Support (The "Happy Path")

* **XGBoost:** JSON format (`booster.save_model("model.json")`). Full support for `gbtree` (binary/multiclass).
* **LightGBM:** JSON dump (`bst.dump_model()`). Full support for standard decision trees.
* **Spark Integration:**
* **CLI Pipe:** `RDD.pipe()` support via standard streams (stdin/stdout).
* **Pandas UDF:** Arrow-backed vectorized inference for PySpark.



### Secondary Support (The "Transmute" Legacy)

* **ONNX:** Tree Ensemble operators (via `transmute` logic).
* **H2O MOJO:** Selected tree models (DRF/GBM).
* **CatBoost:** JSON export (future roadmap, requires oblivious tree backend).

---

## 2. Architecture & Application Layout

Bonsai is designed as a **Cargo Workspace** to keep the core logic lightweight while allowing for heavy optional features (like Python bindings).

### The Workspace Components

1. **`bonsai-core`** (The Engine)
* **Role:** Defines the `Model` enum, the `Predict` trait, and the in-memory tree structures.
* **Deps:** `ndarray` (linear algebra), `thiserror` (error handling), `serde` (serialization foundation).
* **Key Design:** Uses a **linear arena memory layout** (`Vec<Node>` instead of `Box<Node>`) for CPU cache locality and zero-allocation traversal.


2. **`bonsai-graft`** (The Importer)
* **Role:** Parsing logic. Converts external formats (JSON, ONNX protobufs) into `bonsai-core` structures.
* **Deps:** `serde_json`, `prost` (if ONNX), `quick-xml` (if PMML/MOJO).
* **Why separate?** If a user just wants to load a binary Bonsai model, they shouldn't need to compile the JSON parsing logic.


3. **`bonsai-cli`** (The Tool)
* **Role:** The end-user binary. Handles `stdin`/`stdout` piping, file inspection, and basic benchmarking.
* **Deps:** `clap` (args), `anyhow`, `csv` (fast parsing).


4. **`bonsai-py`** (The Spark Connector)
* **Role:** Python bindings for PySpark/Pandas users.
* **Deps:** `pyo3`, `arrow` (optional feature for zero-copy data transfer).



---

## 3. Directory Structure

```text
bonsai/
├── Cargo.toml              # Workspace definition
├── README.md
├── .gitignore
│
├── crates/
│   ├── bonsai-core/        # The heart of the library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tree.rs     # The 'Arena' memory layout
│   │       ├── api.rs      # Traits (Predict, Explain)
│   │       └── math.rs     # Sigmoid, Softmax, etc.
│   │
│   ├── bonsai-graft/       # The parsers (importers)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── xgboost.rs  # XGBoost JSON parser
│   │       ├── lightgbm.rs # LightGBM JSON parser
│   │       └── onnx.rs     # ONNX protobuf parser
│   │
│   ├── bonsai-cli/         # The command line tool
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── commands/
│   │       │   ├── inspect.rs
│   │       │   ├── pipe.rs
│   │       │   └── bench.rs
│   │       └── util.rs
│   │
│   └── bonsai-py/          # Python bindings (PyO3)
│       ├── Cargo.toml
│       ├── pyproject.toml  # Maturin config
│       └── src/
│           └── lib.rs
│
└── examples/
    ├── spark_pipe_job.py   # Example using RDD.pipe()
    └── simple_inference.rs

```

---

## 4. Dependencies Strategy

Keep the dependency tree shallow to ensure fast compilation and small binary sizes.

**`bonsai-core`:**

```toml
[dependencies]
ndarray = "0.15"         # For matrix operations
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
num-traits = "0.2"       # Generic math support

```

**`bonsai-graft`:**

```toml
[dependencies]
bonsai-core = { path = "../bonsai-core" }
serde_json = "1.0"
# Optional features to keep build light
prost = { version = "0.11", optional = true } # Only for ONNX

```

**`bonsai-cli`:**

```toml
[dependencies]
bonsai-core = { path = "../bonsai-core" }
bonsai-graft = { path = "../bonsai-graft" }
clap = { version = "4.0", features = ["derive"] }
csv = "1.2"              # Fast CSV parsing for the pipe
anyhow = "1.0"
jemallocator = "0.5"     # Optional: Better memory management for long-running pipes

```

---

## 5. What Should This Tool Output?

### A. The CLI Output (Commands)

**1. `bonsai inspect model.json**`

* **Goal:** Instant visibility for MLOps.
* **Output (Human Readable):**
```text
Type:       XGBoost (gbtree)
Objective:  binary:logistic
Features:   42 detected
Trees:      150
Max Depth:  6
Size:       4.2 MB
Missing:    True (Default Left)
✅ Model is valid and loadable.

```



**2. `bonsai pipe**`

* **Goal:** High-throughput streaming for Spark/Bash.
* **Input (stdin):** `1.0,0.5,NaN,3.2\n` (CSV or LDJSON)
* **Output (stdout):** `0.874\n` (Raw Score or Probability)
* **Behavior:**
* Must handle `NaN` gracefully (using the native missing direction).
* Must flush stdout appropriately to prevent Spark deadlocks.
* **Logging:** ALL logs must go to `stderr`. `stdout` is exclusively for data.



**3. `bonsai convert` (Optional)**

* **Goal:** Optimization.
* **Action:** Reads `model.json`, parses it, and saves a `.bon` file (bincode/rkyv format).
* **Why:** Loading a binary dump is 100x faster than parsing JSON. Great for Lambda cold starts.

### B. The Library API Output

The Rust API should be type-safe and explicit.

```rust
// The Result Type
pub type Prediction = Result<Array2<f32>, BonsaiError>;

// The Trait
pub trait TreeEnsemble {
    // Returns raw margins (before sigmoid/softmax)
    fn predict_raw(&self, features: &Array2<f32>) -> Prediction;
    
    // Returns probabilities (0.0 to 1.0)
    fn predict_proba(&self, features: &Array2<f32>) -> Prediction;
    
    // Returns feature contributions (SHAP approximation)
    fn explain(&self, features: &Array2<f32>) -> Result<Array3<f32>, BonsaiError>;
}

```

---

## 6. Development Workflow (Next Steps)

1. **Initialize the Workspace:**
`cargo new bonsai` (and setup the sub-crates).
2. **The "Spine":**
Define the `Node` struct in `bonsai-core`.
* *Tip:* Use `u32` indices for children, not pointers.


```rust
struct Node {
    feature_idx: u32,
    threshold: f32,
    left_child: u32,  // Index in the arena Vec
    right_child: u32,
    missing_dir: Direction,
    is_leaf: bool,
    leaf_value: f32,
}

```


3. **The "Graft":**
Write the `from_xgboost_json` parser first. It’s the highest value target.
4. **The "Shears":**
Build the `bonsai inspect` CLI command to verify you are parsing correctly.