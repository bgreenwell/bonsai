#!/usr/bin/env python3
"""
XGBoost benchmark asset generator.

Trains an XGBoost binary-classification model, exports it in two formats
(JSON for bonsai, ONNX for onnxruntime / ort), benchmarks all Python-side
inference methods, and writes timing results to generated/python_results.json.

Usage:
    uv run python generate.py
"""

import json
import timeit
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_FEATURES = 10
N_SAMPLES = 1200
N_TEST = 200
N_TREES = 100
MAX_DEPTH = 5

SINGLE_REPS = 10_000
BATCH_SIZES = [10, 100, 1_000]
BATCH_REPS = 1_000

OUT = Path(__file__).parent / "generated"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

print("=" * 68)
print("XGBoost Benchmark - Binary Classification (10 numeric features)")
print("=" * 68)

print("\n[1/5] Generating synthetic data ...")
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=6,
    n_redundant=2,
    flip_y=0.01,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=N_TEST, random_state=SEED, stratify=y
)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
print(f"   train={X_train.shape[0]}  test={X_test.shape[0]}  features={N_FEATURES}")

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

print("\n[2/5] Training XGBClassifier ...")
clf = XGBClassifier(
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
    objective="binary:logistic",
    random_state=SEED,
    eval_metric="logloss",
)
clf.fit(X_train, y_train)
print(f"   {N_TREES} trees, depth {MAX_DEPTH}")

# ---------------------------------------------------------------------------
# 3. Export model
# ---------------------------------------------------------------------------

print("\n[3/5] Exporting model ...")

# JSON - consumed by bonsai
json_path = OUT / "model.json"
clf.get_booster().save_model(str(json_path))
print(f"   ✓ JSON  → {json_path}")

# ONNX - consumed by onnxruntime / ort crate
# onnxmltools has a dedicated XGBoost converter that produces a TreeEnsemble ONNX
# model with output[0]=labels (int64) and output[1]=probabilities (float32, shape [N,2]).
onnx_path = OUT / "model.onnx"
try:
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = convert_xgboost(
        clf.get_booster(),
        initial_types=[("float_input", FloatTensorType([None, N_FEATURES]))],
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"   ✓ ONNX  → {onnx_path}")
    onnx_available = True
except Exception as e:
    print(f"   ✗ ONNX export failed: {e}")
    onnx_available = False

# Save test features CSV
test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(N_FEATURES)])
test_df["target"] = y_test
test_df.to_csv(OUT / "test_features.csv", index=False)

# Write metadata
meta = {
    "n_features": N_FEATURES,
    "n_trees": N_TREES,
    "max_depth": MAX_DEPTH,
    "objective": "binary:logistic",
    "seed": SEED,
}
with open(OUT / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Python benchmarks
# ---------------------------------------------------------------------------

print("\n[4/5] Benchmarking Python inference ...")

# Use a single representative row - first test sample
sample_row = X_test[:1]  # shape (1, N_FEATURES)
booster = clf.get_booster()
dmat_single = xgb.DMatrix(sample_row)

results = {}

# --- Native XGBoost: single row ---
def _xgb_single():
    booster.predict(dmat_single)

# warm up
for _ in range(200):
    _xgb_single()

t = timeit.timeit(_xgb_single, number=SINGLE_REPS)
us = t / SINGLE_REPS * 1e6
results["xgb_native_single_us"] = us
print(f"   xgb native   single row : {us:8.2f} µs")

# --- Native XGBoost: batch ---
for bs in BATCH_SIZES:
    batch = X_test[:bs] if bs <= len(X_test) else np.tile(X_test, (bs // len(X_test) + 1, 1))[:bs]
    dmat_batch = xgb.DMatrix(batch)

    def _xgb_batch(d=dmat_batch):
        booster.predict(d)

    for _ in range(100):
        _xgb_batch()
    t = timeit.timeit(_xgb_batch, number=BATCH_REPS)
    us_total = t / BATCH_REPS * 1e6
    us_per = us_total / bs
    results[f"xgb_native_batch{bs}_total_us"] = us_total
    results[f"xgb_native_batch{bs}_per_row_us"] = us_per
    print(f"   xgb native   batch {bs:>5}: {us_total:8.2f} µs total  ({us_per:.3f} µs/row)")

# --- ONNX Runtime ---
if onnx_available:
    import onnxruntime as ort

    ort_session = ort.InferenceSession(str(OUT / "model.onnx"))
    input_name = ort_session.get_inputs()[0].name  # "float_input"

    def _ort_single(sess=ort_session, name=input_name, row=sample_row):
        # output[1] = probabilities (shape [1,2]); we just time the call
        sess.run(None, {name: row})

    for _ in range(200):
        _ort_single()

    t = timeit.timeit(_ort_single, number=SINGLE_REPS)
    us = t / SINGLE_REPS * 1e6
    results["ort_py_single_us"] = us
    print(f"   onnxruntime  single row : {us:8.2f} µs")

    for bs in BATCH_SIZES:
        batch = X_test[:bs] if bs <= len(X_test) else np.tile(X_test, (bs // len(X_test) + 1, 1))[:bs]

        def _ort_batch(sess=ort_session, name=input_name, b=batch):
            sess.run(None, {name: b})

        for _ in range(100):
            _ort_batch()
        t = timeit.timeit(_ort_batch, number=BATCH_REPS)
        us_total = t / BATCH_REPS * 1e6
        us_per = us_total / bs
        results[f"ort_py_batch{bs}_total_us"] = us_total
        results[f"ort_py_batch{bs}_per_row_us"] = us_per
        print(f"   onnxruntime  batch {bs:>5}: {us_total:8.2f} µs total  ({us_per:.3f} µs/row)")

# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------

print("\n[5/5] Saving results ...")
with open(OUT / "python_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"   ✓ {OUT / 'python_results.json'}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("─" * 68)
print(f"{'Method':<32} {'Single row':>14}  {'Per-row @ 1k':>14}")
print("─" * 68)

def row(label, single_key, batch_key):
    s = results.get(single_key)
    b = results.get(batch_key)
    s_str = f"{s:>12.2f} µs" if s is not None else f"{'n/a':>12}"
    b_str = f"{b:>12.3f} µs" if b is not None else f"{'n/a':>12}"
    print(f"  {label:<30} {s_str}  {b_str}")

row("XGBoost native (Python)",
    "xgb_native_single_us",
    "xgb_native_batch1000_per_row_us")
if onnx_available:
    row("ONNX Runtime (Python)",
        "ort_py_single_us",
        "ort_py_batch1000_per_row_us")
print("─" * 68)
print("  Run `cargo bench --bench xgboost` for bonsai + ort (Rust) numbers.")
print()

print("✓ Done. Next steps:")
print("  1. cargo run -- transpile \\")
print("          --input  examples/xgboost_benchmark/generated/model.json \\")
print("          --output examples/xgboost_benchmark/generated/model.rs")
print("  2. cargo bench --bench xgboost")
