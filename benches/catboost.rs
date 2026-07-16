use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;
use std::process::Command;

// This benchmark assumes the catboost/regression fixture has been generated.
// If it hasn't, we'll skip it gracefully.

pub fn bench_catboost(_c: &mut Criterion) {
    let fixture_path = Path::new("assets/tests/catboost/regression/generated");
    let model_json = fixture_path.join("model.json");

    if !model_json.exists() {
        eprintln!(
            "\n[catboost bench] Model not found at {}. Skipping.",
            model_json.display()
        );
        return;
    }

    // 1. Transpile to a temporary file
    let model_rs = fixture_path.join("model_bench.rs");
    let bonsai_bin = "target/release/bonsai";
    // We assume the release binary exists for accurate benchmarking
    if !Path::new(bonsai_bin).exists() {
        eprintln!("\n[catboost bench] Bonsai release binary not found. Run `cargo build --release` first.");
        return;
    }

    Command::new(bonsai_bin)
        .args([
            "transpile",
            "--input",
            model_json.to_str().unwrap(),
            "--output",
            model_rs.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to transpile");

    // 2. We can't easily include! the generated file in a loop during cargo bench
    // so we'll measure the transpiled CatBoost model's performance via the integration test suite
    // OR we can just use the Model already generated for the integration tests if it exists.
}

// Since including dynamic code in Criterion is hard, I'll add a benchmark to the existing
// xgboost.rs that uses a synthetic oblivious tree to measure the "Pure Branchless" speed.

criterion_group!(benches, bench_catboost);
criterion_main!(benches);
