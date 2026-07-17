//! End-to-end tests for `bonsai transpile --emit crate`. These generate a
//! full crate in a temp dir and run `cargo test` inside it, so they need a
//! cargo toolchain but no Python.

use std::path::Path;
use std::process::Command;

const SCALAR_MODEL: &str = r#"{
  "learner": {
    "learner_model_param": {"base_score":"0","num_class":"0","num_feature":"2"},
    "objective": {"name": "reg:squarederror"},
    "gradient_booster": {"model": {"trees": [{
      "left_children":  [1, -1, -1],
      "right_children": [2, -1, -1],
      "split_indices":  [0,  0,  0],
      "split_conditions": [0.5, 1.0, -1.0],
      "default_left":   [0,  0,  0]
    }]}}
  }
}"#;

fn run_transpile_crate(dir: &Path, model: &str, extra_args: &[&str]) -> std::process::Output {
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, model).unwrap();

    Command::new(env!("CARGO_BIN_EXE_bonsai"))
        .arg("transpile")
        .arg("--input")
        .arg(&model_path)
        .arg("--output")
        .arg(dir.join("scorer_crate"))
        .arg("--emit")
        .arg("crate")
        .args(extra_args)
        .output()
        .expect("failed to run bonsai transpile")
}

fn cargo_test_in(crate_dir: &Path) -> std::process::Output {
    Command::new("cargo")
        .arg("test")
        .arg("--offline")
        .arg("--quiet")
        .current_dir(crate_dir)
        .output()
        .expect("failed to run cargo test in generated crate")
}

#[test]
fn test_emit_crate_with_golden_data() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("data.csv"),
        "feature_0,feature_1,ground_truth\n0.1,0,1.0\n0.9,0,-1.0\nnan,0,-1.0\n",
    )
    .unwrap();

    let out = run_transpile_crate(
        dir.path(),
        SCALAR_MODEL,
        &["--data", dir.path().join("data.csv").to_str().unwrap()],
    );
    assert!(
        out.status.success(),
        "transpile failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let crate_dir = dir.path().join("scorer_crate");
    assert!(crate_dir.join("Cargo.toml").exists());
    assert!(crate_dir.join("src/lib.rs").exists());
    let golden = std::fs::read_to_string(crate_dir.join("tests/golden.rs")).unwrap();
    assert!(golden.contains("golden_predictions"));
    assert!(golden.contains("f32::NAN"), "NaN row should be baked in");

    let test_out = cargo_test_in(&crate_dir);
    assert!(
        test_out.status.success(),
        "generated crate tests failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&test_out.stdout),
        String::from_utf8_lossy(&test_out.stderr)
    );
}

#[test]
fn test_emit_crate_smoke_test_without_data() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_transpile_crate(dir.path(), SCALAR_MODEL, &[]);
    assert!(
        out.status.success(),
        "transpile failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let crate_dir = dir.path().join("scorer_crate");
    let golden = std::fs::read_to_string(crate_dir.join("tests/golden.rs")).unwrap();
    assert!(golden.contains("model_smoke"));

    let test_out = cargo_test_in(&crate_dir);
    assert!(
        test_out.status.success(),
        "generated crate tests failed:\nstderr: {}",
        String::from_utf8_lossy(&test_out.stderr)
    );
}

#[test]
fn test_data_without_crate_mode_errors() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("model.json");
    std::fs::write(&model_path, SCALAR_MODEL).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_bonsai"))
        .arg("transpile")
        .arg("--input")
        .arg(&model_path)
        .arg("--data")
        .arg("whatever.csv")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--emit crate"));
}
