//! End-to-end tests for `bonsai verify`. These use tiny hand-written models
//! with hand-computable predictions, so they need only rustc (no Python) and
//! run un-ignored.

use std::path::Path;
use std::process::Command;

// Single tree: feature[0] < 0.5 -> leaf 1.0, else leaf -1.0; base_score 0.
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

// Three single-leaf trees, one per class: raw scores 0.1, 0.2, 0.3.
const MULTICLASS_MODEL: &str = r#"{
  "learner": {
    "learner_model_param": {"base_score":"[0,0,0]","num_class":"3","num_feature":"1"},
    "objective": {"name": "multi:softprob"},
    "gradient_booster": {"model": {"trees": [
      {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.1],"default_left":[0]},
      {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.2],"default_left":[0]},
      {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.3],"default_left":[0]}
    ]}}
  }
}"#;

fn run_verify(dir: &Path, model: &str, csv: &str, extra_args: &[&str]) -> std::process::Output {
    let model_path = dir.join("model.json");
    let csv_path = dir.join("data.csv");
    std::fs::write(&model_path, model).unwrap();
    std::fs::write(&csv_path, csv).unwrap();

    Command::new(env!("CARGO_BIN_EXE_bonsai"))
        .arg("verify")
        .arg("--input")
        .arg(&model_path)
        .arg("--data")
        .arg(&csv_path)
        .args(extra_args)
        .output()
        .expect("failed to run bonsai verify")
}

#[test]
fn test_verify_scalar_pass() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "feature_0,feature_1,ground_truth\n0.1,0,1.0\n0.9,0,-1.0\nnan,0,-1.0\n";
    let out = run_verify(dir.path(), SCALAR_MODEL, csv, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "verify failed:\nstdout: {}\nstderr: {}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("all 3 predictions within"), "got: {stdout}");
}

#[test]
fn test_verify_scalar_mismatch_fails() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "feature_0,feature_1,ground_truth\n0.1,0,1.0\n0.9,0,5.0\n";
    let out = run_verify(dir.path(), SCALAR_MODEL, csv, &[]);
    assert!(!out.status.success(), "verify should fail on bad reference");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(combined.contains("row 1"), "got: {combined}");
    assert!(combined.contains("exceed tolerance"), "got: {combined}");
}

#[test]
fn test_verify_tolerance_flag() {
    let dir = tempfile::tempdir().unwrap();
    // Reference off by 0.5: fails at the default 1e-5, passes at 1.0.
    let csv = "feature_0,feature_1,ground_truth\n0.1,0,0.5\n";
    let out = run_verify(dir.path(), SCALAR_MODEL, csv, &[]);
    assert!(!out.status.success());
    let out = run_verify(dir.path(), SCALAR_MODEL, csv, &["--tolerance", "1.0"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn test_verify_multiclass_pass() {
    // Expected probabilities: softmax([0.1, 0.2, 0.3]) computed exactly as
    // the generated code does (f64, max-subtracted, cast to f32 at the end).
    let raw = [0.1f64, 0.2, 0.3];
    let max = raw.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = raw.iter().map(|r| (r - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| (e / sum) as f32).collect();

    let csv = format!(
        "feature_0,target,ground_truth_proba_0,ground_truth_proba_1,ground_truth_proba_2\n\
         0.0,2,{},{},{}\n",
        probs[0], probs[1], probs[2]
    );
    let dir = tempfile::tempdir().unwrap();
    let out = run_verify(dir.path(), MULTICLASS_MODEL, &csv, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "verify failed:\nstdout: {}\nstderr: {}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("all 3 predictions within"), "got: {stdout}");
}

#[test]
fn test_verify_interpreter_engine() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "feature_0,feature_1,ground_truth\n0.1,0,1.0\n0.9,0,-1.0\nnan,0,-1.0\n";
    let out = run_verify(dir.path(), SCALAR_MODEL, csv, &["--engine", "interpret"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("engine: interpreter"), "got: {stdout}");
    assert!(stdout.contains("all 3 predictions within"), "got: {stdout}");
    // The interpreter path never invokes rustc.
    assert!(!stdout.contains("compiled generated code"), "got: {stdout}");
}

#[test]
fn test_verify_array_layout() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "feature_0,feature_1,ground_truth\n0.1,0,1.0\n0.9,0,-1.0\n";
    let out = run_verify(dir.path(), SCALAR_MODEL, csv, &["--layout", "array"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("layout: Array"), "got: {stdout}");
}
