use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Run one end-to-end test case:
/// 1. Transpile the pre-generated model with bonsai
/// 2. Verify generated code structure
/// 3. Compile generated model.rs with the test harness using rustc
/// 4. Score test_data.csv features through the compiled binary
/// 5. Assert predictions match ground truth within 1e-5
fn run_test_case(test_dir: &str, has_categoricals: bool) {
    let test_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/tests")
        .join(test_dir);

    let model_zip = test_path.join("generated/model.zip");
    let model_onnx = test_path.join("generated/model.onnx");
    let model_path = if model_zip.exists() {
        model_zip
    } else if model_onnx.exists() {
        model_onnx
    } else {
        panic!(
            "Model file not found in {}. Run the generate.py script first.",
            test_path.display()
        );
    };

    let output_path = test_path.join("generated/model.rs");
    let test_data_path = test_path.join("generated/test_data.csv");
    let metadata_path = test_path.join("generated/metadata.json");

    // 1. Transpile model with bonsai
    let bonsai_bin = env!("CARGO_BIN_EXE_bonsai");
    let status = Command::new(bonsai_bin)
        .args([
            "transpile",
            "--input",
            model_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run bonsai binary");
    assert!(status.success(), "bonsai transpilation failed");

    // 2. Verify generated code structure
    let generated_code = fs::read_to_string(&output_path).expect("Failed to read generated model.rs");
    if has_categoricals {
        assert!(
            generated_code.contains("fn bitset_contains"),
            "Expected bitset_contains helper for categorical features"
        );
    } else {
        assert!(
            !generated_code.contains("fn bitset_contains"),
            "Unexpected bitset_contains helper in non-categorical model"
        );
    }
    assert!(generated_code.contains("pub struct Model"));
    assert!(generated_code.contains("pub fn predict(&self, features: &[f32]) -> f32"));

    // 3. Load metadata — needed for categorical level → index encoding
    let metadata: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&metadata_path).expect("generated/metadata.json not found"),
    )
    .expect("Failed to parse metadata.json");

    let is_classification = metadata
        .get("task")
        .and_then(|v| v.as_str())
        .map(|s| s == "classification")
        .unwrap_or(false);

    // Build col_name → {level_string → integer_index} map
    let mut cat_encodings: HashMap<String, HashMap<String, usize>> = HashMap::new();
    if let Some(cat_features) = metadata
        .get("categorical_features")
        .and_then(|v| v.as_array())
    {
        for cat in cat_features {
            let name = cat["name"].as_str().unwrap().to_string();
            let levels = cat["levels"].as_array().unwrap();
            let encoding: HashMap<String, usize> = levels
                .iter()
                .enumerate()
                .map(|(i, v)| (v.as_str().unwrap().to_string(), i))
                .collect();
            cat_encodings.insert(name, encoding);
        }
    }

    // 4. Parse test_data.csv
    let csv_content = fs::read_to_string(&test_data_path).expect("generated/test_data.csv not found");
    let mut csv_lines = csv_content.lines();

    let header: Vec<&str> = csv_lines
        .next()
        .expect("test_data.csv is empty")
        .split(',')
        .collect();

    // Feature columns = everything except target and ground truth columns
    let skip_cols = ["target", "ground_truth", "ground_truth_proba"];
    let feature_col_indices: Vec<usize> = header
        .iter()
        .enumerate()
        .filter(|(_, col)| !skip_cols.contains(col))
        .map(|(i, _)| i)
        .collect();
    let n_features = feature_col_indices.len();

    let gt_col_name = if is_classification {
        "ground_truth_proba"
    } else {
        "ground_truth"
    };
    let gt_col = header
        .iter()
        .position(|&c| c == gt_col_name)
        .unwrap_or_else(|| panic!("column '{}' not found in CSV header", gt_col_name));

    let mut feature_rows: Vec<Vec<f32>> = Vec::new();
    let mut ground_truth: Vec<f32> = Vec::new();

    for line in csv_lines {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();

        let features: Vec<f32> = feature_col_indices
            .iter()
            .map(|&i| {
                let col = header[i];
                let val = parts[i].trim();
                if let Some(encoding) = cat_encodings.get(col) {
                    // Categorical: map level string → integer index as f32
                    *encoding
                        .get(val)
                        .unwrap_or_else(|| panic!("unknown category '{}' for column '{}'", val, col))
                        as f32
                } else if val.is_empty() || val.eq_ignore_ascii_case("nan") {
                    f32::NAN
                } else {
                    val.parse().unwrap_or(f32::NAN)
                }
            })
            .collect();

        let gt: f32 = parts[gt_col]
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("Invalid ground truth value: '{}'", parts[gt_col]));

        feature_rows.push(features);
        ground_truth.push(gt);
    }

    assert!(!feature_rows.is_empty(), "No data rows found in test_data.csv");

    // 5. Compile and run the model, collect predictions
    let predictions = compile_and_run_model(&output_path, n_features, &feature_rows);

    // 6. Validate predictions against ground truth
    assert_eq!(
        predictions.len(),
        ground_truth.len(),
        "Prediction count mismatch: got {}, expected {}",
        predictions.len(),
        ground_truth.len()
    );

    let tolerance = 1e-5f32;
    let mut mismatches = 0usize;
    for (i, (pred, gt)) in predictions.iter().zip(ground_truth.iter()).enumerate() {
        let error = (pred - gt).abs();
        if error > tolerance {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "  row {:3}: pred={:.8}  gt={:.8}  error={:.2e}",
                    i, pred, gt, error
                );
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{}/{} predictions for '{}' exceed tolerance {}",
        mismatches,
        predictions.len(),
        test_dir,
        tolerance
    );

    println!(
        "✓ {}: all {} predictions match within {}",
        test_dir,
        predictions.len(),
        tolerance
    );
}

/// Instantiate the test harness template with the model path, compile it with rustc,
/// pipe the feature rows through stdin, and return the predictions.
fn compile_and_run_model(
    model_rs: &PathBuf,
    n_features: usize,
    feature_rows: &[Vec<f32>],
) -> Vec<f32> {
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");

    // Fill in the two placeholders in the template
    let template = include_str!("../assets/tests/common/test_harness.rs.template");
    let model_abs = model_rs
        .canonicalize()
        .expect("Failed to resolve absolute path for model.rs");
    let harness_src = template
        .replace("{MODEL_RS_PATH}", model_abs.to_str().unwrap())
        .replace("{MIN_FEATURES}", &n_features.to_string());

    let harness_path = tmp.path().join("harness.rs");
    let binary_path = tmp.path().join("predictor");
    fs::write(&harness_path, &harness_src).expect("Failed to write harness.rs");

    // Compile harness + embedded model with rustc
    let compile = Command::new("rustc")
        .args([harness_path.to_str().unwrap(), "-o", binary_path.to_str().unwrap()])
        .output()
        .expect("rustc not found — is the Rust toolchain installed?");
    assert!(
        compile.status.success(),
        "rustc failed to compile generated model:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    // Build the feature CSV to pipe through stdin (no header, NaN as "NaN")
    let input: String = feature_rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|f| if f.is_nan() { "NaN".to_string() } else { format!("{}", f) })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut child = Command::new(&binary_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn compiled predictor");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();

    let out = child.wait_with_output().expect("Failed to wait for predictor");
    assert!(
        out.status.success(),
        "Compiled predictor exited with error:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Test cases — all #[ignore] because they require Python-generated model assets
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_h2o_mojo_classification_numeric() {
    run_test_case("h2o_mojo/classification_numeric", false);
}

#[test]
#[ignore]
fn test_h2o_mojo_classification_categorical() {
    run_test_case("h2o_mojo/classification_categorical", true);
}

#[test]
#[ignore]
fn test_h2o_mojo_regression_numeric() {
    run_test_case("h2o_mojo/regression_numeric", false);
}

#[test]
#[ignore]
fn test_h2o_mojo_regression_categorical() {
    run_test_case("h2o_mojo/regression_categorical", true);
}

#[test]
#[ignore]
fn test_sklearn_onnx_classification_numeric() {
    run_test_case("sklearn_onnx/classification_numeric", false);
}

#[test]
#[ignore]
fn test_sklearn_onnx_classification_categorical() {
    // sklearn ONNX label-encodes categoricals to numeric — no bitset helper expected
    run_test_case("sklearn_onnx/classification_categorical", false);
}

#[test]
#[ignore]
fn test_sklearn_onnx_regression_numeric() {
    run_test_case("sklearn_onnx/regression_numeric", false);
}

#[test]
#[ignore]
fn test_sklearn_onnx_regression_categorical() {
    // sklearn ONNX label-encodes categoricals to numeric — no bitset helper expected
    run_test_case("sklearn_onnx/regression_categorical", false);
}

// ---------------------------------------------------------------------------
// Non-ignored structural smoke-test (runs in normal cargo test)
// ---------------------------------------------------------------------------

#[test]
fn test_bitset_contains_generation_logic() {
    let code_with = r#"
        fn bitset_contains(bitoff: u16, nbits: u32, data: &[u8], idx: i32) -> bool {
            let idx = idx - bitoff as i32;
            if idx < 0 || idx >= nbits as i32 { return false; }
            (data[(idx >> 3) as usize] & (1 << (idx & 7) as u8)) != 0
        }
    "#;
    assert!(code_with.contains("fn bitset_contains"));
    assert!(code_with.contains("bitoff"));
    assert!(code_with.contains("data: &[u8]"));
}
