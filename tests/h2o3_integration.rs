//! Integration tests for H2O-3 MOJO and ONNX conversions.
//!
//! Prerequisites:
//! 1. Run `assets/examples/h2o3/train_and_export.py` to generate models and test data
//! 2. Run `python assets/examples/h2o3/validate.py` to generate Rust model files
//!
//! These tests are marked with `#[ignore]` by default because they require
//! external setup. Run with: `cargo test h2o3 -- --ignored`

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const TOLERANCE: f32 = 1e-5;

/// Parse test_data.csv and return (features, ground_truth_predictions)
fn load_test_data() -> (Vec<Vec<f32>>, Vec<f32>) {
    let path = Path::new("assets/examples/h2o3/test_data.csv");
    let file = File::open(path).expect("test_data.csv not found. Run train_and_export.py first.");
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut ground_truth = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.unwrap();

        // Skip header row
        if idx == 0 {
            continue;
        }

        let values: Vec<&str> = line.split(',').collect();
        if values.len() < 12 {
            // Expected: 10 features + target + h2o_pred_proba
            continue;
        }

        // Parse features (columns 0..10)
        let mut row_features = Vec::with_capacity(10);
        for i in 0..10 {
            let val = values[i].trim();
            // Handle NaN and empty strings
            let parsed = if val.is_empty() || val.eq_ignore_ascii_case("nan") {
                f32::NAN
            } else {
                val.parse::<f32>().unwrap_or(f32::NAN)
            };
            row_features.push(parsed);
        }

        // Parse ground truth (last column: h2o_pred_proba)
        let gt = values[11]
            .trim()
            .parse::<f32>()
            .expect("Failed to parse ground truth");

        features.push(row_features);
        ground_truth.push(gt);
    }

    assert!(
        !features.is_empty(),
        "No test data loaded. Check test_data.csv format."
    );

    (features, ground_truth)
}

fn compare_predictions(
    predictions: &[f32],
    ground_truth: &[f32],
    name: &str,
) -> (f32, f32, usize) {
    assert_eq!(predictions.len(), ground_truth.len());

    let mut max_error = 0.0f32;
    let mut sum_error = 0.0f32;
    let mut n_mismatches = 0;

    for (i, (pred, truth)) in predictions.iter().zip(ground_truth.iter()).enumerate() {
        let abs_diff = (pred - truth).abs();
        max_error = max_error.max(abs_diff);
        sum_error += abs_diff;

        if abs_diff > TOLERANCE {
            n_mismatches += 1;
            if n_mismatches <= 5 {
                eprintln!(
                    "   Mismatch row {}: {:.6} vs {:.6} (diff={:.2e})",
                    i, pred, truth, abs_diff
                );
            }
        }
    }

    let mean_error = sum_error / predictions.len() as f32;

    eprintln!("\n{} vs ground truth:", name);
    eprintln!("   Max error:   {:.2e}", max_error);
    eprintln!("   Mean error:  {:.2e}", mean_error);
    eprintln!(
        "   Mismatches:  {} / {} (tolerance={})",
        n_mismatches,
        predictions.len(),
        TOLERANCE
    );

    (max_error, mean_error, n_mismatches)
}

#[test]
#[ignore] // Requires external setup: train_and_export.py + validate.py
fn test_h2o3_mojo_matches_ground_truth() {
    // Load generated MOJO model
    #[path = "../assets/examples/h2o3/h2o_mojo_model.rs"]
    mod mojo_model;

    let (features, ground_truth) = load_test_data();
    let model = mojo_model::Model;

    let mut predictions = Vec::with_capacity(features.len());
    for row in &features {
        predictions.push(model.predict(row));
    }

    let (max_error, _mean_error, n_mismatches) =
        compare_predictions(&predictions, &ground_truth, "MOJO");

    assert_eq!(
        n_mismatches, 0,
        "MOJO predictions differ from H2O-3 ground truth (max error: {:.2e})",
        max_error
    );
}

#[test]
#[ignore] // Requires external setup: train_and_export.py + validate.py
fn test_h2o3_onnx_matches_ground_truth() {
    // Load generated ONNX model
    #[path = "../assets/examples/h2o3/h2o_onnx_model.rs"]
    mod onnx_model;

    let (features, ground_truth) = load_test_data();
    let model = onnx_model::Model;

    let mut predictions = Vec::with_capacity(features.len());
    for row in &features {
        predictions.push(model.predict(row));
    }

    let (max_error, _mean_error, n_mismatches) =
        compare_predictions(&predictions, &ground_truth, "ONNX");

    assert_eq!(
        n_mismatches, 0,
        "ONNX predictions differ from H2O-3 ground truth (max error: {:.2e})",
        max_error
    );
}

#[test]
#[ignore] // Requires external setup: train_and_export.py + validate.py
fn test_h2o3_mojo_and_onnx_identical() {
    // Load both models
    #[path = "../assets/examples/h2o3/h2o_mojo_model.rs"]
    mod mojo_model;

    #[path = "../assets/examples/h2o3/h2o_onnx_model.rs"]
    mod onnx_model;

    let (features, _ground_truth) = load_test_data();
    let mojo = mojo_model::Model;
    let onnx = onnx_model::Model;

    let mut mojo_predictions = Vec::with_capacity(features.len());
    let mut onnx_predictions = Vec::with_capacity(features.len());

    for row in &features {
        mojo_predictions.push(mojo.predict(row));
        onnx_predictions.push(onnx.predict(row));
    }

    let (max_error, _mean_error, n_mismatches) =
        compare_predictions(&mojo_predictions, &onnx_predictions, "MOJO vs ONNX");

    assert_eq!(
        n_mismatches, 0,
        "MOJO and ONNX predictions differ (max error: {:.2e})",
        max_error
    );
}
