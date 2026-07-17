//! `bonsai verify`: transpile a model, compile the generated code with
//! rustc, score a CSV of feature rows, and compare against reference
//! predictions produced by the original framework.
//!
//! CSV contract (matches the integration-test fixtures):
//! - feature columns: every column not named `target`, `ground_truth`, or
//!   `ground_truth_proba_<c>`, in file order
//! - scalar models: a `ground_truth` column
//! - softmax models: one `ground_truth_proba_<c>` column per class
//! - empty cells or `nan` parse as NaN feature values

use crate::backends::rust::{self, CodegenOptions, Layout};
use crate::ir::{Forest, PostTransform};
use anyhow::{anyhow, bail, Context, Result};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// How many individual mismatches to print before summarizing.
const MAX_REPORTED_MISMATCHES: usize = 5;

pub fn run(forest: &Forest, data_path: &Path, tolerance: f32, layout: Layout) -> Result<()> {
    anyhow::ensure!(
        forest.catboost_metadata.is_none(),
        "verify does not support CatBoost categorical (CTR) models yet; \
         their harness needs string categorical inputs"
    );

    let n_classes = match forest.post_transform {
        PostTransform::Softmax { n_classes } => n_classes,
        _ => 1,
    };

    let csv = std::fs::read_to_string(data_path)
        .with_context(|| format!("Failed to read {:?}", data_path))?;
    let parsed = parse_csv(&csv, n_classes)?;
    anyhow::ensure!(!parsed.rows.is_empty(), "No data rows in {:?}", data_path);
    println!(
        "   > {} rows, {} features{}",
        parsed.rows.len(),
        parsed.n_features,
        if n_classes > 1 {
            format!(", {} classes", n_classes)
        } else {
            String::new()
        }
    );

    // --- Generate and compile ---
    let resolved = rust::resolve_layout(forest, layout)?;
    println!("   > code layout: {:?}", resolved);
    let code = rust::generate_with_options(
        forest,
        CodegenOptions {
            layout: resolved,
            no_std: false,
        },
    )?;

    let dir = tempfile::tempdir().context("Failed to create temp dir")?;
    std::fs::write(dir.path().join("model.rs"), &code)?;
    std::fs::write(dir.path().join("main.rs"), harness_source(n_classes))?;
    let harness_bin = dir.path().join("harness");

    let rustc_out = Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg("-O")
        .arg("-o")
        .arg(&harness_bin)
        .arg(dir.path().join("main.rs"))
        .output()
        .context("Failed to invoke rustc — is a Rust toolchain on PATH?")?;
    anyhow::ensure!(
        rustc_out.status.success(),
        "Generated code failed to compile:\n{}",
        String::from_utf8_lossy(&rustc_out.stderr)
    );
    println!("   > compiled generated code");

    // --- Score ---
    let mut child = Command::new(&harness_bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .context("Failed to run verify harness")?;
    {
        let stdin = child.stdin.as_mut().ok_or_else(|| anyhow!("no stdin"))?;
        for row in &parsed.rows {
            let line: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            writeln!(stdin, "{}", line.join(","))?;
        }
    }
    let output = child.wait_with_output()?;
    anyhow::ensure!(output.status.success(), "verify harness exited with error");

    let predictions = parse_predictions(&String::from_utf8_lossy(&output.stdout), n_classes)?;
    anyhow::ensure!(
        predictions.len() == parsed.rows.len(),
        "Prediction count ({}) does not match row count ({})",
        predictions.len(),
        parsed.rows.len()
    );

    // --- Compare ---
    let mut mismatches = 0usize;
    let mut max_abs_error = 0.0f32;
    for (i, (pred_row, ref_row)) in predictions.iter().zip(parsed.references.iter()).enumerate() {
        for (c, (pred, reference)) in pred_row.iter().zip(ref_row.iter()).enumerate() {
            let error = (pred - reference).abs();
            if error > max_abs_error {
                max_abs_error = error;
            }
            if error > tolerance {
                mismatches += 1;
                if mismatches <= MAX_REPORTED_MISMATCHES {
                    if n_classes > 1 {
                        println!(
                            "   ✗ row {} class {}: predicted {} expected {} (error {:.2e})",
                            i, c, pred, reference, error
                        );
                    } else {
                        println!(
                            "   ✗ row {}: predicted {} expected {} (error {:.2e})",
                            i, pred, reference, error
                        );
                    }
                }
            }
        }
    }
    if mismatches > MAX_REPORTED_MISMATCHES {
        println!(
            "   ✗ ... and {} more mismatches",
            mismatches - MAX_REPORTED_MISMATCHES
        );
    }

    let comparisons = parsed.rows.len() * n_classes;
    if mismatches > 0 {
        bail!(
            "{}/{} predictions exceed tolerance {} (max abs error {:.3e})",
            mismatches,
            comparisons,
            tolerance,
            max_abs_error
        );
    }
    println!(
        "✓ all {} predictions within {} (max abs error {:.3e})",
        comparisons, tolerance, max_abs_error
    );
    Ok(())
}

#[derive(Debug)]
struct ParsedData {
    rows: Vec<Vec<f32>>,
    references: Vec<Vec<f32>>,
    n_features: usize,
}

fn parse_csv(content: &str, n_classes: usize) -> Result<ParsedData> {
    let mut lines = content.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| anyhow!("Empty CSV"))?
        .split(',')
        .map(str::trim)
        .collect();

    let is_reference_col = |name: &str| {
        name == "target" || name == "ground_truth" || name.starts_with("ground_truth_proba_")
    };
    let feature_cols: Vec<usize> = header
        .iter()
        .enumerate()
        .filter(|(_, name)| !is_reference_col(name))
        .map(|(i, _)| i)
        .collect();
    anyhow::ensure!(!feature_cols.is_empty(), "No feature columns found");

    let reference_cols: Vec<usize> = if n_classes > 1 {
        (0..n_classes)
            .map(|c| {
                let name = format!("ground_truth_proba_{}", c);
                header.iter().position(|h| *h == name).ok_or_else(|| {
                    anyhow!("Missing column '{}' for a {}-class model", name, n_classes)
                })
            })
            .collect::<Result<_>>()?
    } else {
        vec![header
            .iter()
            .position(|h| *h == "ground_truth")
            .ok_or_else(|| anyhow!("Missing 'ground_truth' column"))?]
    };

    let mut rows = Vec::new();
    let mut references = Vec::new();
    for (lineno, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        anyhow::ensure!(
            parts.len() == header.len(),
            "Row {}: expected {} columns, got {}",
            lineno + 2,
            header.len(),
            parts.len()
        );

        let features: Vec<f32> = feature_cols
            .iter()
            .map(|&i| parse_feature(parts[i], header[i], lineno + 2))
            .collect::<Result<_>>()?;
        let refs: Vec<f32> = reference_cols
            .iter()
            .map(|&i| {
                parts[i].trim().parse::<f32>().with_context(|| {
                    format!("Row {}: invalid reference value '{}'", lineno + 2, parts[i])
                })
            })
            .collect::<Result<_>>()?;

        rows.push(features);
        references.push(refs);
    }

    Ok(ParsedData {
        n_features: feature_cols.len(),
        rows,
        references,
    })
}

fn parse_feature(raw: &str, column: &str, lineno: usize) -> Result<f32> {
    let v = raw.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("nan") {
        return Ok(f32::NAN);
    }
    v.parse::<f32>().map_err(|_| {
        anyhow!(
            "Row {}: column '{}' value '{}' is not numeric — verify supports \
             numeric features only (label-encode categoricals first)",
            lineno,
            column,
            v
        )
    })
}

/// Build the scoring harness: reads comma-separated f32 rows from stdin and
/// prints one prediction line per row.
fn harness_source(n_classes: usize) -> String {
    let predict = if n_classes > 1 {
        r#"let probs = model.predict_proba(&features);
        let strings: Vec<String> = probs.iter().map(|p| p.to_string()).collect();
        println!("{}", strings.join(","));"#
    } else {
        r#"println!("{}", model.predict(&features));"#
    };
    format!(
        r#"#[path = "model.rs"]
#[allow(dead_code)]
mod model;

use std::io::BufRead;

fn main() {{
    let model = model::Model;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {{
        let line = line.unwrap();
        if line.trim().is_empty() {{
            continue;
        }}
        let features: Vec<f32> = line
            .split(',')
            .map(|s| {{
                let t = s.trim();
                if t.is_empty() || t.eq_ignore_ascii_case("nan") {{
                    f32::NAN
                }} else {{
                    t.parse().unwrap_or(f32::NAN)
                }}
            }})
            .collect();
        {predict}
    }}
}}
"#
    )
}

fn parse_predictions(stdout: &str, n_classes: usize) -> Result<Vec<Vec<f32>>> {
    stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let values: Vec<f32> = line
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f32>()
                        .with_context(|| format!("Invalid prediction output '{}'", s))
                })
                .collect::<Result<_>>()?;
            anyhow::ensure!(
                values.len() == n_classes,
                "Expected {} values per prediction line, got {}",
                n_classes,
                values.len()
            );
            Ok(values)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_scalar() {
        let csv = "feat_0,feat_1,ground_truth\n0.5,nan,1.25\n,2.0,-0.5\n";
        let parsed = parse_csv(csv, 1).unwrap();
        assert_eq!(parsed.n_features, 2);
        assert_eq!(parsed.rows.len(), 2);
        assert!((parsed.rows[0][0] - 0.5).abs() < 1e-9);
        assert!(parsed.rows[0][1].is_nan());
        assert!(parsed.rows[1][0].is_nan());
        assert_eq!(parsed.references, vec![vec![1.25], vec![-0.5]]);
    }

    #[test]
    fn test_parse_csv_multiclass_and_target_skipped() {
        let csv = "feat_0,target,ground_truth_proba_0,ground_truth_proba_1\n1.0,0,0.75,0.25\n";
        let parsed = parse_csv(csv, 2).unwrap();
        assert_eq!(parsed.n_features, 1);
        assert_eq!(parsed.references, vec![vec![0.75, 0.25]]);
    }

    #[test]
    fn test_parse_csv_missing_reference_errors() {
        let csv = "feat_0\n1.0\n";
        assert!(parse_csv(csv, 1).is_err());
        let csv = "feat_0,ground_truth_proba_0\n1.0,1.0\n";
        let err = parse_csv(csv, 3).unwrap_err().to_string();
        assert!(err.contains("ground_truth_proba_1"), "got: {err}");
    }

    #[test]
    fn test_parse_csv_non_numeric_feature_errors() {
        let csv = "feat_0,cat,ground_truth\n1.0,red,0.5\n";
        let err = parse_csv(csv, 1).unwrap_err().to_string();
        assert!(err.contains("numeric features only"), "got: {err}");
    }

    #[test]
    fn test_parse_predictions_shapes() {
        assert_eq!(
            parse_predictions("1.5\n-0.25\n", 1).unwrap(),
            vec![vec![1.5], vec![-0.25]]
        );
        assert_eq!(
            parse_predictions("0.6,0.4\n", 2).unwrap(),
            vec![vec![0.6, 0.4]]
        );
        assert!(parse_predictions("0.6,0.4\n", 3).is_err());
        assert!(parse_predictions("abc\n", 1).is_err());
    }
}
