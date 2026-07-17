//! `--emit crate`: write the generated model as a complete cargo crate.
//!
//! Layout:
//! - `Cargo.toml` (adds a `libm` dependency only for no_std models that
//!   need `exp`)
//! - `src/lib.rs`: the generated model, `#![no_std]` when requested
//! - `tests/golden.rs`: with `--data`, the CSV's rows and reference
//!   predictions baked in and asserted within the given tolerance;
//!   otherwise a minimal smoke test

use crate::ir::{Forest, Node, PostTransform};
use crate::verify::{parse_csv, ParsedData};
use anyhow::{Context, Result};
use std::path::Path;

pub fn write(
    forest: &Forest,
    code: &str,
    no_std: bool,
    out_dir: &Path,
    data: Option<&Path>,
    tolerance: f32,
) -> Result<()> {
    let name = crate_name(out_dir);
    let n_classes = match forest.post_transform {
        PostTransform::Softmax { n_classes } => n_classes,
        _ => 1,
    };

    std::fs::create_dir_all(out_dir.join("src"))
        .with_context(|| format!("Failed to create {:?}", out_dir.join("src")))?;
    std::fs::create_dir_all(out_dir.join("tests"))?;

    // --- Cargo.toml ---
    let needs_libm = no_std && !matches!(forest.post_transform, PostTransform::Identity);
    let mut manifest = format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        name
    );
    if needs_libm {
        manifest.push_str("\n[dependencies]\nlibm = \"0.2\"\n");
    }
    std::fs::write(out_dir.join("Cargo.toml"), manifest)?;

    // --- src/lib.rs ---
    // Comments may precede an inner attribute, so the provenance header in
    // `code` cannot lead; emit #![no_std] first.
    let lib = if no_std {
        format!("#![no_std]\n\n{}", code)
    } else {
        code.to_string()
    };
    std::fs::write(out_dir.join("src").join("lib.rs"), lib)?;

    // --- tests/golden.rs ---
    let test_source = match data {
        Some(data_path) => {
            let csv = std::fs::read_to_string(data_path)
                .with_context(|| format!("Failed to read {:?}", data_path))?;
            let parsed = parse_csv(&csv, n_classes)?;
            anyhow::ensure!(!parsed.rows.is_empty(), "No data rows in {:?}", data_path);
            println!(
                "   > baking {} golden rows into tests/golden.rs",
                parsed.rows.len()
            );
            golden_test_source(&name, &parsed, n_classes, no_std, tolerance)
        }
        None => smoke_test_source(&name, forest, n_classes, no_std),
    };
    std::fs::write(out_dir.join("tests").join("golden.rs"), test_source)?;

    println!("✓ crate '{}' written to {:?}", name, out_dir);
    Ok(())
}

/// Derive a valid crate name from the output directory name.
fn crate_name(out_dir: &Path) -> String {
    let raw = out_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut name: String = raw
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if name.is_empty() {
        name = "model".to_string();
    }
    if name.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        name.insert(0, 'm');
    }
    name
}

fn fmt_f32(v: f32) -> String {
    if v.is_nan() {
        "f32::NAN".to_string()
    } else if v == f32::INFINITY {
        "f32::INFINITY".to_string()
    } else if v == f32::NEG_INFINITY {
        "f32::NEG_INFINITY".to_string()
    } else {
        format!("{:?}f32", v)
    }
}

fn rows_literal(rows: &[Vec<f32>]) -> String {
    let items: Vec<String> = rows
        .iter()
        .map(|row| {
            let vals: Vec<String> = row.iter().map(|v| fmt_f32(*v)).collect();
            format!("        &[{}],", vals.join(", "))
        })
        .collect();
    items.join("\n")
}

fn golden_test_source(
    name: &str,
    parsed: &ParsedData,
    n_classes: usize,
    no_std: bool,
    tolerance: f32,
) -> String {
    let assert_block = if n_classes > 1 {
        let get_probs = if no_std {
            format!(
                "let mut probs = [0.0f32; {n}];\n        model.predict_proba_into(row, &mut probs);",
                n = n_classes
            )
        } else {
            "let probs = model.predict_proba(row);".to_string()
        };
        format!(
            r#"{get_probs}
        for (c, (pred, expected)) in probs.iter().zip(exp.iter()).enumerate() {{
            let error = (pred - expected).abs();
            assert!(
                error <= TOLERANCE,
                "row {{i}} class {{c}}: predicted {{pred}}, expected {{expected}} (error {{error:e}})"
            );
        }}"#
        )
    } else {
        r#"let pred = model.predict(row);
        let expected = exp[0];
        let error = (pred - expected).abs();
        assert!(
            error <= TOLERANCE,
            "row {i}: predicted {pred}, expected {expected} (error {error:e})"
        );"#
        .to_string()
    };

    format!(
        r#"// Golden predictions baked in by `bonsai transpile --emit crate --data ...`.

use {name}::Model;

const TOLERANCE: f32 = {tolerance:?};

const ROWS: &[&[f32]] = &[
{rows}
];

const EXPECTED: &[&[f32]] = &[
{expected}
];

#[test]
fn golden_predictions() {{
    let model = Model;
    for (i, (row, exp)) in ROWS.iter().zip(EXPECTED.iter()).enumerate() {{
        {assert_block}
    }}
}}
"#,
        rows = rows_literal(&parsed.rows),
        expected = rows_literal(&parsed.references),
    )
}

fn smoke_test_source(name: &str, forest: &Forest, n_classes: usize, no_std: bool) -> String {
    let n_features = max_feature_index(forest) + 1;
    let body = if n_classes > 1 {
        if no_std {
            format!(
                "let mut probs = [0.0f32; {n_classes}];\n    \
                 model.predict_proba_into(&[0.0f32; {n_features}], &mut probs);\n    \
                 assert!(probs.iter().all(|p| p.is_finite()));"
            )
        } else {
            format!(
                "let probs = model.predict_proba(&[0.0f32; {n_features}]);\n    \
                 assert_eq!(probs.len(), {n_classes});"
            )
        }
    } else {
        format!("assert!(model.predict(&[0.0f32; {n_features}]).is_finite());")
    };
    format!(
        r#"// Smoke test generated by `bonsai transpile --emit crate` (no --data given).

use {name}::Model;

#[test]
fn model_smoke() {{
    let model = Model;
    {body}
}}
"#
    )
}

fn max_feature_index(forest: &Forest) -> usize {
    fn walk(node: &Node) -> usize {
        match node {
            Node::Leaf { .. } => 0,
            Node::Split {
                feature_idx,
                left_child,
                right_child,
                ..
            } => (*feature_idx).max(walk(left_child)).max(walk(right_child)),
        }
    }
    forest
        .trees
        .iter()
        .map(|t| walk(&t.root))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_crate_name_sanitization() {
        assert_eq!(crate_name(&PathBuf::from("out/My-Model.v2")), "my_model_v2");
        assert_eq!(crate_name(&PathBuf::from("out/2fast")), "m2fast");
    }

    #[test]
    fn test_fmt_f32_special_values() {
        assert_eq!(fmt_f32(1.5), "1.5f32");
        assert_eq!(fmt_f32(f32::NAN), "f32::NAN");
        assert_eq!(fmt_f32(f32::INFINITY), "f32::INFINITY");
    }
}
