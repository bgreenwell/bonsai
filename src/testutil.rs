//! Shared helpers for differential tests: compile generated model code with
//! rustc alongside a driver program and capture its output.

use std::process::Command;

/// Compile `model_code` + `driver` (a main.rs that includes "model.rs") in a
/// temp dir and return the harness stdout.
pub(crate) fn compile_and_run(model_code: &str, driver: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.rs"), model_code).unwrap();
    std::fs::write(dir.path().join("main.rs"), driver).unwrap();
    let bin = dir.path().join("harness");
    let out = Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg("-o")
        .arg(&bin)
        .arg(dir.path().join("main.rs"))
        .output()
        .expect("failed to invoke rustc");
    assert!(
        out.status.success(),
        "rustc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&bin).output().expect("failed to run harness");
    assert!(run.status.success());
    String::from_utf8(run.stdout).unwrap()
}

fn row_literal(row: &[f32]) -> String {
    let vals: Vec<String> = row
        .iter()
        .map(|v| {
            if v.is_nan() {
                "f32::NAN".to_string()
            } else {
                format!("{:?}f32", v)
            }
        })
        .collect();
    format!("&[{}]", vals.join(", "))
}

/// Driver printing `predict` output bits (one hex line per row).
pub(crate) fn scalar_driver(rows: &[Vec<f32>]) -> String {
    let literals: Vec<String> = rows
        .iter()
        .map(|r| format!("        {},", row_literal(r)))
        .collect();
    format!(
        r#"mod model {{ include!("model.rs"); }}
fn main() {{
    let rows: &[&[f32]] = &[
{}
    ];
    let m = model::Model;
    for r in rows {{
        println!("{{:08x}}", m.predict(r).to_bits());
    }}
}}
"#,
        literals.join("\n")
    )
}

/// Driver printing `predict_proba` output bits (one hex line per probability).
pub(crate) fn proba_driver(rows: &[Vec<f32>]) -> String {
    let literals: Vec<String> = rows
        .iter()
        .map(|r| format!("        {},", row_literal(r)))
        .collect();
    format!(
        r#"mod model {{ include!("model.rs"); }}
fn main() {{
    let rows: &[&[f32]] = &[
{}
    ];
    let m = model::Model;
    for r in rows {{
        for p in m.predict_proba(r) {{
            println!("{{:08x}}", p.to_bits());
        }}
    }}
}}
"#,
        literals.join("\n")
    )
}
