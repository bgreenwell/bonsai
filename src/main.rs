mod backends;
mod frontends;
mod inspector;
mod ir;
mod parsers;
mod verify;

/// Generated ONNX protobuf bindings (produced by prost-build from src/proto/onnx.proto).
#[allow(clippy::doc_overindented_list_items)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use anyhow::Context;
use clap::{Parser, Subcommand};
use frontends::Frontend;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "bonsai",
    about = "Convert tree-ensemble models to standalone Rust source code",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transpile a model to Rust source code
    Transpile {
        /// Path to the input model file (.zip for H2O MOJO, .onnx or .pb for ONNX, .json for XGBoost/LightGBM)
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Path for the generated Rust source file
        #[arg(short, long, value_name = "FILE", default_value = "model_generated.rs")]
        output: PathBuf,

        /// Code layout: auto picks arrays for large numeric-only forests,
        /// nested if/else otherwise
        #[arg(long, value_enum, default_value_t = LayoutArg::Auto)]
        layout: LayoutArg,

        /// Generate core-only code for no_std targets: softmax models expose
        /// predict_proba_into instead of Vec-returning predict_proba, and
        /// non-identity transforms call exp via the libm crate
        #[arg(long)]
        no_std: bool,
    },

    /// Verify a model end to end: transpile, compile with rustc, score a
    /// CSV, and compare against reference predictions
    Verify {
        /// Path to the input model file
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// CSV with feature columns plus ground_truth (scalar) or
        /// ground_truth_proba_<c> (multiclass) reference columns
        #[arg(short, long, value_name = "FILE")]
        data: PathBuf,

        /// Maximum absolute difference tolerated per prediction
        #[arg(short, long, default_value_t = 1e-5)]
        tolerance: f32,

        /// Code layout to verify
        #[arg(long, value_enum, default_value_t = LayoutArg::Auto)]
        layout: LayoutArg,
    },

    /// Inspect a model's structure and metadata
    Inspect {
        /// Path to the input model file (.zip for H2O MOJO, .onnx or .pb for ONNX, .json for XGBoost/LightGBM)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Show detailed tree structures
        #[arg(short, long)]
        trees: bool,

        /// Number of trees to show in detail (default: 3)
        #[arg(short, long, default_value = "3")]
        num_trees: usize,
    },
}

/// CLI mirror of `backends::rust::Layout` so the backend stays clap-free.
#[derive(Clone, Copy, clap::ValueEnum)]
enum LayoutArg {
    Auto,
    Ifelse,
    Array,
}

impl From<LayoutArg> for backends::rust::Layout {
    fn from(arg: LayoutArg) -> Self {
        match arg {
            LayoutArg::Auto => backends::rust::Layout::Auto,
            LayoutArg::Ifelse => backends::rust::Layout::IfElse,
            LayoutArg::Array => backends::rust::Layout::Array,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Transpile {
            input,
            output,
            layout,
            no_std,
        } => transpile_command(input, output, layout.into(), no_std),
        Commands::Verify {
            input,
            data,
            tolerance,
            layout,
        } => verify_command(input, data, tolerance, layout.into()),
        Commands::Inspect {
            input,
            trees,
            num_trees,
        } => inspect_command(input, trees, num_trees),
    }
}

fn verify_command(
    input: PathBuf,
    data: PathBuf,
    tolerance: f32,
    layout: backends::rust::Layout,
) -> anyhow::Result<()> {
    println!("🔍 bonsai verify: {:?} against {:?}", input, data);
    let forest = parse_model(&input)?;
    println!("   > {} trees in forest", forest.trees.len());
    verify::run(&forest, &data, tolerance, layout)
}

fn transpile_command(
    input: PathBuf,
    output: PathBuf,
    layout: backends::rust::Layout,
    no_std: bool,
) -> anyhow::Result<()> {
    println!("🌱 bonsai: converting {:?}", input);

    let forest = parse_model(&input)?;
    println!("   > {} trees in forest", forest.trees.len());

    // --- Generate Rust source ---
    let resolved = backends::rust::resolve_layout(&forest, layout)?;
    println!("   > code layout: {:?}", resolved);
    if no_std {
        println!("   > no_std mode: core-only output");
    }
    let options = backends::rust::CodegenOptions {
        layout: resolved,
        no_std,
    };
    let rust_code = backends::rust::generate_with_options(&forest, options)?;
    let rust_code = format!(
        "{}{}",
        provenance_header(&input, resolved, no_std)?,
        rust_code
    );
    println!("   > generated {} bytes of Rust source", rust_code.len());

    // --- Write output ---
    std::fs::write(&output, &rust_code)?;
    println!("✓ output written to {:?}", output);

    Ok(())
}

/// Provenance comment for generated code: tool version, source model file
/// and content hash, and codegen settings. Deliberately no timestamp so
/// regenerating an unchanged model produces byte-identical output.
fn provenance_header(
    input: &Path,
    layout: backends::rust::Layout,
    no_std: bool,
) -> anyhow::Result<String> {
    let bytes = std::fs::read(input).with_context(|| format!("Failed to read {:?}", input))?;
    let name = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| input.display().to_string());
    Ok(format!(
        "// Generated by bonsai v{}\n\
         // Source: {} (fnv1a64:{:016x})\n\
         // Layout: {:?}{}\n\n",
        env!("CARGO_PKG_VERSION"),
        name,
        fnv1a64(&bytes),
        layout,
        if no_std { ", no_std" } else { "" },
    ))
}

/// FNV-1a 64-bit content hash — for provenance, not security.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn inspect_command(input: PathBuf, show_trees: bool, num_trees: usize) -> anyhow::Result<()> {
    println!("🔍 bonsai inspect: {:?}\n", input.display());

    let forest = parse_model(&input)?;
    inspector::inspect(&forest, show_trees, num_trees);

    Ok(())
}

/// Read a .json file once, detect the framework from its top-level keys, and parse it.
fn detect_and_parse_json(input: &Path) -> anyhow::Result<ir::Forest> {
    let content =
        std::fs::read_to_string(input).with_context(|| format!("Failed to read {:?}", input))?;
    let root: serde_json::Value = serde_json::from_str(&content)
        .context("Failed to parse JSON — is this a valid model file?")?;

    if root.get("learner").is_some() {
        println!("   > Detected XGBoost JSON");
        frontends::xgboost::parse_json(&root)
    } else if root.get("tree_info").is_some() {
        println!("   > Detected LightGBM JSON");
        frontends::lightgbm::parse_json(&root)
    } else if root.get("oblivious_trees").is_some() {
        println!("   > Detected CatBoost JSON");
        frontends::catboost::parse_json(&root)
    } else {
        anyhow::bail!(
            "Unrecognized JSON model format in '{:?}'. \
             Expected XGBoost JSON (top-level 'learner' key), \
             LightGBM JSON (top-level 'tree_info' key), \
             or CatBoost JSON (top-level 'oblivious_trees' key).",
            input
        )
    }
}

/// Parse a model file into IR based on file extension.
fn parse_model(input: &Path) -> anyhow::Result<ir::Forest> {
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "zip" => frontends::mojo::MojoFrontend::new().parse(input),
        "onnx" | "pb" => frontends::onnx::OnnxFrontend::new().parse(input),
        "json" => detect_and_parse_json(input),
        _ => anyhow::bail!(
            "Unsupported file extension '.{}'. \
             Expected .zip (MOJO), .onnx/.pb (ONNX), or .json (XGBoost/LightGBM/CatBoost).",
            ext
        ),
    }
}
