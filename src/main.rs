mod backends;
mod frontends;
mod inspector;
mod ir;
mod parsers;

/// Generated ONNX protobuf bindings (produced by prost-build from src/proto/onnx.proto).
#[allow(clippy::doc_overindented_list_items)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use anyhow::Context;
use clap::{Parser, Subcommand};
use frontends::Frontend;
use std::path::PathBuf;

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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Transpile { input, output } => transpile_command(input, output),
        Commands::Inspect {
            input,
            trees,
            num_trees,
        } => inspect_command(input, trees, num_trees),
    }
}

fn transpile_command(input: PathBuf, output: PathBuf) -> anyhow::Result<()> {
    println!("🌱 bonsai: converting {:?}", input);

    let forest = parse_model(&input)?;
    println!("   > {} trees in forest", forest.trees.len());

    // --- Generate Rust source ---
    let rust_code = backends::rust::generate(&forest)?;
    println!("   > generated {} bytes of Rust source", rust_code.len());

    // --- Write output ---
    std::fs::write(&output, &rust_code)?;
    println!("✓ output written to {:?}", output);

    Ok(())
}

fn inspect_command(input: PathBuf, show_trees: bool, num_trees: usize) -> anyhow::Result<()> {
    println!("🔍 bonsai inspect: {:?}\n", input);

    let forest = parse_model(&input)?;
    inspector::inspect(&forest, show_trees, num_trees);

    Ok(())
}

/// Peek at a .json file to determine whether it is XGBoost or LightGBM, then parse it.
fn detect_and_parse_json(input: &PathBuf) -> anyhow::Result<ir::Forest> {
    let content =
        std::fs::read_to_string(input).with_context(|| format!("Failed to read {:?}", input))?;
    let probe: serde_json::Value = serde_json::from_str(&content)
        .context("Failed to parse JSON — is this a valid model file?")?;

    if probe.get("learner").is_some() {
        println!("   > Detected XGBoost JSON");
        frontends::xgboost::XgboostFrontend::new().parse(input)
    } else if probe.get("tree_info").is_some() {
        println!("   > Detected LightGBM JSON");
        frontends::lightgbm::LightgbmFrontend::new().parse(input)
    } else {
        anyhow::bail!(
            "Unrecognized JSON model format in '{:?}'. \
             Expected XGBoost JSON (top-level 'learner' key) \
             or LightGBM JSON (top-level 'tree_info' key).",
            input
        )
    }
}

/// Parse a model file into IR based on file extension
fn parse_model(input: &PathBuf) -> anyhow::Result<ir::Forest> {
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "zip" => {
            let fe = frontends::mojo::MojoFrontend::new();
            fe.parse(input)
        }
        "onnx" | "pb" => {
            let fe = frontends::onnx::OnnxFrontend::new();
            fe.parse(input)
        }
        "json" => detect_and_parse_json(input),
        _ => {
            anyhow::bail!(
                "Unsupported file extension '.{}'. Expected .zip (MOJO), .onnx/.pb (ONNX), or .json (XGBoost/LightGBM).",
                ext
            );
        }
    }
}
