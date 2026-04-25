mod backends;
mod frontends;
mod inspector;
mod ir;
mod parsers;

/// Generated ONNX protobuf bindings (produced by prost-build from src/proto/onnx.proto).
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

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
        /// Path to the input model file (.zip for H2O MOJO, .onnx or .pb for ONNX)
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Path for the generated Rust source file
        #[arg(short, long, value_name = "FILE", default_value = "model_generated.rs")]
        output: PathBuf,
    },

    /// Inspect a model's structure and metadata
    Inspect {
        /// Path to the input model file (.zip for H2O MOJO, .onnx or .pb for ONNX)
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
        Commands::Transpile { input, output } => {
            transpile_command(input, output)
        }
        Commands::Inspect { input, trees, num_trees } => {
            inspect_command(input, trees, num_trees)
        }
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

/// Parse a model file into IR based on file extension
fn parse_model(input: &PathBuf) -> anyhow::Result<ir::Forest> {
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "zip" => {
            let fe = frontends::mojo::MojoFrontend::new();
            fe.parse(input)
        }
        "onnx" | "pb" => {
            let fe = frontends::onnx::OnnxFrontend::new();
            fe.parse(input)
        }
        _ => {
            anyhow::bail!(
                "Unsupported file extension '.{}'. Expected .zip (MOJO), .onnx, or .pb (ONNX).",
                ext
            );
        }
    }
}
