// Batch scorer: loads a generated model and scores a CSV or Parquet file
// in parallel using Polars + Rayon.
//
// Requires: cargo build --features scorer --bin polars_score
//
// Usage:
//   # 1. First generate the model:
//   cargo run -- --input model.zip --output model_generated.rs
//   # 2. Then build and run the scorer:
//   cargo build --features scorer --bin polars_score
//   ./target/release/polars_score --input data.parquet --output predictions.parquet

// Include the generated model at compile time.  The path resolves to the
// package root (two directories up from src/bin/).
#[path = "../../model_generated.rs"]
mod model_generated;

use model_generated::Model;

use clap::Parser;
use polars::prelude::*;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about = "Score data using a bonsai generated model")]
struct Cli {
    /// Input file (CSV or Parquet)
    #[arg(short, long)]
    input: PathBuf,

    /// Output file (CSV or Parquet)
    #[arg(short, long)]
    output: PathBuf,

    /// Feature column names (comma-separated).
    /// If omitted, uses all columns except "target" and "prediction".
    #[arg(long, value_delimiter = ',')]
    features: Option<Vec<String>>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let model = Model;

    println!("🚀 polars_score");
    println!("   Input:  {:?}", cli.input);
    println!("   Output: {:?}", cli.output);

    let start_load = Instant::now();

    // --- Load data ---
    let input_str = cli.input.to_string_lossy().to_string();

    let mut df = if cli.input.extension().map_or(false, |ext| ext == "parquet")
        || cli.input.is_dir()
    {
        LazyFrame::scan_parquet(&input_str, ScanArgsParquet::default())?
            .collect()?
    } else {
        LazyCsvReader::new(&input_str).finish()?.collect()?
    };

    println!(
        "   Loaded {} rows in {:.2}s",
        df.height(),
        start_load.elapsed().as_secs_f32()
    );

    // --- Identify feature columns ---
    let feature_names: Vec<String> = match cli.features {
        Some(names) => names,
        None => df
            .get_column_names()
            .iter()
            .filter(|&&name| name != "target" && name != "prediction")
            .map(|&s| s.to_string())
            .collect(),
    };
    println!("   Using {} features.", feature_names.len());

    // --- Extract columns as Float32 ChunkedArrays for fast row access ---
    let feature_series: Vec<Series> = feature_names
        .iter()
        .map(|name| {
            df.column(name)
                .expect("Column not found")
                .cast(&DataType::Float32)
                .unwrap()
        })
        .collect();

    let feature_chunks: Vec<&ChunkedArray<Float32Type>> = feature_series
        .iter()
        .map(|s| s.f32().unwrap())
        .collect();

    // --- Score in parallel ---
    let start_score = Instant::now();
    let rows = df.height();
    let cols = feature_names.len();

    let scores: Vec<f32> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let mut row_vec = Vec::with_capacity(cols);
            for chunk in &feature_chunks {
                row_vec.push(chunk.get(i).unwrap_or(f32::NAN));
            }
            model.predict(&row_vec)
        })
        .collect();

    let duration = start_score.elapsed();
    println!("   Scoring complete in {:.4}s", duration.as_secs_f32());
    println!(
        "   Throughput: {:.0} rows/sec",
        rows as f64 / duration.as_secs_f64()
    );

    // --- Attach predictions and write output ---
    let score_series = Series::new("prediction", scores);
    df.with_column(score_series)?;

    let mut output_file = std::fs::File::create(&cli.output)?;
    if cli.output.extension().map_or(false, |ext| ext == "parquet") {
        ParquetWriter::new(&mut output_file).finish(&mut df)?;
    } else {
        CsvWriter::new(&mut output_file).finish(&mut df)?;
    }

    println!("   Saved predictions to {:?}", cli.output);
    Ok(())
}
