use crate::ir::Forest;
use anyhow::Result;
use std::path::Path;

/// The interface that all model-format parsers must implement.
/// Every frontend produces the same `Forest` IR regardless of input format.
pub trait Frontend {
    fn parse(&self, path: &Path) -> Result<Forest>;
}

pub mod lightgbm;
pub mod mojo;
pub mod onnx;
pub mod xgboost;
