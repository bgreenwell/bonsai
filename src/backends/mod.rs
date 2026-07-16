use crate::ir::Forest;
use anyhow::Result;

/// A backend takes the universal IR and emits source code.
#[allow(dead_code)]
pub trait Backend {
    fn generate(forest: &Forest) -> Result<String>;
}

pub mod rust;
pub mod rust_array;
