use crate::semantic::SymbolStack;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("The resulting WASM failed validation: {0}")]
    ValidateFailed(String),
    #[error("Couldn't write output binary: {0}")]
    BinaryWrite(String),
    #[error("Couldn't compile binary: {0}")]
    CompilationFailed(String),
}

pub fn infer(gamma: SymbolStack) -> Result<(), TypeError> {
    Ok(())
}
