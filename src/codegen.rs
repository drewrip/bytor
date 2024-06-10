use crate::ir::IRNode;
use crate::symbol::{Symbol};
use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodeGenError {
    #[error("The resulting WASM failed validation: {0}")]
    ValidateFailed(String),
    #[error("Couldn't write output binary: {0}")]
    BinaryWrite(String),
    #[error("Couldn't compile binary: {0}")]
    CompilationFailed(String),
}

macro_rules! matches_variant {
    ($val:expr, $var:path) => {
        match $val {
            $var { .. } => true,
            _ => false,
        }
    };
}

pub struct SymbolAssignments {
    counter: u32,
    table: Vec<HashMap<Symbol, u32>>,
}

impl SymbolAssignments {
    pub fn spush(&mut self) {
        self.table.push(HashMap::new());
    }
    pub fn spop(&mut self) {
        self.table.pop();
    }
    pub fn lookup_assignment(&self, symbol: Symbol) -> Option<u32> {
        match self.table.iter().rev().find_map(|table| table.get(&symbol)) {
            Some(assignment) => Some(*assignment),
            None => None,
        }
    }
    pub fn give_assignment(&mut self, symbol: Symbol) -> u32 {
        let used_counter = self.counter;
        self.table
            .last_mut()
            .unwrap()
            .entry(symbol)
            .or_insert(used_counter);
        self.counter += 1;
        used_counter
    }
}

pub struct CodeGenContext {
    pub build_stack: Vec<IRNode>,
    pub outfile: String,
    pub skip_validation: bool,
}

pub fn new(build_stack: Vec<IRNode>, outfile: String, skip_validation: bool) -> CodeGenContext {
    CodeGenContext {
        build_stack: build_stack.into_iter().rev().collect(),
        outfile,
        skip_validation,
    }
}

pub trait CodeGen {
    fn gen(&mut self) -> Result<(), CodeGenError>;
}
