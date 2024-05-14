use crate::ast;
use crate::ast::Frame;
use crate::semantic;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodeGenError {
    #[error("The resulting WASM failed validation: {0}")]
    ValidateFailed(String),
    #[error("Couldn't write output binary: {0}")]
    BinaryWrite(String),
}

macro_rules! matches_variant {
    ($val:expr, $var:path) => {
        match $val {
            $var { .. } => true,
            _ => false,
        }
    };
}

pub struct CodeGen {
    build_stack: Vec<ast::Frame>,
    outfile: String,
}

pub fn new(build_stack: Vec<ast::Frame>, outfile: String) -> CodeGen {
    CodeGen {
        build_stack: build_stack.into_iter().rev().collect(),
        outfile,
    }
}

impl CodeGen {
    pub fn gen(&mut self) -> Result<(), CodeGenError> {
        println!("============================");
        println!("START OF CODEGEN");
        println!("============================");
        for (i, node) in self.build_stack.iter().rev().enumerate() {
            println!("    ({}) : {:?}", i + 1, node);
        }

        let mut module = Module::new();

        // Encode the type section.
        let mut types = TypeSection::new();
        let params = vec![ValType::I32, ValType::I32];
        let results = vec![ValType::I32];
        types.function(params, results);
        module.section(&types);

        // Encode the function section.
        let mut functions = FunctionSection::new();
        let type_index = 0;
        functions.function(type_index);
        module.section(&functions);

        // Encode the export section.
        let mut exports = ExportSection::new();
        exports.export("f", ExportKind::Func, 0);
        module.section(&exports);

        self.gen_globals(&mut module);
        self.gen_program(&mut module);

        let wasm_output = module.finish();
        wasmparser::validate(&wasm_output)
            .map_err(|err| CodeGenError::ValidateFailed(err.message().into()))?;
        let mut file = File::create(self.outfile.clone())
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;
        file.write_all(&wasm_output)
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;

        Ok(())
    }

    pub fn gen_globals(&mut self, module: &mut Module) {
        let mut next_frame: &Frame = self.build_stack.last().unwrap();
        while !matches_variant!(next_frame.node, ast::Node::ProgramNode) {
            // Do something with the node in the global scope
            self.build_stack.pop();
            next_frame = self.build_stack.last().unwrap();
        }
    }

    pub fn gen_program(&mut self, module: &mut Module) {
        let mut codes = CodeSection::new();
        let locals = vec![];
        let mut main_func = Function::new(locals);

        let mut frame: Frame = self.build_stack.pop().unwrap();
        while !self.build_stack.is_empty() {
            frame = self.build_stack.pop().unwrap();
            match frame.node {
                ast::Node::TermNode(term) => {
                    self.gen_term(module, &mut main_func, (*term).clone());
                }
                node => {
                    eprintln!("Frame type not implemented for codegen: {:?}", node);
                }
            };
        }
        println!("{:#?}", main_func);
        main_func.instruction(&Instruction::End);
        codes.function(&main_func);
        module.section(&codes);
    }

    pub fn gen_term(&mut self, module: &mut Module, func: &mut Function, term: ast::Term) {
        match term {
            ast::Term::Num(num) => {
                println!("term: {}", num);
                func.instruction(&Instruction::I32Const(num));
            }
            ast::Term::Id(ident) => {} // TODO: More complicated
            ast::Term::Bool(boolean) => {
                func.instruction(&Instruction::I32Const(boolean as i32));
            }
            ast::Term::Expr(expr) => {} // TODO: Also more complicated
        };
    }
}
