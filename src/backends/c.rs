use crate::ast::{self, Program};
use crate::codegen::{CodeGen, CodeGenContext, CodeGenError};
use crate::ir::{self, IRNode};
use crate::symbol::{new_symbol, Symbol};
use anyhow::{bail, Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

macro_rules! matches_variant {
    ($val:expr, $var:path) => {
        match $val {
            $var { .. } => true,
            _ => false,
        }
    };
}

pub struct CGenContext {
    build_stack: Vec<IRNode>,
    outfile: String,
    skip_validation: bool,
    code_buffer: String,
}

impl From<CodeGenContext> for CGenContext {
    fn from(ctx: CodeGenContext) -> Self {
        CGenContext {
            build_stack: ctx.build_stack,
            outfile: ctx.outfile,
            skip_validation: ctx.skip_validation,
            code_buffer: String::from(""),
        }
    }
}

impl CodeGen for CGenContext {
    fn gen(&mut self) -> Result<(), CodeGenError> {
        self.gen_globals();
        println!("len: {}", self.build_stack.len());
        self.gen_program(self.build_stack.len() - 1);

        let mut file =
            File::create("out.c").map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;
        file.write_all(&self.code_buffer.as_bytes())
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;

        let compile_cmd = Command::new("gcc")
            .arg("out.c")
            .arg("-o")
            .arg(self.outfile.clone())
            .output()
            .map_err(|err| CodeGenError::CompilationFailed(err.to_string()))?;

        if !compile_cmd.status.success() {
            return Err(CodeGenError::CompilationFailed(format!(
                "C compilation failed: {}",
                String::from_utf8(compile_cmd.stderr).unwrap()
            )));
        }

        Ok(())
    }
}

impl CGenContext {
    fn gen_globals(&mut self) {
        // For now, through out all global nodes
        while (*self.build_stack.last().unwrap()).clone()
            != IRNode::Label(ir::Label("_globals_start".into()))
        {
            self.build_stack.pop();
        }

        while (*self.build_stack.last().unwrap()).clone()
            != IRNode::Label(ir::Label("_globals_end".into()))
        {
            self.build_stack.pop();
        }

        // Pop the _globals_end label off
        self.build_stack.pop();
    }

    fn gen_program(&mut self, idx: usize) {
        self.code_buffer.push_str("int main(){");
        let mut node_idx = idx;
        while !self.build_stack.is_empty() && node_idx != 0 {
            println!("idx: {}", node_idx);
            node_idx = match self.build_stack.get(node_idx).unwrap() {
                IRNode::Term(term) => self.gen_term(node_idx).unwrap(),
                IRNode::Eval(eval) => self.gen_eval(node_idx).unwrap(),
                IRNode::IfCase(if_case) => self.gen_if(node_idx).unwrap(),
                IRNode::Label(label) => self.gen_label(node_idx).unwrap(),
                IRNode::Assign(assign) => self.gen_assign(node_idx, assign.clone()).unwrap(),
                IRNode::Reassign(reassign) => {
                    self.gen_reassign(node_idx, reassign.clone()).unwrap()
                }
                IRNode::Return(ret) => self.gen_return(node_idx).unwrap(),
                other => {
                    panic!("Unimplemented IRNode: {:?}", other);
                }
            }
        }
        self.code_buffer.push_str("return 0;");
        self.code_buffer.push_str("}");
    }

    fn gen_term(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }

    fn gen_eval(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }

    fn gen_if(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }

    fn gen_label(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }

    fn gen_assign(&mut self, idx: usize, assign: ir::Assign) -> Result<usize, CodeGenError> {
        println!("{:?}", assign);
        self.code_buffer.push_str("int ");
        self.code_buffer.push_str(&*assign.symbol.ident.clone());
        self.code_buffer.push_str(" = 1;");
        Ok(idx - 1)
    }

    fn gen_reassign(&mut self, idx: usize, reassign: ir::Reassign) -> Result<usize, CodeGenError> {
        println!("{:?}", reassign);
        self.code_buffer.push_str(&*reassign.symbol.ident.clone());
        self.code_buffer.push_str(" = 1;");
        Ok(idx - 1)
    }

    fn gen_return(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }
}
