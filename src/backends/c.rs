use crate::codegen::{CodeGen, CodeGenContext, CodeGenError};
use crate::ir::{self, IRNode};
use crate::types::Type;
use anyhow::{bail, Error, Result};
use std::fs::File;
use std::io::Write;
use std::process::Command;

macro_rules! matches_variant {
    ($val:expr, $var:path) => {
        match $val {
            $var { .. } => true,
            _ => false,
        }
    };
}

pub fn translate_type(type_t: Type) -> String {
    match type_t {
        Type::Int32 => "int32_t",
        Type::Int64 => "int64_t",
        Type::UInt32 => "uint32_t",
        Type::UInt64 => "uint64_t",
        Type::Float32 => "float",
        Type::Float64 => "double",
        Type::Bool => "int32_t",
        Type::String => "char*",
        other => panic!("unknown type: {:?}", other),
    }
    .into()
}

pub struct CGenContext {
    build_stack: Vec<IRNode>,
    outfile: String,
    skip_validation: bool,
    code_buffer: Vec<String>,
}

impl From<CodeGenContext> for CGenContext {
    fn from(ctx: CodeGenContext) -> Self {
        CGenContext {
            build_stack: ctx.build_stack,
            outfile: ctx.outfile,
            skip_validation: ctx.skip_validation,
            code_buffer: vec![],
        }
    }
}

impl CodeGen for CGenContext {
    fn gen(&mut self) -> Result<(), CodeGenError> {
        self.gen_includes();
        self.gen_globals();
        self.gen_program(self.build_stack.len() - 1);

        let final_source = self.code_buffer.join(" ");
        let mut file =
            File::create("out.c").map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;
        file.write_all(final_source.as_bytes())
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
    fn add_code(&mut self, code: &str) {
        self.code_buffer.push(code.into());
    }

    fn gen_includes(&mut self) -> Result<(), CodeGenError> {
        self.add_code("#include \"stdint.h\"\n");
        Ok(())
    }

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
        self.add_code("int main(){");
        let mut node_idx = idx;
        while !self.build_stack.is_empty() && node_idx != 0 {
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
        self.add_code("return 0;");
        self.add_code("}");
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
        self.add_code(&translate_type(assign.type_t));
        self.add_code(&assign.symbol.ident.clone());
        self.add_code(" = 1;");
        Ok(idx - 1)
    }

    fn gen_reassign(&mut self, idx: usize, reassign: ir::Reassign) -> Result<usize, CodeGenError> {
        println!("{:?}", reassign);
        self.add_code(&*reassign.symbol.ident.clone());
        self.add_code(" = 1;");
        Ok(idx - 1)
    }

    fn gen_return(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }
}
