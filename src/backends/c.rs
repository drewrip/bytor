use crate::codegen::{CodeGen, CodeGenContext, CodeGenError};
use crate::ir::{self, IRNode};
use crate::types::Type;
use anyhow::{bail, Error, Result};
use std::fs::File;
use std::io::Write;
use std::process::Command;

use std::collections::HashMap;

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

pub fn translate_value(value: ir::Value) -> String {
    match value {
        ir::Value::Int32(num) => format!("INT32_C({})", num),
        ir::Value::Int64(num) => format!("INT64_C({})", num),
        ir::Value::UInt32(num) => format!("UINT32_C({})", num),
        ir::Value::UInt64(num) => format!("UINT64_C({})", num),
        ir::Value::Float32(num) => format!("{}F", num),
        ir::Value::Float64(num) => format!("{}", num),
        ir::Value::Bool(b) => {
            if b {
                format!("1")
            } else {
                format!("0")
            }
        }
        ir::Value::Id(ident) => format!("{}", ident),
        other => panic!("No value translation for: {:?}", other),
    }
}

pub fn is_expr_node(node: IRNode) -> bool {
    match node {
        IRNode::Term(_) => true,
        IRNode::Eval(_) => true,
        _ => false,
    }
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
                IRNode::Label(label) => self.gen_label(node_idx).unwrap(),
                IRNode::Assign(assign) => self.gen_assign(node_idx, assign.clone()).unwrap(),
                IRNode::Reassign(reassign) => {
                    self.gen_reassign(node_idx, reassign.clone()).unwrap()
                }
                // If Statement
                IRNode::If(if_case) => self.gen_if(node_idx).unwrap(),
                IRNode::IfCase(if_case) => self.gen_if_case(node_idx).unwrap(),
                IRNode::ElseIfCase(if_case) => self.gen_else_if_case(node_idx).unwrap(),
                IRNode::ElseCase(if_case) => self.gen_else_case(node_idx).unwrap(),
                IRNode::EndIf(if_case) => self.gen_end_if(node_idx).unwrap(),
                // Return
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

    fn gen_label(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }

    fn gen_assign(&mut self, idx: usize, assign: ir::Assign) -> Result<usize, CodeGenError> {
        self.add_code(&translate_type(assign.type_t));
        self.add_code(&assign.symbol.ident.clone());
        self.add_code("=");
        self.gen_expr(idx + 1);
        self.add_code(";");
        Ok(idx - 1)
    }

    fn gen_reassign(&mut self, idx: usize, reassign: ir::Reassign) -> Result<usize, CodeGenError> {
        self.add_code(&*reassign.symbol.ident.clone());
        self.add_code("=");
        self.gen_expr(idx + 1);
        self.add_code(";");
        Ok(idx - 1)
    }

    fn gen_expr(&mut self, idx: usize) -> Result<(), CodeGenError> {
        // Collect
        let mut expr: Vec<IRNode> = self
            .build_stack
            .iter()
            .skip(idx)
            .take_while(|node| is_expr_node((*node).clone()))
            .cloned()
            .collect();

        // Use a stack to build the expression
        let mut stack: Vec<String> = vec![];
        for node in expr.into_iter().rev() {
            match node {
                IRNode::Term(term) => stack.push(translate_value(term.value)),
                IRNode::Eval(eval) => {
                    let mut sub_expr: Vec<String> = vec!["(".into()];
                    let evaluated = match eval {
                        ir::Func::Add => {
                            format!("{} + {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Sub => {
                            format!("{} - {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Mult => {
                            format!("{} * {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Div => {
                            format!("{} / {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Lt => {
                            format!("{} < {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Gt => {
                            format!("{} > {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Leq => {
                            format!("{} <= {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Geq => {
                            format!("{} >= {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Eq => {
                            format!("{} == {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::Neq => {
                            format!("{} != {}", stack.pop().unwrap(), stack.pop().unwrap())
                        }
                        ir::Func::DefFunc(_) => {
                            panic!("User defined function calls not handled yet!")
                        }
                    };
                    sub_expr.push(evaluated);
                    sub_expr.push(")".into());
                    stack.push(sub_expr.join(" "))
                }
                _ => panic!("This shouldn't ever happen!"),
            };
        }
        self.add_code(&stack.pop().unwrap());
        Ok(())
    }

    fn gen_if(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }

    fn gen_if_case(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("if");
        self.add_code("(");
        self.gen_expr(idx + 1);
        self.add_code(")");
        self.add_code("{");
        Ok(idx - 1)
    }

    fn gen_else_if_case(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        self.add_code("else if");
        self.add_code("(");
        self.gen_expr(idx + 1);
        self.add_code(")");
        self.add_code("{");
        Ok(idx - 1)
    }

    fn gen_else_case(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        self.add_code("else");
        self.add_code("{");
        Ok(idx - 1)
    }

    fn gen_end_if(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        Ok(idx - 1)
    }

    fn gen_return(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx - 1)
    }
}
