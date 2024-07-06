use crate::codegen::{CodeGen, CodeGenContext, CodeGenError};
use crate::ir::{self, FuncDef, IRNode};
use crate::types::{self, Type};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::process::Command;

pub fn translate_value(value: ir::Value) -> String {
    match value {
        ir::Value::Int32(num) => format!("INT32_C({})", num),
        ir::Value::Int64(num) => format!("INT64_C({})", num),
        ir::Value::UInt32(num) => format!("UINT32_C({})", num),
        ir::Value::UInt64(num) => format!("UINT64_C({})", num),
        ir::Value::Float32(num) => {
            let mut rep = format!("{}", &f32::to_string(&num));
            if !rep.contains(".") {
                rep.push_str(".0");
            }
            rep.push_str("F");
            rep
        }
        ir::Value::Float64(num) => {
            let mut rep = format!("{}", &f64::to_string(&num));
            if !rep.contains(".") {
                rep.push_str(".0");
            }
            rep
        }
        ir::Value::Bool(b) => {
            if b {
                format!("1")
            } else {
                format!("0")
            }
        }
        ir::Value::String(s) => format!("{}", s),
        ir::Value::Id(ident) => format!("{}", ident),
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
    code_buffer: Vec<String>,
    global_idx: usize,
    type_counter: usize,
    type_map: HashMap<types::Type, String>,
    func_name_map: HashMap<String, String>,
}

impl From<CodeGenContext> for CGenContext {
    fn from(ctx: CodeGenContext) -> Self {
        CGenContext {
            build_stack: ctx.build_stack.into_iter().rev().collect(),
            outfile: ctx.outfile,
            code_buffer: vec![],
            global_idx: 0,
            type_counter: 0,
            type_map: HashMap::new(),
            func_name_map: HashMap::new(),
        }
    }
}

impl CodeGen for CGenContext {
    fn gen(&mut self) -> Result<(), CodeGenError> {
        self.gen_includes()?;
        self.save_global_idx();
        let start = self.gen_globals();
        self.gen_program(start);

        let final_source = self.code_buffer.join(" ");
        let mut file = File::create(CGenContext::C_OUTPUT_FILENAME)
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;
        file.write_all(final_source.as_bytes())
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;

        let compile_cmd = Command::new("gcc")
            .arg(CGenContext::C_OUTPUT_FILENAME)
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
    pub const C_OUTPUT_FILENAME: &'static str = "out.c";

    fn add_code(&mut self, code: &str) {
        self.code_buffer.push(code.into());
    }

    fn add_global_code(&mut self, code: &str) {
        self.code_buffer.insert(self.global_idx, code.into());
        self.global_idx += 1;
    }

    fn save_global_idx(&mut self) {
        self.global_idx = self.code_buffer.len();
    }

    fn get_new_type_id(&mut self) -> usize {
        let new_type = self.type_counter;
        self.type_counter += 1;
        new_type
    }

    fn translate_type(&mut self, type_t: Type) -> String {
        match type_t.clone() {
            Type::Int32 => "int32_t".into(),
            Type::Int64 => "int64_t".into(),
            Type::UInt32 => "uint32_t".into(),
            Type::UInt64 => "uint64_t".into(),
            Type::Float32 => "float".into(),
            Type::Float64 => "double".into(),
            Type::Bool => "int32_t".into(),
            Type::String => "char*".into(),
            Type::Function(func) => match self.type_map.get(&type_t.clone()) {
                Some(val) => val.to_string(),
                None => {
                    let typedef_name = format!("_func_type_{}", self.get_new_type_id());
                    self.type_map.insert(type_t.clone(), typedef_name.clone());
                    let param_types: Vec<String> = func
                        .params_t
                        .iter()
                        .map(|t| self.translate_type(t.clone()))
                        .collect();
                    let return_type: String = self.translate_type(*func.return_t.clone());
                    let joined_params = param_types.join(",");
                    self.add_global_code(&format!(
                        "typedef {} (*{})({});",
                        return_type,
                        typedef_name.clone(),
                        joined_params
                    ));
                    typedef_name
                }
            },
            other => panic!("unknown type: {:?}", other),
        }
    }

    fn gen_includes(&mut self) -> Result<(), CodeGenError> {
        self.add_code("#include \"stdint.h\"\n");
        Ok(())
    }

    fn gen_globals(&mut self) -> usize {
        let mut idx = 0;
        // A well formed program must start with a globals section
        // which could be empty
        while (*self.build_stack.get(idx).unwrap()).clone() != IRNode::GlobalSection {
            idx += 1;
        }
        idx += 1;
        let end_of_globals = self
            .build_stack
            .iter()
            .enumerate()
            .find(|(_, ir_node)| matches!(ir_node, IRNode::EndGlobalSection))
            .unwrap()
            .0;
        self.gen_code(idx, end_of_globals) + 1
    }

    fn gen_program(&mut self, idx: usize) -> usize {
        self.add_code("int main(){");
        let new_idx = self.gen_code(idx, self.build_stack.len());
        self.add_code("}");
        new_idx
    }

    fn gen_code(&mut self, idx: usize, end_idx: usize) -> usize {
        let mut node_idx = idx;
        while node_idx < end_idx {
            node_idx = match self.build_stack.get(node_idx).unwrap() {
                IRNode::Term(_) => self.gen_term(node_idx).unwrap(),
                IRNode::Eval(_) => self.gen_eval(node_idx).unwrap(),
                IRNode::Label(_) => self.gen_label(node_idx).unwrap(),
                IRNode::Assign(assign) => self.gen_assign(node_idx, assign.clone()).unwrap(),
                IRNode::Reassign(reassign) => {
                    self.gen_reassign(node_idx, reassign.clone()).unwrap()
                }
                // If Statement
                IRNode::If(_) => self.gen_if(node_idx).unwrap(),
                IRNode::IfCase(_) => self.gen_if_case(node_idx).unwrap(),
                IRNode::ElseIfCase(_) => self.gen_else_if_case(node_idx).unwrap(),
                IRNode::ElseCase(_) => self.gen_else_case(node_idx).unwrap(),
                IRNode::EndIf(_) => self.gen_end_if(node_idx).unwrap(),
                // Function Definitions
                IRNode::FuncDef(def, scope_id) => {
                    self.func_name_map
                        .insert(scope_id.clone(), def.symbol.ident.clone());
                    self.gen_func_def(node_idx, def.clone()).unwrap()
                }
                IRNode::EndFuncDef(_) => self.gen_end_func_def(node_idx).unwrap(),
                // Return
                IRNode::Return => self.gen_return(node_idx).unwrap(),
                IRNode::GlobalSection => {
                    panic!("IRNode::GlobalSection should not be handled as code")
                }
                IRNode::EndGlobalSection => {
                    panic!("IRNode::EndGlobalSection should not be handled as code")
                }
            };
        }
        node_idx
    }

    fn gen_term(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx + 1)
    }

    fn gen_eval(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx + 1)
    }

    fn gen_label(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        Ok(idx + 1)
    }

    fn gen_assign(&mut self, idx: usize, assign: ir::Assign) -> Result<usize, CodeGenError> {
        let assignment_type = &self.translate_type(assign.type_t);
        self.add_code(assignment_type);
        self.add_code(&assign.symbol.ident.clone());
        self.add_code("=");
        match self.build_stack.get(idx - 1).unwrap() {
            IRNode::EndFuncDef(scope_id) => {
                self.add_code(&self.func_name_map.get(scope_id).unwrap().clone());
            }
            _ => {
                self.gen_expr(idx - 1)?;
            }
        };
        self.add_code(";");
        Ok(idx + 1)
    }

    fn gen_reassign(&mut self, idx: usize, reassign: ir::Reassign) -> Result<usize, CodeGenError> {
        self.add_code(&*reassign.symbol.ident.clone());
        self.add_code("=");
        self.gen_expr(idx - 1)?;
        self.add_code(";");
        Ok(idx + 1)
    }

    fn gen_expr(&mut self, idx: usize) -> Result<(), CodeGenError> {
        // Collect
        let expr: Vec<IRNode> = self
            .build_stack
            .iter()
            .rev()
            .skip(self.build_stack.len() - idx - 1)
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
                        ir::Func::Add(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} + {}", lhs, rhs)
                        }
                        ir::Func::Sub(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} - {}", lhs, rhs)
                        }
                        ir::Func::Mult(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} * {}", lhs, rhs)
                        }
                        ir::Func::Div(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} / {}", lhs, rhs)
                        }
                        ir::Func::Lt(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} < {}", lhs, rhs)
                        }
                        ir::Func::Gt(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} > {}", lhs, rhs)
                        }
                        ir::Func::Leq(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} <= {}", lhs, rhs)
                        }
                        ir::Func::Geq(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} >= {}", lhs, rhs)
                        }
                        ir::Func::Eq(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} == {}", lhs, rhs)
                        }
                        ir::Func::Neq(_) => {
                            let rhs = stack.pop().unwrap();
                            let lhs = stack.pop().unwrap();
                            format!("{} != {}", lhs, rhs)
                        }
                        ir::Func::Not(_) => {
                            let u = stack.pop().unwrap();
                            format!("!{}", u)
                        }
                        ir::Func::Neg(_) => {
                            let u = stack.pop().unwrap();
                            format!("-{}", u)
                        }
                        ir::Func::Func(sig) => {
                            let mut call: String = sig.symbol.ident.clone();
                            call.push_str("(");
                            let num_params = sig.params_t.len();
                            for i in 0..num_params {
                                call.push_str(&stack.pop().unwrap());
                                if i != num_params - 1 {
                                    call.push_str(", ");
                                }
                            }
                            call.push_str(")");
                            call
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
        Ok(idx + 1)
    }

    fn gen_if_case(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("if");
        self.add_code("(");
        self.gen_expr(idx - 1)?;
        self.add_code(")");
        self.add_code("{");
        Ok(idx + 1)
    }

    fn gen_else_if_case(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        self.add_code("else if");
        self.add_code("(");
        self.gen_expr(idx - 1)?;
        self.add_code(")");
        self.add_code("{");
        Ok(idx + 1)
    }

    fn gen_else_case(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        self.add_code("else");
        self.add_code("{");
        Ok(idx + 1)
    }

    fn gen_end_if(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        Ok(idx + 1)
    }

    fn gen_func_def(&mut self, idx: usize, def: FuncDef) -> Result<usize, CodeGenError> {
        let return_type = &self.translate_type(def.return_t);
        self.add_code(return_type);
        self.add_code(&def.symbol.ident);
        self.add_code("(");
        let num_params = def.params_t.clone().len();
        for (n, param) in def.params_t.into_iter().enumerate() {
            let param_type = &self.translate_type(param.1);
            self.add_code(param_type);
            self.add_code(&param.0);
            if n != num_params - 1 {
                self.add_code(",");
            }
        }
        self.add_code(")");
        self.add_code("{");
        Ok(idx + 1)
    }

    fn gen_end_func_def(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("}");
        Ok(idx + 1)
    }

    fn gen_return(&mut self, idx: usize) -> Result<usize, CodeGenError> {
        self.add_code("return");
        self.gen_expr(idx - 1)?;
        self.add_code(";");
        Ok(idx + 1)
    }
}
