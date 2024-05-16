use crate::ast::Frame;
use crate::ast::{self, Program};
use crate::semantic;
use crate::symbol::{new_symbol, Symbol};
use anyhow::Result;
use std::collections::HashMap;
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

pub struct CodeGen {
    build_stack: Vec<ast::Frame>,
    outfile: String,
    skip_validation: bool,
    symbol_assignments: SymbolAssignments,
}

pub fn new(build_stack: Vec<ast::Frame>, outfile: String, skip_validation: bool) -> CodeGen {
    CodeGen {
        build_stack: build_stack.into_iter().rev().collect(),
        outfile,
        skip_validation,
        symbol_assignments: SymbolAssignments {
            counter: 0,
            table: vec![],
        },
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
        let main_params = vec![];
        let main_results = vec![ValType::I32]; // Return code
        types.function(main_params, main_results);
        module.section(&types);

        // Encode the function section.
        let mut functions = FunctionSection::new();
        let type_index = 0;
        functions.function(type_index);
        module.section(&functions);

        // Encode the export section.
        let mut exports = ExportSection::new();
        exports.export("main", ExportKind::Func, 0);
        module.section(&exports);

        self.gen_globals(&mut module);
        self.gen_program(&mut module);

        let wasm_output = module.finish();
        if !self.skip_validation {
            wasmparser::validate(&wasm_output)
                .map_err(|err| CodeGenError::ValidateFailed(err.message().into()))?;
        }
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
        let mut locals = vec![];

        let mut frame: Frame = self.build_stack.pop().unwrap();
        match frame.node {
            ast::Node::ProgramNode(_) => {
                self.symbol_assignments.spush();
                for symbol in frame.symbols.unwrap().table.into_keys() {
                    let assignment = self.symbol_assignments.give_assignment(symbol);
                    locals.push((assignment, ValType::I32));
                }
            }
            _ => {
                panic!("first node in gen_program is not a ProgramNode!");
            }
        };
        locals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut main_func = Function::new_with_locals_types(
            locals.iter().map(|local| local.1).collect::<Vec<ValType>>(),
        );
        while !self.build_stack.is_empty() {
            frame = self.build_stack.pop().unwrap();
            match frame.node {
                ast::Node::TermNode(term) => {
                    self.gen_term(module, &mut main_func, (*term).clone());
                }
                ast::Node::ExprNode(expr) => {
                    self.gen_expr(module, &mut main_func, (*expr).clone());
                }
                ast::Node::StmtNode(stmt) => {
                    self.gen_stmt(module, &mut main_func, (*stmt).clone());
                }
                node => {
                    eprintln!("Frame type not implemented for codegen: {:?}", node);
                }
            };
        }
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
            ast::Term::Id(ident) => {
                func.instruction(&Instruction::LocalGet(
                    self.symbol_assignments
                        .lookup_assignment(new_symbol(ident))
                        .unwrap(),
                ));
            } // TODO: More complicated
            ast::Term::Bool(boolean) => {
                func.instruction(&Instruction::I32Const(boolean as i32));
            }
            ast::Term::Expr(expr) => {} // TODO: Also more complicated, skip?
        };
    }

    pub fn gen_expr(&mut self, module: &mut Module, func: &mut Function, expr: ast::Expr) {
        match expr {
            ast::Expr::Add(lhs, rhs) => {
                func.instruction(&Instruction::I32Add);
            }
            ast::Expr::Sub(lhs, rhs) => {
                func.instruction(&Instruction::I32Sub);
            }
            ast::Expr::Mult(lhs, rhs) => {
                func.instruction(&Instruction::I32Mul);
            }
            ast::Expr::Div(lhs, rhs) => {
                func.instruction(&Instruction::I32DivS); // operands currently out of order
            }
            ast::Expr::Term(term) => {} // Skip
            _ => {
                eprintln!("gen expr not implemented: {:?}", expr);
            }
        };
    }

    pub fn gen_stmt(&mut self, module: &mut Module, func: &mut Function, stmt: ast::Stmt) {
        match stmt {
            ast::Stmt::Assign(symbol, var, expr) => {
                func.instruction(&Instruction::LocalSet(
                    self.symbol_assignments.lookup_assignment(symbol).unwrap(),
                ));
            }
            ast::Stmt::Reassign(symbol, var, assign_op, expr) => {
                let op = match assign_op {
                    ast::AssignOp::Assign => Instruction::Nop,
                    ast::AssignOp::AddAssign => Instruction::I32Add,
                    ast::AssignOp::SubAssign => Instruction::I32Sub,
                    ast::AssignOp::MultAssign => Instruction::I32Mul,
                    ast::AssignOp::DivAssign => Instruction::I32DivS,
                };
                if !matches_variant!(op, Instruction::Nop) {
                    func.instruction(&Instruction::LocalGet(
                        self.symbol_assignments
                            .lookup_assignment(symbol.clone())
                            .unwrap(),
                    ));
                    func.instruction(&op);
                }
                func.instruction(&Instruction::LocalSet(
                    self.symbol_assignments.lookup_assignment(symbol).unwrap(),
                ));
            }
            _ => {
                eprintln!("gen stmt not implemented: {:?}", stmt);
            }
        };
    }
}
