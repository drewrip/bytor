use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast;
use crate::types;

type Result<T> = std::result::Result<T, SemanticError>;

#[derive(Debug, Clone)]
pub enum SemanticError {
    SomeError,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantic Error")
    }
}

#[derive(Debug)]
pub struct Var {
    type_t: types::Type,
    name: String,
}

#[derive(Debug)]
pub struct Symbol {
    type_t: types::Type,
    name: String,
}

pub fn new_symbol(type_t: types::Type, name: String) -> Symbol {
    Symbol { type_t, name }
}

#[derive(Debug)]
pub struct SymbolTable {
    table: HashMap<Symbol, Var>,
}

pub fn new_empty_symbol_table() -> SymbolTable {
    SymbolTable {
        table: HashMap::new(),
    }
}

pub type SymbolStack = Vec<SymbolTable>;

#[derive(Debug)]
pub struct ProgramState {
    stack: SymbolStack,
    ast: Arc<ast::Root>,
}

pub fn new_state(ast: Arc<ast::Root>) -> ProgramState {
    ProgramState { stack: vec![], ast }
}

impl ProgramState {
    pub fn build(&mut self) -> Result<()> {
        let base_frame = new_empty_symbol_table();

        // First pass: discover types and signatures of global identifiers
        let pre_idents: Vec<Symbol> = self
            .ast
            .preblock
            .iter()
            .filter_map(|stmt| match (*stmt.clone()).clone() {
                ast::Stmt::Assign(left_var, _) => {
                    Some(new_symbol(left_var.type_t.clone(), left_var.ident.clone()))
                }
                ast::Stmt::FuncDef(func) => {
                    Some(new_symbol(func.ret_t.clone(), func.ident.clone()))
                }
                _ => None,
            })
            .collect();

        let post_idents: Vec<Symbol> = self
            .ast
            .postblock
            .iter()
            .filter_map(|stmt| match (*stmt.clone()).clone() {
                ast::Stmt::Assign(left_var, _) => {
                    Some(new_symbol(left_var.type_t.clone(), left_var.ident.clone()))
                }
                ast::Stmt::FuncDef(func) => {
                    Some(new_symbol(func.ret_t.clone(), func.ident.clone()))
                }
                _ => None,
            })
            .collect();

        Ok(())
    }
}
