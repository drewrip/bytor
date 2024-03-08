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

#[derive(Debug, Clone)]
pub struct Var {
    type_t: types::Type,
    name: String,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Symbol {
    type_t: types::Type,
    name: String,
}

pub fn new_symbol(type_t: types::Type, name: String) -> Symbol {
    Symbol { type_t, name }
}

pub trait Symbolic {
    fn get_symbol(&self) -> Symbol;
}

impl Symbolic for ast::Program {
    fn get_symbol(&self) -> Symbol {
        match self {
            ast::Program::NoWith(name, _) => Symbol {
                name: name.clone(),
                type_t: types::Type::Program(types::ProgramType { with_t: vec![] }),
            },
            ast::Program::With(name, with_vars, _) => Symbol {
                name: name.clone(),
                type_t: types::Type::Program(types::ProgramType {
                    with_t: with_vars
                        .iter()
                        .map(|with_var| match *with_var.clone() {
                            ast::WithVar::NonMut(_) => types::WithType::Imm,
                            ast::WithVar::Mut(_) => types::WithType::Mut,
                        })
                        .collect(),
                }),
            },
        }
    }
}

#[derive(Debug, Clone)]
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
        self.stack.push(new_empty_symbol_table());
        // Find the signature of `program` blocks
        //  -> if there are none, we can abort compilation :)
        self.program_signature_discovery()?;
        // Discover the functions and variables in the global scope
        //  -> but, don't parse function bodies
        self.global_ident_discovery()?;

        Ok(())
    }

    fn program_signature_discovery(&mut self) -> Result<()> {
        let prog_symbol = self.ast.program.get_symbol();
        self.stack
            .first_mut()
            .expect("No base frame found!")
            .table
            .insert(
                prog_symbol,
                Var {
                    type_t: types::Type::UInt64,
                    name: String::from("foo"),
                },
            );
        Ok(())
    }

    fn global_ident_discovery(&mut self) -> Result<()> {
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

        let base_frame = self.stack.first_mut().expect("No base frame!");

        for pre_ident in pre_idents {
            base_frame.table.insert(
                pre_ident,
                Var {
                    type_t: types::Type::UInt64,
                    name: String::from("foo"),
                },
            );
        }

        for post_ident in post_idents {
            base_frame.table.insert(
                post_ident,
                Var {
                    type_t: types::Type::UInt64,
                    name: String::from("foo"),
                },
            );
        }
        Ok(())
    }
}
