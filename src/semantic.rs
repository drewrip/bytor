use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast;
use crate::symbol::{new_symbol, Symbol, Symbolic, Var};
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
pub struct SymbolTable {
    pub table: HashMap<Symbol, Var>,
}

pub fn new_empty_symbol_table() -> SymbolTable {
    SymbolTable {
        table: HashMap::new(),
    }
}

pub type SymbolStack = Vec<SymbolTable>;

#[derive(Debug)]
pub struct ProgramState {
    pub stack: SymbolStack,
    pub ast: Arc<ast::Root>,
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

        // Now we can 'recursively' check the bodies of the
        // globally defined variables, functions and 'programs'.
        self.check_global_definitions()?;
        Ok(())
    }

    fn program_signature_discovery(&mut self) -> Result<()> {
        let gen_prog_symbol = self.ast.program.get_symbol();
        if let Some(prog_symbol) = gen_prog_symbol {
            self.stack
                .first_mut()
                .expect("No base frame found!")
                .table
                .insert(
                    prog_symbol,
                    Var {
                        type_t: types::Type::UInt64,
                        name: String::from("prog"),
                    },
                );
        }
        Ok(())
    }

    fn global_ident_discovery(&mut self) -> Result<()> {
        // First pass: discover types and signatures of global identifiers
        let pre_idents: Vec<Symbol> = self
            .ast
            .preblock
            .iter()
            .filter_map(|stmt| stmt.get_symbol())
            .collect();

        let post_idents: Vec<Symbol> = self
            .ast
            .postblock
            .iter()
            .filter_map(|stmt| stmt.get_symbol())
            .collect();

        let base_frame = self.stack.first_mut().expect("No base frame!");

        for pre_ident in pre_idents {
            base_frame.table.insert(
                pre_ident.clone(),
                Var {
                    type_t: pre_ident.type_t,
                    name: String::from("foo"),
                },
            );
        }

        for post_ident in post_idents {
            base_frame.table.insert(
                post_ident.clone(),
                Var {
                    type_t: post_ident.type_t,
                    name: String::from("foo"),
                },
            );
        }
        Ok(())
    }

    fn check_global_definitions(&mut self) -> Result<()> {
        Ok(())
    }
}
