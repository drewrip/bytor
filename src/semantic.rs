use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast;
use crate::symbol::{new_symbol, IdentMapping, Symbol, Symbolic, Var};
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
    pub fn slookup(&self, symbol: Symbol) -> Option<&Var> {
        self.stack
            .iter()
            .rev()
            .find_map(|table| table.table.get(&symbol))
    }

    // Assume this to be into the top of the stack
    pub fn sinsert(&mut self, symbol: Symbol, var: Var) -> Option<Var> {
        self.stack
            .last_mut()
            .expect("No frames in symbol table!")
            .table
            .insert(symbol, var)
    }

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
                .insert(prog_symbol.symbol, prog_symbol.var);
        }
        Ok(())
    }

    fn global_ident_discovery(&mut self) -> Result<()> {
        // First pass: discover types and signatures of global identifiers
        let pre_idents: Vec<IdentMapping> = self
            .ast
            .preblock
            .iter()
            .filter_map(|stmt| stmt.get_symbol())
            .collect();

        let post_idents: Vec<IdentMapping> = self
            .ast
            .postblock
            .iter()
            .filter_map(|stmt| stmt.get_symbol())
            .collect();

        let base_frame = self.stack.first_mut().expect("No base frame!");

        for pre_ident in pre_idents {
            base_frame.table.insert(pre_ident.symbol, pre_ident.var);
        }

        for post_ident in post_idents {
            base_frame.table.insert(post_ident.symbol, post_ident.var);
        }
        Ok(())
    }

    fn check_global_definitions(&mut self) -> Result<()> {
        let base_frame = self.stack.first_mut().expect("No base frame!");
        // All of the global statements
        let stmts: Vec<ast::Stmt> = self
            .ast
            .preblock
            .iter()
            .chain(self.ast.postblock.iter())
            .map(|stmt| (*stmt.clone()).clone())
            .collect();

        // These checks all depend on the state of the symbol
        // tables, so I have them in a for loop
        for stmt in stmts {}

        Ok(())
    }
}
