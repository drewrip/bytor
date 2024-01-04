use std::fmt;
use std::collections::HashMap;
use std::sync::Arc;

use crate::ast;

type Result<T> = std::result::Result<T, SemanticError>;

#[derive(Debug, Clone)]
pub enum SemanticError{
    SomeError,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantic Error")
    }
}

#[derive(Debug)]
pub enum Type {
    Unknown,
}

#[derive(Debug)]
pub struct Var {
    type_t: Type,
    name: String,
}

#[derive(Debug)]
pub struct Symbol {
    type_t: Type,
    name: String,
}

#[derive(Debug)]
pub struct SymbolTable {
    table: HashMap<Symbol, Var>,
}

pub type SymbolStack = Vec<SymbolTable>;

#[derive(Debug)]
pub struct CompilerState {
    stack: SymbolStack,
    ast: Arc<ast::Root>,
}

pub fn new_state(ast: Arc<ast::Root>) -> CompilerState {
    CompilerState {
        stack: vec![],
        ast: ast,
    }
}

impl CompilerState {
    pub fn check(&self) -> Result<()> {
        self.check_prepost(
            self.ast.preblock.clone(),
            self.ast.preblock.clone()
        )?;
        self.check_program(self.ast.program.clone())?;
        Ok(())
    }

    fn check_prepost(&self, pre: ast::Block, post: ast::Block) -> Result<()> { 
        let mut prepost = vec![];
        prepost.extend(pre.clone());
        prepost.extend(post.clone());
        self.check_block(prepost)?;
        Ok(())
    }

    fn check_program(&self, prog: Arc<ast::Program>) -> Result<()> {
        
        Ok(())
    }

    fn check_block(&self, block: ast::Block) -> Result<()> {
        for s in block {
            self.check_stmt(s)?;
        }
        Ok(())
    }

    fn check_stmt(&self, stmt: Arc<ast::Stmt>) -> Result<()> {
        match (*stmt).clone() {
            ast::Stmt::Assign(left, right) => self.check_assign(left, right),
            ast::Stmt::Reassign(left, op, right) => self.check_reassign(left, op, right),
            ast::Stmt::Call(id, args) => self.check_call(id, args),
            ast::Stmt::FuncDef(func) => self.check_funcdef(func),
        };
        Ok(())
    }


    fn check_assign(&self, left: Arc<ast::Var>, right: Arc<ast::Expr>) -> Result<()> {
        Ok(())
    }

    fn check_reassign(&self, left: Arc<ast::Var>, op: ast::AssignOp, right: Arc<ast::Expr>) -> Result<()> {
        Ok(())
    }

    fn check_call(&self, id: String, args: Vec<Arc<ast::Expr>>) -> Result<()> {
        Ok(())
    }

    fn check_funcdef(&self, func: Arc<ast::Func>) -> Result<()> {

        Ok(())
    }
}
