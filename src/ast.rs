use std::sync::Arc;

use crate::semantic;
use crate::types;

pub type Block = Vec<Arc<Stmt>>;

#[derive(Debug)]
pub enum Node {
    RootNode(Root),
    ProgramNode(Program),
    WithNode(With),
    WithVarNode(WithVar),
    BlockNode(Block),
    ExprNode(Expr),
    AssignOpNode(AssignOp),
    StmtNode(Stmt),
    ParamsNode(Params),
    FuncNode(Func),
    VarNode(Var),
    TypeNode(types::Type),
}

#[derive(Debug)]
pub struct Root {
    pub preblock: Block,
    pub program: Arc<Program>,
    pub postblock: Block,
}

#[derive(Debug, Clone)]
pub enum Program {
    NoWith(String, Block),
    With(String, With, Block),
}

pub type With = Vec<Arc<WithVar>>;

#[derive(Debug)]
pub enum WithVar {
    NonMut(String),
    Mut(String),
}

#[derive(Debug)]
pub enum Expr {
    Term(Arc<Term>),
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mult(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    Call(String, Vec<Arc<Expr>>),
}

#[derive(Debug)]
pub enum Term {
    Num(i32),
    Expr(Arc<Expr>),
}

#[derive(Debug, Clone)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MultAssign,
    DivAssign,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign(Arc<Var>, Arc<Expr>),
    Reassign(Arc<Var>, AssignOp, Arc<Expr>),
    Call(String, Vec<Arc<Expr>>),
    FuncDef(Arc<Func>),
}

pub type Params = Vec<Arc<Var>>;

#[derive(Debug)]
pub struct Func {
    pub ret_t: types::Type,
    pub params: Params,
    pub with: With,
    pub ident: String,
    pub block: Block,
}

#[derive(Debug)]
pub struct Var {
    pub type_t: types::Type,
    pub ident: String,
}
