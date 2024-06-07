use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::semantic;
use crate::symbol::{Symbol, Var};
use crate::types;

pub type Block = Vec<Arc<Stmt>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    RootNode(Arc<Root>),
    ProgramNode(Arc<Program>),
    WithNode(Arc<With>),
    WithVarNode(Arc<WithVar>),
    BlockNode(Arc<Block>),
    ExprNode(Arc<Expr>),
    AssignOpNode(AssignOp),
    StmtNode(Arc<Stmt>),
    ArgsNode(Arc<Args>),
    ParamsNode(Arc<Params>),
    FuncNode(Arc<Func>),
    TypeNode(types::Type),
    TermNode(Arc<Term>),
    SymbolNode(Symbol),
    VarNode(Arc<Var>),
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Root {
    pub preblock: Block,
    pub program: Arc<Program>,
    pub postblock: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Program {
    NoWith(Symbol, Block),
    With(Symbol, With, Block),
}

pub type With = Vec<Arc<WithVar>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WithVar {
    Imm(Symbol),
    Mut(Symbol),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Term(Arc<Term>),
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mult(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    Eq(Arc<Expr>, Arc<Expr>),
    Neq(Arc<Expr>, Arc<Expr>),
    Leq(Arc<Expr>, Arc<Expr>),
    Geq(Arc<Expr>, Arc<Expr>),
    LessThan(Arc<Expr>, Arc<Expr>),
    GreaterThan(Arc<Expr>, Arc<Expr>),
    Call(Symbol, Arc<Args>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    Id(String),
    Num(i32),
    Bool(bool),
    Expr(Arc<Expr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MultAssign,
    DivAssign,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfCase {
    pub condition: Arc<Expr>,
    pub block: Block,
    pub is_else: bool,
}

pub type IfCases = Vec<Arc<IfCase>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    Assign(Symbol, Arc<Var>, Arc<Expr>),
    Reassign(Symbol, Arc<Var>, AssignOp, Arc<Expr>),
    If(IfCases),
    Call(Symbol, Arc<Args>),
    FuncDef(Arc<Func>),
    Return(Arc<Expr>),
}

pub type Args = Vec<Arc<Expr>>;

pub type Params = Vec<Arc<Param>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub type_t: types::Type,
    pub ident: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Func {
    pub ret_t: types::Type,
    pub params: Params,
    pub with: With,
    pub ident: String,
    pub block: Block,
}
