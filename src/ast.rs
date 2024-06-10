use serde::{Deserialize, Serialize};
use crate::symbol::{Symbol, Var};
use crate::types;

pub type Block = Vec<Box<Stmt>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    RootNode(Box<Root>),
    ProgramNode(Box<Program>),
    WithNode(Box<With>),
    WithVarNode(Box<WithVar>),
    BlockNode(Box<Block>),
    ExprNode(Box<Expr>),
    AssignOpNode(AssignOp),
    StmtNode(Box<Stmt>),
    ArgsNode(Box<Args>),
    ParamsNode(Box<Params>),
    FuncNode(Box<Func>),
    TypeNode(types::Type),
    TermNode(Box<Term>),
    SymbolNode(Symbol),
    VarNode(Box<Var>),
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Root {
    pub preblock: Block,
    pub program: Box<Program>,
    pub postblock: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Program {
    NoWith(Symbol, Block),
    With(Symbol, With, Block),
}

pub type With = Vec<Box<WithVar>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WithVar {
    Imm(Symbol),
    Mut(Symbol),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Term(Box<Term>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Neq(Box<Expr>, Box<Expr>),
    Leq(Box<Expr>, Box<Expr>),
    Geq(Box<Expr>, Box<Expr>),
    LessThan(Box<Expr>, Box<Expr>),
    GreaterThan(Box<Expr>, Box<Expr>),
    Call(Symbol, Box<Args>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    Id(String),
    Num(i32),
    Bool(bool),
    Expr(Box<Expr>),
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
    pub condition: Box<Expr>,
    pub block: Block,
    pub is_else: bool,
}

pub type IfCases = Vec<Box<IfCase>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    Assign(Symbol, Box<Var>, Box<Expr>),
    Reassign(Symbol, Box<Var>, AssignOp, Box<Expr>),
    If(IfCases),
    Call(Symbol, Box<Args>),
    FuncDef(Box<Func>),
    Return(Box<Expr>),
}

pub type Args = Vec<Box<Expr>>;

pub type Params = Vec<Box<Param>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
