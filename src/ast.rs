use crate::ir;
use crate::symbol::{Symbol, Var};
use crate::types;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ASTError {
    #[error("Conversion from value in the AST to IR failed: {0}")]
    NoRep(String),
}

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
    // Unary Operators
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Call(Symbol, Box<Args>),
    LambdaFunc(LambdaFunc),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    Id(String),
    Num(Num),
    Bool(bool),
    String(String),
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
    pub return_t: types::Type,
    pub params: Params,
    pub ident: String,
    pub block: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaFunc {
    pub return_t: types::Type,
    pub params: Params,
    pub block: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Num {
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
}

impl TryFrom<Num> for ir::Value {
    type Error = ASTError;
    fn try_from(num: Num) -> Result<Self, Self::Error> {
        match num {
            Num::Int32(n) => Ok(ir::Value::Int32(n)),
            Num::Int64(n) => Ok(ir::Value::Int64(n)),
            Num::UInt32(n) => Ok(ir::Value::UInt32(n)),
            Num::UInt64(n) => Ok(ir::Value::UInt64(n)),
            Num::Float32(n) => Ok(ir::Value::Float32(n)),
            Num::Float64(n) => Ok(ir::Value::Float64(n)),
            other => Err(ASTError::NoRep(String::from("Unknown AST num"))),
        }
    }
}

impl TryFrom<Num> for types::Type {
    type Error = ASTError;
    fn try_from(num: Num) -> Result<Self, Self::Error> {
        match num {
            Num::Int32(n) => Ok(types::Type::Int32),
            Num::Int64(n) => Ok(types::Type::Int64),
            Num::UInt32(n) => Ok(types::Type::UInt32),
            Num::UInt64(n) => Ok(types::Type::UInt64),
            Num::Float32(n) => Ok(types::Type::Float32),
            Num::Float64(n) => Ok(types::Type::Float64),
            other => Err(ASTError::NoRep(String::from("Unknown AST num"))),
        }
    }
}
