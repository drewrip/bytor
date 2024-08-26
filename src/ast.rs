use crate::ir;
use crate::symbol::{Symbol, Var};
use crate::types::Type;
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
    BlockNode(Box<Block>),
    ExprNode(Box<TypedExpr>),
    AssignOpNode(AssignOp),
    StmtNode(Box<Stmt>),
    ArgsNode(Box<Args>),
    ParamsNode(Box<Params>),
    FuncNode(Box<Func>),
    TypeNode(Type),
    TermNode(Box<TypedTerm>),
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
pub struct Program(pub Symbol, pub Block);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedExpr {
    pub type_t: Type,
    pub expr: Expr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Term(Box<TypedTerm>),
    Add(Box<TypedExpr>, Box<TypedExpr>),
    Sub(Box<TypedExpr>, Box<TypedExpr>),
    Mult(Box<TypedExpr>, Box<TypedExpr>),
    Div(Box<TypedExpr>, Box<TypedExpr>),
    Eq(Box<TypedExpr>, Box<TypedExpr>),
    Neq(Box<TypedExpr>, Box<TypedExpr>),
    Leq(Box<TypedExpr>, Box<TypedExpr>),
    Geq(Box<TypedExpr>, Box<TypedExpr>),
    LessThan(Box<TypedExpr>, Box<TypedExpr>),
    GreaterThan(Box<TypedExpr>, Box<TypedExpr>),
    // Unary Operators
    Not(Box<TypedExpr>),
    Neg(Box<TypedExpr>),
    Call(Symbol, Args),
    LambdaFunc(LambdaFunc),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedTerm {
    pub type_t: Type,
    pub term: Term,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    Id(String),
    Num(Num),
    Bool(bool),
    String(String),
    Expr(Box<TypedExpr>),
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
    pub condition: Box<TypedExpr>,
    pub block: Block,
    pub is_else: bool,
}

pub type IfCases = Vec<Box<IfCase>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    Assign(Symbol, Box<Var>, Box<TypedExpr>),
    Reassign(Symbol, Box<Var>, AssignOp, Box<TypedExpr>),
    If(IfCases),
    Call(Symbol, Args),
    FuncDef(Func),
    Return(Box<TypedExpr>),
    TypeDecl(TypeDecl),
    ImplTrait(ImplTrait),
}

pub type Args = Vec<Box<TypedExpr>>;

pub type Params = Vec<Box<Param>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Param {
    pub type_t: Type,
    pub ident: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Func {
    pub return_t: Type,
    pub params: Params,
    pub ident: String,
    pub block: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaFunc {
    pub return_t: Type,
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
        }
    }
}

impl TryFrom<Num> for Type {
    type Error = ASTError;
    fn try_from(num: Num) -> Result<Self, Self::Error> {
        match num {
            Num::Int32(_) => Ok(Type::Int32),
            Num::Int64(_) => Ok(Type::Int64),
            Num::UInt32(_) => Ok(Type::UInt32),
            Num::UInt64(_) => Ok(Type::UInt64),
            Num::Float32(_) => Ok(Type::Float32),
            Num::Float64(_) => Ok(Type::Float64),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeDecl {
    Trait(TraitDecl),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub ident: String,
    pub params_t: Vec<Type>,
    pub return_t: Type,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitDecl {
    pub ident: String,
    // List of generic names used
    pub generics: Vec<Type>,
    pub signatures: Vec<Signature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplTrait {
    pub ident: String,
    // List of generic names used
    pub generics: Vec<Type>,
    pub func_defs: Vec<Func>,
}
