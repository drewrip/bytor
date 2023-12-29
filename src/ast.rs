pub type Block = Vec<Box<Stmt>>;

#[derive(Debug)]
pub struct Root {
    pub preblock: Block,
    pub program: Box<Program>,
    pub postblock: Block,
}

#[derive(Debug)]
pub enum Program {
    NoWith(String, Block),
    With(String, With, Block),
}

pub type With = Vec<Box<WithVar>>;

#[derive(Debug)]
pub enum WithVar {
    NonMut(String),
    Mut(String),
}

#[derive(Debug)]
pub enum Expr {
    Term(Box<Term>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Call(String, Vec<Box<Expr>>),
}

#[derive(Debug)]
pub enum Term {
    Num(i32),
    Expr(Box<Expr>),
}

#[derive(Debug)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MultAssign,
    DivAssign,
}

#[derive(Debug)]
pub enum Stmt {
    Assign(Box<Var>, Box<Expr>),
    Reassign(Box<Var>, AssignOp, Box<Expr>),
    Call(String, Vec<Box<Expr>>),
    FuncDef(Box<Func>),
}

pub type Params = Vec<Box<Var>>;

#[derive(Debug)]
pub struct Func {
    pub ret_t: Type,
    pub params: Params,
    pub with: With,
    pub ident: String,
    pub block: Block,
}

#[derive(Debug)]
pub struct Var {
    pub type_t: Type,
    pub ident: String,
}

#[derive(Debug)]
pub enum Type {
    Int64,
    Int32,
    UInt64,
    UInt32,
    Float64,
    Float32,
    Unknown,
    Nil,
}
