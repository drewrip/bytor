#[derive(Debug)]
pub enum Expr {
    Term(Box<Term>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
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
    Assign(Box<Var>, AssignOp, Box<Expr>),
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
}
