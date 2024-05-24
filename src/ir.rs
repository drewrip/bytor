use crate::ast::Node;
use crate::semantic;
use crate::symbol::Symbol;
use crate::types::Type;

#[derive(Debug, Clone)]
pub enum IRNode {
    Label(Label),
    Assign(Assign),
    Reassign(Reassign),
    IfCase(IfCase),
    Term(Term),
    Eval(Func),
    Return(Label),
}

#[derive(Debug, Clone)]
pub struct Label(pub String);

#[derive(Debug, Clone)]
pub struct Assign {
    pub type_t: Type,
    pub symbol: Symbol,
}

#[derive(Debug, Clone)]
pub struct Reassign {
    pub type_t: Type,
    pub symbol: Symbol,
}

#[derive(Debug, Clone)]
pub struct IfCase {
    pub end_block_label: Label,
    pub end_if_label: Label,
}

#[derive(Debug, Clone)]
pub struct Term {
    pub type_t: Type,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub enum Value {
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    Id(String),
}

#[derive(Debug, Clone)]
pub enum Func {
    Add,
    Sub,
    Mult,
    Div,
    Lt,
    Gt,
    Leq,
    Geq,
    Eq,
    Neq,
    DefFunc(Symbol),
}
