use crate::symbol::{new_symbol, Symbol};
use crate::types::Type;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IRNode {
    // Mutate State
    Assign(Assign),
    Reassign(Reassign),
    // If statements
    If(String),
    IfCase(String),
    ElseIfCase(String),
    ElseCase(String),
    EndIf(String),
    // Expression nodes
    Term(Term),
    Eval(Func),
    // Function Definition
    FuncDef(FuncDef, String),
    EndFuncDef(String),
    // Globals
    GlobalSection,
    EndGlobalSection,
    // Extra
    Return,
    Label(Label),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assign {
    pub type_t: Type,
    pub symbol: Symbol,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reassign {
    pub type_t: Type,
    pub symbol: Symbol,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IfCase {
    pub end_block_label: Label,
    pub end_if_label: Label,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Term {
    pub type_t: Type,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Func {
    Add(Signature),
    Sub(Signature),
    Mult(Signature),
    Div(Signature),
    Lt(Signature),
    Gt(Signature),
    Leq(Signature),
    Geq(Signature),
    Eq(Signature),
    Neq(Signature),
    Func(Signature),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signature {
    pub symbol: Symbol,
    pub params_t: Vec<Type>,
    pub return_t: Type,
}

pub fn new_sig(ident: &str, params_t: Vec<Type>, return_t: Type) -> Signature {
    Signature {
        symbol: new_symbol(ident.to_string()),
        params_t,
        return_t,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuncDef {
    pub symbol: Symbol,
    pub params_t: Vec<(String, Type)>,
    pub return_t: Type,
}

pub fn new_func_def(ident: &str, params_t: Vec<(String, Type)>, return_t: Type) -> FuncDef {
    FuncDef {
        symbol: new_symbol(ident.to_string()),
        params_t,
        return_t,
    }
}
