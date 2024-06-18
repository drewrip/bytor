use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum Type {
    // Primitives
    Int64,
    Int32,
    UInt64,
    UInt32,
    Float64,
    Float32,
    String,
    Bool,
    // User defined types
    Function(FunctionType),
    Program(ProgramType),
    // Compiler and existence
    Unknown,
    Nil,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct FunctionType {
    pub params_t: Vec<Type>,
    pub return_t: Box<Type>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum WithType {
    Mut,
    Imm,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramType {
    pub with_t: Vec<WithType>,
}
