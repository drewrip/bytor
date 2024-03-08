#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Type {
    // Primitives
    Int64,
    Int32,
    UInt64,
    UInt32,
    Float64,
    Float32,
    String,
    // User defined types
    Function(FunctionType),
    Program(ProgramType),
    // Compiler and existence
    Unknown,
    Nil,
}
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct FunctionType {
    pub return_t: Vec<Type>,
    pub params_t: Vec<Type>,
    pub with_t: Vec<WithType>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum WithType {
    Mut,
    Imm,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ProgramType {
    pub with_t: Vec<WithType>,
}
