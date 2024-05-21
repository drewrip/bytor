use crate::ast::Node;
use crate::semantic;
use crate::symbol::Symbol;
use crate::types::Type;

#[derive(Debug, Clone)]
pub enum IRNode {
    Label(Label),
    Assign(Assign),
    IfCase(IfCase),
    Term(Term),
    Eval(Func),
}

#[derive(Debug, Clone)]
pub struct Label(String);

#[derive(Debug, Clone)]
pub struct Assign {
    type_t: Type,
    symbol: Symbol,
}

#[derive(Debug, Clone)]
pub struct IfCase {
    end_block_label: Label,
    end_if_label: Label,
}

#[derive(Debug, Clone)]
pub struct Term {
    type_t: Type,
    value: usize, // TODO: will need a generic value
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
}
