use crate::ast;
use crate::types;

#[derive(Debug, Clone)]
pub struct Var {
    pub type_t: types::Type,
    pub name: String,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Symbol {
    pub type_t: types::Type,
    pub name: String,
}

pub fn new_symbol(type_t: types::Type, name: String) -> Symbol {
    Symbol { type_t, name }
}

pub trait Symbolic {
    fn get_symbol(&self) -> Option<Symbol>;
}

impl Symbolic for ast::Program {
    fn get_symbol(&self) -> Option<Symbol> {
        match self {
            ast::Program::NoWith(name, _) => Some(Symbol {
                name: name.clone(),
                type_t: types::Type::Program(types::ProgramType { with_t: vec![] }),
            }),
            ast::Program::With(name, with_vars, _) => Some(Symbol {
                name: name.clone(),
                type_t: types::Type::Program(types::ProgramType {
                    with_t: with_vars
                        .iter()
                        .map(|with_var| match *with_var.clone() {
                            ast::WithVar::Imm(_) => types::WithType::Imm,
                            ast::WithVar::Mut(_) => types::WithType::Mut,
                        })
                        .collect(),
                }),
            }),
        }
    }
}

impl Symbolic for ast::Stmt {
    fn get_symbol(&self) -> Option<Symbol> {
        match self {
            ast::Stmt::Assign(var, expr) => Some(Symbol {
                name: var.name.clone(),
                type_t: var.type_t.clone(),
            }),
            ast::Stmt::FuncDef(func) => Some(Symbol {
                name: func.ident.clone(),
                type_t: types::Type::Function(types::FunctionType {
                    return_t: vec![func.ret_t.clone()],
                    params_t: func.params.iter().map(|var| var.type_t.clone()).collect(),
                    with_t: func
                        .with
                        .iter()
                        .map(|with_var| match *with_var.clone() {
                            ast::WithVar::Imm(_) => types::WithType::Imm,
                            ast::WithVar::Mut(_) => types::WithType::Mut,
                        })
                        .collect(),
                }),
            }),
            _ => None,
        }
    }
}
