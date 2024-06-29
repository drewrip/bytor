use serde::{Deserialize, Serialize};

use crate::ast;
use crate::types;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentMapping {
    pub symbol: Symbol,
    pub var: Var,
}

pub fn new_identmapping(symbol: Symbol, var: Var) -> IdentMapping {
    IdentMapping { symbol, var }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Var {
    pub type_t: types::Type,
    pub node: ast::Node,
}

pub fn new_var(type_t: types::Type, node: ast::Node) -> Var {
    Var { type_t, node }
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Symbol {
    pub ident: String,
}

pub fn new_symbol(ident: String) -> Symbol {
    Symbol { ident }
}

pub trait Symbolic {
    fn get_symbol(&self) -> Option<IdentMapping>;
}

impl Symbolic for ast::Program {
    fn get_symbol(&self) -> Option<IdentMapping> {
        Some(IdentMapping {
            symbol: Symbol {
                ident: self.0.ident.clone(),
            },
            var: Var {
                type_t: types::Type::Program,
                node: ast::Node::BlockNode(Box::new(self.1.clone())),
            },
        })
    }
}

impl Symbolic for ast::Stmt {
    fn get_symbol(&self) -> Option<IdentMapping> {
        match self {
            ast::Stmt::Assign(symbol, var, expr) => Some(IdentMapping {
                symbol: symbol.clone(),
                var: (*var.clone()).clone(),
            }),
            ast::Stmt::FuncDef(func) => Some(IdentMapping {
                symbol: Symbol {
                    ident: func.ident.clone(),
                },
                var: Var {
                    type_t: types::Type::Function(types::FunctionType {
                        return_t: Box::new(func.return_t.clone()),
                        params_t: func
                            .params
                            .iter()
                            .map(|param| param.type_t.clone())
                            .collect(),
                    }),
                    node: ast::Node::BlockNode(Box::new(func.block.clone())),
                },
            }),
            _ => None,
        }
    }
}
