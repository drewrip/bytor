use crate::ast::Expr;
use crate::semantic::SymbolStack;
use crate::types::Type;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("No definition for variable: {0}")]
    IdentNotFound(String),
}

pub fn infer(gamma: SymbolStack, expr: Expr) -> Result<Type, TypeError> {
    Ok(Type::Nil)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Node, Num, Term};
    use crate::semantic::SymbolTable;
    use crate::symbol::{new_symbol, new_var};
    use crate::types::Type;

    #[test]
    fn simple_passing() {
        let ctx: SymbolStack = vec![SymbolTable {
            table: vec![(new_symbol("x".into()), new_var(Type::Int32, Node::Null))]
                .into_iter()
                .collect(),
        }];
        let expr: Expr = Expr::Add(
            Box::new(Expr::Term(Box::new(Term::Id("x".into())))),
            Box::new(Expr::Term(Box::new(Term::Num(Num::Int32(5))))),
        );
        let res = infer(ctx, expr);
        assert!(res.is_ok());
    }
}
