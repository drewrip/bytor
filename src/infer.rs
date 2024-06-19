use crate::semantic::SymbolStack;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("No definition for variable: {0}")]
    IdentNotFound(String),
}

pub fn infer(gamma: SymbolStack) -> Result<(), TypeError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Node;
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
        let res = infer(ctx);
        assert!(res.is_ok());
    }
}
