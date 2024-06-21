use crate::ast::Expr;
use crate::semantic::SymbolStack;
use crate::types::Type;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("No definition for variable: {0}")]
    IdentNotFound(String),
    #[error("Couldn't unify types: {0}")]
    UnifyFailed(String),
}

#[derive(Debug, Clone)]
pub struct Subst(Type, Type);

fn has_sub(sub: Vec<Subst>, t: Type) -> bool {
    sub.iter().find(|s| s.0 == t).is_some()
}
fn get_sub(sub: Vec<Subst>, t: Type) -> Option<Type> {
    match sub.into_iter().find(|s| s.0 == t) {
        Some(found_sub) => Some(found_sub.1),
        None => None,
    }
}

// Find most general unifier between T1 and T2
// ---
// I'm starting with a simple implementation of unification
// If I find it to be prohibitively slow, I'll work on implementing
// something like "An Efficient Unification Algorithm" from
// Martelli and Montanari
// Currently using: Unifcation by recursive descent - Baader and Snyder
pub fn mgu(t1: Type, t2: Type) -> Result<Vec<Subst>, TypeError> {
    let mut sub: Vec<Subst> = vec![];
    unify(t1, t2, sub)
}

fn occurs_check(var: Type, t: Type, sub: Vec<Subst>) -> bool {
    if !matches!(var, Type::TypeVar(_)) {
        false
    } else if var == t {
        true
    } else if matches!(t, Type::TypeVar(_)) && has_sub(sub.clone(), t.clone()) {
        occurs_check(var, get_sub(sub.clone(), t.clone()).unwrap(), sub)
    } else if matches!(t, Type::Function(_)) {
        match t {
            Type::Function(f) => f
                .params_t
                .iter()
                .chain(vec![*f.return_t.clone()].iter())
                .any(|arg| occurs_check(var.clone(), arg.clone(), sub.clone())),
            _ => false,
        }
    } else {
        false
    }
}

fn unify(t1: Type, t2: Type, sub: Vec<Subst>) -> Result<Vec<Subst>, TypeError> {
    if t1 == t2 {
        Ok(sub)
    } else if matches!(t1, Type::TypeVar(_)) {
        unify_var(t1, t2, sub)
    } else if matches!(t2, Type::TypeVar(_)) {
        unify_var(t2, t1, sub)
    } else if matches!(t1, Type::Function(_)) && matches!(t2, Type::Function(_)) {
        match (t1, t2) {
            (Type::Function(f1), Type::Function(f2)) => {
                if f1.params_t.len() == f2.params_t.len() {
                    let mut sig_subs = sub.clone();
                    for (p1, p2) in f1.params_t.into_iter().zip(f2.params_t.into_iter()) {
                        sig_subs.extend(unify(p1, p2, sig_subs.clone())?);
                    }
                    sig_subs.extend(unify(
                        *f1.return_t.clone(),
                        *f2.return_t.clone(),
                        sig_subs.clone(),
                    )?);
                    Ok(sig_subs)
                } else {
                    Err(TypeError::UnifyFailed("Symbol clash".into()))
                }
            }
            (_, _) => Err(TypeError::UnifyFailed("Functions aren't functions".into())),
        }
    } else {
        Err(TypeError::UnifyFailed(
            "Couldn't unify types, not matching case".into(),
        ))
    }
}

fn unify_var(var: Type, t: Type, sub: Vec<Subst>) -> Result<Vec<Subst>, TypeError> {
    if !matches!(var, Type::TypeVar(_)) {
        return Err(TypeError::UnifyFailed(
            "unify_var, var isn't a type variable".to_string(),
        ));
    }

    if has_sub(sub.clone(), var.clone()) {
        unify(get_sub(sub.clone(), var).unwrap(), t, sub.clone())
    } else if matches!(t, Type::TypeVar(_)) && has_sub(sub.clone(), t.clone()) {
        unify(var, get_sub(sub.clone(), t).unwrap(), sub)
    } else if occurs_check(var.clone(), t.clone(), sub.clone()) {
        Err(TypeError::UnifyFailed("occurs_check failed".to_string()))
    } else {
        Ok([sub, vec![Subst(var, t)]].concat())
    }
}

// TODO
pub fn solve() {}

// Implement inference with:
// Generalizing Hindley-Milner Type Inference Algorithms
// from Heeren, Hage, and Swierstra
pub fn infer(gamma: SymbolStack, expr: Expr) -> Result<Type, TypeError> {
    Ok(Type::Nil)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Node, Num, Term};
    use crate::semantic::SymbolTable;
    use crate::symbol::{new_symbol, new_var};
    use crate::types::{FunctionType, Type};

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

    #[test]
    fn simple_unify() {
        let subs = mgu(Type::TypeVar(0), Type::Int32);
        println!("subs: {:?}", subs);
        assert!(subs.is_ok());
    }

    #[test]
    fn simple_func_unify() {
        let subs = mgu(
            Type::Function(FunctionType {
                params_t: vec![Type::TypeVar(0), Type::Int64],
                return_t: Box::new(Type::TypeVar(1)),
            }),
            Type::Function(FunctionType {
                params_t: vec![Type::Int32, Type::TypeVar(2)],
                return_t: Box::new(Type::String),
            }),
        );
        println!("subs: {:?}", subs);
        assert!(subs.is_ok());
    }

    #[test]
    fn simple_whole_function() {
        let subs = mgu(
            Type::Function(FunctionType {
                params_t: vec![Type::String, Type::Int64, Type::Float32],
                return_t: Box::new(Type::Int32),
            }),
            Type::TypeVar(0),
        );
        println!("subs: {:?}", subs);
        assert!(subs.is_ok());
    }
}
