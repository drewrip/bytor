use crate::ast::{Expr, Func, Param, Program, Root, Stmt, Term, TypedExpr, TypedTerm};
use crate::semantic::SymbolStack;
use crate::traverse::Traverse;
use crate::types::{FunctionType, Type};

use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("No definition for variable: {0}")]
    IdentNotFound(String),
    #[error("Couldn't unify types: {0}")]
    UnifyFailed(String),
    #[error("Numeric type couldn't be derived: {0}")]
    DeriveNumType(String),
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Eq(Type, Type),
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
    let sub: Vec<Subst> = vec![];
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

fn subst(sub: Vec<Subst>, t: Type) -> Type {
    match t.clone() {
        Type::Function(func) => {
            let mut new_params = vec![];
            for p in func.params_t {
                new_params.push(subst(sub.clone(), p));
            }
            Type::Function(FunctionType {
                params_t: new_params,
                return_t: Box::new(subst(sub, *func.return_t.clone())),
            })
        }
        Type::TypeVar(_) => match get_sub(sub.clone(), t.clone()) {
            Some(repl_type) => repl_type,
            None => t,
        },
        _ => t,
    }
}

fn subst_into_constr(sub: Vec<Subst>, constr: Constraint) -> Constraint {
    match constr {
        Constraint::Eq(t1, t2) => Constraint::Eq(subst(sub.clone(), t1), subst(sub.clone(), t2)),
    }
}

fn solve_helper(
    constraints: Vec<Constraint>,
    sub: &mut Vec<Subst>,
) -> Result<Vec<Subst>, TypeError> {
    match constraints.first() {
        Some(cst) => match cst.clone() {
            Constraint::Eq(t1, t2) => {
                let new_subs = mgu(t1, t2)?;
                sub.extend(new_subs);
                solve_helper(
                    constraints
                        .iter()
                        .skip(1)
                        .map(|c| subst_into_constr(sub.clone(), c.clone()))
                        .collect(),
                    sub,
                )
            }
        },
        None => Ok(vec![]),
    }
}

pub fn solve(constraints: Vec<Constraint>) -> Result<Vec<Subst>, TypeError> {
    let mut substitutions = vec![];
    solve_helper(constraints, &mut substitutions)?;
    Ok(substitutions)
}

// Implement inference with:
// Generalizing Hindley-Milner Type Inference Algorithms
// from Heeren, Hage, and Swierstra
pub fn infer(gamma: SymbolStack, expr: Expr) -> Result<Type, TypeError> {
    Ok(Type::Nil)
}

pub struct TypingState {
    type_var_counter: u32,
    type_mapping: HashMap<Type, Type>,
}

pub struct InferState {
    contraints: Vec<Constraint>,
    symbols: SymbolStack,
}

impl Traverse for TypingState {
    type Error = TypeError;

    fn visit_expr(&mut self, expr: &mut TypedExpr) -> Result<(), Self::Error> {
        match expr.expr {
            Expr::Add(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Sub(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Mult(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Div(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Eq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Neq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Leq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Geq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::LessThan(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::GreaterThan(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
            }
            Expr::Not(ref mut u) => {
                self.visit_expr(u)?;
            }
            Expr::Neg(ref mut u) => {
                self.visit_expr(u)?;
            }
            Expr::Term(ref mut t) => {
                self.visit_term(t)?;
            }
            Expr::Call(_, ref mut args) => {
                for mut arg in args {
                    self.visit_expr(&mut arg)?;
                }
            }
            Expr::LambdaFunc(ref mut lf) => {
                lf.params = lf
                    .params
                    .iter()
                    .map(|p| {
                        Box::new(Param {
                            ident: p.ident.clone(),
                            type_t: match p.type_t.clone() {
                                Type::Unknown => self.get_new_type_var(),
                                _ => p.type_t.clone(),
                            },
                        })
                    })
                    .collect();
                lf.return_t = match lf.return_t {
                    Type::Unknown => self.get_new_type_var(),
                    _ => lf.return_t.clone(),
                };
                self.visit_block(&mut lf.block)?;
            }
        }
        expr.type_t = match expr.type_t {
            Type::Unknown => self.get_new_type_var(),
            _ => expr.type_t.clone(),
        };
        Ok(())
    }

    fn visit_term(&mut self, term: &mut TypedTerm) -> Result<(), Self::Error> {
        match term.term {
            Term::Id(_) => {
                term.type_t = match term.type_t {
                    Type::Unknown => self.get_new_type_var(),
                    _ => term.type_t.clone(),
                };
            }
            Term::Num(ref num) => {
                term.type_t = match Type::try_from(num.clone()) {
                    Ok(t) => Ok(t),
                    Err(_) => Err(TypeError::DeriveNumType("Couldn't derive Num type".into())),
                }?;
            }
            Term::Bool(_) => {
                term.type_t = Type::Bool;
            }
            Term::String(_) => {
                term.type_t = Type::String;
            }
            Term::Expr(ref mut expr) => {
                self.visit_expr(expr)?;
                term.type_t = match term.type_t {
                    Type::Unknown => self.get_new_type_var(),
                    _ => term.type_t.clone(),
                };
            }
        };
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) -> Result<(), Self::Error> {
        match stmt {
            Stmt::If(if_cases) => {
                for if_case in if_cases {
                    self.visit_expr(&mut if_case.condition)?;
                    self.visit_block(&mut if_case.block)?;
                }
            }
            Stmt::Assign(_, var, expr) => {
                self.visit_expr(expr)?;
                var.type_t = match var.type_t.clone() {
                    Type::Unknown => self.get_new_type_var(),
                    other => other.clone(),
                };
            }
            Stmt::Reassign(_, var, _, expr) => {
                self.visit_expr(expr)?;
                var.type_t = match var.type_t.clone() {
                    Type::Unknown => self.get_new_type_var(),
                    other => other.clone(),
                };
            }
            Stmt::Call(_, args) => {
                for arg in args {
                    self.visit_expr(arg)?;
                }
            }
            Stmt::FuncDef(func) => {
                func.params = func
                    .params
                    .clone()
                    .into_iter()
                    .map(|p| {
                        Box::new(Param {
                            ident: p.ident,
                            type_t: match p.type_t {
                                Type::Unknown => self.get_new_type_var(),
                                _ => p.type_t,
                            },
                        })
                    })
                    .collect();
                func.return_t = match func.return_t {
                    Type::Unknown => self.get_new_type_var(),
                    _ => func.return_t.clone(),
                };
                self.visit_block(&mut func.block)?;
            }
            Stmt::Return(expr) => {
                self.visit_expr(expr)?;
            }
        };
        Ok(())
    }
}

impl Traverse for InferState {
    type Error = TypeError;
}

impl TypingState {
    pub fn new() -> Self {
        TypingState {
            type_var_counter: 0,
            type_mapping: HashMap::new(),
        }
    }

    pub fn get_new_type_var(&mut self) -> Type {
        let new_tv = Type::TypeVar(self.type_var_counter);
        self.type_mapping.insert(new_tv.clone(), Type::Unknown);
        self.type_var_counter += 1;
        new_tv
    }
}

impl InferState {
    pub fn new() -> Self {
        InferState {
            contraints: vec![],
            symbols: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Node, Num, Term};
    use crate::semantic::SymbolTable;
    use crate::symbol::{new_symbol, new_var};
    use crate::types::{FunctionType, Type};

    #[test]
    fn infer_simple_passing() {
        let ctx: SymbolStack = vec![SymbolTable {
            table: vec![(new_symbol("x".into()), new_var(Type::Int32, Node::Null))]
                .into_iter()
                .collect(),
        }];
        let expr: Expr = Expr::Add(
            Box::new(TypedExpr {
                type_t: Type::Unknown,
                expr: Expr::Term(Box::new(TypedTerm {
                    type_t: Type::Unknown,
                    term: Term::Id("x".into()),
                })),
            }),
            Box::new(TypedExpr {
                type_t: Type::Unknown,
                expr: Expr::Term(Box::new(TypedTerm {
                    type_t: Type::Unknown,
                    term: Term::Num(Num::Int32(5)),
                })),
            }),
        );
        let res = infer(ctx, expr);
        assert!(res.is_ok());
    }

    #[test]
    fn unify_simple_unify() {
        let subs = mgu(Type::TypeVar(0), Type::Int32);
        println!("subs: {:?}", subs);
        assert!(subs.is_ok());
    }

    #[test]
    fn unify_simple_func_unify() {
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
    fn unify_simple_whole_function() {
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

    #[test]
    fn solve_simple_whole_function() {
        let t1 = Type::Function(FunctionType {
            params_t: vec![Type::String, Type::Int64, Type::Float32],
            return_t: Box::new(Type::Int32),
        });
        let t2 = Type::TypeVar(0);
        let subs = solve(vec![Constraint::Eq(t1, t2)]);
        println!("solved: {:?}", subs);
        assert!(subs.is_ok());
    }
}
