use crate::ast::{Block, Expr, Node, Param, Root, Stmt, Term, TypedExpr, TypedTerm};
use crate::semantic::{new_empty_symbol_table, sinsert, slookup, SymbolStack, SymbolTable};
use crate::symbol::{new_symbol, new_var};
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
    #[error("Couldn't perform substitution: {0}")]
    SubstitutionError(String),
    #[error("The types in the program could not be fully infered: {0}")]
    UndeterminedVariables(String),
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
        Err(TypeError::UnifyFailed(format!(
            "Couldn't unify types, not matching case, (t1={:?}, t2={:?})",
            t1, t2
        )))
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

// Implement inference with:
// Generalizing Hindley-Milner Type Inference Algorithms
// from Heeren, Hage, and Swierstra
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

pub struct TypingState {
    type_var_counter: u32,
}

pub struct InferState {
    pub constraints: Vec<Constraint>,
    symbols: SymbolStack,
    type_mapping: HashMap<Type, Type>,
    surrounding_func_types: Vec<Type>,
}

pub struct SubState {
    type_mapping: HashMap<Type, Type>,
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

    fn visit_root(&mut self, root: &mut Root) -> Result<(), Self::Error> {
        let Root {
            preblock,
            program,
            postblock,
        } = root;
        self.spush()?;
        self.visit_preblock(preblock)?;
        self.visit_postblock(postblock)?;
        self.surrounding_func_types
            .push(Type::Function(FunctionType {
                params_t: vec![],
                return_t: Box::new(Type::Int32),
            }));
        self.visit_program(program)?;
        self.surrounding_func_types.pop();
        self.spop();
        Ok(())
    }

    fn visit_preblock(&mut self, preblock: &mut Block) -> Result<(), Self::Error> {
        for stmt in preblock {
            self.visit_stmt(stmt)?;
        }
        Ok(())
    }

    fn visit_postblock(&mut self, postblock: &mut Block) -> Result<(), Self::Error> {
        for stmt in postblock {
            self.visit_stmt(stmt)?;
        }
        Ok(())
    }

    fn visit_block(&mut self, block: &mut Block) -> Result<(), Self::Error> {
        self.spush()?;
        for stmt in block {
            self.visit_stmt(stmt)?;
        }
        self.spop();
        Ok(())
    }

    fn visit_expr(&mut self, expr: &mut TypedExpr) -> Result<(), Self::Error> {
        match expr.expr {
            Expr::Add(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), expr.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(rhs.type_t.clone(), expr.type_t.clone()));
            }
            Expr::Sub(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), expr.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(rhs.type_t.clone(), expr.type_t.clone()));
            }
            Expr::Mult(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), expr.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(rhs.type_t.clone(), expr.type_t.clone()));
            }
            Expr::Div(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), expr.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(rhs.type_t.clone(), expr.type_t.clone()));
            }
            Expr::Eq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), rhs.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), Type::Bool));
            }
            Expr::Neq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), rhs.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), Type::Bool));
            }
            Expr::Leq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), rhs.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), Type::Bool));
            }
            Expr::Geq(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), rhs.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), Type::Bool));
            }
            Expr::LessThan(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), rhs.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), Type::Bool));
            }
            Expr::GreaterThan(ref mut lhs, ref mut rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)?;
                self.constraints
                    .push(Constraint::Eq(lhs.type_t.clone(), rhs.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), Type::Bool));
            }
            Expr::Not(ref mut u) => {
                self.visit_expr(u)?;
                self.constraints
                    .push(Constraint::Eq(u.type_t.clone(), expr.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(Type::Bool, expr.type_t.clone()));
            }
            Expr::Neg(ref mut u) => {
                self.visit_expr(u)?;
                self.constraints
                    .push(Constraint::Eq(u.type_t.clone(), expr.type_t.clone()));
                self.constraints
                    .push(Constraint::Eq(Type::Bool, expr.type_t.clone()));
            }
            Expr::Term(ref mut t) => {
                self.visit_term(t)?;
                self.constraints
                    .push(Constraint::Eq(t.type_t.clone(), expr.type_t.clone()));
            }
            Expr::Call(ref symbol, ref mut args) => {
                let target_func = slookup(&self.symbols, symbol.clone()).ok_or(
                    TypeError::IdentNotFound(format!("Function {:?} not found", symbol.clone())),
                )?;
                self.constraints.push(Constraint::Eq(
                    target_func.type_t.clone(),
                    Type::Function(FunctionType {
                        params_t: args.iter().map(|p| p.type_t.clone()).collect(),
                        return_t: Box::new(expr.type_t.clone()),
                    }),
                ));
            }
            Expr::LambdaFunc(ref mut lf) => {
                self.spush()?;
                let func_type = Type::Function(FunctionType {
                    params_t: lf.params.iter().map(|p| p.type_t.clone()).collect(),
                    return_t: Box::new(lf.return_t.clone()),
                });
                self.surrounding_func_types.push(func_type.clone());
                for param in lf.params.clone() {
                    let Param { type_t, ident } = *param;
                    sinsert(
                        &mut self.symbols,
                        new_symbol(ident),
                        new_var(type_t, Node::Null),
                    );
                }
                self.visit_block(&mut lf.block)?;
                self.constraints
                    .push(Constraint::Eq(func_type.clone(), expr.type_t.clone()));
                self.surrounding_func_types.pop();
                self.spop();
            }
        }
        Ok(())
    }

    fn visit_term(&mut self, term: &mut TypedTerm) -> Result<(), Self::Error> {
        match term.term {
            Term::Id(ref ident) => {
                let symbol = new_symbol(ident.clone());
                let found_term = slookup(&self.symbols, symbol.clone()).ok_or(
                    TypeError::IdentNotFound(format!("Ident {:?} not found", symbol.clone())),
                )?;
                self.constraints.push(Constraint::Eq(
                    term.type_t.clone(),
                    found_term.type_t.clone(),
                ));
            }
            Term::Num(_) => {}
            Term::Bool(_) => {}
            Term::String(_) => {}
            Term::Expr(ref mut expr) => {
                self.visit_expr(expr)?;
                self.constraints
                    .push(Constraint::Eq(expr.type_t.clone(), term.type_t.clone()));
            }
        };
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) -> Result<(), Self::Error> {
        match stmt {
            Stmt::If(if_cases) => {
                for if_case in if_cases {
                    self.visit_expr(&mut if_case.condition)?;
                    self.constraints
                        .push(Constraint::Eq(if_case.condition.type_t.clone(), Type::Bool));
                    self.visit_block(&mut if_case.block)?;
                }
            }
            Stmt::Assign(symbol, var, expr) => {
                self.visit_expr(expr)?;
                sinsert(&mut self.symbols, symbol.clone(), *var.clone());
                self.constraints
                    .push(Constraint::Eq(var.type_t.clone(), expr.type_t.clone()));
            }
            Stmt::Reassign(symbol, var, _, expr) => {
                self.visit_expr(expr)?;
                sinsert(&mut self.symbols, symbol.clone(), *var.clone());
                self.constraints
                    .push(Constraint::Eq(var.type_t.clone(), expr.type_t.clone()));
            }
            Stmt::Call(_, _) => {
                // Inferring types based of these call statements needs to be treated
                // differently since there is no return type to deduce
                /*
                self.constraints.push(Constraint::Eq(
                    target_func.type_t.clone(),
                    Type::Function(FunctionType {
                        params_t: args.iter().map(|p| p.type_t.clone()).collect(),
                        return_t: Box::new(expr.type_t.clone()),
                    }),
                ));
                */
            }
            Stmt::FuncDef(func) => {
                sinsert(
                    &mut self.symbols,
                    new_symbol(func.ident.clone()),
                    new_var(
                        Type::Function(FunctionType {
                            params_t: func.params.iter().map(|p| p.type_t.clone()).collect(),
                            return_t: Box::new(func.return_t.clone()),
                        }),
                        Node::Null,
                    ),
                );
                self.spush()?;
                self.surrounding_func_types
                    .push(Type::Function(FunctionType {
                        params_t: func.params.iter().map(|p| p.type_t.clone()).collect(),
                        return_t: Box::new(func.return_t.clone()),
                    }));
                for param in func.params.clone() {
                    let Param { type_t, ident } = *param;
                    sinsert(
                        &mut self.symbols,
                        new_symbol(ident),
                        new_var(type_t, Node::Null),
                    );
                }
                self.visit_block(&mut func.block)?;
                self.surrounding_func_types.pop();
                self.spop();
            }
            Stmt::Return(expr) => {
                self.visit_expr(expr)?;
                let surround_type = self.surrounding_func_types.iter().last().unwrap();
                let surround_func_type = match surround_type {
                    Type::Function(func) => Ok(func),
                    _ => Err(TypeError::IdentNotFound(
                        "Function type on func stack is not function".into(),
                    )),
                }?;
                self.constraints.push(Constraint::Eq(
                    expr.type_t.clone(),
                    *surround_func_type.return_t.clone(),
                ));
            }
        };
        Ok(())
    }
}

impl Traverse for SubState {
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
                            type_t: self.resolve_type(p.type_t.clone()),
                        })
                    })
                    .collect();
                lf.return_t = self.resolve_type(lf.return_t.clone());
                self.visit_block(&mut lf.block)?;
            }
        }
        expr.type_t = self.resolve_type(expr.type_t.clone());
        Ok(())
    }

    fn visit_term(&mut self, term: &mut TypedTerm) -> Result<(), Self::Error> {
        match term.term {
            Term::Id(_) => {
                term.type_t = self.resolve_type(term.type_t.clone());
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
                term.type_t = self.resolve_type(term.type_t.clone());
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
                var.type_t = self.resolve_type(var.type_t.clone());
            }
            Stmt::Reassign(_, var, _, expr) => {
                self.visit_expr(expr)?;
                var.type_t = self.resolve_type(var.type_t.clone());
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
                            type_t: self.resolve_type(p.type_t),
                        })
                    })
                    .collect();
                func.return_t = self.resolve_type(func.return_t.clone());
                self.visit_block(&mut func.block)?;
            }
            Stmt::Return(expr) => {
                self.visit_expr(expr)?;
            }
        };
        Ok(())
    }
}

impl TypingState {
    pub fn new() -> Self {
        TypingState {
            type_var_counter: 0,
        }
    }

    pub fn get_new_type_var(&mut self) -> Type {
        let new_tv = Type::TypeVar(self.type_var_counter);
        self.type_var_counter += 1;
        new_tv
    }

    pub fn augment(&mut self, root: &mut Root) -> Result<(), TypeError> {
        self.visit_root(root)
    }
}

impl InferState {
    pub fn new() -> Self {
        InferState {
            constraints: vec![],
            symbols: vec![],
            type_mapping: HashMap::new(),
            surrounding_func_types: vec![],
        }
    }

    pub fn get_type_mapping(&self) -> HashMap<Type, Type> {
        self.type_mapping.clone()
    }

    pub fn spush(&mut self) -> Result<(), TypeError> {
        self.symbols.push(new_empty_symbol_table());
        Ok(())
    }

    pub fn spop(&mut self) -> Option<SymbolTable> {
        self.symbols.pop()
    }

    pub fn constrain(&mut self, root: &mut Root) -> Result<(), TypeError> {
        self.visit_root(root)
    }

    fn type_is_determined(t: Type) -> bool {
        match t.clone() {
            Type::TypeVar(_) => false,
            Type::Function(func) => !func
                .params_t
                .iter()
                .chain(vec![*func.return_t.clone()].iter())
                .any(|t| !Self::type_is_determined(t.clone())),
            _ => true,
        }
    }

    pub fn is_fully_determined(&mut self) -> Result<(), TypeError> {
        for (t1, t2) in self.type_mapping.clone().iter() {
            match (t1.clone(), t2.clone()) {
                (Type::TypeVar(tv1), Type::TypeVar(_)) => {
                    Err(TypeError::UndeterminedVariables(format!(
                        "TypeVar({}) left undetermined: {:?} -> {:?}",
                        tv1,
                        t1.clone(),
                        t2.clone()
                    )))
                }
                (Type::TypeVar(_), mapped_t) => {
                    if Self::type_is_determined(mapped_t) {
                        Ok(())
                    } else {
                        Err(TypeError::UndeterminedVariables(format!(
                            "Type mapped onto is not fully determined: {:?} -> {:?}",
                            t1.clone(),
                            t2.clone()
                        )))
                    }
                }
                (other_t1, other_t2) => Err(TypeError::UndeterminedVariables(format!(
                    "Mapping between non-variables: {:?} -> {:?}",
                    other_t1, other_t2,
                ))),
            }?;
        }
        Ok(())
    }

    pub fn resolve(&mut self) -> Result<(), TypeError> {
        // Solve the constraint system for the TypeVars
        let subs = solve(self.constraints.clone())?;
        for sub in subs {
            match (sub.0, sub.1) {
                (Type::TypeVar(t), other) => {
                    self.type_mapping.insert(Type::TypeVar(t), other);
                }
                (other, Type::TypeVar(t)) => {
                    self.type_mapping.insert(Type::TypeVar(t), other);
                }
                (_, _) => {
                    return Err(TypeError::SubstitutionError("Nothing to substitute".into()));
                }
            }
        }
        Ok(())
    }
}

impl SubState {
    pub fn new(subs: HashMap<Type, Type>) -> Self {
        SubState { type_mapping: subs }
    }

    pub fn resolve_type(&self, t: Type) -> Type {
        self.type_mapping.get(&t).unwrap_or(&t).clone()
    }

    pub fn substitute(&mut self, root: &mut Root) -> Result<(), TypeError> {
        self.visit_root(root)
    }
}

#[cfg(test)]
mod tests {
    use crate::infer::*;
    use crate::rascal::RootParser;
    use crate::types::{FunctionType, Type};

    #[test]
    fn test_mgu_simple() {
        let subs = mgu(Type::TypeVar(0), Type::Int32);
        println!("subs: {:?}", subs);
        assert!(subs.is_ok());
    }

    #[test]
    fn test_mgu_lambda() {
        let subs = mgu(
            Type::Function(FunctionType {
                params_t: vec![Type::TypeVar(0), Type::Int32],
                return_t: Box::new(Type::TypeVar(2)),
            }),
            Type::Function(FunctionType {
                params_t: vec![Type::Int32, Type::TypeVar(1)],
                return_t: Box::new(Type::Int32),
            }),
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

    #[test]
    fn infer_lambda_function() {
        let source = r#"
            program test_funcs begin
                    let g = fun (x: int32) -> (x + x);
                    return g(2);
            end
        "#;
        let mut root = RootParser::new().parse(&source).unwrap();

        let mut typing_state = TypingState::new();
        let tv_typing_result = typing_state.augment(&mut root);
        assert!(tv_typing_result.is_ok());
        let mut infer_state = InferState::new();
        let constraint_gen_result = infer_state.constrain(&mut root);
        assert!(constraint_gen_result.is_ok());
        let resolve_result = infer_state.resolve();
        assert!(resolve_result.is_ok());
        let fully_determined = infer_state.is_fully_determined();
        assert!(fully_determined.is_ok());
        let mut sub_state = SubState::new(infer_state.get_type_mapping());
        let sub_gen_result = sub_state.substitute(&mut root);
        assert!(sub_gen_result.is_ok());
    }

    #[test]
    fn test_infer_nested_lambda() {
        let source = r#"
            program nested begin
                    let f = fun (x: int32) begin
                            let g = fun (y: int32) -> (y * 2);
                            return g(x);
                    end;
                    return f(3);
            end
        "#;

        let mut root = RootParser::new().parse(&source).unwrap();

        let mut typing_state = TypingState::new();
        let tv_typing_result = typing_state.augment(&mut root);
        assert!(tv_typing_result.is_ok());
        let mut infer_state = InferState::new();
        let constraint_gen_result = infer_state.constrain(&mut root);
        assert!(constraint_gen_result.is_ok());
        let resolve_result = infer_state.resolve();
        assert!(resolve_result.is_ok());
        let fully_determined = infer_state.is_fully_determined();
        assert!(fully_determined.is_ok());
        let mut sub_state = SubState::new(infer_state.get_type_mapping());
        let sub_gen_result = sub_state.substitute(&mut root);
        assert!(sub_gen_result.is_ok());
    }

    #[test]
    fn test_infer_complex_function() {
        let source = r#"
            fun foo(x: int32) -> (fun (int32) -> int32) begin
                    return fun (n: int32) -> (n + x + 1);
            end
            program nested begin
                    let f = foo(2);
                    return f(4);
            end
        "#;

        let mut root = RootParser::new().parse(&source).unwrap();

        let mut typing_state = TypingState::new();
        let tv_typing_result = typing_state.augment(&mut root);
        assert!(tv_typing_result.is_ok());
        let mut infer_state = InferState::new();
        let constraint_gen_result = infer_state.constrain(&mut root);
        assert!(constraint_gen_result.is_ok());
        let resolve_result = infer_state.resolve();
        assert!(resolve_result.is_ok());
        let fully_determined = infer_state.is_fully_determined();
        assert!(fully_determined.is_ok());
        let mut sub_state = SubState::new(infer_state.get_type_mapping());
        let sub_gen_result = sub_state.substitute(&mut root);
        assert!(sub_gen_result.is_ok());
    }
}
