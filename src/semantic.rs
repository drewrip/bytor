use std::collections::HashMap;

use crate::ast::{
    AssignOp, Block, Expr, Func, IfCases, LambdaFunc, Node, Program, Root, Stmt, Term, TypedExpr,
    TypedTerm,
};
use crate::ir::{self, IRNode};
use crate::symbol::{new_symbol, new_var, IdentMapping, Symbol, Symbolic, Var};
use crate::traverse::Traverse;
use crate::types::{self, Type};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BuildIRError {
    #[error("Couldn't build IR: {0}")]
    SomeError(String),
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub table: HashMap<Symbol, Var>,
}

pub fn new_empty_symbol_table() -> SymbolTable {
    SymbolTable {
        table: HashMap::new(),
    }
}

pub type SymbolStack = Vec<SymbolTable>;

pub fn slookup(stack: &SymbolStack, symbol: Symbol) -> Option<&Var> {
    stack
        .iter()
        .rev()
        .find_map(|table| table.table.get(&symbol))
}

// Assume this to be into the top of the stack
pub fn sinsert(stack: &mut SymbolStack, symbol: Symbol, var: Var) -> Option<Var> {
    stack
        .last_mut()
        .expect("No frames in symbol table!")
        .table
        .insert(symbol, var)
}

#[derive(Debug)]
pub struct ProgramState {
    pub stack: SymbolStack,
    pub ast: Box<Root>,
    pub build_stack: Vec<IRNode>,
    pub scope_counter: usize,
}

impl Traverse for ProgramState {
    type Error = BuildIRError;

    fn visit_root(&mut self, root: &mut Root) -> Result<(), Self::Error> {
        let Root {
            preblock,
            program,
            postblock,
        } = root;
        self.build_stack.push(IRNode::GlobalSection);
        self.visit_preblock(preblock)?;
        self.visit_postblock(postblock)?;
        self.build_stack.push(IRNode::EndGlobalSection);
        self.visit_program(program)?;
        Ok(())
    }

    fn visit_preblock(&mut self, preblock: &mut Block) -> Result<(), Self::Error> {
        self.visit_block(preblock)
    }

    fn visit_postblock(&mut self, postblock: &mut Block) -> Result<(), Self::Error> {
        self.visit_block(postblock)
    }

    fn visit_program(&mut self, program: &mut Program) -> Result<(), Self::Error> {
        self.visit_block(&mut program.1)
    }

    fn visit_block(&mut self, block: &mut Block) -> Result<(), Self::Error> {
        self.spush()?;
        for s in block {
            self.visit_stmt(s)?;
        }
        let _ = self.spop();
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) -> Result<(), Self::Error> {
        match stmt {
            Stmt::If(ifcases) => {
                self.visit_if_cases(ifcases)?;
            }
            Stmt::Call(symbol, args) => {
                self.visit_args(args)?;
                let resolved_ret_t = match slookup(&self.stack, symbol.clone()) {
                    Some(var) => &var.type_t,
                    None => {
                        return Err(BuildIRError::SomeError("Missing symbol".into()));
                    }
                };
                self.build_stack
                    .push(IRNode::Eval(ir::Func::Func(ir::Signature {
                        symbol: symbol.clone(),
                        params_t: args.iter().map(|p| p.type_t.clone()).collect(),
                        return_t: resolved_ret_t.clone(),
                    })));
            }
            Stmt::Assign(symbol, var, expr) => {
                self.visit_expr(expr)?;
                sinsert(
                    &mut self.stack,
                    symbol.clone(),
                    new_var(expr.type_t.clone(), Node::Null),
                );
                self.build_stack.push(IRNode::Assign(ir::Assign {
                    type_t: var.type_t.clone(),
                    symbol: symbol.clone(),
                }));
            }
            Stmt::Reassign(symbol, var, assign_op, expr) => {
                let mut new_expr = match assign_op {
                    AssignOp::Assign => expr.clone(),
                    AssignOp::AddAssign => Box::new(TypedExpr {
                        type_t: expr.type_t.clone(),
                        expr: Expr::Add(
                            Box::new(TypedExpr {
                                type_t: expr.type_t.clone(),
                                expr: Expr::Term(Box::new(TypedTerm {
                                    type_t: expr.type_t.clone(),
                                    term: Term::Id(symbol.ident.clone()),
                                })),
                            }),
                            expr.clone(),
                        ),
                    }),
                    AssignOp::SubAssign => Box::new(TypedExpr {
                        type_t: expr.type_t.clone(),
                        expr: Expr::Sub(
                            Box::new(TypedExpr {
                                type_t: expr.type_t.clone(),
                                expr: Expr::Term(Box::new(TypedTerm {
                                    type_t: expr.type_t.clone(),
                                    term: Term::Id(symbol.ident.clone()),
                                })),
                            }),
                            expr.clone(),
                        ),
                    }),
                    AssignOp::MultAssign => Box::new(TypedExpr {
                        type_t: expr.type_t.clone(),
                        expr: Expr::Mult(
                            Box::new(TypedExpr {
                                type_t: expr.type_t.clone(),
                                expr: Expr::Term(Box::new(TypedTerm {
                                    type_t: expr.type_t.clone(),
                                    term: Term::Id(symbol.ident.clone()),
                                })),
                            }),
                            expr.clone(),
                        ),
                    }),
                    AssignOp::DivAssign => Box::new(TypedExpr {
                        type_t: expr.type_t.clone(),
                        expr: Expr::Div(
                            Box::new(TypedExpr {
                                type_t: expr.type_t.clone(),
                                expr: Expr::Term(Box::new(TypedTerm {
                                    type_t: expr.type_t.clone(),
                                    term: Term::Id(symbol.ident.clone()),
                                })),
                            }),
                            expr.clone(),
                        ),
                    }),
                };
                self.visit_expr(&mut new_expr)?;
                sinsert(
                    &mut self.stack,
                    symbol.clone(),
                    new_var(new_expr.type_t, Node::Null),
                );
                self.build_stack.push(IRNode::Reassign(ir::Reassign {
                    type_t: var.type_t.clone(),
                    symbol: symbol.clone(),
                }));
            }
            Stmt::FuncDef(func) => {
                self.visit_func(func)?;
            }
            Stmt::Return(expr) => {
                self.visit_expr(expr)?;
                self.build_stack.push(IRNode::Return);
            }
        };
        Ok(())
    }

    fn visit_expr(&mut self, expr: &mut TypedExpr) -> Result<(), Self::Error> {
        match expr.expr.clone() {
            Expr::Add(_, _) => self.binary_op(expr.clone()),
            Expr::Sub(_, _) => self.binary_op(expr.clone()),
            Expr::Mult(_, _) => self.binary_op(expr.clone()),
            Expr::Div(_, _) => self.binary_op(expr.clone()),
            Expr::Eq(_, _) => self.binary_op(expr.clone()),
            Expr::Neq(_, _) => self.binary_op(expr.clone()),
            Expr::Leq(_, _) => self.binary_op(expr.clone()),
            Expr::Geq(_, _) => self.binary_op(expr.clone()),
            Expr::LessThan(_, _) => self.binary_op(expr.clone()),
            Expr::GreaterThan(_, _) => self.binary_op(expr.clone()),
            Expr::Not(_) => self.unary_op(expr.clone()),
            Expr::Neg(_) => self.unary_op(expr.clone()),
            Expr::Term(mut term) => self.visit_term(&mut term),
            Expr::Call(symbol, mut args) => {
                self.visit_args(&mut args)?;
                self.build_stack
                    .push(IRNode::Eval(ir::Func::Func(ir::Signature {
                        symbol,
                        params_t: args.iter().map(|p| p.type_t.clone()).collect(),
                        return_t: expr.type_t.clone(),
                    })));
                Ok(())
            }
            Expr::LambdaFunc(mut lf) => self.visit_lambda_func(&mut lf),
        }
    }

    fn visit_term(&mut self, term: &mut TypedTerm) -> Result<(), Self::Error> {
        match term.term.clone() {
            Term::Id(ident) => {
                self.build_stack.push(IRNode::Term(ir::Term {
                    type_t: term.type_t.clone(),
                    value: ir::Value::Id(ident),
                }));
            }
            Term::Expr(mut expr) => {
                self.visit_expr(&mut expr)?;
            }
            Term::Bool(b) => {
                self.build_stack.push(IRNode::Term(ir::Term {
                    type_t: term.type_t.clone(),
                    value: ir::Value::Bool(b),
                }));
            }
            Term::String(s) => {
                self.build_stack.push(IRNode::Term(ir::Term {
                    type_t: term.type_t.clone(),
                    value: ir::Value::String(s),
                }));
            }
            Term::Num(num) => {
                self.build_stack.push(IRNode::Term(ir::Term {
                    type_t: term.type_t.clone(),
                    value: num.try_into().unwrap(),
                }));
            }
        }
        Ok(())
    }

    fn visit_func(&mut self, func: &mut Func) -> Result<(), Self::Error> {
        let func_ir_num = self.get_new_scope();
        let func_ir_id = format!("_func_def_{}", func_ir_num);
        let return_type = func.return_t.clone();
        let param_types: Vec<Type> = func.params.iter().map(|p| p.type_t.clone()).collect();
        let func_symbol = new_symbol(func.ident.clone());
        sinsert(
            &mut self.stack,
            func_symbol.clone(),
            new_var(
                Type::Function(types::FunctionType {
                    params_t: param_types.clone(),
                    return_t: Box::new(return_type.clone()),
                }),
                Node::Null,
            ),
        );
        self.build_stack.push(IRNode::FuncDef(
            ir::FuncDef {
                symbol: func_symbol,
                return_t: return_type.clone(),
                params_t: func
                    .params
                    .iter()
                    .map(|p| (p.ident.clone(), p.type_t.clone()))
                    .collect(),
            },
            func_ir_id.clone(),
        ));
        self.visit_block(&mut func.block)?;
        self.build_stack.push(IRNode::EndFuncDef(func_ir_id));
        Ok(())
    }

    fn visit_lambda_func(&mut self, lf: &mut LambdaFunc) -> Result<(), Self::Error> {
        let func_ir_num = self.get_new_scope();
        let func_ir_id = format!("_func_def_{}", func_ir_num);
        let return_type = lf.return_t.clone();
        let param_types: Vec<Type> = lf.params.iter().map(|p| p.type_t.clone()).collect();
        let func_symbol = new_symbol(format!("_anon_func_{}", func_ir_num));
        sinsert(
            &mut self.stack,
            func_symbol.clone(),
            new_var(
                Type::Function(types::FunctionType {
                    params_t: param_types.clone(),
                    return_t: Box::new(return_type.clone()),
                }),
                Node::Null,
            ),
        );
        self.build_stack.push(IRNode::FuncDef(
            ir::FuncDef {
                symbol: func_symbol,
                return_t: return_type.clone(),
                params_t: lf
                    .params
                    .iter()
                    .map(|p| (p.ident.clone(), p.type_t.clone()))
                    .collect(),
            },
            func_ir_id.clone(),
        ));
        self.visit_block(&mut lf.block)?;
        self.build_stack.push(IRNode::EndFuncDef(func_ir_id));
        Ok(())
    }

    fn visit_if_cases(&mut self, if_cases: &mut IfCases) -> Result<(), Self::Error> {
        let if_ir_num = self.get_new_scope();
        let if_ir_id = format!("_if_stmt_{}", if_ir_num);
        self.build_stack.push(IRNode::If(if_ir_id.clone()));
        for (n, if_case) in if_cases.into_iter().enumerate() {
            self.visit_expr(&mut if_case.condition.clone())?;
            if n == 0 {
                self.build_stack.push(IRNode::IfCase(if_ir_id.clone()));
            } else if if_case.is_else {
                self.build_stack.push(IRNode::ElseCase(if_ir_id.clone()));
            } else {
                self.build_stack.push(IRNode::ElseIfCase(if_ir_id.clone()));
            }
            self.visit_block(&mut if_case.block)?;
        }
        self.build_stack.push(IRNode::EndIf(if_ir_id.clone()));
        Ok(())
    }
}

impl ProgramState {
    pub const IR_OUTPUT_FILENAME: &'static str = "out.ir";

    pub fn new(ast: Box<Root>) -> ProgramState {
        ProgramState {
            stack: vec![],
            build_stack: vec![],
            ast,
            scope_counter: 0,
        }
    }

    pub fn spush(&mut self) -> Result<(), BuildIRError> {
        self.stack.push(new_empty_symbol_table());
        Ok(())
    }

    pub fn spop(&mut self) -> Option<SymbolTable> {
        self.stack.pop()
    }

    pub fn get_new_scope(&mut self) -> usize {
        let new_scope = self.scope_counter;
        self.scope_counter += 1;
        new_scope
    }

    pub fn build_ir(&mut self) -> Result<(), BuildIRError> {
        self.spush()?;
        // Find the signature of `program` blocks
        //  -> if there are none, we can abort compilation :)
        self.program_signature_discovery()?;
        // Discover the functions and variables in the global scope
        //  -> but, don't parse function bodies
        self.global_ident_discovery()?;
        // check the program
        let mut ast = self.ast.clone();
        self.visit_root(&mut ast)?;
        Ok(())
    }

    fn program_signature_discovery(&mut self) -> Result<(), BuildIRError> {
        let gen_prog_symbol = self.ast.program.get_symbol();
        if let Some(prog_symbol) = gen_prog_symbol {
            self.stack
                .first_mut()
                .expect("No base node found!")
                .table
                .insert(prog_symbol.symbol, prog_symbol.var);
        }
        Ok(())
    }

    fn global_ident_discovery(&mut self) -> Result<(), BuildIRError> {
        // First pass: discover types and signatures of global identifiers
        let pre_idents: Vec<IdentMapping> = self
            .ast
            .preblock
            .iter()
            .filter_map(|stmt| stmt.get_symbol())
            .collect();

        let post_idents: Vec<IdentMapping> = self
            .ast
            .postblock
            .iter()
            .filter_map(|stmt| stmt.get_symbol())
            .collect();

        let base_node = self.stack.first_mut().expect("No base node!");

        for pre_ident in pre_idents {
            base_node.table.insert(pre_ident.symbol, pre_ident.var);
        }

        for post_ident in post_idents {
            base_node.table.insert(post_ident.symbol, post_ident.var);
        }
        Ok(())
    }

    fn binary_op(&mut self, operator: TypedExpr) -> Result<(), BuildIRError> {
        // Resolve the signature of the function that should be added in the IR
        let resolved_func = match operator.expr {
            Expr::Add(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Add(ir::new_sig(
                    "Add",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Sub(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Sub(ir::new_sig(
                    "Sub",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Mult(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Mult(ir::new_sig(
                    "Mult",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Div(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Div(ir::new_sig(
                    "Div",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Eq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Eq(ir::new_sig(
                    "Eq",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Neq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Neq(ir::new_sig(
                    "Neq",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Leq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Leq(ir::new_sig(
                    "Leq",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::Geq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Geq(ir::new_sig(
                    "Geq",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::LessThan(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Lt(ir::new_sig(
                    "Lt",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            Expr::GreaterThan(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
                ir::Func::Gt(ir::new_sig(
                    "Gt",
                    vec![lhs.type_t.clone(), rhs.type_t.clone()],
                    operator.type_t,
                ))
            }
            _ => panic!("Not sure how to represent {:?} in IR!", operator),
        };
        self.build_stack.push(IRNode::Eval(resolved_func));
        Ok(())
    }

    fn unary_op(&mut self, operator: TypedExpr) -> Result<(), BuildIRError> {
        // Resolve the signature of the function that should be added in the IR
        let resolved_func = match operator.expr {
            Expr::Not(mut u) => {
                self.visit_expr(&mut u)?;
                ir::Func::Add(ir::new_sig("Not", vec![u.type_t.clone()], operator.type_t))
            }
            Expr::Neg(mut u) => {
                self.visit_expr(&mut u)?;
                ir::Func::Add(ir::new_sig("Neg", vec![u.type_t.clone()], operator.type_t))
            }
            _ => panic!("Not sure how to represent {:?} in IR!", operator),
        };
        self.build_stack.push(IRNode::Eval(resolved_func));
        Ok(())
    }
}
