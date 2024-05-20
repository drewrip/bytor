use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast;
use crate::ir::{new_ir_node, IRNode};
use crate::symbol::{new_symbol, new_var, IdentMapping, Symbol, Symbolic, Var};
use crate::types::{self, Type};

type Result<T> = std::result::Result<T, SemanticError>;

#[derive(Debug, Clone)]
pub enum SemanticError {
    SomeError,
    StackCorruption,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantic Error: {:?}", self)
    }
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

#[derive(Debug)]
pub struct ProgramState {
    pub stack: SymbolStack,
    pub ast: Arc<ast::Root>,
    pub build_stack: Vec<IRNode>,
}

pub fn new_state(ast: Arc<ast::Root>) -> ProgramState {
    ProgramState {
        stack: vec![],
        build_stack: vec![],
        ast,
    }
}

impl ProgramState {
    pub fn slookup(&self, symbol: Symbol) -> Option<&Var> {
        self.stack
            .iter()
            .rev()
            .find_map(|table| table.table.get(&symbol))
    }

    // Assume this to be into the top of the stack
    pub fn sinsert(&mut self, symbol: Symbol, var: Var) -> Option<Var> {
        self.stack
            .last_mut()
            .expect("No frames in symbol table!")
            .table
            .insert(symbol, var)
    }

    pub fn spush(&mut self) -> Result<()> {
        self.stack.push(new_empty_symbol_table());
        Ok(())
    }

    pub fn spop(&mut self) -> Option<SymbolTable> {
        self.stack.pop()
    }

    pub fn build(&mut self) -> Result<()> {
        self.spush();
        // Find the signature of `program` blocks
        //  -> if there are none, we can abort compilation :)
        self.program_signature_discovery()?;
        // Discover the functions and variables in the global scope
        //  -> but, don't parse function bodies
        self.global_ident_discovery()?;
        // Now we can 'recursively' check the bodies of the
        // globally defined variables, functions and 'programs'.
        self.check_global_definitions()?;
        // check the program
        self.check_program()?;
        self.spop();
        Ok(())
    }
    fn program_signature_discovery(&mut self) -> Result<()> {
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

    fn global_ident_discovery(&mut self) -> Result<()> {
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

    fn check_global_definitions(&mut self) -> Result<()> {
        let base_node = self.stack.first_mut().expect("No base node!");
        // All of the global statements
        let stmts: Vec<Arc<ast::Stmt>> = self
            .ast
            .preblock
            .iter()
            .chain(self.ast.postblock.iter())
            .map(|stmt| (stmt.clone()).clone())
            .collect();

        let mut stack: Vec<IRNode> = vec![];

        // These checks all depend on the state of the symbol
        // tables, so I have them in a for loop
        for stmt in stmts {
            let node: ast::Node = ast::Node::StmtNode(stmt.clone());
            let mut ir_node: IRNode = new_ir_node(node.clone(), types::Type::Unknown, 0, false);
            let mut idx: usize = 0;
            stack.push(ir_node.clone());
            while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
                let (new_idx, new_ir_node) = stack
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, x)| !x.get_checked())
                    .unwrap();

                idx = new_idx;
                ir_node = new_ir_node.clone();

                match ir_node.node.clone() {
                    ast::Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                    ast::Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                    ast::Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                    ast::Node::BlockNode(block) => {
                        self.check_block(&mut stack, idx, (*block).clone())
                    }
                    ast::Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                    _ => {
                        println!("node -> {:?}", ir_node.node.clone());
                        panic!("AST node not yet implemented!")
                    }
                };
            }
        }

        Ok(())
    }

    fn check_program(&mut self) -> Result<()> {
        let mut stack: Vec<IRNode> = vec![];
        let program = self.ast.program.clone();
        stack.push(new_ir_node(
            ast::Node::ProgramNode(program.clone()),
            Type::Program(types::ProgramType { with_t: vec![] }),
            0,
            false,
        ));
        let block = match (*program).clone() {
            ast::Program::NoWith(_, block) => block,
            ast::Program::With(_, _, block) => block,
        };
        self.npop(&mut stack);
        let node: ast::Node = ast::Node::BlockNode(block.into());
        let mut ir_node: IRNode = new_ir_node(node.clone(), types::Type::Unknown, 0, false);
        let mut idx: usize = 0;
        self.spush();
        stack.push(ir_node.clone());
        while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
            let (new_idx, new_ir_node) = stack
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| !x.get_checked())
                .unwrap();

            idx = new_idx;
            ir_node = new_ir_node.clone();

            match ir_node.node.clone() {
                ast::Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                ast::Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                ast::Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                ast::Node::BlockNode(block) => self.check_block(&mut stack, idx, (*block).clone()),
                ast::Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                _ => {
                    println!("node -> {:?}", ir_node.node.clone());
                    panic!("AST node not yet implemented!")
                }
            };
        }
        Ok(())
    }

    fn check_stmt(
        &mut self,
        stack: &mut Vec<IRNode>,
        node_idx: usize,
        stmt: ast::Stmt,
    ) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match stmt {
            ast::Stmt::Assign(symbol, var, expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_ir_node(
                            ast::Node::SymbolNode(symbol),
                            var.type_t.clone(),
                            0,
                            true,
                        ));
                        stack.push(new_ir_node(
                            ast::Node::VarNode(var.clone()),
                            var.type_t.clone(),
                            0,
                            true,
                        ));
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(expr),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let oper1 = stack.get(node_idx + 3).unwrap().get_type();
                        // Don't increase the parent, because there isn't one right now
                        let var_node = stack.get(node_idx + 2).unwrap();
                        let mut var = match var_node.node.clone() {
                            ast::Node::VarNode(var) => (*var).clone(),
                            _ => panic!("no var!"),
                        };
                        if oper1 != var_node.type_t && var_node.type_t != types::Type::Unknown {
                            println!(
                                "declared -> {:?}, expression -> {:?}",
                                var_node.type_t, oper1
                            );
                            panic!("type of expression doesn't match declared type");
                        }
                        var.type_t = oper1.clone();
                        let symbol = match stack.get(node_idx + 1).unwrap().node.clone() {
                            ast::Node::SymbolNode(symbol) => symbol,
                            _ => panic!("no symbol!"),
                        };
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1.clone());
                        self.sinsert(symbol, var);
                        self.rebase_stack(stack, node_idx);
                        self.npop(stack);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::Assign",
                        other
                    ),
                }
            }
            ast::Stmt::Reassign(symbol, var, assign_op, expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_ir_node(
                            ast::Node::SymbolNode(symbol.clone()),
                            types::Type::Unknown,
                            0,
                            true,
                        ));
                        stack.push(new_ir_node(
                            ast::Node::VarNode(var.clone()),
                            types::Type::Unknown,
                            0,
                            true,
                        ));
                        stack.push(new_ir_node(
                            ast::Node::AssignOpNode(assign_op.clone()),
                            types::Type::Unknown,
                            0,
                            true,
                        ));
                        let assign_op_expr = match assign_op {
                            ast::AssignOp::Assign => expr,
                            ast::AssignOp::AddAssign => Arc::new(ast::Expr::Add(
                                Arc::new(ast::Expr::Term(Arc::new(ast::Term::Id(
                                    symbol.ident.clone(),
                                )))),
                                expr,
                            )),
                            ast::AssignOp::SubAssign => Arc::new(ast::Expr::Sub(
                                Arc::new(ast::Expr::Term(Arc::new(ast::Term::Id(
                                    symbol.ident.clone(),
                                )))),
                                expr,
                            )),
                            ast::AssignOp::MultAssign => Arc::new(ast::Expr::Mult(
                                Arc::new(ast::Expr::Term(Arc::new(ast::Term::Id(
                                    symbol.ident.clone(),
                                )))),
                                expr,
                            )),
                            ast::AssignOp::DivAssign => Arc::new(ast::Expr::Div(
                                Arc::new(ast::Expr::Term(Arc::new(ast::Term::Id(
                                    symbol.ident.clone(),
                                )))),
                                expr,
                            )),
                        };
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(assign_op_expr.clone()),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                        stack.get_mut(node_idx).unwrap().node = ast::Node::StmtNode(Arc::new(
                            ast::Stmt::Assign(symbol, var, assign_op_expr),
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let oper1 = stack.get(node_idx + 4).unwrap().get_type();
                        // Later we must also check that the assign_op is defined for the types
                        let symbol = match stack.get(node_idx + 1).unwrap().node.clone() {
                            ast::Node::SymbolNode(symbol) => symbol,
                            other => {
                                panic!("stack node isn't symbol: {:?}\nstack: {:?}", other, stack)
                            }
                        };
                        let lookup_type = match self.slookup(symbol) {
                            Some(var) => var.type_t.clone(),
                            None => panic!("ident not found!"),
                        };
                        if oper1 != lookup_type {
                            println!("stack: {:?}", stack);
                            panic!("type error: {:?} == {:?}", oper1, lookup_type);
                        }
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1);
                        self.rebase_stack(stack, node_idx);
                        self.npop(stack);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::Reassign",
                        other
                    ),
                }
            }
            ast::Stmt::If(if_cases) => match progress {
                0 => {
                    stack.get_mut(node_idx).unwrap().set_total(1);
                    for if_case in if_cases.iter() {
                        stack.push(new_ir_node(
                            ast::Node::BlockNode(if_case.block.clone().into()),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(if_case.condition.clone()),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                }
                1 => {
                    stack.get_mut(node_idx).unwrap().set_checked();
                    self.rebase_stack(stack, node_idx);
                    self.npop(stack);
                }
                other => panic!(
                    "error: progress={} doesn't match any possible for Stmt::If",
                    other
                ),
            },
            ast::Stmt::Call(symbol, args) => match progress {
                0 => {
                    stack.get_mut(node_idx).unwrap().set_total(1);
                    let func_var = self.slookup(symbol.clone()).unwrap();
                    let func_type = match func_var.type_t.clone() {
                        types::Type::Function(func_type) => func_type,
                        _ => panic!("not a function!"),
                    };
                    let param_types = func_type.params_t;
                    if param_types.len() != args.len() {
                        panic!("calling function without correct number of parameters");
                    }
                    for arg in (*args).clone() {
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(arg),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                }
                1 => {
                    let func_var = self.slookup(symbol.clone()).unwrap();
                    let func_type = match func_var.type_t.clone() {
                        types::Type::Function(func_type) => func_type,
                        _ => panic!("not a function!"),
                    };
                    let param_types = func_type.params_t;
                    let mut resolved_types: Vec<Type> = vec![];
                    while stack.len() - 1 != node_idx {
                        resolved_types.push(self.npop(stack).unwrap().get_type());
                    }
                    for (arg_type, param_type) in resolved_types.iter().rev().zip(param_types) {
                        println!(
                            "{}: {:?} == {:?}",
                            symbol.ident.clone(),
                            arg_type,
                            param_type
                        );
                        if param_type != *arg_type {
                            panic!(
                                "argument to function {}(...) is incorrect type",
                                symbol.ident
                            )
                        }
                    }
                    let return_t = func_type.return_t.first().unwrap().clone();
                    stack.get_mut(node_idx).unwrap().set_type(return_t);
                    stack.get_mut(node_idx).unwrap().set_checked();
                    self.rebase_stack(stack, node_idx);
                    self.npop(stack);
                }
                other => panic!(
                    "error: progress={} doesn't match any possible for Stmt::Call",
                    other
                ),
            },
            ast::Stmt::FuncDef(func) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        let params = func.params.iter().map(|p| p.type_t.clone()).collect();
                        let return_t = func.ret_t.clone();
                        let symbol = new_symbol(func.ident.clone());
                        let block = func.block.clone();
                        let function_type = types::Type::Function(types::FunctionType {
                            params_t: params,
                            return_t: vec![return_t],
                            with_t: vec![],
                        });
                        self.sinsert(
                            new_symbol(func.ident.clone()),
                            new_var(
                                function_type.clone(),
                                stack.get(node_idx).unwrap().node.clone(),
                            ),
                        );
                        stack.push(new_ir_node(
                            ast::Node::FuncNode(func),
                            function_type,
                            0,
                            false,
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let func_type = stack.get(node_idx + 1).unwrap().get_type();
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(func_type);
                        self.rebase_stack(stack, node_idx);
                        self.npop(stack);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::FuncDef",
                        other
                    ),
                }
            }
            ast::Stmt::Return(expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(expr),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                    1 => {
                        // Nothing to check just yet
                        stack.get_mut(node_idx).unwrap().set_checked();
                        self.rebase_stack(stack, node_idx);
                        self.npop(stack);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::Return",
                        other
                    ),
                }
            }
        }

        Ok(())
    }

    fn check_expr(
        &mut self,
        stack: &mut Vec<IRNode>,
        node_idx: usize,
        expr: ast::Expr,
    ) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match expr {
            ast::Expr::Add(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "add".into(), lhs, rhs);
            }
            ast::Expr::Sub(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "sub".into(), lhs, rhs);
            }
            ast::Expr::Mult(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "mult".into(), lhs, rhs);
            }
            ast::Expr::Div(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "div".into(), lhs, rhs);
            }
            ast::Expr::Term(term) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_ir_node(
                            ast::Node::TermNode(term),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                    1 => {
                        // Term is complete
                        let oper1 = stack.get(node_idx + 1).unwrap().get_type();
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1);
                        // Increment progress of parent
                        self.inc_parent(stack);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Expr::Term",
                        other
                    ),
                }
            }
            ast::Expr::Eq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "eq".into(), lhs, rhs);
            }
            ast::Expr::Neq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "neq".into(), lhs, rhs);
            }
            ast::Expr::Leq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "leq".into(), lhs, rhs);
            }
            ast::Expr::Geq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "geq".into(), lhs, rhs);
            }
            ast::Expr::LessThan(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "leq".into(), lhs, rhs);
            }
            ast::Expr::GreaterThan(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "geq".into(), lhs, rhs);
            }
            ast::Expr::Call(symbol, args) => match progress {
                0 => {
                    let func_var = self.slookup(symbol.clone()).unwrap();
                    let func_type = match func_var.type_t.clone() {
                        types::Type::Function(func_type) => func_type,
                        _ => panic!("not a function!"),
                    };
                    let param_types = func_type.params_t;
                    if param_types.len() != args.len() {
                        panic!("calling function without correct number of parameters");
                    }
                    for arg in (*args).clone() {
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(arg),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                }
                1 => {
                    let func_var = self.slookup(symbol.clone()).unwrap();
                    let func_type = match func_var.type_t.clone() {
                        types::Type::Function(func_type) => func_type,
                        _ => panic!("not a function!"),
                    };
                    let param_types = func_type.params_t;
                    let mut resolved_types: Vec<Type> = vec![];
                    while stack.len() - 1 != node_idx {
                        resolved_types.push(self.npop(stack).unwrap().get_type());
                    }
                    for (arg_type, param_type) in resolved_types.iter().rev().zip(param_types) {
                        println!(
                            "{}: {:?} == {:?}",
                            symbol.ident.clone(),
                            arg_type,
                            param_type
                        );
                        if param_type != *arg_type {
                            panic!(
                                "argument to function {}(...) is incorrect type",
                                symbol.ident
                            )
                        }
                    }
                    let return_t = func_type.return_t.first().unwrap().clone();
                    stack.get_mut(node_idx).unwrap().set_type(return_t);
                    stack.get_mut(node_idx).unwrap().set_checked();
                    self.inc_parent(stack);
                }
                other => panic!(
                    "error: progress={} doesn't match any possible for Expr::Call",
                    other
                ),
            },
        }
        Ok(())
    }

    fn check_term(
        &mut self,
        stack: &mut Vec<IRNode>,
        node_idx: usize,
        term: ast::Term,
    ) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match term {
            ast::Term::Id(ident) => {
                stack.get_mut(node_idx).unwrap().set_total(0);
                // Lookup type of identifier
                let lookup_type = match self.slookup(new_symbol(ident.clone())) {
                    Some(var) => var.type_t.clone(),
                    None => panic!("ident not found! -> {:?}", ident),
                };

                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().set_type(lookup_type);
                // Increment progress of parent
                self.inc_parent(stack);
            }
            ast::Term::Num(num) => {
                stack.get_mut(node_idx).unwrap().set_total(0);
                stack.get_mut(node_idx).unwrap().set_checked();
                stack
                    .get_mut(node_idx)
                    .unwrap()
                    .set_type(types::Type::Int32);
                // Increment progress of parent
                self.inc_parent(stack);
            }
            ast::Term::Bool(bool_value) => {
                stack.get_mut(node_idx).unwrap().set_total(0);
                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().set_type(types::Type::Bool);
                // Increment progress of parent
                self.inc_parent(stack);
            }
            ast::Term::Expr(expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_ir_node(
                            ast::Node::ExprNode(expr),
                            types::Type::Unknown,
                            0,
                            false,
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let oper1 = stack.get(node_idx + 1).unwrap().get_type();
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1);
                        // Increment progress of parent
                        self.inc_parent(stack);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Term::Expr",
                        other
                    ),
                }
            }
        }
        Ok(())
    }

    fn check_block(
        &mut self,
        stack: &mut Vec<IRNode>,
        node_idx: usize,
        block: ast::Block,
    ) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match progress {
            0 => {
                self.rebase_stack(stack, node_idx);
                self.spush();
                stack.get_mut(node_idx).unwrap().set_total(1);
                for stmt in block.iter().rev() {
                    stack.push(new_ir_node(
                        ast::Node::StmtNode(stmt.clone()),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                }
                stack.get_mut(node_idx).unwrap().inc_prog();
            }
            1 => {
                self.rebase_stack(stack, node_idx);
                stack.get_mut(node_idx).unwrap().set_checked();
                let found_symbols = self.spop().unwrap().clone();
                attach_symbols_parent(&mut self.build_stack, found_symbols);
                self.npop(stack);
                // Increment progress of parent
                self.inc_parent(stack);
            }
            other => panic!(
                "error: progress={} doesn't match any possible for block",
                other
            ),
        }
        Ok(())
    }

    fn check_func(
        &mut self,
        stack: &mut Vec<IRNode>,
        node_idx: usize,
        func: ast::Func,
    ) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match progress {
            0 => {
                stack.get_mut(node_idx).unwrap().set_total(1);
                let block = func.block;
                self.spush();
                let params = func.params;
                for param in params {
                    let param_type = param.type_t.clone();
                    let symbol = new_symbol(param.ident.clone());
                    let var = new_var(param_type, ast::Node::SymbolNode(symbol.clone()));
                    self.sinsert(symbol, var);
                }
                stack.push(new_ir_node(
                    ast::Node::BlockNode(block.into()),
                    types::Type::Unknown,
                    0,
                    false,
                ));
            }
            1 => {
                // Both sub expressions checked!
                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().add_symbols(self.spop());
                // Increment progress of parent
                self.inc_parent(stack);
            }
            other => panic!(
                "error: progress={} doesn't match any possible for func",
                other
            ),
        }
        Ok(())
    }

    fn binary_op(
        &self,
        stack: &mut Vec<IRNode>,
        node_idx: usize,
        progress: usize,
        operator: String,
        lhs: Arc<ast::Expr>,
        rhs: Arc<ast::Expr>,
    ) -> Result<()> {
        match progress {
            0 => {
                stack.get_mut(node_idx).unwrap().set_total(2);
                stack.push(new_ir_node(
                    ast::Node::ExprNode(rhs),
                    types::Type::Unknown,
                    0,
                    false,
                ));
            }
            1 => {
                stack.push(new_ir_node(
                    ast::Node::ExprNode(lhs),
                    types::Type::Unknown,
                    0,
                    false,
                ));
            }
            2 => {
                // Both sub expressions checked!
                let oper1 = stack.get(node_idx + 1).unwrap().get_type();
                let oper2 = stack.get(node_idx + 2).unwrap().get_type();
                if oper1 != oper2 {
                    println!("{:?}", stack);
                    println!("lhs: {:?}", stack.get(node_idx + 1).unwrap());
                    println!("rhs: {:?}", stack.get(node_idx + 2).unwrap());
                    panic!("type error: {}({:?}, {:?})", operator, oper1, oper2);
                }
                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().set_type(oper1);
                // Increment progress of parent
                self.inc_parent(stack);
            }
            other => panic!(
                "error: progress={} doesn't match any possible for {}",
                other, operator
            ),
        }
        Ok(())
    }

    fn npop(&mut self, stack: &mut Vec<IRNode>) -> Result<IRNode> {
        let checked_node = stack.pop();
        match checked_node {
            Some(node) => {
                let build_copy = node.clone();
                self.build_stack.push(build_copy);
                Ok(node)
            }
            None => Err(SemanticError::StackCorruption),
        }
    }

    fn rebase_stack(&mut self, stack: &mut Vec<IRNode>, node_idx: usize) -> Result<()> {
        if stack.len() - 1 == node_idx {
            Ok(())
        } else {
            while stack.len() - 1 != node_idx {
                self.npop(stack)?;
            }
            Ok(())
        }
    }

    fn inc_parent(&self, stack: &mut Vec<IRNode>) {
        match stack.iter_mut().rev().find(|x| !x.get_checked()) {
            Some(parent) => parent.inc_prog(),
            None => (),
        };
    }
}

fn attach_symbols_parent(stack: &mut Vec<IRNode>, symbols: SymbolTable) {
    match stack.iter_mut().rev().find(|x| !x.get_checked()) {
        Some(parent) => parent.add_symbols(Some(symbols)),
        None => panic!("no parent node to attach symbols to"),
    };
}
