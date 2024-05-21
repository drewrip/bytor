use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast::{AssignOp, Block, Expr, Func, Node, Program, Root, Stmt, Term};
use crate::ir::IRNode;
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

pub fn new_sem_node(
    ast_node: Node,
    type_t: types::Type,
    total: usize,
    checked: bool,
    parent: Option<usize>,
) -> SemNode {
    SemNode {
        progress: 0,
        total,
        checked,
        ast_node,
        type_t,
        symbols: None,
        parent,
    }
}

#[derive(Debug, Clone)]
pub struct SemNode {
    pub progress: usize,
    pub total: usize,
    pub checked: bool,
    pub ast_node: Node,
    pub type_t: types::Type,
    pub symbols: Option<SymbolTable>,
    pub parent: Option<usize>, // Optionally the node_idx of the parent AST node
                               // NOTE: we also now need a way to reference the location of a function or jump target
}

impl SemNode {
    pub fn get_prog(&self) -> usize {
        self.progress
    }

    pub fn set_prog(&mut self, progress: usize) {
        self.progress = progress;
    }

    pub fn inc_prog(&mut self) {
        self.progress += 1;
    }

    pub fn get_total(&self) -> usize {
        self.total
    }

    pub fn set_total(&mut self, total: usize) {
        self.total = total;
    }

    pub fn get_type(&self) -> types::Type {
        self.type_t.clone()
    }

    pub fn set_type(&mut self, type_t: types::Type) {
        self.type_t = type_t;
    }

    pub fn set_checked(&mut self) {
        self.checked = true;
    }

    pub fn get_checked(&self) -> bool {
        self.checked
    }

    pub fn add_symbols(&mut self, symbol_table: Option<SymbolTable>) {
        self.symbols = symbol_table;
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
    pub ast: Arc<Root>,
    pub build_stack: Vec<IRNode>,
}

pub fn new_state(ast: Arc<Root>) -> ProgramState {
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
        let stmts: Vec<Arc<Stmt>> = self
            .ast
            .preblock
            .iter()
            .chain(self.ast.postblock.iter())
            .map(|stmt| (stmt.clone()).clone())
            .collect();

        let mut stack: Vec<SemNode> = vec![];

        // These checks all depend on the state of the symbol
        // tables, so I have them in a for loop
        for stmt in stmts {
            let node: Node = Node::StmtNode(stmt.clone());
            let mut sem_node: SemNode =
                new_sem_node(node.clone(), types::Type::Unknown, 0, false, None);
            let mut idx: usize = 0;
            stack.push(sem_node.clone());
            while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
                let (new_idx, new_sem_node) = stack
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, x)| !x.get_checked())
                    .unwrap();

                idx = new_idx;
                sem_node = new_sem_node.clone();

                match sem_node.ast_node.clone() {
                    Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                    Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                    Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                    Node::BlockNode(block) => self.check_block(&mut stack, idx, (*block).clone()),
                    Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                    _ => {
                        println!("node -> {:?}", sem_node.ast_node.clone());
                        panic!("AST node not yet implemented!")
                    }
                };
            }
        }

        Ok(())
    }

    fn check_program(&mut self) -> Result<()> {
        let mut stack: Vec<SemNode> = vec![];
        let program = self.ast.program.clone();
        stack.push(new_sem_node(
            Node::ProgramNode(program.clone()),
            Type::Program(types::ProgramType { with_t: vec![] }),
            0,
            false,
            None,
        ));
        let block = match (*program).clone() {
            Program::NoWith(_, block) => block,
            Program::With(_, _, block) => block,
        };
        stack.pop();
        let node: Node = Node::BlockNode(block.into());
        let mut sem_node: SemNode = new_sem_node(
            node.clone(),
            types::Type::Unknown,
            0,
            false,
            Some(stack.len() - 1),
        );
        let mut idx: usize = 0;
        self.spush();
        stack.push(sem_node.clone());
        while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
            let (new_idx, new_sem_node) = stack
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| !x.get_checked())
                .unwrap();

            idx = new_idx;
            sem_node = new_sem_node.clone();

            match sem_node.ast_node.clone() {
                Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                Node::BlockNode(block) => self.check_block(&mut stack, idx, (*block).clone()),
                Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                _ => {
                    println!("node -> {:?}", sem_node.ast_node.clone());
                    panic!("AST node not yet implemented!")
                }
            };
        }
        Ok(())
    }

    fn check_stmt(&mut self, stack: &mut Vec<SemNode>, node_idx: usize, stmt: Stmt) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match stmt {
            Stmt::Assign(symbol, var, expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_sem_node(
                            Node::SymbolNode(symbol),
                            var.type_t.clone(),
                            0,
                            true,
                            Some(node_idx),
                        ));
                        stack.push(new_sem_node(
                            Node::VarNode(var.clone()),
                            var.type_t.clone(),
                            0,
                            true,
                            Some(node_idx),
                        ));
                        stack.push(new_sem_node(
                            Node::ExprNode(expr),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let oper1 = stack.get(node_idx + 3).unwrap().get_type();
                        // Don't increase the parent, because there isn't one right now
                        let var_node = stack.get(node_idx + 2).unwrap();
                        let mut var = match var_node.ast_node.clone() {
                            Node::VarNode(var) => (*var).clone(),
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
                        let symbol = match stack.get(node_idx + 1).unwrap().ast_node.clone() {
                            Node::SymbolNode(symbol) => symbol,
                            _ => panic!("no symbol!"),
                        };
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1.clone());
                        self.sinsert(symbol, var);
                        self.rebase_stack(stack, node_idx);
                        stack.pop();
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::Assign",
                        other
                    ),
                }
            }
            Stmt::Reassign(symbol, var, assign_op, expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_sem_node(
                            Node::SymbolNode(symbol.clone()),
                            types::Type::Unknown,
                            0,
                            true,
                            Some(node_idx),
                        ));
                        stack.push(new_sem_node(
                            Node::VarNode(var.clone()),
                            types::Type::Unknown,
                            0,
                            true,
                            Some(node_idx),
                        ));
                        stack.push(new_sem_node(
                            Node::AssignOpNode(assign_op.clone()),
                            types::Type::Unknown,
                            0,
                            true,
                            Some(node_idx),
                        ));
                        let assign_op_expr = match assign_op {
                            AssignOp::Assign => expr,
                            AssignOp::AddAssign => Arc::new(Expr::Add(
                                Arc::new(Expr::Term(Arc::new(Term::Id(symbol.ident.clone())))),
                                expr,
                            )),
                            AssignOp::SubAssign => Arc::new(Expr::Sub(
                                Arc::new(Expr::Term(Arc::new(Term::Id(symbol.ident.clone())))),
                                expr,
                            )),
                            AssignOp::MultAssign => Arc::new(Expr::Mult(
                                Arc::new(Expr::Term(Arc::new(Term::Id(symbol.ident.clone())))),
                                expr,
                            )),
                            AssignOp::DivAssign => Arc::new(Expr::Div(
                                Arc::new(Expr::Term(Arc::new(Term::Id(symbol.ident.clone())))),
                                expr,
                            )),
                        };
                        stack.push(new_sem_node(
                            Node::ExprNode(assign_op_expr.clone()),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                        stack.get_mut(node_idx).unwrap().ast_node =
                            Node::StmtNode(Arc::new(Stmt::Assign(symbol, var, assign_op_expr)));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let oper1 = stack.get(node_idx + 4).unwrap().get_type();
                        // Later we must also check that the assign_op is defined for the types
                        let symbol = match stack.get(node_idx + 1).unwrap().ast_node.clone() {
                            Node::SymbolNode(symbol) => symbol,
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
                        stack.pop();
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::Reassign",
                        other
                    ),
                }
            }
            Stmt::If(if_cases) => match progress {
                0 => {
                    stack.get_mut(node_idx).unwrap().set_total(1);
                    for if_case in if_cases.iter() {
                        stack.push(new_sem_node(
                            Node::BlockNode(if_case.block.clone().into()),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                        stack.push(new_sem_node(
                            Node::ExprNode(if_case.condition.clone()),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                    }
                }
                1 => {
                    stack.get_mut(node_idx).unwrap().set_checked();
                    self.rebase_stack(stack, node_idx);
                    stack.pop();
                }
                other => panic!(
                    "error: progress={} doesn't match any possible for Stmt::If",
                    other
                ),
            },
            Stmt::Call(symbol, args) => match progress {
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
                        stack.push(new_sem_node(
                            Node::ExprNode(arg),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
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
                        resolved_types.push(stack.pop().unwrap().get_type());
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
                    stack.pop();
                }
                other => panic!(
                    "error: progress={} doesn't match any possible for Stmt::Call",
                    other
                ),
            },
            Stmt::FuncDef(func) => {
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
                                stack.get(node_idx).unwrap().ast_node.clone(),
                            ),
                        );
                        stack.push(new_sem_node(
                            Node::FuncNode(func),
                            function_type,
                            0,
                            false,
                            Some(node_idx),
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let func_type = stack.get(node_idx + 1).unwrap().get_type();
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(func_type);
                        self.rebase_stack(stack, node_idx);
                        stack.pop();
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Stmt::FuncDef",
                        other
                    ),
                }
            }
            Stmt::Return(expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_sem_node(
                            Node::ExprNode(expr),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                    }
                    1 => {
                        // Nothing to check just yet
                        stack.get_mut(node_idx).unwrap().set_checked();
                        self.rebase_stack(stack, node_idx);
                        stack.pop();
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

    fn check_expr(&mut self, stack: &mut Vec<SemNode>, node_idx: usize, expr: Expr) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match expr {
            Expr::Add(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "add".into(), lhs, rhs);
            }
            Expr::Sub(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "sub".into(), lhs, rhs);
            }
            Expr::Mult(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "mult".into(), lhs, rhs);
            }
            Expr::Div(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "div".into(), lhs, rhs);
            }
            Expr::Term(term) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_sem_node(
                            Node::TermNode(term),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                    }
                    1 => {
                        // Term is complete
                        let oper1 = stack.get(node_idx + 1).unwrap().get_type();
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1);
                        // Increment progress of parent
                        self.inc_parent(stack, node_idx);
                    }
                    other => panic!(
                        "error: progress={} doesn't match any possible for Expr::Term",
                        other
                    ),
                }
            }
            Expr::Eq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "eq".into(), lhs, rhs);
            }
            Expr::Neq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "neq".into(), lhs, rhs);
            }
            Expr::Leq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "leq".into(), lhs, rhs);
            }
            Expr::Geq(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "geq".into(), lhs, rhs);
            }
            Expr::LessThan(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "leq".into(), lhs, rhs);
            }
            Expr::GreaterThan(lhs, rhs) => {
                self.binary_op(stack, node_idx, progress, "geq".into(), lhs, rhs);
            }
            Expr::Call(symbol, args) => match progress {
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
                        stack.push(new_sem_node(
                            Node::ExprNode(arg),
                            types::Type::Unknown,
                            0,
                            false,
                            None,
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
                        resolved_types.push(stack.pop().unwrap().get_type());
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
                    self.inc_parent(stack, node_idx);
                }
                other => panic!(
                    "error: progress={} doesn't match any possible for Expr::Call",
                    other
                ),
            },
        }
        Ok(())
    }

    fn check_term(&mut self, stack: &mut Vec<SemNode>, node_idx: usize, term: Term) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match term {
            Term::Id(ident) => {
                stack.get_mut(node_idx).unwrap().set_total(0);
                // Lookup type of identifier
                let lookup_type = match self.slookup(new_symbol(ident.clone())) {
                    Some(var) => var.type_t.clone(),
                    None => panic!("ident not found! -> {:?}", ident),
                };

                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().set_type(lookup_type);
                // Increment progress of parent
                self.inc_parent(stack, node_idx);
            }
            Term::Num(num) => {
                stack.get_mut(node_idx).unwrap().set_total(0);
                stack.get_mut(node_idx).unwrap().set_checked();
                stack
                    .get_mut(node_idx)
                    .unwrap()
                    .set_type(types::Type::Int32);
                // Increment progress of parent
                self.inc_parent(stack, node_idx);
            }
            Term::Bool(bool_value) => {
                stack.get_mut(node_idx).unwrap().set_total(0);
                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().set_type(types::Type::Bool);
                // Increment progress of parent
                self.inc_parent(stack, node_idx);
            }
            Term::Expr(expr) => {
                match progress {
                    0 => {
                        stack.get_mut(node_idx).unwrap().set_total(1);
                        stack.push(new_sem_node(
                            Node::ExprNode(expr),
                            types::Type::Unknown,
                            0,
                            false,
                            Some(node_idx),
                        ));
                    }
                    1 => {
                        // Both sub expressions checked!
                        let oper1 = stack.get(node_idx + 1).unwrap().get_type();
                        stack.get_mut(node_idx).unwrap().set_checked();
                        stack.get_mut(node_idx).unwrap().set_type(oper1);
                        // Increment progress of parent
                        self.inc_parent(stack, node_idx);
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
        stack: &mut Vec<SemNode>,
        node_idx: usize,
        block: Block,
    ) -> Result<()> {
        let progress = stack.get_mut(node_idx).unwrap().get_prog();
        match progress {
            0 => {
                self.rebase_stack(stack, node_idx);
                self.spush();
                stack.get_mut(node_idx).unwrap().set_total(1);
                for stmt in block.iter().rev() {
                    stack.push(new_sem_node(
                        Node::StmtNode(stmt.clone()),
                        types::Type::Unknown,
                        0,
                        false,
                        None, // TODO: might want to make the parent the block node
                    ));
                }
                stack.get_mut(node_idx).unwrap().inc_prog();
            }
            1 => {
                self.rebase_stack(stack, node_idx);
                stack.get_mut(node_idx).unwrap().set_checked();
                let found_symbols = self.spop().unwrap().clone();
                attach_symbols_parent(stack, node_idx, found_symbols);
                stack.pop();
                // Increment progress of parent
                self.inc_parent(stack, node_idx);
            }
            other => panic!(
                "error: progress={} doesn't match any possible for block",
                other
            ),
        }
        Ok(())
    }

    fn check_func(&mut self, stack: &mut Vec<SemNode>, node_idx: usize, func: Func) -> Result<()> {
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
                    let var = new_var(param_type, Node::SymbolNode(symbol.clone()));
                    self.sinsert(symbol, var);
                }
                stack.push(new_sem_node(
                    Node::BlockNode(block.into()),
                    types::Type::Unknown,
                    0,
                    false,
                    Some(node_idx),
                ));
            }
            1 => {
                // Both sub expressions checked!
                stack.get_mut(node_idx).unwrap().set_checked();
                stack.get_mut(node_idx).unwrap().add_symbols(self.spop());
                // Increment progress of parent
                self.inc_parent(stack, node_idx);
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
        stack: &mut Vec<SemNode>,
        node_idx: usize,
        progress: usize,
        operator: String,
        lhs: Arc<Expr>,
        rhs: Arc<Expr>,
    ) -> Result<()> {
        match progress {
            0 => {
                stack.get_mut(node_idx).unwrap().set_total(2);
                stack.push(new_sem_node(
                    Node::ExprNode(rhs),
                    types::Type::Unknown,
                    0,
                    false,
                    Some(node_idx),
                ));
            }
            1 => {
                stack.push(new_sem_node(
                    Node::ExprNode(lhs),
                    types::Type::Unknown,
                    0,
                    false,
                    Some(node_idx),
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
                self.inc_parent(stack, node_idx);
            }
            other => panic!(
                "error: progress={} doesn't match any possible for {}",
                other, operator
            ),
        }
        Ok(())
    }

    fn rebase_stack(&mut self, stack: &mut Vec<SemNode>, node_idx: usize) -> Result<()> {
        if stack.len() - 1 == node_idx {
            Ok(())
        } else {
            while stack.len() - 1 != node_idx {
                stack.pop();
            }
            Ok(())
        }
    }

    fn inc_parent(&self, stack: &mut Vec<SemNode>, node_idx: usize) {
        match stack.get(node_idx).unwrap().parent {
            Some(parent) => stack.get_mut(parent).unwrap().inc_prog(),
            None => (),
        };
    }
}

fn attach_symbols_parent(stack: &mut Vec<SemNode>, node_idx: usize, symbols: SymbolTable) {
    match stack.get(node_idx).unwrap().parent {
        Some(parent) => stack.get_mut(parent).unwrap().add_symbols(Some(symbols)),
        None => eprintln!("no parent node to attach symbols to"),
    };
}
