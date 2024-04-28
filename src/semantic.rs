use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast;
use crate::symbol::{new_symbol, new_var, IdentMapping, Symbol, Symbolic, Var};
use crate::types;

type Result<T> = std::result::Result<T, SemanticError>;

#[derive(Debug, Clone)]
pub enum SemanticError {
    SomeError,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantic Error")
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
}

pub fn new_state(ast: Arc<ast::Root>) -> ProgramState {
    ProgramState { stack: vec![], ast }
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

    pub fn spop(&mut self) -> Result<()> {
        self.stack.pop();
        Ok(())
    }

    pub fn build(&mut self) -> Result<()> {
        self.stack.push(new_empty_symbol_table());
        // Find the signature of `program` blocks
        //  -> if there are none, we can abort compilation :)
        self.program_signature_discovery()?;
        // Discover the functions and variables in the global scope
        //  -> but, don't parse function bodies
        self.global_ident_discovery()?;
        // Now we can 'recursively' check the bodies of the
        // globally defined variables, functions and 'programs'.
        self.check_global_definitions()?;
        // TODO: check the program
        self.check_program()?;
        Ok(())
    }
    fn program_signature_discovery(&mut self) -> Result<()> {
        let gen_prog_symbol = self.ast.program.get_symbol();
        if let Some(prog_symbol) = gen_prog_symbol {
            self.stack
                .first_mut()
                .expect("No base frame found!")
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

        let base_frame = self.stack.first_mut().expect("No base frame!");

        for pre_ident in pre_idents {
            base_frame.table.insert(pre_ident.symbol, pre_ident.var);
        }

        for post_ident in post_idents {
            base_frame.table.insert(post_ident.symbol, post_ident.var);
        }
        Ok(())
    }

    fn check_global_definitions(&mut self) -> Result<()> {
        let base_frame = self.stack.first_mut().expect("No base frame!");
        // All of the global statements
        let stmts: Vec<Arc<ast::Stmt>> = self
            .ast
            .preblock
            .iter()
            .chain(self.ast.postblock.iter())
            .map(|stmt| (stmt.clone()).clone())
            .collect();

        let mut stack: Vec<ast::Frame> = vec![];

        // These checks all depend on the state of the symbol
        // tables, so I have them in a for loop
        for stmt in stmts {
            let node: ast::Node = ast::Node::StmtNode(stmt.clone());
            let mut frame: ast::Frame =
                ast::new_frame(node.clone(), types::Type::Unknown, 0, false);
            let mut idx: usize = 0;
            stack.push(frame.clone());
            while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
                let (new_idx, new_frame) = stack
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, x)| !x.get_checked())
                    .unwrap();

                idx = new_idx;
                frame = new_frame.clone();

                match frame.node.clone() {
                    ast::Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                    ast::Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                    ast::Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                    ast::Node::BlockNode(block) => {
                        self.check_block(&mut stack, idx, (*block).clone())
                    }
                    ast::Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                    _ => {
                        println!("node -> {:?}", frame.node.clone());
                        panic!("AST node not yet implemented!")
                    }
                };
            }
        }

        Ok(())
    }

    fn check_program(&mut self) -> Result<()> {
        let program = self.ast.program.clone();
        let block = match (*program).clone() {
            ast::Program::NoWith(_, block) => block,
            ast::Program::With(_, _, block) => block,
        };
        let node: ast::Node = ast::Node::BlockNode(block.into());
        let mut stack: Vec<ast::Frame> = vec![];
        let mut frame: ast::Frame = ast::new_frame(node.clone(), types::Type::Unknown, 0, false);
        let mut idx: usize = 0;
        self.spush();
        stack.push(frame.clone());
        while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
            let (new_idx, new_frame) = stack
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| !x.get_checked())
                .unwrap();

            idx = new_idx;
            frame = new_frame.clone();

            match frame.node.clone() {
                ast::Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                ast::Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                ast::Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                ast::Node::BlockNode(block) => self.check_block(&mut stack, idx, (*block).clone()),
                ast::Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                _ => {
                    println!("node -> {:?}", frame.node.clone());
                    panic!("AST node not yet implemented!")
                }
            };
        }

        Ok(())
    }

    fn resolve_expr(&mut self, target_expr: ast::Expr) -> Result<types::Type> {
        let node: ast::Node = ast::Node::ExprNode(target_expr.into());
        let mut stack: Vec<ast::Frame> = vec![];
        let mut frame: ast::Frame = ast::new_frame(node.clone(), types::Type::Unknown, 0, false);
        let mut idx: usize = 0;
        stack.push(frame.clone());
        while !stack.is_empty() && stack.iter().any(|f| !f.get_checked()) {
            let (new_idx, new_frame) = stack
                .iter()
                .enumerate()
                .rev()
                .find(|(_, x)| !x.get_checked())
                .unwrap();

            idx = new_idx;
            frame = new_frame.clone();

            match frame.node.clone() {
                ast::Node::StmtNode(stmt) => self.check_stmt(&mut stack, idx, (*stmt).clone()),
                ast::Node::ExprNode(expr) => self.check_expr(&mut stack, idx, (*expr).clone()),
                ast::Node::TermNode(term) => self.check_term(&mut stack, idx, (*term).clone()),
                ast::Node::BlockNode(block) => self.check_block(&mut stack, idx, (*block).clone()),
                ast::Node::FuncNode(func) => self.check_func(&mut stack, idx, (*func).clone()),
                _ => {
                    println!("node -> {:?}", frame.node.clone());
                    panic!("AST node not yet implemented!")
                }
            };
        }
        let resolved_type = stack.first().unwrap().type_t.clone();
        Ok(resolved_type)
    }

    fn check_stmt(
        &mut self,
        stack: &mut Vec<ast::Frame>,
        frame_idx: usize,
        stmt: ast::Stmt,
    ) -> Result<()> {
        let progress = stack.get_mut(frame_idx).unwrap().get_prog();
        match stmt {
            ast::Stmt::Assign(symbol, var, expr) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(1);
                    stack.push(ast::new_frame(
                        ast::Node::SymbolNode(symbol),
                        var.type_t.clone(),
                        0,
                        true,
                    ));
                    stack.push(ast::new_frame(
                        ast::Node::VarNode(var.clone()),
                        var.type_t.clone(),
                        0,
                        true,
                    ));
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(expr),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    // Don't increase the parent, because there isn't one right now
                    let var_frame = stack.pop().unwrap();
                    println!("var frame : {:?}", var_frame);
                    let mut var = match var_frame.node {
                        ast::Node::VarNode(var) => (*var).clone(),
                        _ => panic!("no var!"),
                    };
                    if oper1 != var_frame.type_t && var_frame.type_t != types::Type::Unknown {
                        println!(
                            "declared -> {:?}, expression -> {:?}",
                            var_frame.type_t, oper1
                        );
                        panic!("type of expression doesn't match declared type");
                    }
                    var.type_t = oper1.clone();
                    let symbol = match stack.pop().unwrap().node {
                        ast::Node::SymbolNode(symbol) => symbol,
                        _ => panic!("no symbol!"),
                    };
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1.clone());
                    stack.pop(); // stmt
                    self.sinsert(symbol, var);
                }
            }
            ast::Stmt::Reassign(symbol, var, assign_op, expr) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(1);
                    stack.push(ast::new_frame(
                        ast::Node::SymbolNode(symbol),
                        types::Type::Unknown,
                        0,
                        true,
                    ));
                    stack.push(ast::new_frame(
                        ast::Node::VarNode(var),
                        types::Type::Unknown,
                        0,
                        true,
                    ));
                    stack.push(ast::new_frame(
                        ast::Node::AssignOpNode(assign_op),
                        types::Type::Unknown,
                        0,
                        true,
                    ));
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(expr),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    // Later we must also check that the assign_op is defined for the types
                    stack.pop(); // assign_op
                    stack.pop(); // var
                    let symbol = match stack.pop().unwrap().node {
                        ast::Node::SymbolNode(symbol) => symbol,
                        _ => panic!("stack frame isn't symbol"),
                    };
                    let lookup_type = match self.slookup(symbol) {
                        Some(var) => var.type_t.clone(),
                        None => panic!("ident not found!"),
                    };
                    if oper1 != lookup_type {
                        panic!("type error!");
                    }
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    stack.pop(); // stmt
                                 // Don't increment progress of parent
                }
            }
            ast::Stmt::If(if_cases) => {}
            ast::Stmt::Call(symbol, args) => {
                let func_var = self.slookup(symbol.clone()).unwrap();
                let func_type = match func_var.type_t.clone() {
                    types::Type::Function(func_type) => func_type,
                    _ => panic!("not a function!"),
                };
                let param_types = func_type.params_t;
                if param_types.len() != args.len() {
                    panic!("calling function without correct number of parameters");
                }
                for arg_expr in args.iter() {
                    self.resolve_expr((**arg_expr).clone());
                }
                stack.get_mut(frame_idx).unwrap().set_checked();
            }
            ast::Stmt::FuncDef(func) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(1);
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
                            stack.get(frame_idx).unwrap().node.clone(),
                        ),
                    );
                    stack.push(ast::new_frame(
                        ast::Node::FuncNode(func),
                        function_type,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    // Both sub expressions checked!
                    let func_type = stack.pop().unwrap().get_type();
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(func_type);
                    stack.pop(); // stmt
                                 // Don't increment progress of parent, there might not be one!
                }
            }
        }

        Ok(())
    }

    fn check_expr(
        &mut self,
        stack: &mut Vec<ast::Frame>,
        frame_idx: usize,
        expr: ast::Expr,
    ) -> Result<()> {
        let progress = stack.get_mut(frame_idx).unwrap().get_prog();
        match expr {
            ast::Expr::Add(lhs, rhs) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(2);
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(lhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(rhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 2 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    let oper2 = stack.pop().unwrap().get_type();
                    if oper1 != oper2 {
                        panic!("type error!");
                    }
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    // Increment progress of parent
                    stack
                        .iter_mut()
                        .rev()
                        .find(|x| !x.get_checked())
                        .unwrap()
                        .inc_prog();
                }
            }
            ast::Expr::Sub(lhs, rhs) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(2);
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(lhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(rhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 2 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    let oper2 = stack.pop().unwrap().get_type();
                    if oper1 != oper2 {
                        panic!("type error!");
                    }
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    // Increment progress of parent
                    stack
                        .iter_mut()
                        .rev()
                        .find(|x| !x.get_checked())
                        .unwrap()
                        .inc_prog();
                }
            }
            ast::Expr::Mult(lhs, rhs) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(2);
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(lhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(rhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 2 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    let oper2 = stack.pop().unwrap().get_type();
                    if oper1 != oper2 {
                        panic!("type error!");
                    }
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    // Increment progress of parent
                    stack
                        .iter_mut()
                        .rev()
                        .find(|x| !x.get_checked())
                        .unwrap()
                        .inc_prog();
                }
            }
            ast::Expr::Div(lhs, rhs) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(2);
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(lhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(rhs),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 2 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    let oper2 = stack.pop().unwrap().get_type();
                    if oper1 != oper2 {
                        panic!("type error!");
                    }
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    // Increment progress of parent
                    stack
                        .iter_mut()
                        .rev()
                        .find(|x| !x.get_checked())
                        .unwrap()
                        .inc_prog();
                }
            }
            ast::Expr::Term(term) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(1);
                    stack.push(ast::new_frame(
                        ast::Node::TermNode(term),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    // Term is complete
                    let oper1 = stack.pop().unwrap().get_type();
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    // Increment progress of parent
                    stack
                        .iter_mut()
                        .rev()
                        .find(|x| !x.get_checked())
                        .map(|x| x.inc_prog());
                }
            }
            ast::Expr::Eq(lhs, rhs) => {}  // TODO: implement checking
            ast::Expr::Neq(lhs, rhs) => {} // TODO: implement checking
            ast::Expr::Leq(lhs, rhs) => {} // TODO: implement checking
            ast::Expr::Geq(lhs, rhs) => {} // TODO: implement checking
            ast::Expr::LessThan(lhs, rhs) => {} // TODO: implement checking
            ast::Expr::GreaterThan(lhs, rhs) => {} // TODO: implement checking
            ast::Expr::Call(symbol, args) => {
                let func_var = self.slookup(symbol.clone()).unwrap();
                let func_type = match func_var.type_t.clone() {
                    types::Type::Function(func_type) => func_type,
                    _ => panic!("not a function!"),
                };
                let param_types = func_type.params_t;
                if param_types.len() != args.len() {
                    panic!("calling function without correct number of parameters");
                }
                for (arg_expr, param_type) in args.iter().zip(param_types) {
                    let arg_type = self.resolve_expr((**arg_expr).clone()).unwrap();
                    if param_type != arg_type {
                        panic!(
                            "argument to function {}(...) is incorrect type",
                            symbol.ident
                        )
                    }
                }
                stack.get_mut(frame_idx).unwrap().set_checked();
                let return_t = func_type.return_t.first().unwrap().clone();
                stack.get_mut(frame_idx).unwrap().set_type(return_t);
                stack
                    .iter_mut()
                    .rev()
                    .find(|x| !x.get_checked())
                    .map(|x| x.inc_prog());
            }
        }
        Ok(())
    }

    fn check_term(
        &mut self,
        stack: &mut Vec<ast::Frame>,
        frame_idx: usize,
        term: ast::Term,
    ) -> Result<()> {
        let progress = stack.get_mut(frame_idx).unwrap().get_prog();
        match term {
            ast::Term::Id(ident) => {
                stack.get_mut(frame_idx).unwrap().set_total(0);
                // Lookup type of identifier
                let lookup_type = match self.slookup(new_symbol(ident.clone())) {
                    Some(var) => var.type_t.clone(),
                    None => panic!("ident not found! -> {:?}", ident),
                };

                stack.get_mut(frame_idx).unwrap().set_checked();
                stack.get_mut(frame_idx).unwrap().set_type(lookup_type);
                // Increment progress of parent
                stack
                    .iter_mut()
                    .rev()
                    .find(|x| !x.get_checked())
                    .unwrap()
                    .inc_prog();
            }
            ast::Term::Num(num) => {
                stack.get_mut(frame_idx).unwrap().set_total(0);
                stack.get_mut(frame_idx).unwrap().set_checked();
                stack
                    .get_mut(frame_idx)
                    .unwrap()
                    .set_type(types::Type::Int32);
                // Increment progress of parent
                stack
                    .iter_mut()
                    .rev()
                    .find(|x| !x.get_checked())
                    .unwrap()
                    .inc_prog();
            }
            ast::Term::Bool(bool_value) => {
                stack.get_mut(frame_idx).unwrap().set_total(0);
                stack.get_mut(frame_idx).unwrap().set_checked();
                stack
                    .get_mut(frame_idx)
                    .unwrap()
                    .set_type(types::Type::Bool);
                // Increment progress of parent
                stack
                    .iter_mut()
                    .rev()
                    .find(|x| !x.get_checked())
                    .unwrap()
                    .inc_prog();
            }
            ast::Term::Expr(expr) => {
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(1);
                    stack.push(ast::new_frame(
                        ast::Node::ExprNode(expr),
                        types::Type::Unknown,
                        0,
                        false,
                    ));
                } else if progress == 1 {
                    // Both sub expressions checked!
                    let oper1 = stack.pop().unwrap().get_type();
                    stack.get_mut(frame_idx).unwrap().set_checked();
                    stack.get_mut(frame_idx).unwrap().set_type(oper1);
                    // Increment progress of parent
                    stack
                        .iter_mut()
                        .rev()
                        .find(|x| !x.get_checked())
                        .unwrap()
                        .inc_prog();
                }
            }
        }
        Ok(())
    }

    fn check_block(
        &mut self,
        stack: &mut Vec<ast::Frame>,
        frame_idx: usize,
        block: ast::Block,
    ) -> Result<()> {
        let progress = stack.get_mut(frame_idx).unwrap().get_prog();
        if progress == 0 {
            stack.get_mut(frame_idx).unwrap().set_total(1);
            for stmt in block.iter().rev() {
                stack.push(ast::new_frame(
                    ast::Node::StmtNode(stmt.clone()),
                    types::Type::Unknown,
                    0,
                    false,
                ));
            }
            stack.get_mut(frame_idx).unwrap().inc_prog();
        } else if progress == 1 {
            self.spop();
            while stack.len() - 1 != frame_idx {
                stack.pop();
            }
            stack.get_mut(frame_idx).unwrap().set_checked();
            // Increment progress of parent
            stack
                .iter_mut()
                .rev()
                .find(|x| !x.get_checked())
                .map(|x| x.inc_prog());
        }

        Ok(())
    }

    fn check_func(
        &mut self,
        stack: &mut Vec<ast::Frame>,
        frame_idx: usize,
        func: ast::Func,
    ) -> Result<()> {
        let progress = stack.get_mut(frame_idx).unwrap().get_prog();
        if progress == 0 {
            stack.get_mut(frame_idx).unwrap().set_total(1);
            let block = func.block;
            self.spush();
            let params = func.params;
            for param in params {
                let param_type = param.type_t.clone();
                let symbol = new_symbol(param.ident.clone());
                let var = new_var(param_type, ast::Node::SymbolNode(symbol.clone()));
                self.sinsert(symbol, var);
            }
            stack.push(ast::new_frame(
                ast::Node::BlockNode(block.into()),
                types::Type::Unknown,
                0,
                false,
            ));
        } else if progress == 1 {
            // Both sub expressions checked!
            stack.pop();
            stack.get_mut(frame_idx).unwrap().set_checked();
            // Increment progress of parent
            stack
                .iter_mut()
                .rev()
                .find(|x| !x.get_checked())
                .map(|x| x.inc_prog());
        }
        Ok(())
    }
}
