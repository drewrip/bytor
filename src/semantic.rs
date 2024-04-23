use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::ast;
use crate::symbol::{new_symbol, IdentMapping, Symbol, Symbolic, Var};
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
            while !stack.is_empty() {
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
                    _ => panic!("AST node not yet implemented!"),
                };

                println!("{:?}", stack);
            }
        }

        Ok(())
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
                    // Don't increase the parent, because there isn't one right now
                    stack.pop(); // var
                    stack.pop(); // symbol
                    stack.pop(); // stmt
                }
            }
            ast::Stmt::Reassign(symbol, var, assign_op, expr) => {
                stack.get_mut(frame_idx).unwrap().set_total(4);
                stack.push(ast::new_frame(
                    ast::Node::SymbolNode(symbol),
                    types::Type::Unknown,
                    0,
                    true,
                ));
                stack.get_mut(frame_idx).unwrap().set_prog(1);
                stack.push(ast::new_frame(
                    ast::Node::VarNode(var),
                    types::Type::Unknown,
                    0,
                    true,
                ));
                stack.get_mut(frame_idx).unwrap().set_prog(2);
                stack.push(ast::new_frame(
                    ast::Node::AssignOpNode(assign_op),
                    types::Type::Unknown,
                    0,
                    true,
                ));
                stack.get_mut(frame_idx).unwrap().set_prog(3);
                stack.push(ast::new_frame(
                    ast::Node::ExprNode(expr),
                    types::Type::Unknown,
                    0,
                    false,
                ));
            }
            ast::Stmt::Call(ident, args) => {
                stack.get_mut(frame_idx).unwrap().set_total(2);
                stack.push(ast::new_frame(
                    ast::Node::SymbolNode(ident),
                    types::Type::Unknown,
                    0,
                    true,
                ));
                stack.get_mut(frame_idx).unwrap().set_prog(1);
                stack.push(ast::new_frame(
                    ast::Node::ArgsNode(args),
                    types::Type::Unknown,
                    0,
                    false,
                ));
            }
            ast::Stmt::FuncDef(func) => {
                stack.get_mut(frame_idx).unwrap().set_total(1);
                stack.push(ast::new_frame(
                    ast::Node::FuncNode(func),
                    types::Type::Unknown,
                    0,
                    false,
                ));
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
                        .unwrap()
                        .inc_prog();
                }
            }
            ast::Expr::Call(symbol, args) => (),
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
                if progress == 0 {
                    stack.get_mut(frame_idx).unwrap().set_total(0);
                    // Lookup type of identifier
                    let lookup_type = match self.slookup(new_symbol(ident)) {
                        Some(var) => var.type_t.clone(),
                        None => panic!("ident not found!"),
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
            }
            ast::Term::Num(num) => {
                if progress == 0 {
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

    fn check_program(&mut self) -> Result<()> {
        Ok(())
    }
}
