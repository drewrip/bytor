use crate::ast::*;

pub trait Traverse {
    type Result;

    fn visit_root(&mut self, root: &mut Root) {
        let Root {
            preblock,
            program,
            postblock,
        } = root;
        self.visit_preblock(preblock);
        self.visit_postblock(postblock);
        self.visit_program(program);
    }

    fn visit_preblock(&mut self, preblock: &mut Block) {
        self.visit_block(preblock);
    }

    fn visit_postblock(&mut self, postblock: &mut Block) {
        self.visit_block(postblock);
    }

    fn visit_program(&mut self, program: &mut Program) {
        match program {
            Program::With(_, _, _) => {}
            Program::NoWith(_, block) => {
                self.visit_block(block);
            }
        };
    }

    fn visit_block(&mut self, block: &mut Block) {
        for b in block {
            self.visit_stmt(b);
        }
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::If(cases) => {
                self.visit_if_cases(cases);
            }
            Stmt::Assign(symbol, var, expr) => {
                self.visit_expr(expr);
            }
            Stmt::Reassign(symbol, var, assign_op, expr) => {
                self.visit_expr(expr);
            }
            Stmt::Call(symbol, args) => {
                self.visit_args(args);
            }
            Stmt::FuncDef(func) => {
                self.visit_func(func);
            }
            Stmt::Return(expr) => {
                self.visit_expr(expr);
            }
        };
    }

    fn visit_expr(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Term(term) => {
                self.visit_term(term);
            }
            Expr::Add(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Sub(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Mult(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Div(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Eq(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Neq(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Leq(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Geq(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::LessThan(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::GreaterThan(lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Not(uex) => {
                self.visit_expr(uex);
            }
            Expr::Neg(uex) => {
                self.visit_expr(uex);
            }
            Expr::Call(_, args) => {
                self.visit_args(args);
            }
            Expr::LambdaFunc(lf) => {
                self.visit_lambda_func(lf);
            }
        };
    }

    fn visit_term(&mut self, term: &mut Term) {
        println!("{:?}", term);
        match term {
            Term::Expr(expr) => {
                self.visit_expr(expr);
            }
            Term::Id(_) => {}
            Term::Num(_) => {}
            Term::Bool(_) => {}
            Term::String(_) => {}
        };
    }

    fn visit_args(&mut self, args: &mut Args) {
        for arg in args {
            self.visit_expr(arg);
        }
    }

    fn visit_lambda_func(&mut self, lf: &mut LambdaFunc) {
        self.visit_block(&mut lf.block);
    }

    fn visit_func(&mut self, func: &mut Func) {
        self.visit_block(&mut func.block);
    }

    fn visit_if_cases(&mut self, cases: &mut IfCases) {
        for ifcase in cases {
            self.visit_expr(&mut ifcase.condition);
            self.visit_block(&mut ifcase.block);
        }
    }
}
