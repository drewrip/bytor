use crate::ast::*;

pub trait Traverse {
    type Error;

    fn visit_root(&mut self, root: &mut Root) -> Result<(), Self::Error> {
        let Root {
            preblock,
            program,
            postblock,
        } = root;
        self.visit_preblock(preblock)?;
        self.visit_postblock(postblock)?;
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
        for b in block {
            self.visit_stmt(b)?;
        }
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) -> Result<(), Self::Error> {
        match stmt {
            Stmt::If(cases) => self.visit_if_cases(cases),
            Stmt::Assign(symbol, var, expr) => self.visit_expr(expr),
            Stmt::Reassign(symbol, var, assign_op, expr) => self.visit_expr(expr),
            Stmt::Call(symbol, args) => self.visit_args(args),
            Stmt::FuncDef(func) => self.visit_func(func),
            Stmt::Return(expr) => self.visit_expr(expr),
        }
    }

    fn visit_expr(&mut self, expr: &mut TypedExpr) -> Result<(), Self::Error> {
        match expr.expr.clone() {
            Expr::Term(mut term) => {
                self.visit_term(&mut term)?;
            }
            Expr::Add(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Sub(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Mult(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Div(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Eq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Neq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Leq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Geq(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::LessThan(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::GreaterThan(mut lhs, mut rhs) => {
                self.visit_expr(&mut lhs)?;
                self.visit_expr(&mut rhs)?;
            }
            Expr::Not(mut u) => {
                self.visit_expr(&mut u)?;
            }
            Expr::Neg(mut u) => {
                self.visit_expr(&mut u)?;
            }
            Expr::Call(_, mut args) => {
                self.visit_args(&mut args)?;
            }
            Expr::LambdaFunc(mut lf) => {
                self.visit_lambda_func(&mut lf)?;
            }
        };
        Ok(())
    }

    fn visit_term(&mut self, term: &mut TypedTerm) -> Result<(), Self::Error> {
        match term.term.clone() {
            Term::Expr(mut expr) => self.visit_expr(&mut expr),
            Term::Id(_) => Ok(()),
            Term::Num(_) => Ok(()),
            Term::Bool(_) => Ok(()),
            Term::String(_) => Ok(()),
        }
    }

    fn visit_args(&mut self, args: &mut Args) -> Result<(), Self::Error> {
        for arg in args {
            self.visit_expr(arg)?;
        }
        Ok(())
    }

    fn visit_lambda_func(&mut self, lf: &mut LambdaFunc) -> Result<(), Self::Error> {
        self.visit_block(&mut lf.block)
    }

    fn visit_func(&mut self, func: &mut Func) -> Result<(), Self::Error> {
        self.visit_block(&mut func.block)
    }

    fn visit_if_cases(&mut self, cases: &mut IfCases) -> Result<(), Self::Error> {
        for ifcase in cases {
            self.visit_expr(&mut ifcase.condition)?;
            self.visit_block(&mut ifcase.block)?;
        }
        Ok(())
    }
}
