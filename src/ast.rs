use std::sync::Arc;

use crate::semantic;
use crate::symbol::{Symbol, Var};
use crate::types;

pub type Block = Vec<Arc<Stmt>>;

#[derive(Debug, Clone)]
pub struct Frame {
    pub progress: usize,
    pub total: usize,
    pub checked: bool,
    pub node: Node,
    pub type_t: types::Type,
}

pub fn new_frame(node: Node, type_t: types::Type, total: usize, checked: bool) -> Frame {
    Frame {
        progress: 0,
        total,
        checked,
        node,
        type_t,
    }
}

impl Frame {
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
}

#[derive(Debug, Clone)]
pub enum Node {
    RootNode(Arc<Root>),
    ProgramNode(Arc<Program>),
    WithNode(Arc<With>),
    WithVarNode(Arc<WithVar>),
    BlockNode(Arc<Block>),
    ExprNode(Arc<Expr>),
    AssignOpNode(AssignOp),
    StmtNode(Arc<Stmt>),
    ArgsNode(Arc<Args>),
    ParamsNode(Arc<Params>),
    FuncNode(Arc<Func>),
    TypeNode(types::Type),
    TermNode(Arc<Term>),
    SymbolNode(Symbol),
    VarNode(Arc<Var>),
}

#[derive(Debug, Clone)]
pub struct Root {
    pub preblock: Block,
    pub program: Arc<Program>,
    pub postblock: Block,
}

#[derive(Debug, Clone)]
pub enum Program {
    NoWith(Symbol, Block),
    With(Symbol, With, Block),
}

pub type With = Vec<Arc<WithVar>>;

#[derive(Debug, Clone)]
pub enum WithVar {
    Imm(Symbol),
    Mut(Symbol),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Term(Arc<Term>),
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mult(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    Eq(Arc<Expr>, Arc<Expr>),
    Neq(Arc<Expr>, Arc<Expr>),
    Leq(Arc<Expr>, Arc<Expr>),
    Geq(Arc<Expr>, Arc<Expr>),
    LessThan(Arc<Expr>, Arc<Expr>),
    GreaterThan(Arc<Expr>, Arc<Expr>),
    Call(Symbol, Arc<Args>),
}

#[derive(Debug, Clone)]
pub enum Term {
    Id(String),
    Num(i32),
    Bool(bool),
    Expr(Arc<Expr>),
}

#[derive(Debug, Clone)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MultAssign,
    DivAssign,
}

#[derive(Debug, Clone)]
pub struct IfCase {
    pub condition: Arc<Expr>,
    pub block: Block,
}

pub type IfCases = Vec<Arc<IfCase>>;

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign(Symbol, Arc<Var>, Arc<Expr>),
    Reassign(Symbol, Arc<Var>, AssignOp, Arc<Expr>),
    If(IfCases),
    Call(Symbol, Arc<Args>),
    FuncDef(Arc<Func>),
}

pub type Args = Vec<Arc<Expr>>;

pub type Params = Vec<Arc<Param>>;

#[derive(Debug, Clone)]
pub struct Param {
    pub type_t: types::Type,
    pub ident: String,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub ret_t: types::Type,
    pub params: Params,
    pub with: With,
    pub ident: String,
    pub block: Block,
}
