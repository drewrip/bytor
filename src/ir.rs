use crate::ast::Node;
use crate::semantic;
use crate::types;

#[derive(Debug, Clone)]
pub struct IRNode {
    pub progress: usize,
    pub total: usize,
    pub checked: bool,
    pub node: Node,
    pub type_t: types::Type,
    pub symbols: Option<semantic::SymbolTable>,
}

pub fn new_ir_node(node: Node, type_t: types::Type, total: usize, checked: bool) -> IRNode {
    IRNode {
        progress: 0,
        total,
        checked,
        node,
        type_t,
        symbols: None,
    }
}

impl IRNode {
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

    pub fn add_symbols(&mut self, symbol_table: Option<semantic::SymbolTable>) {
        self.symbols = symbol_table;
    }
}

pub fn requires_block(node: Node) -> bool {
    match node {
        Node::ProgramNode(_) => true,
        Node::FuncNode(_) => true,
        // TODO: if statements can have blocks,
        _ => false,
    }
}
