use crate::ast::{self};
use crate::codegen::{CodeGen, CodeGenContext, CodeGenError};
use crate::ir::IRNode;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

pub struct WasmGenContext {
    build_stack: Vec<IRNode>,
    outfile: String,
    skip_validation: bool,
}

impl From<CodeGenContext> for WasmGenContext {
    fn from(ctx: CodeGenContext) -> Self {
        WasmGenContext {
            build_stack: ctx.build_stack,
            outfile: ctx.outfile,
            skip_validation: ctx.skip_validation,
        }
    }
}

impl CodeGen for WasmGenContext {
    fn gen(&mut self) -> Result<(), CodeGenError> {
        let mut module = Module::new();

        // Encode the type section.
        let mut types = TypeSection::new();
        let main_params = vec![];
        let main_results = vec![ValType::I32]; // Return code
        types.function(main_params, main_results);
        module.section(&types);

        // Encode the function section.
        let mut functions = FunctionSection::new();
        let type_index = 0;
        functions.function(type_index);
        module.section(&functions);

        // Encode the export section.
        let mut exports = ExportSection::new();
        exports.export("main", ExportKind::Func, 0);
        module.section(&exports);

        self.gen_globals(&mut module);
        self.gen_program(&mut module);

        let wasm_output = module.finish();
        if !self.skip_validation {
            wasmparser::validate(&wasm_output)
                .map_err(|err| CodeGenError::ValidateFailed(err.message().into()))?;
        }
        let mut file = File::create(self.outfile.clone())
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;
        file.write_all(&wasm_output)
            .map_err(|err| CodeGenError::BinaryWrite(err.to_string()))?;

        Ok(())
    }
}

impl WasmGenContext {
    fn gen_globals(&mut self, module: &mut Module) {}

    fn gen_program(&mut self, module: &mut Module) {
        let mut codes = CodeSection::new();
        let mut main_func = Function::new_with_locals_types(vec![]);

        main_func.instruction(&Instruction::End);
        codes.function(&main_func);
        module.section(&codes);
    }

    fn gen_term(&mut self, module: &mut Module, func: &mut Function, term: ast::Term) {}

    fn gen_expr(&mut self, module: &mut Module, func: &mut Function, expr: ast::Expr) {}

    fn gen_stmt(&mut self, module: &mut Module, func: &mut Function, stmt: ast::Stmt) {}
}
