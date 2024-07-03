use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use clap::{Parser, ValueEnum};
use lalrpop_util::lalrpop_mod;
use thiserror::Error;

pub mod ast;
pub mod backends;
pub mod codegen;
pub mod infer;
pub mod ir;
pub mod semantic;
pub mod symbol;
pub mod traverse;
pub mod types;

use backends::{c::CGenContext, wasm::WasmGenContext};
use codegen::CodeGen;
use semantic::ProgramState;

/// Compiler for the Rascal language
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input Rascal source file
    infile: String,

    /// Name of output binary
    #[arg(short = 'o', long = "outfile", default_value = "a.out")]
    outfile: String,

    /// Skip the WASM Validation after codegen
    #[arg(long = "skip-validation", default_value = "false")]
    skip_validation: bool,

    /// Backend: options will be C, or WASM
    #[arg(short = 'b', long = "backend", value_enum, default_value_t = BackendArgs::C)]
    backend: BackendArgs,

    // Emit: options will be any or both of ir, or C for dumping intermediate reps to file
    #[arg(short = 'e', long = "emit", value_parser, value_delimiter = ',')]
    emit: Option<Vec<EmitArgs>>,
}

#[derive(Clone, Debug, ValueEnum)]
enum BackendArgs {
    C,
    WASM,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EmitArgs {
    Ir,
    C,
}

lalrpop_mod!(pub rascal);

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("There was an issue with the input file: {0}")]
    Input(String),
    #[error("There was a problem creating the output: {0}")]
    Output(String),
}
fn main() -> Result<(), BuildError> {
    let args = Args::parse();
    let save_c: bool;
    let save_ir: bool;
    if let Some(emit) = args.emit {
        save_c = emit.iter().any(|x| matches!(x, EmitArgs::C));
        save_ir = emit.iter().any(|x| matches!(x, EmitArgs::Ir));
    } else {
        (save_c, save_ir) = (false, false);
    }
    let src_file =
        fs::read_to_string(&args.infile).map_err(|err| BuildError::Input(err.to_string()))?;
    let file_extension = Path::new(&args.infile)
        .extension()
        .ok_or(BuildError::Input("Problem with filename".to_string()))?;
    let build_stack = if file_extension == "ir" {
        serde_json::from_str(&src_file).map_err(|err| BuildError::Input(err.to_string()))?
    } else {
        let mut root = rascal::RootParser::new()
            .parse(&src_file)
            .map_err(|err| BuildError::Input(err.to_string()))?;

        let mut typing_state = infer::TypingState::new();
        typing_state
            .augment(&mut root)
            .map_err(|err| BuildError::Output(err.to_string()))?;
        let mut infer_state = infer::InferState::new();
        infer_state
            .constrain(&mut root)
            .map_err(|err| BuildError::Output(err.to_string()))?;
        infer_state
            .resolve()
            .map_err(|err| BuildError::Output(err.to_string()))?;
        let mut sub_state = infer::SubState::new(infer_state.get_type_mapping());
        let _sub_gen_result = sub_state.substitute(&mut root);

        let mut state = semantic::ProgramState::new(root.clone());
        state
            .build_ir()
            .map_err(|err| BuildError::Output(err.to_string()))?;

        if save_ir {
            let serialized_ir = serde_json::to_string(&state.build_stack)
                .map_err(|err| BuildError::Output(err.to_string()))?;
            let mut file = File::create(ProgramState::IR_OUTPUT_FILENAME)
                .map_err(|err| BuildError::Output(err.to_string()))?;
            write!(&mut file, "{serialized_ir}")
                .map_err(|err| BuildError::Output(err.to_string()))?;
        }
        state.build_stack
    };

    // Generate code
    let ctx = codegen::new(build_stack, args.outfile, args.skip_validation);
    let build_result = match args.backend {
        BackendArgs::C => CGenContext::from(ctx).gen(),
        BackendArgs::WASM => WasmGenContext::from(ctx).gen(),
    };
    if !save_c {
        fs::remove_file(CGenContext::C_OUTPUT_FILENAME)
            .map_err(|_| BuildError::Output("Cannot delete C output file".to_string()))?;
    }
    build_result.map_err(|err| BuildError::Output(err.to_string()))?;
    Ok(())
}
