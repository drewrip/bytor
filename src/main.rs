use std::{
    fs::{self, File},
    io::Write,
};

use clap::{Parser, ValueEnum};
use lalrpop_util::lalrpop_mod;

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

fn main() {
    let args = Args::parse();
    let save_c: bool;
    let save_ir: bool;
    if let Some(emit) = args.emit {
        save_c = emit.iter().any(|x| matches!(x, EmitArgs::C));
        save_ir = emit.iter().any(|x| matches!(x, EmitArgs::Ir));
    } else {
        (save_c, save_ir) = (false, false);
    }
    let src_file = fs::read_to_string(args.infile).expect("ERROR: couldn't find source file");
    let mut root = rascal::RootParser::new().parse(&src_file).unwrap();

    let mut typing_state = infer::TypingState::new();
    let _tv_typing_result = typing_state
        .augment(&mut root)
        .expect("Couldn't replace unknown types with TypeVar's");
    let mut infer_state = infer::InferState::new();
    let _constraint_gen_result = infer_state
        .constrain(&mut root)
        .expect("Couldn't add contraints");
    let _resolve_result = infer_state.resolve().expect("Couldn't resolve types");
    let mut sub_state = infer::SubState::new(infer_state.get_type_mapping());
    let _sub_gen_result = sub_state.substitute(&mut root);

    // Perform semantic checks and type checking
    let mut state = semantic::ProgramState::new(root.clone());
    state.build_ir().expect("couldn't generate IR");

    if save_ir {
        let serialized_ir = serde_json::to_string(&state.build_stack).expect("Serialization error");
        let mut file =
            File::create(ProgramState::IR_OUTPUT_FILENAME).expect("Cannot create IR file");
        write!(&mut file, "{serialized_ir}").expect("Cannot write to IR file");
    }

    // Generate code
    let ctx = codegen::new(state.build_stack, args.outfile, args.skip_validation);
    let build_result = match args.backend {
        BackendArgs::C => CGenContext::from(ctx).gen(),
        BackendArgs::WASM => WasmGenContext::from(ctx).gen(),
    };
    if !save_c {
        fs::remove_file(CGenContext::C_OUTPUT_FILENAME).expect("Unable to delete C output file");
    }
    build_result.expect("Build failed!");
}
