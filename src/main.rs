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
    let root = rascal::RootParser::new().parse(&src_file).unwrap();
    // Perform semantic checks and type checking
    let mut state = semantic::new_state(root);
    state.build().unwrap();
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

#[test]
fn root_parser_passing1() {
    let source = r#"
    program passing1
        let x = 9;
        let y = 10;
    end
    "#;
    assert!(rascal::ProgramParser::new().parse(source).is_ok());
}

#[test]
fn root_parser_failing1() {
    let source = r#"
    program failing1
        1 + 2;
    end
    "#;
    assert!(rascal::ProgramParser::new().parse(source).is_err());
}

#[test]
fn type_checking_passing1() {
    let source = r#"
    program passing1
        let x: int32 = 431;
    end
    "#;
    let root = rascal::RootParser::new().parse(source).unwrap();
    let mut state = semantic::new_state(root);
    // Perform semantic checks and type checking
    let build_res = state.build();

    assert!(build_res.is_ok());
}

#[test]
fn type_checking_passing2() {
    let source = r#"
    fun foo(a: float32, b: float32) -> float32
        let c = a;
        c *= b;
    end

    fun baz(x: int32, y: int32, z: int32) -> int32
        let w = x + y + z;
    end

    program passing1
        let x: int32 = 431;
        let test = baz(x, x, x);
    end
    "#;
    let root = rascal::RootParser::new().parse(source).unwrap();
    let mut state = semantic::new_state(root);
    // Perform semantic checks and type checking
    let build_res = state.build();
    assert!(build_res.is_ok());
}

#[test]
#[should_panic]
fn type_checking_func_failing1() {
    let source = r#"
    fun foo(a: int32, b: float32) -> float32
        let c = a;
        let d = b;
    end

    program passing1
        let x: int32 = 431;
        let test = foo(x, x);
    end
    "#;
    let root = rascal::RootParser::new().parse(source).unwrap();
    let mut state = semantic::new_state(root);
    // Perform semantic checks and type checking
    let build_res = state.build();
}

#[test]
fn type_checking_ifs_passing1() {
    let source = r#"
        let x: int32 = 4;

        program test_if
            if x == 5 then
                x = 10;
            end
        end
    "#;
    let root = rascal::RootParser::new().parse(source).unwrap();
    let mut state = semantic::new_state(root);
    // Perform semantic checks and type checking
    let build_res = state.build();
    assert!(build_res.is_ok());
}

#[test]
fn type_checking_ifs_passing2() {
    let source = r#"
        let x = 4;

        program test_if
            if x >= 4 then
                let y = 2;
            end

            if x != 5 then
                x = 10;
            else if x == 6 then
                x = 11;
            else then
                x = 12;
            end
        end
    "#;
    let root = rascal::RootParser::new().parse(source).unwrap();
    let mut state = semantic::new_state(root);
    // Perform semantic checks and type checking
    let build_res = state.build();
    assert!(build_res.is_ok());
}
