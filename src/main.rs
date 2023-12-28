use std::fs;

use lalrpop_util::lalrpop_mod;
use clap::Parser;

pub mod ast;

/// Compiler for the Rascal language
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input Rascal source file
    infile: String,

    /// Name of output binary
    #[arg(short, long, default_value = "a.out")]
    outfile: String,
}

lalrpop_mod!(pub rascal_grammar);

fn main() {
    let args = Args::parse();
    
    let src_file = fs::read_to_string(args.infile).expect("ERROR: couldn't find source file");
    let root = rascal_grammar::ProgramParser::new().parse(&src_file);
    println!("{:?}", root);
}

#[test]
fn root_parser_passing1() {
    let source = r#"
    program
        let x = 9;
        let y = 10;
    end
    "#;
    assert!(rascal_grammar::ProgramParser::new().parse(source).is_ok());
}

#[test]
fn root_parser_failing1() {
    let source = r#"
    program
        1 + 2;
    end
    "#;
    assert!(rascal_grammar::ProgramParser::new().parse(source).is_err());
}
