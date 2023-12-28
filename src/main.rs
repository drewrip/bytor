use lalrpop_util::lalrpop_mod;

pub mod ast;

lalrpop_mod!(pub rascal_grammar);

fn main() {
    let source 
    println!("Hello from the Rascal compiler");
}

#[test]
fn root_parser_passing1() {
    let source = r#"
    program
        22 + 2
    end
    "#;
    assert!(rascal_grammar::ProgramParser::new().parse(source).is_ok());
}

#[test]
fn root_parser_failing1() {
    let source = r#"
    program
        22 +
    end
    "#;
    assert!(rascal_grammar::ProgramParser::new().parse(source).is_err());
}
