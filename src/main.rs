use lalrpop_util::lalrpop_mod;


pub mod ast;
lalrpop_mod!(pub rascal_grammar); // synthesized by LALRPOP

#[test]
fn calculator1() {
    assert!(rascal_grammar::TermParser::new().parse("22 + 22").is_ok());
    assert!(rascal_grammar::TermParser::new().parse("22 +").is_err());
    assert!(rascal_grammar::TermParser::new().parse("(22 + 12) * 2").is_err());
}

fn main() {
    println!("Hello from the Rascal compiler");
}
