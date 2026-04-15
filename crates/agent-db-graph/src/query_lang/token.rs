/// Token types for MinnsQL lexer.
///
/// ## Reserved vs soft keywords
///
/// Only operators, literals, clause delimiters, and top-level statement verbs
/// retain specialized `Token` variants. Everything else — `KEY`, `TABLE`,
/// `PRIMARY`, `VALUES`, `SET`, etc. — lexes as `Token::Ident(String)` and is
/// recognized by string comparison at the specific parser sites where it has
/// structural meaning. This means a user column named `key` or `table` is a
/// normal identifier everywhere in expression context and does not need any
/// special handling in the expression parser.
///
/// The set of truly-reserved keywords is the minimum set where letting the
/// token act as an identifier would change the parse of a currently-valid
/// query: boolean and comparison operators, null/true/false literals, and
/// tokens the parser pattern-matches to delimit clauses or dispatch statements.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Clause delimiters and query structure
    Match,
    When,
    Where,
    Return,
    Order,
    By,
    Limit,
    As,
    Of,
    Distinct,
    Asc,
    Desc,
    From,
    Join,
    Left,
    Inner,
    On,
    Group,
    Having,

    // Expression operators
    And,
    Or,
    Not,
    Is,
    In,
    Between,
    Like,
    Contains,
    StartsWith,

    // Literal keywords
    Null,
    True,
    False,

    // Statement-top-level verbs (only matched at statement dispatch)
    Create,
    Drop,
    Insert,
    Update,
    Delete,
    Alter,
    Subscribe,
    Unsubscribe,

    // Literals
    Ident(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),

    // Symbols
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Colon,
    Comma,
    Dot,
    Arrow,     // ->
    BackArrow, // <-
    Dash,      // -
    Star,      // *
    DotDot,    // ..
    Eq,        // =
    Neq,       // !=
    Lt,        // <
    Gt,        // >
    Lte,       // <=
    Gte,       // >=

    Eof,
}

#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub span: (usize, usize),
}
