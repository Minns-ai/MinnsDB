/// Token types for MinnsQL lexer.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Match,
    When,
    Where,
    Return,
    Order,
    By,
    Limit,
    As,
    Of,
    And,
    Or,
    Not,
    Is,
    Null,
    Distinct,
    Asc,
    Desc,
    All,
    Last,
    To,
    True,
    False,
    Contains,
    StartsWith,
    Subscribe,
    Unsubscribe,

    // DDL/DML/Table keywords
    Create,
    Table,
    Drop,
    Insert,
    Into,
    Values,
    Update,
    Set,
    Delete,
    From,
    Join,
    On,
    Primary,
    Key,
    Unique,
    References,
    Graph,

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
