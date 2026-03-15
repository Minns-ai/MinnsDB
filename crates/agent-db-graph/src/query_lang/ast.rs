/// AST types for MinnsQL.

#[derive(Debug, Clone)]
pub struct Query {
    pub match_clauses: Vec<Pattern>,
    pub when: Option<WhenClause>,
    pub as_of: Option<Expr>,
    pub where_clause: Option<BoolExpr>,
    pub returns: Vec<ReturnItem>,
    pub order_by: Vec<OrderItem>,
    pub limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern, Direction),
}

#[derive(Debug, Clone)]
pub struct NodePattern {
    pub var: Option<String>,
    pub labels: Vec<String>,
    pub props: Vec<(String, Literal)>,
}

#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub var: Option<String>,
    pub edge_type: Option<String>,
    pub range: Option<(u32, Option<u32>)>,
    pub props: Vec<(String, Literal)>,
}

#[derive(Debug, Clone)]
pub enum WhenClause {
    PointInTime(Expr),
    Range(Expr, Expr),
    Last(String),
    All,
}

#[derive(Debug, Clone)]
pub enum BoolExpr {
    Comparison(Expr, CompOp, Expr),
    IsNull(Expr),
    IsNotNull(Expr),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    Not(Box<BoolExpr>),
    Paren(Box<BoolExpr>),
    /// A boolean-returning function call in WHERE (e.g. SUCCESSIVE, CHANGED, overlap).
    FuncPredicate(String, Vec<Expr>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Property(String, String),
    Literal(Literal),
    FuncCall(String, Vec<Expr>),
    Var(String),
    Star,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompOp {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    Contains,
    StartsWith,
}

#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
    pub distinct: bool,
}

#[derive(Debug, Clone)]
pub struct OrderItem {
    pub expr: Expr,
    pub descending: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Direction {
    Out,
    In,
}
