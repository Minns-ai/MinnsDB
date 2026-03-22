// AST types for MinnsQL.

/// Top-level statement: a query, subscription, DDL, or DML.
#[derive(Debug, Clone)]
pub enum Statement {
    /// Regular MATCH/FROM ... RETURN query.
    Query(Query),
    /// SUBSCRIBE MATCH/FROM ... RETURN — creates a live subscription.
    Subscribe(Query),
    /// UNSUBSCRIBE <id> — removes a subscription.
    Unsubscribe(u64),
    /// CREATE TABLE ...
    CreateTable(CreateTableStmt),
    /// DROP TABLE name
    DropTable(String),
    /// INSERT INTO table VALUES ...
    InsertInto(InsertStmt),
    /// UPDATE table SET ... WHERE ...
    UpdateTable(UpdateStmt),
    /// DELETE FROM table WHERE ...
    DeleteFrom(DeleteStmt),
    /// CREATE INDEX name ON table (columns)
    CreateIndex(CreateIndexStmt),
    /// ALTER TABLE name ADD COLUMN col_def [, col_def ...]
    AlterTable(AlterTableStmt),
}

#[derive(Debug, Clone)]
pub struct AlterTableStmt {
    pub table: String,
    pub add_columns: Vec<ColumnDefAst>,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub match_clauses: Vec<Pattern>,
    /// FROM table_name [AS alias] (driving table for table queries).
    pub from_table: Option<TableRef>,
    /// JOIN clauses (zero or more).
    pub joins: Vec<JoinClause>,
    pub when: Option<WhenClause>,
    pub as_of: Option<Expr>,
    pub where_clause: Option<BoolExpr>,
    /// GROUP BY expressions.
    pub group_by: Vec<Expr>,
    /// HAVING (post-aggregation filter).
    pub having: Option<BoolExpr>,
    pub returns: Vec<ReturnItem>,
    pub order_by: Vec<OrderItem>,
    pub limit: Option<u64>,
}

/// A table reference with optional alias: `table_name [AS alias]`.
#[derive(Debug, Clone)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
}

// -- DDL/DML AST nodes --

#[derive(Debug, Clone)]
pub struct CreateTableStmt {
    pub name: String,
    pub columns: Vec<ColumnDefAst>,
    pub constraints: Vec<ConstraintAst>,
}

#[derive(Debug, Clone)]
pub struct ColumnDefAst {
    pub name: String,
    pub col_type: String,
    pub nullable: bool,
    pub is_primary_key: bool,
    pub default_value: Option<Literal>,
    pub autoincrement: bool,
}

#[derive(Debug, Clone)]
pub struct ConstraintAst {
    pub kind: ConstraintKind,
}

#[derive(Debug, Clone)]
pub enum ConstraintKind {
    PrimaryKey(Vec<String>),
    Unique(Vec<String>),
    NotNull(String),
    ReferencesGraph(String),
}

#[derive(Debug, Clone)]
pub struct InsertStmt {
    pub table: String,
    pub columns: Option<Vec<String>>,
    pub rows: Vec<Vec<Literal>>,
}

#[derive(Debug, Clone)]
pub struct UpdateStmt {
    pub table: String,
    pub assignments: Vec<(String, Expr)>,
    pub where_clause: BoolExpr,
}

#[derive(Debug, Clone)]
pub struct DeleteStmt {
    pub table: String,
    pub where_clause: BoolExpr,
}

#[derive(Debug, Clone)]
pub struct CreateIndexStmt {
    pub index_name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub unique: bool,
}

// -- JOIN AST --

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone)]
pub struct JoinClause {
    pub join_type: JoinType,
    pub table: String,
    pub alias: Option<String>,
    pub on_left: JoinSide,
    pub on_right: JoinSide,
}

#[derive(Debug, Clone)]
pub enum JoinSide {
    /// Table column: table_name.column_name
    TableColumn { table: String, column: String },
    /// Graph variable (for graph-to-table joins)
    GraphVar(String),
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
    /// expr IN (val1, val2, ...)
    In(Expr, Vec<Expr>),
    /// expr NOT IN (val1, val2, ...)
    NotIn(Expr, Vec<Expr>),
    /// expr BETWEEN low AND high
    Between(Expr, Expr, Expr),
    /// expr LIKE pattern
    Like(Expr, String),
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

impl Eq for Literal {}

impl std::hash::Hash for Literal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Literal::String(s) => s.hash(state),
            Literal::Int(i) => i.hash(state),
            Literal::Float(f) => f.to_bits().hash(state),
            Literal::Bool(b) => b.hash(state),
            Literal::Null => {},
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
