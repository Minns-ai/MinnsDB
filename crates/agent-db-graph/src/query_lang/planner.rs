/// Query planner for MinnsQL.
///
/// The planner converts an AST `Query` into an `ExecutionPlan` — a sequence of
/// physical `PlanStep`s that the executor evaluates against the graph.
use std::collections::HashMap;

use super::ast::*;
use super::types::QueryError;
use agent_db_core::types::Timestamp;

// ---------------------------------------------------------------------------
// Slot-based variable table
// ---------------------------------------------------------------------------

/// Index into a `BindingRow`'s slot array.
pub type SlotIdx = u8;

/// Maps variable names to monotonic slot indices during planning.
/// Strings are owned here; the executor never touches them.
#[derive(Debug, Clone)]
pub struct VarTable {
    map: HashMap<String, SlotIdx>,
    pub count: u8,
}

impl Default for VarTable {
    fn default() -> Self {
        Self::new()
    }
}

impl VarTable {
    pub fn new() -> Self {
        VarTable {
            map: HashMap::new(),
            count: 0,
        }
    }

    /// Get or assign a slot index for `name`.
    pub fn get_or_insert(&mut self, name: &str) -> SlotIdx {
        if let Some(&idx) = self.map.get(name) {
            return idx;
        }
        let idx = self.count;
        self.count = self.count.checked_add(1).expect("exceeded 255 variables");
        self.map.insert(name.to_string(), idx);
        idx
    }

    /// Look up an existing slot. Returns `None` if the variable was never bound.
    pub fn lookup(&self, name: &str) -> Option<SlotIdx> {
        self.map.get(name).copied()
    }
}

// ---------------------------------------------------------------------------
// Resolved expression types (executor never does string lookups)
// ---------------------------------------------------------------------------

/// Expression with variable references resolved to slot indices.
#[derive(Debug, Clone)]
pub enum RExpr {
    Property(SlotIdx, String),
    Literal(Literal),
    FuncCall(String, Vec<RExpr>),
    Var(SlotIdx),
    Star,
}

/// Boolean expression with resolved variable references.
#[derive(Debug, Clone)]
pub enum RBoolExpr {
    Comparison(RExpr, CompOp, RExpr),
    IsNull(RExpr),
    IsNotNull(RExpr),
    And(Box<RBoolExpr>, Box<RBoolExpr>),
    Or(Box<RBoolExpr>, Box<RBoolExpr>),
    Not(Box<RBoolExpr>),
    Paren(Box<RBoolExpr>),
    FuncPredicate(String, Vec<RExpr>),
}

/// Resolve an AST `Expr` into an `RExpr` using the VarTable.
fn resolve_expr(expr: &Expr, vt: &VarTable) -> Result<RExpr, QueryError> {
    match expr {
        Expr::Property(var, prop) => {
            let slot = vt.lookup(var).ok_or_else(|| {
                QueryError::PlanError(format!("Unbound variable `{}` in expression", var))
            })?;
            Ok(RExpr::Property(slot, prop.clone()))
        },
        Expr::Literal(lit) => Ok(RExpr::Literal(lit.clone())),
        Expr::FuncCall(name, args) => {
            let resolved_args: Result<Vec<RExpr>, QueryError> =
                args.iter().map(|a| resolve_expr(a, vt)).collect();
            Ok(RExpr::FuncCall(name.clone(), resolved_args?))
        },
        Expr::Var(var) => {
            let slot = vt.lookup(var).ok_or_else(|| {
                QueryError::PlanError(format!("Unbound variable `{}` in expression", var))
            })?;
            Ok(RExpr::Var(slot))
        },
        Expr::Star => Ok(RExpr::Star),
    }
}

/// Resolve an AST `BoolExpr` into an `RBoolExpr`.
fn resolve_bool_expr(expr: &BoolExpr, vt: &VarTable) -> Result<RBoolExpr, QueryError> {
    match expr {
        BoolExpr::Comparison(lhs, op, rhs) => Ok(RBoolExpr::Comparison(
            resolve_expr(lhs, vt)?,
            op.clone(),
            resolve_expr(rhs, vt)?,
        )),
        BoolExpr::IsNull(e) => Ok(RBoolExpr::IsNull(resolve_expr(e, vt)?)),
        BoolExpr::IsNotNull(e) => Ok(RBoolExpr::IsNotNull(resolve_expr(e, vt)?)),
        BoolExpr::And(a, b) => Ok(RBoolExpr::And(
            Box::new(resolve_bool_expr(a, vt)?),
            Box::new(resolve_bool_expr(b, vt)?),
        )),
        BoolExpr::Or(a, b) => Ok(RBoolExpr::Or(
            Box::new(resolve_bool_expr(a, vt)?),
            Box::new(resolve_bool_expr(b, vt)?),
        )),
        BoolExpr::Not(inner) => Ok(RBoolExpr::Not(Box::new(resolve_bool_expr(inner, vt)?))),
        BoolExpr::Paren(inner) => Ok(RBoolExpr::Paren(Box::new(resolve_bool_expr(inner, vt)?))),
        BoolExpr::FuncPredicate(name, args) => {
            let resolved: Result<Vec<RExpr>, QueryError> =
                args.iter().map(|a| resolve_expr(a, vt)).collect();
            Ok(RBoolExpr::FuncPredicate(name.clone(), resolved?))
        },
    }
}

// ---------------------------------------------------------------------------
// Plan types
// ---------------------------------------------------------------------------

/// A fully resolved execution plan ready for the executor.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub steps: Vec<PlanStep>,
    pub projections: Vec<Projection>,
    pub aggregations: Vec<Aggregation>,
    pub group_by_keys: Vec<String>,
    pub ordering: Vec<OrderSpec>,
    pub limit: Option<u64>,
    pub temporal_viewport: TemporalViewport,
    /// Transaction-time cutoff from AS OF clause.
    /// When set, only nodes/edges with `created_at <= cutoff` are visible.
    pub transaction_cutoff: Option<Timestamp>,
    /// Number of variable slots needed by the executor.
    pub var_count: u8,
}

/// A single physical operation in the plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    /// Scan nodes matching optional labels and property filters, binding to `var`.
    ScanNodes {
        var: SlotIdx,
        labels: Vec<String>,
        props: Vec<(String, Literal)>,
    },
    /// Expand from a bound node along edges.
    Expand {
        from_var: SlotIdx,
        edge_var: Option<SlotIdx>,
        to_var: SlotIdx,
        edge_type: Option<String>,
        direction: Direction,
        /// Variable-length range: (min_hops, max_hops). `None` means single hop.
        range: Option<(u32, Option<u32>)>,
    },
    /// Filter rows by a resolved boolean expression.
    Filter(RBoolExpr),
    /// Scan all rows from a table (FROM table_name).
    ScanTable { table_name: String },
    /// Join with a table. Follows a ScanTable or graph steps.
    JoinTable {
        table_name: String,
        on: JoinCondition,
    },
}

/// Join condition for JoinTable plan step.
#[derive(Debug, Clone)]
pub enum JoinCondition {
    /// table.column = other_table.column (table-to-table equi-join)
    TableToTable {
        left_table: String,
        left_col: String,
        right_col: String,
    },
    /// table.noderef_column = graph_var (graph-to-table join)
    GraphToTable {
        graph_var: SlotIdx,
        table_col: String,
    },
}

/// Temporal window for edge visibility.
#[derive(Debug, Clone)]
pub enum TemporalViewport {
    /// Only edges with `valid_until == None` (currently active).
    ActiveOnly,
    /// Edges valid at a specific point in time.
    PointInTime(Timestamp),
    /// Edges valid during a time range.
    Range(Timestamp, Timestamp),
    /// All edges regardless of temporal validity.
    All,
}

/// A column in the output.
#[derive(Debug, Clone)]
pub struct Projection {
    pub expr: RExpr,
    pub alias: String,
    pub distinct: bool,
}

/// An aggregate function applied to a column.
#[derive(Debug, Clone)]
pub struct Aggregation {
    pub function: AggregateFunction,
    pub input_expr: RExpr,
    pub output_alias: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Collect,
}

/// Ordering specification for ORDER BY.
#[derive(Debug, Clone)]
pub struct OrderSpec {
    pub column_alias: String,
    pub descending: bool,
}

// ---------------------------------------------------------------------------
// Plan construction
// ---------------------------------------------------------------------------

/// Convert an AST `Query` into an `ExecutionPlan`.
pub fn plan(query: Query) -> Result<ExecutionPlan, QueryError> {
    let mut steps = Vec::new();
    let mut vt = VarTable::new();

    // 1a. Process FROM table (if present) into a ScanTable step.
    // Register table names as pseudo-variables so property resolution works.
    if let Some(ref table_name) = query.from_table {
        vt.get_or_insert(table_name);
        steps.push(PlanStep::ScanTable {
            table_name: table_name.clone(),
        });
    }

    // 1b. Process MATCH patterns into scan/expand steps.
    for pattern in &query.match_clauses {
        plan_pattern(pattern, &mut steps, &mut vt)?;
    }

    // 1c. Process JOIN clauses into JoinTable steps.
    for join in &query.joins {
        // Register joined table name as pseudo-variable for property resolution.
        vt.get_or_insert(&join.table);
        let on = plan_join_condition(join, &vt)?;
        steps.push(PlanStep::JoinTable {
            table_name: join.table.clone(),
            on,
        });
    }

    // 2. Process WHEN clause into a temporal viewport.
    let temporal_viewport = plan_when(&query.when)?;

    // 2b. Process AS OF clause into a transaction-time cutoff.
    let transaction_cutoff = match &query.as_of {
        Some(expr) => Some(resolve_timestamp_expr(expr)?),
        None => None,
    };

    // 3. Process WHERE clause into a Filter step (resolve to RBoolExpr).
    if let Some(ref cond) = query.where_clause {
        // For table queries, WHERE references may be table.column (Property nodes).
        // resolve_bool_expr handles Expr::Property via VarTable, but table columns
        // don't have slot variables. We pass them through as-is; the executor
        // resolves table.column references at runtime.
        steps.push(PlanStep::Filter(resolve_bool_expr(cond, &vt)?));
    }

    // 4. Process RETURN into projections, aggregations, and group-by keys.
    let (projections, aggregations, group_by_keys) = plan_return(&query.returns, &vt)?;

    // 5. ORDER BY
    let ordering: Vec<OrderSpec> = query
        .order_by
        .iter()
        .map(|o| OrderSpec {
            column_alias: expr_to_alias(&o.expr, 0),
            descending: o.descending,
        })
        .collect();

    Ok(ExecutionPlan {
        steps,
        temporal_viewport,
        transaction_cutoff,
        projections,
        ordering,
        limit: query.limit,
        aggregations,
        group_by_keys,
        var_count: vt.count,
    })
}

/// Walk pattern elements (Node, Edge, Node, Edge, ...) and emit plan steps.
///
/// The first node becomes a `ScanNodes`; subsequent (Edge, Node) pairs become
/// `Expand` steps chained from the previous node variable.
fn plan_pattern(
    pattern: &Pattern,
    steps: &mut Vec<PlanStep>,
    vt: &mut VarTable,
) -> Result<(), QueryError> {
    let mut iter = pattern.elements.iter();

    // First element must be a Node.
    let first = iter
        .next()
        .ok_or_else(|| QueryError::PlanError("Empty pattern".into()))?;
    let first_slot = match first {
        PatternElement::Node(np) => {
            let var_name = np
                .var
                .clone()
                .unwrap_or_else(|| format!("_anon_{}", steps.len()));
            let slot = vt.get_or_insert(&var_name);
            steps.push(PlanStep::ScanNodes {
                var: slot,
                labels: np.labels.clone(),
                props: np.props.clone(),
            });
            slot
        },
        _ => {
            return Err(QueryError::PlanError(
                "Pattern must start with a node".into(),
            ))
        },
    };

    let mut current_slot = first_slot;

    // Remaining elements arrive in (Edge, Node) pairs.
    while let Some(edge_elem) = iter.next() {
        let (ep, dir) = match edge_elem {
            PatternElement::Edge(ep, dir) => (ep, dir),
            _ => return Err(QueryError::PlanError("Expected edge in pattern".into())),
        };

        let node_elem = iter
            .next()
            .ok_or_else(|| QueryError::PlanError("Edge must be followed by a node".into()))?;
        let np = match node_elem {
            PatternElement::Node(np) => np,
            _ => return Err(QueryError::PlanError("Expected node after edge".into())),
        };

        let edge_slot = ep.var.as_ref().map(|v| vt.get_or_insert(v));

        let to_var_name = np
            .var
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", steps.len()));
        let to_slot = vt.get_or_insert(&to_var_name);

        steps.push(PlanStep::Expand {
            from_var: current_slot,
            edge_var: edge_slot,
            to_var: to_slot,
            edge_type: ep.edge_type.clone(),
            direction: dir.clone(),
            range: ep.range,
        });

        current_slot = to_slot;
    }

    Ok(())
}

/// Convert an AST JoinClause into a JoinCondition.
fn plan_join_condition(join: &JoinClause, vt: &VarTable) -> Result<JoinCondition, QueryError> {
    match (&join.on_left, &join.on_right) {
        // table.col = table.col (table-to-table)
        (
            JoinSide::TableColumn {
                table: lt,
                column: lc,
            },
            JoinSide::TableColumn {
                table: _rt,
                column: rc,
            },
        ) => Ok(JoinCondition::TableToTable {
            left_table: lt.clone(),
            left_col: lc.clone(),
            right_col: rc.clone(),
        }),
        // table.col = graph_var
        (
            JoinSide::TableColumn {
                table: _,
                column: tc,
            },
            JoinSide::GraphVar(gv),
        ) => {
            let slot = vt.lookup(gv).ok_or_else(|| {
                QueryError::PlanError(format!("Unbound graph variable `{}` in JOIN ON", gv))
            })?;
            Ok(JoinCondition::GraphToTable {
                graph_var: slot,
                table_col: tc.clone(),
            })
        },
        // graph_var = table.col (reversed)
        (
            JoinSide::GraphVar(gv),
            JoinSide::TableColumn {
                table: _,
                column: tc,
            },
        ) => {
            let slot = vt.lookup(gv).ok_or_else(|| {
                QueryError::PlanError(format!("Unbound graph variable `{}` in JOIN ON", gv))
            })?;
            Ok(JoinCondition::GraphToTable {
                graph_var: slot,
                table_col: tc.clone(),
            })
        },
        // graph_var = graph_var (not supported)
        (JoinSide::GraphVar(_), JoinSide::GraphVar(_)) => Err(QueryError::PlanError(
            "JOIN ON must reference at least one table column".into(),
        )),
    }
}

/// Translate the optional WHEN clause into a `TemporalViewport`.
fn plan_when(when: &Option<WhenClause>) -> Result<TemporalViewport, QueryError> {
    match when {
        None => Ok(TemporalViewport::ActiveOnly),
        Some(WhenClause::All) => Ok(TemporalViewport::All),
        Some(WhenClause::Last(duration)) => {
            let nanos = parse_duration_nanos(duration).map_err(|e| {
                QueryError::PlanError(format!("Invalid duration '{}': {}", duration, e))
            })?;
            let now = now_nanos();
            Ok(TemporalViewport::Range(now.saturating_sub(nanos), now))
        },
        Some(WhenClause::PointInTime(expr)) => {
            let ts = resolve_timestamp_expr(expr)?;
            Ok(TemporalViewport::PointInTime(ts))
        },
        Some(WhenClause::Range(from, to)) => {
            let t1 = resolve_timestamp_expr(from)?;
            let t2 = resolve_timestamp_expr(to)?;
            Ok(TemporalViewport::Range(t1, t2))
        },
    }
}

/// Resolve a timestamp expression used inside a WHEN clause.
fn resolve_timestamp_expr(expr: &Expr) -> Result<u64, QueryError> {
    match expr {
        Expr::Literal(Literal::String(s)) => parse_timestamp_str(s),
        Expr::Literal(Literal::Int(i)) => Ok(*i as u64),
        Expr::FuncCall(name, args) if name == "ago" => {
            if let Some(Expr::Literal(Literal::String(dur))) = args.first() {
                let nanos = parse_duration_nanos(dur)
                    .map_err(|e| QueryError::PlanError(format!("Invalid duration: {}", e)))?;
                let now = now_nanos();
                Ok(now.saturating_sub(nanos))
            } else {
                Err(QueryError::PlanError(
                    "ago() requires a duration string argument".into(),
                ))
            }
        },
        Expr::FuncCall(name, _) if name == "now" => Ok(now_nanos()),
        _ => Err(QueryError::PlanError(
            "Expected timestamp string or function in WHEN clause".into(),
        )),
    }
}

/// Current time as nanoseconds since the Unix epoch.
fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Parse an ISO-8601 date/datetime string to nanoseconds since epoch.
///
/// Accepted formats: epoch milliseconds, `YYYY-MM`, `YYYY-MM-DD`,
/// `YYYY-MM-DDTHH:MM:SS`.
pub fn parse_timestamp_str(s: &str) -> Result<u64, QueryError> {
    // Try epoch milliseconds first.
    if let Ok(ms) = s.parse::<u64>() {
        return Ok(ms * 1_000_000); // ms -> nanos
    }

    let parts: Vec<&str> = s.split('T').collect();
    let date_parts: Vec<&str> = parts[0].split('-').collect();

    if date_parts.len() < 2 {
        return Err(QueryError::PlanError(format!(
            "Cannot parse timestamp: '{}'",
            s
        )));
    }

    let year: i32 = date_parts[0]
        .parse()
        .map_err(|_| QueryError::PlanError(format!("Invalid year in '{}'", s)))?;
    let month: u32 = date_parts[1]
        .parse()
        .map_err(|_| QueryError::PlanError(format!("Invalid month in '{}'", s)))?;
    let day: u32 = if date_parts.len() > 2 {
        date_parts[2]
            .parse()
            .map_err(|_| QueryError::PlanError(format!("Invalid day in '{}'", s)))?
    } else {
        1
    };

    let (hour, min, sec) = if parts.len() > 1 {
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        let h = time_parts
            .first()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        let m = time_parts
            .get(1)
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        let s = time_parts
            .get(2)
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        (h, m, s)
    } else {
        (0, 0, 0)
    };

    let days_since_epoch = days_from_civil(year, month, day);
    let total_secs =
        days_since_epoch as u64 * 86400 + hour as u64 * 3600 + min as u64 * 60 + sec as u64;
    Ok(total_secs * 1_000_000_000)
}

/// Days from a civil date to the Unix epoch.
///
/// Uses Howard Hinnant's algorithm.
pub(crate) fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y as i64;
    let m = m as i64;
    let d = d as i64;
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as u64;
    era * 146097 + doe as i64 - 719468
}

/// Inverse of `days_from_civil`: convert a day count since Unix epoch to (year, month, day).
///
/// Uses Howard Hinnant's algorithm.
pub(crate) fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d)
}

/// Extract (year, month, day) from a nanosecond timestamp.
pub(crate) fn nanos_to_civil(nanos: u64) -> (i32, u32, u32) {
    let days = (nanos / (86_400 * 1_000_000_000)) as i64;
    civil_from_days(days)
}

/// Convert a civil date back to nanoseconds (midnight UTC).
pub(crate) fn civil_to_nanos(y: i32, m: u32, d: u32) -> u64 {
    let days = days_from_civil(y, m, d);
    (days as u64) * 86_400 * 1_000_000_000
}

/// Parse a duration string such as `"30d"`, `"6h"`, `"1y"`, `"2w"` to nanoseconds.
pub fn parse_duration_nanos(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty duration".into());
    }

    let num_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if num_end == 0 {
        return Err(format!("no numeric value in duration '{}'", s));
    }

    let value: u64 = s[..num_end]
        .parse()
        .map_err(|e| format!("invalid number: {}", e))?;
    let unit = &s[num_end..];

    let multiplier_nanos: u64 = match unit {
        "ns" => 1,
        "us" | "\u{b5}s" => 1_000,
        "ms" => 1_000_000,
        "s" => 1_000_000_000,
        "m" | "min" => 60 * 1_000_000_000,
        "h" => 3_600 * 1_000_000_000,
        "d" => 86_400 * 1_000_000_000,
        "w" => 7 * 86_400 * 1_000_000_000,
        "y" => 365 * 86_400 * 1_000_000_000,
        _ => {
            return Err(format!(
                "unknown duration unit '{}' (expected d, h, m, s, w, y, ms, us, ns)",
                unit
            ))
        },
    };

    Ok(value * multiplier_nanos)
}

/// Translate RETURN items into projections, aggregations, and group-by keys.
fn plan_return(
    items: &[ReturnItem],
    vt: &VarTable,
) -> Result<(Vec<Projection>, Vec<Aggregation>, Vec<String>), QueryError> {
    let mut projections = Vec::new();
    let mut aggregations = Vec::new();
    let mut group_by_keys = Vec::new();

    for (i, item) in items.iter().enumerate() {
        let alias = item
            .alias
            .clone()
            .unwrap_or_else(|| expr_to_alias(&item.expr, i));

        if let Some(agg_fn) = extract_aggregation(&item.expr) {
            if let Expr::FuncCall(_, args) = &item.expr {
                let input_ast = args.first().cloned().unwrap_or(Expr::Star);
                aggregations.push(Aggregation {
                    function: agg_fn,
                    input_expr: resolve_expr(&input_ast, vt)?,
                    output_alias: alias.clone(),
                });
            }
        } else {
            group_by_keys.push(alias.clone());
        }

        projections.push(Projection {
            expr: resolve_expr(&item.expr, vt)?,
            alias,
            distinct: item.distinct,
        });
    }

    Ok((projections, aggregations, group_by_keys))
}

/// If the expression is an aggregation function call, return the corresponding
/// `AggregateFunction` variant.
fn extract_aggregation(expr: &Expr) -> Option<AggregateFunction> {
    if let Expr::FuncCall(name, _) = expr {
        match name.as_str() {
            "count" => Some(AggregateFunction::Count),
            "sum" => Some(AggregateFunction::Sum),
            "avg" => Some(AggregateFunction::Avg),
            "min" => Some(AggregateFunction::Min),
            "max" => Some(AggregateFunction::Max),
            "collect" => Some(AggregateFunction::Collect),
            _ => None,
        }
    } else {
        None
    }
}

/// Derive a human-readable alias from an expression.
fn expr_to_alias(expr: &Expr, index: usize) -> String {
    match expr {
        Expr::Property(var, prop) => format!("{}.{}", var, prop),
        Expr::Var(var) => var.clone(),
        Expr::FuncCall(name, _) => name.clone(),
        Expr::Literal(lit) => format!("{:?}", lit),
        Expr::Star => format!("_col_{}", index),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: minimal query with no match/where/when/order/limit.
    fn empty_query() -> Query {
        Query {
            match_clauses: vec![],
            from_table: None,
            joins: vec![],
            when: None,
            as_of: None,
            where_clause: None,
            returns: vec![],
            order_by: vec![],
            limit: None,
        }
    }

    // 1. Simple MATCH (node) RETURN query produces a ScanNodes step.
    #[test]
    fn test_plan_single_node_scan() {
        let query = Query {
            match_clauses: vec![Pattern {
                elements: vec![PatternElement::Node(NodePattern {
                    var: Some("p".into()),
                    labels: vec!["Person".into()],
                    props: vec![("name".into(), Literal::String("Alice".into()))],
                })],
            }],
            returns: vec![ReturnItem {
                expr: Expr::Var("p".into()),
                alias: None,
                distinct: false,
            }],
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        assert_eq!(ep.steps.len(), 1);
        match &ep.steps[0] {
            PlanStep::ScanNodes { var, labels, props } => {
                assert_eq!(*var, 0); // first variable gets slot 0
                assert_eq!(labels, &["Person"]);
                assert_eq!(props.len(), 1);
                assert_eq!(props[0].0, "name");
            },
            other => panic!("expected ScanNodes, got {:?}", other),
        }
        assert_eq!(ep.var_count, 1);
    }

    // 2. MATCH with edge produces ScanNodes + Expand.
    #[test]
    fn test_plan_node_edge_node() {
        let query = Query {
            match_clauses: vec![Pattern {
                elements: vec![
                    PatternElement::Node(NodePattern {
                        var: Some("a".into()),
                        labels: vec!["Person".into()],
                        props: vec![],
                    }),
                    PatternElement::Edge(
                        EdgePattern {
                            var: Some("r".into()),
                            edge_type: Some("KNOWS".into()),
                            range: None,
                            props: vec![],
                        },
                        Direction::Out,
                    ),
                    PatternElement::Node(NodePattern {
                        var: Some("b".into()),
                        labels: vec!["Person".into()],
                        props: vec![],
                    }),
                ],
            }],
            returns: vec![ReturnItem {
                expr: Expr::Var("b".into()),
                alias: None,
                distinct: false,
            }],
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        assert_eq!(ep.steps.len(), 2);
        assert!(matches!(&ep.steps[0], PlanStep::ScanNodes { var, .. } if *var == 0));
        match &ep.steps[1] {
            PlanStep::Expand {
                from_var,
                edge_var,
                to_var,
                edge_type,
                direction,
                range,
            } => {
                assert_eq!(*from_var, 0); // "a"
                assert_eq!(*edge_var, Some(1)); // "r"
                assert_eq!(*to_var, 2); // "b"
                assert_eq!(edge_type.as_deref(), Some("KNOWS"));
                assert!(matches!(direction, Direction::Out));
                assert!(range.is_none()); // single hop uses None
            },
            other => panic!("expected Expand, got {:?}", other),
        }
        assert_eq!(ep.var_count, 3);
    }

    // 3. WHEN ALL yields TemporalViewport::All.
    #[test]
    fn test_plan_when_all() {
        let query = Query {
            when: Some(WhenClause::All),
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        assert!(matches!(ep.temporal_viewport, TemporalViewport::All));
    }

    // 4. WHEN LAST "30d" yields TemporalViewport::Range covering the last 30 days.
    #[test]
    fn test_plan_when_last_30d() {
        let query = Query {
            when: Some(WhenClause::Last("30d".into())),
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        match ep.temporal_viewport {
            TemporalViewport::Range(start, end) => {
                let thirty_days_nanos: u64 = 30 * 86_400 * 1_000_000_000;
                let diff = end - start;
                // Allow a small margin for clock drift during test execution.
                assert!(
                    diff >= thirty_days_nanos - 1_000_000_000
                        && diff <= thirty_days_nanos + 1_000_000_000,
                    "range diff {} not close to 30d in nanos {}",
                    diff,
                    thirty_days_nanos
                );
            },
            other => panic!("expected Range, got {:?}", other),
        }
    }

    // 5. WHERE clause produces a Filter step.
    #[test]
    fn test_plan_where_clause() {
        let condition = BoolExpr::Comparison(
            Expr::Property("p".into(), "age".into()),
            CompOp::Gt,
            Expr::Literal(Literal::Int(21)),
        );
        let query = Query {
            match_clauses: vec![Pattern {
                elements: vec![PatternElement::Node(NodePattern {
                    var: Some("p".into()),
                    labels: vec![],
                    props: vec![],
                })],
            }],
            where_clause: Some(condition),
            returns: vec![ReturnItem {
                expr: Expr::Var("p".into()),
                alias: None,
                distinct: false,
            }],
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        // ScanNodes + Filter
        assert_eq!(ep.steps.len(), 2);
        assert!(matches!(&ep.steps[1], PlanStep::Filter(_)));
    }

    // 6. RETURN with aggregation detects aggregation and group_by keys.
    #[test]
    fn test_plan_return_aggregation() {
        let query = Query {
            match_clauses: vec![Pattern {
                elements: vec![PatternElement::Node(NodePattern {
                    var: Some("p".into()),
                    labels: vec![],
                    props: vec![],
                })],
            }],
            returns: vec![
                ReturnItem {
                    expr: Expr::Property("p".into(), "city".into()),
                    alias: Some("city".into()),
                    distinct: false,
                },
                ReturnItem {
                    expr: Expr::FuncCall("count".into(), vec![Expr::Star]),
                    alias: Some("cnt".into()),
                    distinct: false,
                },
            ],
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        assert_eq!(ep.aggregations.len(), 1);
        assert_eq!(ep.aggregations[0].function, AggregateFunction::Count);
        assert_eq!(ep.aggregations[0].output_alias, "cnt");
        assert_eq!(ep.group_by_keys, vec!["city"]);
    }

    // 7. parse_duration_nanos for various units.
    #[test]
    fn test_parse_duration_nanos() {
        assert_eq!(parse_duration_nanos("1s").unwrap(), 1_000_000_000);
        assert_eq!(parse_duration_nanos("500ms").unwrap(), 500_000_000);
        assert_eq!(
            parse_duration_nanos("2h").unwrap(),
            2 * 3_600 * 1_000_000_000
        );
        assert_eq!(
            parse_duration_nanos("7d").unwrap(),
            7 * 86_400 * 1_000_000_000
        );
        assert_eq!(
            parse_duration_nanos("1w").unwrap(),
            7 * 86_400 * 1_000_000_000
        );
        assert_eq!(
            parse_duration_nanos("1y").unwrap(),
            365 * 86_400 * 1_000_000_000
        );
        assert_eq!(parse_duration_nanos("100ns").unwrap(), 100);
        assert_eq!(parse_duration_nanos("5m").unwrap(), 5 * 60 * 1_000_000_000);
        assert_eq!(
            parse_duration_nanos("5min").unwrap(),
            5 * 60 * 1_000_000_000
        );

        // Error cases.
        assert!(parse_duration_nanos("").is_err());
        assert!(parse_duration_nanos("abc").is_err());
        assert!(parse_duration_nanos("10x").is_err());
    }

    // 8. parse_timestamp for ISO dates.
    #[test]
    fn test_parse_timestamp_iso() {
        // 2024-01-01T00:00:00 UTC
        let ts = parse_timestamp_str("2024-01-01").unwrap();
        let expected_nanos = days_from_civil(2024, 1, 1) as u64 * 86_400 * 1_000_000_000;
        assert_eq!(ts, expected_nanos);

        // With time component.
        let ts2 = parse_timestamp_str("2024-06-15T10:30:00").unwrap();
        let day_nanos = days_from_civil(2024, 6, 15) as u64 * 86_400 * 1_000_000_000;
        let time_nanos = (10 * 3600 + 30 * 60) as u64 * 1_000_000_000;
        assert_eq!(ts2, day_nanos + time_nanos);

        // Year-month only.
        let ts3 = parse_timestamp_str("2024-06").unwrap();
        let expected3 = days_from_civil(2024, 6, 1) as u64 * 86_400 * 1_000_000_000;
        assert_eq!(ts3, expected3);

        // Epoch milliseconds.
        let ts4 = parse_timestamp_str("1700000000000").unwrap();
        assert_eq!(ts4, 1_700_000_000_000u64 * 1_000_000);
    }

    // Default temporal viewport when no WHEN clause is provided.
    #[test]
    fn test_plan_default_temporal_viewport() {
        let query = empty_query();
        let ep = plan(query).expect("plan should succeed");
        assert!(matches!(ep.temporal_viewport, TemporalViewport::ActiveOnly));
    }

    // Variable-length edge range is preserved.
    #[test]
    fn test_plan_variable_length_edge() {
        let query = Query {
            match_clauses: vec![Pattern {
                elements: vec![
                    PatternElement::Node(NodePattern {
                        var: Some("a".into()),
                        labels: vec![],
                        props: vec![],
                    }),
                    PatternElement::Edge(
                        EdgePattern {
                            var: None,
                            edge_type: Some("FRIEND".into()),
                            range: Some((1, Some(3))),
                            props: vec![],
                        },
                        Direction::Out,
                    ),
                    PatternElement::Node(NodePattern {
                        var: Some("b".into()),
                        labels: vec![],
                        props: vec![],
                    }),
                ],
            }],
            returns: vec![ReturnItem {
                expr: Expr::Var("b".into()),
                alias: None,
                distinct: false,
            }],
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        match &ep.steps[1] {
            PlanStep::Expand { range, .. } => {
                assert_eq!(*range, Some((1, Some(3))));
            },
            other => panic!("expected Expand, got {:?}", other),
        }
    }

    // ORDER BY and LIMIT are propagated.
    #[test]
    fn test_plan_order_and_limit() {
        let query = Query {
            order_by: vec![OrderItem {
                expr: Expr::Property("p".into(), "age".into()),
                descending: true,
            }],
            limit: Some(10),
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        assert_eq!(ep.ordering.len(), 1);
        assert!(ep.ordering[0].descending);
        assert_eq!(ep.limit, Some(10));
    }

    // Empty pattern is rejected.
    #[test]
    fn test_plan_empty_pattern_error() {
        let query = Query {
            match_clauses: vec![Pattern { elements: vec![] }],
            ..empty_query()
        };
        assert!(plan(query).is_err());
    }

    // VarTable assigns monotonic indices and deduplicates.
    #[test]
    fn test_var_table_basics() {
        let mut vt = VarTable::new();
        assert_eq!(vt.get_or_insert("a"), 0);
        assert_eq!(vt.get_or_insert("b"), 1);
        assert_eq!(vt.get_or_insert("a"), 0); // same slot
        assert_eq!(vt.count, 2);
        assert_eq!(vt.lookup("a"), Some(0));
        assert_eq!(vt.lookup("c"), None);
    }

    // Shared variables across MATCH clauses get the same slot.
    #[test]
    fn test_shared_variable_same_slot() {
        let query = Query {
            match_clauses: vec![
                Pattern {
                    elements: vec![
                        PatternElement::Node(NodePattern {
                            var: Some("u".into()),
                            labels: vec![],
                            props: vec![],
                        }),
                        PatternElement::Edge(
                            EdgePattern {
                                var: None,
                                edge_type: None,
                                range: None,
                                props: vec![],
                            },
                            Direction::Out,
                        ),
                        PatternElement::Node(NodePattern {
                            var: Some("a".into()),
                            labels: vec![],
                            props: vec![],
                        }),
                    ],
                },
                Pattern {
                    elements: vec![
                        PatternElement::Node(NodePattern {
                            var: Some("u".into()),
                            labels: vec![],
                            props: vec![],
                        }),
                        PatternElement::Edge(
                            EdgePattern {
                                var: None,
                                edge_type: None,
                                range: None,
                                props: vec![],
                            },
                            Direction::Out,
                        ),
                        PatternElement::Node(NodePattern {
                            var: Some("b".into()),
                            labels: vec![],
                            props: vec![],
                        }),
                    ],
                },
            ],
            returns: vec![ReturnItem {
                expr: Expr::Var("u".into()),
                alias: None,
                distinct: false,
            }],
            ..empty_query()
        };

        let ep = plan(query).expect("plan should succeed");
        // Both ScanNodes for "u" should use the same slot
        match (&ep.steps[0], &ep.steps[2]) {
            (PlanStep::ScanNodes { var: v1, .. }, PlanStep::ScanNodes { var: v2, .. }) => {
                assert_eq!(*v1, *v2, "shared variable 'u' should have the same slot");
            },
            _ => panic!("expected two ScanNodes"),
        }
    }

    // resolve_expr catches unbound variables.
    #[test]
    fn test_resolve_unbound_variable_error() {
        let vt = VarTable::new();
        let expr = Expr::Var("x".into());
        assert!(resolve_expr(&expr, &vt).is_err());
    }
}
