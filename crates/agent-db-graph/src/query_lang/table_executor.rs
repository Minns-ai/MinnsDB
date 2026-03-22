//! Table executor: executes ScanTable and JoinTable plan steps against the TableCatalog.
//!
//! This module handles FROM/JOIN queries. It converts between agent-db-tables
//! CellValue and query_lang Value, resolves table.column property references,
//! and integrates with the existing projection/aggregation/ordering pipeline.

use std::collections::HashMap;

use agent_db_tables::catalog::TableCatalog;
use agent_db_tables::row_codec::DecodedRow;
use agent_db_tables::types::{CellValue, GroupId};

use super::planner::{
    Aggregation, ExecutionPlan, JoinCondition, OrderSpec, PlanStep, RExpr, TemporalViewport,
};
use super::types::{QueryError, QueryOutput, QueryStats, Value};

/// Maximum intermediate rows to prevent OOM on large scans/joins.
const MAX_TABLE_ROWS: usize = 100_000;

/// Execute a table query (FROM/JOIN) against the catalog.
///
/// This handles plans that contain ScanTable/JoinTable steps. It produces
/// result rows as Vec<Value>, then applies projection, aggregation, ordering,
/// and limit — same as the graph executor's output pipeline.
pub fn execute_table_query(
    catalog: &TableCatalog,
    plan: &ExecutionPlan,
    group_id: GroupId,
) -> Result<QueryOutput, QueryError> {
    let start = std::time::Instant::now();

    // Phase 1: Execute scan/join steps to produce raw table rows.
    // Each row is a HashMap<String, Value> keyed by "table.column".
    let mut rows: Vec<HashMap<String, Value>> = Vec::new();
    let mut active_tables: Vec<String> = Vec::new();

    // Check for predicate pushdown: if ScanTable is followed by Filter with PK equality,
    // use direct lookup instead of full scan.
    let pk_lookup = try_extract_pk_lookup(&plan.steps, catalog, group_id);

    for step in plan.steps.iter() {
        match step {
            PlanStep::ScanTable { table_name } => {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;

                // If we have a PK lookup optimization, use it
                if let Some((ref opt_table, row_id)) = pk_lookup {
                    if opt_table == table_name {
                        if let Some(decoded) = table.get_active(group_id, row_id) {
                            rows = vec![decoded_row_to_map(
                                table_name,
                                &table.schema.columns,
                                &decoded,
                            )];
                        }
                        active_tables.push(table_name.clone());
                        continue;
                    }
                }

                let decoded = match &plan.temporal_viewport {
                    TemporalViewport::ActiveOnly => table.scan_active(group_id),
                    TemporalViewport::All => table.scan_all(group_id),
                    TemporalViewport::PointInTime(ts) => table.scan_as_of(group_id, *ts),
                    TemporalViewport::Range(start, end) => table.scan_range(group_id, *start, *end),
                };

                rows = decoded
                    .into_iter()
                    .take(MAX_TABLE_ROWS)
                    .map(|r| decoded_row_to_map(table_name, &table.schema.columns, &r))
                    .collect();
                if rows.len() >= MAX_TABLE_ROWS {
                    return Err(QueryError::ExecutionError(format!(
                        "table scan exceeded {} row limit — add a WHERE filter or LIMIT",
                        MAX_TABLE_ROWS
                    )));
                }
                active_tables.push(table_name.clone());
            },
            PlanStep::JoinTable { table_name, on } => {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;

                let mut new_rows = Vec::new();
                for left_row in &rows {
                    let matches = execute_join_lookup(
                        catalog,
                        table_name,
                        on,
                        left_row,
                        group_id,
                        &plan.temporal_viewport,
                    )?;
                    for right_decoded in matches {
                        let mut combined = left_row.clone();
                        let right_map =
                            decoded_row_to_map(table_name, &table.schema.columns, &right_decoded);
                        combined.extend(right_map);
                        new_rows.push(combined);
                    }
                }
                rows = new_rows;
                if rows.len() > MAX_TABLE_ROWS {
                    return Err(QueryError::ExecutionError(format!(
                        "join produced {} rows, exceeding {} limit — add a WHERE filter or LIMIT",
                        rows.len(),
                        MAX_TABLE_ROWS
                    )));
                }
                active_tables.push(table_name.clone());
            },
            PlanStep::Filter(expr) => {
                rows.retain(|row| eval_filter(expr, row));
            },
            // Graph steps should not reach the table executor.
            // Mixed queries (MATCH...JOIN) go through execute_mixed_query().
            PlanStep::ScanNodes { .. } | PlanStep::Expand { .. } => {
                return Err(QueryError::ExecutionError(
                    "Graph steps in table executor. Mixed MATCH+JOIN queries \
                     must be routed through execute_mixed_query()."
                        .into(),
                ));
            },
        }
    }

    // Phase 2: Project rows into output columns.
    let columns: Vec<String> = plan.projections.iter().map(|p| p.alias.clone()).collect();

    let mut result_rows: Vec<Vec<Value>> = Vec::new();
    for row in &rows {
        let mut out = Vec::with_capacity(plan.projections.len());
        for proj in &plan.projections {
            out.push(eval_projection(&proj.expr, row));
        }
        result_rows.push(out);
    }

    // Phase 3: Aggregation (if any).
    if !plan.aggregations.is_empty() {
        result_rows = apply_aggregations(&result_rows, &plan.aggregations, &columns);
    }

    // Phase 4: Ordering.
    if !plan.ordering.is_empty() {
        apply_ordering(&mut result_rows, &plan.ordering, &columns);
    }

    // Phase 5: Limit.
    if let Some(limit) = plan.limit {
        result_rows.truncate(limit as usize);
    }

    let elapsed = start.elapsed();

    Ok(QueryOutput {
        columns,
        rows: result_rows,
        stats: QueryStats {
            nodes_scanned: 0,
            edges_traversed: 0,
            execution_time_ms: elapsed.as_millis() as u64,
        },
    })
}

/// Convert a DecodedRow into a flat HashMap keyed by "table.column".
fn decoded_row_to_map(
    table_name: &str,
    schema_cols: &[agent_db_tables::schema::ColumnDef],
    row: &DecodedRow,
) -> HashMap<String, Value> {
    let mut map = HashMap::new();

    // System columns
    map.insert(
        format!("{}.row_id", table_name),
        Value::Int(row.row_id as i64),
    );
    map.insert(
        format!("{}.version_id", table_name),
        Value::Int(row.version_id as i64),
    );
    map.insert(
        format!("{}.valid_from", table_name),
        Value::Int(row.valid_from as i64),
    );
    map.insert(
        format!("{}.valid_until", table_name),
        match row.valid_until {
            Some(t) => Value::Int(t as i64),
            None => Value::Null,
        },
    );

    // User columns — only insert bare name if not already present (first table wins for unqualified)
    for (i, col) in schema_cols.iter().enumerate() {
        let key = format!("{}.{}", table_name, col.name);
        let val = if i < row.values.len() {
            cell_to_value(&row.values[i])
        } else {
            Value::Null // schema has more columns than row data (possible after schema migration)
        };
        // Bare column name: only if not already claimed by a previous table in a join
        map.entry(col.name.clone()).or_insert_with(|| val.clone());
        // Qualified name always inserted
        map.insert(key, val);
    }

    map
}

/// Convert CellValue to query Value.
fn cell_to_value(cell: &CellValue) -> Value {
    match cell {
        CellValue::String(s) => Value::String(s.clone()),
        CellValue::Int64(i) => Value::Int(*i),
        CellValue::Float64(f) => Value::Float(*f),
        CellValue::Bool(b) => Value::Bool(*b),
        CellValue::Timestamp(t) => Value::Int(*t as i64),
        CellValue::Json(v) => json_to_value(v),
        CellValue::NodeRef(n) => Value::Int(*n as i64),
        CellValue::Null => Value::Null,
    }
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        },
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => Value::Map(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
    }
}

/// Execute a join lookup: for one left row, find matching rows in the right table.
fn execute_join_lookup(
    catalog: &TableCatalog,
    right_table_name: &str,
    on: &JoinCondition,
    left_row: &HashMap<String, Value>,
    group_id: GroupId,
    viewport: &TemporalViewport,
) -> Result<Vec<DecodedRow>, QueryError> {
    let right_table = catalog.get_table(right_table_name).ok_or_else(|| {
        QueryError::ExecutionError(format!("table not found: {}", right_table_name))
    })?;

    match on {
        JoinCondition::TableToTable {
            left_table,
            left_col,
            right_col,
        } => {
            // Get the join key from the left row
            let key_name = format!("{}.{}", left_table, left_col);
            let join_key = left_row.get(&key_name).cloned().unwrap_or(Value::Null);

            // Scan right table and filter by join column
            let right_rows = match viewport {
                TemporalViewport::ActiveOnly => right_table.scan_active(group_id),
                TemporalViewport::All => right_table.scan_all(group_id),
                TemporalViewport::PointInTime(ts) => right_table.scan_as_of(group_id, *ts),
                TemporalViewport::Range(s, e) => right_table.scan_range(group_id, *s, *e),
            };

            // Find the column index for the right join column
            let right_col_idx = right_table.schema.column_index(right_col).ok_or_else(|| {
                QueryError::ExecutionError(format!(
                    "column '{}' not found in table '{}'",
                    right_col, right_table_name
                ))
            })?;

            let matches: Vec<DecodedRow> = right_rows
                .into_iter()
                .filter(|r| {
                    let right_val = cell_to_value(&r.values[right_col_idx]);
                    values_equal(&join_key, &right_val)
                })
                .collect();

            Ok(matches)
        },
        JoinCondition::GraphToTable {
            graph_var: _,
            table_col: _,
        } => {
            // Graph-to-table joins in pure table queries are not valid — they
            // require graph binding rows. Mixed queries (MATCH...JOIN) go through
            // execute_mixed_query() which handles this correctly.
            Err(QueryError::ExecutionError(
                "Graph-to-table JOIN requires a MATCH clause. \
                 This query was routed to the table executor without graph bindings. \
                 Use execute_mixed_query() for MATCH...JOIN queries."
                    .into(),
            ))
        },
    }
}

/// Evaluate a resolved boolean filter expression against a table row map.
fn eval_filter(expr: &super::planner::RBoolExpr, row: &HashMap<String, Value>) -> bool {
    use super::planner::RBoolExpr;
    match expr {
        RBoolExpr::Comparison(lhs, op, rhs) => {
            let l = eval_rexpr(lhs, row);
            let r = eval_rexpr(rhs, row);
            match op {
                super::ast::CompOp::Eq => values_equal(&l, &r),
                super::ast::CompOp::Neq => !values_equal(&l, &r),
                super::ast::CompOp::Lt => l.partial_cmp(&r) == Some(std::cmp::Ordering::Less),
                super::ast::CompOp::Gt => l.partial_cmp(&r) == Some(std::cmp::Ordering::Greater),
                super::ast::CompOp::Lte => l
                    .partial_cmp(&r)
                    .is_some_and(|o| o != std::cmp::Ordering::Greater),
                super::ast::CompOp::Gte => l
                    .partial_cmp(&r)
                    .is_some_and(|o| o != std::cmp::Ordering::Less),
                super::ast::CompOp::Contains => {
                    if let (Value::String(haystack), Value::String(needle)) = (&l, &r) {
                        haystack.contains(needle.as_str())
                    } else {
                        false
                    }
                },
                super::ast::CompOp::StartsWith => {
                    if let (Value::String(haystack), Value::String(prefix)) = (&l, &r) {
                        haystack.starts_with(prefix.as_str())
                    } else {
                        false
                    }
                },
            }
        },
        RBoolExpr::IsNull(e) => matches!(eval_rexpr(e, row), Value::Null),
        RBoolExpr::IsNotNull(e) => !matches!(eval_rexpr(e, row), Value::Null),
        RBoolExpr::And(a, b) => eval_filter(a, row) && eval_filter(b, row),
        RBoolExpr::Or(a, b) => eval_filter(a, row) || eval_filter(b, row),
        RBoolExpr::Not(inner) => !eval_filter(inner, row),
        RBoolExpr::Paren(inner) => eval_filter(inner, row),
        RBoolExpr::FuncPredicate(_, _) => false, // Function predicates not evaluated for tables — filter out
    }
}

/// Evaluate a resolved expression against a table row map.
fn eval_rexpr(expr: &RExpr, row: &HashMap<String, Value>) -> Value {
    match expr {
        RExpr::Property(_slot, prop) => {
            // The slot was assigned from a table name. Look up "table.prop" in the row map.
            // We try the qualified name first, then fall back to bare column name.
            // Since we don't have the table name from the slot here, try bare first.
            row.get(prop).cloned().unwrap_or_else(|| {
                // Try all qualified variants
                for (k, v) in row {
                    if k.ends_with(&format!(".{}", prop)) {
                        return v.clone();
                    }
                }
                Value::Null
            })
        },
        RExpr::Literal(lit) => literal_to_value(lit),
        RExpr::Var(_slot) => Value::Null, // Bare variable not meaningful for tables
        RExpr::Star => Value::Null,
        RExpr::FuncCall(name, args) => {
            let arg_vals: Vec<Value> = args.iter().map(|a| eval_rexpr(a, row)).collect();
            eval_builtin_func(name, &arg_vals)
        },
    }
}

fn literal_to_value(lit: &super::ast::Literal) -> Value {
    match lit {
        super::ast::Literal::String(s) => Value::String(s.clone()),
        super::ast::Literal::Int(i) => Value::Int(*i),
        super::ast::Literal::Float(f) => Value::Float(*f),
        super::ast::Literal::Bool(b) => Value::Bool(*b),
        super::ast::Literal::Null => Value::Null,
    }
}

fn eval_builtin_func(name: &str, args: &[Value]) -> Value {
    match name.to_lowercase().as_str() {
        "count" => Value::Int(1), // Per-row count; actual aggregation happens later
        "coalesce" => args
            .iter()
            .find(|v| !v.is_null())
            .cloned()
            .unwrap_or(Value::Null),
        "toupper" | "to_upper" => match args.first() {
            Some(Value::String(s)) => Value::String(s.to_uppercase()),
            _ => Value::Null,
        },
        "tolower" | "to_lower" => match args.first() {
            Some(Value::String(s)) => Value::String(s.to_lowercase()),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

/// Evaluate a projection expression for output.
fn eval_projection(expr: &RExpr, row: &HashMap<String, Value>) -> Value {
    eval_rexpr(expr, row)
}

/// Compare two Values for equality (used in join matching).
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => {
            (*x as f64 - y).abs() < f64::EPSILON
        },
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Null, Value::Null) => false, // NULL != NULL in SQL semantics
        _ => false,
    }
}

/// Apply aggregations to result rows.
fn apply_aggregations(
    rows: &[Vec<Value>],
    aggregations: &[Aggregation],
    _columns: &[String],
) -> Vec<Vec<Value>> {
    if rows.is_empty() {
        // Return one row of zeros/nulls for aggregation on empty set
        return vec![aggregations
            .iter()
            .map(|agg| match agg.function {
                super::planner::AggregateFunction::Count => Value::Int(0),
                _ => Value::Null,
            })
            .collect()];
    }

    // Single group (no GROUP BY support yet — that comes with full SQL)
    let mut result = Vec::new();
    for (i, agg) in aggregations.iter().enumerate() {
        let vals: Vec<&Value> = rows.iter().map(|r| &r[i]).collect();
        let aggregated = match agg.function {
            super::planner::AggregateFunction::Count => {
                Value::Int(vals.iter().filter(|v| !v.is_null()).count() as i64)
            },
            super::planner::AggregateFunction::Sum => {
                let sum: f64 = vals
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .sum();
                Value::Float(sum)
            },
            super::planner::AggregateFunction::Avg => {
                let nums: Vec<f64> = vals
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                if nums.is_empty() {
                    Value::Null
                } else {
                    Value::Float(nums.iter().sum::<f64>() / nums.len() as f64)
                }
            },
            super::planner::AggregateFunction::Min => vals
                .iter()
                .filter(|v| !v.is_null())
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .cloned()
                .unwrap_or(Value::Null),
            super::planner::AggregateFunction::Max => vals
                .iter()
                .filter(|v| !v.is_null())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .cloned()
                .unwrap_or(Value::Null),
            super::planner::AggregateFunction::Collect => {
                Value::List(vals.into_iter().cloned().collect())
            },
        };
        result.push(aggregated);
    }
    vec![result]
}

/// Apply ordering to result rows.
fn apply_ordering(rows: &mut [Vec<Value>], ordering: &[OrderSpec], columns: &[String]) {
    rows.sort_by(|a, b| {
        for spec in ordering {
            let col_idx = columns
                .iter()
                .position(|c| c == &spec.column_alias)
                .unwrap_or(0);
            let cmp = a[col_idx]
                .partial_cmp(&b[col_idx])
                .unwrap_or(std::cmp::Ordering::Equal);
            let cmp = if spec.descending { cmp.reverse() } else { cmp };
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
}

/// Try to extract a PK lookup from the plan: ScanTable followed by Filter with `id = N`.
/// Returns (table_name, row_id) if applicable.
fn try_extract_pk_lookup(
    steps: &[PlanStep],
    _catalog: &TableCatalog,
    _group_id: GroupId,
) -> Option<(String, u64)> {
    // Pattern: ScanTable { name } followed by Filter(id = N)
    if steps.len() < 2 {
        return None;
    }
    let table_name = match &steps[0] {
        PlanStep::ScanTable { table_name } => table_name.clone(),
        _ => return None,
    };
    match &steps[1] {
        PlanStep::Filter(filter) => extract_id_eq_from_filter(filter),
        _ => None,
    }
    .map(|row_id| (table_name, row_id))
}

/// Extract `id = N` or `row_id = N` from a resolved filter expression.
fn extract_id_eq_from_filter(filter: &super::planner::RBoolExpr) -> Option<u64> {
    use super::planner::RBoolExpr;
    match filter {
        RBoolExpr::Comparison(lhs, super::ast::CompOp::Eq, rhs) => {
            // Check: property.name == "id" and literal is int
            match (lhs, rhs) {
                (RExpr::Property(_, prop), RExpr::Literal(super::ast::Literal::Int(id)))
                | (RExpr::Literal(super::ast::Literal::Int(id)), RExpr::Property(_, prop))
                    if (prop == "id" || prop == "row_id") && *id >= 0 =>
                {
                    Some(*id as u64)
                },
                _ => None,
            }
        },
        // Also check AND: left might be id=N and right is some other filter
        RBoolExpr::And(a, b) => {
            extract_id_eq_from_filter(a).or_else(|| extract_id_eq_from_filter(b))
        },
        _ => None,
    }
}

/// Check if a plan contains table steps (ScanTable or JoinTable).
pub fn is_table_query(plan: &ExecutionPlan) -> bool {
    plan.steps
        .iter()
        .any(|s| matches!(s, PlanStep::ScanTable { .. } | PlanStep::JoinTable { .. }))
}

/// Check if a plan has both graph steps (ScanNodes/Expand) and table steps (JoinTable).
/// This requires two-phase execution: graph first, then table join.
pub fn is_mixed_query(plan: &ExecutionPlan) -> bool {
    let has_graph = plan
        .steps
        .iter()
        .any(|s| matches!(s, PlanStep::ScanNodes { .. } | PlanStep::Expand { .. }));
    let has_table_join = plan
        .steps
        .iter()
        .any(|s| matches!(s, PlanStep::JoinTable { .. }));
    has_graph && has_table_join
}

/// Execute a mixed MATCH...JOIN query.
///
/// Phase 1: Run graph steps (ScanNodes, Expand, Filter) through the graph executor
///          to produce binding rows with node IDs.
/// Phase 2: For each graph result row, execute the JoinTable step by looking up
///          matching table rows via the NodeRef index.
/// Phase 3: Project, aggregate, order, limit as normal.
pub fn execute_mixed_query(
    catalog: &TableCatalog,
    plan: &ExecutionPlan,
    group_id: GroupId,
    graph: &crate::structures::Graph,
    ontology: &crate::ontology::OntologyRegistry,
) -> Result<QueryOutput, QueryError> {
    let start = std::time::Instant::now();

    // Phase 1: Build a graph-only plan from the graph steps
    let graph_steps: Vec<PlanStep> = plan
        .steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                PlanStep::ScanNodes { .. } | PlanStep::Expand { .. } | PlanStep::Filter(_)
            )
        })
        .cloned()
        .collect();

    // Find the JoinTable steps and their conditions
    let join_steps: Vec<&PlanStep> = plan
        .steps
        .iter()
        .filter(|s| matches!(s, PlanStep::JoinTable { .. }))
        .collect();

    // Execute graph steps to get binding rows with node IDs
    let graph_plan = ExecutionPlan {
        steps: graph_steps,
        projections: vec![], // We'll project later
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: plan.limit, // Push limit to graph phase
        temporal_viewport: plan.temporal_viewport.clone(),
        transaction_cutoff: plan.transaction_cutoff,
        var_count: plan.var_count,
    };

    // Execute graph plan — returns QueryOutput with node IDs
    // We need the raw binding rows, not projected output.
    // Use execute_with_bindings to get the slot bindings.
    let (graph_output, graph_bindings) =
        super::executor::Executor::execute_with_bindings(graph, ontology, graph_plan)
            .map_err(|e| QueryError::ExecutionError(format!("graph phase: {}", e)))?;

    if graph_bindings.is_empty() {
        return Ok(QueryOutput {
            columns: plan.projections.iter().map(|p| p.alias.clone()).collect(),
            rows: vec![],
            stats: QueryStats {
                nodes_scanned: graph_output.stats.nodes_scanned,
                edges_traversed: graph_output.stats.edges_traversed,
                execution_time_ms: start.elapsed().as_millis() as u64,
            },
        });
    }

    // Phase 2: For each graph binding, execute table joins
    let mut result_rows: Vec<HashMap<String, Value>> = Vec::new();

    for binding in &graph_bindings {
        // Build a row map from the graph binding
        let mut row: HashMap<String, Value> = HashMap::new();

        // Extract node properties from each bound entity
        for (slot_idx, entity) in binding.iter() {
            match entity {
                crate::subscription::incremental::BoundEntityId::Node(node_id) => {
                    // Find the variable name for this slot from the plan
                    // We store it as the graph variable with its properties
                    if let Some(node) = graph.get_node(*node_id) {
                        // Use a helper to get common properties
                        let var_name = format!("_slot_{}", slot_idx);
                        row.insert(format!("{}.id", var_name), Value::Int(*node_id as i64));
                        row.insert(
                            format!("{}.type", var_name),
                            Value::String(format!("{:?}", node.node_type)),
                        );
                        // Store the raw node ID keyed by slot for join lookup
                        row.insert(
                            format!("__slot_{}_node_id", slot_idx),
                            Value::Int(*node_id as i64),
                        );
                    }
                },
                crate::subscription::incremental::BoundEntityId::Edge(edge_id) => {
                    row.insert(
                        format!("__slot_{}_edge_id", slot_idx),
                        Value::Int(*edge_id as i64),
                    );
                },
            }
        }

        // Execute each JoinTable step
        for join_step in &join_steps {
            if let PlanStep::JoinTable { table_name, on } = join_step {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;

                match on {
                    JoinCondition::GraphToTable {
                        graph_var,
                        table_col: _,
                    } => {
                        // Extract the node ID from the specific graph variable slot
                        let node_id_key = format!("__slot_{}_node_id", graph_var);
                        let node_id = match row.get(&node_id_key) {
                            Some(Value::Int(id)) => *id as u64,
                            _ => continue, // No node ID for this slot, skip
                        };

                        // Look up table rows via NodeRef index
                        let matching_rows = table.rows_by_node(group_id, node_id);

                        for decoded in matching_rows {
                            let mut joined_row = row.clone();
                            let table_map =
                                decoded_row_to_map(table_name, &table.schema.columns, &decoded);
                            joined_row.extend(table_map);
                            result_rows.push(joined_row);
                        }
                    },
                    JoinCondition::TableToTable { .. } => {
                        // Table-to-table join in a mixed query (unusual but possible)
                        let matches = execute_join_lookup(
                            catalog,
                            table_name,
                            on,
                            &row,
                            group_id,
                            &plan.temporal_viewport,
                        )?;
                        let table = catalog.get_table(table_name).unwrap();
                        for decoded in matches {
                            let mut joined_row = row.clone();
                            let table_map =
                                decoded_row_to_map(table_name, &table.schema.columns, &decoded);
                            joined_row.extend(table_map);
                            result_rows.push(joined_row);
                        }
                    },
                }
            }
        }
    }

    // Cap result size
    if result_rows.len() > MAX_TABLE_ROWS {
        return Err(QueryError::ExecutionError(format!(
            "mixed query produced {} rows, exceeding {} limit",
            result_rows.len(),
            MAX_TABLE_ROWS
        )));
    }

    // Phase 3: Project
    let columns: Vec<String> = plan.projections.iter().map(|p| p.alias.clone()).collect();

    let mut projected: Vec<Vec<Value>> = Vec::new();
    for row in &result_rows {
        let mut out = Vec::with_capacity(plan.projections.len());
        for proj in &plan.projections {
            out.push(eval_projection(&proj.expr, row));
        }
        projected.push(out);
    }

    // Aggregation
    if !plan.aggregations.is_empty() {
        projected = apply_aggregations(&projected, &plan.aggregations, &columns);
    }

    // Ordering
    if !plan.ordering.is_empty() {
        apply_ordering(&mut projected, &plan.ordering, &columns);
    }

    // Limit
    if let Some(limit) = plan.limit {
        projected.truncate(limit as usize);
    }

    let elapsed = start.elapsed();

    Ok(QueryOutput {
        columns,
        rows: projected,
        stats: QueryStats {
            nodes_scanned: graph_output.stats.nodes_scanned,
            edges_traversed: graph_output.stats.edges_traversed,
            execution_time_ms: elapsed.as_millis() as u64,
        },
    })
}
