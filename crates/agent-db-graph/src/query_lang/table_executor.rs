//! Table executor: executes ScanTable and JoinTable plan steps against the TableCatalog.
//!
//! This module handles FROM/JOIN queries. It converts between agent-db-tables
//! CellValue and query_lang Value, resolves table.column property references,
//! and integrates with the existing projection/aggregation/ordering pipeline.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use agent_db_tables::catalog::TableCatalog;
use agent_db_tables::row_codec::DecodedRow;
use agent_db_tables::types::{CellValue, GroupId};

use super::ast::Literal;
use super::planner::{
    AggregateFunction, Aggregation, ExecutionPlan, JoinCondition, JoinType, OrderSpec, PlanStep,
    Projection, RExpr, TemporalViewport,
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

    // COUNT(*) optimization: if the plan is a single ScanTable with no filters,
    // no joins, and the only aggregation is COUNT, read the row count directly.
    if let Some(count_result) = try_count_star_optimization(plan, catalog, group_id) {
        return Ok(count_result);
    }

    // Phase 1: Execute scan/join steps to produce raw table rows.
    // Each row is a HashMap<String, Value> keyed by "table.column".
    let mut rows: Vec<HashMap<String, Value>> = Vec::new();
    let mut active_tables: Vec<String> = Vec::new();

    // Check for predicate pushdown: if ScanTable is followed by Filter with PK equality,
    // use direct lookup instead of full scan.
    let pk_lookup = try_extract_pk_lookup(&plan.steps, catalog, group_id);

    for step in plan.steps.iter() {
        match step {
            PlanStep::ScanTable {
                table_name,
                alias,
                scan_limit,
            } => {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;
                let map_name = alias.as_deref().unwrap_or(table_name.as_str());

                // PK lookup optimization: only valid for ActiveOnly viewport since
                // get_active() returns only the current version. For WHEN ALL / point-in-time /
                // range queries we must fall through to the full scan path.
                if matches!(plan.temporal_viewport, TemporalViewport::ActiveOnly) {
                    if let Some((ref opt_table, row_id)) = pk_lookup {
                        if opt_table == table_name {
                            if let Some(decoded) = table.get_active(group_id, row_id) {
                                rows = vec![decoded_row_to_map(
                                    map_name,
                                    &table.schema.columns,
                                    &decoded,
                                )];
                            }
                            active_tables.push(map_name.to_string());
                            continue;
                        }
                    }
                }

                let decoded = match &plan.temporal_viewport {
                    TemporalViewport::ActiveOnly => table.scan_active(group_id),
                    TemporalViewport::All => table.scan_all(group_id),
                    TemporalViewport::PointInTime(ts) => table.scan_as_of(group_id, *ts),
                    TemporalViewport::Range(start, end) => table.scan_range(group_id, *start, *end),
                };

                let limit = scan_limit.unwrap_or(MAX_TABLE_ROWS);
                rows = decoded
                    .into_iter()
                    .take(limit)
                    .map(|r| decoded_row_to_map(map_name, &table.schema.columns, &r))
                    .collect();
                if rows.len() >= MAX_TABLE_ROWS {
                    return Err(QueryError::ExecutionError(format!(
                        "table scan exceeded {} row limit — add a WHERE filter or LIMIT",
                        MAX_TABLE_ROWS
                    )));
                }
                active_tables.push(map_name.to_string());
            },
            PlanStep::JoinTable {
                table_name,
                alias,
                join_type,
                on,
            } => {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;
                let map_name = alias.as_deref().unwrap_or(table_name.as_str());

                // For table-to-table joins, build hash table once and probe per left row.
                // This is O(N + M) instead of O(N * M) for nested-loop.
                let hash_table_opt = match on {
                    JoinCondition::TableToTable { right_col, .. } => {
                        let (_, ht) = build_join_hash_table(
                            table,
                            right_col,
                            group_id,
                            &plan.temporal_viewport,
                        )?;
                        Some(ht)
                    },
                    _ => None,
                };

                let mut new_rows = Vec::new();
                for left_row in &rows {
                    let matches: Vec<DecodedRow> = if let Some(ref ht) = hash_table_opt {
                        // Hash join: probe pre-built hash table
                        if let JoinCondition::TableToTable {
                            left_table,
                            left_col,
                            ..
                        } = on
                        {
                            let key_name = format!("{}.{}", left_table, left_col);
                            let join_key = left_row.get(&key_name).cloned().unwrap_or(Value::Null);
                            probe_join_hash_table(ht, &join_key).to_vec()
                        } else {
                            vec![]
                        }
                    } else {
                        // Fallback (graph-to-table)
                        execute_join_lookup(
                            catalog,
                            table_name,
                            on,
                            left_row,
                            group_id,
                            &plan.temporal_viewport,
                        )?
                    };

                    if matches.is_empty() && *join_type == JoinType::Left {
                        // LEFT JOIN: emit left row with NULLs for right table columns
                        let mut combined = left_row.clone();
                        for col in &table.schema.columns {
                            combined.insert(format!("{}.{}", map_name, col.name), Value::Null);
                            combined.entry(col.name.clone()).or_insert(Value::Null);
                        }
                        combined.insert(format!("{}.row_id", map_name), Value::Null);
                        combined.insert(format!("{}.version_id", map_name), Value::Null);
                        combined.insert(format!("{}.valid_from", map_name), Value::Null);
                        combined.insert(format!("{}.valid_until", map_name), Value::Null);
                        new_rows.push(combined);
                    } else {
                        for right_decoded in matches {
                            let mut combined = left_row.clone();
                            let right_map =
                                decoded_row_to_map(map_name, &table.schema.columns, &right_decoded);
                            combined.extend(right_map);
                            new_rows.push(combined);
                        }
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
                active_tables.push(map_name.to_string());
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
            PlanStep::IndexScan {
                table_name,
                alias,
                index_columns,
                key_values,
                is_point,
                scan_limit,
            } => {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;
                let map_name = alias.as_deref().unwrap_or(table_name.as_str());

                if *is_point {
                    // Point lookup: convert key_values to CellValue, probe index
                    let key_cells: Vec<CellValue> =
                        key_values.iter().map(literal_to_cell_value).collect();
                    if let Some(decoded) = table.get_by_index(group_id, index_columns, &key_cells) {
                        rows = vec![decoded_row_to_map(
                            map_name,
                            &table.schema.columns,
                            &decoded,
                        )];
                    } else {
                        rows = vec![];
                    }
                } else {
                    // Non-point (range): fall back to filtered scan
                    let decoded = match &plan.temporal_viewport {
                        TemporalViewport::ActiveOnly => table.scan_active(group_id),
                        TemporalViewport::All => table.scan_all(group_id),
                        TemporalViewport::PointInTime(ts) => table.scan_as_of(group_id, *ts),
                        TemporalViewport::Range(start, end) => {
                            table.scan_range(group_id, *start, *end)
                        },
                    };
                    let limit = scan_limit.unwrap_or(MAX_TABLE_ROWS);
                    rows = decoded
                        .into_iter()
                        .take(limit)
                        .map(|r| decoded_row_to_map(map_name, &table.schema.columns, &r))
                        .collect();
                }
                active_tables.push(map_name.to_string());
            },
            PlanStep::IndexJoin {
                table_name,
                alias,
                join_type,
                index_column,
                left_table,
                left_col,
            } => {
                let table = catalog.get_table(table_name).ok_or_else(|| {
                    QueryError::ExecutionError(format!("table not found: {}", table_name))
                })?;
                let map_name = alias.as_deref().unwrap_or(table_name.as_str());
                let index_cols = vec![index_column.clone()];

                let mut new_rows = Vec::new();
                for left_row in &rows {
                    let key_name = format!("{}.{}", left_table, left_col);
                    let join_key = left_row.get(&key_name).cloned().unwrap_or(Value::Null);

                    let matches: Vec<DecodedRow> =
                        if let Some(cell_key) = value_to_cell_value(&join_key) {
                            // Probe the index for this key
                            match table.get_by_index(group_id, &index_cols, &[cell_key]) {
                                Some(decoded) => vec![decoded],
                                None => vec![],
                            }
                        } else {
                            vec![]
                        };

                    if matches.is_empty() && *join_type == JoinType::Left {
                        let mut combined = left_row.clone();
                        for col in &table.schema.columns {
                            combined.insert(format!("{}.{}", map_name, col.name), Value::Null);
                            combined.entry(col.name.clone()).or_insert(Value::Null);
                        }
                        combined.insert(format!("{}.row_id", map_name), Value::Null);
                        combined.insert(format!("{}.version_id", map_name), Value::Null);
                        combined.insert(format!("{}.valid_from", map_name), Value::Null);
                        combined.insert(format!("{}.valid_until", map_name), Value::Null);
                        new_rows.push(combined);
                    } else {
                        for right_decoded in matches {
                            let mut combined = left_row.clone();
                            let right_map =
                                decoded_row_to_map(map_name, &table.schema.columns, &right_decoded);
                            combined.extend(right_map);
                            new_rows.push(combined);
                        }
                    }
                }
                rows = new_rows;
                if rows.len() > MAX_TABLE_ROWS {
                    return Err(QueryError::ExecutionError(format!(
                        "index join produced {} rows, exceeding {} limit",
                        rows.len(),
                        MAX_TABLE_ROWS
                    )));
                }
                active_tables.push(map_name.to_string());
            },
        }
    }

    // Phase 2: Project rows into output columns.
    // For aggregate columns, evaluate the aggregate's input expression (e.g. the
    // `price` in `sum(price)`) rather than the full FuncCall.  Phase 3 applies
    // the actual aggregate function over the collected per-row values.
    let columns: Vec<String> = plan.projections.iter().map(|p| p.alias.clone()).collect();

    let agg_input: HashMap<&str, &Aggregation> = plan
        .aggregations
        .iter()
        .map(|a| (a.output_alias.as_str(), a))
        .collect();

    let mut result_rows: Vec<Vec<Value>> = Vec::new();
    for row in &rows {
        result_rows.push(project_row_for_aggregation(
            &plan.projections,
            &agg_input,
            row,
        ));
    }

    // Phase 3: Aggregation (if any).
    if !plan.aggregations.is_empty() {
        result_rows = apply_aggregations(
            &result_rows,
            &plan.aggregations,
            &columns,
            &plan.group_by_keys,
        );
    }

    // Phase 3b: HAVING filter (post-aggregation).
    if let Some(ref having_expr) = plan.having {
        result_rows.retain(|row| {
            let row_map: HashMap<String, Value> = columns
                .iter()
                .zip(row.iter())
                .map(|(c, v)| (c.clone(), v.clone()))
                .collect();
            eval_filter(having_expr, &row_map)
        });
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

/// Convert a query Value to CellValue (for index probing).
fn value_to_cell_value(v: &Value) -> Option<CellValue> {
    match v {
        Value::Int(i) => Some(CellValue::Int64(*i)),
        Value::Float(f) => Some(CellValue::Float64(*f)),
        Value::String(s) => Some(CellValue::String(s.clone())),
        Value::Bool(b) => Some(CellValue::Bool(*b)),
        Value::Null => None,
        _ => None,
    }
}

/// Convert an AST Literal to CellValue (for IndexScan key values).
fn literal_to_cell_value(lit: &Literal) -> CellValue {
    match lit {
        Literal::String(s) => CellValue::String(s.clone()),
        Literal::Int(i) => CellValue::Int64(*i),
        Literal::Float(f) => CellValue::Float64(*f),
        Literal::Bool(b) => CellValue::Bool(*b),
        Literal::Null => CellValue::Null,
        Literal::NodeRef(id) => CellValue::NodeRef(*id),
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

/// Stable, deterministic string key for a Value used in hash joins and grouping.
/// Unlike Debug format, this is canonical: Int and Float are never conflated,
/// and NULLs are excluded from join hash tables entirely.
fn value_hash_key(v: &Value) -> String {
    match v {
        Value::Null => String::new(), // should never be used as a key
        Value::Int(i) => format!("I:{}", i),
        Value::Float(f) => format!("F:{}", f.to_bits()), // exact bit representation
        Value::String(s) => format!("S:{}", s),
        Value::Bool(b) => format!("B:{}", b),
        Value::List(l) => {
            let parts: Vec<String> = l.iter().map(value_hash_key).collect();
            format!("L:[{}]", parts.join(","))
        },
        Value::Map(m) => {
            let mut parts: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{}={}", k, value_hash_key(v)))
                .collect();
            parts.sort(); // deterministic order
            format!("M:{{{}}}", parts.join(","))
        },
    }
}

/// Build a hash table from the right side of a join, keyed by join column value.
/// NULL keys are excluded — SQL semantics require NULL != NULL in joins.
fn build_join_hash_table(
    right_table: &agent_db_tables::table::Table,
    right_col: &str,
    group_id: GroupId,
    viewport: &TemporalViewport,
) -> Result<(usize, HashMap<String, Vec<DecodedRow>>), QueryError> {
    let right_col_idx = right_table.schema.column_index(right_col).ok_or_else(|| {
        QueryError::ExecutionError(format!(
            "column '{}' not found in table '{}'",
            right_col, right_table.schema.name
        ))
    })?;

    let right_rows = match viewport {
        TemporalViewport::ActiveOnly => right_table.scan_active(group_id),
        TemporalViewport::All => right_table.scan_all(group_id),
        TemporalViewport::PointInTime(ts) => right_table.scan_as_of(group_id, *ts),
        TemporalViewport::Range(s, e) => right_table.scan_range(group_id, *s, *e),
    };

    let mut hash_table: HashMap<String, Vec<DecodedRow>> = HashMap::new();
    for row in right_rows {
        let val = cell_to_value(&row.values[right_col_idx]);
        if matches!(val, Value::Null) {
            continue; // NULL keys never match in SQL joins
        }
        let key = value_hash_key(&val);
        hash_table.entry(key).or_default().push(row);
    }

    Ok((right_col_idx, hash_table))
}

/// Probe a pre-built hash table with a join key value. Returns empty for NULL keys.
fn probe_join_hash_table<'a>(
    hash_table: &'a HashMap<String, Vec<DecodedRow>>,
    join_key: &Value,
) -> &'a [DecodedRow] {
    if matches!(join_key, Value::Null) {
        return &[]; // NULL never matches
    }
    let key = value_hash_key(join_key);
    hash_table.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
}

/// Execute a join lookup: for one left row, find matching rows in the right table.
/// This is the fallback path for graph-to-table joins; table-to-table joins use hash join.
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
            let key_name = format!("{}.{}", left_table, left_col);
            let join_key = left_row.get(&key_name).cloned().unwrap_or(Value::Null);
            // NULL keys never match in SQL joins
            if matches!(join_key, Value::Null) {
                return Ok(vec![]);
            }
            let (_, hash_table) =
                build_join_hash_table(right_table, right_col, group_id, viewport)?;
            Ok(probe_join_hash_table(&hash_table, &join_key).to_vec())
        },
        JoinCondition::GraphToTable {
            graph_var: _,
            table_col: _,
        } => Err(QueryError::ExecutionError(
            "Graph-to-table JOIN requires a MATCH clause. \
             Use execute_mixed_query() for MATCH...JOIN queries."
                .into(),
        )),
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
        RBoolExpr::In(e, vals) => {
            let v = eval_rexpr(e, row);
            vals.iter()
                .any(|candidate| values_equal(&v, &eval_rexpr(candidate, row)))
        },
        RBoolExpr::NotIn(e, vals) => {
            let v = eval_rexpr(e, row);
            !vals
                .iter()
                .any(|candidate| values_equal(&v, &eval_rexpr(candidate, row)))
        },
        RBoolExpr::Between(e, low, high) => {
            let v = eval_rexpr(e, row);
            let lo = eval_rexpr(low, row);
            let hi = eval_rexpr(high, row);
            v.partial_cmp(&lo)
                .is_some_and(|o| o != std::cmp::Ordering::Less)
                && v.partial_cmp(&hi)
                    .is_some_and(|o| o != std::cmp::Ordering::Greater)
        },
        RBoolExpr::Like(e, pattern) => {
            if let Value::String(s) = eval_rexpr(e, row) {
                like_match(&s, pattern)
            } else {
                false
            }
        },
        RBoolExpr::And(a, b) => eval_filter(a, row) && eval_filter(b, row),
        RBoolExpr::Or(a, b) => eval_filter(a, row) || eval_filter(b, row),
        RBoolExpr::Not(inner) => !eval_filter(inner, row),
        RBoolExpr::Paren(inner) => eval_filter(inner, row),
        RBoolExpr::FuncPredicate(_, _) => false, // Function predicates not evaluated for tables
    }
}

/// SQL LIKE pattern matching: `%` matches any sequence, `_` matches any single character.
pub fn like_match(s: &str, pattern: &str) -> bool {
    let s_bytes = s.as_bytes();
    let p = pattern.as_bytes();
    let plen = p.len();

    // dp[j] = true if s[0..i] matches p[0..j]
    let mut dp = vec![false; plen + 1];
    dp[0] = true;
    // Initialize: leading % matches empty
    for (j, &ch) in p.iter().enumerate() {
        if ch == b'%' {
            dp[j + 1] = dp[j];
        } else {
            break;
        }
    }

    for &sc in s_bytes {
        let mut new_dp = vec![false; plen + 1];
        for j in 0..plen {
            if p[j] == b'%' {
                new_dp[j + 1] = dp[j + 1] || dp[j] || new_dp[j];
            } else if p[j] == b'_' || p[j] == sc {
                new_dp[j + 1] = dp[j];
            }
        }
        dp = new_dp;
    }
    dp[plen]
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
        super::ast::Literal::NodeRef(id) => Value::Int(*id as i64),
    }
}

fn eval_builtin_func(name: &str, args: &[Value]) -> Value {
    match name.to_lowercase().as_str() {
        // Aggregate functions (count, sum, avg, min, max, collect) are handled
        // in project_row_for_aggregation() — they never reach this function.
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

/// Project a single row, using `agg.input_expr` for aggregate columns.
///
/// For aggregate projections (sum, avg, min, max, collect), Phase 2 must
/// produce the raw input value — the aggregate function is applied later in
/// Phase 3 by `compute_aggregate`.  `count(*)` is special-cased to a
/// non-null sentinel so the null-filtering count works correctly.
fn project_row_for_aggregation(
    projections: &[Projection],
    agg_input: &HashMap<&str, &Aggregation>,
    row: &HashMap<String, Value>,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(projections.len());
    for proj in projections {
        if let Some(agg) = agg_input.get(proj.alias.as_str()) {
            if matches!(agg.function, AggregateFunction::Count)
                && matches!(agg.input_expr, RExpr::Star)
            {
                out.push(Value::Int(1));
            } else {
                out.push(eval_projection(&agg.input_expr, row));
            }
        } else {
            out.push(eval_projection(&proj.expr, row));
        }
    }
    out
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

/// Compute a single aggregate value from a column of values.
fn compute_aggregate(vals: &[&Value], function: &super::planner::AggregateFunction) -> Value {
    match function {
        super::planner::AggregateFunction::Count => {
            Value::Int(vals.iter().filter(|v| !v.is_null()).count() as i64)
        },
        super::planner::AggregateFunction::Sum => {
            // Preserve integer type when all inputs are Int; promote to
            // Float on overflow or if any input is Float.
            let mut int_sum: i64 = 0;
            let mut overflowed = false;
            let mut has_float = false;
            let mut has_value = false;
            for v in vals.iter() {
                match *v {
                    Value::Int(i) => {
                        has_value = true;
                        match int_sum.checked_add(*i) {
                            Some(s) => int_sum = s,
                            None => {
                                overflowed = true;
                                break;
                            },
                        }
                    },
                    Value::Float(_) => {
                        has_float = true;
                        break;
                    },
                    _ => {},
                }
            }
            if has_float || overflowed {
                let sum: f64 = vals
                    .iter()
                    .filter_map(|v| match *v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .sum();
                Value::Float(sum)
            } else if has_value {
                Value::Int(int_sum)
            } else {
                Value::Null
            }
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
            Value::List(vals.iter().cloned().cloned().collect())
        },
    }
}

/// Apply aggregations to result rows, with GROUP BY support.
///
/// When `group_by_keys` is non-empty, rows are partitioned by their group key
/// values using a hash map, and each group produces one output row. When empty,
/// all rows form a single group.
fn apply_aggregations(
    rows: &[Vec<Value>],
    aggregations: &[Aggregation],
    columns: &[String],
    group_by_keys: &[String],
) -> Vec<Vec<Value>> {
    if rows.is_empty() {
        // Empty set: return one row of zeros/nulls (standard SQL aggregation behavior)
        return vec![aggregations
            .iter()
            .map(|agg| match agg.function {
                super::planner::AggregateFunction::Count => Value::Int(0),
                _ => Value::Null,
            })
            .collect()];
    }

    // Find column indices for group-by keys
    let key_indices: Vec<usize> = group_by_keys
        .iter()
        .filter_map(|k| columns.iter().position(|c| c == k))
        .collect();

    // Build a map from output column alias → aggregation for fast lookup
    let agg_map: HashMap<&str, &Aggregation> = aggregations
        .iter()
        .map(|a| (a.output_alias.as_str(), a))
        .collect();

    let group_by_set: HashSet<&str> = group_by_keys.iter().map(|s| s.as_str()).collect();

    if key_indices.is_empty() {
        // Single group — all rows aggregated together.
        let all_indices: Vec<usize> = (0..rows.len()).collect();
        let out_row = build_group_row(rows, columns, &group_by_set, &agg_map, &[], &all_indices);
        return vec![out_row];
    }

    // Hash-based grouping: partition rows by hashing group key columns
    // in-place (no String allocation per row).
    let mut groups: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
    let mut key_to_group: HashMap<u64, usize> = HashMap::new();

    for (row_idx, row) in rows.iter().enumerate() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &ci in &key_indices {
            std::mem::discriminant(&row[ci]).hash(&mut hasher);
            match &row[ci] {
                Value::String(s) => s.hash(&mut hasher),
                Value::Int(v) => v.hash(&mut hasher),
                Value::Float(f) => f.to_bits().hash(&mut hasher),
                Value::Bool(b) => b.hash(&mut hasher),
                Value::Null => 0u8.hash(&mut hasher),
                Value::List(items) => items.len().hash(&mut hasher),
                Value::Map(m) => m.len().hash(&mut hasher),
            }
        }
        let hash_key = hasher.finish();

        if let Some(&group_idx) = key_to_group.get(&hash_key) {
            groups[group_idx].1.push(row_idx);
        } else {
            let group_idx = groups.len();
            key_to_group.insert(hash_key, group_idx);
            let key_vals: Vec<Value> = key_indices.iter().map(|&i| row[i].clone()).collect();
            groups.push((key_vals, vec![row_idx]));
        }
    }

    // For each group, compute aggregations
    groups
        .iter()
        .map(|(key_vals, row_indices)| {
            build_group_row(
                rows,
                columns,
                &group_by_set,
                &agg_map,
                key_vals,
                row_indices,
            )
        })
        .collect()
}

/// Build one output row for a group. Matches each output column to either a group-by
/// key value or the correct aggregation function, by alias name.
fn build_group_row(
    rows: &[Vec<Value>],
    columns: &[String],
    group_by_set: &HashSet<&str>,
    agg_map: &HashMap<&str, &Aggregation>,
    key_vals: &[Value],
    row_indices: &[usize],
) -> Vec<Value> {
    let mut out_row = Vec::with_capacity(columns.len());
    let mut key_idx = 0;

    for (col_idx, col_name) in columns.iter().enumerate() {
        if group_by_set.contains(col_name.as_str()) {
            // Group-by key — emit its value
            if key_idx < key_vals.len() {
                out_row.push(key_vals[key_idx].clone());
                key_idx += 1;
            } else {
                // Fallback: take from first row
                out_row.push(rows[row_indices[0]][col_idx].clone());
            }
        } else if let Some(agg) = agg_map.get(col_name.as_str()) {
            // Aggregation column — collect values from this column across group rows
            let vals: Vec<&Value> = row_indices
                .iter()
                .filter_map(|&i| rows.get(i).and_then(|r| r.get(col_idx)))
                .collect();
            out_row.push(compute_aggregate(&vals, &agg.function));
        } else {
            // Non-aggregated, non-grouped column: take first value in group
            out_row.push(
                rows.get(row_indices[0])
                    .and_then(|r| r.get(col_idx))
                    .cloned()
                    .unwrap_or(Value::Null),
            );
        }
    }
    out_row
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
/// COUNT(*) optimization: if the query is `SELECT count(*) FROM table` with no WHERE,
/// no GROUP BY, and only active-only viewport, return the row count directly
/// without scanning any pages.
fn try_count_star_optimization(
    plan: &ExecutionPlan,
    catalog: &TableCatalog,
    group_id: GroupId,
) -> Option<QueryOutput> {
    // Must be: 1 ScanTable step, no filters, no joins, no GROUP BY
    if plan.steps.len() != 1 || !plan.group_by_keys.is_empty() || plan.having.is_some() {
        return None;
    }
    let table_name = match &plan.steps[0] {
        PlanStep::ScanTable { table_name, .. } => table_name,
        _ => return None,
    };
    // Must have exactly one aggregation: COUNT
    if plan.aggregations.len() != 1
        || plan.aggregations[0].function != super::planner::AggregateFunction::Count
    {
        return None;
    }
    // Must be active-only viewport (no temporal range needed)
    if !matches!(plan.temporal_viewport, TemporalViewport::ActiveOnly) {
        return None;
    }

    let table = catalog.get_table(table_name)?;
    let count = table.active_row_count_for_group(group_id);

    Some(QueryOutput {
        columns: vec![plan.aggregations[0].output_alias.clone()],
        rows: vec![vec![Value::Int(count as i64)]],
        stats: QueryStats {
            nodes_scanned: 0,
            edges_traversed: 0,
            execution_time_ms: 0,
        },
    })
}

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
        PlanStep::ScanTable { table_name, .. } => table_name.clone(),
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
    plan.steps.iter().any(|s| {
        matches!(
            s,
            PlanStep::ScanTable { .. }
                | PlanStep::JoinTable { .. }
                | PlanStep::IndexScan { .. }
                | PlanStep::IndexJoin { .. }
        )
    })
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
        .any(|s| matches!(s, PlanStep::JoinTable { .. } | PlanStep::IndexJoin { .. }));
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
        having: None,
        ordering: vec![],
        limit: plan.limit, // Push limit to graph phase
        temporal_viewport: plan.temporal_viewport.clone(),
        transaction_cutoff: plan.transaction_cutoff,
        var_count: plan.var_count,
        slot_to_table: Vec::new(),
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
            if let PlanStep::JoinTable {
                table_name,
                alias,
                join_type: _,
                on,
            } = join_step
            {
                let _ = alias; // alias used in pure table queries; mixed queries use table name
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
                        let table = catalog.get_table(table_name).ok_or_else(|| {
                            QueryError::ExecutionError(format!("table not found: {}", table_name))
                        })?;
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

    // Phase 3: Project — use input_expr for aggregate columns (same as table query).
    let columns: Vec<String> = plan.projections.iter().map(|p| p.alias.clone()).collect();
    let agg_input: HashMap<&str, &Aggregation> = plan
        .aggregations
        .iter()
        .map(|a| (a.output_alias.as_str(), a))
        .collect();

    let mut projected: Vec<Vec<Value>> = Vec::new();
    for row in &result_rows {
        projected.push(project_row_for_aggregation(
            &plan.projections,
            &agg_input,
            row,
        ));
    }

    // Aggregation
    if !plan.aggregations.is_empty() {
        projected = apply_aggregations(
            &projected,
            &plan.aggregations,
            &columns,
            &plan.group_by_keys,
        );
    }

    // HAVING
    if let Some(ref having_expr) = plan.having {
        projected.retain(|row| {
            let row_map: HashMap<String, Value> = columns
                .iter()
                .zip(row.iter())
                .map(|(c, v)| (c.clone(), v.clone()))
                .collect();
            eval_filter(having_expr, &row_map)
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_tables::catalog::TableCatalog;
    use agent_db_tables::schema::ColumnDef;
    use agent_db_tables::types::ColumnType;

    fn col(name: &str, col_type: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type,
            nullable: true,
            default_value: None,
            autoincrement: false,
        }
    }

    /// Parse a MinnsQL query and execute it against the table catalog.
    fn exec(query: &str, catalog: &TableCatalog) -> QueryOutput {
        let ast = crate::query_lang::parser::Parser::parse(query).unwrap();
        let plan = crate::query_lang::planner::plan(ast).unwrap();
        execute_table_query(catalog, &plan, 0).unwrap()
    }

    /// Helper: create a table with 3 rows, run a query, return the result.
    fn run_table_query(query: &str) -> QueryOutput {
        let mut catalog = TableCatalog::new();
        catalog
            .create_table(
                "items".into(),
                vec![
                    col("price", ColumnType::Int64),
                    col("category", ColumnType::String),
                ],
                vec![],
            )
            .unwrap();
        let table = catalog.get_table_mut("items").unwrap();
        table
            .insert(
                0,
                vec![CellValue::Int64(100), CellValue::String("A".into())],
            )
            .unwrap();
        table
            .insert(
                0,
                vec![CellValue::Int64(200), CellValue::String("A".into())],
            )
            .unwrap();
        table
            .insert(
                0,
                vec![CellValue::Int64(150), CellValue::String("B".into())],
            )
            .unwrap();

        exec(query, &catalog)
    }

    #[test]
    fn test_table_count_star() {
        let out = run_table_query("FROM items RETURN count(*)");
        assert_eq!(out.rows.len(), 1);
        assert_eq!(out.rows[0][0], Value::Int(3));
    }

    #[test]
    fn test_table_sum_int64_returns_int() {
        let out = run_table_query("FROM items RETURN sum(items.price)");
        assert_eq!(out.rows.len(), 1);
        assert_eq!(out.rows[0][0], Value::Int(450));
    }

    #[test]
    fn test_table_avg() {
        let out = run_table_query("FROM items RETURN avg(items.price)");
        assert_eq!(out.rows.len(), 1);
        match &out.rows[0][0] {
            Value::Float(f) => assert!((f - 150.0).abs() < 0.001),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_table_min_max() {
        let out = run_table_query("FROM items RETURN min(items.price), max(items.price)");
        assert_eq!(out.rows.len(), 1);
        assert_eq!(out.rows[0][0], Value::Int(100));
        assert_eq!(out.rows[0][1], Value::Int(200));
    }

    #[test]
    fn test_table_group_by_sum() {
        let out = run_table_query("FROM items RETURN items.category, sum(items.price)");
        assert_eq!(out.rows.len(), 2);
        for row in &out.rows {
            match row[0].as_str() {
                Some("A") => assert_eq!(row[1], Value::Int(300)),
                Some("B") => assert_eq!(row[1], Value::Int(150)),
                other => panic!("unexpected category: {:?}", other),
            }
        }
    }

    #[test]
    fn test_table_count_empty() {
        let mut catalog = TableCatalog::new();
        catalog
            .create_table("empty".into(), vec![col("x", ColumnType::Int64)], vec![])
            .unwrap();
        let out = exec("FROM empty RETURN count(*)", &catalog);
        assert_eq!(out.rows.len(), 1);
        assert_eq!(out.rows[0][0], Value::Int(0));
    }
}
