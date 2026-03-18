// MinnsQL query handler

use crate::errors::ApiError;
use crate::state::AppState;
use agent_db_graph::query_lang::ast::Statement;
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default)]
    pub group_id: Option<String>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub stats: Option<QueryStatsResponse>,
    /// Present when the query is a SUBSCRIBE statement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscription_id: Option<u64>,
    /// Subscription maintenance strategy (if subscribed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    /// Present when the query is an UNSUBSCRIBE statement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsubscribed: Option<u64>,
}

#[derive(Serialize)]
pub struct QueryStatsResponse {
    pub nodes_scanned: u64,
    pub edges_traversed: u64,
    pub execution_time_ms: u64,
}

fn map_query_error(e: &agent_db_graph::query_lang::QueryError) -> ApiError {
    match e {
        agent_db_graph::query_lang::QueryError::ParseError { message, position } => {
            ApiError::BadRequest(format!("Parse error at position {}: {}", position, message))
        },
        agent_db_graph::query_lang::QueryError::Timeout => {
            ApiError::GatewayTimeout("Query execution timed out".into())
        },
        _ => ApiError::Internal(format!("{}", e)),
    }
}

/// POST /api/query - Execute a MinnsQL query or subscription statement
pub async fn minnsql_query(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    if request.query.len() > 4096 {
        return Err(ApiError::BadRequest(
            "Query too long (max 4096 bytes)".into(),
        ));
    }

    info!("MinnsQL: '{}'", request.query);

    // Parse as a Statement to detect SUBSCRIBE/UNSUBSCRIBE.
    let stmt = agent_db_graph::query_lang::parser::Parser::parse_statement(&request.query)
        .map_err(|e| map_query_error(&e))?;

    let group_id: u64 = request
        .group_id
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    match stmt {
        Statement::Query(ref query) => {
            // Determine if this is a table query or graph query.
            let plan = agent_db_graph::query_lang::planner::plan(query.clone())
                .map_err(|e| map_query_error(&e))?;

            if agent_db_graph::query_lang::table_executor::is_table_query(&plan) {
                // Table query — execute against table catalog.
                let catalog = state.table_catalog.read().await;
                let result = agent_db_graph::query_lang::table_executor::execute_table_query(
                    &catalog, &plan, group_id,
                )
                .map_err(|e| map_query_error(&e))?;

                Ok(Json(QueryResponse {
                    columns: result.columns,
                    rows: result
                        .rows
                        .into_iter()
                        .map(|row| row.into_iter().map(|v| v.to_json()).collect())
                        .collect(),
                    stats: Some(QueryStatsResponse {
                        nodes_scanned: result.stats.nodes_scanned,
                        edges_traversed: result.stats.edges_traversed,
                        execution_time_ms: result.stats.execution_time_ms,
                    }),
                    subscription_id: None,
                    strategy: None,
                    unsubscribed: None,
                }))
            } else {
                // Graph query — use the existing execute_query path.
                let _permit = state
                    .read_gate
                    .acquire()
                    .await
                    .map_err(ApiError::ServiceUnavailable)?;

                let inference = state.engine.inference().read().await;
                let graph = inference.graph();
                let ontology = state.engine.ontology();

                // Execute the already-planned query directly (avoid re-parsing).
                let result =
                    agent_db_graph::query_lang::executor::Executor::execute(graph, ontology, plan)
                        .map_err(|e| map_query_error(&e))?;

                Ok(Json(QueryResponse {
                    columns: result.columns,
                    rows: result
                        .rows
                        .into_iter()
                        .map(|row| row.into_iter().map(|v| v.to_json()).collect())
                        .collect(),
                    stats: Some(QueryStatsResponse {
                        nodes_scanned: result.stats.nodes_scanned,
                        edges_traversed: result.stats.edges_traversed,
                        execution_time_ms: result.stats.execution_time_ms,
                    }),
                    subscription_id: None,
                    strategy: None,
                    unsubscribed: None,
                }))
            }
        },
        Statement::Subscribe(query) => {
            // Plan the query.
            let plan = agent_db_graph::query_lang::planner::plan(query)
                .map_err(|e| map_query_error(&e))?;

            // Subscribe under locks.
            let (sub_id, initial_output, strategy) = {
                let _permit = state
                    .read_gate
                    .acquire()
                    .await
                    .map_err(ApiError::ServiceUnavailable)?;

                let inference = state.engine.inference().read().await;
                let graph = inference.graph();
                let ontology = state.engine.ontology();

                let mut sub_mgr = state.subscription_manager.lock().await;
                let (sub_id, output) =
                    sub_mgr
                        .subscribe(plan, graph, ontology)
                        .map_err(|e| match e {
                            agent_db_graph::query_lang::QueryError::ExecutionError(ref msg)
                                if msg.contains("Subscription limit") =>
                            {
                                ApiError::ServiceUnavailable(format!("{}", e))
                            },
                            _ => ApiError::Internal(format!("Subscription failed: {}", e)),
                        })?;

                let strategy = if let Some(sub_state) = sub_mgr.get_subscription(sub_id) {
                    match &sub_state.incremental_plan.strategy {
                        agent_db_graph::subscription::incremental::MaintenanceStrategy::Incremental => {
                            "incremental".to_string()
                        }
                        agent_db_graph::subscription::incremental::MaintenanceStrategy::FullRerun {
                            reason,
                        } => format!("full_rerun: {}", reason),
                    }
                } else {
                    "unknown".to_string()
                };

                (sub_id, output, strategy)
            };

            // Store query text.
            state
                .subscription_queries
                .lock()
                .await
                .insert(sub_id, request.query);

            Ok(Json(QueryResponse {
                columns: initial_output.columns,
                rows: initial_output
                    .rows
                    .into_iter()
                    .map(|row| row.into_iter().map(|v| v.to_json()).collect())
                    .collect(),
                stats: None,
                subscription_id: Some(sub_id),
                strategy: Some(strategy),
                unsubscribed: None,
            }))
        },
        Statement::Unsubscribe(id) => {
            let removed = {
                let mut sub_mgr = state.subscription_manager.lock().await;
                sub_mgr.unsubscribe(id)
            };
            if removed {
                state.subscription_queries.lock().await.remove(&id);
                Ok(Json(QueryResponse {
                    columns: vec![],
                    rows: vec![],
                    stats: None,
                    subscription_id: None,
                    strategy: None,
                    unsubscribed: Some(id),
                }))
            } else {
                Err(ApiError::NotFound(format!("Subscription {} not found", id)))
            }
        },
        Statement::CreateTable(ct) => {
            let mut columns: Vec<agent_db_tables::schema::ColumnDef> = Vec::new();
            for c in &ct.columns {
                columns.push(agent_db_tables::schema::ColumnDef {
                    name: c.name.clone(),
                    col_type: parse_column_type(&c.col_type)?,
                    nullable: c.nullable,
                    default_value: None,
                });
            }

            let constraints: Vec<agent_db_tables::schema::Constraint> = ct
                .constraints
                .iter()
                .map(|c| match &c.kind {
                    agent_db_graph::query_lang::ast::ConstraintKind::PrimaryKey(cols) => {
                        agent_db_tables::schema::Constraint::PrimaryKey(cols.clone())
                    },
                    agent_db_graph::query_lang::ast::ConstraintKind::Unique(cols) => {
                        agent_db_tables::schema::Constraint::Unique(cols.clone())
                    },
                    agent_db_graph::query_lang::ast::ConstraintKind::NotNull(col) => {
                        agent_db_tables::schema::Constraint::NotNull(col.clone())
                    },
                    agent_db_graph::query_lang::ast::ConstraintKind::ReferencesGraph(col) => {
                        agent_db_tables::schema::Constraint::ForeignKeyGraph(col.clone())
                    },
                })
                .collect();

            let mut catalog = state.table_catalog.write().await;
            let table_id = catalog
                .create_table(ct.name.clone(), columns, constraints)
                .map_err(|e| ApiError::BadRequest(e.to_string()))?;

            info!("Created table '{}' (id={})", ct.name, table_id);

            Ok(Json(QueryResponse {
                columns: vec!["table_id".into(), "name".into()],
                rows: vec![vec![
                    serde_json::Value::Number(table_id.into()),
                    serde_json::Value::String(ct.name),
                ]],
                stats: None,
                subscription_id: None,
                strategy: None,
                unsubscribed: None,
            }))
        },
        Statement::DropTable(name) => {
            let mut catalog = state.table_catalog.write().await;
            let table_id = catalog
                .drop_table(&name)
                .map_err(|e| ApiError::BadRequest(e.to_string()))?;

            info!("Dropped table '{}' (id={})", name, table_id);

            Ok(Json(QueryResponse {
                columns: vec!["dropped".into()],
                rows: vec![vec![serde_json::Value::Bool(true)]],
                stats: None,
                subscription_id: None,
                strategy: None,
                unsubscribed: None,
            }))
        },
        Statement::InsertInto(ins) => {
            let mut catalog = state.table_catalog.write().await;
            let table = catalog
                .get_table_mut(&ins.table)
                .ok_or_else(|| ApiError::NotFound(format!("table not found: {}", ins.table)))?;

            // If column list is provided, map values to schema order.
            // If not, values must be in schema order.
            let col_indices: Option<Vec<usize>> = if let Some(ref cols) = ins.columns {
                let mut indices = Vec::with_capacity(cols.len());
                for col_name in cols {
                    let idx = table.schema.column_index(col_name).ok_or_else(|| {
                        ApiError::BadRequest(format!("unknown column in INSERT: '{}'", col_name))
                    })?;
                    indices.push(idx);
                }
                Some(indices)
            } else {
                None
            };

            let mut results = Vec::new();
            for row_lits in &ins.rows {
                let values = if let Some(ref indices) = col_indices {
                    // Reorder: build a full-width row with NULLs, then fill specified columns
                    if row_lits.len() != indices.len() {
                        return Err(ApiError::BadRequest(format!(
                            "INSERT column count ({}) does not match VALUES count ({})",
                            indices.len(),
                            row_lits.len()
                        )));
                    }
                    let mut full_row =
                        vec![agent_db_tables::types::CellValue::Null; table.schema.columns.len()];
                    for (val_idx, &col_idx) in indices.iter().enumerate() {
                        full_row[col_idx] = literal_to_cell_value(&row_lits[val_idx]);
                    }
                    full_row
                } else {
                    row_lits.iter().map(literal_to_cell_value).collect()
                };
                let (rid, vid) = table
                    .insert(group_id, values)
                    .map_err(|e| ApiError::BadRequest(e.to_string()))?;
                results.push(vec![
                    serde_json::Value::Number(rid.into()),
                    serde_json::Value::Number(vid.into()),
                ]);
            }

            Ok(Json(QueryResponse {
                columns: vec!["row_id".into(), "version_id".into()],
                rows: results,
                stats: None,
                subscription_id: None,
                strategy: None,
                unsubscribed: None,
            }))
        },
        Statement::UpdateTable(upd) => {
            let mut catalog = state.table_catalog.write().await;
            let table = catalog
                .get_table_mut(&upd.table)
                .ok_or_else(|| ApiError::NotFound(format!("table not found: {}", upd.table)))?;

            // Find matching rows by evaluating WHERE against active rows
            let matching_row_ids = find_matching_rows(table, group_id, &upd.where_clause)?;
            if matching_row_ids.is_empty() {
                return Err(ApiError::NotFound("no rows match WHERE condition".into()));
            }

            let mut results = Vec::new();
            for row_id in matching_row_ids {
                let current = table
                    .get_active(group_id, row_id)
                    .ok_or_else(|| ApiError::NotFound(format!("row {} not found", row_id)))?;

                let mut new_values = current.values;
                for (col_name, expr) in &upd.assignments {
                    let col_idx = table.schema.column_index(col_name).ok_or_else(|| {
                        ApiError::BadRequest(format!("unknown column: {}", col_name))
                    })?;
                    new_values[col_idx] = expr_to_cell_value(expr).map_err(|e| {
                        ApiError::BadRequest(format!("column '{}': {}", col_name, e))
                    })?;
                }

                let (old_vid, new_vid) = table
                    .update(group_id, row_id, new_values)
                    .map_err(|e| ApiError::BadRequest(e.to_string()))?;
                results.push(vec![
                    serde_json::Value::Number(old_vid.into()),
                    serde_json::Value::Number(new_vid.into()),
                ]);
            }

            Ok(Json(QueryResponse {
                columns: vec!["old_version_id".into(), "new_version_id".into()],
                rows: results,
                stats: None,
                subscription_id: None,
                strategy: None,
                unsubscribed: None,
            }))
        },
        Statement::DeleteFrom(del) => {
            let mut catalog = state.table_catalog.write().await;
            let table = catalog
                .get_table_mut(&del.table)
                .ok_or_else(|| ApiError::NotFound(format!("table not found: {}", del.table)))?;

            let matching_row_ids = find_matching_rows(table, group_id, &del.where_clause)?;
            if matching_row_ids.is_empty() {
                return Err(ApiError::NotFound("no rows match WHERE condition".into()));
            }

            let mut results = Vec::new();
            for row_id in matching_row_ids {
                let vid = table
                    .delete(group_id, row_id)
                    .map_err(|e| ApiError::BadRequest(e.to_string()))?;
                results.push(vec![serde_json::Value::Number(vid.into())]);
            }

            Ok(Json(QueryResponse {
                columns: vec!["version_id".into()],
                rows: results,
                stats: None,
                subscription_id: None,
                strategy: None,
                unsubscribed: None,
            }))
        },
    }
}

fn parse_column_type(s: &str) -> Result<agent_db_tables::types::ColumnType, ApiError> {
    match s.to_lowercase().as_str() {
        "string" | "text" | "varchar" => Ok(agent_db_tables::types::ColumnType::String),
        "int64" | "int" | "integer" | "bigint" => Ok(agent_db_tables::types::ColumnType::Int64),
        "float64" | "float" | "double" | "real" => Ok(agent_db_tables::types::ColumnType::Float64),
        "bool" | "boolean" => Ok(agent_db_tables::types::ColumnType::Bool),
        "timestamp" => Ok(agent_db_tables::types::ColumnType::Timestamp),
        "json" | "jsonb" => Ok(agent_db_tables::types::ColumnType::Json),
        "noderef" => Ok(agent_db_tables::types::ColumnType::NodeRef),
        _ => Err(ApiError::BadRequest(format!("unknown column type: {}", s))),
    }
}

fn literal_to_cell_value(
    lit: &agent_db_graph::query_lang::ast::Literal,
) -> agent_db_tables::types::CellValue {
    match lit {
        agent_db_graph::query_lang::ast::Literal::String(s) => {
            agent_db_tables::types::CellValue::String(s.clone())
        },
        agent_db_graph::query_lang::ast::Literal::Int(i) => {
            agent_db_tables::types::CellValue::Int64(*i)
        },
        agent_db_graph::query_lang::ast::Literal::Float(f) => {
            agent_db_tables::types::CellValue::Float64(*f)
        },
        agent_db_graph::query_lang::ast::Literal::Bool(b) => {
            agent_db_tables::types::CellValue::Bool(*b)
        },
        agent_db_graph::query_lang::ast::Literal::Null => agent_db_tables::types::CellValue::Null,
    }
}

fn expr_to_cell_value(
    expr: &agent_db_graph::query_lang::ast::Expr,
) -> Result<agent_db_tables::types::CellValue, String> {
    match expr {
        agent_db_graph::query_lang::ast::Expr::Literal(lit) => Ok(literal_to_cell_value(lit)),
        other => Err(format!(
            "only literal values are supported in UPDATE SET, got {:?}",
            other
        )),
    }
}

/// Evaluate a WHERE clause against active rows and return matching RowIds.
/// Supports general comparisons on any column.
fn find_matching_rows(
    table: &agent_db_tables::table::Table,
    group_id: u64,
    where_clause: &agent_db_graph::query_lang::ast::BoolExpr,
) -> Result<Vec<u64>, ApiError> {
    let active = table.scan_active(group_id);
    let mut matching = Vec::new();

    for row in &active {
        if eval_where_on_row(where_clause, row, &table.schema) {
            matching.push(row.row_id);
        }
    }

    Ok(matching)
}

/// Evaluate a BoolExpr against a decoded row.
fn eval_where_on_row(
    expr: &agent_db_graph::query_lang::ast::BoolExpr,
    row: &agent_db_tables::row_codec::DecodedRow,
    schema: &agent_db_tables::schema::TableSchema,
) -> bool {
    use agent_db_graph::query_lang::ast::{BoolExpr, CompOp};

    match expr {
        BoolExpr::Comparison(lhs, op, rhs) => {
            let l = eval_expr_on_row(lhs, row, schema);
            let r = eval_expr_on_row(rhs, row, schema);
            match op {
                CompOp::Eq => cell_values_equal(&l, &r),
                CompOp::Neq => !cell_values_equal(&l, &r),
                CompOp::Lt => cell_values_cmp(&l, &r) == Some(std::cmp::Ordering::Less),
                CompOp::Gt => cell_values_cmp(&l, &r) == Some(std::cmp::Ordering::Greater),
                CompOp::Lte => {
                    cell_values_cmp(&l, &r).is_some_and(|o| o != std::cmp::Ordering::Greater)
                },
                CompOp::Gte => {
                    cell_values_cmp(&l, &r).is_some_and(|o| o != std::cmp::Ordering::Less)
                },
                CompOp::Contains => match (&l, &r) {
                    (
                        agent_db_tables::types::CellValue::String(h),
                        agent_db_tables::types::CellValue::String(n),
                    ) => h.contains(n.as_str()),
                    _ => false,
                },
                CompOp::StartsWith => match (&l, &r) {
                    (
                        agent_db_tables::types::CellValue::String(h),
                        agent_db_tables::types::CellValue::String(n),
                    ) => h.starts_with(n.as_str()),
                    _ => false,
                },
            }
        },
        BoolExpr::And(a, b) => {
            eval_where_on_row(a, row, schema) && eval_where_on_row(b, row, schema)
        },
        BoolExpr::Or(a, b) => {
            eval_where_on_row(a, row, schema) || eval_where_on_row(b, row, schema)
        },
        BoolExpr::Not(inner) => !eval_where_on_row(inner, row, schema),
        BoolExpr::Paren(inner) => eval_where_on_row(inner, row, schema),
        BoolExpr::IsNull(e) => matches!(
            eval_expr_on_row(e, row, schema),
            agent_db_tables::types::CellValue::Null
        ),
        BoolExpr::IsNotNull(e) => !matches!(
            eval_expr_on_row(e, row, schema),
            agent_db_tables::types::CellValue::Null
        ),
        BoolExpr::FuncPredicate(_, _) => false,
    }
}

fn eval_expr_on_row(
    expr: &agent_db_graph::query_lang::ast::Expr,
    row: &agent_db_tables::row_codec::DecodedRow,
    schema: &agent_db_tables::schema::TableSchema,
) -> agent_db_tables::types::CellValue {
    use agent_db_graph::query_lang::ast::Expr;

    match expr {
        Expr::Var(name) | Expr::Property(_, name) => {
            // System columns
            match name.as_str() {
                "id" | "row_id" => {
                    return agent_db_tables::types::CellValue::Int64(row.row_id as i64)
                },
                "version_id" => {
                    return agent_db_tables::types::CellValue::Int64(row.version_id as i64)
                },
                _ => {},
            }
            // User columns
            if let Some(idx) = schema.column_index(name) {
                if idx < row.values.len() {
                    return row.values[idx].clone();
                }
            }
            agent_db_tables::types::CellValue::Null
        },
        Expr::Literal(lit) => literal_to_cell_value(lit),
        _ => agent_db_tables::types::CellValue::Null,
    }
}

fn cell_values_equal(
    a: &agent_db_tables::types::CellValue,
    b: &agent_db_tables::types::CellValue,
) -> bool {
    use agent_db_tables::types::CellValue;
    match (a, b) {
        (CellValue::Int64(x), CellValue::Int64(y)) => x == y,
        (CellValue::Float64(x), CellValue::Float64(y)) => (x - y).abs() < f64::EPSILON,
        (CellValue::String(x), CellValue::String(y)) => x == y,
        (CellValue::Bool(x), CellValue::Bool(y)) => x == y,
        (CellValue::Timestamp(x), CellValue::Timestamp(y)) => x == y,
        (CellValue::NodeRef(x), CellValue::NodeRef(y)) => x == y,
        _ => false,
    }
}

fn cell_values_cmp(
    a: &agent_db_tables::types::CellValue,
    b: &agent_db_tables::types::CellValue,
) -> Option<std::cmp::Ordering> {
    use agent_db_tables::types::CellValue;
    match (a, b) {
        (CellValue::Int64(x), CellValue::Int64(y)) => Some(x.cmp(y)),
        (CellValue::Float64(x), CellValue::Float64(y)) => x.partial_cmp(y),
        (CellValue::String(x), CellValue::String(y)) => Some(x.cmp(y)),
        (CellValue::Timestamp(x), CellValue::Timestamp(y)) => Some(x.cmp(y)),
        _ => None,
    }
}
