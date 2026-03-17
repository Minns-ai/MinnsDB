// MinnsQL query handler

use crate::errors::ApiError;
use crate::state::AppState;
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
        }
        agent_db_graph::query_lang::QueryError::Timeout => {
            ApiError::GatewayTimeout("Query execution timed out".into())
        }
        _ => ApiError::Internal(format!("{}", e)),
    }
}

/// POST /api/query - Execute a MinnsQL query or subscription statement
pub async fn minnsql_query(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    if request.query.len() > 4096 {
        return Err(ApiError::BadRequest("Query too long (max 4096 bytes)".into()));
    }

    info!("MinnsQL: '{}'", request.query);

    // Parse as a Statement to detect SUBSCRIBE/UNSUBSCRIBE.
    let stmt = agent_db_graph::query_lang::parser::Parser::parse_statement(&request.query)
        .map_err(|e| map_query_error(&e))?;

    match stmt {
        agent_db_graph::query_lang::Statement::Query(_query) => {
            // Regular query — use the existing execute_query path.
            let _permit = state
                .read_gate
                .acquire()
                .await
                .map_err(ApiError::ServiceUnavailable)?;

            let inference = state.engine.inference().read().await;
            let graph = inference.graph();
            let ontology = state.engine.ontology();

            let result = agent_db_graph::query_lang::execute_query(
                &request.query,
                graph,
                &ontology,
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
        }
        agent_db_graph::query_lang::Statement::Subscribe(query) => {
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
                let (sub_id, output) = sub_mgr
                    .subscribe(plan, graph, &ontology)
                    .map_err(|e| match e {
                        agent_db_graph::query_lang::QueryError::ExecutionError(ref msg)
                            if msg.contains("Subscription limit") =>
                        {
                            ApiError::ServiceUnavailable(format!("{}", e))
                        }
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
        }
        agent_db_graph::query_lang::Statement::Unsubscribe(id) => {
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
        }
    }
}
