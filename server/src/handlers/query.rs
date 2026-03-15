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
    pub stats: QueryStatsResponse,
}

#[derive(Serialize)]
pub struct QueryStatsResponse {
    pub nodes_scanned: u64,
    pub edges_traversed: u64,
    pub execution_time_ms: u64,
}

/// POST /api/query - Execute a MinnsQL query against the graph
pub async fn minnsql_query(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    if request.query.len() > 4096 {
        return Err(ApiError::BadRequest("Query too long (max 4096 bytes)".into()));
    }

    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!("MinnsQL query: '{}'", request.query);

    let inference = state.engine.inference().read().await;
    let graph = inference.graph();
    let ontology = state.engine.ontology();

    let result = agent_db_graph::query_lang::execute_query(
        &request.query,
        graph,
        &ontology,
    )
    .map_err(|e| match &e {
        agent_db_graph::query_lang::QueryError::ParseError { message, position } => {
            ApiError::BadRequest(format!("Parse error at position {}: {}", position, message))
        }
        agent_db_graph::query_lang::QueryError::Timeout => {
            ApiError::GatewayTimeout("Query execution timed out".into())
        }
        _ => ApiError::Internal(format!("{}", e)),
    })?;

    Ok(Json(QueryResponse {
        columns: result.columns,
        rows: result
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(|v| v.to_json()).collect())
            .collect(),
        stats: QueryStatsResponse {
            nodes_scanned: result.stats.nodes_scanned,
            edges_traversed: result.stats.edges_traversed,
            execution_time_ms: result.stats.execution_time_ms,
        },
    }))
}
