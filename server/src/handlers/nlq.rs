// Natural language query handler

use crate::errors::ApiError;
use crate::models::{NlqEntity, NlqRequest, NlqResponseBody};
use crate::state::AppState;
use axum::{extract::State, Json};
use tracing::info;

/// POST /api/nlq - Execute a natural language query against the graph
pub async fn nlq_query(
    State(state): State<AppState>,
    Json(request): Json<NlqRequest>,
) -> Result<Json<NlqResponseBody>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!("NLQ query: '{}'", request.question);

    let pagination = agent_db_graph::nlq::NlqPagination {
        limit: request.limit,
        offset: request.offset,
    };

    let response = state
        .engine
        .natural_language_query(
            &request.question,
            &pagination,
            request.session_id.as_deref(),
        )
        .await
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;

    let entities: Vec<NlqEntity> = response
        .entities_resolved
        .iter()
        .map(|e| NlqEntity {
            text: e.mention.text.clone(),
            node_id: e.node_id,
            node_type: e.node_type.clone(),
            confidence: e.confidence,
        })
        .collect();

    let result_count = agent_db_graph::nlq::formatter::result_count(&response.result);

    Ok(Json(NlqResponseBody {
        answer: response.answer,
        intent: agent_db_graph::nlq::intent::intent_display_name(&response.intent).to_string(),
        entities_resolved: entities,
        confidence: response.confidence,
        result_count,
        execution_time_ms: response.execution_time_ms,
        query_used: format!("{:?}", response.query_used),
        explanation: response.explanation,
        total_count: response.total_count,
    }))
}
