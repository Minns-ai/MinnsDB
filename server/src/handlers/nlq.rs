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
        .natural_language_query_with_options(
            &request.question,
            &pagination,
            request.session_id.as_deref(),
            request.include_context,
        )
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("timed out") || msg.contains("synthesis failed") {
                ApiError::GatewayTimeout(msg)
            } else {
                ApiError::BadRequest(msg)
            }
        })?;

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

    // Retrieve related context when requested
    let (related_memories, related_strategies) = if request.include_context {
        let memories = retrieve_related_memories(&state, &request.question).await;
        let strategies = retrieve_related_strategies(&state, &request.question).await;
        (memories, strategies)
    } else {
        (vec![], vec![])
    };

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
        related_memories,
        related_strategies,
    }))
}

/// Retrieve related memories via BM25-only multi-signal retrieval.
async fn retrieve_related_memories(
    state: &AppState,
    question: &str,
) -> Vec<agent_db_graph::MemorySummary> {
    let query = agent_db_graph::MemoryRetrievalQuery {
        query_text: question.to_string(),
        query_embedding: vec![],
        context: None,
        anchor_node: None,
        agent_id: None,
        session_id: None,
        now: None,
        limit: 5,
    };
    let memories = state
        .engine
        .retrieve_memories_multi_signal(query, None)
        .await;
    memories
        .iter()
        .map(agent_db_graph::MemorySummary::from_memory)
        .collect()
}

/// Retrieve related strategies via BM25-only multi-signal retrieval.
async fn retrieve_related_strategies(
    state: &AppState,
    question: &str,
) -> Vec<agent_db_graph::StrategySummary> {
    let query = agent_db_graph::StrategyRetrievalQuery {
        query_text: question.to_string(),
        query_embedding: vec![],
        anchor_node: None,
        now: None,
        limit: 3,
    };
    let strategies = state
        .engine
        .retrieve_strategies_multi_signal(query, None)
        .await;
    strategies
        .iter()
        .map(agent_db_graph::StrategySummary::from_strategy)
        .collect()
}
