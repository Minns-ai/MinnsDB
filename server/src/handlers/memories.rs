// Memory-related handlers

use crate::errors::ApiError;
use crate::models::{ContextMemoriesRequest, MemoryResponse, PaginationQuery};
use crate::state::AppState;
use agent_db_core::types::AgentId;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use tracing::info;

fn memory_type_label(memory_type: &agent_db_graph::memory::MemoryType) -> String {
    match memory_type {
        agent_db_graph::memory::MemoryType::Episodic { .. } => "Episodic".to_string(),
        agent_db_graph::memory::MemoryType::Working => "Working".to_string(),
        agent_db_graph::memory::MemoryType::Semantic => "Semantic".to_string(),
        agent_db_graph::memory::MemoryType::Negative { .. } => "Negative".to_string(),
    }
}

// GET /api/memories/agent/:agent_id - Get memories for an agent
pub async fn get_agent_memories(
    State(state): State<AppState>,
    Path(agent_id): Path<AgentId>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<MemoryResponse>>, ApiError> {
    info!("Getting memories for agent: {}", agent_id);

    let memories = state
        .engine
        .get_agent_memories(agent_id, pagination.limit)
        .await;

    let response: Vec<MemoryResponse> = memories
        .into_iter()
        .map(memory_to_response)
        .collect();

    Ok(Json(response))
}

// POST /api/memories/context - Get memories for a similar context
pub async fn get_memories_by_context(
    State(state): State<AppState>,
    Json(payload): Json<ContextMemoriesRequest>,
) -> Result<Json<Vec<MemoryResponse>>, ApiError> {
    info!(
        "Getting memories for context hash: {}",
        payload.context.fingerprint
    );

    let min_similarity = payload.min_similarity.unwrap_or(0.6);
    let memories = state
        .engine
        .retrieve_memories_by_context_similar(
            &payload.context,
            payload.limit,
            min_similarity,
            payload.agent_id,
            payload.session_id,
        )
        .await;

    let response: Vec<MemoryResponse> = memories
        .into_iter()
        .map(memory_to_response)
        .collect();

    Ok(Json(response))
}

fn memory_to_response(m: agent_db_graph::memory::Memory) -> MemoryResponse {
    MemoryResponse {
        id: m.id,
        agent_id: m.agent_id,
        session_id: m.session_id,
        summary: m.summary.clone(),
        takeaway: m.takeaway.clone(),
        causal_note: m.causal_note.clone(),
        tier: format!("{:?}", m.tier),
        consolidation_status: format!("{:?}", m.consolidation_status),
        schema_id: None, // populated for consolidated memories
        consolidated_from: Vec::new(),
        strength: m.strength,
        relevance_score: m.relevance_score,
        access_count: m.access_count,
        formed_at: m.formed_at,
        last_accessed: m.last_accessed,
        context_hash: m.context.fingerprint,
        context: m.context.clone(),
        outcome: format!("{:?}", m.outcome),
        memory_type: memory_type_label(&m.memory_type),
    }
}
