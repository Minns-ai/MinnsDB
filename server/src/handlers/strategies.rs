// Strategy-related handlers

use crate::errors::ApiError;
use crate::models::{
    ActionSuggestionResponse, ActionSuggestionsQuery, PaginationQuery, ReasoningStepResponse,
    SimilarStrategyResponse, StrategyResponse, StrategySimilarityRequest,
};
use crate::state::AppState;
use agent_db_core::types::AgentId;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use tracing::info;

// GET /api/strategies/agent/:agent_id - Get strategies for an agent
pub async fn get_agent_strategies(
    State(state): State<AppState>,
    Path(agent_id): Path<AgentId>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<StrategyResponse>>, ApiError> {
    info!("Getting strategies for agent: {}", agent_id);

    let strategies = state
        .engine
        .get_agent_strategies(agent_id, pagination.limit)
        .await;

    let response: Vec<StrategyResponse> = strategies
        .into_iter()
        .map(|s| StrategyResponse {
            id: s.id,
            name: s.name.clone(),
            agent_id: s.agent_id,
            quality_score: s.quality_score,
            success_count: s.success_count,
            failure_count: s.failure_count,
            reasoning_steps: s
                .reasoning_steps
                .iter()
                .map(|step| ReasoningStepResponse {
                    description: step.description.clone(),
                    sequence_order: step.sequence_order,
                })
                .collect(),
            strategy_type: format!("{:?}", s.strategy_type),
            support_count: s.support_count,
            expected_success: s.expected_success,
            expected_cost: s.expected_cost,
            expected_value: s.expected_value,
            confidence: s.confidence,
            goal_bucket_id: s.goal_bucket_id,
            behavior_signature: s.behavior_signature.clone(),
            precondition: s.precondition.clone(),
            action_hint: s.action_hint.clone(),
        })
        .collect();

    Ok(Json(response))
}

// POST /api/strategies/similar - Find similar strategies
pub async fn get_similar_strategies(
    State(state): State<AppState>,
    Json(payload): Json<StrategySimilarityRequest>,
) -> Result<Json<Vec<SimilarStrategyResponse>>, ApiError> {
    let min_score = payload.min_score.unwrap_or(0.2);
    let query = agent_db_graph::strategies::StrategySimilarityQuery {
        goal_ids: payload.goal_ids,
        tool_names: payload.tool_names,
        result_types: payload.result_types,
        context_hash: payload.context_hash,
        agent_id: payload.agent_id,
        min_score,
        limit: payload.limit,
    };

    let strategies = state.engine.get_similar_strategies(query).await;

    let response: Vec<SimilarStrategyResponse> = strategies
        .into_iter()
        .map(|(s, score)| SimilarStrategyResponse {
            score,
            id: s.id,
            name: s.name.clone(),
            agent_id: s.agent_id,
            quality_score: s.quality_score,
            success_count: s.success_count,
            failure_count: s.failure_count,
            reasoning_steps: s
                .reasoning_steps
                .iter()
                .map(|step| ReasoningStepResponse {
                    description: step.description.clone(),
                    sequence_order: step.sequence_order,
                })
                .collect(),
            strategy_type: format!("{:?}", s.strategy_type),
            support_count: s.support_count,
            expected_success: s.expected_success,
            expected_cost: s.expected_cost,
            expected_value: s.expected_value,
            confidence: s.confidence,
            goal_bucket_id: s.goal_bucket_id,
            behavior_signature: s.behavior_signature.clone(),
            precondition: s.precondition.clone(),
            action_hint: s.action_hint.clone(),
        })
        .collect();

    Ok(Json(response))
}

// GET /api/suggestions - Get action suggestions (Policy Guide)
pub async fn get_action_suggestions(
    State(state): State<AppState>,
    Query(query): Query<ActionSuggestionsQuery>,
) -> Result<Json<Vec<ActionSuggestionResponse>>, ApiError> {
    info!(
        "Getting action suggestions for context: {}",
        query.context_hash
    );

    let suggestions = state
        .engine
        .get_next_action_suggestions(query.context_hash, query.last_action_node, query.limit)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let response: Vec<ActionSuggestionResponse> = suggestions
        .into_iter()
        .map(|s| ActionSuggestionResponse {
            action_name: s.action_name,
            success_probability: s.success_probability,
            evidence_count: s.evidence_count,
            reasoning: s.reasoning,
        })
        .collect();

    Ok(Json(response))
}
