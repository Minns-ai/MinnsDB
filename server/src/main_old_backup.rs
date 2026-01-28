// EventGraphDB REST API Server
//
// Provides HTTP endpoints for the self-evolving agent database

use agent_db_core::types::{AgentId, AgentType, ContextHash, SessionId};
use agent_db_events::core::EventContext;
use agent_db_events::core::EventType;
use agent_db_events::Event;
use agent_db_graph::integration::StorageBackend;
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
struct AppState {
    engine: Arc<GraphEngine>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct ProcessEventRequest {
    event: Event,
    /// Enable semantic memory processing (NER + claim extraction + embeddings)
    #[serde(default)]
    enable_semantic: bool,
}

#[derive(Debug, Serialize)]
struct ProcessEventResponse {
    success: bool,
    nodes_created: usize,
    patterns_detected: usize,
    processing_time_ms: u64,
}

#[derive(Debug, Deserialize)]
struct PaginationQuery {
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Deserialize)]
struct ActionSuggestionsQuery {
    context_hash: ContextHash,
    #[serde(default)]
    last_action_node: Option<u64>,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Debug, Serialize)]
struct MemoryResponse {
    id: u64,
    agent_id: AgentId,
    session_id: SessionId,
    strength: f32,
    relevance_score: f32,
    access_count: u32,
    formed_at: u64,
    last_accessed: u64,
    context_hash: ContextHash,
    context: EventContext,
    outcome: String,
    memory_type: String,
}

#[derive(Debug, Serialize)]
struct StrategyResponse {
    id: u64,
    name: String,
    agent_id: AgentId,
    quality_score: f32,
    success_count: u32,
    failure_count: u32,
    reasoning_steps: Vec<ReasoningStepResponse>,
    strategy_type: String,
    support_count: u32,
    expected_success: f32,
    expected_cost: f32,
    expected_value: f32,
    confidence: f32,
    goal_bucket_id: u64,
    behavior_signature: String,
    precondition: String,
    action_hint: String,
}

#[derive(Debug, Serialize)]
struct SimilarStrategyResponse {
    score: f32,
    id: u64,
    name: String,
    agent_id: AgentId,
    quality_score: f32,
    success_count: u32,
    failure_count: u32,
    reasoning_steps: Vec<ReasoningStepResponse>,
    strategy_type: String,
    support_count: u32,
    expected_success: f32,
    expected_cost: f32,
    expected_value: f32,
    confidence: f32,
    goal_bucket_id: u64,
    behavior_signature: String,
    precondition: String,
    action_hint: String,
}

#[derive(Debug, Serialize)]
struct ReasoningStepResponse {
    description: String,
    sequence_order: usize,
}

#[derive(Debug, Serialize)]
struct ActionSuggestionResponse {
    action_name: String,
    success_probability: f32,
    evidence_count: u32,
    reasoning: String,
}

#[derive(Debug, Serialize)]
struct EpisodeResponse {
    id: u64,
    agent_id: AgentId,
    event_count: usize,
    significance: f32,
    outcome: Option<String>,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    total_events_processed: u64,
    total_nodes_created: u64,
    total_episodes_detected: u64,
    total_memories_formed: u64,
    total_strategies_extracted: u64,
    total_reinforcements_applied: u64,
    average_processing_time_ms: f64,
}

#[derive(Debug, Serialize)]
struct GraphResponse {
    nodes: Vec<GraphNodeResponse>,
    edges: Vec<GraphEdgeResponse>,
}

#[derive(Debug, Deserialize)]
struct GraphQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    session_id: Option<SessionId>,
    #[serde(default)]
    agent_type: Option<AgentType>,
}

#[derive(Debug, Deserialize)]
struct GraphContextQuery {
    context_hash: ContextHash,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    session_id: Option<SessionId>,
    #[serde(default)]
    agent_type: Option<AgentType>,
}

#[derive(Debug, Deserialize)]
struct ContextMemoriesRequest {
    context: EventContext,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    min_similarity: Option<f32>,
    #[serde(default)]
    agent_id: Option<AgentId>,
    #[serde(default)]
    session_id: Option<SessionId>,
}

#[derive(Debug, Deserialize)]
struct StrategySimilarityRequest {
    #[serde(default)]
    goal_ids: Vec<u64>,
    #[serde(default)]
    tool_names: Vec<String>,
    #[serde(default)]
    result_types: Vec<String>,
    #[serde(default)]
    context_hash: Option<ContextHash>,
    #[serde(default)]
    agent_id: Option<AgentId>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    min_score: Option<f32>,
}

#[derive(Debug, Serialize)]
struct GraphNodeResponse {
    id: u64,
    label: String,
    node_type: String,
    created_at: u64,
    properties: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GraphEdgeResponse {
    id: u64,
    from: u64,
    to: u64,
    edge_type: String,
    weight: f32,
    confidence: f32,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_seconds: u64,
    is_healthy: bool,
    node_count: usize,
    edge_count: usize,
    processing_rate: f64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    details: Option<String>,
}

// NEW: Advanced Graph Features Response Types

#[derive(Debug, Serialize)]
struct AnalyticsResponse {
    node_count: usize,
    edge_count: usize,
    connected_components: usize,
    largest_component_size: usize,
    average_path_length: f32,
    diameter: u32,
    clustering_coefficient: f32,
    average_clustering: f32,
    modularity: f32,
    community_count: usize,
    learning_metrics: LearningMetricsResponse,
}

#[derive(Debug, Serialize)]
struct LearningMetricsResponse {
    total_events: usize,
    unique_contexts: usize,
    learned_patterns: usize,
    strong_memories: usize,
    overall_success_rate: f32,
    average_edge_weight: f32,
}

#[derive(Debug, Serialize)]
struct IndexStatsResponse {
    insert_count: u64,
    query_count: u64,
    range_query_count: u64,
    hit_count: u64,
    miss_count: u64,
    last_accessed: u64,
}

#[derive(Debug, Serialize)]
struct CommunityResponse {
    community_id: u64,
    node_ids: Vec<u64>,
    size: usize,
}

// ============================================================================
// Semantic Memory Response Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct ClaimSearchRequest {
    query_text: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_min_similarity")]
    min_similarity: f32,
}

fn default_top_k() -> usize {
    10
}

fn default_min_similarity() -> f32 {
    0.7
}

#[derive(Debug, Serialize)]
struct ClaimResponse {
    claim_id: u64,
    claim_text: String,
    confidence: f32,
    source_event_id: u128,
    similarity: Option<f32>,
    evidence_spans: Vec<EvidenceSpanResponse>,
    support_count: u32,
    status: String,
    created_at: u64,
    last_accessed: u64,
}

#[derive(Debug, Serialize)]
struct EvidenceSpanResponse {
    start_offset: usize,
    end_offset: usize,
    text_snippet: String,
}

#[derive(Debug, Serialize)]
struct EmbeddingProcessResponse {
    claims_processed: usize,
    success: bool,
}

#[derive(Debug, Deserialize)]
struct ClaimListQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    event_id: Option<u128>,
}

#[derive(Debug, Serialize)]
struct CommunitiesResponse {
    communities: Vec<CommunityResponse>,
    modularity: f32,
    iterations: usize,
    community_count: usize,
}

#[derive(Debug, Serialize)]
struct CentralityScoresResponse {
    node_id: u64,
    degree: f32,
    betweenness: f32,
    closeness: f32,
    eigenvector: f32,
    pagerank: f32,
    combined: f32,
}

// ============================================================================
// Error Handling
// ============================================================================

#[allow(dead_code)]
enum ApiError {
    Internal(String),
    BadRequest(String),
    NotFound(String),
    NotImplemented(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, details) = match self {
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal Server Error".to_string(),
                Some(msg),
            ),
            ApiError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "Bad Request".to_string(),
                Some(msg),
            ),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "Not Found".to_string(), Some(msg)),
            ApiError::NotImplemented(msg) => (
                StatusCode::NOT_IMPLEMENTED,
                "Not Implemented".to_string(),
                Some(msg),
            ),
        };

        let body = Json(ErrorResponse {
            error: message,
            details,
        });

        (status, body).into_response()
    }
}

impl From<Box<dyn std::error::Error>> for ApiError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ApiError::Internal(err.to_string())
    }
}

fn event_type_name(event_type: &EventType) -> &'static str {
    match event_type {
        EventType::Action { .. } => "Action",
        EventType::Observation { .. } => "Observation",
        EventType::Cognitive { .. } => "Cognitive",
        EventType::Communication { .. } => "Communication",
        EventType::Learning { .. } => "Learning",
        EventType::Context { .. } => "Context",
    }
}

fn memory_type_label(memory_type: &agent_db_graph::memory::MemoryType) -> String {
    match memory_type {
        agent_db_graph::memory::MemoryType::Episodic { .. } => "Episodic".to_string(),
        agent_db_graph::memory::MemoryType::Working => "Working".to_string(),
        agent_db_graph::memory::MemoryType::Semantic => "Semantic".to_string(),
        agent_db_graph::memory::MemoryType::Negative { .. } => "Negative".to_string(),
    }
}

// ============================================================================
// API Endpoints
// ============================================================================

// POST /api/events - Process a new event
async fn process_event(
    State(state): State<AppState>,
    Json(payload): Json<ProcessEventRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    let start = std::time::Instant::now();
    let event_id = payload.event.id;
    info!(
        "Processing event: id={} agent_id={} session_id={} type={}",
        event_id,
        payload.event.agent_id,
        payload.event.session_id,
        event_type_name(&payload.event.event_type)
    );

    // Auto-compute fingerprint if not provided (fingerprint == 0)
    let mut event = payload.event;
    if event.context.fingerprint == 0 {
        info!("Event {} missing fingerprint, computing.", event.id);
        event.context.fingerprint = event.context.compute_fingerprint();
    }

    let result = state
        .engine
        .process_event_with_options(event, Some(payload.enable_semantic))
        .await
        .map_err(|e| {
            info!("Error processing event: {:?}", e);
            ApiError::Internal(e.to_string())
        })?;

    info!(
        "Processed event {}: nodes_created={} patterns_detected={} processing_time_ms={} total_handler_ms={}",
        event_id,
        result.nodes_created.len(),
        result.patterns_detected.len(),
        result.processing_time_ms,
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        nodes_created: result.nodes_created.len(),
        patterns_detected: result.patterns_detected.len(),
        processing_time_ms: result.processing_time_ms,
    }))
}

// GET /api/memories/agent/:agent_id - Get memories for an agent
async fn get_agent_memories(
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
        .map(|m| MemoryResponse {
            id: m.id,
            agent_id: m.agent_id,
            session_id: m.session_id,
            strength: m.strength,
            relevance_score: m.relevance_score,
            access_count: m.access_count,
            formed_at: m.formed_at,
            last_accessed: m.last_accessed,
            context_hash: m.context.fingerprint,
            context: m.context.clone(),
            outcome: format!("{:?}", m.outcome),
            memory_type: memory_type_label(&m.memory_type),
        })
        .collect();

    Ok(Json(response))
}

// POST /api/memories/context - Get memories for a similar context
async fn get_memories_by_context(
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
        .map(|m| MemoryResponse {
            id: m.id,
            agent_id: m.agent_id,
            session_id: m.session_id,
            strength: m.strength,
            relevance_score: m.relevance_score,
            access_count: m.access_count,
            formed_at: m.formed_at,
            last_accessed: m.last_accessed,
            context_hash: m.context.fingerprint,
            context: m.context.clone(),
            outcome: format!("{:?}", m.outcome),
            memory_type: memory_type_label(&m.memory_type),
        })
        .collect();

    Ok(Json(response))
}

// GET /api/strategies/agent/:agent_id - Get strategies for an agent
async fn get_agent_strategies(
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
async fn get_similar_strategies(
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
async fn get_action_suggestions(
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

// GET /api/episodes - Get completed episodes
async fn get_episodes(
    State(state): State<AppState>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<EpisodeResponse>>, ApiError> {
    info!("Getting episodes");

    let episodes = state.engine.get_completed_episodes().await;

    let response: Vec<EpisodeResponse> = episodes
        .into_iter()
        .take(pagination.limit)
        .map(|e| EpisodeResponse {
            id: e.id,
            agent_id: e.agent_id,
            event_count: e.events.len(),
            significance: e.significance,
            outcome: e.outcome.map(|o| format!("{:?}", o)),
        })
        .collect();

    Ok(Json(response))
}

// GET /api/events - Get recent events
async fn get_events(
    State(state): State<AppState>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<Event>>, ApiError> {
    info!("Getting recent events");
    let events = state.engine.get_recent_events(pagination.limit).await;
    Ok(Json(events))
}

// GET /api/stats - Get system statistics
async fn get_stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, ApiError> {
    info!("Getting system statistics");

    let stats = state.engine.get_engine_stats().await;

    Ok(Json(StatsResponse {
        total_events_processed: stats.total_events_processed,
        total_nodes_created: stats.total_nodes_created,
        total_episodes_detected: stats.total_episodes_detected,
        total_memories_formed: stats.total_memories_formed,
        total_strategies_extracted: stats.total_strategies_extracted,
        total_reinforcements_applied: stats.total_reinforcements_applied,
        average_processing_time_ms: stats.average_processing_time_ms,
    }))
}

// GET /api/graph - Get graph structure
async fn get_graph(
    State(state): State<AppState>,
    Query(query): Query<GraphQuery>,
) -> Result<Json<GraphResponse>, ApiError> {
    info!("Getting graph structure");

    let graph_data = state
        .engine
        .get_graph_structure(query.limit, query.session_id, query.agent_type)
        .await;

    let nodes: Vec<GraphNodeResponse> = graph_data
        .nodes
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            label: n.label.unwrap_or_else(|| format!("Node {}", n.id)),
            node_type: n.node_type,
            created_at: n.created_at,
            properties: n.properties,
        })
        .collect();

    let edges: Vec<GraphEdgeResponse> = graph_data
        .edges
        .into_iter()
        .map(|e| GraphEdgeResponse {
            id: e.id,
            from: e.from,
            to: e.to,
            edge_type: e.edge_type,
            weight: e.weight,
            confidence: e.confidence,
        })
        .collect();

    Ok(Json(GraphResponse { nodes, edges }))
}

// GET /api/graph/context - Get context-centered graph structure
async fn get_graph_for_context(
    State(state): State<AppState>,
    Query(query): Query<GraphContextQuery>,
) -> Result<Json<GraphResponse>, ApiError> {
    info!(
        "Getting graph structure for context: {}",
        query.context_hash
    );

    let graph_data = state
        .engine
        .get_graph_structure_for_context(
            query.context_hash,
            query.limit,
            query.session_id,
            query.agent_type,
        )
        .await;

    let nodes: Vec<GraphNodeResponse> = graph_data
        .nodes
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            label: n.label.unwrap_or_else(|| format!("Node {}", n.id)),
            node_type: n.node_type,
            created_at: n.created_at,
            properties: n.properties,
        })
        .collect();

    let edges: Vec<GraphEdgeResponse> = graph_data
        .edges
        .into_iter()
        .map(|e| GraphEdgeResponse {
            id: e.id,
            from: e.from,
            to: e.to,
            edge_type: e.edge_type,
            weight: e.weight,
            confidence: e.confidence,
        })
        .collect();

    Ok(Json(GraphResponse { nodes, edges }))
}

// GET /api/health - Health check endpoint
async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let health = state.engine.get_health_metrics().await;

    Ok(Json(HealthResponse {
        status: if health.is_healthy {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        },
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // TODO: track uptime
        is_healthy: health.is_healthy,
        node_count: health.node_count,
        edge_count: health.edge_count,
        processing_rate: health.processing_rate,
    }))
}

// GET / - Root endpoint
async fn root() -> &'static str {
    "EventGraphDB REST API Server v0.2.0\n\n\
     Core Endpoints:\n\
     POST /api/events - Process event\n\
     GET /api/memories/agent/:id - Get agent memories\n\
     POST /api/memories/context - Get memories by context\n\
     GET /api/strategies/agent/:id - Get agent strategies\n\
     POST /api/strategies/similar - Find similar strategies\n\
     GET /api/suggestions - Get action suggestions\n\
     GET /api/episodes - Get episodes\n\
     GET /api/stats - Get statistics\n\
     GET /api/graph - Get graph visualization data\n\
     GET /api/graph/context - Get context graph visualization data\n\
     GET /api/health - Health check\n\n\
     NEW - Advanced Graph Features:\n\
     GET /api/analytics - Graph analytics with learning metrics\n\
     GET /api/indexes - Property index statistics\n\
     GET /api/communities - Community detection (Louvain)\n\
     GET /api/centrality - Node centrality scores\n\n\
     GET /docs - API documentation"
}

// GET /docs - API documentation
async fn docs() -> &'static str {
    "EventGraphDB API Documentation\n\n\
     See API_REFERENCE.md for complete documentation."
}

// ============================================================================
// NEW: Advanced Graph Features Endpoints
// ============================================================================

// GET /api/analytics - Get comprehensive graph analytics
async fn get_analytics(State(state): State<AppState>) -> Result<Json<AnalyticsResponse>, ApiError> {
    let metrics = state
        .engine
        .get_analytics()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get analytics: {}", e)))?;

    Ok(Json(AnalyticsResponse {
        node_count: metrics.node_count,
        edge_count: metrics.edge_count,
        connected_components: metrics.connected_components,
        largest_component_size: metrics.largest_component_size,
        average_path_length: metrics.average_path_length,
        diameter: metrics.diameter,
        clustering_coefficient: metrics.clustering_coefficient,
        average_clustering: metrics.average_clustering,
        modularity: metrics.modularity,
        community_count: metrics.community_count,
        learning_metrics: LearningMetricsResponse {
            total_events: metrics.learning_metrics.total_events,
            unique_contexts: metrics.learning_metrics.unique_contexts,
            learned_patterns: metrics.learning_metrics.learned_patterns,
            strong_memories: metrics.learning_metrics.strong_memories,
            overall_success_rate: metrics.learning_metrics.overall_success_rate,
            average_edge_weight: metrics.learning_metrics.average_edge_weight,
        },
    }))
}

// GET /api/indexes - Get property index statistics
async fn get_indexes(
    State(state): State<AppState>,
) -> Result<Json<Vec<IndexStatsResponse>>, ApiError> {
    let index_stats = state.engine.get_index_stats().await;

    let response: Vec<IndexStatsResponse> = index_stats
        .into_iter()
        .map(|stats| IndexStatsResponse {
            insert_count: stats.insert_count,
            query_count: stats.query_count,
            range_query_count: stats.range_query_count,
            hit_count: stats.hit_count,
            miss_count: stats.miss_count,
            last_accessed: stats.last_accessed,
        })
        .collect();

    Ok(Json(response))
}

// GET /api/communities - Detect and return graph communities
async fn get_communities(
    State(state): State<AppState>,
) -> Result<Json<CommunitiesResponse>, ApiError> {
    let result = state
        .engine
        .detect_communities()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to detect communities: {}", e)))?;

    let communities: Vec<CommunityResponse> = result
        .communities
        .into_iter()
        .map(|(id, nodes)| CommunityResponse {
            community_id: id,
            size: nodes.len(),
            node_ids: nodes,
        })
        .collect();

    Ok(Json(CommunitiesResponse {
        communities,
        modularity: result.modularity,
        iterations: result.iterations,
        community_count: result.community_count,
    }))
}

// GET /api/centrality - Get centrality scores for all nodes
async fn get_centrality(
    State(state): State<AppState>,
) -> Result<Json<Vec<CentralityScoresResponse>>, ApiError> {
    let all_centralities = state
        .engine
        .get_all_centrality_scores()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to calculate centrality: {}", e)))?;

    // Get all node IDs from one of the centrality maps
    let node_ids: Vec<u64> = all_centralities.degree.keys().copied().collect();

    let mut scores: Vec<CentralityScoresResponse> = node_ids
        .into_iter()
        .map(|node_id| {
            let combined = all_centralities.combined_score(node_id);

            CentralityScoresResponse {
                node_id,
                degree: all_centralities
                    .degree
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                betweenness: all_centralities
                    .betweenness
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                closeness: all_centralities
                    .closeness
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                eigenvector: all_centralities
                    .eigenvector
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                pagerank: all_centralities
                    .pagerank
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                combined,
            }
        })
        .collect();

    // Sort by combined score descending
    scores.sort_by(|a, b| {
        b.combined
            .partial_cmp(&a.combined)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(Json(scores))
}

// ============================================================================
// Semantic Memory Handlers
// ============================================================================

async fn search_claims(
    State(state): State<AppState>,
    Json(payload): Json<ClaimSearchRequest>,
) -> Result<Json<Vec<ClaimResponse>>, ApiError> {
    info!(
        "Searching for claims: query={} top_k={} min_similarity={}",
        payload.query_text, payload.top_k, payload.min_similarity
    );

    let results = state
        .engine
        .search_similar_claims(&payload.query_text, payload.top_k, payload.min_similarity)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to search claims: {}", e)))?;

    let responses: Vec<ClaimResponse> = results
        .into_iter()
        .map(|(claim, similarity)| ClaimResponse {
            claim_id: claim.id,
            claim_text: claim.claim_text,
            confidence: claim.confidence,
            source_event_id: claim.source_event_id,
            similarity: Some(similarity),
            evidence_spans: claim
                .supporting_evidence
                .into_iter()
                .map(|span| EvidenceSpanResponse {
                    start_offset: span.start_offset,
                    end_offset: span.end_offset,
                    text_snippet: span.text_snippet,
                })
                .collect(),
            support_count: claim.support_count,
            status: format!("{:?}", claim.status),
            created_at: claim.created_at,
            last_accessed: claim.last_accessed,
        })
        .collect();

    info!("Found {} similar claims", responses.len());

    Ok(Json(responses))
}

async fn process_embeddings(
    State(state): State<AppState>,
    Query(params): Query<PaginationQuery>,
) -> Result<Json<EmbeddingProcessResponse>, ApiError> {
    info!("Processing pending embeddings (batch_size={})", params.limit);

    let count = state
        .engine
        .process_pending_embeddings(params.limit)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to process embeddings: {}", e)))?;

    info!("Processed {} claims for embedding generation", count);

    Ok(Json(EmbeddingProcessResponse {
        claims_processed: count,
        success: true,
    }))
}

async fn get_claim(
    State(state): State<AppState>,
    Path(claim_id): Path<u64>,
) -> Result<Json<ClaimResponse>, ApiError> {
    info!("Fetching claim {}", claim_id);

    let claim_store = state
        .engine
        .claim_store()
        .ok_or_else(|| ApiError::Internal("Claim store not initialized".to_string()))?;

    let claim = claim_store
        .get(claim_id)
        .map_err(|e| ApiError::Internal(format!("Failed to retrieve claim: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Claim {} not found", claim_id)))?;

    Ok(Json(ClaimResponse {
        claim_id: claim.id,
        claim_text: claim.claim_text,
        confidence: claim.confidence,
        source_event_id: claim.source_event_id,
        similarity: None,
        evidence_spans: claim
            .supporting_evidence
            .into_iter()
            .map(|span| EvidenceSpanResponse {
                start_offset: span.start_offset,
                end_offset: span.end_offset,
                text_snippet: span.text_snippet,
            })
            .collect(),
        support_count: claim.support_count,
        status: format!("{:?}", claim.status),
        created_at: claim.created_at,
        last_accessed: claim.last_accessed,
    }))
}

async fn list_claims(
    State(state): State<AppState>,
    Query(params): Query<ClaimListQuery>,
) -> Result<Json<Vec<ClaimResponse>>, ApiError> {
    info!("Listing claims (limit={})", params.limit);

    let claim_store = state
        .engine
        .claim_store()
        .ok_or_else(|| ApiError::Internal("Claim store not initialized".to_string()))?;

    let claims = claim_store
        .get_all_active(params.limit)
        .map_err(|e| ApiError::Internal(format!("Failed to retrieve claims: {}", e)))?;

    let responses: Vec<ClaimResponse> = claims
        .into_iter()
        .filter(|claim| {
            if let Some(event_id) = params.event_id {
                claim.source_event_id == event_id
            } else {
                true
            }
        })
        .map(|claim| ClaimResponse {
            claim_id: claim.id,
            claim_text: claim.claim_text,
            confidence: claim.confidence,
            source_event_id: claim.source_event_id,
            similarity: None,
            evidence_spans: claim
                .supporting_evidence
                .into_iter()
                .map(|span| EvidenceSpanResponse {
                    start_offset: span.start_offset,
                    end_offset: span.end_offset,
                    text_snippet: span.text_snippet,
                })
                .collect(),
            support_count: claim.support_count,
            status: format!("{:?}", claim.status),
            created_at: claim.created_at,
            last_accessed: claim.last_accessed,
        })
        .collect();

    info!("Found {} claims", responses.len());

    Ok(Json(responses))
}

// ============================================================================
// Main Application
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("🚀 Starting EventGraphDB REST API Server");

    // Initialize GraphEngine with persistent storage
    info!("Initializing GraphEngine with persistent storage...");
    let mut config = GraphEngineConfig::default();

    // Configure persistent storage backend
    config.storage_backend = StorageBackend::Persistent;
    config.redb_path = PathBuf::from("./data/eventgraph.redb");
    config.redb_cache_size_mb = 128; // 128MB redb cache
    config.memory_cache_size = 10_000; // 10K memories in RAM (~20MB)
    config.strategy_cache_size = 5_000; // 5K strategies in RAM (~15MB)
    config.enable_louvain = env::var("ENABLE_LOUVAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(false);
    config.louvain_interval = env::var("LOUVAIN_INTERVAL").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);

    // Configure semantic memory (always available)
    config.enable_semantic_memory = true;
    config.ner_workers = 2;
    config.ner_service_url = env::var("NER_SERVICE_URL")
        .unwrap_or_else(|_| "http://localhost:8081/ner".to_string());
    config.ner_request_timeout_ms = env::var("NER_REQUEST_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(5_000);
    config.claim_workers = 4;
    config.embedding_workers = 2;
    config.ner_storage_path = Some(PathBuf::from("./data/ner_features.redb"));
    config.claim_storage_path = Some(PathBuf::from("./data/claims.redb"));
    config.openai_api_key = env::var("OPENAI_API_KEY").ok();
    config.llm_model = env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    config.claim_min_confidence = 0.7;
    config.claim_max_per_input = 10;
    config.enable_embedding_generation = true;
    info!("✓ Semantic memory enabled");
    info!("  NER workers: {}", config.ner_workers);
    info!("  Claim workers: {}", config.claim_workers);
    info!("  Embedding workers: {}", config.embedding_workers);

    // Create data directory if it doesn't exist
    if let Some(parent) = config.redb_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let engine = GraphEngine::with_config(config).await?;
    info!("✓ GraphEngine initialized with persistent storage at ./data/eventgraph.redb");
    info!("  Memory cache: 10,000 items (~20MB)");
    info!("  Strategy cache: 5,000 items (~15MB)");
    info!("  Redb cache: 128MB");

    // Create application state
    let state = AppState {
        engine: Arc::new(engine),
    };

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/docs", get(docs))
        .route("/api/health", get(health_check))
        .route("/api/events", post(process_event).get(get_events))
        .route("/api/memories/agent/:agent_id", get(get_agent_memories))
        .route("/api/memories/context", post(get_memories_by_context))
        .route("/api/strategies/agent/:agent_id", get(get_agent_strategies))
        .route("/api/strategies/similar", post(get_similar_strategies))
        .route("/api/suggestions", get(get_action_suggestions))
        .route("/api/episodes", get(get_episodes))
        .route("/api/stats", get(get_stats))
        .route("/api/graph", get(get_graph))
        .route("/api/graph/context", get(get_graph_for_context))
        // NEW: Advanced graph features endpoints
        .route("/api/analytics", get(get_analytics))
        .route("/api/indexes", get(get_indexes))
        .route("/api/communities", get(get_communities))
        .route("/api/centrality", get(get_centrality))
        // Semantic memory endpoints
        .route("/api/claims", get(list_claims))
        .route("/api/claims/:id", get(get_claim))
        .route("/api/claims/search", post(search_claims))
        .route("/api/embeddings/process", post(process_embeddings))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:3000";
    info!("🌐 Server listening on http://{}", addr);
    info!("📚 API documentation: http://{}/docs", addr);
    info!("❤️  Health check: http://{}/api/health", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
