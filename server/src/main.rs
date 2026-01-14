// EventGraphDB REST API Server
//
// Provides HTTP endpoints for the self-evolving agent database

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response, Json},
    routing::{get, post},
    Router,
};
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use agent_db_events::Event;
use agent_db_core::types::{AgentId, ContextHash};
use serde::{Deserialize, Serialize};
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
    strength: f32,
    relevance_score: f32,
    access_count: u32,
    formed_at: u64,
    last_accessed: u64,
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

// ============================================================================
// Error Handling
// ============================================================================

enum ApiError {
    Internal(String),
    BadRequest(String),
    NotFound(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, details) = match self {
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error".to_string(), Some(msg)),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "Bad Request".to_string(), Some(msg)),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "Not Found".to_string(), Some(msg)),
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

// ============================================================================
// API Endpoints
// ============================================================================

// POST /api/events - Process a new event
async fn process_event(
    State(state): State<AppState>,
    Json(payload): Json<ProcessEventRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    info!("Processing event: id={}", payload.event.id);

    let result = state.engine.process_event(payload.event).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

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

    let memories = state.engine.get_agent_memories(agent_id, pagination.limit).await;

    let response: Vec<MemoryResponse> = memories.into_iter().map(|m| MemoryResponse {
        id: m.id,
        agent_id: m.agent_id,
        strength: m.strength,
        relevance_score: m.relevance_score,
        access_count: m.access_count,
        formed_at: m.formed_at,
        last_accessed: m.last_accessed,
    }).collect();

    Ok(Json(response))
}

// GET /api/strategies/agent/:agent_id - Get strategies for an agent
async fn get_agent_strategies(
    State(state): State<AppState>,
    Path(agent_id): Path<AgentId>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<StrategyResponse>>, ApiError> {
    info!("Getting strategies for agent: {}", agent_id);

    let strategies = state.engine.get_agent_strategies(agent_id, pagination.limit).await;

    let response: Vec<StrategyResponse> = strategies.into_iter().map(|s| StrategyResponse {
        id: s.id,
        name: s.name.clone(),
        agent_id: s.agent_id,
        quality_score: s.quality_score,
        success_count: s.success_count,
        failure_count: s.failure_count,
        reasoning_steps: s.reasoning_steps.iter().map(|step| ReasoningStepResponse {
            description: step.description.clone(),
            sequence_order: step.sequence_order,
        }).collect(),
    }).collect();

    Ok(Json(response))
}

// GET /api/suggestions - Get action suggestions (Policy Guide)
async fn get_action_suggestions(
    State(state): State<AppState>,
    Query(query): Query<ActionSuggestionsQuery>,
) -> Result<Json<Vec<ActionSuggestionResponse>>, ApiError> {
    info!("Getting action suggestions for context: {}", query.context_hash);

    let suggestions = state.engine.get_next_action_suggestions(
        query.context_hash,
        query.last_action_node,
        query.limit
    ).await.map_err(|e| ApiError::Internal(e.to_string()))?;

    let response: Vec<ActionSuggestionResponse> = suggestions.into_iter().map(|s| ActionSuggestionResponse {
        action_name: s.action_name,
        success_probability: s.success_probability,
        evidence_count: s.evidence_count,
        reasoning: s.reasoning,
    }).collect();

    Ok(Json(response))
}

// GET /api/episodes - Get completed episodes
async fn get_episodes(
    State(state): State<AppState>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<EpisodeResponse>>, ApiError> {
    info!("Getting episodes");

    let episodes = state.engine.get_completed_episodes().await;

    let response: Vec<EpisodeResponse> = episodes.into_iter()
        .take(pagination.limit)
        .map(|e| EpisodeResponse {
            id: e.id,
            agent_id: e.agent_id,
            event_count: e.events.len(),
            significance: e.significance,
            outcome: e.outcome.map(|o| format!("{:?}", o)),
        }).collect();

    Ok(Json(response))
}

// GET /api/stats - Get system statistics
async fn get_stats(
    State(state): State<AppState>,
) -> Result<Json<StatsResponse>, ApiError> {
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

// GET /api/health - Health check endpoint
async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, ApiError> {
    let health = state.engine.get_health_metrics().await;

    Ok(Json(HealthResponse {
        status: if health.is_healthy { "healthy".to_string() } else { "degraded".to_string() },
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
    "EventGraphDB REST API Server v0.1.0\n\nEndpoints:\n\
     POST /api/events - Process event\n\
     GET /api/memories/agent/:id - Get agent memories\n\
     GET /api/strategies/agent/:id - Get agent strategies\n\
     GET /api/suggestions - Get action suggestions\n\
     GET /api/episodes - Get episodes\n\
     GET /api/stats - Get statistics\n\
     GET /api/health - Health check\n\
     GET /docs - API documentation"
}

// GET /docs - API documentation
async fn docs() -> &'static str {
    "EventGraphDB API Documentation\n\n\
     See API_REFERENCE.md for complete documentation."
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

    // Initialize GraphEngine
    info!("Initializing GraphEngine with automatic self-evolution...");
    let config = GraphEngineConfig::default();
    let engine = GraphEngine::with_config(config).await?;
    info!("✓ GraphEngine initialized");

    // Create application state
    let state = AppState {
        engine: Arc::new(engine),
    };

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/docs", get(docs))
        .route("/api/health", get(health_check))
        .route("/api/events", post(process_event))
        .route("/api/memories/agent/:agent_id", get(get_agent_memories))
        .route("/api/strategies/agent/:agent_id", get(get_agent_strategies))
        .route("/api/suggestions", get(get_action_suggestions))
        .route("/api/episodes", get(get_episodes))
        .route("/api/stats", get(get_stats))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = "127.0.0.1:3000";
    info!("🌐 Server listening on http://{}", addr);
    info!("📚 API documentation: http://{}/docs", addr);
    info!("❤️  Health check: http://{}/api/health", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
