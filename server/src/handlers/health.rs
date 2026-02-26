// Health check and informational endpoints

use crate::errors::ApiError;
use crate::models::HealthResponse;
use crate::state::AppState;
use axum::{extract::State, Json};

// GET /api/health - Health check endpoint
pub async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let health = state.engine.get_health_metrics().await;

    Ok(Json(HealthResponse {
        status: if health.is_healthy {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        },
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.started_at.elapsed().as_secs(),
        is_healthy: health.is_healthy,
        node_count: health.node_count,
        edge_count: health.edge_count,
        processing_rate: health.processing_rate,
    }))
}

// GET / - Root endpoint
pub async fn root() -> &'static str {
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
     Advanced Graph Features:\n\
     GET /api/analytics - Graph analytics with learning metrics\n\
     GET /api/indexes - Property index statistics\n\
     GET /api/communities - Community detection (Louvain)\n\
     GET /api/centrality - Node centrality scores\n\n\
     Natural Language & Conversation:\n\
     POST /api/nlq - Natural language graph query\n\
     POST /api/conversations/ingest - Ingest conversation sessions\n\
     POST /api/conversations/query - Query with conversation context\n\n\
     GET /docs - API documentation"
}

// GET /docs - API documentation
pub async fn docs() -> &'static str {
    "EventGraphDB API Documentation\n\n\
     See API_REFERENCE.md for complete documentation."
}
