// Router setup for EventGraphDB REST API

use crate::handlers;
use crate::state::AppState;
use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Root and docs
        .route("/", get(handlers::root))
        .route("/docs", get(handlers::docs))
        // Health check
        .route("/api/health", get(handlers::health_check))
        // Events
        .route(
            "/api/events",
            post(handlers::process_event).get(handlers::get_events),
        )
        .route("/api/events/simple", post(handlers::process_simple_event))
        .route("/api/episodes", get(handlers::get_episodes))
        // Memories
        .route(
            "/api/memories/agent/:agent_id",
            get(handlers::get_agent_memories),
        )
        .route(
            "/api/memories/context",
            post(handlers::get_memories_by_context),
        )
        // Strategies
        .route(
            "/api/strategies/agent/:agent_id",
            get(handlers::get_agent_strategies),
        )
        .route(
            "/api/strategies/similar",
            post(handlers::get_similar_strategies),
        )
        .route("/api/suggestions", get(handlers::get_action_suggestions))
        // Graph visualization
        .route("/api/graph", get(handlers::get_graph))
        .route("/api/graph/context", get(handlers::get_graph_for_context))
        .route("/api/stats", get(handlers::get_stats))
        // Advanced analytics
        .route("/api/analytics", get(handlers::get_analytics))
        .route("/api/indexes", get(handlers::get_indexes))
        .route("/api/communities", get(handlers::get_communities))
        .route("/api/centrality", get(handlers::get_centrality))
        // Search
        .route("/api/search", post(handlers::search))
        // Semantic memory / claims
        .route("/api/claims", get(handlers::list_claims))
        .route("/api/claims/:id", get(handlers::get_claim))
        .route("/api/claims/search", post(handlers::search_claims))
        .route(
            "/api/embeddings/process",
            post(handlers::process_embeddings),
        )
        // Apply middleware
        .layer(CorsLayer::permissive())
        .with_state(state)
}
