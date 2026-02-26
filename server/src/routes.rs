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
        .route(
            "/api/events/state-change",
            post(handlers::process_state_change_event),
        )
        .route(
            "/api/events/transaction",
            post(handlers::process_transaction_event),
        )
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
        .route("/api/graph/persist", post(handlers::persist_graph))
        .route("/api/stats", get(handlers::get_stats))
        .route(
            "/api/world-model/stats",
            get(handlers::get_world_model_stats),
        )
        // Planning
        .route(
            "/api/planning/strategies",
            post(handlers::generate_strategies),
        )
        .route("/api/planning/actions", post(handlers::generate_actions))
        .route("/api/planning/plan", post(handlers::plan_for_goal))
        .route("/api/planning/execute", post(handlers::start_execution))
        .route(
            "/api/planning/validate",
            post(handlers::validate_execution_event),
        )
        // Advanced analytics
        .route("/api/analytics", get(handlers::get_analytics))
        .route("/api/indexes", get(handlers::get_indexes))
        .route("/api/communities", get(handlers::get_communities))
        .route("/api/centrality", get(handlers::get_centrality))
        .route("/api/ppr", get(handlers::get_ppr))
        .route("/api/reachability", get(handlers::get_reachability))
        .route("/api/causal-path", get(handlers::get_causal_path))
        // Search
        .route("/api/search", post(handlers::search))
        // Natural language query
        .route("/api/nlq", post(handlers::nlq_query))
        // Semantic memory / claims
        .route("/api/claims", get(handlers::list_claims))
        .route("/api/claims/:id", get(handlers::get_claim))
        .route("/api/claims/search", post(handlers::search_claims))
        .route(
            "/api/embeddings/process",
            post(handlers::process_embeddings),
        )
        // Structured Memory
        .route(
            "/api/structured-memory",
            post(handlers::upsert_structured_memory).get(handlers::list_structured_memory_keys),
        )
        .route(
            "/api/structured-memory/:key",
            get(handlers::get_structured_memory).delete(handlers::delete_structured_memory),
        )
        .route(
            "/api/structured-memory/ledger/:key/append",
            post(handlers::ledger_append),
        )
        .route(
            "/api/structured-memory/ledger/:key/balance",
            get(handlers::ledger_balance),
        )
        .route(
            "/api/structured-memory/state/:key/transition",
            post(handlers::state_transition),
        )
        .route(
            "/api/structured-memory/state/:key/current",
            get(handlers::state_current),
        )
        .route(
            "/api/structured-memory/preference/:key/update",
            post(handlers::preference_update),
        )
        .route(
            "/api/structured-memory/tree/:key/add-child",
            post(handlers::tree_add_child),
        )
        // Conversation Ingestion
        .route(
            "/api/conversations/ingest",
            post(handlers::ingest_conversation),
        )
        .route(
            "/api/conversations/query",
            post(handlers::query_conversation),
        )
        // Admin: Export/Import
        .route("/api/admin/export", post(handlers::export_handler))
        .route("/api/admin/import", post(handlers::import_handler))
        // Apply middleware
        .layer(CorsLayer::permissive())
        .with_state(state)
}
