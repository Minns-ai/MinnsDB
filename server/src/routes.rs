// Router setup for MinnsDB REST API

use crate::handlers;
use crate::state::AppState;
use axum::http::{header, Method};
use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post, put},
    Router,
};
use tower_http::cors::CorsLayer;

/// Build CORS layer from environment configuration.
///
/// Set `CORS_ALLOWED_ORIGINS` to a comma-separated list of allowed origins
/// (e.g. `https://my-app.com,https://staging.my-app.com`).
/// If unset or empty, falls back to permissive CORS for local development.
fn build_cors_layer() -> CorsLayer {
    let origins = std::env::var("CORS_ALLOWED_ORIGINS").unwrap_or_default();
    if origins.is_empty() {
        // Development default: permissive
        CorsLayer::permissive()
    } else {
        let allowed: Vec<_> = origins
            .split(',')
            .filter_map(|o| o.trim().parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(allowed)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::OPTIONS,
            ])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE, header::ACCEPT])
    }
}

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
        // Code intelligence
        .route(
            "/api/events/code-review",
            post(handlers::process_code_review),
        )
        .route("/api/events/code-file", post(handlers::process_code_file))
        .route("/api/code/search", post(handlers::search_code))
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
        .route("/api/graph/nodes/:id", delete(handlers::delete_node))
        .route("/api/graph/edges/:id", delete(handlers::delete_edge))
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
        // MinnsQL structured query
        .route("/api/query", post(handlers::minnsql_query))
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
        // Workflows
        .route(
            "/api/workflows",
            post(handlers::create_workflow).get(handlers::list_workflows),
        )
        .route(
            "/api/workflows/:id",
            get(handlers::get_workflow)
                .put(handlers::update_workflow)
                .delete(handlers::delete_workflow),
        )
        .route(
            "/api/workflows/:id/steps/:step_id/transition",
            post(handlers::workflow_step_transition),
        )
        .route(
            "/api/workflows/:id/feedback",
            post(handlers::workflow_feedback),
        )
        // Agent registry
        .route("/api/agents/register", post(handlers::register_agent))
        .route("/api/agents", get(handlers::list_agents))
        // Single message endpoint
        .route("/api/messages", post(handlers::accept_message))
        // Conversation Ingestion (batch)
        .route(
            "/api/conversations/ingest",
            post(handlers::ingest_conversation),
        )
        // Ontology
        .route(
            "/api/ontology/properties",
            get(handlers::list_ontology_properties),
        )
        .route("/api/ontology/upload", post(handlers::upload_ontology))
        .route("/api/ontology/discover", post(handlers::discover_ontology))
        .route(
            "/api/ontology/cascade-inference",
            post(handlers::run_cascade_inference),
        )
        .route(
            "/api/ontology/observations",
            get(handlers::list_ontology_observations),
        )
        .route(
            "/api/ontology/proposals",
            get(handlers::list_ontology_proposals),
        )
        .route(
            "/api/ontology/proposals/:id",
            get(handlers::get_ontology_proposal),
        )
        .route(
            "/api/ontology/proposals/:id/approve",
            post(handlers::approve_ontology_proposal),
        )
        .route(
            "/api/ontology/proposals/:id/reject",
            post(handlers::reject_ontology_proposal),
        )
        .route(
            "/api/ontology/stats",
            get(handlers::ontology_evolution_stats),
        )
        // Subscriptions (live queries)
        .route(
            "/api/subscriptions",
            post(handlers::create_subscription).get(handlers::list_subscriptions),
        )
        .route(
            "/api/subscriptions/:id",
            delete(handlers::delete_subscription),
        )
        .route(
            "/api/subscriptions/:id/poll",
            get(handlers::poll_subscription),
        )
        .route("/api/subscriptions/ws", get(handlers::ws_handler))
        // Graph import (bulk knowledge graph)
        .route("/api/graph/import", post(handlers::import_graph))
        // Temporal Tables
        .route(
            "/api/tables",
            post(handlers::tables::create_table).get(handlers::tables::list_tables),
        )
        .route("/api/tables/:name", delete(handlers::tables::drop_table))
        .route(
            "/api/tables/:name/schema",
            get(handlers::tables::get_schema),
        )
        .route(
            "/api/tables/:name/rows",
            post(handlers::tables::insert_rows).get(handlers::tables::scan_rows),
        )
        .route(
            "/api/tables/:name/rows/:id",
            put(handlers::tables::update_row).delete(handlers::tables::delete_row),
        )
        .route(
            "/api/tables/:name/by-node/:node_id",
            get(handlers::tables::rows_by_node),
        )
        .route(
            "/api/tables/:name/compact",
            post(handlers::tables::compact_table),
        )
        .route(
            "/api/tables/:name/stats",
            get(handlers::tables::table_stats),
        )
        // WASM Agent Modules
        .route(
            "/api/modules",
            post(handlers::modules::upload_module).get(handlers::modules::list_modules),
        )
        .route(
            "/api/modules/:name",
            get(handlers::modules::get_module).delete(handlers::modules::delete_module),
        )
        .route(
            "/api/modules/:name/call/:function",
            post(handlers::modules::call_function),
        )
        .route(
            "/api/modules/:name/enable",
            put(handlers::modules::enable_module),
        )
        .route(
            "/api/modules/:name/disable",
            put(handlers::modules::disable_module),
        )
        .route(
            "/api/modules/:name/usage",
            get(handlers::modules::get_usage),
        )
        .route(
            "/api/modules/:name/usage/reset",
            post(handlers::modules::reset_usage),
        )
        .route(
            "/api/modules/:name/schedules",
            get(handlers::modules::list_schedules).post(handlers::modules::create_schedule),
        )
        .route(
            "/api/modules/:name/schedules/:id",
            delete(handlers::modules::delete_schedule),
        )
        // API Key Management
        .route(
            "/api/keys",
            post(handlers::auth::create_key).get(handlers::auth::list_keys),
        )
        .route("/api/keys/:name", delete(handlers::auth::delete_key))
        // Admin: Export/Import
        .route("/api/admin/export", post(handlers::export_handler))
        .route("/api/admin/import", post(handlers::import_handler))
        // Apply middleware
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::auth_middleware::auth_layer,
        ))
        .layer(build_cors_layer())
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .with_state(state)
}
