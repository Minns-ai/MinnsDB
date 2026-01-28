// EventGraphDB REST API Server
//
// Provides HTTP endpoints for the self-evolving agent database

mod config;
mod errors;
mod handlers;
mod models;
mod routes;
mod state;

use agent_db_graph::GraphEngine;
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("🚀 Starting EventGraphDB REST API Server");

    // Load configuration
    info!("Initializing GraphEngine with persistent storage...");
    let config = config::create_engine_config()?;

    // Initialize GraphEngine
    let engine = GraphEngine::with_config(config).await?;
    info!("✓ GraphEngine initialized with persistent storage at ./data/eventgraph.redb");
    info!("  Memory cache: 10,000 items (~20MB)");
    info!("  Strategy cache: 5,000 items (~15MB)");
    info!("  Redb cache: 128MB");

    // Create application state
    let state = state::AppState {
        engine: Arc::new(engine),
    };

    // Build router
    let app = routes::create_router(state);

    // Start server
    let addr = "0.0.0.0:3000";
    info!("🌐 Server listening on http://{}", addr);
    info!("📚 API documentation: http://{}/docs", addr);
    info!("❤️  Health check: http://{}/api/health", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
