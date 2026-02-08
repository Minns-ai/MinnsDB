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
use std::time::Instant;
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

    let engine = Arc::new(engine);

    // Create application state
    let state = state::AppState {
        engine: engine.clone(),
        started_at: Instant::now(),
    };

    // Build router
    let app = routes::create_router(state);

    // Start server with configurable port
    let port = std::env::var("SERVER_PORT").unwrap_or_else(|_| "3000".to_string());
    let host = std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr = format!("{}:{}", host, port);

    info!("🌐 Server listening on http://{}", addr);
    info!("📚 API documentation: http://{}/docs", addr);
    info!("❤️  Health check: http://{}/api/health", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Serve with graceful shutdown on SIGINT (Ctrl+C) / SIGTERM
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // ── Post-shutdown: flush all in-flight work to redb ──
    info!("🛑 Server stopped accepting connections — flushing engine buffers");
    engine.shutdown().await;
    info!("✅ Graceful shutdown complete");

    Ok(())
}

/// Returns a future that resolves when the process receives a termination
/// signal.  On Unix this listens for SIGTERM (container orchestrators) and
/// SIGINT (Ctrl+C).  On Windows it listens for Ctrl+C only.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>(); // no SIGTERM on Windows

    tokio::select! {
        _ = ctrl_c => info!("Received SIGINT (Ctrl+C) — initiating graceful shutdown"),
        _ = terminate => info!("Received SIGTERM — initiating graceful shutdown"),
    }
}
