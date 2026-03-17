// MinnsDB REST API Server
//
// Provides HTTP endpoints for the self-evolving agent database

mod config;
mod errors;
mod handlers;
mod models;
mod read_gate;
mod routes;
mod sequence;
mod state;
mod subscription_task;
mod write_lanes;

use agent_db_graph::GraphEngine;
use agent_db_graph::subscription::manager::SubscriptionManager;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("Starting MinnsDB REST API Server");

    // Load configuration
    info!("Initializing GraphEngine with persistent storage...");
    let config = config::create_engine_config()?;

    // Initialize GraphEngine
    let engine = GraphEngine::with_config(config).await?;
    info!("GraphEngine initialized with persistent storage at ./data/minns.redb");

    let engine = Arc::new(engine);

    // Start background maintenance loop (memory decay + strategy pruning)
    let _maintenance_handle = engine.start_maintenance_loop();

    // ── Write Lanes ──────────────────────────────────────────────────────
    let num_lanes = std::env::var("WRITE_LANE_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (num_cpus::get() / 2).clamp(2, 8));
    let lane_capacity = std::env::var("WRITE_LANE_CAPACITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let write_lanes = Arc::new(write_lanes::WriteLanes::new(
        engine.clone(),
        num_lanes,
        lane_capacity,
    ));
    info!(
        "Write lanes: {} lanes x {} capacity",
        write_lanes.num_lanes(),
        lane_capacity
    );

    // ── Read Gate ────────────────────────────────────────────────────────
    let read_permits = std::env::var("READ_GATE_PERMITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| num_cpus::get() * 2);
    let read_gate = Arc::new(read_gate::ReadGate::new(read_permits));
    info!("Read gate: {} permits", read_permits);

    // ── Sequence Tracker ─────────────────────────────────────────────────
    let seq_tracker = Arc::new(sequence::SequenceTracker::new());

    // ── Subscription Manager ─────────────────────────────────────────────
    let subscription_rx = {
        let mut inference = engine.inference().write().await;
        let graph = inference.graph_mut();
        graph.enable_subscriptions()
    };
    let subscription_manager = Arc::new(Mutex::new(SubscriptionManager::new(subscription_rx)));
    let _subscription_task = subscription_task::spawn(engine.clone(), subscription_manager.clone());
    info!("Subscription system enabled (background processing active)");

    // Create application state
    let state = state::AppState {
        engine: engine.clone(),
        write_lanes: write_lanes.clone(),
        read_gate,
        seq_tracker,
        started_at: Instant::now(),
        subscription_manager,
        subscription_queries: Arc::new(Mutex::new(HashMap::new())),
    };

    // Build router
    let app = routes::create_router(state);

    // Start server with configurable port
    let port = std::env::var("SERVER_PORT").unwrap_or_else(|_| "3000".to_string());
    let host = std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr = format!("{}:{}", host, port);

    info!("Server listening on http://{}", addr);
    info!("API documentation: http://{}/docs", addr);
    info!("Health check: http://{}/api/health", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Serve with graceful shutdown on SIGINT (Ctrl+C) / SIGTERM
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // ── Post-shutdown: drain write lanes, then flush engine to redb ──
    info!("Server stopped accepting connections — draining write lanes");
    write_lanes.drain().await;
    info!("Write lanes drained — flushing engine buffers");
    engine.shutdown().await;
    info!("Graceful shutdown complete");

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
