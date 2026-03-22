// MinnsDB REST API Server
//
// Provides HTTP endpoints for the self-evolving agent database

mod auth_middleware;
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

use agent_db_graph::subscription::manager::SubscriptionManager;
use agent_db_graph::GraphEngine;
use agent_db_tables::catalog::TableCatalog;
use minns_auth::store::KeyStore;
use minns_wasm_runtime::registry::ModuleRegistry;
use minns_wasm_runtime::runtime::{RuntimeConfig, WasmRuntime};
use minns_wasm_runtime::scheduler::ScheduleRunner;
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

    // ── Table Catalog ──────────────────────────────────────────────────
    let table_catalog = if let Some(backend) = engine.redb_backend() {
        match agent_db_tables::persistence::load_catalog(backend) {
            Ok(catalog) => {
                let count = catalog.table_count();
                info!("Loaded {} temporal tables from ReDB", count);
                catalog
            },
            Err(e) => {
                tracing::warn!("Failed to load table catalog: {} — starting fresh", e);
                TableCatalog::new()
            },
        }
    } else {
        info!("No ReDB backend — temporal tables will be in-memory only");
        TableCatalog::new()
    };
    let table_catalog = Arc::new(tokio::sync::RwLock::new(table_catalog));

    // ── WASM Runtime ──────────────────────────────────────────────────
    let wasm_runtime = Arc::new(
        WasmRuntime::new(RuntimeConfig::default()).expect("failed to create WASM runtime"),
    );
    let module_registry = if let Some(backend) = engine.redb_backend() {
        match minns_wasm_runtime::persistence::load_registry(backend) {
            Ok(mut reg) => {
                let errors = reg.recompile_all(&wasm_runtime, table_catalog.clone(), Some(engine.clone()));
                for (name, err) in &errors {
                    tracing::warn!("Failed to load WASM module '{}': {}", name, err);
                }
                info!(
                    "Loaded {} WASM modules ({} failed)",
                    reg.module_count(),
                    errors.len()
                );
                reg
            },
            Err(e) => {
                tracing::warn!("Failed to load WASM registry: {} — starting fresh", e);
                ModuleRegistry::new()
            },
        }
    } else {
        ModuleRegistry::new()
    };
    let module_registry = Arc::new(tokio::sync::RwLock::new(module_registry));
    let mut schedule_runner_inner = ScheduleRunner::new();
    if let Some(backend) = engine.redb_backend() {
        match minns_wasm_runtime::persistence::load_schedules(backend) {
            Ok(schedules) => {
                let count = schedules.len();
                schedule_runner_inner.restore(schedules);
                if count > 0 {
                    info!("Loaded {} schedules from ReDB", count);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load schedules: {}", e);
            }
        }
    }
    let schedule_runner = Arc::new(tokio::sync::RwLock::new(schedule_runner_inner));
    info!("WASM agent runtime initialized");

    // ── Authentication ─────────────────────────────────────────────────
    // Auth is OFF by default. Set MINNS_AUTH_ENABLED=true to require API keys.
    let auth_enabled = std::env::var("MINNS_AUTH_ENABLED")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let mut key_store = if let Some(backend) = engine.redb_backend() {
        KeyStore::load(backend).unwrap_or_else(|e| {
            tracing::warn!("Failed to load API keys: {} — starting fresh", e);
            KeyStore::new()
        })
    } else {
        KeyStore::new()
    };

    if let Some(root_key) = key_store.init_root_key_if_empty() {
        info!("========================================");
        info!("  ROOT API KEY (save this — shown once):");
        info!("  {}", root_key);
        info!("========================================");
    } else {
        info!("API keys loaded: {} keys", key_store.count());
    }

    if auth_enabled {
        info!("Authentication ENABLED (MINNS_AUTH_ENABLED=true) — API key required on all endpoints except /api/health");
    } else {
        info!("Authentication disabled (default) — set MINNS_AUTH_ENABLED=true to require API keys");
    }

    let key_store = Arc::new(tokio::sync::RwLock::new(key_store));

    // Create application state
    let state = state::AppState {
        engine: engine.clone(),
        write_lanes: write_lanes.clone(),
        read_gate,
        seq_tracker,
        started_at: Instant::now(),
        subscription_manager,
        subscription_queries: Arc::new(Mutex::new(HashMap::new())),
        table_catalog: table_catalog.clone(),
        wasm_runtime: wasm_runtime.clone(),
        module_registry: module_registry.clone(),
        schedule_runner: schedule_runner.clone(),
        key_store: key_store.clone(),
        auth_enabled,
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

    // ── Post-shutdown: drain write lanes, persist tables, then flush engine ──
    info!("Server stopped accepting connections — draining write lanes");
    write_lanes.drain().await;

    // Persist table catalog
    if let Some(backend) = engine.redb_backend() {
        info!("Persisting temporal tables to ReDB...");
        let mut catalog = table_catalog.write().await;
        if let Err(e) = agent_db_tables::persistence::persist_catalog(backend, &mut catalog) {
            tracing::error!("Failed to persist table catalog: {}", e);
        } else {
            info!("Temporal tables persisted successfully");
        }
    }

    // Persist WASM module registry
    if let Some(backend) = engine.redb_backend() {
        info!("Persisting WASM module registry...");
        let registry = module_registry.read().await;
        if let Err(e) = minns_wasm_runtime::persistence::persist_registry(backend, &registry) {
            tracing::error!("Failed to persist WASM registry: {}", e);
        } else {
            info!("WASM module registry persisted");
        }
    }

    // Persist schedules
    if let Some(backend) = engine.redb_backend() {
        let runner = schedule_runner.read().await;
        if let Err(e) = minns_wasm_runtime::persistence::persist_schedules(backend, &runner) {
            tracing::error!("Failed to persist schedules: {}", e);
        } else {
            info!("Schedules persisted");
        }
    }

    // Persist API keys
    if let Some(backend) = engine.redb_backend() {
        let store = key_store.read().await;
        if let Err(e) = store.persist(backend) {
            tracing::error!("Failed to persist API keys: {}", e);
        } else {
            info!("API keys persisted");
        }
    }

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
