//! HTTP-level tests for DELETE /api/graph/nodes/:id and /api/graph/edges/:id.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tower::ServiceExt;

async fn build_test_app() -> Router {
    use agent_db_graph::integration::StorageBackend;
    use agent_db_graph::subscription::manager::SubscriptionManager;
    use agent_db_graph::GraphEngine;
    use agent_db_graph::GraphEngineConfig;
    use agent_db_tables::catalog::TableCatalog;
    use minns_wasm_runtime::registry::ModuleRegistry;
    use minns_wasm_runtime::runtime::{RuntimeConfig, WasmRuntime};
    use minns_wasm_runtime::scheduler::ScheduleRunner;

    let config = GraphEngineConfig {
        storage_backend: StorageBackend::InMemory,
        enable_semantic_memory: false,
        ner_storage_path: None,
        claim_storage_path: None,
        ..GraphEngineConfig::default()
    };
    let engine = Arc::new(GraphEngine::with_config(config).await.unwrap());

    let write_lanes = Arc::new(minnsdb_server::write_lanes::WriteLanes::new(
        engine.clone(),
        2,
        64,
    ));
    let read_gate = Arc::new(minnsdb_server::read_gate::ReadGate::new(8));
    let seq_tracker = Arc::new(minnsdb_server::sequence::SequenceTracker::new());

    let subscription_rx = {
        let mut inference = engine.inference().write().await;
        let graph = inference.graph_mut();
        graph.enable_subscriptions()
    };
    let subscription_manager = Arc::new(Mutex::new(SubscriptionManager::new(subscription_rx)));

    let table_catalog = Arc::new(tokio::sync::RwLock::new(TableCatalog::new()));
    let wasm_runtime = Arc::new(WasmRuntime::new(RuntimeConfig::default()).unwrap());
    let module_registry = Arc::new(tokio::sync::RwLock::new(ModuleRegistry::new()));
    let schedule_runner = Arc::new(tokio::sync::RwLock::new(ScheduleRunner::new()));

    let state = minnsdb_server::state::AppState {
        engine,
        write_lanes,
        read_gate,
        seq_tracker,
        started_at: Instant::now(),
        subscription_manager,
        subscription_queries: Arc::new(Mutex::new(HashMap::new())),
        table_catalog,
        wasm_runtime,
        module_registry,
        schedule_runner,
        key_store: Arc::new(tokio::sync::RwLock::new(minns_auth::store::KeyStore::new())),
        auth_enabled: false,
        export_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
    };

    minnsdb_server::routes::create_router(state)
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn delete(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
    (status, body)
}

/// Seed two concept nodes and one edge between them, returning
/// (alice_id, bob_id, edge_id).
async fn seed(app: &Router) -> (u64, u64, u64) {
    let (status, body) = post(
        app,
        "/api/graph/import",
        json!({
            "nodes": [
                { "name": "alice", "type": "concept" },
                { "name": "bob",   "type": "concept" },
            ],
            "edges": [
                { "source": "alice", "target": "bob", "type": "association", "label": "knows" },
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "import failed: {:?}", body);
    assert_eq!(body["nodes_created"], 2);
    assert_eq!(body["edges_created"], 1);

    let (status, graph) = get(app, "/api/graph").await;
    assert_eq!(status, StatusCode::OK);
    let nodes = graph["nodes"].as_array().expect("nodes array");
    let edges = graph["edges"].as_array().expect("edges array");

    let alice = nodes
        .iter()
        .find(|n| n["label"].as_str() == Some("alice"))
        .expect("alice node")["id"]
        .as_u64()
        .unwrap();
    let bob = nodes
        .iter()
        .find(|n| n["label"].as_str() == Some("bob"))
        .expect("bob node")["id"]
        .as_u64()
        .unwrap();
    let edge = edges
        .iter()
        .find(|e| e["from"].as_u64() == Some(alice) && e["to"].as_u64() == Some(bob))
        .expect("alice→bob edge")["id"]
        .as_u64()
        .unwrap();

    (alice, bob, edge)
}

#[tokio::test]
async fn delete_edge_endpoint_removes_edge_but_keeps_nodes() {
    let app = build_test_app().await;
    let (alice, bob, edge) = seed(&app).await;

    let (status, body) = delete(&app, &format!("/api/graph/edges/{}", edge)).await;
    assert_eq!(status, StatusCode::OK, "delete edge: {:?}", body);
    assert_eq!(body["deleted"], true);
    assert_eq!(body["edge_id"].as_u64(), Some(edge));

    let (_, graph) = get(&app, "/api/graph").await;
    let edges = graph["edges"].as_array().unwrap();
    assert!(
        edges.iter().all(|e| e["id"].as_u64() != Some(edge)),
        "edge should be gone: {:?}",
        edges
    );
    let node_ids: Vec<u64> = graph["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["id"].as_u64().unwrap())
        .collect();
    assert!(node_ids.contains(&alice));
    assert!(node_ids.contains(&bob));
}

#[tokio::test]
async fn delete_node_endpoint_removes_node_and_cascades_edges() {
    let app = build_test_app().await;
    let (alice, bob, edge) = seed(&app).await;

    let (status, body) = delete(&app, &format!("/api/graph/nodes/{}", alice)).await;
    assert_eq!(status, StatusCode::OK, "delete node: {:?}", body);
    assert_eq!(body["deleted"], true);
    assert_eq!(body["node_id"].as_u64(), Some(alice));

    let (_, graph) = get(&app, "/api/graph").await;
    let node_ids: Vec<u64> = graph["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["id"].as_u64().unwrap())
        .collect();
    assert!(!node_ids.contains(&alice), "alice should be gone");
    assert!(node_ids.contains(&bob), "bob should remain");

    let edges = graph["edges"].as_array().unwrap();
    assert!(
        edges.iter().all(|e| e["id"].as_u64() != Some(edge)),
        "incident edge should be cascaded: {:?}",
        edges
    );
}

#[tokio::test]
async fn delete_missing_node_returns_404() {
    let app = build_test_app().await;
    let (status, _body) = delete(&app, "/api/graph/nodes/999999").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_missing_edge_returns_404() {
    let app = build_test_app().await;
    let (status, _body) = delete(&app, "/api/graph/edges/999999").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
