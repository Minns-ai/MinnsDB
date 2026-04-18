#![allow(dead_code, unused_imports)]
//! End-to-end integration test for MinnsDB.
//!
//! Tests the full stack: tables, graph, MinnsQL, WASM modules, subscriptions,
//! temporal queries, concurrent requests — all through the HTTP API layer.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tower::ServiceExt;

// -- Helper: build the full app router with in-memory backend --

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
        ..GraphEngineConfig::default()
    };
    let engine = Arc::new(GraphEngine::with_config(config).await.unwrap());

    // Write lanes
    let write_lanes = Arc::new(minnsdb_server::write_lanes::WriteLanes::new(
        engine.clone(),
        2,
        64,
    ));
    let read_gate = Arc::new(minnsdb_server::read_gate::ReadGate::new(8));
    let seq_tracker = Arc::new(minnsdb_server::sequence::SequenceTracker::new());

    // Subscriptions
    let subscription_rx = {
        let mut inference = engine.inference().write().await;
        let graph = inference.graph_mut();
        graph.enable_subscriptions()
    };
    let subscription_manager = Arc::new(Mutex::new(SubscriptionManager::new(subscription_rx)));

    // Tables
    let table_catalog = Arc::new(tokio::sync::RwLock::new(TableCatalog::new()));

    // WASM runtime
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

// -- Helper: make a JSON POST request --

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

async fn put(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("PUT")
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

// ============================================================================
// THE TEST
// ============================================================================

#[tokio::test]
async fn test_end_to_end() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("info")
        .try_init()
        .ok();

    let app = build_test_app().await;
    tracing::info!("=== E2E test started ===");

    // ── 1. Health check ─────────────────────────────────────────────
    tracing::info!("--- Step 1: Health check ---");
    let (status, body) = get(&app, "/api/health").await;
    assert_eq!(status, StatusCode::OK, "health check failed: {:?}", body);
    assert_eq!(body["status"], "healthy");
    tracing::info!(
        "Health: OK, nodes={}, edges={}",
        body["node_count"],
        body["edge_count"]
    );

    // ── 2. Create tables via MinnsQL ────────────────────────────────
    tracing::info!("--- Step 2: Create tables via MinnsQL ---");

    let (status, body) = post(&app, "/api/query", json!({
        "query": "CREATE TABLE customers (id Int64 PRIMARY KEY, name String NOT NULL, region String)"
    })).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "create customers failed: {:?}",
        body
    );
    tracing::info!("Created table 'customers': {:?}", body);

    let (status, body) = post(&app, "/api/query", json!({
        "query": "CREATE TABLE orders (id Int64 PRIMARY KEY, customer_id Int64 NOT NULL, amount Float64, status String)"
    })).await;
    assert_eq!(status, StatusCode::OK, "create orders failed: {:?}", body);
    tracing::info!("Created table 'orders': {:?}", body);

    let (status, body) = post(&app, "/api/query", json!({
        "query": "CREATE TABLE shipments (id Int64 PRIMARY KEY, order_id Int64, tracking String, status String)"
    })).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "create shipments failed: {:?}",
        body
    );
    tracing::info!("Created table 'shipments': {:?}", body);

    // ── 3. Insert data via MinnsQL ──────────────────────────────────
    tracing::info!("--- Step 3: Insert data via MinnsQL ---");

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": r#"INSERT INTO customers VALUES (1, "Alice", "EU")"#
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "insert Alice failed: {:?}", body);

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": r#"INSERT INTO customers VALUES (2, "Bob", "US")"#
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "insert Bob failed: {:?}", body);

    let (status, body) = post(&app, "/api/query", json!({
        "query": r#"INSERT INTO customers (id, name, region) VALUES (3, "Charlie", "EU"), (4, "Diana", "US")"#
    })).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "insert Charlie/Diana failed: {:?}",
        body
    );
    tracing::info!("Inserted 4 customers: {:?}", body);

    // Insert orders
    for i in 1..=6 {
        let cid = ((i - 1) % 4) + 1;
        let amount = 50.0 + (i as f64) * 25.0;
        let status_str = if i <= 3 { "pending" } else { "shipped" };
        // Use .0 suffix for Float64 columns
        let q = format!(
            r#"INSERT INTO orders VALUES ({}, {}, {:.1}, "{}")"#,
            i, cid, amount, status_str
        );
        let (status, body) = post(&app, "/api/query", json!({ "query": q })).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "insert order {} failed: {:?}",
            i,
            body
        );
    }
    tracing::info!("Inserted 6 orders");

    // Insert shipments
    let (status, _) = post(&app, "/api/query", json!({
        "query": r#"INSERT INTO shipments VALUES (1, 4, "TRK-001", "delivered"), (2, 5, "TRK-002", "in_transit")"#
    })).await;
    assert_eq!(status, StatusCode::OK);
    tracing::info!("Inserted 2 shipments");

    // ── 4. Insert data via REST API ─────────────────────────────────
    tracing::info!("--- Step 4: Insert via REST API ---");

    let (status, body) = post(
        &app,
        "/api/tables/customers/rows",
        json!({
            "group_id": 0,
            "values": [5, "Eve", "APAC"]
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "REST insert failed: {:?}",
        body
    );
    tracing::info!("REST insert customer Eve: row_id={}", body["row_id"]);

    // ── 5. Query tables ─────────────────────────────────────────────
    tracing::info!("--- Step 5: Query tables ---");

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "FROM customers RETURN customers.id, customers.name, customers.region"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "scan customers failed: {:?}", body);
    let row_count = body["rows"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(
        row_count, 5,
        "expected 5 customers, got {}: {:?}",
        row_count, body
    );
    tracing::info!("Customers scan: {} rows", row_count);

    // Query with WHERE
    let (status, body) = post(&app, "/api/query", json!({
        "query": r#"FROM orders WHERE orders.status = "pending" RETURN orders.id, orders.customer_id, orders.amount ORDER BY orders.amount DESC"#
    })).await;
    assert_eq!(status, StatusCode::OK);
    let pending_count = body["rows"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(pending_count, 3, "expected 3 pending orders");
    tracing::info!("Pending orders: {} rows", pending_count);

    // ── 6. Table-to-table JOIN ──────────────────────────────────────
    tracing::info!("--- Step 6: Table-to-table JOIN ---");

    let (status, body) = post(&app, "/api/query", json!({
        "query": "FROM orders JOIN customers ON orders.customer_id = customers.id RETURN customers.name, orders.amount, orders.status LIMIT 100"
    })).await;
    assert_eq!(status, StatusCode::OK, "join failed: {:?}", body);
    let join_count = body["rows"].as_array().map(|a| a.len()).unwrap_or(0);
    assert!(
        join_count >= 6,
        "expected >= 6 join rows, got {}",
        join_count
    );
    tracing::info!("orders JOIN customers: {} rows", join_count);

    // ── 7. Update via MinnsQL ───────────────────────────────────────
    tracing::info!("--- Step 7: Update rows ---");

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": r#"UPDATE orders SET status = "shipped" WHERE id = 1"#
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "update failed: {:?}", body);
    tracing::info!("Updated order 1: {:?}", body);

    // Verify update
    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "FROM orders WHERE orders.id = 1 RETURN orders.id, orders.status"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    tracing::info!("Order 1 after update: {:?}", body["rows"]);

    // ── 8. Temporal query — WHEN ALL shows history ──────────────────
    tracing::info!("--- Step 8: Temporal queries ---");

    let (status, body) = post(&app, "/api/query", json!({
        "query": "FROM orders WHEN ALL RETURN orders.id, orders.status, orders.valid_from, orders.valid_until"
    })).await;
    assert_eq!(status, StatusCode::OK);
    let all_versions = body["rows"].as_array().map(|a| a.len()).unwrap_or(0);
    assert!(
        all_versions > 6,
        "expected > 6 versions (6 inserts + 1 update = 7+), got {}",
        all_versions
    );
    tracing::info!(
        "WHEN ALL: {} total versions (includes history)",
        all_versions
    );

    // ── 9. Ingest events into the graph ─────────────────────────────
    tracing::info!("--- Step 9: Ingest events into graph ---");

    for i in 1..=3 {
        let (status, body) = post(
            &app,
            "/api/events/simple",
            json!({
                "agent_id": 1,
                "agent_type": "test-agent",
                "session_id": 100 + i,
                "action": format!("test_action_{}", i),
                "data": {
                    "message": format!("Test event {}", i),
                    "value": i * 10
                },
                "success": true,
                "enable_semantic": false
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "event {} failed: {:?}", i, body);
        tracing::info!(
            "Event {}: success={}, nodes_created={}",
            i,
            body["success"],
            body["nodes_created"]
        );
    }

    // Check graph has nodes
    let (status, body) = get(&app, "/api/health").await;
    assert_eq!(status, StatusCode::OK);
    let node_count = body["node_count"].as_u64().unwrap_or(0);
    assert!(node_count > 0, "graph should have nodes after events");
    tracing::info!(
        "Graph after events: {} nodes, {} edges",
        body["node_count"],
        body["edge_count"]
    );

    // ── 10. Graph query via MinnsQL ─────────────────────────────────
    tracing::info!("--- Step 10: Graph queries ---");

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "MATCH (n) RETURN n.id, n.type LIMIT 10"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "graph query failed: {:?}", body);
    let graph_rows = body["rows"].as_array().map(|a| a.len()).unwrap_or(0);
    tracing::info!("Graph MATCH: {} nodes returned", graph_rows);

    // ── 11. Subscription ────────────────────────────────────────────
    tracing::info!("--- Step 11: Subscriptions ---");

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "SUBSCRIBE MATCH (n) RETURN n.id, n.type LIMIT 50"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "subscribe failed: {:?}", body);
    let sub_id = body["subscription_id"].as_u64();
    assert!(sub_id.is_some(), "expected subscription_id in response");
    tracing::info!(
        "Subscription created: id={}, strategy={}, initial_rows={}",
        sub_id.unwrap(),
        body["strategy"],
        body["rows"].as_array().map(|a| a.len()).unwrap_or(0)
    );

    // Submit more events to trigger subscription update
    let (status, _) = post(
        &app,
        "/api/events/simple",
        json!({
            "agent_id": 1,
            "agent_type": "test-agent",
            "session_id": 200,
            "action": "subscription_trigger",
            "data": {"trigger": true},
            "success": true,
            "enable_semantic": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Poll subscription
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let poll_uri = format!("/api/subscriptions/{}/poll", sub_id.unwrap());
    let (status, body) = get(&app, &poll_uri).await;
    tracing::info!("Subscription poll: status={}, body={:?}", status, body);

    // Clean up subscription
    let del_uri = format!("/api/subscriptions/{}", sub_id.unwrap());
    let (status, _) = delete(&app, &del_uri).await;
    assert_eq!(status, StatusCode::OK);
    tracing::info!("Subscription deleted");

    // ── 12. Table stats ─────────────────────────────────────────────
    tracing::info!("--- Step 12: Table stats ---");

    let (status, body) = get(&app, "/api/tables/orders/stats").await;
    assert_eq!(status, StatusCode::OK);
    tracing::info!(
        "Orders stats: active={}, versions={}, pages={}, gen={}",
        body["active_rows"],
        body["total_versions"],
        body["pages"],
        body["generation"]
    );

    // ── 13. Delete rows and table ───────────────────────────────────
    tracing::info!("--- Step 13: Delete rows and table ---");

    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "DELETE FROM shipments WHERE id = 1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "delete shipment failed: {:?}", body);
    tracing::info!("Deleted shipment 1: {:?}", body);

    // Drop the shipments table
    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "DROP TABLE shipments"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "drop table failed: {:?}", body);
    tracing::info!("Dropped table 'shipments'");

    // Verify it's gone
    let (status, _) = get(&app, "/api/tables/shipments/stats").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    tracing::info!("Confirmed: shipments table no longer exists");

    // ── 14. Parallel requests ───────────────────────────────────────
    tracing::info!("--- Step 14: Parallel requests ---");

    let start = Instant::now();
    let mut handles = Vec::new();

    // 10 concurrent reads
    for i in 0..10 {
        let app_clone = app.clone();
        handles.push(tokio::spawn(async move {
            let (status, body) = post(
                &app_clone,
                "/api/query",
                json!({ "query": "FROM customers RETURN customers.id, customers.name" }),
            )
            .await;
            tracing::debug!("Parallel read {}: status={}", i, status);
            assert_eq!(status, StatusCode::OK, "parallel read {} failed", i);
            body
        }));
    }

    // 5 concurrent writes
    for i in 0..5 {
        let app_clone = app.clone();
        handles.push(tokio::spawn(async move {
            let (status, body) = post(
                &app_clone,
                "/api/tables/orders/rows",
                json!({
                    "group_id": 0,
                    "values": [100 + i, 1, 999.99, "parallel"]
                }),
            )
            .await;
            tracing::debug!("Parallel write {}: status={}", i, status);
            assert_eq!(
                status,
                StatusCode::CREATED,
                "parallel write {} failed: {:?}",
                i,
                body
            );
            body
        }));
    }

    // 5 concurrent graph queries
    for i in 0..5 {
        let app_clone = app.clone();
        handles.push(tokio::spawn(async move {
            let (status, body) = post(
                &app_clone,
                "/api/query",
                json!({ "query": "MATCH (n) RETURN n.id LIMIT 5" }),
            )
            .await;
            tracing::debug!("Parallel graph query {}: status={}", i, status);
            assert_eq!(status, StatusCode::OK, "parallel graph query {} failed", i);
            body
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let failures = results.iter().filter(|r| r.is_err()).count();
    tracing::info!(
        "Parallel: {} requests in {:?} ({} failures)",
        results.len(),
        elapsed,
        failures
    );
    assert_eq!(failures, 0, "some parallel requests failed");

    // ── 15. Verify final state ──────────────────────────────────────
    tracing::info!("--- Step 15: Final state verification ---");

    let (status, body) = get(&app, "/api/health").await;
    assert_eq!(status, StatusCode::OK);
    tracing::info!(
        "Final health: nodes={}, edges={}, uptime={}s",
        body["node_count"],
        body["edge_count"],
        body["uptime_seconds"]
    );

    // Tables list should have 2 tables (customers + orders, shipments was dropped)
    let (status, body) = get(&app, "/api/tables").await;
    assert_eq!(status, StatusCode::OK);
    let table_count = body.as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(
        table_count, 2,
        "expected 2 tables (customers + orders), got {}",
        table_count
    );
    tracing::info!("Tables remaining: {}", table_count);

    // Orders should have original 6 + 5 parallel = 11 active (minus any we updated/deleted)
    let (status, body) = get(&app, "/api/tables/orders/stats").await;
    assert_eq!(status, StatusCode::OK);
    let active = body["active_rows"].as_u64().unwrap_or(0);
    assert!(active >= 11, "expected >= 11 active orders, got {}", active);
    tracing::info!(
        "Orders final: {} active rows, {} total versions",
        active,
        body["total_versions"]
    );

    tracing::info!("=== E2E test PASSED ===");
}
