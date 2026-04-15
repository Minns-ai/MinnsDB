//! End-to-end regression test for keyword-named columns in MinnsQL.
//!
//! This test exercises the full HTTP → parser → planner → table executor
//! path with a column literally named `key` — the specific failure mode
//! that motivated the soft-keyword refactor. It verifies:
//!
//!   1. `CREATE TABLE` accepts `key` and `value` as column names
//!   2. `INSERT INTO` with a `(key, value)` column list works
//!   3. Qualified `FROM t WHERE t.key = "x" RETURN t.value` executes and
//!      returns the right row
//!   4. Bare `FROM t WHERE key = "x" RETURN value` executes and returns
//!      the right row (requires the planner scope-resolution fix)
//!   5. `UPDATE t SET value = "y" WHERE key = "x"` works
//!   6. `DELETE FROM t WHERE key = "x"` works
//!   7. Genuinely reserved keywords still need backtick quoting — the
//!      escape hatch covers the edge case

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

/// Helper: POST a query to /api/query and unwrap to OK + response body.
async fn run_query(app: &Router, sql: &str) -> Value {
    let (status, body) = post(app, "/api/query", json!({ "query": sql })).await;
    assert_eq!(status, StatusCode::OK, "query `{}` failed: {:?}", sql, body);
    body
}

/// Helper: extract `rows` from a table-query response. A row is a flat
/// array of `Value`s, one per projected column in the order of the RETURN
/// clause (see `QueryResponse::rows: Vec<Vec<Value>>` in the handler).
fn rows_of(resp: &Value) -> &Vec<Value> {
    resp["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("response missing `rows` array: {}", resp))
}

/// Helper: extract the first column of the first row as a string. Panics
/// with a readable message if the row is empty or the value is not a string.
fn first_col_string(resp: &Value) -> String {
    let rows = rows_of(resp);
    assert_eq!(rows.len(), 1, "expected exactly one row, got {:?}", rows);
    let cols = rows[0]
        .as_array()
        .expect("row should be an array of column values");
    cols.first()
        .expect("row should have at least one column")
        .as_str()
        .expect("first column should be a string")
        .to_string()
}

async fn seed_app_store(app: &Router) {
    run_query(
        app,
        "CREATE TABLE app_store (key String PRIMARY KEY, value String)",
    )
    .await;
    run_query(
        app,
        r#"INSERT INTO app_store (key, value) VALUES ("env_overrides", "{\"flag\":true}")"#,
    )
    .await;
    run_query(
        app,
        r#"INSERT INTO app_store (key, value) VALUES ("feature_flags", "{\"dark_mode\":true}")"#,
    )
    .await;
}

/// The exact query the user reported as failing. Qualified form.
#[tokio::test]
async fn user_query_qualified_form_works_end_to_end() {
    let app = build_test_app().await;
    seed_app_store(&app).await;

    let resp = run_query(
        &app,
        r#"FROM app_store WHERE app_store.key = "env_overrides" RETURN app_store.value"#,
    )
    .await;

    let value = first_col_string(&resp);
    assert!(
        value.contains("flag"),
        "expected row value to contain `flag`, got {:?}",
        value
    );
}

/// Bare form: no qualification. Previously failed at the planner with
/// "Unbound variable". Now works because the planner promotes bare
/// `Expr::Var("key")` to `RExpr::Property(app_store_slot, "key")` when
/// exactly one table is in scope.
#[tokio::test]
async fn user_query_bare_form_works_end_to_end() {
    let app = build_test_app().await;
    seed_app_store(&app).await;

    let resp = run_query(
        &app,
        r#"FROM app_store WHERE key = "env_overrides" RETURN value"#,
    )
    .await;

    let value = first_col_string(&resp);
    assert!(value.contains("flag"));
}

/// UPDATE with a keyword-named column in WHERE.
#[tokio::test]
async fn update_with_keyword_column_where() {
    let app = build_test_app().await;
    seed_app_store(&app).await;

    run_query(
        &app,
        r#"UPDATE app_store SET value = "new_value" WHERE app_store.key = "env_overrides""#,
    )
    .await;

    let resp = run_query(
        &app,
        r#"FROM app_store WHERE app_store.key = "env_overrides" RETURN app_store.value"#,
    )
    .await;
    let value = first_col_string(&resp);
    assert_eq!(value, "new_value");
}

/// DELETE with a keyword-named column in WHERE.
#[tokio::test]
async fn delete_with_keyword_column_where() {
    let app = build_test_app().await;
    seed_app_store(&app).await;

    run_query(
        &app,
        r#"DELETE FROM app_store WHERE app_store.key = "env_overrides""#,
    )
    .await;

    let resp = run_query(
        &app,
        r#"FROM app_store WHERE app_store.key = "env_overrides" RETURN app_store.value"#,
    )
    .await;
    let rows = rows_of(&resp);
    assert_eq!(rows.len(), 0, "row should be gone");

    // The other row should still be there.
    let resp2 = run_query(
        &app,
        r#"FROM app_store WHERE app_store.key = "feature_flags" RETURN app_store.value"#,
    )
    .await;
    assert_eq!(rows_of(&resp2).len(), 1);
}

/// Multi-table query with a bare column reference should fail with a
/// clear ambiguity error naming both tables. This pins the Phase 3
/// correctness rule — the planner never silently picks one column over
/// another.
#[tokio::test]
async fn multi_table_bare_column_is_ambiguity_error() {
    let app = build_test_app().await;

    run_query(
        &app,
        "CREATE TABLE orders (id Int64 PRIMARY KEY, customer_id Int64, amount Float64)",
    )
    .await;
    run_query(
        &app,
        "CREATE TABLE customers (id Int64 PRIMARY KEY, name String)",
    )
    .await;

    // Bare `id` is ambiguous: both tables have a column named `id`.
    let (status, body) = post(
        &app,
        "/api/query",
        json!({
            "query": "FROM orders JOIN customers ON orders.customer_id = customers.id RETURN id"
        }),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "expected 400, got {}: {:?}",
        status,
        body
    );
    // The error response shape is `{ "error": "Bad Request", "details": "..." }`.
    // The useful message lives in `details`.
    let details = body["details"]
        .as_str()
        .unwrap_or_else(|| panic!("response missing `details` field: {}", body));
    assert!(
        details.to_lowercase().contains("ambiguous"),
        "details should mention ambiguity: {}",
        details
    );
    assert!(
        details.contains("orders") && details.contains("customers"),
        "details should list both tables: {}",
        details
    );
}

/// Backtick escape hatch: a column literally named `where` (a genuinely
/// reserved keyword) must be reachable via backtick quoting.
#[tokio::test]
async fn backtick_escape_for_reserved_keyword_column() {
    let app = build_test_app().await;

    run_query(
        &app,
        "CREATE TABLE t (`where` String, id Int64 PRIMARY KEY)",
    )
    .await;
    run_query(&app, r#"INSERT INTO t VALUES ("home", 1)"#).await;

    let resp = run_query(&app, r#"FROM t WHERE t.`where` = "home" RETURN t.`where`"#).await;
    let rows = rows_of(&resp);
    assert_eq!(rows.len(), 1);
}

/// Bug 1 regression: an aliased single-table query with a bare column
/// reference must not trip the multi-table ambiguity check. Before the
/// fix this errored with "Ambiguous bare column reference `key` …
/// (tables in scope: s, app_store)" even though there's only one table.
#[tokio::test]
async fn aliased_single_table_bare_column_is_not_ambiguous() {
    let app = build_test_app().await;
    seed_app_store(&app).await;

    // Alias the table and reference a bare column. Plan must succeed and
    // row must come back.
    let resp = run_query(
        &app,
        r#"FROM app_store AS s WHERE key = "env_overrides" RETURN value"#,
    )
    .await;
    let value = first_col_string(&resp);
    assert!(value.contains("flag"));

    // Also verify that alias-qualified and real-name-qualified references
    // both still work after the fix — the secondary VarTable entry for the
    // real name is still there, just not in table_slots.
    let resp_alias = run_query(
        &app,
        r#"FROM app_store AS s WHERE s.key = "env_overrides" RETURN s.value"#,
    )
    .await;
    assert_eq!(first_col_string(&resp_alias), first_col_string(&resp));

    let resp_name = run_query(
        &app,
        r#"FROM app_store AS s WHERE app_store.key = "env_overrides" RETURN app_store.value"#,
    )
    .await;
    assert_eq!(first_col_string(&resp_name), first_col_string(&resp));
}
