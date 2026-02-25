// Structured memory handlers

use crate::errors::ApiError;
use crate::models::{
    LedgerAppendRequest, PreferenceUpdateRequest, StateTransitionRequest, StructuredMemoryKeyQuery,
    StructuredMemoryRequest, TreeAddChildRequest,
};
use crate::state::AppState;
use axum::extract::{Path, Query, State};
use axum::Json;
use serde_json::json;
use tracing::info;

/// POST /api/structured-memory — upsert a structured memory
pub async fn upsert_structured_memory(
    State(state): State<AppState>,
    Json(request): Json<StructuredMemoryRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!("Upsert structured memory: key={}", request.key);
    let mut store = state.engine.structured_memory().write().await;
    store.upsert(&request.key, request.template);
    Ok(Json(json!({ "success": true, "key": request.key })))
}

/// GET /api/structured-memory/:key — get by key
pub async fn get_structured_memory(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.engine.structured_memory().read().await;
    match store.get(&key) {
        Some(template) => Ok(Json(json!({
            "key": key,
            "template": template,
        }))),
        None => Err(ApiError::NotFound(format!(
            "Structured memory not found: {}",
            key
        ))),
    }
}

/// GET /api/structured-memory — list keys (optional ?prefix=)
pub async fn list_structured_memory_keys(
    State(state): State<AppState>,
    Query(query): Query<StructuredMemoryKeyQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.engine.structured_memory().read().await;
    let prefix = query.prefix.as_deref().unwrap_or("");
    let keys: Vec<&str> = store.list_keys(prefix);
    Ok(Json(json!({ "keys": keys, "count": keys.len() })))
}

/// DELETE /api/structured-memory/:key — remove
pub async fn delete_structured_memory(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut store = state.engine.structured_memory().write().await;
    match store.remove(&key) {
        Some(_) => Ok(Json(json!({ "success": true, "key": key }))),
        None => Err(ApiError::NotFound(format!(
            "Structured memory not found: {}",
            key
        ))),
    }
}

/// POST /api/structured-memory/ledger/:key/append — append ledger entry
pub async fn ledger_append(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(request): Json<LedgerAppendRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !request.amount.is_finite() {
        return Err(ApiError::BadRequest(format!(
            "Invalid amount: {} (must be finite)",
            request.amount
        )));
    }
    let mut store = state.engine.structured_memory().write().await;
    let entry = agent_db_graph::LedgerEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
        amount: request.amount,
        description: request.description,
        direction: request.direction,
    };
    match store.ledger_append(&key, entry) {
        Ok(balance) => Ok(Json(json!({ "success": true, "balance": balance }))),
        Err(e) => Err(ApiError::BadRequest(e)),
    }
}

/// GET /api/structured-memory/ledger/:key/balance — get balance
pub async fn ledger_balance(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.engine.structured_memory().read().await;
    match store.ledger_balance(&key) {
        Some(balance) => Ok(Json(json!({ "key": key, "balance": balance }))),
        None => Err(ApiError::NotFound(format!(
            "Ledger not found or wrong type: {}",
            key
        ))),
    }
}

/// POST /api/structured-memory/state/:key/transition — transition state
pub async fn state_transition(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(request): Json<StateTransitionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut store = state.engine.structured_memory().write().await;
    match store.state_transition(&key, &request.new_state, &request.trigger, timestamp) {
        Ok(()) => Ok(Json(
            json!({ "success": true, "new_state": request.new_state }),
        )),
        Err(e) => Err(ApiError::BadRequest(e)),
    }
}

/// GET /api/structured-memory/state/:key/current — get current state
pub async fn state_current(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.engine.structured_memory().read().await;
    match store.state_current(&key) {
        Some(current) => Ok(Json(json!({ "key": key, "current_state": current }))),
        None => Err(ApiError::NotFound(format!(
            "State machine not found or wrong type: {}",
            key
        ))),
    }
}

/// POST /api/structured-memory/preference/:key/update — update preference
pub async fn preference_update(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(request): Json<PreferenceUpdateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut store = state.engine.structured_memory().write().await;
    match store.preference_update(&key, &request.item, request.rank, request.score) {
        Ok(()) => Ok(Json(json!({ "success": true }))),
        Err(e) => Err(ApiError::BadRequest(e)),
    }
}

/// POST /api/structured-memory/tree/:key/add-child — add tree child
pub async fn tree_add_child(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(request): Json<TreeAddChildRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut store = state.engine.structured_memory().write().await;
    match store.tree_add_child(&key, &request.parent, &request.child) {
        Ok(()) => Ok(Json(json!({ "success": true }))),
        Err(e) => Err(ApiError::BadRequest(e)),
    }
}
