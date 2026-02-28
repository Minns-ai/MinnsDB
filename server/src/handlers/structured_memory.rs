// Structured memory handlers

use crate::errors::ApiError;
use crate::models::{
    LedgerAppendRequest, PreferenceUpdateRequest, StateTransitionRequest, StructuredMemoryKeyQuery,
    StructuredMemoryRequest, TreeAddChildRequest,
};
use crate::state::AppState;
use crate::write_lanes::{WriteError, WriteJob};
use axum::extract::{Path, Query, State};
use axum::Json;
use serde_json::json;
use std::sync::Arc;
use tracing::info;

/// POST /api/structured-memory — upsert a structured memory
pub async fn upsert_structured_memory(
    State(state): State<AppState>,
    Json(request): Json<StructuredMemoryRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!("Upsert structured memory: key={}", request.key);

    let key = request.key.clone();
    let routing_key = hash_string_key(&key);
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let mut store = engine.structured_memory().write().await;
                    store.upsert(&request.key, request.template);
                    Ok(json!({ "success": true, "key": request.key }))
                })
            }),
            result_tx: tx,
        })
        .await?;

    Ok(Json(result))
}

/// GET /api/structured-memory/:key — get by key
pub async fn get_structured_memory(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    let routing_key = hash_string_key(&key);
    let key_clone = key.clone();
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let mut store = engine.structured_memory().write().await;
                    match store.remove(&key_clone) {
                        Some(_) => Ok(json!({ "success": true, "key": key_clone })),
                        None => Err(format!("Structured memory not found: {}", key_clone)),
                    }
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::NotFound(e)),
        Err(e) => Err(ApiError::from(e)),
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
    let routing_key = hash_string_key(&key);
    let key_clone = key.clone();
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let mut store = engine.structured_memory().write().await;
                    let entry = agent_db_graph::LedgerEntry {
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64,
                        amount: request.amount,
                        description: request.description,
                        direction: request.direction,
                    };
                    match store.ledger_append(&key_clone, entry) {
                        Ok(balance) => Ok(json!({ "success": true, "balance": balance })),
                        Err(e) => Err(e),
                    }
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::BadRequest(e)),
        Err(e) => Err(ApiError::from(e)),
    }
}

/// GET /api/structured-memory/ledger/:key/balance — get balance
pub async fn ledger_balance(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    let routing_key = hash_string_key(&key);
    let key_clone = key.clone();
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    let mut store = engine.structured_memory().write().await;
                    match store.state_transition(
                        &key_clone,
                        &request.new_state,
                        &request.trigger,
                        timestamp,
                    ) {
                        Ok(()) => Ok(json!({ "success": true, "new_state": request.new_state })),
                        Err(e) => Err(e),
                    }
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::BadRequest(e)),
        Err(e) => Err(ApiError::from(e)),
    }
}

/// GET /api/structured-memory/state/:key/current — get current state
pub async fn state_current(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    let routing_key = hash_string_key(&key);
    let key_clone = key.clone();
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let mut store = engine.structured_memory().write().await;
                    match store.preference_update(
                        &key_clone,
                        &request.item,
                        request.rank,
                        request.score,
                    ) {
                        Ok(()) => Ok(json!({ "success": true })),
                        Err(e) => Err(e),
                    }
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::BadRequest(e)),
        Err(e) => Err(ApiError::from(e)),
    }
}

/// POST /api/structured-memory/tree/:key/add-child — add tree child
pub async fn tree_add_child(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(request): Json<TreeAddChildRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let routing_key = hash_string_key(&key);
    let key_clone = key.clone();
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let mut store = engine.structured_memory().write().await;
                    match store.tree_add_child(&key_clone, &request.parent, &request.child) {
                        Ok(()) => Ok(json!({ "success": true })),
                        Err(e) => Err(e),
                    }
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::BadRequest(e)),
        Err(e) => Err(ApiError::from(e)),
    }
}

/// Hash a string key to a u64 for write lane routing.
fn hash_string_key(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}
