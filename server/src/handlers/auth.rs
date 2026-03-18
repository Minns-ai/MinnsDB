//! API key management endpoints and auth extraction helper.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use minns_auth::key::permissions;
use minns_auth::middleware::AuthIdentity;

use crate::state::AppState;

/// Extract and verify the auth identity from a request's Authorization header.
/// If auth is disabled (dev mode), returns a default admin identity.
pub async fn extract_auth(
    state: &AppState,
    auth_header: Option<&str>,
) -> Result<AuthIdentity, (StatusCode, Json<serde_json::Value>)> {
    if !state.auth_enabled {
        return Ok(AuthIdentity {
            key_name: "dev".into(),
            group_id: 0,
            is_admin: true,
            permissions: vec![permissions::ADMIN.into()],
        });
    }

    let store = state.key_store.read().await;
    minns_auth::middleware::verify_request(&store, auth_header).map_err(|e| {
        let status = match &e {
            minns_auth::error::AuthError::MissingHeader => StatusCode::UNAUTHORIZED,
            minns_auth::error::AuthError::InvalidFormat => StatusCode::UNAUTHORIZED,
            minns_auth::error::AuthError::InvalidKey => StatusCode::UNAUTHORIZED,
            minns_auth::error::AuthError::KeyDisabled => StatusCode::FORBIDDEN,
            minns_auth::error::AuthError::PermissionDenied(_) => StatusCode::FORBIDDEN,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(serde_json::json!({ "error": e.to_string() })))
    })
}

// -- Key management endpoints (admin only) --

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub name: String,
    pub group_id: Option<u64>,
    #[serde(default)]
    pub permissions: Vec<String>,
}

/// POST /api/keys — create a new API key (admin only)
pub async fn create_key(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CreateKeyRequest>,
) -> impl IntoResponse {
    let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
    let identity = match extract_auth(&state, auth_header).await {
        Ok(id) => id,
        Err(e) => return e,
    };

    if !identity.has_permission(permissions::ADMIN) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": "admin permission required to create keys" })),
        );
    }

    let mut store = state.key_store.write().await;
    let raw_key = store.create_key(req.name.clone(), req.group_id, req.permissions.clone());

    tracing::info!(
        "API key created: name='{}', group_id={:?}",
        req.name,
        req.group_id
    );

    (
        StatusCode::CREATED,
        Json(serde_json::json!({
            "key": raw_key,
            "name": req.name,
            "group_id": req.group_id,
            "permissions": req.permissions,
            "warning": "Save this key — it cannot be retrieved again."
        })),
    )
}

/// GET /api/keys — list all keys (admin only, no secrets exposed)
pub async fn list_keys(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
    let identity = match extract_auth(&state, auth_header).await {
        Ok(id) => id,
        Err(e) => return e,
    };

    if !identity.has_permission(permissions::ADMIN) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": "admin permission required" })),
        );
    }

    let store = state.key_store.read().await;
    let keys = store.list();
    (
        StatusCode::OK,
        Json(
            serde_json::to_value(keys)
                .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()})),
        ),
    )
}

/// DELETE /api/keys/:name — delete a key (admin only)
pub async fn delete_key(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
    let identity = match extract_auth(&state, auth_header).await {
        Ok(id) => id,
        Err(e) => return e,
    };

    if !identity.has_permission(permissions::ADMIN) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": "admin permission required" })),
        );
    }

    if name == "root" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "cannot delete the root key" })),
        );
    }

    let mut store = state.key_store.write().await;
    if store.delete(&name) {
        tracing::info!("API key deleted: name='{}'", name);
        (StatusCode::OK, Json(serde_json::json!({ "deleted": true })))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("key '{}' not found", name) })),
        )
    }
}
