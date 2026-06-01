//! Axum middleware layer that enforces API key auth on all endpoints
//! except a small allow-list of public read-only telemetry surfaces:
//! `/api/health`, `/metrics`, `/docs`, and the root path. Only active
//! when `AppState.auth_enabled` is true.

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

use crate::state::AppState;

/// Middleware function: check Authorization header on every request.
/// Skips a small allow-list of public read-only surfaces:
/// - `/api/health` (load balancers, uptime monitors)
/// - `/metrics` (Prometheus scrapers; aggregate counts only, no user data)
/// - `/docs` (OpenAPI documentation, public by design)
/// - root path
///
/// When `auth_enabled` is false, all requests pass through.
pub async fn auth_layer(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, Response> {
    // If auth is disabled, pass through
    if !state.auth_enabled {
        return Ok(next.run(request).await);
    }

    // Public allow-list — read-only surfaces with no sensitive data.
    let path = request.uri().path().trim_end_matches('/');
    if path == "/api/health"
        || path == "/metrics"
        || path.is_empty()
        || path == "/docs"
    {
        return Ok(next.run(request).await);
    }

    // Extract and verify the API key
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let store = state.key_store.read().await;
    match minns_auth::middleware::verify_request(&store, auth_header) {
        Ok(_identity) => Ok(next.run(request).await),
        Err(e) => {
            let status = match &e {
                minns_auth::error::AuthError::MissingHeader => StatusCode::UNAUTHORIZED,
                minns_auth::error::AuthError::InvalidFormat => StatusCode::UNAUTHORIZED,
                minns_auth::error::AuthError::InvalidKey => StatusCode::UNAUTHORIZED,
                minns_auth::error::AuthError::KeyDisabled => StatusCode::FORBIDDEN,
                _ => StatusCode::UNAUTHORIZED,
            };
            tracing::warn!("Auth rejected: {} (path: {})", e, path);
            Err((status, Json(serde_json::json!({"error": "Unauthorized"}))).into_response())
        },
    }
}
