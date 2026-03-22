//! Axum middleware layer that enforces API key auth on all endpoints except /api/health.
//! Only active when AppState.auth_enabled is true.

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};

use crate::state::AppState;

/// Middleware function: check Authorization header on every request.
/// Skips /api/health (needed for load balancers and uptime monitors).
/// When auth_enabled is false, all requests pass through.
pub async fn auth_layer(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // If auth is disabled, pass through
    if !state.auth_enabled {
        return Ok(next.run(request).await);
    }

    // Health endpoint is always open
    let path = request.uri().path();
    if path == "/api/health" || path == "/" || path == "/docs" {
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
            Err(status)
        },
    }
}
