// Error handling for MinnsDB REST API

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub details: Option<String>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum ApiError {
    Internal(String),
    BadRequest(String),
    /// 403 Forbidden — authenticated but not permitted (e.g. non-admin key
    /// on an admin-only endpoint).
    Forbidden(String),
    NotFound(String),
    NotImplemented(String),
    /// 503 Service Unavailable — returned when write lanes or read gate are exhausted.
    ServiceUnavailable(String),
    /// 429 Too Many Requests — returned when a rate or concurrency limit is hit.
    TooManyRequests(String),
    /// 504 Gateway Timeout — returned when LLM synthesis times out.
    GatewayTimeout(String),
}

/// Human-readable rendering for log lines and persisted JobState bodies.
/// Use this in preference to `{:?}` (Debug) anywhere the message lands
/// in user-visible output — Debug leaks the enum-variant name into the
/// payload, which is internal implementation detail.
impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, msg) = match self {
            ApiError::Internal(m) => ("internal", m),
            ApiError::BadRequest(m) => ("bad_request", m),
            ApiError::Forbidden(m) => ("forbidden", m),
            ApiError::NotFound(m) => ("not_found", m),
            ApiError::NotImplemented(m) => ("not_implemented", m),
            ApiError::ServiceUnavailable(m) => ("service_unavailable", m),
            ApiError::TooManyRequests(m) => ("too_many_requests", m),
            ApiError::GatewayTimeout(m) => ("gateway_timeout", m),
        };
        write!(f, "{}: {}", kind, msg)
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, details) = match self {
            ApiError::Internal(msg) => {
                tracing::error!("Internal error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal Server Error".to_string(),
                    None,
                )
            },
            ApiError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "Bad Request".to_string(),
                Some(msg),
            ),
            ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, "Forbidden".to_string(), Some(msg)),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "Not Found".to_string(), Some(msg)),
            ApiError::NotImplemented(msg) => (
                StatusCode::NOT_IMPLEMENTED,
                "Not Implemented".to_string(),
                Some(msg),
            ),
            ApiError::TooManyRequests(msg) => (
                StatusCode::TOO_MANY_REQUESTS,
                "Too Many Requests".to_string(),
                Some(msg),
            ),
            ApiError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Service Unavailable".to_string(),
                Some(msg),
            ),
            ApiError::GatewayTimeout(msg) => (
                StatusCode::GATEWAY_TIMEOUT,
                "Gateway Timeout".to_string(),
                Some(msg),
            ),
        };

        let body = Json(ErrorResponse {
            error: message,
            details,
        });

        (status, body).into_response()
    }
}

impl From<Box<dyn std::error::Error>> for ApiError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ApiError::Internal(err.to_string())
    }
}

impl From<crate::write_lanes::WriteError> for ApiError {
    fn from(err: crate::write_lanes::WriteError) -> Self {
        match err {
            crate::write_lanes::WriteError::LaneUnavailable(msg) => {
                ApiError::ServiceUnavailable(msg)
            },
            crate::write_lanes::WriteError::WorkerDropped => {
                ApiError::Internal("Write worker dropped".to_string())
            },
            crate::write_lanes::WriteError::OperationFailed(msg) => ApiError::Internal(msg),
        }
    }
}
