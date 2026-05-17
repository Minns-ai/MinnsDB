//! Errors returned by [`VectorStore`](crate::VectorStore) implementations.

use thiserror::Error;

/// Errors a [`VectorStore`](crate::VectorStore) implementation may return.
///
/// `Backend` wraps any opaque error from the underlying engine (network failure,
/// authentication, gRPC stream issues, etc.); the message is preserved as a
/// string so callers can log it without depending on the backend crate. The
/// remaining variants are conditions the trait promises to signal in a uniform
/// way regardless of which backend is in use.
#[derive(Debug, Error)]
pub enum VectorError {
    /// The vector backend reported a failure (transport, auth, server-side, etc.).
    #[error("vector backend error: {0}")]
    Backend(String),

    /// A query or upsert provided a vector whose dimensionality does not match
    /// the collection's configured dimension.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The collection identified by name does not exist on the backend.
    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    /// The query is structurally invalid (e.g. `top_k == 0`).
    #[error("invalid query: {0}")]
    InvalidQuery(String),
}

/// A `Result` alias for [`VectorError`].
pub type VectorResult<T> = std::result::Result<T, VectorError>;
