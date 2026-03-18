#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("missing authorization header")]
    MissingHeader,
    #[error("invalid authorization header format — expected: Bearer mndb_<key>")]
    InvalidFormat,
    #[error("invalid API key")]
    InvalidKey,
    #[error("API key is disabled")]
    KeyDisabled,
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("persistence error: {0}")]
    PersistenceError(String),
}
