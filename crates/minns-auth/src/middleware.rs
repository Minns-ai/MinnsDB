//! Axum middleware for API key authentication.

use crate::error::AuthError;
use crate::store::KeyStore;

/// Authenticated identity extracted from the API key.
/// Inserted into request extensions by the auth middleware.
#[derive(Debug, Clone)]
pub struct AuthIdentity {
    /// The key name (e.g. "root", "my-app").
    pub key_name: String,
    /// The effective group_id for this request.
    pub group_id: u64,
    /// Whether this is an admin key.
    pub is_admin: bool,
    /// Granted permissions.
    pub permissions: Vec<String>,
}

impl AuthIdentity {
    /// Check if this identity has a specific permission.
    pub fn has_permission(&self, required: &str) -> bool {
        self.is_admin || self.permissions.iter().any(|p| p == required)
    }
}

/// Extract the API key from the Authorization header.
/// Expected format: `Authorization: Bearer mndb_<hex>`
pub fn extract_key_from_header(header_value: &str) -> Result<&str, AuthError> {
    let stripped = header_value
        .strip_prefix("Bearer ")
        .or_else(|| header_value.strip_prefix("bearer "))
        .ok_or(AuthError::InvalidFormat)?;
    Ok(stripped.trim())
}

/// Verify a request's API key and return the identity.
pub fn verify_request(
    store: &KeyStore,
    auth_header: Option<&str>,
) -> Result<AuthIdentity, AuthError> {
    let header = auth_header.ok_or(AuthError::MissingHeader)?;
    let raw_key = extract_key_from_header(header)?;
    let record = store.verify(raw_key)?;

    Ok(AuthIdentity {
        key_name: record.name.clone(),
        group_id: record.effective_group_id(),
        is_admin: record.has_permission(crate::key::permissions::ADMIN),
        permissions: record.permissions.clone(),
    })
}
