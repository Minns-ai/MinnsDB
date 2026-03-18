//! Permission system for WASM modules.
//!
//! Modules declare required permissions in their descriptor.
//! Admin approves at upload. Host enforces at runtime.

use crate::error::WasmError;
use serde::{Deserialize, Serialize};

/// Set of permissions granted to a module.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionSet {
    grants: Vec<String>,
}

impl PermissionSet {
    pub fn new(grants: Vec<String>) -> Self {
        PermissionSet { grants }
    }

    pub fn empty() -> Self {
        PermissionSet { grants: Vec::new() }
    }

    /// Check if a specific permission is granted.
    /// Supports wildcard matching: "table:*:read" matches "table:orders:read".
    pub fn check(&self, required: &str) -> Result<(), WasmError> {
        if self.has(required) {
            Ok(())
        } else {
            Err(WasmError::PermissionDenied(required.to_string()))
        }
    }

    /// Check if a permission is granted (without error).
    pub fn has(&self, required: &str) -> bool {
        for grant in &self.grants {
            if grant == required {
                return true;
            }
            // Wildcard matching: "table:*:read" matches "table:orders:read"
            if matches_wildcard(grant, required) {
                return true;
            }
        }
        false
    }

    /// Grant a permission.
    pub fn grant(&mut self, permission: String) {
        if !self.grants.contains(&permission) {
            self.grants.push(permission);
        }
    }

    /// Revoke a permission.
    pub fn revoke(&mut self, permission: &str) {
        self.grants.retain(|g| g != permission);
    }

    /// All granted permissions.
    pub fn grants(&self) -> &[String] {
        &self.grants
    }

    /// Check table read permission for a specific table.
    pub fn check_table_read(&self, table_name: &str) -> Result<(), WasmError> {
        self.check(&format!("table:{}:read", table_name))
    }

    /// Check table write permission for a specific table.
    pub fn check_table_write(&self, table_name: &str) -> Result<(), WasmError> {
        self.check(&format!("table:{}:write", table_name))
    }

    /// Check graph query permission.
    pub fn check_graph_query(&self) -> Result<(), WasmError> {
        self.check("graph:query")
    }

    /// Check graph subscribe permission.
    pub fn check_graph_subscribe(&self) -> Result<(), WasmError> {
        self.check("graph:subscribe")
    }

    /// Check HTTP fetch permission for a domain.
    pub fn check_http_fetch(&self, domain: &str) -> Result<(), WasmError> {
        self.check(&format!("http:fetch:{}", domain))
    }
}

/// Wildcard matching: each colon-separated segment is matched independently.
/// A `*` segment matches any single segment.
fn matches_wildcard(pattern: &str, target: &str) -> bool {
    let pat_parts: Vec<&str> = pattern.split(':').collect();
    let tgt_parts: Vec<&str> = target.split(':').collect();

    if pat_parts.len() != tgt_parts.len() {
        return false;
    }

    pat_parts
        .iter()
        .zip(tgt_parts.iter())
        .all(|(p, t)| *p == "*" || p == t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let ps = PermissionSet::new(vec!["table:orders:read".into()]);
        assert!(ps.has("table:orders:read"));
        assert!(!ps.has("table:orders:write"));
        assert!(!ps.has("table:customers:read"));
    }

    #[test]
    fn test_wildcard() {
        let ps = PermissionSet::new(vec!["table:*:read".into()]);
        assert!(ps.has("table:orders:read"));
        assert!(ps.has("table:customers:read"));
        assert!(!ps.has("table:orders:write"));
    }

    #[test]
    fn test_http_wildcard() {
        let ps = PermissionSet::new(vec!["http:fetch:*".into()]);
        assert!(ps.has("http:fetch:api.stripe.com"));
        assert!(ps.has("http:fetch:example.com"));
    }

    #[test]
    fn test_check_returns_error() {
        let ps = PermissionSet::empty();
        assert!(ps.check("table:orders:read").is_err());
    }

    #[test]
    fn test_grant_revoke() {
        let mut ps = PermissionSet::empty();
        ps.grant("graph:query".into());
        assert!(ps.has("graph:query"));
        ps.revoke("graph:query");
        assert!(!ps.has("graph:query"));
    }
}
