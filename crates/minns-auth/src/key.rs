//! API key generation, hashing, and verification.

use rand::Rng;
use serde::{Deserialize, Serialize};

use agent_db_core::types::Timestamp;

/// Prefix for all MinnsDB API keys.
pub const KEY_PREFIX: &str = "mndb_";

/// Length of the random portion in bytes (32 bytes = 64 hex chars).
const KEY_RANDOM_BYTES: usize = 32;

/// Stored API key record. The raw key is never stored — only the blake3 hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyRecord {
    /// blake3 hash of the raw key.
    pub key_hash: [u8; 32],
    /// Human-readable name (e.g. "root", "my-app", "staging").
    pub name: String,
    /// Group ID scope. None = admin (access all groups). Some(id) = scoped to one group.
    pub group_id: Option<u64>,
    /// Granted permissions.
    pub permissions: Vec<String>,
    /// When the key was created.
    pub created_at: Timestamp,
    /// Whether the key is active.
    pub enabled: bool,
}

/// Generate a new random API key. Returns (raw_key, hash).
/// The raw key is `mndb_` + 64 hex chars (32 random bytes).
pub fn generate_key() -> (String, [u8; 32]) {
    let mut rng = rand::thread_rng();
    let mut random_bytes = [0u8; KEY_RANDOM_BYTES];
    rng.fill(&mut random_bytes);

    let hex = hex_encode(&random_bytes);
    let raw_key = format!("{}{}", KEY_PREFIX, hex);
    let hash = hash_key(&raw_key);

    (raw_key, hash)
}

/// Hash a raw API key with blake3. This is what gets stored and compared.
pub fn hash_key(raw_key: &str) -> [u8; 32] {
    *blake3::hash(raw_key.as_bytes()).as_bytes()
}

/// Validate that a string looks like a valid API key format.
pub fn is_valid_format(key: &str) -> bool {
    key.starts_with(KEY_PREFIX)
        && key.len() == KEY_PREFIX.len() + KEY_RANDOM_BYTES * 2
        && key[KEY_PREFIX.len()..].chars().all(|c| c.is_ascii_hexdigit())
}

/// Known permission strings.
pub mod permissions {
    /// Full admin access — can manage keys, access all groups, all operations.
    pub const ADMIN: &str = "admin";
    /// Read tables and graph.
    pub const READ: &str = "read";
    /// Write to tables (insert, update, delete).
    pub const WRITE: &str = "write";
    /// Execute MinnsQL queries.
    pub const QUERY: &str = "query";
    /// Manage tables (create, drop, compact).
    pub const TABLES: &str = "tables";
    /// Manage WASM modules (upload, delete, call).
    pub const MODULES: &str = "modules";
    /// Ingest events and conversations.
    pub const INGEST: &str = "ingest";
    /// Manage subscriptions.
    pub const SUBSCRIBE: &str = "subscribe";
}

impl ApiKeyRecord {
    /// Check if this key has a specific permission.
    /// Admin keys have all permissions.
    pub fn has_permission(&self, required: &str) -> bool {
        self.permissions.iter().any(|p| p == permissions::ADMIN || p == required)
    }

    /// Check if this key can access a specific group_id.
    /// Admin keys (group_id = None) can access all groups.
    pub fn can_access_group(&self, target_group: u64) -> bool {
        match self.group_id {
            None => true, // admin — all groups
            Some(gid) => gid == target_group,
        }
    }

    /// Get the effective group_id for this key.
    /// Admin keys default to group 0 unless overridden.
    pub fn effective_group_id(&self) -> u64 {
        self.group_id.unwrap_or(0)
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_key() {
        let (key, hash) = generate_key();
        assert!(key.starts_with(KEY_PREFIX));
        assert_eq!(key.len(), KEY_PREFIX.len() + 64);
        assert!(is_valid_format(&key));
        // Hash should match
        assert_eq!(hash, hash_key(&key));
    }

    #[test]
    fn test_format_validation() {
        assert!(is_valid_format("mndb_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
        assert!(!is_valid_format("wrong_prefix"));
        assert!(!is_valid_format("mndb_tooshort"));
        assert!(!is_valid_format("mndb_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg")); // 'g' invalid
    }

    #[test]
    fn test_deterministic_hash() {
        let key = "mndb_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(hash_key(key), hash_key(key));
    }

    #[test]
    fn test_permissions() {
        let admin = ApiKeyRecord {
            key_hash: [0; 32],
            name: "root".into(),
            group_id: None,
            permissions: vec![permissions::ADMIN.into()],
            created_at: 0,
            enabled: true,
        };
        assert!(admin.has_permission(permissions::READ));
        assert!(admin.has_permission(permissions::WRITE));
        assert!(admin.has_permission("anything"));
        assert!(admin.can_access_group(42));

        let scoped = ApiKeyRecord {
            key_hash: [0; 32],
            name: "app".into(),
            group_id: Some(5),
            permissions: vec![permissions::READ.into(), permissions::WRITE.into()],
            created_at: 0,
            enabled: true,
        };
        assert!(scoped.has_permission(permissions::READ));
        assert!(!scoped.has_permission(permissions::MODULES));
        assert!(scoped.can_access_group(5));
        assert!(!scoped.can_access_group(6));
    }
}
