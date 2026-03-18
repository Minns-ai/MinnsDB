//! API key store — in-memory with ReDB persistence.

use rustc_hash::FxHashMap;

use agent_db_core::types::current_timestamp;
use agent_db_storage::RedbBackend;

use crate::error::AuthError;
use crate::key::{self, ApiKeyRecord};

const REDB_TABLE: &str = "api_keys";

/// Manages API keys. Thread-safe via external RwLock (caller responsibility).
pub struct KeyStore {
    /// Keys indexed by hash for O(1) lookup.
    keys: FxHashMap<[u8; 32], ApiKeyRecord>,
}

impl KeyStore {
    pub fn new() -> Self {
        KeyStore {
            keys: FxHashMap::default(),
        }
    }

    /// Create the root admin key if no keys exist.
    /// Returns the raw key string (printed once, never stored).
    /// Returns None if keys already exist.
    pub fn init_root_key_if_empty(&mut self) -> Option<String> {
        if !self.keys.is_empty() {
            return None;
        }

        let (raw_key, hash) = key::generate_key();
        let record = ApiKeyRecord {
            key_hash: hash,
            name: "root".into(),
            group_id: None, // admin — all groups
            permissions: vec![key::permissions::ADMIN.into()],
            created_at: current_timestamp(),
            enabled: true,
        };
        self.keys.insert(hash, record);

        Some(raw_key)
    }

    /// Create a new API key. Returns the raw key (caller must save it).
    pub fn create_key(
        &mut self,
        name: String,
        group_id: Option<u64>,
        permissions: Vec<String>,
    ) -> String {
        let (raw_key, hash) = key::generate_key();
        let record = ApiKeyRecord {
            key_hash: hash,
            name,
            group_id,
            permissions,
            created_at: current_timestamp(),
            enabled: true,
        };
        self.keys.insert(hash, record);
        raw_key
    }

    /// Verify a raw API key. Returns the record if valid and enabled.
    pub fn verify(&self, raw_key: &str) -> Result<&ApiKeyRecord, AuthError> {
        if !key::is_valid_format(raw_key) {
            return Err(AuthError::InvalidFormat);
        }

        let hash = key::hash_key(raw_key);
        let record = self.keys.get(&hash).ok_or(AuthError::InvalidKey)?;

        if !record.enabled {
            return Err(AuthError::KeyDisabled);
        }

        Ok(record)
    }

    /// Disable a key by name. Returns true if found.
    pub fn disable(&mut self, name: &str) -> bool {
        for record in self.keys.values_mut() {
            if record.name == name {
                record.enabled = false;
                return true;
            }
        }
        false
    }

    /// Enable a key by name. Returns true if found.
    pub fn enable(&mut self, name: &str) -> bool {
        for record in self.keys.values_mut() {
            if record.name == name {
                record.enabled = true;
                return true;
            }
        }
        false
    }

    /// Delete a key by name. Returns true if found.
    pub fn delete(&mut self, name: &str) -> bool {
        let hash = self
            .keys
            .iter()
            .find(|(_, r)| r.name == name)
            .map(|(h, _)| *h);
        if let Some(h) = hash {
            self.keys.remove(&h);
            true
        } else {
            false
        }
    }

    /// List all keys (without hashes — for API responses).
    pub fn list(&self) -> Vec<KeyInfo> {
        self.keys
            .values()
            .map(|r| KeyInfo {
                name: r.name.clone(),
                group_id: r.group_id,
                permissions: r.permissions.clone(),
                created_at: r.created_at,
                enabled: r.enabled,
            })
            .collect()
    }

    /// Number of keys.
    pub fn count(&self) -> usize {
        self.keys.len()
    }

    /// Persist all keys to ReDB.
    pub fn persist(&self, backend: &RedbBackend) -> Result<(), AuthError> {
        for (hash, record) in &self.keys {
            let value = rmp_serde::to_vec(record)
                .map_err(|e| AuthError::PersistenceError(e.to_string()))?;
            backend
                .put_raw(REDB_TABLE, hash.as_slice(), &value)
                .map_err(|e| AuthError::PersistenceError(e.to_string()))?;
        }
        Ok(())
    }

    /// Load keys from ReDB.
    pub fn load(backend: &RedbBackend) -> Result<Self, AuthError> {
        let entries = backend
            .scan_all_raw(REDB_TABLE)
            .map_err(|e| AuthError::PersistenceError(e.to_string()))?;

        let mut keys = FxHashMap::default();
        for (key_bytes, value) in &entries {
            if key_bytes.len() != 32 {
                continue;
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(key_bytes);
            if let Ok(record) = rmp_serde::from_slice::<ApiKeyRecord>(value) {
                keys.insert(hash, record);
            }
        }

        Ok(KeyStore { keys })
    }
}

/// Key info for API responses (no hash exposed).
#[derive(Debug, Clone, serde::Serialize)]
pub struct KeyInfo {
    pub name: String,
    pub group_id: Option<u64>,
    pub permissions: Vec<String>,
    pub created_at: u64,
    pub enabled: bool,
}

impl Default for KeyStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_root_key() {
        let mut store = KeyStore::new();
        let root_key = store.init_root_key_if_empty();
        assert!(root_key.is_some());
        assert_eq!(store.count(), 1);

        // Second call returns None
        assert!(store.init_root_key_if_empty().is_none());
    }

    #[test]
    fn test_create_and_verify() {
        let mut store = KeyStore::new();
        let raw = store.create_key("test-app".into(), Some(5), vec!["read".into()]);

        let record = store.verify(&raw).unwrap();
        assert_eq!(record.name, "test-app");
        assert_eq!(record.group_id, Some(5));
        assert!(record.has_permission("read"));
    }

    #[test]
    fn test_invalid_key() {
        let store = KeyStore::new();
        assert!(store.verify("bad_key").is_err());
        assert!(store
            .verify("mndb_0000000000000000000000000000000000000000000000000000000000000000")
            .is_err());
    }

    #[test]
    fn test_disable_enable() {
        let mut store = KeyStore::new();
        let raw = store.create_key("app".into(), None, vec!["admin".into()]);
        assert!(store.verify(&raw).is_ok());

        store.disable("app");
        assert!(matches!(store.verify(&raw), Err(AuthError::KeyDisabled)));

        store.enable("app");
        assert!(store.verify(&raw).is_ok());
    }

    #[test]
    fn test_delete() {
        let mut store = KeyStore::new();
        let raw = store.create_key("temp".into(), None, vec![]);
        assert_eq!(store.count(), 1);

        store.delete("temp");
        assert_eq!(store.count(), 0);
        assert!(store.verify(&raw).is_err());
    }

    #[test]
    fn test_list() {
        let mut store = KeyStore::new();
        store.create_key("a".into(), Some(1), vec!["read".into()]);
        store.create_key("b".into(), Some(2), vec!["write".into()]);

        let list = store.list();
        assert_eq!(list.len(), 2);
    }
}
