//! ModuleRegistry: manage loaded WASM modules.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use agent_db_core::types::Timestamp;
use agent_db_tables::catalog::TableCatalog;

use crate::abi::ModuleDescriptor;
use crate::error::WasmError;
use crate::module::ModuleInstance;
use crate::permissions::PermissionSet;
use crate::runtime::WasmRuntime;
use crate::usage::{ModuleUsage, ModuleUsageCounters};

/// Persistent record stored in ReDB for each module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRecord {
    pub name: String,
    pub blob_hash: [u8; 32],
    pub permissions: Vec<String>,
    pub descriptor: Vec<u8>, // cached MessagePack descriptor
    pub uploaded_at: Timestamp,
    pub enabled: bool,
    pub group_id: u64,
    pub module_id: u64,
}

/// Registry of loaded WASM modules.
pub struct ModuleRegistry {
    /// Module instances keyed by name.
    modules: FxHashMap<String, ModuleInstance>,
    /// Persistent records keyed by name (for reload without WASM bytes).
    records: FxHashMap<String, ModuleRecord>,
    /// WASM blobs keyed by blake3 hash (content-addressed, deduplicated).
    blobs: FxHashMap<[u8; 32], Vec<u8>>,
    /// Next module ID.
    next_module_id: u64,
    /// Usage period start.
    period_start: Timestamp,
    /// Usage data to restore after recompilation (loaded from persistence).
    usage_to_restore: FxHashMap<String, ModuleUsage>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        ModuleRegistry {
            modules: FxHashMap::default(),
            records: FxHashMap::default(),
            blobs: FxHashMap::default(),
            next_module_id: 1,
            period_start: agent_db_core::types::current_timestamp(),
            usage_to_restore: FxHashMap::default(),
        }
    }

    /// Upload and register a new module.
    pub fn upload(
        &mut self,
        runtime: &WasmRuntime,
        name: String,
        wasm_bytes: Vec<u8>,
        permissions: Vec<String>,
        group_id: u64,
        table_catalog: Arc<RwLock<TableCatalog>>,
    ) -> Result<&ModuleInstance, WasmError> {
        if self.modules.contains_key(&name) {
            return Err(WasmError::ModuleAlreadyExists(name));
        }

        let module_id = self.next_module_id;
        self.next_module_id += 1;

        let perm_set = PermissionSet::new(permissions.clone());
        let instance = ModuleInstance::load(
            runtime,
            &wasm_bytes,
            perm_set,
            group_id,
            module_id,
            table_catalog,
        )?;

        let blob_hash = instance.blob_hash;
        let descriptor_bytes = crate::abi::to_msgpack(&instance.descriptor).unwrap_or_default();

        let record = ModuleRecord {
            name: name.clone(),
            blob_hash,
            permissions,
            descriptor: descriptor_bytes,
            uploaded_at: agent_db_core::types::current_timestamp(),
            enabled: true,
            group_id,
            module_id,
        };

        self.blobs.entry(blob_hash).or_insert(wasm_bytes);
        self.records.insert(name.clone(), record);
        self.modules.insert(name.clone(), instance);

        Ok(self.modules.get(&name).unwrap())
    }

    /// Unload and remove a module. Also cleans up the blob if no other module uses it.
    pub fn unload(&mut self, name: &str) -> Result<ModuleRecord, WasmError> {
        self.modules.remove(name);
        let record = self
            .records
            .remove(name)
            .ok_or_else(|| WasmError::ModuleNotFound(name.into()))?;

        // Clean up blob if no other module references it
        let blob_hash = record.blob_hash;
        let still_used = self.records.values().any(|r| r.blob_hash == blob_hash);
        if !still_used {
            self.blobs.remove(&blob_hash);
        }

        Ok(record)
    }

    /// Get a module instance by name.
    pub fn get(&self, name: &str) -> Option<&ModuleInstance> {
        self.modules.get(name)
    }

    /// Get a module record by name.
    pub fn get_record(&self, name: &str) -> Option<&ModuleRecord> {
        self.records.get(name)
    }

    /// List all module names.
    pub fn list(&self) -> Vec<&str> {
        self.records.keys().map(|k| k.as_str()).collect()
    }

    /// List all module records.
    pub fn list_records(&self) -> Vec<&ModuleRecord> {
        self.records.values().collect()
    }

    /// Enable a module.
    pub fn enable(&mut self, name: &str) -> Result<(), WasmError> {
        if let Some(instance) = self.modules.get_mut(name) {
            instance.enabled = true;
        }
        if let Some(record) = self.records.get_mut(name) {
            record.enabled = true;
            Ok(())
        } else {
            Err(WasmError::ModuleNotFound(name.into()))
        }
    }

    /// Disable a module.
    pub fn disable(&mut self, name: &str) -> Result<(), WasmError> {
        if let Some(instance) = self.modules.get_mut(name) {
            instance.enabled = false;
        }
        if let Some(record) = self.records.get_mut(name) {
            record.enabled = false;
            Ok(())
        } else {
            Err(WasmError::ModuleNotFound(name.into()))
        }
    }

    /// Get usage for a module.
    pub fn get_usage(&self, name: &str) -> Result<ModuleUsage, WasmError> {
        let instance = self
            .modules
            .get(name)
            .ok_or_else(|| WasmError::ModuleNotFound(name.into()))?;
        Ok(instance.usage.snapshot(name, self.period_start))
    }

    /// Reset usage for a module. Returns the previous period's usage.
    /// Updates period_start to now for the new billing period.
    pub fn reset_usage(&mut self, name: &str) -> Result<ModuleUsage, WasmError> {
        let instance = self
            .modules
            .get(name)
            .ok_or_else(|| WasmError::ModuleNotFound(name.into()))?;
        let previous = instance.usage.reset(name, self.period_start);
        self.period_start = agent_db_core::types::current_timestamp();
        Ok(previous)
    }

    /// Get all module records (for persistence).
    pub fn all_records(&self) -> &FxHashMap<String, ModuleRecord> {
        &self.records
    }

    /// Get all blobs (for persistence).
    pub fn all_blobs(&self) -> &FxHashMap<[u8; 32], Vec<u8>> {
        &self.blobs
    }

    /// Get all usage snapshots (for persistence).
    pub fn all_usage(&self) -> Vec<ModuleUsage> {
        self.modules
            .iter()
            .map(|(name, inst)| inst.usage.snapshot(name, self.period_start))
            .collect()
    }

    /// Set next_module_id (used during load from persistence).
    pub fn set_next_module_id(&mut self, id: u64) {
        self.next_module_id = id;
    }

    /// Insert a record without the instance (used during load before recompilation).
    pub fn insert_record(&mut self, record: ModuleRecord) {
        self.records.insert(record.name.clone(), record);
    }

    /// Insert a blob.
    pub fn insert_blob(&mut self, hash: [u8; 32], bytes: Vec<u8>) {
        self.blobs.insert(hash, bytes);
    }

    /// Set usage data to restore after recompilation.
    pub fn set_usage_to_restore(&mut self, usage: FxHashMap<String, ModuleUsage>) {
        self.usage_to_restore = usage;
    }

    /// Recompile and instantiate all modules from stored blobs.
    /// Restores usage counters from persisted data.
    pub fn recompile_all(
        &mut self,
        runtime: &WasmRuntime,
        table_catalog: Arc<RwLock<TableCatalog>>,
    ) -> Vec<(String, WasmError)> {
        let mut errors = Vec::new();
        let records: Vec<ModuleRecord> = self.records.values().cloned().collect();

        for record in records {
            if let Some(wasm_bytes) = self.blobs.get(&record.blob_hash) {
                let perm_set = PermissionSet::new(record.permissions.clone());
                match ModuleInstance::load(
                    runtime,
                    wasm_bytes,
                    perm_set,
                    record.group_id,
                    record.module_id,
                    table_catalog.clone(),
                ) {
                    Ok(mut instance) => {
                        instance.enabled = record.enabled;
                        // Restore usage counters if available
                        if let Some(usage) = self.usage_to_restore.get(&record.name) {
                            instance.usage.restore(usage);
                            if usage.period_start > 0 {
                                self.period_start = usage.period_start;
                            }
                        }
                        self.modules.insert(record.name.clone(), instance);
                    },
                    Err(e) => {
                        errors.push((record.name.clone(), e));
                    },
                }
            } else {
                // Blob missing — log but don't fail other modules
                errors.push((
                    record.name.clone(),
                    WasmError::PersistenceError("blob not found".into()),
                ));
            }
        }

        self.usage_to_restore.clear();
        errors
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let reg = ModuleRegistry::new();
        assert_eq!(reg.module_count(), 0);
        assert!(reg.list().is_empty());
    }
}
