//! ReDB persistence for WASM modules, schedules, and usage.

use agent_db_storage::RedbBackend;

use crate::error::WasmError;
use crate::registry::{ModuleRecord, ModuleRegistry};
use crate::usage::ModuleUsage;

const MODULE_REGISTRY: &str = "module_registry";
const MODULE_BLOBS: &str = "module_blobs";
const MODULE_USAGE: &str = "module_usage";
#[allow(dead_code)]
const MODULE_SCHEDULES: &str = "module_schedules";

/// Persist all module records, blobs, and usage to ReDB.
/// Also deletes records for modules that have been removed since last persist.
pub fn persist_registry(backend: &RedbBackend, registry: &ModuleRegistry) -> Result<(), WasmError> {
    // Delete records not in current registry (handles module removal)
    let existing = backend
        .scan_all_raw(MODULE_REGISTRY)
        .map_err(|e| WasmError::PersistenceError(e.to_string()))?;
    let current_names: std::collections::HashSet<&str> =
        registry.all_records().keys().map(|k| k.as_str()).collect();
    for (key, _) in &existing {
        let name = String::from_utf8_lossy(key);
        if !current_names.contains(name.as_ref()) {
            let _ = backend.delete(MODULE_REGISTRY, key.as_slice());
            let _ = backend.delete(MODULE_USAGE, key.as_slice());
        }
    }

    // Persist records
    for (name, record) in registry.all_records() {
        let key = name.as_bytes();
        let value =
            rmp_serde::to_vec(record).map_err(|e| WasmError::PersistenceError(e.to_string()))?;
        backend
            .put_raw(MODULE_REGISTRY, key, &value)
            .map_err(|e| WasmError::PersistenceError(e.to_string()))?;
    }

    // Persist blobs
    for (hash, bytes) in registry.all_blobs() {
        backend
            .put_raw(MODULE_BLOBS, hash.as_slice(), bytes)
            .map_err(|e| WasmError::PersistenceError(e.to_string()))?;
    }

    // Persist usage
    for usage in registry.all_usage() {
        let key = usage.module_name.as_bytes();
        let value =
            rmp_serde::to_vec(&usage).map_err(|e| WasmError::PersistenceError(e.to_string()))?;
        backend
            .put_raw(MODULE_USAGE, key, &value)
            .map_err(|e| WasmError::PersistenceError(e.to_string()))?;
    }

    Ok(())
}

/// Load module registry from ReDB.
pub fn load_registry(backend: &RedbBackend) -> Result<ModuleRegistry, WasmError> {
    let mut registry = ModuleRegistry::new();
    let mut max_module_id = 0u64;

    // Load records
    let record_entries = backend
        .scan_all_raw(MODULE_REGISTRY)
        .map_err(|e| WasmError::PersistenceError(e.to_string()))?;

    for (_key, value) in &record_entries {
        let record: ModuleRecord =
            rmp_serde::from_slice(value).map_err(|e| WasmError::PersistenceError(e.to_string()))?;
        if record.module_id >= max_module_id {
            max_module_id = record.module_id + 1;
        }
        registry.insert_record(record);
    }

    registry.set_next_module_id(max_module_id);

    // Load blobs
    let blob_entries = backend
        .scan_all_raw(MODULE_BLOBS)
        .map_err(|e| WasmError::PersistenceError(e.to_string()))?;

    for (key, value) in blob_entries {
        if key.len() == 32 {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&key);
            registry.insert_blob(hash, value);
        }
    }

    // Load usage — stored for restoring after recompilation
    let usage_entries = backend
        .scan_all_raw(MODULE_USAGE)
        .map_err(|e| WasmError::PersistenceError(e.to_string()))?;

    let mut usage_map: rustc_hash::FxHashMap<String, ModuleUsage> =
        rustc_hash::FxHashMap::default();
    for (_key, value) in &usage_entries {
        if let Ok(usage) = rmp_serde::from_slice::<ModuleUsage>(value) {
            usage_map.insert(usage.module_name.clone(), usage);
        }
    }

    registry.set_usage_to_restore(usage_map);

    Ok(registry)
}
