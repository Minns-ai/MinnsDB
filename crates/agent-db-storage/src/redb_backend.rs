//! redb backend for persistent storage
//!
//! This module provides the core redb infrastructure with 20 tables
//! for storing episodes, memories, strategies, transitions, telemetry, and graph structure.
//!
//! ## Why redb?
//! - Pure Rust (no C++ compiler needed)
//! - 2-3x faster than RocksDB for many workloads
//! - Better memory safety guarantees
//! - Simpler API, easier to reason about
//! - ACID transactions with MVCC
//!
//! ## Architecture
//! - Catalog tables: episode_catalog, partition_map
//! - Memory tables: memory_records, mem_by_bucket, mem_by_context_hash, mem_feature_postings
//! - Strategy tables: strategy_records, strategy_by_bucket, strategy_by_signature, strategy_feature_postings
//! - Learning tables: transition_stats, motif_stats
//! - Telemetry tables: decision_trace, outcome_signals
//! - Operational tables: id_allocator, schema_versions

use crate::{StorageError, StorageResult};
use redb::{Database, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Error type for `for_each_prefix_raw` that distinguishes storage errors from
/// callback-initiated early exits.
#[derive(Debug)]
pub enum ForEachError<E> {
    Storage(StorageError),
    Callback(E),
}

/// Table definitions (16 total)
/// Keys and values are stored as bytes (MessagePack serialization)
mod table_defs {
    use super::*;

    // Catalogs
    pub const EPISODE_CATALOG: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("episode_catalog");
    pub const PARTITION_MAP: TableDefinition<&[u8], &[u8]> = TableDefinition::new("partition_map");

    // Memory store
    pub const MEMORY_RECORDS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("memory_records");
    pub const MEM_BY_BUCKET: TableDefinition<&[u8], &[u8]> = TableDefinition::new("mem_by_bucket");
    pub const MEM_BY_CONTEXT_HASH: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("mem_by_context_hash");
    pub const MEM_BY_GOAL_BUCKET: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("mem_by_goal_bucket");
    pub const MEM_FEATURE_POSTINGS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("mem_feature_postings");

    // Strategy store
    pub const STRATEGY_RECORDS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("strategy_records");
    pub const STRATEGY_BY_BUCKET: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("strategy_by_bucket");
    pub const STRATEGY_BY_SIGNATURE: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("strategy_by_signature");
    pub const STRATEGY_FEATURE_POSTINGS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("strategy_feature_postings");

    // Learning stats
    pub const TRANSITION_STATS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("transition_stats");
    pub const MOTIF_STATS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("motif_stats");

    // Telemetry
    pub const DECISION_TRACE: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("decision_trace");
    pub const OUTCOME_SIGNALS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("outcome_signals");

    // Operational
    pub const ID_ALLOCATOR: TableDefinition<&[u8], &[u8]> = TableDefinition::new("id_allocator");
    pub const SCHEMA_VERSIONS: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("schema_versions");

    // Graph persistence (Phase 5B)
    pub const GRAPH_NODES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("graph_nodes");
    pub const GRAPH_ADJACENCY: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("graph_adjacency");
    pub const GRAPH_EDGES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("graph_edges");

    // World model (Phase 3)
    pub const WORLD_MODEL: TableDefinition<&[u8], &[u8]> = TableDefinition::new("world_model");

    // Temporal tables
    /// Key: [table_id: 8B BE] -> msgpack(TableSchema)
    pub const TABLE_SCHEMAS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("table_schemas");
    /// Key: [table_id: 8B BE][page_id: 4B BE] -> raw [u8; 8192]
    pub const TABLE_PAGES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("table_pages");
    /// Key: [table_id: 8B BE] -> msgpack(TableMeta)
    pub const TABLE_META: TableDefinition<&[u8], &[u8]> = TableDefinition::new("table_meta");

    // WASM agent modules
    pub const MODULE_REGISTRY: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("module_registry");
    pub const MODULE_BLOBS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("module_blobs");
    pub const MODULE_USAGE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("module_usage");
    pub const MODULE_SCHEDULES: TableDefinition<&[u8], &[u8]> =
        TableDefinition::new("module_schedules");

    // Authentication
    pub const API_KEYS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("api_keys");
}

/// Table names (for string-based access)
pub mod table_names {
    pub const EPISODE_CATALOG: &str = "episode_catalog";
    pub const PARTITION_MAP: &str = "partition_map";
    pub const MEMORY_RECORDS: &str = "memory_records";
    pub const MEM_BY_BUCKET: &str = "mem_by_bucket";
    pub const MEM_BY_CONTEXT_HASH: &str = "mem_by_context_hash";
    pub const MEM_BY_GOAL_BUCKET: &str = "mem_by_goal_bucket";
    pub const MEM_FEATURE_POSTINGS: &str = "mem_feature_postings";
    pub const STRATEGY_RECORDS: &str = "strategy_records";
    pub const STRATEGY_BY_BUCKET: &str = "strategy_by_bucket";
    pub const STRATEGY_BY_SIGNATURE: &str = "strategy_by_signature";
    pub const STRATEGY_FEATURE_POSTINGS: &str = "strategy_feature_postings";
    pub const TRANSITION_STATS: &str = "transition_stats";
    pub const MOTIF_STATS: &str = "motif_stats";
    pub const DECISION_TRACE: &str = "decision_trace";
    pub const OUTCOME_SIGNALS: &str = "outcome_signals";
    pub const ID_ALLOCATOR: &str = "id_allocator";
    pub const SCHEMA_VERSIONS: &str = "schema_versions";
    pub const GRAPH_NODES: &str = "graph_nodes";
    pub const GRAPH_ADJACENCY: &str = "graph_adjacency";
    pub const GRAPH_EDGES: &str = "graph_edges";
    pub const WORLD_MODEL: &str = "world_model";
    pub const TABLE_SCHEMAS: &str = "table_schemas";
    pub const TABLE_PAGES: &str = "table_pages";
    pub const TABLE_META: &str = "table_meta";
    pub const MODULE_REGISTRY: &str = "module_registry";
    pub const MODULE_BLOBS: &str = "module_blobs";
    pub const MODULE_USAGE: &str = "module_usage";
    pub const MODULE_SCHEDULES: &str = "module_schedules";
    pub const API_KEYS: &str = "api_keys";
}

/// redb backend configuration
#[derive(Debug, Clone)]
pub struct RedbConfig {
    /// Database file path
    pub data_path: PathBuf,

    /// Cache size in bytes (default: 256 MB)
    pub cache_size_bytes: usize,

    /// Enable repair on open (slow, only for recovery)
    pub repair_on_open: bool,
}

impl Default for RedbConfig {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("./data/minns.redb"),
            cache_size_bytes: 256 * 1024 * 1024, // 256 MB
            repair_on_open: false,
        }
    }
}

/// redb backend with 16 tables
pub struct RedbBackend {
    db: Arc<Database>,
    config: RedbConfig,
}

impl RedbBackend {
    /// Open or create redb database with all tables
    pub fn open(config: RedbConfig) -> StorageResult<Self> {
        // Create parent directory
        if let Some(parent) = config.data_path.parent() {
            std::fs::create_dir_all(parent).map_err(StorageError::Io)?;
        }

        // Open database
        let db = if config.repair_on_open {
            Database::builder()
                .set_cache_size(config.cache_size_bytes)
                .set_repair_callback(|_progress| {
                    tracing::info!("Repairing database...");
                })
                .open(&config.data_path)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?
        } else {
            Database::builder()
                .set_cache_size(config.cache_size_bytes)
                .create(&config.data_path)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?
        };

        // Create all tables (idempotent)
        let write_txn = db
            .begin_write()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        {
            // Catalogs
            let _ = write_txn
                .open_table(table_defs::EPISODE_CATALOG)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::PARTITION_MAP)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Memory store
            let _ = write_txn
                .open_table(table_defs::MEMORY_RECORDS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MEM_BY_BUCKET)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MEM_BY_CONTEXT_HASH)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MEM_BY_GOAL_BUCKET)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MEM_FEATURE_POSTINGS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Strategy store
            let _ = write_txn
                .open_table(table_defs::STRATEGY_RECORDS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::STRATEGY_BY_BUCKET)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::STRATEGY_BY_SIGNATURE)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::STRATEGY_FEATURE_POSTINGS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Learning stats
            let _ = write_txn
                .open_table(table_defs::TRANSITION_STATS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MOTIF_STATS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Telemetry
            let _ = write_txn
                .open_table(table_defs::DECISION_TRACE)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::OUTCOME_SIGNALS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Operational
            let _ = write_txn
                .open_table(table_defs::ID_ALLOCATOR)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::SCHEMA_VERSIONS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Graph persistence (Phase 5B)
            let _ = write_txn
                .open_table(table_defs::GRAPH_NODES)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::GRAPH_ADJACENCY)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::GRAPH_EDGES)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // World model (Phase 3)
            let _ = write_txn
                .open_table(table_defs::WORLD_MODEL)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // WASM agent modules
            let _ = write_txn
                .open_table(table_defs::MODULE_REGISTRY)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MODULE_BLOBS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MODULE_USAGE)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::MODULE_SCHEDULES)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Authentication
            let _ = write_txn
                .open_table(table_defs::API_KEYS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Temporal tables
            let _ = write_txn
                .open_table(table_defs::TABLE_SCHEMAS)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::TABLE_PAGES)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
            let _ = write_txn
                .open_table(table_defs::TABLE_META)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        }

        write_txn
            .commit()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        tracing::info!(
            "redb opened at {:?} with 20 tables, cache size: {} MB",
            config.data_path,
            config.cache_size_bytes / (1024 * 1024)
        );

        let backend = Self {
            db: Arc::new(db),
            config,
        };

        // Check / stamp schema version
        crate::schema::check_schema_version(&backend).map_err(|e| {
            StorageError::DatabaseError(format!("Schema version check failed: {}", e))
        })?;

        Ok(backend)
    }

    /// Helper: get table definition constant by name
    fn get_table_def(
        table_name: &str,
    ) -> StorageResult<TableDefinition<'static, &'static [u8], &'static [u8]>> {
        Ok(match table_name {
            table_names::EPISODE_CATALOG => table_defs::EPISODE_CATALOG,
            table_names::PARTITION_MAP => table_defs::PARTITION_MAP,
            table_names::MEMORY_RECORDS => table_defs::MEMORY_RECORDS,
            table_names::MEM_BY_BUCKET => table_defs::MEM_BY_BUCKET,
            table_names::MEM_BY_CONTEXT_HASH => table_defs::MEM_BY_CONTEXT_HASH,
            table_names::MEM_BY_GOAL_BUCKET => table_defs::MEM_BY_GOAL_BUCKET,
            table_names::MEM_FEATURE_POSTINGS => table_defs::MEM_FEATURE_POSTINGS,
            table_names::STRATEGY_RECORDS => table_defs::STRATEGY_RECORDS,
            table_names::STRATEGY_BY_BUCKET => table_defs::STRATEGY_BY_BUCKET,
            table_names::STRATEGY_BY_SIGNATURE => table_defs::STRATEGY_BY_SIGNATURE,
            table_names::STRATEGY_FEATURE_POSTINGS => table_defs::STRATEGY_FEATURE_POSTINGS,
            table_names::TRANSITION_STATS => table_defs::TRANSITION_STATS,
            table_names::MOTIF_STATS => table_defs::MOTIF_STATS,
            table_names::DECISION_TRACE => table_defs::DECISION_TRACE,
            table_names::OUTCOME_SIGNALS => table_defs::OUTCOME_SIGNALS,
            table_names::ID_ALLOCATOR => table_defs::ID_ALLOCATOR,
            table_names::SCHEMA_VERSIONS => table_defs::SCHEMA_VERSIONS,
            table_names::GRAPH_NODES => table_defs::GRAPH_NODES,
            table_names::GRAPH_ADJACENCY => table_defs::GRAPH_ADJACENCY,
            table_names::GRAPH_EDGES => table_defs::GRAPH_EDGES,
            table_names::WORLD_MODEL => table_defs::WORLD_MODEL,
            _ => {
                return Err(StorageError::DatabaseError(format!(
                    "Unknown table: {}",
                    table_name
                )))
            },
        })
    }

    /// Put a value (serialize with MessagePack)
    pub fn put<K, V>(&self, table_name: &str, key: K, value: &V) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
        V: Serialize,
    {
        let table_def = Self::get_table_def(table_name)?;
        let value_bytes =
            rmp_serde::to_vec(value).map_err(|e| StorageError::Serialization(e.to_string()))?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        {
            let mut table = write_txn
                .open_table(table_def)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            table
                .insert(key.as_ref(), value_bytes.as_slice())
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        }

        write_txn
            .commit()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Put raw bytes (no serialization)
    pub fn put_raw<K>(&self, table_name: &str, key: K, value: &[u8]) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
    {
        let table_def = Self::get_table_def(table_name)?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        {
            let mut table = write_txn
                .open_table(table_def)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            table
                .insert(key.as_ref(), value)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        }

        write_txn
            .commit()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Get a value (deserialize with MessagePack)
    pub fn get<K, V>(&self, table_name: &str, key: K) -> StorageResult<Option<V>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        let table_def = Self::get_table_def(table_name)?;

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let table = read_txn
            .open_table(table_def)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        match table
            .get(key.as_ref())
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?
        {
            Some(access_guard) => {
                let bytes = access_guard.value();
                let value = rmp_serde::from_slice(bytes)
                    .map_err(|e| StorageError::Deserialization(e.to_string()))?;
                Ok(Some(value))
            },
            None => Ok(None),
        }
    }

    /// Get raw bytes (no deserialization)
    pub fn get_raw<K>(&self, table_name: &str, key: K) -> StorageResult<Option<Vec<u8>>>
    where
        K: AsRef<[u8]>,
    {
        let table_def = Self::get_table_def(table_name)?;

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let table = read_txn
            .open_table(table_def)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        match table
            .get(key.as_ref())
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?
        {
            Some(access_guard) => Ok(Some(access_guard.value().to_vec())),
            None => Ok(None),
        }
    }

    /// Delete a key
    pub fn delete<K>(&self, table_name: &str, key: K) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
    {
        let table_def = Self::get_table_def(table_name)?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        {
            let mut table = write_txn
                .open_table(table_def)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            table
                .remove(key.as_ref())
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        }

        write_txn
            .commit()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Scan with prefix (returns all key-value pairs with matching prefix)
    pub fn scan_prefix<K, V>(&self, table_name: &str, prefix: K) -> StorageResult<Vec<(Vec<u8>, V)>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        let table_def = Self::get_table_def(table_name)?;
        let prefix_bytes = prefix.as_ref();

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let table = read_txn
            .open_table(table_def)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let mut results = Vec::new();

        // Range scan from prefix to prefix+1
        let iter = table
            .range(prefix_bytes..)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Check if key still has prefix
            if !key.value().starts_with(prefix_bytes) {
                break;
            }

            let deserialized = rmp_serde::from_slice(value.value())
                .map_err(|e| StorageError::Deserialization(e.to_string()))?;

            results.push((key.value().to_vec(), deserialized));
        }

        Ok(results)
    }

    /// Scan with prefix, returning raw bytes (no deserialization)
    pub fn scan_prefix_raw<K>(
        &self,
        table_name: &str,
        prefix: K,
    ) -> StorageResult<Vec<(Vec<u8>, Vec<u8>)>>
    where
        K: AsRef<[u8]>,
    {
        let table_def = Self::get_table_def(table_name)?;
        let prefix_bytes = prefix.as_ref();

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let table = read_txn
            .open_table(table_def)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let mut results = Vec::new();

        // Range scan from prefix to prefix+1
        let iter = table
            .range(prefix_bytes..)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Check if key still has prefix
            if !key.value().starts_with(prefix_bytes) {
                break;
            }

            results.push((key.value().to_vec(), value.value().to_vec()));
        }

        Ok(results)
    }

    /// Batch write operation (atomic)
    pub fn write_batch(&self, operations: Vec<BatchOperation>) -> StorageResult<()> {
        use std::collections::HashMap;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        // Group ops by table for fewer open_table calls.
        let mut by_table: HashMap<String, Vec<BatchOperation>> = HashMap::new();
        for op in operations {
            let table_name = match &op {
                BatchOperation::Put { table_name, .. } => table_name.clone(),
                BatchOperation::Delete { table_name, .. } => table_name.clone(),
            };
            by_table.entry(table_name).or_default().push(op);
        }

        for (table_name, ops) in by_table {
            let table_def = Self::get_table_def(&table_name)?;
            let mut table = write_txn
                .open_table(table_def)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            for op in ops {
                match op {
                    BatchOperation::Put { key, value, .. } => {
                        table
                            .insert(key.as_slice(), value.as_slice())
                            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
                    },
                    BatchOperation::Delete { key, .. } => {
                        table
                            .remove(key.as_slice())
                            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
                    },
                }
            }
        }

        write_txn
            .commit()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// Put a value with versioned envelope (serialize with versioned msgpack)
    pub fn put_versioned<K, V>(&self, table_name: &str, key: K, value: &V) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
        V: Serialize,
    {
        let value_bytes = crate::versioned::serialize_versioned(value)?;
        self.put_raw(table_name, key, &value_bytes)
    }

    /// Get a value with versioned envelope (deserialize versioned-or-legacy msgpack)
    pub fn get_versioned<K, V>(&self, table_name: &str, key: K) -> StorageResult<Option<V>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        match self.get_raw(table_name, key)? {
            Some(bytes) => {
                let value = crate::versioned::deserialize_versioned(&bytes)?;
                Ok(Some(value))
            },
            None => Ok(None),
        }
    }

    /// Scan all key-value pairs in a table as raw bytes (no prefix filter).
    pub fn scan_all_raw(&self, table_name: &str) -> StorageResult<Vec<(Vec<u8>, Vec<u8>)>> {
        // Reuse scan_prefix_raw with empty prefix to get all entries
        self.scan_prefix_raw(table_name, &[] as &[u8])
    }

    /// Return the number of key-value pairs in the given table. O(1).
    pub fn table_len(&self, table_name: &str) -> StorageResult<u64> {
        let table_def = Self::get_table_def(table_name)?;
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        let table = read_txn
            .open_table(table_def)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        table
            .len()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Internal iterator: calls `f(key, value)` for each row whose key starts
    /// with `prefix`. Borrowed slices — zero allocation per row. Stops early
    /// when `f` returns `Err(E)`.
    pub fn for_each_prefix_raw<K, F, E>(
        &self,
        table_name: &str,
        prefix: K,
        mut f: F,
    ) -> Result<(), ForEachError<E>>
    where
        K: AsRef<[u8]>,
        F: FnMut(&[u8], &[u8]) -> Result<(), E>,
    {
        let table_def = Self::get_table_def(table_name).map_err(ForEachError::Storage)?;
        let prefix_bytes = prefix.as_ref();

        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| ForEachError::Storage(StorageError::DatabaseError(e.to_string())))?;

        let table = read_txn
            .open_table(table_def)
            .map_err(|e| ForEachError::Storage(StorageError::DatabaseError(e.to_string())))?;

        let iter = table
            .range(prefix_bytes..)
            .map_err(|e| ForEachError::Storage(StorageError::DatabaseError(e.to_string())))?;

        for item in iter {
            let (key, value) = item
                .map_err(|e| ForEachError::Storage(StorageError::DatabaseError(e.to_string())))?;

            if !key.value().starts_with(prefix_bytes) {
                break;
            }

            f(key.value(), value.value()).map_err(ForEachError::Callback)?;
        }

        Ok(())
    }

    /// Get approximate disk usage (bytes)
    pub fn disk_usage(&self) -> StorageResult<u64> {
        let metadata = std::fs::metadata(&self.config.data_path).map_err(StorageError::Io)?;
        Ok(metadata.len())
    }
}

/// Batch operation for atomic writes
#[derive(Debug, Clone)]
pub enum BatchOperation {
    Put {
        table_name: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        table_name: String,
        key: Vec<u8>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_redb_open() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };

        let backend = RedbBackend::open(config).unwrap();

        // Verify database created
        assert!(backend.config.data_path.exists());
    }

    #[test]
    fn test_put_get() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };

        let backend = RedbBackend::open(config).unwrap();

        // Test serialization
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        struct TestData {
            id: u64,
            name: String,
        }

        let data = TestData {
            id: 42,
            name: "test".to_string(),
        };

        // Put
        backend
            .put(table_names::MEMORY_RECORDS, b"test_key", &data)
            .unwrap();

        // Get
        let retrieved: Option<TestData> = backend
            .get(table_names::MEMORY_RECORDS, b"test_key")
            .unwrap();
        assert_eq!(retrieved, Some(data));

        // Get non-existent
        let none: Option<TestData> = backend
            .get(table_names::MEMORY_RECORDS, b"missing")
            .unwrap();
        assert_eq!(none, None);
    }

    #[test]
    fn test_scan_prefix() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };

        let backend = RedbBackend::open(config).unwrap();

        // Insert multiple keys with same prefix
        backend
            .put(table_names::MEMORY_RECORDS, b"agent_1_mem_1", &100u64)
            .unwrap();
        backend
            .put(table_names::MEMORY_RECORDS, b"agent_1_mem_2", &200u64)
            .unwrap();
        backend
            .put(table_names::MEMORY_RECORDS, b"agent_2_mem_1", &300u64)
            .unwrap();

        // Scan with prefix
        let results: Vec<(Vec<u8>, u64)> = backend
            .scan_prefix(table_names::MEMORY_RECORDS, b"agent_1")
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, 100);
        assert_eq!(results[1].1, 200);
    }

    #[test]
    fn test_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };

        let backend = RedbBackend::open(config).unwrap();

        let operations = vec![
            BatchOperation::Put {
                table_name: table_names::MEMORY_RECORDS.to_string(),
                key: b"key1".to_vec(),
                value: rmp_serde::to_vec(&100u64).unwrap(),
            },
            BatchOperation::Put {
                table_name: table_names::MEMORY_RECORDS.to_string(),
                key: b"key2".to_vec(),
                value: rmp_serde::to_vec(&200u64).unwrap(),
            },
        ];

        backend.write_batch(operations).unwrap();

        let val1: Option<u64> = backend.get(table_names::MEMORY_RECORDS, b"key1").unwrap();
        let val2: Option<u64> = backend.get(table_names::MEMORY_RECORDS, b"key2").unwrap();

        assert_eq!(val1, Some(100));
        assert_eq!(val2, Some(200));
    }

    #[test]
    fn test_delete() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };

        let backend = RedbBackend::open(config).unwrap();

        backend
            .put(table_names::MEMORY_RECORDS, b"key1", &100u64)
            .unwrap();
        backend
            .delete(table_names::MEMORY_RECORDS, b"key1")
            .unwrap();

        let val: Option<u64> = backend.get(table_names::MEMORY_RECORDS, b"key1").unwrap();
        assert_eq!(val, None);
    }
}
