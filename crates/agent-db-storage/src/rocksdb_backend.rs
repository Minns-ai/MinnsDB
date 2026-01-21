//! RocksDB backend for persistent storage
//!
//! This module provides the core RocksDB infrastructure with 16 column families
//! for storing episodes, memories, strategies, transitions, and telemetry.
//!
//! ## Architecture
//! - Catalog CFs: episode_catalog, partition_map
//! - Memory CFs: memory_records, mem_by_bucket, mem_by_context_hash, mem_feature_postings
//! - Strategy CFs: strategy_records, strategy_by_bucket, strategy_by_signature, strategy_feature_postings
//! - Learning CFs: transition_stats, motif_stats
//! - Telemetry CFs: decision_trace, outcome_signals
//! - Operational CFs: id_allocator, schema_versions

use crate::{StorageError, StorageResult};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, DBWithThreadMode, MultiThreaded, Options, WriteOptions, DB};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Column family names (16 total)
pub mod cf_names {
    // Catalogs
    pub const EPISODE_CATALOG: &str = "episode_catalog";
    pub const PARTITION_MAP: &str = "partition_map";

    // Memory store
    pub const MEMORY_RECORDS: &str = "memory_records";
    pub const MEM_BY_BUCKET: &str = "mem_by_bucket";
    pub const MEM_BY_CONTEXT_HASH: &str = "mem_by_context_hash";
    pub const MEM_FEATURE_POSTINGS: &str = "mem_feature_postings";

    // Strategy store
    pub const STRATEGY_RECORDS: &str = "strategy_records";
    pub const STRATEGY_BY_BUCKET: &str = "strategy_by_bucket";
    pub const STRATEGY_BY_SIGNATURE: &str = "strategy_by_signature";
    pub const STRATEGY_FEATURE_POSTINGS: &str = "strategy_feature_postings";

    // Learning stats
    pub const TRANSITION_STATS: &str = "transition_stats";
    pub const MOTIF_STATS: &str = "motif_stats";

    // Telemetry
    pub const DECISION_TRACE: &str = "decision_trace";
    pub const OUTCOME_SIGNALS: &str = "outcome_signals";

    // Operational
    pub const ID_ALLOCATOR: &str = "id_allocator";
    pub const SCHEMA_VERSIONS: &str = "schema_versions";
}

/// RocksDB backend configuration
#[derive(Debug, Clone)]
pub struct RocksDBConfig {
    /// Database path
    pub data_dir: PathBuf,

    /// Create if missing
    pub create_if_missing: bool,

    /// Enable Write-Ahead Log
    pub enable_wal: bool,

    /// WAL bytes per sync (0 = OS decides)
    pub wal_bytes_per_sync: u64,

    /// Max background jobs
    pub max_background_jobs: i32,

    /// Block cache size (MB)
    pub block_cache_size_mb: usize,

    /// Write buffer size (MB)
    pub write_buffer_size_mb: usize,

    /// Enable statistics
    pub enable_statistics: bool,
}

impl Default for RocksDBConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data/rocksdb"),
            create_if_missing: true,
            enable_wal: true,
            wal_bytes_per_sync: 1024 * 1024, // 1MB
            max_background_jobs: 4,
            block_cache_size_mb: 256,
            write_buffer_size_mb: 64,
            enable_statistics: true,
        }
    }
}

/// RocksDB backend with 16 column families
pub struct RocksDBBackend {
    db: Arc<DBWithThreadMode<MultiThreaded>>,
    config: RocksDBConfig,
}

impl RocksDBBackend {
    /// Open or create RocksDB with all column families
    pub fn open(config: RocksDBConfig) -> StorageResult<Self> {
        // Create data directory
        std::fs::create_dir_all(&config.data_dir)
            .map_err(|e| StorageError::Io(e))?;

        // Configure RocksDB options
        let mut db_opts = Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_background_jobs(config.max_background_jobs);
        db_opts.set_wal_bytes_per_sync(config.wal_bytes_per_sync);

        // Block cache for reads
        let cache = rocksdb::Cache::new_lru_cache(config.block_cache_size_mb * 1024 * 1024);
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        block_opts.set_bloom_filter(10.0, false); // 10 bits per key
        db_opts.set_block_based_table_factory(&block_opts);

        // Write buffer
        db_opts.set_write_buffer_size(config.write_buffer_size_mb * 1024 * 1024);

        // Enable statistics
        if config.enable_statistics {
            db_opts.enable_statistics();
        }

        // Define all 16 column families
        let cf_descriptors = vec![
            // Catalogs
            ColumnFamilyDescriptor::new(cf_names::EPISODE_CATALOG, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::PARTITION_MAP, Options::default()),

            // Memory store
            ColumnFamilyDescriptor::new(cf_names::MEMORY_RECORDS, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::MEM_BY_BUCKET, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::MEM_BY_CONTEXT_HASH, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::MEM_FEATURE_POSTINGS, Options::default()),

            // Strategy store
            ColumnFamilyDescriptor::new(cf_names::STRATEGY_RECORDS, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::STRATEGY_BY_BUCKET, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::STRATEGY_BY_SIGNATURE, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::STRATEGY_FEATURE_POSTINGS, Options::default()),

            // Learning stats
            ColumnFamilyDescriptor::new(cf_names::TRANSITION_STATS, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::MOTIF_STATS, Options::default()),

            // Telemetry
            ColumnFamilyDescriptor::new(cf_names::DECISION_TRACE, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::OUTCOME_SIGNALS, Options::default()),

            // Operational
            ColumnFamilyDescriptor::new(cf_names::ID_ALLOCATOR, Options::default()),
            ColumnFamilyDescriptor::new(cf_names::SCHEMA_VERSIONS, Options::default()),
        ];

        // Open database
        let db = DB::open_cf_descriptors(&db_opts, &config.data_dir, cf_descriptors)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        tracing::info!(
            "RocksDB opened at {:?} with 16 column families",
            config.data_dir
        );

        Ok(Self {
            db: Arc::new(db),
            config,
        })
    }

    /// Get column family handle by name
    fn cf(&self, name: &str) -> StorageResult<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| StorageError::DatabaseError(format!("Column family not found: {}", name)))
    }

    /// Put a value (serialize with bincode)
    pub fn put<K, V>(&self, cf_name: &str, key: K, value: &V) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
        V: Serialize,
    {
        let cf = self.cf(cf_name)?;
        let value_bytes = bincode::serialize(value)
            .map_err(|e| StorageError::Serialization(e))?;

        self.db
            .put_cf(cf, key.as_ref(), value_bytes)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Put raw bytes (no serialization)
    pub fn put_raw<K>(&self, cf_name: &str, key: K, value: &[u8]) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
    {
        let cf = self.cf(cf_name)?;
        self.db
            .put_cf(cf, key.as_ref(), value)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Get a value (deserialize with bincode)
    pub fn get<K, V>(&self, cf_name: &str, key: K) -> StorageResult<Option<V>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        let cf = self.cf(cf_name)?;
        let value_bytes = self.db
            .get_cf(cf, key.as_ref())
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        match value_bytes {
            Some(bytes) => {
                let value = bincode::deserialize(&bytes)
                    .map_err(|e| StorageError::Deserialization(e))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Get raw bytes (no deserialization)
    pub fn get_raw<K>(&self, cf_name: &str, key: K) -> StorageResult<Option<Vec<u8>>>
    where
        K: AsRef<[u8]>,
    {
        let cf = self.cf(cf_name)?;
        self.db
            .get_cf(cf, key.as_ref())
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Delete a key
    pub fn delete<K>(&self, cf_name: &str, key: K) -> StorageResult<()>
    where
        K: AsRef<[u8]>,
    {
        let cf = self.cf(cf_name)?;
        self.db
            .delete_cf(cf, key.as_ref())
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Scan with prefix (returns iterator of (key, value) pairs)
    pub fn scan_prefix<K, V>(
        &self,
        cf_name: &str,
        prefix: K,
    ) -> StorageResult<Vec<(Vec<u8>, V)>>
    where
        K: AsRef<[u8]>,
        V: for<'de> Deserialize<'de>,
    {
        let cf = self.cf(cf_name)?;
        let prefix_bytes = prefix.as_ref();

        let mut results = Vec::new();
        let iter = self.db.prefix_iterator_cf(cf, prefix_bytes);

        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            // Check if key still has prefix
            if !key.starts_with(prefix_bytes) {
                break;
            }

            let deserialized = bincode::deserialize(&value)
                .map_err(|e| StorageError::Deserialization(e))?;

            results.push((key.to_vec(), deserialized));
        }

        Ok(results)
    }

    /// Batch write operation (atomic)
    pub fn write_batch(&self, operations: Vec<BatchOperation>) -> StorageResult<()> {
        let mut batch = rocksdb::WriteBatch::default();

        for op in operations {
            match op {
                BatchOperation::Put { cf_name, key, value } => {
                    let cf = self.cf(&cf_name)?;
                    batch.put_cf(cf, key, value);
                }
                BatchOperation::Delete { cf_name, key } => {
                    let cf = self.cf(&cf_name)?;
                    batch.delete_cf(cf, key);
                }
            }
        }

        self.db
            .write(batch)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Flush all memtables to disk (for testing/shutdown)
    pub fn flush(&self) -> StorageResult<()> {
        self.db
            .flush()
            .map_err(|e| StorageError::DatabaseError(e.to_string()))
    }

    /// Get statistics (if enabled)
    pub fn get_statistics(&self) -> Option<String> {
        self.db.property_value("rocksdb.stats")
    }

    /// Get approximate disk usage (bytes)
    pub fn disk_usage(&self) -> StorageResult<u64> {
        // Sum up all CF sizes
        let mut total = 0u64;

        for cf_name in [
            cf_names::EPISODE_CATALOG,
            cf_names::PARTITION_MAP,
            cf_names::MEMORY_RECORDS,
            cf_names::MEM_BY_BUCKET,
            cf_names::MEM_BY_CONTEXT_HASH,
            cf_names::MEM_FEATURE_POSTINGS,
            cf_names::STRATEGY_RECORDS,
            cf_names::STRATEGY_BY_BUCKET,
            cf_names::STRATEGY_BY_SIGNATURE,
            cf_names::STRATEGY_FEATURE_POSTINGS,
            cf_names::TRANSITION_STATS,
            cf_names::MOTIF_STATS,
            cf_names::DECISION_TRACE,
            cf_names::OUTCOME_SIGNALS,
            cf_names::ID_ALLOCATOR,
            cf_names::SCHEMA_VERSIONS,
        ] {
            let cf = self.cf(cf_name)?;
            if let Some(size_str) = self.db.property_int_value_cf(cf, "rocksdb.total-sst-files-size") {
                total += size_str;
            }
        }

        Ok(total)
    }
}

/// Batch operation for atomic writes
#[derive(Debug, Clone)]
pub enum BatchOperation {
    Put {
        cf_name: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        cf_name: String,
        key: Vec<u8>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_rocksdb_open() {
        let temp_dir = TempDir::new().unwrap();
        let config = RocksDBConfig {
            data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let backend = RocksDBBackend::open(config).unwrap();

        // Verify all 16 column families exist
        assert!(backend.cf(cf_names::EPISODE_CATALOG).is_ok());
        assert!(backend.cf(cf_names::MEMORY_RECORDS).is_ok());
        assert!(backend.cf(cf_names::STRATEGY_RECORDS).is_ok());
        assert!(backend.cf(cf_names::TRANSITION_STATS).is_ok());
        assert!(backend.cf(cf_names::DECISION_TRACE).is_ok());
        assert!(backend.cf(cf_names::ID_ALLOCATOR).is_ok());
        assert!(backend.cf(cf_names::SCHEMA_VERSIONS).is_ok());
    }

    #[test]
    fn test_put_get() {
        let temp_dir = TempDir::new().unwrap();
        let config = RocksDBConfig {
            data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let backend = RocksDBBackend::open(config).unwrap();

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
        backend.put(cf_names::MEMORY_RECORDS, b"test_key", &data).unwrap();

        // Get
        let retrieved: Option<TestData> = backend.get(cf_names::MEMORY_RECORDS, b"test_key").unwrap();
        assert_eq!(retrieved, Some(data));

        // Get non-existent
        let none: Option<TestData> = backend.get(cf_names::MEMORY_RECORDS, b"missing").unwrap();
        assert_eq!(none, None);
    }

    #[test]
    fn test_scan_prefix() {
        let temp_dir = TempDir::new().unwrap();
        let config = RocksDBConfig {
            data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let backend = RocksDBBackend::open(config).unwrap();

        // Insert multiple keys with same prefix
        backend.put(cf_names::MEMORY_RECORDS, b"agent_1_mem_1", &100u64).unwrap();
        backend.put(cf_names::MEMORY_RECORDS, b"agent_1_mem_2", &200u64).unwrap();
        backend.put(cf_names::MEMORY_RECORDS, b"agent_2_mem_1", &300u64).unwrap();

        // Scan with prefix
        let results: Vec<(Vec<u8>, u64)> = backend
            .scan_prefix(cf_names::MEMORY_RECORDS, b"agent_1")
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, 100);
        assert_eq!(results[1].1, 200);
    }

    #[test]
    fn test_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let config = RocksDBConfig {
            data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let backend = RocksDBBackend::open(config).unwrap();

        let operations = vec![
            BatchOperation::Put {
                cf_name: cf_names::MEMORY_RECORDS.to_string(),
                key: b"key1".to_vec(),
                value: bincode::serialize(&100u64).unwrap(),
            },
            BatchOperation::Put {
                cf_name: cf_names::MEMORY_RECORDS.to_string(),
                key: b"key2".to_vec(),
                value: bincode::serialize(&200u64).unwrap(),
            },
        ];

        backend.write_batch(operations).unwrap();

        let val1: Option<u64> = backend.get(cf_names::MEMORY_RECORDS, b"key1").unwrap();
        let val2: Option<u64> = backend.get(cf_names::MEMORY_RECORDS, b"key2").unwrap();

        assert_eq!(val1, Some(100));
        assert_eq!(val2, Some(200));
    }
}
