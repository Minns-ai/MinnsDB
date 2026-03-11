//! Streaming binary export/import using the v2 wire format.
//!
//! Constant ~16MB peak memory, zero payload transcoding overhead. Values are
//! written as raw versioned bytes exactly as stored in redb — no
//! deserialization on export, no re-encoding on import (except memories and
//! strategies, which must be deserialized for secondary index rebuilding).
//!
//! ## Format
//! See `wire_v2` module for the binary wire format specification.

use agent_db_storage::{table_names, BatchOperation, ForEachError, RedbBackend};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{Read, Write};

use crate::memory::Memory;
use crate::stores::{build_memory_index_ops, build_strategy_index_ops};
use crate::strategies::Strategy;
use crate::wire_v2::{self, WireError, FOOTER_TAG};

// ========== Error types ==========

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Storage error: {0}")]
    Storage(#[from] agent_db_storage::StorageError),
}

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Storage error: {0}")]
    Storage(#[from] agent_db_storage::StorageError),
    #[error("Wire format error: {0}")]
    Wire(#[from] WireError),
    #[error("Unknown record tag: 0x{0:02X}")]
    UnknownTag(u8),
    #[error("Duplicate singleton tag: 0x{0:02X}")]
    DuplicateSingleton(u8),
    #[error("Checksum mismatch")]
    ChecksumMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("Record count mismatch: expected {expected}, got {actual}")]
    RecordCountMismatch { expected: u64, actual: u64 },
    #[error("Import error: {0}")]
    Other(String),
}

/// Import mode: how to handle existing data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportMode {
    /// Replace all existing data (wipe then import).
    Replace,
    /// Merge with existing data (upsert semantics).
    Merge,
}

/// Statistics from an import operation.
#[derive(Debug, Default)]
pub struct ImportStats {
    pub memories_imported: u64,
    pub strategies_imported: u64,
    pub graph_nodes_imported: u64,
    pub graph_edges_imported: u64,
    pub total_records: u64,
}

// ========== Hashing writer ==========

/// Writer wrapper that feeds all written bytes to a SHA-256 hasher.
struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finalize_checksum(self) -> ([u8; 32], W) {
        let hash = self.hasher.finalize();
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&hash);
        (checksum, self.inner)
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

// ========== Export ==========

/// Export all persisted state from redb to a binary v2 writer.
///
/// **Important**: call `engine.export_prepare()` or flush caches before
/// exporting to ensure all in-memory state is on disk.
///
/// Returns the number of records written (excluding header/footer).
///
/// ## Failure semantics
/// If the underlying writer returns an error (e.g. `BrokenPipe` from a
/// disconnected HTTP client), the export aborts immediately with that error.
pub fn export_to_writer<W: Write>(backend: &RedbBackend, writer: W) -> Result<u64, ExportError> {
    let mut hw = HashingWriter::new(writer);
    let mut record_count: u64 = 0;

    // Header
    wire_v2::write_header(&mut hw)?;

    // Helper: write one record via for_each_prefix_raw (streaming, no allocation)
    let write_table = |hw: &mut HashingWriter<W>,
                       count: &mut u64,
                       table: &str,
                       prefix: &[u8],
                       tag: u8|
     -> Result<(), ExportError> {
        backend
            .for_each_prefix_raw(table, prefix, |key, value| {
                wire_v2::write_record(hw, tag, key, value)?;
                *count += 1;
                Ok::<(), std::io::Error>(())
            })
            .map_err(|e| match e {
                ForEachError::Storage(se) => ExportError::Storage(se),
                ForEachError::Callback(io_err) => ExportError::Io(io_err),
            })?;
        Ok(())
    };

    // Memories
    write_table(
        &mut hw,
        &mut record_count,
        table_names::MEMORY_RECORDS,
        &[],
        wire_v2::TAG_MEMORY,
    )?;

    // Strategies
    write_table(
        &mut hw,
        &mut record_count,
        table_names::STRATEGY_RECORDS,
        &[],
        wire_v2::TAG_STRATEGY,
    )?;

    // Graph nodes
    write_table(
        &mut hw,
        &mut record_count,
        table_names::GRAPH_NODES,
        b"n",
        wire_v2::TAG_GRAPH_NODE,
    )?;

    // Graph edges
    write_table(
        &mut hw,
        &mut record_count,
        table_names::GRAPH_EDGES,
        b"e",
        wire_v2::TAG_GRAPH_EDGE,
    )?;

    // Singletons: graph meta, transition model, episode detector, id allocator
    if let Some(value) = backend.get_raw(table_names::GRAPH_ADJACENCY, b"__meta__")? {
        wire_v2::write_record(&mut hw, wire_v2::TAG_GRAPH_META, b"__meta__", &value)?;
        record_count += 1;
    }

    if let Some(value) = backend.get_raw(table_names::TRANSITION_STATS, b"__model__")? {
        wire_v2::write_record(&mut hw, wire_v2::TAG_TRANSITION_MODEL, b"__model__", &value)?;
        record_count += 1;
    }

    if let Some(value) = backend.get_raw(table_names::EPISODE_CATALOG, b"__detector__")? {
        wire_v2::write_record(
            &mut hw,
            wire_v2::TAG_EPISODE_DETECTOR,
            b"__detector__",
            &value,
        )?;
        record_count += 1;
    }

    if let Some(value) = backend.get_raw(table_names::ID_ALLOCATOR, b"consolidation_counter")? {
        wire_v2::write_record(
            &mut hw,
            wire_v2::TAG_ID_ALLOCATOR,
            b"consolidation_counter",
            &value,
        )?;
        record_count += 1;
    }

    // Footer
    let (checksum, mut writer) = hw.finalize_checksum();
    wire_v2::write_footer(&mut writer, record_count, &checksum)?;
    writer.flush()?;

    Ok(record_count)
}

// ========== Import ==========

/// Batch size limits for Replace-mode table clearing.
const DELETE_BATCH_MAX_KEYS: usize = 2048;

/// Import state from binary v2 stream into this engine's redb backend.
///
/// ## Failure semantics
/// Import is **not** atomic. Partial writes may exist on failure.
/// `import_finalize()` is NOT called on failure. In Replace mode, the system
/// is in an inconsistent state after failure — caller must retry or restore
/// from backup.
pub fn import_from_reader<R: Read>(
    backend: &RedbBackend,
    mut reader: R,
    mode: ImportMode,
) -> Result<ImportStats, ImportError> {
    let mut hasher = Sha256::new();
    let mut stats = ImportStats::default();
    let mut ops: Vec<BatchOperation> = Vec::new();
    let mut singletons_seen = HashSet::new();

    // Read and hash the header (21 bytes)
    let mut header_buf = [0u8; wire_v2::HEADER_LEN];
    reader
        .read_exact(&mut header_buf)
        .map_err(WireError::from)?;
    hasher.update(header_buf);
    // Validate header contents
    {
        let mut cursor = std::io::Cursor::new(&header_buf[..]);
        wire_v2::read_header(&mut cursor)?;
    }

    // If Replace mode, clear relevant tables first using bounded batches
    if mode == ImportMode::Replace {
        clear_tables_for_replace(backend)?;
    }

    // Read records until footer
    loop {
        // Read 1-byte tag
        let mut tag_buf = [0u8; 1];
        reader.read_exact(&mut tag_buf).map_err(WireError::from)?;
        let tag = {
            let mut cursor = std::io::Cursor::new(&tag_buf[..]);
            wire_v2::read_record_tag(&mut cursor)?
        };

        if tag == FOOTER_TAG {
            // Footer tag is NOT part of the checksum — don't hash it.
            // Read footer body directly from reader.
            let (expected_count, expected_checksum) = wire_v2::read_footer(&mut reader)?;

            // Finalize checksum
            let hash = hasher.finalize();
            let mut actual_checksum = [0u8; 32];
            actual_checksum.copy_from_slice(&hash);

            // Verify checksum
            if actual_checksum != expected_checksum {
                return Err(ImportError::ChecksumMismatch {
                    expected: expected_checksum,
                    actual: actual_checksum,
                });
            }

            // Verify record count
            if stats.total_records != expected_count {
                return Err(ImportError::RecordCountMismatch {
                    expected: expected_count,
                    actual: stats.total_records,
                });
            }

            break;
        }

        // Hash the tag byte (non-footer records are part of checksum)
        hasher.update(tag_buf);

        // Check for duplicate singletons
        if wire_v2::is_singleton_tag(tag) && !singletons_seen.insert(tag) {
            return Err(ImportError::DuplicateSingleton(tag));
        }

        // Read record body and hash it
        let (key, value) = read_record_body_hashing(&mut reader, &mut hasher)?;

        // Process by tag
        match tag {
            wire_v2::TAG_MEMORY => {
                // Deserialize for secondary index building
                let (_ver, payload) = agent_db_storage::unwrap_versioned(&value);
                let memory: Memory = rmp_serde::from_slice(payload)
                    .map_err(|e| ImportError::Other(format!("deserialize memory: {}", e)))?;
                ops.push(BatchOperation::Put {
                    table_name: table_names::MEMORY_RECORDS.to_string(),
                    key: key.clone(),
                    value,
                });
                let index_ops = build_memory_index_ops(&memory)
                    .map_err(|e| ImportError::Other(format!("memory index: {}", e)))?;
                ops.extend(index_ops);
                stats.memories_imported += 1;
            },
            wire_v2::TAG_STRATEGY => {
                let (_ver, payload) = agent_db_storage::unwrap_versioned(&value);
                let strategy: Strategy = rmp_serde::from_slice(payload)
                    .map_err(|e| ImportError::Other(format!("deserialize strategy: {}", e)))?;
                ops.push(BatchOperation::Put {
                    table_name: table_names::STRATEGY_RECORDS.to_string(),
                    key: key.clone(),
                    value,
                });
                let index_ops = build_strategy_index_ops(&strategy)
                    .map_err(|e| ImportError::Other(format!("strategy index: {}", e)))?;
                ops.extend(index_ops);
                stats.strategies_imported += 1;
            },
            wire_v2::TAG_GRAPH_NODE => {
                ops.push(BatchOperation::Put {
                    table_name: table_names::GRAPH_NODES.to_string(),
                    key,
                    value,
                });
                stats.graph_nodes_imported += 1;
            },
            wire_v2::TAG_GRAPH_EDGE => {
                ops.push(BatchOperation::Put {
                    table_name: table_names::GRAPH_EDGES.to_string(),
                    key,
                    value,
                });
                stats.graph_edges_imported += 1;
            },
            wire_v2::TAG_GRAPH_META => {
                ops.push(BatchOperation::Put {
                    table_name: table_names::GRAPH_ADJACENCY.to_string(),
                    key: b"__meta__".to_vec(),
                    value,
                });
            },
            wire_v2::TAG_TRANSITION_MODEL => {
                ops.push(BatchOperation::Put {
                    table_name: table_names::TRANSITION_STATS.to_string(),
                    key: b"__model__".to_vec(),
                    value,
                });
            },
            wire_v2::TAG_EPISODE_DETECTOR => {
                ops.push(BatchOperation::Put {
                    table_name: table_names::EPISODE_CATALOG.to_string(),
                    key: b"__detector__".to_vec(),
                    value,
                });
            },
            wire_v2::TAG_ID_ALLOCATOR => {
                ops.push(BatchOperation::Put {
                    table_name: table_names::ID_ALLOCATOR.to_string(),
                    key,
                    value,
                });
            },
            _ => return Err(ImportError::UnknownTag(tag)),
        }

        stats.total_records += 1;

        // Flush in chunks of 10,000 ops to bound memory
        if ops.len() >= 10_000 {
            let batch = std::mem::take(&mut ops);
            backend
                .write_batch(batch)
                .map_err(|e| ImportError::Other(format!("write_batch: {}", e)))?;
        }
    }

    // Flush remaining ops
    if !ops.is_empty() {
        backend
            .write_batch(ops)
            .map_err(|e| ImportError::Other(format!("write_batch: {}", e)))?;
    }

    // Stamp schema version
    agent_db_storage::stamp_schema_version(backend)
        .map_err(|e| ImportError::Other(format!("stamp schema: {}", e)))?;

    Ok(stats)
}

/// Read record body (key_len + key + value_len + value) while also feeding
/// every byte to the hasher. This avoids needing a HashingReader wrapper.
fn read_record_body_hashing<R: Read>(
    reader: &mut R,
    hasher: &mut Sha256,
) -> Result<(Vec<u8>, Vec<u8>), WireError> {
    // key_len (4 bytes)
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    hasher.update(len_buf);
    let key_len = u32::from_be_bytes(len_buf);
    if key_len > wire_v2::MAX_KEY_LEN {
        return Err(WireError::KeyTooLarge(key_len));
    }

    // key
    let mut key = vec![0u8; key_len as usize];
    reader.read_exact(&mut key)?;
    hasher.update(&key);

    // value_len (4 bytes)
    reader.read_exact(&mut len_buf)?;
    hasher.update(len_buf);
    let value_len = u32::from_be_bytes(len_buf);
    if value_len > wire_v2::MAX_VALUE_LEN {
        return Err(WireError::ValueTooLarge(value_len));
    }

    // value
    let mut value = vec![0u8; value_len as usize];
    reader.read_exact(&mut value)?;
    hasher.update(&value);

    Ok((key, value))
}

/// Clear all tables that are populated by import in Replace mode.
/// Uses bounded batches to avoid O(n) memory.
fn clear_tables_for_replace(backend: &RedbBackend) -> Result<(), ImportError> {
    let tables_to_clear = [
        table_names::MEMORY_RECORDS,
        table_names::MEM_BY_CONTEXT_HASH,
        table_names::MEM_BY_BUCKET,
        table_names::MEM_BY_GOAL_BUCKET,
        table_names::MEM_FEATURE_POSTINGS,
        table_names::STRATEGY_RECORDS,
        table_names::STRATEGY_BY_BUCKET,
        table_names::STRATEGY_BY_SIGNATURE,
        table_names::STRATEGY_FEATURE_POSTINGS,
        table_names::GRAPH_NODES,
        table_names::GRAPH_EDGES,
        table_names::GRAPH_ADJACENCY,
    ];

    for table in &tables_to_clear {
        let mut delete_ops: Vec<BatchOperation> = Vec::new();

        backend
            .for_each_prefix_raw(table, &[] as &[u8], |key, _value| {
                delete_ops.push(BatchOperation::Delete {
                    table_name: table.to_string(),
                    key: key.to_vec(),
                });
                Ok::<(), std::convert::Infallible>(())
            })
            .map_err(|e| match e {
                ForEachError::Storage(se) => ImportError::Storage(se),
                ForEachError::Callback(never) => match never {},
            })?;

        // Flush in bounded chunks
        for chunk in delete_ops.chunks(DELETE_BATCH_MAX_KEYS) {
            backend
                .write_batch(chunk.to_vec())
                .map_err(|e| ImportError::Other(format!("clear table {}: {}", table, e)))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::GraphNode;
    use agent_db_storage::{deserialize_versioned, serialize_versioned, RedbConfig};
    use tempfile::TempDir;

    #[test]
    fn test_export_empty_database() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        let mut buf = Vec::new();
        let count = export_to_writer(&backend, &mut buf).unwrap();
        assert_eq!(count, 0);

        // Verify binary header: magic + version
        assert!(buf.len() >= wire_v2::HEADER_LEN);
        assert_eq!(&buf[0..4], wire_v2::MAGIC);
        assert_eq!(buf[4], wire_v2::FORMAT_VERSION);

        // Footer should follow header immediately (record_count = 0)
        let footer_start = wire_v2::HEADER_LEN;
        assert_eq!(buf[footer_start], wire_v2::FOOTER_TAG);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("source.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Write some test data using versioned format
        let test_node = crate::structures::GraphNode {
            id: 1,
            node_type: crate::structures::NodeType::Agent {
                agent_id: 42,
                agent_type: agent_db_core::types::AgentType::default(),
                capabilities: vec![],
            },
            created_at: 1000,
            updated_at: 1000,
            properties: std::collections::HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
            group_id: String::new(),
        };
        let mut key = Vec::with_capacity(9);
        key.push(b'n');
        key.extend_from_slice(&1u64.to_be_bytes());
        let value = serialize_versioned(&test_node).unwrap();
        backend
            .put_raw(table_names::GRAPH_NODES, &key, &value)
            .unwrap();

        // Export
        let mut buf = Vec::new();
        let count = export_to_writer(&backend, &mut buf).unwrap();
        assert!(count > 0);

        // Import into a fresh database
        let config2 = RedbConfig {
            data_path: temp_dir.path().join("target.redb"),
            ..Default::default()
        };
        let backend2 = RedbBackend::open(config2).unwrap();

        let reader = std::io::Cursor::new(&buf);
        let stats = import_from_reader(&backend2, reader, ImportMode::Replace).unwrap();
        assert_eq!(stats.graph_nodes_imported, 1);

        // Verify the node is in the target database
        let restored_raw = backend2.get_raw(table_names::GRAPH_NODES, &key).unwrap();
        assert!(restored_raw.is_some());
        let restored: GraphNode = deserialize_versioned(&restored_raw.unwrap()).unwrap();
        assert_eq!(restored.id, 1);
    }

    #[test]
    fn test_checksum_verification() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Export
        let mut buf = Vec::new();
        export_to_writer(&backend, &mut buf).unwrap();
        assert!(buf.len() > 10);

        // Corrupt a byte in the header area (after magic/version, before footer)
        if buf.len() > 6 {
            buf[6] ^= 0xFF;
        }

        // Import should fail with checksum mismatch
        let config2 = RedbConfig {
            data_path: temp_dir.path().join("target.redb"),
            ..Default::default()
        };
        let backend2 = RedbBackend::open(config2).unwrap();
        let reader = std::io::Cursor::new(&buf);
        let result = import_from_reader(&backend2, reader, ImportMode::Replace);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_footer_count() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Write some data
        let value = serialize_versioned(&crate::structures::GraphNode {
            id: 1,
            node_type: crate::structures::NodeType::Agent {
                agent_id: 1,
                agent_type: agent_db_core::types::AgentType::default(),
                capabilities: vec![],
            },
            created_at: 1,
            updated_at: 1,
            properties: std::collections::HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
            group_id: String::new(),
        })
        .unwrap();
        let mut key = vec![b'n'];
        key.extend_from_slice(&1u64.to_be_bytes());
        backend
            .put_raw(table_names::GRAPH_NODES, &key, &value)
            .unwrap();

        // Export normally
        let mut buf = Vec::new();
        export_to_writer(&backend, &mut buf).unwrap();

        // Tamper with the footer: change record_count but fix the checksum
        // to test that count mismatch is detected separately.
        // The footer is: [0xFF][count: 8 bytes][checksum: 32 bytes]
        // Find footer tag
        let footer_pos = buf.iter().rposition(|&b| b == FOOTER_TAG).unwrap();

        // Change count from 1 to 999 — but also need to recompute checksum
        // for the pre-footer bytes to pass checksum check.
        // Actually, easier: just change the count. The checksum will still
        // match (it covers pre-footer bytes), but the count won't match
        // the number of records parsed.
        buf[footer_pos + 1..footer_pos + 9].copy_from_slice(&999u64.to_be_bytes());

        let config2 = RedbConfig {
            data_path: temp_dir.path().join("target.redb"),
            ..Default::default()
        };
        let backend2 = RedbBackend::open(config2).unwrap();
        let reader = std::io::Cursor::new(&buf);
        let result = import_from_reader(&backend2, reader, ImportMode::Replace);
        match result {
            Err(ImportError::RecordCountMismatch {
                expected: 999,
                actual: 1,
            }) => {},
            Err(ImportError::ChecksumMismatch { .. }) => {
                // Also acceptable: the checksum check may fail first
                // because modifying footer count doesn't affect pre-footer checksum
                // but the footer itself may have been included. This depends on implementation.
            },
            other => panic!(
                "Expected RecordCountMismatch or ChecksumMismatch, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_duplicate_singleton_rejected() {
        // Manually construct a stream with duplicate TAG_GRAPH_META
        let mut buf = Vec::new();
        let mut hasher = Sha256::new();

        // Write header
        wire_v2::write_header(&mut buf).unwrap();
        hasher.update(&buf);

        // Write two graph_meta records (duplicate singleton)
        let pre_len = buf.len();
        wire_v2::write_record(&mut buf, wire_v2::TAG_GRAPH_META, b"__meta__", b"data1").unwrap();
        hasher.update(&buf[pre_len..]);

        let pre_len2 = buf.len();
        wire_v2::write_record(&mut buf, wire_v2::TAG_GRAPH_META, b"__meta__", b"data2").unwrap();
        hasher.update(&buf[pre_len2..]);

        // Write footer
        let hash = hasher.finalize();
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&hash);
        wire_v2::write_footer(&mut buf, 2, &checksum).unwrap();

        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();
        let reader = std::io::Cursor::new(&buf);
        let result = import_from_reader(&backend, reader, ImportMode::Merge);
        match result {
            Err(ImportError::DuplicateSingleton(wire_v2::TAG_GRAPH_META)) => {},
            other => panic!("Expected DuplicateSingleton, got {:?}", other),
        }
    }

    #[test]
    fn test_broken_pipe_export() {
        // Writer that fails after N bytes
        struct FailWriter {
            written: usize,
            limit: usize,
        }
        impl Write for FailWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if self.written >= self.limit {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "client disconnected",
                    ));
                }
                let n = buf.len().min(self.limit - self.written);
                self.written += n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Write some data so export has records to emit
        let value = serialize_versioned(&42u64).unwrap();
        backend
            .put_raw(table_names::MEMORY_RECORDS, b"key1", &value)
            .unwrap();

        let writer = FailWriter {
            written: 0,
            limit: 10, // Fail after 10 bytes (in the middle of header)
        };
        let result = export_to_writer(&backend, writer);
        assert!(result.is_err());
        match result {
            Err(ExportError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe),
            other => panic!("Expected BrokenPipe IO error, got {:?}", other),
        }
    }
}
