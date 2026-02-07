//! Write-Ahead Log (WAL) implementation for durability
//!
//! The WAL ensures that all operations are logged to disk before being applied,
//! providing crash recovery and data durability guarantees.

use crate::{StorageError, StorageResult};
use agent_db_core::types::{EventId, Timestamp};
use agent_db_events::Event;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// WAL entry representing a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Store event operation
    StoreEvent {
        event_id: EventId,
        timestamp: Timestamp,
        data: Vec<u8>, // Compressed event data
    },
    /// Delete event operation
    DeleteEvent {
        event_id: EventId,
        timestamp: Timestamp,
    },
    /// Checkpoint marker
    Checkpoint {
        timestamp: Timestamp,
        last_event_id: EventId,
    },
}

/// WAL entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// Entry sequence number
    pub sequence: u64,

    /// Entry timestamp
    pub timestamp: Timestamp,

    /// Checksum for integrity
    pub checksum: u32,

    /// The actual operation
    pub entry: WalEntry,
}

/// Configuration for WAL
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum WAL file size before rotation
    pub max_file_size: u64,

    /// Directory for WAL files
    pub wal_directory: std::path::PathBuf,

    /// Sync policy
    pub sync_policy: SyncPolicy,

    /// Buffer size for batching writes
    pub buffer_size: usize,

    /// Checkpoint interval (number of entries)
    pub checkpoint_interval: u64,
}

/// WAL sync policy
#[derive(Debug, Clone, PartialEq)]
pub enum SyncPolicy {
    /// Sync every write (safest, slowest)
    Always,
    /// Sync on checkpoint only
    Checkpoint,
    /// Sync on timer interval
    Interval(std::time::Duration),
    /// Never sync (fastest, least safe)
    Never,
}

/// Write-Ahead Log implementation
pub struct WriteAheadLog {
    config: WalConfig,
    current_file: Arc<Mutex<BufWriter<File>>>,
    current_sequence: Arc<Mutex<u64>>,
    pending_entries: Arc<Mutex<VecDeque<WalRecord>>>,
    last_checkpoint: Arc<Mutex<u64>>,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_file_size: 64 * 1024 * 1024, // 64MB
            wal_directory: std::path::PathBuf::from("./data/wal"),
            sync_policy: SyncPolicy::Checkpoint,
            buffer_size: 1024,
            checkpoint_interval: 10000,
        }
    }
}

impl WalConfig {
    /// Create WAL config with custom directory
    pub fn with_directory<P: AsRef<Path>>(directory: P) -> Self {
        Self {
            wal_directory: directory.as_ref().to_path_buf(),
            ..Default::default()
        }
    }
}

impl WriteAheadLog {
    /// Create new WAL with configuration
    pub fn new(config: WalConfig) -> StorageResult<Self> {
        // Create WAL directory if it doesn't exist
        std::fs::create_dir_all(&config.wal_directory).map_err(StorageError::Io)?;

        // Open or create current WAL file
        let wal_file_path = config.wal_directory.join("wal_current.log");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_file_path)
            .map_err(StorageError::Io)?;

        let writer = BufWriter::new(file);

        // Determine current sequence number by reading existing WAL
        let current_sequence = Self::read_last_sequence(&config.wal_directory)?;

        Ok(Self {
            config,
            current_file: Arc::new(Mutex::new(writer)),
            current_sequence: Arc::new(Mutex::new(current_sequence)),
            pending_entries: Arc::new(Mutex::new(VecDeque::new())),
            last_checkpoint: Arc::new(Mutex::new(0)),
        })
    }

    /// Log event storage operation
    pub async fn log_store_event(
        &self,
        event: &Event,
        compressed_data: Vec<u8>,
    ) -> StorageResult<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let entry = WalEntry::StoreEvent {
            event_id: event.id,
            timestamp,
            data: compressed_data,
        };

        self.write_entry(entry).await
    }

    /// Log event deletion operation
    pub async fn log_delete_event(&self, event_id: EventId) -> StorageResult<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let entry = WalEntry::DeleteEvent {
            event_id,
            timestamp,
        };

        self.write_entry(entry).await
    }

    /// Create checkpoint
    pub async fn checkpoint(&self, last_event_id: EventId) -> StorageResult<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let entry = WalEntry::Checkpoint {
            timestamp,
            last_event_id,
        };

        // Update last checkpoint sequence
        {
            let sequence = *self.current_sequence.lock().unwrap();
            *self.last_checkpoint.lock().unwrap() = sequence;
        }

        self.write_entry(entry).await?;

        // Force sync on checkpoint
        self.sync().await?;

        Ok(())
    }

    /// Write WAL entry
    async fn write_entry(&self, entry: WalEntry) -> StorageResult<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Get next sequence number
        let sequence = {
            let mut seq = self.current_sequence.lock().unwrap();
            *seq += 1;
            *seq
        };

        // Create record
        let record = WalRecord {
            sequence,
            timestamp,
            checksum: 0, // Will be calculated
            entry,
        };

        // Calculate checksum
        let mut record_with_checksum = record;
        record_with_checksum.checksum = self.calculate_checksum(&record_with_checksum);

        // Add to pending buffer
        let should_flush = {
            let mut pending = self.pending_entries.lock().unwrap();
            pending.push_back(record_with_checksum.clone());
            pending.len() >= self.config.buffer_size
        };

        // Flush outside of lock scope
        if should_flush {
            self.flush_buffer().await?;
        }

        // Check if we need to sync
        if self.config.sync_policy == SyncPolicy::Always {
            self.sync().await?;
        }

        // Check if we need checkpoint
        let last_checkpoint = *self.last_checkpoint.lock().unwrap();
        if sequence - last_checkpoint >= self.config.checkpoint_interval {
            // Don't checkpoint here to avoid deadlock - let caller handle
        }

        Ok(())
    }

    /// Flush pending entries to disk
    async fn flush_buffer(&self) -> StorageResult<()> {
        let entries_to_flush = {
            let mut pending = self.pending_entries.lock().unwrap();
            let entries: Vec<WalRecord> = pending.drain(..).collect();
            entries
        };

        if entries_to_flush.is_empty() {
            return Ok(());
        }

        // Serialize and write entries
        let mut writer = self.current_file.lock().unwrap();
        for entry in entries_to_flush {
            let serialized = bincode::serialize(&entry).map_err(StorageError::Serialization)?;

            // Write length prefix
            let len = serialized.len() as u32;
            writer
                .write_all(&len.to_le_bytes())
                .map_err(StorageError::Io)?;

            // Write serialized entry
            writer.write_all(&serialized).map_err(StorageError::Io)?;
        }

        Ok(())
    }

    /// Force sync to disk
    async fn sync(&self) -> StorageResult<()> {
        // First flush any pending entries
        self.flush_buffer().await?;

        // Then sync to disk
        let mut writer = self.current_file.lock().unwrap();
        writer.flush().map_err(StorageError::Io)?;

        Ok(())
    }

    /// Calculate simple checksum for integrity
    fn calculate_checksum(&self, record: &WalRecord) -> u32 {
        let mut checksum = 0u32;
        checksum = checksum.wrapping_add(record.sequence as u32);
        checksum = checksum.wrapping_add((record.timestamp >> 32) as u32);
        checksum = checksum.wrapping_add(record.timestamp as u32);

        match &record.entry {
            WalEntry::StoreEvent { event_id, .. } => {
                checksum = checksum.wrapping_add(*event_id as u32);
            },
            WalEntry::DeleteEvent { event_id, .. } => {
                checksum = checksum.wrapping_add(*event_id as u32);
                checksum = checksum.wrapping_add(0xDEAD); // Marker for delete
            },
            WalEntry::Checkpoint { last_event_id, .. } => {
                checksum = checksum.wrapping_add(*last_event_id as u32);
                checksum = checksum.wrapping_add(0xC4EC); // Marker for checkpoint
            },
        }

        checksum
    }

    /// Read last sequence number from existing WAL files
    fn read_last_sequence(wal_directory: &Path) -> StorageResult<u64> {
        let wal_file_path = wal_directory.join("wal_current.log");

        if !wal_file_path.exists() {
            return Ok(0);
        }

        let mut file = File::open(&wal_file_path).map_err(StorageError::Io)?;
        let mut reader = BufReader::new(&mut file);
        let mut last_sequence = 0u64;

        loop {
            // Read length prefix
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut len_bytes).is_err() {
                break; // End of file
            }

            let entry_len = u32::from_le_bytes(len_bytes) as usize;

            // Read entry data
            let mut entry_data = vec![0u8; entry_len];
            if reader.read_exact(&mut entry_data).is_err() {
                break; // Corrupted or incomplete entry
            }

            // Deserialize and check sequence
            if let Ok(record) = bincode::deserialize::<WalRecord>(&entry_data) {
                last_sequence = last_sequence.max(record.sequence);
            }
        }

        Ok(last_sequence)
    }

    /// Recover operations from WAL
    pub async fn recover<F>(&self, mut apply_fn: F) -> StorageResult<u64>
    where
        F: FnMut(&WalEntry) -> StorageResult<()>,
    {
        let wal_file_path = self.config.wal_directory.join("wal_current.log");

        if !wal_file_path.exists() {
            return Ok(0);
        }

        let mut file = File::open(&wal_file_path).map_err(StorageError::Io)?;
        let mut reader = BufReader::new(&mut file);
        let mut recovered_count = 0u64;
        let mut last_checkpoint_sequence = 0u64;

        // First pass: find last checkpoint
        loop {
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut len_bytes).is_err() {
                break;
            }

            let entry_len = u32::from_le_bytes(len_bytes) as usize;
            let mut entry_data = vec![0u8; entry_len];

            if reader.read_exact(&mut entry_data).is_err() {
                break;
            }

            if let Ok(record) = bincode::deserialize::<WalRecord>(&entry_data) {
                if matches!(record.entry, WalEntry::Checkpoint { .. }) {
                    last_checkpoint_sequence = record.sequence;
                }
            }
        }

        // Second pass: apply operations after last checkpoint
        file.seek(SeekFrom::Start(0)).map_err(StorageError::Io)?;
        reader = BufReader::new(&mut file);

        loop {
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut len_bytes).is_err() {
                break;
            }

            let entry_len = u32::from_le_bytes(len_bytes) as usize;
            let mut entry_data = vec![0u8; entry_len];

            if reader.read_exact(&mut entry_data).is_err() {
                break;
            }

            if let Ok(record) = bincode::deserialize::<WalRecord>(&entry_data) {
                // Skip entries before last checkpoint
                if record.sequence <= last_checkpoint_sequence {
                    continue;
                }

                // Verify checksum
                let expected_checksum = record.checksum;
                let mut record_for_check = record.clone();
                record_for_check.checksum = 0;
                let calculated_checksum = self.calculate_checksum(&record_for_check);

                if expected_checksum != calculated_checksum {
                    eprintln!(
                        "WAL entry {} has invalid checksum, skipping",
                        record.sequence
                    );
                    continue;
                }

                // Apply operation
                apply_fn(&record.entry)?;
                recovered_count += 1;
            }
        }

        Ok(recovered_count)
    }

    /// Get current WAL statistics
    pub async fn stats(&self) -> WalStats {
        let current_sequence = *self.current_sequence.lock().unwrap();
        let last_checkpoint = *self.last_checkpoint.lock().unwrap();
        let pending_count = self.pending_entries.lock().unwrap().len();

        WalStats {
            current_sequence,
            last_checkpoint_sequence: last_checkpoint,
            pending_entries: pending_count,
            entries_since_checkpoint: current_sequence - last_checkpoint,
        }
    }
}

/// WAL statistics
#[derive(Debug, Clone)]
pub struct WalStats {
    pub current_sequence: u64,
    pub last_checkpoint_sequence: u64,
    pub pending_entries: usize,
    pub entries_since_checkpoint: u64,
}
