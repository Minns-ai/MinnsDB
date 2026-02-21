//! Core storage engine implementation

use crate::wal::{WalConfig, WalEntry, WriteAheadLog};
use crate::{StorageError, StorageResult};
use agent_db_core::types::{EventId, Timestamp};
use agent_db_events::Event;
use crossbeam::channel::{bounded, Receiver, Sender};
use lz4::block::{compress, decompress};
use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as AsyncMutex;

/// Compression type for stored data
#[derive(Debug, Clone)]
pub enum CompressionType {
    None,
    Lz4,
    Gzip,
}

/// Storage engine configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Base directory for storage files (legacy compatibility)
    pub data_directory: PathBuf,

    /// Base directory as string (new API)
    pub data_dir: String,

    /// WAL configuration
    pub wal_config: WalConfig,

    /// Compression type
    pub compression: CompressionType,

    /// Enable compression (legacy compatibility)
    pub enable_compression: bool,

    /// Maximum file size in MB
    pub max_file_size_mb: u64,

    /// File size for data segments
    pub segment_size: u64,

    /// Cache size (number of events)
    pub cache_size: usize,

    /// Enable checksums for data integrity
    pub enable_checksums: bool,

    /// Sync interval in seconds
    pub sync_interval_secs: u64,

    /// Background flush interval
    pub flush_interval: std::time::Duration,

    /// Segment compaction interval (0 disables compaction)
    pub compaction_interval: std::time::Duration,

    /// Minimum number of segments before compaction runs
    pub compaction_min_segments: u32,
}

/// Storage engine statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total number of events stored
    pub total_events: u64,

    /// Total size in bytes
    pub total_size_bytes: u64,

    /// Compression ratio (if compression enabled)
    pub compression_ratio: f64,

    /// Number of segments
    pub segment_count: u32,

    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Event storage index entry
#[derive(Debug, Clone)]
struct IndexEntry {
    /// File segment containing the event
    segment_id: u32,

    /// Offset within the segment
    offset: u64,

    /// Compressed size
    size: u32,

    /// Timestamp for ordering
    #[allow(dead_code)]
    timestamp: Timestamp,

    /// Whether data is compressed
    compressed: bool,
}

/// In-memory cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    event: Event,
    last_accessed: Timestamp,
    access_count: u64,
}

/// Storage engine for events with WAL, compression, and memory mapping
pub struct StorageEngine {
    config: StorageConfig,
    wal: Arc<WriteAheadLog>,
    index: Arc<RwLock<HashMap<EventId, IndexEntry>>>,
    cache: Arc<AsyncMutex<HashMap<EventId, CacheEntry>>>,
    current_segment: Arc<AsyncMutex<Option<DataSegment>>>,
    flush_sender: Sender<FlushRequest>,
    counters: Arc<StorageCounters>,
}

/// Data segment for storing events
struct DataSegment {
    id: u32,
    #[allow(dead_code)]
    file: File,
    mmap: MmapMut,
    current_offset: u64,
}

/// Request for background flush
#[derive(Debug)]
enum FlushRequest {
    Event {
        event: Box<Event>,
        compressed_data: Vec<u8>,
        is_compressed: bool,
    },
    Sync,
    #[allow(dead_code)]
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct CompactionStats {
    pub original_segments: u32,
    pub compacted_segments: u32,
    pub events_copied: u64,
    pub bytes_written: u64,
}

#[derive(Debug, Default)]
struct StorageCounters {
    total_events_written: AtomicU64,
    total_raw_bytes: AtomicU64,
    total_stored_bytes: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    segments_created: AtomicU64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_directory: PathBuf::from("./data"),
            data_dir: "./data".to_string(),
            wal_config: WalConfig::default(),
            compression: CompressionType::Lz4,
            enable_compression: true,
            max_file_size_mb: 128,
            segment_size: 128 * 1024 * 1024, // 128MB segments
            cache_size: 10000,               // Cache up to 10K events
            enable_checksums: true,
            sync_interval_secs: 30,
            flush_interval: std::time::Duration::from_millis(100),
            compaction_interval: std::time::Duration::from_secs(1800),
            compaction_min_segments: 3,
        }
    }
}

impl StorageConfig {
    /// Create config with custom directory
    pub fn with_directory<P: Into<PathBuf>>(directory: P) -> Self {
        let mut config = Self::default();
        let path = directory.into();
        config.data_directory = path.clone();
        config.data_dir = path.to_string_lossy().to_string();
        config.wal_config = WalConfig::with_directory(config.data_directory.join("wal"));
        config
    }
}

impl StorageEngine {
    /// Create new storage engine
    pub async fn new(config: StorageConfig) -> StorageResult<Self> {
        // Use data_dir if set, otherwise fall back to data_directory
        let data_path = if !config.data_dir.is_empty() {
            PathBuf::from(&config.data_dir)
        } else {
            config.data_directory.clone()
        };

        // Create data directory
        std::fs::create_dir_all(&data_path).map_err(StorageError::Io)?;
        std::fs::create_dir_all(data_path.join("segments")).map_err(StorageError::Io)?;

        // Initialize WAL
        let wal = Arc::new(WriteAheadLog::new(config.wal_config.clone())?);

        // Initialize index and cache
        let index = Arc::new(RwLock::new(HashMap::new()));
        let cache = Arc::new(AsyncMutex::new(HashMap::new()));

        // Create background flush channel
        let (flush_sender, flush_receiver) = bounded(1000);

        let engine = Self {
            config,
            wal,
            index,
            cache,
            current_segment: Arc::new(AsyncMutex::new(None)),
            flush_sender,
            counters: Arc::new(StorageCounters::default()),
        };

        // Start background flush task
        let flush_engine = engine.clone_for_flush();
        tokio::spawn(async move {
            flush_engine.background_flush_task(flush_receiver).await;
        });

        if !engine.config.compaction_interval.is_zero() {
            let compaction_engine = engine.clone_for_flush();
            tokio::task::spawn_blocking(move || {
                compaction_engine.background_compaction_task_blocking();
            });
        }

        // Recover from WAL if needed
        engine.recover().await?;

        Ok(engine)
    }

    /// Store event with durability guarantees
    pub async fn store_event(&self, event: Event) -> StorageResult<()> {
        // Serialize and optionally compress event
        let serialized =
            rmp_serde::to_vec(&event).map_err(|e| StorageError::Serialization(e.to_string()))?;

        let raw_len = serialized.len();
        let (data, is_compressed) = match self.config.compression {
            CompressionType::Lz4 => {
                match compress(&serialized, None, true) {
                    Ok(compressed_data) => (compressed_data, true),
                    Err(_) => (serialized, false), // Fall back to uncompressed
                }
            },
            CompressionType::Gzip => {
                // Implement gzip compression here if needed
                (serialized, false)
            },
            CompressionType::None => (serialized, false),
        };

        self.counters
            .total_events_written
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .total_raw_bytes
            .fetch_add(raw_len as u64, Ordering::Relaxed);
        self.counters
            .total_stored_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);

        // Log to WAL first (durability)
        self.wal.log_store_event(&event, data.clone()).await?;

        // Update cache
        self.update_cache(event.id, event.clone()).await;

        // Queue for background storage
        self.flush_sender
            .send(FlushRequest::Event {
                event: Box::new(event.clone()),
                compressed_data: data,
                is_compressed,
            })
            .map_err(|_| {
                StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "Background flush channel closed",
                ))
            })?;

        Ok(())
    }

    /// Retrieve event by ID
    pub async fn retrieve_event(&self, event_id: EventId) -> StorageResult<Option<Event>> {
        // Check cache first
        if let Some(event) = self.get_from_cache(event_id).await {
            return Ok(Some(event));
        }

        self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Check index for disk location
        let index_entry = {
            let index = self.index.read().unwrap();
            index.get(&event_id).cloned()
        };

        if let Some(entry) = index_entry {
            // Read from disk
            let event = self.read_from_segment(&entry).await?;

            // Update cache
            if let Some(ref evt) = event {
                self.update_cache(event_id, evt.clone()).await;
            }

            Ok(event)
        } else {
            Ok(None)
        }
    }

    /// Delete event
    pub async fn delete_event(&self, event_id: EventId) -> StorageResult<()> {
        // Log deletion to WAL
        self.wal.log_delete_event(event_id).await?;

        // Remove from cache
        self.cache.lock().await.remove(&event_id);

        // Remove from index (actual data cleanup will happen during compaction)
        {
            let mut index = self.index.write().unwrap();
            index.remove(&event_id);
        }

        Ok(())
    }

    /// Get storage statistics
    pub async fn stats(&self) -> StorageStats {
        let index_size = self.index.read().unwrap().len();
        let cache = self.cache.lock().await;
        let _cache_size = cache.len();
        drop(cache);

        let (segment_count, total_size_bytes) = self.scan_segment_usage();
        let raw_bytes = self.counters.total_raw_bytes.load(Ordering::Relaxed);
        let stored_bytes = self.counters.total_stored_bytes.load(Ordering::Relaxed);
        let cache_hits = self.counters.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.counters.cache_misses.load(Ordering::Relaxed);
        let cache_total = cache_hits + cache_misses;
        let cache_hit_rate = if cache_total > 0 {
            cache_hits as f64 / cache_total as f64
        } else {
            0.0
        };
        let compression_ratio = if stored_bytes > 0 {
            raw_bytes as f64 / stored_bytes as f64
        } else {
            0.0
        };

        StorageStats {
            total_events: index_size as u64,
            total_size_bytes,
            compression_ratio,
            segment_count,
            cache_hit_rate,
        }
    }

    /// Force sync to disk
    pub async fn sync(&self) -> StorageResult<()> {
        self.flush_sender.send(FlushRequest::Sync).map_err(|_| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Background flush channel closed",
            ))
        })?;
        Ok(())
    }

    /// Compact all segments by rewriting only live events into new segments.
    ///
    /// NOTE: This blocks segment writes while compaction runs.
    pub async fn compact_segments(&self) -> StorageResult<CompactionStats> {
        let engine = self.clone_for_flush();
        tokio::task::spawn_blocking(move || engine.compact_segments_blocking())
            .await
            .map_err(|err| StorageError::Io(std::io::Error::other(err)))?
    }

    fn compact_segments_blocking(&self) -> StorageResult<CompactionStats> {
        // Block background flush writes while we compact.
        let _segment_guard = self.current_segment.blocking_lock();

        let entries: Vec<(EventId, IndexEntry)> = {
            let index = self.index.read().unwrap();
            index
                .iter()
                .map(|(id, entry)| (*id, entry.clone()))
                .collect()
        };

        let segments_dir = self.config.data_directory.join("segments");
        let original_segments = self.scan_segment_usage().0;

        let temp_dir = self.config.data_directory.join(format!(
            "segments_compact_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ));
        std::fs::create_dir_all(&temp_dir).map_err(StorageError::Io)?;

        struct CompactionSegment {
            id: u32,
            _file: File,
            mmap: MmapMut,
            current_offset: u64,
        }

        let mut current_segment: Option<CompactionSegment> = None;
        let mut new_index: HashMap<EventId, IndexEntry> = HashMap::new();
        let mut bytes_written = 0u64;
        let mut raw_bytes = 0u64;
        let mut compacted_segments = 0u32;

        for (event_id, entry) in entries {
            let Some(event) = self.read_from_segment_sync(&entry)? else {
                continue;
            };

            let serialized = rmp_serde::to_vec(&event)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            let raw_len = serialized.len();
            let (data, is_compressed) = match self.config.compression {
                CompressionType::Lz4 => match compress(&serialized, None, true) {
                    Ok(compressed_data) => (compressed_data, true),
                    Err(_) => (serialized, false),
                },
                CompressionType::Gzip => (serialized, false),
                CompressionType::None => (serialized, false),
            };

            raw_bytes = raw_bytes.saturating_add(raw_len as u64);

            if current_segment.is_none()
                || current_segment.as_ref().unwrap().current_offset + data.len() as u64
                    > self.config.segment_size
            {
                let segment_id = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as u32)
                    .wrapping_add(compacted_segments);
                let segment_path = temp_dir.join(format!("segment_{:08}.dat", segment_id));
                let file = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .read(true)
                    .write(true)
                    .open(&segment_path)
                    .map_err(StorageError::Io)?;
                file.set_len(self.config.segment_size)
                    .map_err(StorageError::Io)?;
                let mmap = unsafe { memmap2::MmapMut::map_mut(&file).map_err(StorageError::Io)? };
                current_segment = Some(CompactionSegment {
                    id: segment_id,
                    _file: file,
                    mmap,
                    current_offset: 0,
                });
                compacted_segments = compacted_segments.saturating_add(1);
            }

            let segment = current_segment.as_mut().unwrap();
            let offset = segment.current_offset;
            let data_len = data.len();

            segment.mmap[offset as usize..(offset as usize + data_len)].copy_from_slice(&data);
            segment.current_offset += data_len as u64;

            new_index.insert(
                event_id,
                IndexEntry {
                    segment_id: segment.id,
                    offset,
                    size: data_len as u32,
                    timestamp: event.timestamp,
                    compressed: is_compressed,
                },
            );

            bytes_written = bytes_written.saturating_add(data_len as u64);
        }

        if let Some(segment) = current_segment {
            segment.mmap.flush().map_err(StorageError::Io)?;
            drop(segment);
        }

        let old_dir = self.config.data_directory.join(format!(
            "segments_old_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ));
        if segments_dir.exists() {
            std::fs::rename(&segments_dir, &old_dir).map_err(StorageError::Io)?;
        }
        std::fs::rename(&temp_dir, &segments_dir).map_err(StorageError::Io)?;
        if old_dir.exists() {
            let _ = std::fs::remove_dir_all(&old_dir);
        }

        {
            let mut index = self.index.write().unwrap();
            *index = new_index;
        }

        self.counters
            .total_events_written
            .store(self.index.read().unwrap().len() as u64, Ordering::Relaxed);
        self.counters
            .total_raw_bytes
            .store(raw_bytes, Ordering::Relaxed);
        self.counters
            .total_stored_bytes
            .store(bytes_written, Ordering::Relaxed);
        self.counters
            .segments_created
            .store(compacted_segments as u64, Ordering::Relaxed);

        Ok(CompactionStats {
            original_segments,
            compacted_segments,
            events_copied: self.index.read().unwrap().len() as u64,
            bytes_written,
        })
    }

    // Private helper methods

    fn clone_for_flush(&self) -> StorageEngine {
        StorageEngine {
            config: self.config.clone(),
            wal: self.wal.clone(),
            index: self.index.clone(),
            cache: self.cache.clone(),
            current_segment: self.current_segment.clone(),
            flush_sender: self.flush_sender.clone(),
            counters: self.counters.clone(),
        }
    }

    async fn background_flush_task(&self, receiver: Receiver<FlushRequest>) {
        while let Ok(request) = receiver.recv() {
            match request {
                FlushRequest::Event {
                    event,
                    compressed_data,
                    is_compressed,
                } => {
                    if let Err(e) = self
                        .write_event_to_segment(&event, compressed_data, is_compressed)
                        .await
                    {
                        eprintln!("Background flush error: {}", e);
                    }
                },
                FlushRequest::Sync => {
                    if let Err(e) = self.force_sync().await {
                        eprintln!("Sync error: {}", e);
                    }
                },
                FlushRequest::Shutdown => break,
            }
        }
    }

    fn background_compaction_task_blocking(&self) {
        let interval = self.config.compaction_interval;
        loop {
            std::thread::sleep(interval);
            let (segment_count, _) = self.scan_segment_usage();
            if segment_count < self.config.compaction_min_segments {
                continue;
            }
            if let Err(err) = self.compact_segments_blocking() {
                eprintln!("Background compaction error: {}", err);
            }
        }
    }

    async fn write_event_to_segment(
        &self,
        event: &Event,
        data: Vec<u8>,
        is_compressed: bool,
    ) -> StorageResult<()> {
        let mut current_segment = self.current_segment.lock().await;

        // Create new segment if needed
        if current_segment.is_none()
            || current_segment.as_ref().unwrap().current_offset + data.len() as u64
                > self.config.segment_size
        {
            let segment_id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32;

            let new_segment = self.create_segment(segment_id).await?;
            *current_segment = Some(new_segment);
        }

        // Write to current segment
        if let Some(ref mut segment) = *current_segment {
            let offset = segment.current_offset;
            let data_len = data.len();

            // Ensure we have space
            if offset + data_len as u64 > segment.mmap.len() as u64 {
                return Err(StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "Segment full",
                )));
            }

            // Write data
            segment.mmap[offset as usize..(offset as usize + data_len)].copy_from_slice(&data);
            segment.current_offset += data_len as u64;

            // Update index
            {
                let mut index = self.index.write().unwrap();
                index.insert(
                    event.id,
                    IndexEntry {
                        segment_id: segment.id,
                        offset,
                        size: data_len as u32,
                        timestamp: event.timestamp,
                        compressed: is_compressed,
                    },
                );
            }
        }

        Ok(())
    }

    async fn create_segment(&self, segment_id: u32) -> StorageResult<DataSegment> {
        let segment_path = self
            .config
            .data_directory
            .join("segments")
            .join(format!("segment_{:08}.dat", segment_id));

        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&segment_path)
            .map_err(StorageError::Io)?;

        // Set file size
        file.set_len(self.config.segment_size)
            .map_err(StorageError::Io)?;

        let mmap = unsafe { memmap2::MmapMut::map_mut(&file).map_err(StorageError::Io)? };

        self.counters
            .segments_created
            .fetch_add(1, Ordering::Relaxed);

        Ok(DataSegment {
            id: segment_id,
            file,
            mmap,
            current_offset: 0,
        })
    }

    async fn read_from_segment(&self, entry: &IndexEntry) -> StorageResult<Option<Event>> {
        self.read_from_segment_sync(entry)
    }

    fn read_from_segment_sync(&self, entry: &IndexEntry) -> StorageResult<Option<Event>> {
        let segment_path = self
            .config
            .data_directory
            .join("segments")
            .join(format!("segment_{:08}.dat", entry.segment_id));

        if !segment_path.exists() {
            return Ok(None);
        }

        let file = File::open(&segment_path).map_err(StorageError::Io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(StorageError::Io)? };

        let data_slice = &mmap[entry.offset as usize..(entry.offset + entry.size as u64) as usize];

        // Decompress if needed
        let event_data = if entry.compressed {
            decompress(data_slice, None)
                .map_err(|e| StorageError::Compression(format!("Decompression failed: {}", e)))?
        } else {
            data_slice.to_vec()
        };

        // Deserialize
        let event = rmp_serde::from_slice(&event_data)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        Ok(Some(event))
    }

    async fn update_cache(&self, event_id: EventId, event: Event) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut cache = self.cache.lock().await;

        // Add or update cache entry
        match cache.get_mut(&event_id) {
            Some(entry) => {
                entry.event = event;
                entry.last_accessed = timestamp;
                entry.access_count += 1;
            },
            None => {
                // Check cache size and evict if necessary
                if cache.len() >= self.config.cache_size {
                    self.evict_lru_cache_entry(&mut cache);
                }

                cache.insert(
                    event_id,
                    CacheEntry {
                        event,
                        last_accessed: timestamp,
                        access_count: 1,
                    },
                );
            },
        }
    }

    async fn get_from_cache(&self, event_id: EventId) -> Option<Event> {
        let mut cache = self.cache.lock().await;

        if let Some(entry) = cache.get_mut(&event_id) {
            entry.last_accessed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            entry.access_count += 1;
            self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.event.clone())
        } else {
            None
        }
    }

    fn evict_lru_cache_entry(&self, cache: &mut HashMap<EventId, CacheEntry>) {
        if cache.is_empty() {
            return;
        }

        // Find least recently used entry
        let lru_key = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| *key);

        if let Some(key) = lru_key {
            cache.remove(&key);
        }
    }

    async fn recover(&self) -> StorageResult<()> {
        let mut recovered_entries = Vec::new();
        let recovered_count = self
            .wal
            .recover(|entry| {
                recovered_entries.push(entry.clone());
                Ok(())
            })
            .await?;

        if recovered_count == 0 {
            return Ok(());
        }

        for entry in recovered_entries {
            match entry {
                WalEntry::StoreEvent { data, .. } => {
                    let (event, is_compressed) = self.decode_wal_event(&data)?;
                    self.write_event_to_segment(&event, data.clone(), is_compressed)
                        .await?;
                    self.update_recovery_counters(&event, &data);
                },
                WalEntry::DeleteEvent { event_id, .. } => {
                    self.cache.lock().await.remove(&event_id);
                    {
                        let mut index = self.index.write().unwrap();
                        index.remove(&event_id);
                    }
                },
                WalEntry::Checkpoint { .. } => {},
            }
        }

        Ok(())
    }

    async fn force_sync(&self) -> StorageResult<()> {
        // Sync current segment
        if let Some(ref segment) = *self.current_segment.lock().await {
            segment.mmap.flush().map_err(StorageError::Io)?;
        }

        Ok(())
    }

    /// Query events by session and time range
    pub async fn query_events(
        &self,
        session_id: u64,
        start_time: u64,
        end_time: u64,
    ) -> StorageResult<Vec<Event>> {
        let mut results = Vec::new();

        // For now, scan through cache for matching events
        let cache = self.cache.lock().await;
        for (_, cache_entry) in cache.iter() {
            let event = &cache_entry.event;
            if event.session_id == session_id
                && event.timestamp >= start_time
                && event.timestamp <= end_time
            {
                results.push(event.clone());
            }
        }

        // TODO: Also scan disk storage for events not in cache
        results.sort_by_key(|e| e.timestamp);
        Ok(results)
    }

    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> StorageStats {
        self.stats().await
    }

    fn scan_segment_usage(&self) -> (u32, u64) {
        let segments_dir = self.config.data_directory.join("segments");
        let mut total_size = 0u64;
        let mut count = 0u32;

        if let Ok(entries) = std::fs::read_dir(segments_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        total_size = total_size.saturating_add(metadata.len());
                        count = count.saturating_add(1);
                    }
                }
            }
        }

        (count, total_size)
    }

    fn decode_wal_event(&self, data: &[u8]) -> StorageResult<(Event, bool)> {
        if let Ok(decompressed) = decompress(data, None) {
            if let Ok(event) = rmp_serde::from_slice::<Event>(&decompressed) {
                return Ok((event, true));
            }
        }

        let event = rmp_serde::from_slice::<Event>(data)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        Ok((event, false))
    }

    fn update_recovery_counters(&self, event: &Event, stored_bytes: &[u8]) {
        if let Ok(serialized) = rmp_serde::to_vec(event) {
            self.counters
                .total_events_written
                .fetch_add(1, Ordering::Relaxed);
            self.counters
                .total_raw_bytes
                .fetch_add(serialized.len() as u64, Ordering::Relaxed);
            self.counters
                .total_stored_bytes
                .fetch_add(stored_bytes.len() as u64, Ordering::Relaxed);
        }
    }
}
