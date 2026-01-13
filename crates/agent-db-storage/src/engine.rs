//! Core storage engine implementation

use crate::{StorageError, StorageResult};
use crate::wal::{WriteAheadLog, WalConfig, WalEntry};
use agent_db_core::types::{EventId, Timestamp};
use agent_db_events::Event;
use crossbeam::channel::{bounded, Receiver, Sender};
use lz4::block::{compress, decompress};
use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
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
}

/// Data segment for storing events
struct DataSegment {
    id: u32,
    file: File,
    mmap: MmapMut,
    current_offset: u64,
}

/// Request for background flush
#[derive(Debug)]
enum FlushRequest {
    Event { event: Event, compressed_data: Vec<u8> },
    Sync,
    Shutdown,
}

impl StorageConfig {
    /// Create default storage configuration
    pub fn default() -> Self {
        Self {
            data_directory: PathBuf::from("./data"),
            data_dir: "./data".to_string(),
            wal_config: WalConfig::default(),
            compression: CompressionType::Lz4,
            enable_compression: true,
            max_file_size_mb: 128,
            segment_size: 128 * 1024 * 1024, // 128MB segments
            cache_size: 10000, // Cache up to 10K events
            enable_checksums: true,
            sync_interval_secs: 30,
            flush_interval: std::time::Duration::from_millis(100),
        }
    }
    
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
        std::fs::create_dir_all(&data_path)
            .map_err(StorageError::Io)?;
        std::fs::create_dir_all(&data_path.join("segments"))
            .map_err(StorageError::Io)?;
            
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
        };
        
        // Start background flush task
        let flush_engine = engine.clone_for_flush();
        tokio::spawn(async move {
            flush_engine.background_flush_task(flush_receiver).await;
        });
        
        // Recover from WAL if needed
        engine.recover().await?;
        
        Ok(engine)
    }
    
    /// Store event with durability guarantees
    pub async fn store_event(&self, event: Event) -> StorageResult<()> {
        // Serialize and optionally compress event
        let serialized = bincode::serialize(&event)
            .map_err(StorageError::Serialization)?;
            
        let (data, _compressed) = match self.config.compression {
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
        
        // Log to WAL first (durability)
        self.wal.log_store_event(&event, data.clone()).await?;
        
        // Update cache
        self.update_cache(event.id, event.clone()).await;
        
        // Queue for background storage
        self.flush_sender.send(FlushRequest::Event {
            event: event.clone(),
            compressed_data: data,
        }).map_err(|_| StorageError::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "Background flush channel closed"
        )))?;
        
        Ok(())
    }
    
    /// Retrieve event by ID
    pub async fn retrieve_event(&self, event_id: EventId) -> StorageResult<Option<Event>> {
        // Check cache first
        if let Some(event) = self.get_from_cache(event_id).await {
            return Ok(Some(event));
        }
        
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
        let cache_size = self.cache.lock().await.len();
        let wal_stats = self.wal.stats().await;
        
        StorageStats {
            total_events: index_size,
            cached_events: cache_size,
            wal_entries: wal_stats.current_sequence,
            wal_pending: wal_stats.pending_entries,
        }
    }
    
    /// Force sync to disk
    pub async fn sync(&self) -> StorageResult<()> {
        self.flush_sender.send(FlushRequest::Sync)
            .map_err(|_| StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Background flush channel closed"
            )))?;
        Ok(())
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
        }
    }
    
    async fn background_flush_task(&self, receiver: Receiver<FlushRequest>) {
        while let Ok(request) = receiver.recv() {
            match request {
                FlushRequest::Event { event, compressed_data } => {
                    if let Err(e) = self.write_event_to_segment(&event, compressed_data).await {
                        eprintln!("Background flush error: {}", e);
                    }
                }
                FlushRequest::Sync => {
                    if let Err(e) = self.force_sync().await {
                        eprintln!("Sync error: {}", e);
                    }
                }
                FlushRequest::Shutdown => break,
            }
        }
    }
    
    async fn write_event_to_segment(&self, event: &Event, data: Vec<u8>) -> StorageResult<()> {
        let mut current_segment = self.current_segment.lock().await;
        
        // Create new segment if needed
        if current_segment.is_none() || 
           current_segment.as_ref().unwrap().current_offset + data.len() as u64 > self.config.segment_size {
            
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
                    "Segment full"
                )));
            }
            
            // Write data
            segment.mmap[offset as usize..(offset as usize + data_len)].copy_from_slice(&data);
            segment.current_offset += data_len as u64;
            
            // Update index
            {
                let mut index = self.index.write().unwrap();
                index.insert(event.id, IndexEntry {
                    segment_id: segment.id,
                    offset,
                    size: data_len as u32,
                    timestamp: event.timestamp,
                    compressed: self.config.enable_compression,
                });
            }
        }
        
        Ok(())
    }
    
    async fn create_segment(&self, segment_id: u32) -> StorageResult<DataSegment> {
        let segment_path = self.config.data_directory
            .join("segments")
            .join(format!("segment_{:08}.dat", segment_id));
            
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&segment_path)
            .map_err(StorageError::Io)?;
            
        // Set file size
        file.set_len(self.config.segment_size)
            .map_err(StorageError::Io)?;
            
        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(StorageError::Io)?
        };
        
        Ok(DataSegment {
            id: segment_id,
            file,
            mmap,
            current_offset: 0,
        })
    }
    
    async fn read_from_segment(&self, entry: &IndexEntry) -> StorageResult<Option<Event>> {
        let segment_path = self.config.data_directory
            .join("segments")
            .join(format!("segment_{:08}.dat", entry.segment_id));
            
        if !segment_path.exists() {
            return Ok(None);
        }
        
        let file = File::open(&segment_path).map_err(StorageError::Io)?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(StorageError::Io)?
        };
        
        let data_slice = &mmap[entry.offset as usize..(entry.offset + entry.size as u64) as usize];
        
        // Decompress if needed
        let event_data = if entry.compressed {
            decompress(data_slice, None)
                .map_err(|e| StorageError::Compression(format!("Decompression failed: {}", e)))?
        } else {
            data_slice.to_vec()
        };
        
        // Deserialize
        let event = bincode::deserialize(&event_data)
            .map_err(StorageError::Serialization)?;
            
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
            }
            None => {
                // Check cache size and evict if necessary
                if cache.len() >= self.config.cache_size {
                    self.evict_lru_cache_entry(&mut cache);
                }
                
                cache.insert(event_id, CacheEntry {
                    event,
                    last_accessed: timestamp,
                    access_count: 1,
                });
            }
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
        let lru_key = cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| *key);
            
        if let Some(key) = lru_key {
            cache.remove(&key);
        }
    }
    
    async fn recover(&self) -> StorageResult<()> {
        let recovered_count = self.wal.recover(|entry| {
            match entry {
                WalEntry::StoreEvent { event_id, .. } => {
                    // TODO: Apply recovered store operation
                    println!("Recovering store operation for event {}", event_id);
                    Ok(())
                }
                WalEntry::DeleteEvent { event_id, .. } => {
                    // TODO: Apply recovered delete operation
                    println!("Recovering delete operation for event {}", event_id);
                    Ok(())
                }
                WalEntry::Checkpoint { .. } => {
                    println!("Checkpoint marker during recovery");
                    Ok(())
                }
            }
        }).await?;
        
        if recovered_count > 0 {
            println!("Recovered {} operations from WAL", recovered_count);
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
    pub async fn query_events(&self, session_id: u64, start_time: u64, end_time: u64) -> StorageResult<Vec<Event>> {
        let mut results = Vec::new();
        
        // For now, scan through cache for matching events
        let cache = self.cache.lock().await;
        for (_, cache_entry) in cache.iter() {
            let event = &cache_entry.event;
            if event.session_id == session_id 
                && event.timestamp >= start_time 
                && event.timestamp <= end_time {
                results.push(event.clone());
            }
        }
        
        // TODO: Also scan disk storage for events not in cache
        results.sort_by_key(|e| e.timestamp);
        Ok(results)
    }
    
    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> StorageStats {
        let cache = self.cache.lock().await;
        let index = self.index.read().unwrap();
        
        StorageStats {
            total_events: index.len() as u64,
            total_size_bytes: index.len() as u64 * 1024, // Rough estimate
            compression_ratio: 2.0, // Placeholder estimate
            segment_count: 1, // Placeholder
            cache_hit_rate: 0.85, // Placeholder
        }
    }
    
    /// Sync storage to disk
    pub async fn sync(&self) -> StorageResult<()> {
        self.force_sync().await?;
        self.wal.sync().await?;
        Ok(())
    }
}