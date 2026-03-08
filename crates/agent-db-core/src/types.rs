//! Core type definitions for the Agentic Database

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// High-precision timestamp (nanoseconds since Unix epoch)
pub type Timestamp = u64;

/// Unique event identifier using UUID
pub type EventId = u128;

/// Agent identifier
pub type AgentId = u64;

/// Agent type classifier (e.g., "coding-assistant", "data-analyst", "task-manager")
pub type AgentType = String;

/// Session identifier for grouping related events
pub type SessionId = u64;

/// Node identifier in the graph
pub type NodeId = u64;

/// Memory identifier
pub type MemoryId = u64;

/// Pattern identifier
pub type PatternId = u64;

/// Goal identifier
pub type GoalId = u64;

/// Context hash for fast context matching
pub type ContextHash = u64;

/// Partition identifier for storage
pub type PartitionId = u64;

pub use crate::event_time::EventTime;

/// Current timestamp in nanoseconds
pub fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Generate unique event ID
pub fn generate_event_id() -> EventId {
    uuid::Uuid::new_v4().as_u128()
}

/// Configuration for the database system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Base directory for data storage
    pub data_directory: std::path::PathBuf,

    /// Event ingestion configuration
    pub ingestion: IngestionConfig,

    /// Storage configuration
    pub storage: StorageConfig,

    /// Graph configuration
    pub graph: GraphConfig,

    /// Memory system configuration
    pub memory: MemoryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Buffer size for batching events
    pub buffer_size: usize,

    /// Flush interval for buffered events
    pub flush_interval: Duration,

    /// Maximum event size in bytes
    pub max_event_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Partition duration (e.g., 1 hour, 1 day)
    pub partition_duration: Duration,

    /// Number of hot partitions to keep in memory
    pub hot_partitions: usize,

    /// Compression level (0-9)
    pub compression_level: u8,

    /// WAL sync frequency
    pub wal_sync_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Maximum edges per node before compression
    pub max_edges_per_node: usize,

    /// Edge weight decay rate
    pub edge_decay_rate: f32,

    /// Minimum edge weight before pruning
    pub min_edge_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memories per agent
    pub max_memories_per_agent: usize,

    /// Memory decay rate
    pub memory_decay_rate: f32,

    /// Consolidation frequency
    pub consolidation_interval: Duration,
}
