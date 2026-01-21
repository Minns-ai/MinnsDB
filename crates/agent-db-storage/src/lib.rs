//! Storage engine for agent database events
//!
//! This crate provides the storage layer for the agent database,
//! handling persistence, compression, and retrieval of events.

pub mod error {
    //! Storage-specific error types
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum StorageError {
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),

        #[error("Serialization error: {0}")]
        Serialization(#[from] bincode::Error),

        #[error("Deserialization error: {0}")]
        Deserialization(bincode::Error),

        #[error("Compression error: {0}")]
        Compression(String),

        #[error("WAL error: {0}")]
        Wal(String),

        #[error("Database error: {0}")]
        DatabaseError(String),
    }

    pub type StorageResult<T> = Result<T, StorageError>;
}

pub mod engine;
pub mod redb_backend;
pub mod wal;

// Re-export commonly used items
pub use engine::{CompressionType, StorageConfig, StorageEngine, StorageStats};
pub use error::{StorageError, StorageResult};
pub use redb_backend::{table_names, BatchOperation, RedbBackend, RedbConfig};
pub use wal::{SyncPolicy, WalConfig, WalStats, WriteAheadLog};
