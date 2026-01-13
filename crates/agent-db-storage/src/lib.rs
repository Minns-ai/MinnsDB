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
        
        #[error("Compression error: {0}")]
        Compression(String),
        
        #[error("WAL error: {0}")]
        Wal(String),
    }
    
    pub type StorageResult<T> = Result<T, StorageError>;
}

pub mod wal;
pub mod engine;

// Re-export commonly used items
pub use error::{StorageError, StorageResult};
pub use engine::{StorageEngine, StorageConfig, StorageStats, CompressionType};
pub use wal::{WriteAheadLog, WalConfig, WalStats, SyncPolicy};