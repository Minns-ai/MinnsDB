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
        Serialization(String),

        #[error("Deserialization error: {0}")]
        Deserialization(String),

        #[error("Compression error: {0}")]
        Compression(String),

        #[error("WAL error: {0}")]
        Wal(String),

        #[error("Database error: {0}")]
        DatabaseError(String),
    }

    pub type StorageResult<T> = Result<T, StorageError>;
}

pub mod redb_backend;
pub mod schema;
pub mod versioned;
pub mod wal;

// Re-export commonly used items
pub use error::{StorageError, StorageResult};
pub use redb_backend::{table_names, BatchOperation, ForEachError, RedbBackend, RedbConfig};
pub use schema::{check_schema_version, stamp_schema_version, SchemaError, SchemaVersion};
pub use versioned::{
    deserialize_versioned, serialize_versioned, unwrap_versioned, wrap_versioned,
    CURRENT_DATA_VERSION, VERSION_MAGIC,
};
pub use wal::{SyncPolicy, WalConfig, WalStats, WriteAheadLog};
