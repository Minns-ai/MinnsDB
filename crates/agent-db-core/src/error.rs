//! Comprehensive error handling for the Agentic Database

use thiserror::Error;

/// Result type for database operations
pub type DatabaseResult<T> = Result<T, DatabaseError>;

/// Comprehensive error types for all database operations
#[derive(Debug, Error)]
pub enum DatabaseError {
    // Event-related errors
    #[error("Invalid event structure: {0}")]
    InvalidEvent(String),
    
    #[error("Event validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Causality violation: event {event_id} references non-existent parent {parent_id}")]
    CausalityViolation {
        event_id: crate::types::EventId,
        parent_id: crate::types::EventId,
    },
    
    // Storage-related errors
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Partition not found: {0}")]
    PartitionNotFound(crate::types::PartitionId),
    
    #[error("Corruption detected in partition {0}")]
    DataCorruption(crate::types::PartitionId),
    
    #[error("Write-ahead log error: {0}")]
    WalError(String),
    
    // Graph-related errors
    #[error("Node not found: {0}")]
    NodeNotFound(crate::types::NodeId),
    
    #[error("Graph cycle detected")]
    GraphCycleDetected,
    
    #[error("Invalid graph operation: {0}")]
    InvalidGraphOperation(String),
    
    // Memory-related errors
    #[error("Memory not found: {0}")]
    MemoryNotFound(crate::types::MemoryId),
    
    #[error("Memory formation failed: {0}")]
    MemoryFormationFailed(String),
    
    #[error("Context matching error: {0}")]
    ContextMatchingError(String),
    
    // System-level errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    #[error("Timeout: operation took longer than {0:?}")]
    Timeout(std::time::Duration),
    
    #[error("Concurrent modification detected")]
    ConcurrentModification,
    
    // I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    // Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    // Network errors (for future distributed functionality)
    #[error("Network error: {0}")]
    NetworkError(String),
    
    // Generic catch-all
    #[error("Internal error: {0}")]
    Internal(String),
}

impl DatabaseError {
    /// Create a new storage error
    pub fn storage(msg: impl Into<String>) -> Self {
        DatabaseError::StorageError(msg.into())
    }
    
    /// Create a new validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        DatabaseError::ValidationFailed(msg.into())
    }
    
    /// Create a new internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        DatabaseError::Internal(msg.into())
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            DatabaseError::Timeout(_) => true,
            DatabaseError::ResourceExhausted(_) => true,
            DatabaseError::NetworkError(_) => true,
            DatabaseError::ConcurrentModification => true,
            _ => false,
        }
    }
    
    /// Check if error indicates data corruption
    pub fn is_corruption(&self) -> bool {
        matches!(self, DatabaseError::DataCorruption(_))
    }
}

/// Convert from bincode errors
impl From<Box<bincode::ErrorKind>> for DatabaseError {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        DatabaseError::SerializationError(format!("Bincode error: {}", err))
    }
}

/// Convert from serde_json errors
impl From<serde_json::Error> for DatabaseError {
    fn from(err: serde_json::Error) -> Self {
        DatabaseError::SerializationError(format!("JSON error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_classification() {
        let timeout_err = DatabaseError::Timeout(std::time::Duration::from_secs(5));
        assert!(timeout_err.is_recoverable());
        assert!(!timeout_err.is_corruption());
        
        let corruption_err = DatabaseError::DataCorruption(123);
        assert!(!corruption_err.is_recoverable());
        assert!(corruption_err.is_corruption());
    }
}