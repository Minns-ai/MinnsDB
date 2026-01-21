//! Configuration management for the Agentic Database

use crate::types::{IngestionConfig, StorageConfig, GraphConfig, MemoryConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Base directory for data storage
    pub data_directory: PathBuf,
    
    /// Event ingestion configuration
    pub ingestion: IngestionConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Graph configuration
    pub graph: GraphConfig,
    
    /// Memory system configuration
    pub memory: MemoryConfig,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            data_directory: PathBuf::from("./agent_db_data"),
            ingestion: IngestionConfig::default(),
            storage: StorageConfig::default(),
            graph: GraphConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10_000,
            flush_interval: std::time::Duration::from_millis(100),
            max_event_size: 1024 * 1024, // 1MB
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            partition_duration: std::time::Duration::from_secs(3600), // 1 hour
            hot_partitions: 24,
            compression_level: 6,
            wal_sync_interval: std::time::Duration::from_millis(50),
        }
    }
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_edges_per_node: 1000,
            edge_decay_rate: 0.001,
            min_edge_weight: 0.01,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memories_per_agent: 10_000,
            memory_decay_rate: 0.0001,
            consolidation_interval: std::time::Duration::from_secs(300), // 5 minutes
        }
    }
}

impl DatabaseConfig {
    /// Load configuration from file
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: DatabaseConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.ingestion.buffer_size == 0 {
            return Err("Buffer size must be greater than 0".to_string());
        }
        
        if self.storage.hot_partitions == 0 {
            return Err("Hot partitions must be greater than 0".to_string());
        }
        
        if self.storage.compression_level > 9 {
            return Err("Compression level must be between 0 and 9".to_string());
        }
        
        if self.graph.min_edge_weight <= 0.0 {
            return Err("Minimum edge weight must be positive".to_string());
        }
        
        Ok(())
    }
    
    /// Get total memory budget in bytes
    pub fn estimated_memory_usage(&self) -> u64 {
        let events_memory = self.ingestion.buffer_size as u64 * 1000; // rough estimate
        let graph_memory = self.graph.max_edges_per_node as u64 * 1000 * 100; // rough estimate
        let memory_system = self.memory.max_memories_per_agent as u64 * 1000 * 100; // rough estimate
        
        events_memory + graph_memory + memory_system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = DatabaseConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.estimated_memory_usage() > 0);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = DatabaseConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: DatabaseConfig = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(config.ingestion.buffer_size, deserialized.ingestion.buffer_size);
        assert_eq!(config.storage.hot_partitions, deserialized.storage.hot_partitions);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = DatabaseConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid buffer size
        config.ingestion.buffer_size = 0;
        assert!(config.validate().is_err());
        
        // Reset and test invalid edge weight
        config = DatabaseConfig::default();
        config.graph.min_edge_weight = -1.0;
        assert!(config.validate().is_err());
    }
}