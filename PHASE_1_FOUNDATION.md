# Phase 1: Foundation Architecture (Weeks 1-8)

**Goal**: Establish solid foundation for the Agentic Database with core abstractions, basic event system, simple storage, and initial graph structures.

---

## Module Architecture 🏗️

### Core Module Hierarchy

```
agent-database/
├── Cargo.toml                    # Workspace definition
├── crates/
│   ├── agent-db-core/           # Core traits, types, errors
│   │   ├── src/
│   │   │   ├── lib.rs           # Public API exports
│   │   │   ├── types.rs         # Core type definitions
│   │   │   ├── traits.rs        # Core trait definitions  
│   │   │   ├── error.rs         # Error handling framework
│   │   │   ├── config.rs        # Configuration structures
│   │   │   └── utils.rs         # Common utilities
│   │   └── Cargo.toml
│   ├── agent-db-events/         # Event system implementation
│   │   ├── src/
│   │   │   ├── lib.rs           # Public event API
│   │   │   ├── core.rs          # Event structures
│   │   │   ├── validation.rs    # Event validation logic
│   │   │   ├── buffer.rs        # Event buffering
│   │   │   ├── serde.rs         # Serialization/deserialization
│   │   │   ├── context.rs       # Context handling
│   │   │   └── causality.rs     # Causality chain validation
│   │   └── Cargo.toml
│   ├── agent-db-storage/        # Storage engine
│   │   ├── src/
│   │   │   ├── lib.rs           # Storage API
│   │   │   ├── files.rs         # File management
│   │   │   ├── mmap.rs          # Memory mapping
│   │   │   ├── wal.rs           # Write-ahead logging
│   │   │   ├── partitions.rs    # Time-based partitioning
│   │   │   ├── compression.rs   # Compression utilities
│   │   │   └── recovery.rs      # Crash recovery
│   │   └── Cargo.toml
│   └── agent-db-graph/          # Graph structures
│       ├── src/
│       │   ├── lib.rs           # Graph API
│       │   ├── core.rs          # Node/Edge definitions
│       │   ├── adjacency.rs     # Adjacency list storage
│       │   ├── traversal.rs     # Graph traversal algorithms
│       │   ├── storage.rs       # Graph persistence
│       │   └── algorithms.rs    # Basic graph algorithms
│       └── Cargo.toml
```

---

## Week 1-2: Project Setup & Core Abstractions

### Core Types (`agent-db-core/src/types.rs`)

```rust
//! Core type definitions for the Agentic Database

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// High-precision timestamp (nanoseconds since Unix epoch)
pub type Timestamp = u64;

/// Unique event identifier using UUID
pub type EventId = u128;

/// Agent identifier
pub type AgentId = u64;

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
```

### Core Traits (`agent-db-core/src/traits.rs`)

```rust
//! Core trait definitions for the Agentic Database

use crate::error::DatabaseResult;
use crate::types::*;
use async_trait::async_trait;
use std::future::Future;

/// Core database operations for event ingestion and querying
#[async_trait]
pub trait Database: Send + Sync {
    /// Ingest a single event
    async fn ingest_event(&mut self, event: Event) -> DatabaseResult<EventId>;
    
    /// Ingest multiple events atomically
    async fn ingest_events(&mut self, events: Vec<Event>) -> DatabaseResult<Vec<EventId>>;
    
    /// Retrieve events within a time range
    async fn get_events_in_range(
        &self,
        start: Timestamp,
        end: Timestamp,
    ) -> DatabaseResult<Vec<Event>>;
    
    /// Get events for a specific agent
    async fn get_agent_events(
        &self,
        agent_id: AgentId,
        limit: Option<usize>,
    ) -> DatabaseResult<Vec<Event>>;
}

/// Storage layer abstraction
#[async_trait]
pub trait Storage: Send + Sync {
    /// Write events to storage
    async fn write_events(&mut self, events: &[Event]) -> DatabaseResult<()>;
    
    /// Read events from storage
    async fn read_events(&self, partition: PartitionId) -> DatabaseResult<Vec<Event>>;
    
    /// Get available partitions
    async fn list_partitions(&self) -> DatabaseResult<Vec<PartitionId>>;
    
    /// Compact storage (remove deleted events, optimize layout)
    async fn compact(&mut self) -> DatabaseResult<()>;
}

/// Graph operations interface
#[async_trait]
pub trait GraphEngine: Send + Sync {
    /// Add node to the graph
    async fn add_node(&mut self, node: Node) -> DatabaseResult<NodeId>;
    
    /// Add edge between nodes
    async fn add_edge(&mut self, edge: Edge) -> DatabaseResult<()>;
    
    /// Get node by ID
    async fn get_node(&self, id: NodeId) -> DatabaseResult<Option<Node>>;
    
    /// Get edges from a node
    async fn get_edges_from(&self, node_id: NodeId) -> DatabaseResult<Vec<Edge>>;
    
    /// Perform graph traversal
    async fn traverse(
        &self,
        start: NodeId,
        depth: u8,
        algorithm: TraversalAlgorithm,
    ) -> DatabaseResult<Vec<NodeId>>;
}

/// Memory formation and retrieval interface
#[async_trait]
pub trait MemoryEngine: Send + Sync {
    /// Form new memory from events
    async fn form_memory(&mut self, events: &[Event]) -> DatabaseResult<MemoryId>;
    
    /// Retrieve memories relevant to context
    async fn retrieve_memories(
        &self,
        context: &EventContext,
        limit: usize,
    ) -> DatabaseResult<Vec<Memory>>;
    
    /// Update memory strength based on access
    async fn access_memory(&mut self, memory_id: MemoryId) -> DatabaseResult<()>;
}

/// Event validation interface
pub trait EventValidator: Send + Sync {
    /// Validate event structure and content
    fn validate_event(&self, event: &Event) -> DatabaseResult<()>;
    
    /// Validate causality chain
    fn validate_causality(&self, chain: &[EventId]) -> DatabaseResult<()>;
    
    /// Validate event context
    fn validate_context(&self, context: &EventContext) -> DatabaseResult<()>;
}

// Forward declarations for types used in traits
// These will be defined in their respective modules

pub struct Event; // Defined in agent-db-events
pub struct EventContext; // Defined in agent-db-events
pub struct Node; // Defined in agent-db-graph  
pub struct Edge; // Defined in agent-db-graph
pub struct Memory; // Defined in agent-db-memory
pub enum TraversalAlgorithm { BreadthFirst, DepthFirst } // Defined in agent-db-graph
```

### Error Handling (`agent-db-core/src/error.rs`)

```rust
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
```

### Dependency Configuration

#### Root `Cargo.toml`
```toml
[workspace]
members = [
    "crates/agent-db-core",
    "crates/agent-db-events", 
    "crates/agent-db-storage",
    "crates/agent-db-graph",
]

[workspace.dependencies]
# Async runtime
tokio = { version = "1.28", features = ["full"] }
async-trait = "0.1"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Utilities
uuid = { version = "1.3", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }

# Collections and data structures
indexmap = "1.9"
smallvec = "1.10"

# Performance
rayon = "1.7"
crossbeam = "0.8"

# Storage
memmap2 = "0.5"
lz4 = "1.24"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Testing
criterion = "0.5"

[profile.dev]
opt-level = 1
debug = true
overflow-checks = true

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.bench]
opt-level = 3
debug = false
lto = true
```

#### Core Module `Cargo.toml`
```toml
[package]
name = "agent-db-core"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }

[dev-dependencies]
tokio-test = "0.4"
```

---

## Week 3-4: Event System Implementation

### Event Core (`agent-db-events/src/core.rs`)

```rust
//! Core event structures and types

use agent_db_core::types::*;
use agent_db_core::error::{DatabaseError, DatabaseResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete event structure with all metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique event identifier
    pub id: EventId,
    
    /// High-precision timestamp
    pub timestamp: Timestamp,
    
    /// Agent that generated this event
    pub agent_id: AgentId,
    
    /// Session identifier for grouping
    pub session_id: SessionId,
    
    /// Type and payload of the event
    pub event_type: EventType,
    
    /// Parent events in causality chain
    pub causality_chain: Vec<EventId>,
    
    /// Environmental context
    pub context: EventContext,
    
    /// Additional metadata
    pub metadata: HashMap<String, MetadataValue>,
}

/// Different types of events the system can handle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Agent actions and decisions
    Action {
        action_name: String,
        parameters: serde_json::Value,
        outcome: ActionOutcome,
        duration_ns: u64,
    },
    
    /// Environmental observations
    Observation {
        observation_type: String,
        data: serde_json::Value,
        confidence: f32,
        source: String,
    },
    
    /// Cognitive processes
    Cognitive {
        process_type: CognitiveType,
        input: serde_json::Value,
        output: serde_json::Value,
        reasoning_trace: Vec<String>,
    },
    
    /// Communication events
    Communication {
        message_type: String,
        sender: AgentId,
        recipient: AgentId,
        content: serde_json::Value,
    },
}

/// Outcome of an action event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionOutcome {
    Success { result: serde_json::Value },
    Failure { error: String, error_code: u32 },
    Partial { result: serde_json::Value, issues: Vec<String> },
}

/// Types of cognitive processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveType {
    GoalFormation,
    Planning,
    Reasoning,
    MemoryRetrieval,
    LearningUpdate,
}

/// Environmental context at the time of event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContext {
    /// Environment state snapshot
    pub environment: EnvironmentState,
    
    /// Active goals
    pub active_goals: Vec<Goal>,
    
    /// Available resources
    pub resources: ResourceState,
    
    /// Context fingerprint for fast matching
    pub fingerprint: ContextHash,
    
    /// Context embeddings for similarity
    pub embeddings: Option<Vec<f32>>,
}

/// Environment state variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentState {
    /// Key-value environment variables
    pub variables: HashMap<String, serde_json::Value>,
    
    /// Spatial context if applicable
    pub spatial: Option<SpatialContext>,
    
    /// Temporal context
    pub temporal: TemporalContext,
}

/// Goal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub description: String,
    pub priority: f32,
    pub deadline: Option<Timestamp>,
    pub progress: f32,
    pub subgoals: Vec<GoalId>,
}

/// Resource availability state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceState {
    /// Available computational resources
    pub computational: ComputationalResources,
    
    /// Available external resources
    pub external: HashMap<String, ResourceAvailability>,
}

/// Spatial context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialContext {
    pub location: (f64, f64, f64), // x, y, z coordinates
    pub bounds: Option<BoundingBox>,
    pub reference_frame: String,
}

/// Bounding box for spatial context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: (f64, f64, f64),
    pub max: (f64, f64, f64),
}

/// Temporal context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Time of day effects
    pub time_of_day: Option<TimeOfDay>,
    
    /// Active deadlines
    pub deadlines: Vec<Deadline>,
    
    /// Temporal patterns
    pub patterns: Vec<TemporalPattern>,
}

/// Computational resource availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalResources {
    pub cpu_percent: f32,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
    pub network_bandwidth: u64,
}

/// External resource availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    pub available: bool,
    pub capacity: f32,
    pub current_usage: f32,
    pub estimated_cost: Option<f32>,
}

/// Time of day information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeOfDay {
    pub hour: u8,
    pub minute: u8,
    pub timezone: String,
}

/// Deadline information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deadline {
    pub goal_id: GoalId,
    pub timestamp: Timestamp,
    pub priority: f32,
}

/// Temporal pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_name: String,
    pub frequency: Duration,
    pub phase: f32,
}

/// Extensible metadata value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(serde_json::Value),
}

/// Duration type for convenience
use std::time::Duration;

impl Event {
    /// Create a new event with current timestamp
    pub fn new(
        agent_id: AgentId,
        session_id: SessionId,
        event_type: EventType,
        context: EventContext,
    ) -> Self {
        Self {
            id: generate_event_id(),
            timestamp: current_timestamp(),
            agent_id,
            session_id,
            event_type,
            causality_chain: Vec::new(),
            context,
            metadata: HashMap::new(),
        }
    }
    
    /// Add parent event to causality chain
    pub fn with_parent(mut self, parent_id: EventId) -> Self {
        self.causality_chain.push(parent_id);
        self
    }
    
    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: MetadataValue) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Get event size in bytes (for storage calculations)
    pub fn size_bytes(&self) -> usize {
        // Rough estimate - would be more precise with actual serialization
        std::mem::size_of::<Self>() 
            + self.causality_chain.len() * std::mem::size_of::<EventId>()
            + self.metadata.len() * 64 // rough estimate for metadata
    }
    
    /// Check if event references specific parent
    pub fn has_parent(&self, parent_id: EventId) -> bool {
        self.causality_chain.contains(&parent_id)
    }
}

impl EventContext {
    /// Create new context with computed fingerprint
    pub fn new(environment: EnvironmentState, active_goals: Vec<Goal>, resources: ResourceState) -> Self {
        let mut context = Self {
            environment,
            active_goals,
            resources,
            fingerprint: 0,
            embeddings: None,
        };
        context.fingerprint = context.compute_fingerprint();
        context
    }
    
    /// Compute context fingerprint for fast matching
    fn compute_fingerprint(&self) -> ContextHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash environment variables
        for (key, value) in &self.environment.variables {
            key.hash(&mut hasher);
            // Would need to implement Hash for serde_json::Value or convert to string
        }
        
        // Hash active goals
        for goal in &self.active_goals {
            goal.id.hash(&mut hasher);
            goal.priority.to_bits().hash(&mut hasher);
        }
        
        // Hash resource state
        self.resources.computational.cpu_percent.to_bits().hash(&mut hasher);
        self.resources.computational.memory_bytes.hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Calculate similarity to another context (0.0 to 1.0)
    pub fn similarity(&self, other: &EventContext) -> f32 {
        if let (Some(embed1), Some(embed2)) = (&self.embeddings, &other.embeddings) {
            // Cosine similarity if embeddings available
            cosine_similarity(embed1, embed2)
        } else {
            // Fallback to simple fingerprint comparison
            if self.fingerprint == other.fingerprint {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_creation() {
        let agent_id = 123;
        let session_id = 456;
        let event_type = EventType::Action {
            action_name: "test_action".to_string(),
            parameters: serde_json::Value::Null,
            outcome: ActionOutcome::Success { result: serde_json::Value::Null },
            duration_ns: 1000,
        };
        let context = EventContext::new(
            EnvironmentState {
                variables: HashMap::new(),
                spatial: None,
                temporal: TemporalContext {
                    time_of_day: None,
                    deadlines: Vec::new(),
                    patterns: Vec::new(),
                },
            },
            Vec::new(),
            ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 50.0,
                    memory_bytes: 1024 * 1024,
                    storage_bytes: 1024 * 1024 * 1024,
                    network_bandwidth: 1000,
                },
                external: HashMap::new(),
            },
        );
        
        let event = Event::new(agent_id, session_id, event_type, context);
        
        assert_eq!(event.agent_id, agent_id);
        assert_eq!(event.session_id, session_id);
        assert!(!event.id.to_string().is_empty());
        assert!(event.timestamp > 0);
    }
    
    #[test]
    fn test_causality_chain() {
        let parent_id = generate_event_id();
        let agent_id = 123;
        let session_id = 456;
        let event_type = EventType::Cognitive {
            process_type: CognitiveType::Reasoning,
            input: serde_json::Value::Null,
            output: serde_json::Value::Null,
            reasoning_trace: vec!["step1".to_string()],
        };
        let context = EventContext::new(
            EnvironmentState {
                variables: HashMap::new(),
                spatial: None,
                temporal: TemporalContext {
                    time_of_day: None,
                    deadlines: Vec::new(),
                    patterns: Vec::new(),
                },
            },
            Vec::new(),
            ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 50.0,
                    memory_bytes: 1024 * 1024,
                    storage_bytes: 1024 * 1024 * 1024,
                    network_bandwidth: 1000,
                },
                external: HashMap::new(),
            },
        );
        
        let event = Event::new(agent_id, session_id, event_type, context)
            .with_parent(parent_id);
        
        assert!(event.has_parent(parent_id));
        assert_eq!(event.causality_chain.len(), 1);
    }
}
```

---

## Success Criteria for Phase 1 ✅

### Week 1-2 Success Criteria:
- [ ] All crates compile without warnings
- [ ] Core traits and types are well-defined
- [ ] Error handling is comprehensive
- [ ] CI/CD pipeline runs successfully
- [ ] Basic benchmarking harness works

### Week 3-4 Success Criteria:
- [ ] Event creation and validation work
- [ ] Event serialization/deserialization is correct
- [ ] Context fingerprinting performs well
- [ ] Causality chain validation is robust
- [ ] Event buffer handles 10K events/sec

### Week 5-6 Success Criteria:
- [ ] Events persist across restarts
- [ ] WAL provides durability guarantees
- [ ] Memory-mapped access to recent data
- [ ] Partitioning works correctly
- [ ] 20K events/sec sustained write performance

### Week 7-8 Success Criteria:
- [ ] Graph with 100K nodes loads in <1s
- [ ] Basic traversal algorithms work
- [ ] Graph persistence and recovery
- [ ] Edge weight management functional
- [ ] Memory usage scales linearly

### Overall Phase 1 Goals:
- **Event Ingestion**: 50K events/sec
- **Graph Operations**: 100K nodes, 1M edges
- **Storage**: Crash recovery works
- **Foundation**: All core abstractions defined and working

Ready to proceed with detailed implementation of any specific module?