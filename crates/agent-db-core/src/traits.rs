//! Core trait definitions for the Agentic Database

use crate::error::DatabaseResult;
use crate::types::*;
use async_trait::async_trait;

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

#[derive(Debug, Clone)]
pub struct Event {
    pub id: EventId,
    pub timestamp: Timestamp,
    pub agent_id: AgentId,
    // Placeholder - full definition will be in agent-db-events
}

#[derive(Debug, Clone)]
pub struct EventContext {
    pub fingerprint: ContextHash,
    // Placeholder - full definition will be in agent-db-events
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    // Placeholder - full definition will be in agent-db-graph
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,
    // Placeholder - full definition will be in agent-db-graph
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub id: MemoryId,
    // Placeholder - full definition will be in agent-db-memory
}

#[derive(Debug, Clone)]
pub enum TraversalAlgorithm {
    BreadthFirst,
    DepthFirst,
    // More will be added in agent-db-graph
}
