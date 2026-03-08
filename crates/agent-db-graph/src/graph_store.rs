// Graph persistence layer - Persistent storage for graph structure
//
// Provides trait-based abstraction for storing graph nodes and edges in persistent storage.
// Implementation uses goal-bucket partitioning for semantic sharding and efficient memory management.
//
// Types are unified with structures.rs — GraphNode/GraphEdge/NodeType/EdgeType are the
// single source of truth. NodeHeader provides a lightweight scoring record (~40 bytes)
// for streaming importance scoring without full deserialization.

use crate::structures::{AdjList, EdgeId, GoalBucketId, NodeId};
use agent_db_core::types::Timestamp;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

// Re-export structures.rs types so downstream can `use graph_store::{GraphNode, ..}`
pub use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

// ============================================================================
// NodeHeader — lightweight scoring record
// ============================================================================

/// Eviction tier determines how aggressively a node can be evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvictionTier {
    /// Never evict: Agent, Goal
    Protected = 0,
    /// Evict only under heavy pressure: Strategy, Memory, Tool
    Important = 1,
    /// Normal eviction: Concept, Claim
    Standard = 2,
    /// Evict first: Context, Episode, Event, Result
    Ephemeral = 3,
}

/// Lightweight scoring record (~40 bytes) that avoids full node deserialization.
///
/// Stored alongside full nodes in the cold store; scanned during pruning
/// to determine importance without loading the full `GraphNode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHeader {
    pub node_id: NodeId,
    pub node_type_discriminant: u8,
    pub signal: f32,
    pub tier: EvictionTier,
    pub degree: u32,
    pub updated_at: Timestamp,
    pub created_at: Timestamp,
    pub goal_bucket: GoalBucketId,
}

/// Fixed binary size of a NodeHeader (42 bytes).
pub const NODE_HEADER_BYTES: usize = 42;

impl NodeHeader {
    /// Build a header from a full node.
    pub fn from_node(node: &GraphNode, bucket: GoalBucketId) -> Self {
        Self {
            node_id: node.id,
            node_type_discriminant: node.node_type.discriminant(),
            signal: node.node_type.signal(),
            tier: node.node_type.eviction_tier(),
            degree: node.degree,
            updated_at: node.updated_at,
            created_at: node.created_at,
            goal_bucket: bucket,
        }
    }

    /// Compute importance score (higher = more important, keep in RAM longer).
    ///
    /// Formula: `signal * tier_boost * recency_factor * degree_factor`
    pub fn score(&self, now: Timestamp) -> f32 {
        let tier_boost = match self.tier {
            EvictionTier::Protected => 100.0,
            EvictionTier::Important => 10.0,
            EvictionTier::Standard => 1.0,
            EvictionTier::Ephemeral => 0.1,
        };

        // Recency: exponential decay over time (nanosecond timestamps)
        let age_ns = now.saturating_sub(self.updated_at) as f64;
        let age_hours = age_ns / 3_600_000_000_000.0;
        let recency = (-age_hours / 168.0).exp() as f32; // half-life ~1 week

        let degree_factor = 1.0 + (self.degree as f32).ln().max(0.0);

        self.signal * tier_boost * recency * degree_factor
    }

    // ====================================================================
    // Fixed-layout binary serialization (42 bytes, zero-alloc decode)
    // ====================================================================
    // Layout (little-endian):
    //   [0.. 8)  node_id       : u64
    //   [8.. 9)  type_disc     : u8
    //   [9..13)  signal        : f32
    //   [13..14) tier          : u8
    //   [14..18) degree        : u32
    //   [18..26) updated_at    : u64
    //   [26..34) created_at    : u64
    //   [34..42) goal_bucket   : u64

    /// Serialize to a fixed 42-byte binary layout.
    pub fn to_bytes(&self) -> [u8; NODE_HEADER_BYTES] {
        let mut buf = [0u8; NODE_HEADER_BYTES];
        buf[0..8].copy_from_slice(&self.node_id.to_le_bytes());
        buf[8] = self.node_type_discriminant;
        buf[9..13].copy_from_slice(&self.signal.to_le_bytes());
        buf[13] = self.tier as u8;
        buf[14..18].copy_from_slice(&self.degree.to_le_bytes());
        buf[18..26].copy_from_slice(&self.updated_at.to_le_bytes());
        buf[26..34].copy_from_slice(&self.created_at.to_le_bytes());
        buf[34..42].copy_from_slice(&self.goal_bucket.to_le_bytes());
        buf
    }

    /// Deserialize from the fixed 42-byte binary layout.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < NODE_HEADER_BYTES {
            return None;
        }
        Some(Self {
            node_id: u64::from_le_bytes(bytes[0..8].try_into().ok()?),
            node_type_discriminant: bytes[8],
            signal: f32::from_le_bytes(bytes[9..13].try_into().ok()?),
            tier: match bytes[13] {
                0 => EvictionTier::Protected,
                1 => EvictionTier::Important,
                2 => EvictionTier::Standard,
                _ => EvictionTier::Ephemeral,
            },
            degree: u32::from_le_bytes(bytes[14..18].try_into().ok()?),
            updated_at: u64::from_le_bytes(bytes[18..26].try_into().ok()?),
            created_at: u64::from_le_bytes(bytes[26..34].try_into().ok()?),
            goal_bucket: u64::from_le_bytes(bytes[34..42].try_into().ok()?),
        })
    }

    /// Compute importance score directly from raw bytes without constructing
    /// a full `NodeHeader`. Reads only the 4 scoring fields:
    /// `signal(4B) + tier(1B) + degree(4B) + updated_at(8B)` = 17 bytes.
    ///
    /// Returns `Some((node_id, score))` or `None` if bytes are too short.
    pub fn score_from_bytes(bytes: &[u8], now: Timestamp) -> Option<(NodeId, f32)> {
        if bytes.len() < NODE_HEADER_BYTES {
            return None;
        }
        let node_id = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let signal = f32::from_le_bytes(bytes[9..13].try_into().ok()?);
        let tier_boost = match bytes[13] {
            0 => 100.0f32,
            1 => 10.0,
            2 => 1.0,
            _ => 0.1,
        };
        let degree = u32::from_le_bytes(bytes[14..18].try_into().ok()?);
        let updated_at = u64::from_le_bytes(bytes[18..26].try_into().ok()?);

        let age_ns = now.saturating_sub(updated_at) as f64;
        let age_hours = age_ns / 3_600_000_000_000.0;
        let recency = (-age_hours / 168.0).exp() as f32;
        let degree_factor = 1.0 + (degree as f32).ln().max(0.0);

        Some((node_id, signal * tier_boost * recency * degree_factor))
    }
}

/// Path through the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<GraphEdge>,
    pub total_weight: f32,
    pub length: usize,
}

/// Subgraph centered on a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    pub center: NodeId,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub radius: u32,
}

/// Statistics for a goal bucket partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketInfo {
    pub bucket_id: GoalBucketId,
    pub node_count: u64,
    pub edge_count: u64,
    pub size_bytes: u64,
    pub last_modified: u64,
}

/// Trait for persistent graph storage
///
/// Provides operations for storing and querying graph structure with goal-bucket partitioning.
/// All types are from `structures.rs` — no separate persistence-layer types.
///
/// Edge operations use `EdgeId` for multi-edge support (multiple edges between
/// the same pair of nodes with different EdgeTypes).
pub trait GraphStore: Send + Sync {
    // ============================================================================
    // Node Operations
    // ============================================================================

    /// Add a node to the graph
    fn add_node(&mut self, bucket: GoalBucketId, node: GraphNode) -> Result<(), GraphStoreError>;

    /// Get a node by ID
    fn get_node(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Option<GraphNode>, GraphStoreError>;

    /// Delete a node (and all its edges)
    fn delete_node(&mut self, bucket: GoalBucketId, node_id: NodeId)
        -> Result<(), GraphStoreError>;

    /// Check if a node exists
    fn has_node(&self, bucket: GoalBucketId, node_id: NodeId) -> Result<bool, GraphStoreError>;

    /// Get all nodes in a bucket (for iteration)
    fn get_all_nodes(&self, bucket: GoalBucketId) -> Result<Vec<GraphNode>, GraphStoreError>;

    // ============================================================================
    // Edge Operations (EdgeId-based for multi-edge support)
    // ============================================================================

    /// Add an edge between two nodes
    fn add_edge(&mut self, bucket: GoalBucketId, edge: GraphEdge) -> Result<(), GraphStoreError>;

    /// Get edge by EdgeId
    fn get_edge(
        &self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError>;

    /// Delete an edge by EdgeId
    fn delete_edge(&mut self, bucket: GoalBucketId, edge_id: EdgeId)
        -> Result<(), GraphStoreError>;

    /// Get all outgoing neighbors (forward adjacency)
    fn get_neighbors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError>;

    /// Get all incoming neighbors (reverse adjacency / backlinks)
    fn get_predecessors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError>;

    /// Get all edges from a node
    fn get_outgoing_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError>;

    /// Get all edges to a node
    fn get_incoming_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError>;

    // ============================================================================
    // Header Operations (for streaming scoring)
    // ============================================================================

    /// Store a node header for fast scoring
    fn store_header(
        &mut self,
        bucket: GoalBucketId,
        header: NodeHeader,
    ) -> Result<(), GraphStoreError>;

    /// Scan headers across all buckets (bounded by limit)
    fn scan_headers(&self, limit: usize) -> Result<Vec<NodeHeader>, GraphStoreError>;

    /// Delete a node header
    fn delete_header(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError>;

    // ============================================================================
    // Partition Operations
    // ============================================================================

    /// Load a partition into memory (for LRU cache management)
    fn load_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError>;

    /// Unload a partition from memory
    fn unload_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError>;

    /// Check if partition is loaded
    fn is_partition_loaded(&self, bucket: GoalBucketId) -> bool;

    /// Get partition statistics
    fn get_partition_stats(&self, bucket: GoalBucketId) -> Result<BucketInfo, GraphStoreError>;

    /// Get all bucket IDs
    fn get_all_buckets(&self) -> Result<Vec<GoalBucketId>, GraphStoreError>;

    // ============================================================================
    // Traversal Operations
    // ============================================================================

    /// Breadth-first search from a starting node
    fn traverse_bfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError>;

    /// Depth-first search from a starting node
    fn traverse_dfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError>;

    /// Find paths between two nodes
    fn find_paths(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
        max_depth: u32,
    ) -> Result<Vec<GraphPath>, GraphStoreError>;

    /// Get subgraph centered on a node
    fn get_subgraph(
        &self,
        bucket: GoalBucketId,
        center: NodeId,
        radius: u32,
    ) -> Result<Subgraph, GraphStoreError>;

    // ============================================================================
    // Batch Operations
    // ============================================================================

    /// Add multiple nodes at once (more efficient)
    fn add_nodes(
        &mut self,
        bucket: GoalBucketId,
        nodes: Vec<GraphNode>,
    ) -> Result<(), GraphStoreError> {
        for node in nodes {
            self.add_node(bucket, node)?;
        }
        Ok(())
    }

    /// Add multiple edges at once (more efficient)
    fn add_edges(
        &mut self,
        bucket: GoalBucketId,
        edges: Vec<GraphEdge>,
    ) -> Result<(), GraphStoreError> {
        for edge in edges {
            self.add_edge(bucket, edge)?;
        }
        Ok(())
    }

    /// Scan nodes with a push-down filter.
    ///
    /// The default implementation loads all nodes and filters in Rust.
    /// Persistent stores (like `RedbGraphStore`) override this to scan
    /// the compact `NodeHeader` index first, only deserializing nodes
    /// that pass the filter.
    fn scan_nodes_filtered(
        &self,
        bucket: GoalBucketId,
        filter: &NodeFilter,
        limit: usize,
    ) -> Result<Vec<GraphNode>, GraphStoreError> {
        // Default: brute-force — load all, filter, truncate
        let all = self.get_all_nodes(bucket)?;
        let filtered: Vec<GraphNode> = all
            .into_iter()
            .filter(|node| {
                let header = NodeHeader::from_node(node, bucket);
                filter.matches_header(&header)
            })
            .take(limit)
            .collect();
        Ok(filtered)
    }
}

/// Errors that can occur during graph storage operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum GraphStoreError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Node not found: bucket={0}, node_id={1}")]
    NodeNotFound(GoalBucketId, NodeId),

    #[error("Edge not found: bucket={0}, edge_id={1}")]
    EdgeNotFound(GoalBucketId, EdgeId),

    #[error("Partition not found: {0}")]
    PartitionNotFound(GoalBucketId),

    #[error("Partition already loaded: {0}")]
    PartitionAlreadyLoaded(GoalBucketId),

    #[error("Invalid node ID: {0}")]
    InvalidNodeId(NodeId),

    #[error("Invalid bucket ID: {0}")]
    InvalidBucketId(GoalBucketId),

    #[error("Graph constraint violation: {0}")]
    ConstraintViolation(String),
}

// ============================================================================
// Push-down filters — pre-filter at the storage layer using NodeHeader
// ============================================================================

/// Push-down filter for storage-layer node scanning.
///
/// Instead of loading all nodes and filtering in Rust, these filters are
/// evaluated against the compact 42-byte `NodeHeader`, skipping full
/// deserialization for non-matching nodes. This reduces I/O and
/// deserialization work by up to 99% for selective queries.
#[derive(Debug, Clone)]
pub enum NodeFilter {
    /// Filter by node type discriminant (matches `NodeHeader.node_type_discriminant`)
    ByTypeDiscriminant(u8),
    /// Filter by updated_at >= timestamp
    UpdatedAfter(Timestamp),
    /// Filter by degree >= threshold
    MinDegree(u32),
    /// Filter by eviction tier (exact match)
    ByTier(EvictionTier),
    /// Filter by minimum signal strength
    MinSignal(f32),
    /// Logical AND: all sub-filters must match
    All(Vec<NodeFilter>),
    /// Logical OR: any sub-filter must match
    Any(Vec<NodeFilter>),
}

impl NodeFilter {
    /// Evaluate this filter against a `NodeHeader` (42 bytes, zero-alloc).
    pub fn matches_header(&self, header: &NodeHeader) -> bool {
        match self {
            NodeFilter::ByTypeDiscriminant(d) => header.node_type_discriminant == *d,
            NodeFilter::UpdatedAfter(ts) => header.updated_at >= *ts,
            NodeFilter::MinDegree(min) => header.degree >= *min,
            NodeFilter::ByTier(tier) => header.tier == *tier,
            NodeFilter::MinSignal(min) => header.signal >= *min,
            NodeFilter::All(filters) => filters.iter().all(|f| f.matches_header(header)),
            NodeFilter::Any(filters) => filters.iter().any(|f| f.matches_header(header)),
        }
    }

    /// Evaluate this filter directly against raw header bytes without
    /// constructing a full `NodeHeader`. Uses the fixed binary layout:
    ///   [0..8) node_id, [8) type_disc, [9..13) signal, [13) tier,
    ///   [14..18) degree, [18..26) updated_at, [26..34) created_at,
    ///   [34..42) goal_bucket
    pub fn matches_bytes(&self, bytes: &[u8]) -> bool {
        if bytes.len() < NODE_HEADER_BYTES {
            return false;
        }
        match self {
            NodeFilter::ByTypeDiscriminant(d) => bytes[8] == *d,
            NodeFilter::UpdatedAfter(ts) => {
                let updated = u64::from_le_bytes(bytes[18..26].try_into().unwrap_or([0; 8]));
                updated >= *ts
            },
            NodeFilter::MinDegree(min) => {
                let degree = u32::from_le_bytes(bytes[14..18].try_into().unwrap_or([0; 4]));
                degree >= *min
            },
            NodeFilter::ByTier(tier) => bytes[13] == *tier as u8,
            NodeFilter::MinSignal(min) => {
                let signal = f32::from_le_bytes(bytes[9..13].try_into().unwrap_or([0; 4]));
                signal >= *min
            },
            NodeFilter::All(filters) => filters.iter().all(|f| f.matches_bytes(bytes)),
            NodeFilter::Any(filters) => filters.iter().any(|f| f.matches_bytes(bytes)),
        }
    }
}

// ============================================================================
// In-memory implementation for testing
// ============================================================================

/// In-memory implementation of GraphStore for testing.
///
/// Stores structures.rs types directly. Edges are keyed by EdgeId
/// with forward/reverse adjacency lists tracking EdgeIds.
pub struct InMemoryGraphStore {
    nodes: FxHashMap<(GoalBucketId, NodeId), GraphNode>,
    edges: FxHashMap<(GoalBucketId, EdgeId), GraphEdge>,
    forward_edges: FxHashMap<(GoalBucketId, NodeId), AdjList>,
    reverse_edges: FxHashMap<(GoalBucketId, NodeId), AdjList>,
    headers: FxHashMap<(GoalBucketId, NodeId), NodeHeader>,
    loaded_partitions: std::collections::HashSet<GoalBucketId>,
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            edges: FxHashMap::default(),
            forward_edges: FxHashMap::default(),
            reverse_edges: FxHashMap::default(),
            headers: FxHashMap::default(),
            loaded_partitions: std::collections::HashSet::new(),
        }
    }
}

impl Default for InMemoryGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStore for InMemoryGraphStore {
    fn add_node(&mut self, bucket: GoalBucketId, node: GraphNode) -> Result<(), GraphStoreError> {
        self.nodes.insert((bucket, node.id), node);
        Ok(())
    }

    fn get_node(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Option<GraphNode>, GraphStoreError> {
        Ok(self.nodes.get(&(bucket, node_id)).cloned())
    }

    fn delete_node(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        self.nodes.remove(&(bucket, node_id));

        // Remove all outgoing edges
        if let Some(edge_ids) = self.forward_edges.remove(&(bucket, node_id)) {
            for eid in &edge_ids {
                if let Some(edge) = self.edges.remove(&(bucket, *eid)) {
                    // Clean up reverse adjacency of the target
                    if let Some(rev) = self.reverse_edges.get_mut(&(bucket, edge.target)) {
                        rev.retain(|e| e != eid);
                    }
                }
            }
        }

        // Remove all incoming edges
        if let Some(edge_ids) = self.reverse_edges.remove(&(bucket, node_id)) {
            for eid in &edge_ids {
                if let Some(edge) = self.edges.remove(&(bucket, *eid)) {
                    // Clean up forward adjacency of the source
                    if let Some(fwd) = self.forward_edges.get_mut(&(bucket, edge.source)) {
                        fwd.retain(|e| e != eid);
                    }
                }
            }
        }

        // Remove header
        self.headers.remove(&(bucket, node_id));

        Ok(())
    }

    fn has_node(&self, bucket: GoalBucketId, node_id: NodeId) -> Result<bool, GraphStoreError> {
        Ok(self.nodes.contains_key(&(bucket, node_id)))
    }

    fn get_all_nodes(&self, bucket: GoalBucketId) -> Result<Vec<GraphNode>, GraphStoreError> {
        Ok(self
            .nodes
            .iter()
            .filter(|((b, _), _)| *b == bucket)
            .map(|(_, node)| node.clone())
            .collect())
    }

    fn add_edge(&mut self, bucket: GoalBucketId, edge: GraphEdge) -> Result<(), GraphStoreError> {
        let edge_id = edge.id;
        let source = edge.source;
        let target = edge.target;

        // Add to forward adjacency
        self.forward_edges
            .entry((bucket, source))
            .or_default()
            .push(edge_id);

        // Add to reverse adjacency
        self.reverse_edges
            .entry((bucket, target))
            .or_default()
            .push(edge_id);

        // Store edge metadata
        self.edges.insert((bucket, edge_id), edge);

        Ok(())
    }

    fn get_edge(
        &self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError> {
        Ok(self.edges.get(&(bucket, edge_id)).cloned())
    }

    fn delete_edge(
        &mut self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<(), GraphStoreError> {
        if let Some(edge) = self.edges.remove(&(bucket, edge_id)) {
            // Remove from forward adjacency
            if let Some(fwd) = self.forward_edges.get_mut(&(bucket, edge.source)) {
                fwd.retain(|e| *e != edge_id);
            }

            // Remove from reverse adjacency
            if let Some(rev) = self.reverse_edges.get_mut(&(bucket, edge.target)) {
                rev.retain(|e| *e != edge_id);
            }
        }

        Ok(())
    }

    fn get_neighbors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let edge_ids = self
            .forward_edges
            .get(&(bucket, node_id))
            .cloned()
            .unwrap_or_default();

        let mut neighbors: Vec<NodeId> = edge_ids
            .iter()
            .filter_map(|eid| self.edges.get(&(bucket, *eid)).map(|e| e.target))
            .collect();
        neighbors.sort();
        neighbors.dedup();
        Ok(neighbors)
    }

    fn get_predecessors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let edge_ids = self
            .reverse_edges
            .get(&(bucket, node_id))
            .cloned()
            .unwrap_or_default();

        let mut preds: Vec<NodeId> = edge_ids
            .iter()
            .filter_map(|eid| self.edges.get(&(bucket, *eid)).map(|e| e.source))
            .collect();
        preds.sort();
        preds.dedup();
        Ok(preds)
    }

    fn get_outgoing_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let edge_ids = self
            .forward_edges
            .get(&(bucket, node_id))
            .cloned()
            .unwrap_or_default();

        Ok(edge_ids
            .iter()
            .filter_map(|eid| self.edges.get(&(bucket, *eid)).cloned())
            .collect())
    }

    fn get_incoming_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let edge_ids = self
            .reverse_edges
            .get(&(bucket, node_id))
            .cloned()
            .unwrap_or_default();

        Ok(edge_ids
            .iter()
            .filter_map(|eid| self.edges.get(&(bucket, *eid)).cloned())
            .collect())
    }

    fn store_header(
        &mut self,
        bucket: GoalBucketId,
        header: NodeHeader,
    ) -> Result<(), GraphStoreError> {
        self.headers.insert((bucket, header.node_id), header);
        Ok(())
    }

    fn scan_headers(&self, limit: usize) -> Result<Vec<NodeHeader>, GraphStoreError> {
        Ok(self.headers.values().take(limit).cloned().collect())
    }

    fn delete_header(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        self.headers.remove(&(bucket, node_id));
        Ok(())
    }

    fn load_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        if self.loaded_partitions.contains(&bucket) {
            return Err(GraphStoreError::PartitionAlreadyLoaded(bucket));
        }
        self.loaded_partitions.insert(bucket);
        Ok(())
    }

    fn unload_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        self.loaded_partitions.remove(&bucket);
        Ok(())
    }

    fn is_partition_loaded(&self, bucket: GoalBucketId) -> bool {
        self.loaded_partitions.contains(&bucket)
    }

    fn get_partition_stats(&self, bucket: GoalBucketId) -> Result<BucketInfo, GraphStoreError> {
        let node_count = self.nodes.keys().filter(|(b, _)| *b == bucket).count() as u64;
        let edge_count = self.edges.keys().filter(|(b, _)| *b == bucket).count() as u64;

        Ok(BucketInfo {
            bucket_id: bucket,
            node_count,
            edge_count,
            size_bytes: 0,
            last_modified: 0,
        })
    }

    fn get_all_buckets(&self) -> Result<Vec<GoalBucketId>, GraphStoreError> {
        let mut buckets: Vec<_> = self
            .nodes
            .keys()
            .map(|(b, _)| *b)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        buckets.sort();
        Ok(buckets)
    }

    fn traverse_bfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();

        queue.push_back((start, 0));
        visited.insert(start);

        while let Some((node, depth)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }

            result.push(node);

            if depth < max_depth {
                for neighbor in self.get_neighbors(bucket, node)? {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    fn traverse_dfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();

        fn dfs_helper(
            store: &InMemoryGraphStore,
            bucket: GoalBucketId,
            node: NodeId,
            depth: u32,
            max_depth: u32,
            visited: &mut std::collections::HashSet<NodeId>,
            result: &mut Vec<NodeId>,
        ) -> Result<(), GraphStoreError> {
            if depth > max_depth || visited.contains(&node) {
                return Ok(());
            }

            visited.insert(node);
            result.push(node);

            if depth < max_depth {
                for neighbor in store.get_neighbors(bucket, node)? {
                    dfs_helper(
                        store,
                        bucket,
                        neighbor,
                        depth + 1,
                        max_depth,
                        visited,
                        result,
                    )?;
                }
            }

            Ok(())
        }

        dfs_helper(self, bucket, start, 0, max_depth, &mut visited, &mut result)?;
        Ok(result)
    }

    fn find_paths(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
        max_depth: u32,
    ) -> Result<Vec<GraphPath>, GraphStoreError> {
        let mut paths = Vec::new();
        let mut current_path = Vec::new();
        let mut visited = std::collections::HashSet::new();

        #[allow(clippy::too_many_arguments)]
        fn find_paths_helper(
            store: &InMemoryGraphStore,
            bucket: GoalBucketId,
            current: NodeId,
            target: NodeId,
            depth: u32,
            max_depth: u32,
            current_path: &mut Vec<NodeId>,
            visited: &mut std::collections::HashSet<NodeId>,
            paths: &mut Vec<GraphPath>,
        ) -> Result<(), GraphStoreError> {
            if depth > max_depth {
                return Ok(());
            }

            current_path.push(current);
            visited.insert(current);

            if current == target {
                // Collect edges for this path
                let mut edges = Vec::new();
                let mut total_weight = 0.0;

                for window in current_path.windows(2) {
                    // Find an edge between window[0] and window[1]
                    let outgoing = store.get_outgoing_edges(bucket, window[0])?;
                    if let Some(edge) = outgoing.into_iter().find(|e| e.target == window[1]) {
                        total_weight += edge.weight;
                        edges.push(edge);
                    }
                }

                paths.push(GraphPath {
                    nodes: current_path.clone(),
                    edges,
                    total_weight,
                    length: current_path.len() - 1,
                });
            } else if depth < max_depth {
                for neighbor in store.get_neighbors(bucket, current)? {
                    if !visited.contains(&neighbor) {
                        find_paths_helper(
                            store,
                            bucket,
                            neighbor,
                            target,
                            depth + 1,
                            max_depth,
                            current_path,
                            visited,
                            paths,
                        )?;
                    }
                }
            }

            current_path.pop();
            visited.remove(&current);

            Ok(())
        }

        find_paths_helper(
            self,
            bucket,
            from,
            to,
            0,
            max_depth,
            &mut current_path,
            &mut visited,
            &mut paths,
        )?;
        Ok(paths)
    }

    fn get_subgraph(
        &self,
        bucket: GoalBucketId,
        center: NodeId,
        radius: u32,
    ) -> Result<Subgraph, GraphStoreError> {
        let node_ids = self.traverse_bfs(bucket, center, radius)?;

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for &node_id in &node_ids {
            if let Some(node) = self.get_node(bucket, node_id)? {
                nodes.push(node);
            }

            for edge in self.get_outgoing_edges(bucket, node_id)? {
                if node_ids.contains(&edge.target) {
                    edges.push(edge);
                }
            }
        }

        Ok(Subgraph {
            center,
            nodes,
            edges,
            radius,
        })
    }
}

// ============================================================================
// ShardedGraphStore — per-bucket locking for concurrent access
// ============================================================================

/// Internal shard holding all data for a single goal bucket.
struct BucketShard {
    nodes: FxHashMap<NodeId, GraphNode>,
    edges: FxHashMap<EdgeId, GraphEdge>,
    forward_edges: FxHashMap<NodeId, AdjList>,
    reverse_edges: FxHashMap<NodeId, AdjList>,
    headers: FxHashMap<NodeId, NodeHeader>,
}

impl BucketShard {
    fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            edges: FxHashMap::default(),
            forward_edges: FxHashMap::default(),
            reverse_edges: FxHashMap::default(),
            headers: FxHashMap::default(),
        }
    }
}

/// Sharded graph store with per-bucket `RwLock` for fine-grained concurrency.
///
/// Edges are **bucket-local only** — both endpoints must reside in the same bucket.
/// This is enforced at `add_edge()`.
pub struct ShardedGraphStore {
    shards: FxHashMap<GoalBucketId, parking_lot::RwLock<BucketShard>>,
    loaded_partitions: parking_lot::RwLock<std::collections::HashSet<GoalBucketId>>,
}

impl ShardedGraphStore {
    pub fn new() -> Self {
        Self {
            shards: FxHashMap::default(),
            loaded_partitions: parking_lot::RwLock::new(std::collections::HashSet::new()),
        }
    }

    fn ensure_shard(&mut self, bucket: GoalBucketId) {
        self.shards
            .entry(bucket)
            .or_insert_with(|| parking_lot::RwLock::new(BucketShard::new()));
    }
}

impl Default for ShardedGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStore for ShardedGraphStore {
    fn add_node(&mut self, bucket: GoalBucketId, node: GraphNode) -> Result<(), GraphStoreError> {
        self.ensure_shard(bucket);
        let shard = self.shards.get(&bucket).unwrap();
        let mut s = shard.write();
        s.nodes.insert(node.id, node);
        Ok(())
    }

    fn get_node(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Option<GraphNode>, GraphStoreError> {
        match self.shards.get(&bucket) {
            Some(shard) => Ok(shard.read().nodes.get(&node_id).cloned()),
            None => Ok(None),
        }
    }

    fn delete_node(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        let shard = match self.shards.get(&bucket) {
            Some(s) => s,
            None => return Ok(()),
        };
        let mut s = shard.write();
        s.nodes.remove(&node_id);

        // Remove outgoing edges
        if let Some(edge_ids) = s.forward_edges.remove(&node_id) {
            for eid in &edge_ids {
                if let Some(edge) = s.edges.remove(eid) {
                    if let Some(rev) = s.reverse_edges.get_mut(&edge.target) {
                        rev.retain(|e| e != eid);
                    }
                }
            }
        }

        // Remove incoming edges
        if let Some(edge_ids) = s.reverse_edges.remove(&node_id) {
            for eid in &edge_ids {
                if let Some(edge) = s.edges.remove(eid) {
                    if let Some(fwd) = s.forward_edges.get_mut(&edge.source) {
                        fwd.retain(|e| e != eid);
                    }
                }
            }
        }

        s.headers.remove(&node_id);
        Ok(())
    }

    fn has_node(&self, bucket: GoalBucketId, node_id: NodeId) -> Result<bool, GraphStoreError> {
        match self.shards.get(&bucket) {
            Some(shard) => Ok(shard.read().nodes.contains_key(&node_id)),
            None => Ok(false),
        }
    }

    fn get_all_nodes(&self, bucket: GoalBucketId) -> Result<Vec<GraphNode>, GraphStoreError> {
        match self.shards.get(&bucket) {
            Some(shard) => Ok(shard.read().nodes.values().cloned().collect()),
            None => Ok(Vec::new()),
        }
    }

    fn add_edge(&mut self, bucket: GoalBucketId, edge: GraphEdge) -> Result<(), GraphStoreError> {
        self.ensure_shard(bucket);
        let shard = self.shards.get(&bucket).unwrap();
        let mut s = shard.write();

        // Enforce bucket-local constraint
        if !s.nodes.contains_key(&edge.source) || !s.nodes.contains_key(&edge.target) {
            return Err(GraphStoreError::ConstraintViolation(
                "Both endpoints must be in the same bucket".to_string(),
            ));
        }

        let edge_id = edge.id;
        let source = edge.source;
        let target = edge.target;

        s.forward_edges.entry(source).or_default().push(edge_id);
        s.reverse_edges.entry(target).or_default().push(edge_id);
        s.edges.insert(edge_id, edge);
        Ok(())
    }

    fn get_edge(
        &self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError> {
        match self.shards.get(&bucket) {
            Some(shard) => Ok(shard.read().edges.get(&edge_id).cloned()),
            None => Ok(None),
        }
    }

    fn delete_edge(
        &mut self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<(), GraphStoreError> {
        let shard = match self.shards.get(&bucket) {
            Some(s) => s,
            None => return Ok(()),
        };
        let mut s = shard.write();
        if let Some(edge) = s.edges.remove(&edge_id) {
            if let Some(fwd) = s.forward_edges.get_mut(&edge.source) {
                fwd.retain(|e| *e != edge_id);
            }
            if let Some(rev) = s.reverse_edges.get_mut(&edge.target) {
                rev.retain(|e| *e != edge_id);
            }
        }
        Ok(())
    }

    fn get_neighbors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let shard = match self.shards.get(&bucket) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let s = shard.read();
        let edge_ids = match s.forward_edges.get(&node_id) {
            Some(ids) => ids,
            None => return Ok(Vec::new()),
        };
        let mut neighbors: Vec<NodeId> = edge_ids
            .iter()
            .filter_map(|eid| s.edges.get(eid).map(|e| e.target))
            .collect();
        neighbors.sort();
        neighbors.dedup();
        Ok(neighbors)
    }

    fn get_predecessors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let shard = match self.shards.get(&bucket) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let s = shard.read();
        let edge_ids = match s.reverse_edges.get(&node_id) {
            Some(ids) => ids,
            None => return Ok(Vec::new()),
        };
        let mut preds: Vec<NodeId> = edge_ids
            .iter()
            .filter_map(|eid| s.edges.get(eid).map(|e| e.source))
            .collect();
        preds.sort();
        preds.dedup();
        Ok(preds)
    }

    fn get_outgoing_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let shard = match self.shards.get(&bucket) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let s = shard.read();
        let edge_ids = match s.forward_edges.get(&node_id) {
            Some(ids) => ids,
            None => return Ok(Vec::new()),
        };
        Ok(edge_ids
            .iter()
            .filter_map(|eid| s.edges.get(eid).cloned())
            .collect())
    }

    fn get_incoming_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let shard = match self.shards.get(&bucket) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let s = shard.read();
        let edge_ids = match s.reverse_edges.get(&node_id) {
            Some(ids) => ids,
            None => return Ok(Vec::new()),
        };
        Ok(edge_ids
            .iter()
            .filter_map(|eid| s.edges.get(eid).cloned())
            .collect())
    }

    fn store_header(
        &mut self,
        bucket: GoalBucketId,
        header: NodeHeader,
    ) -> Result<(), GraphStoreError> {
        self.ensure_shard(bucket);
        let shard = self.shards.get(&bucket).unwrap();
        shard.write().headers.insert(header.node_id, header);
        Ok(())
    }

    fn scan_headers(&self, limit: usize) -> Result<Vec<NodeHeader>, GraphStoreError> {
        let mut result = Vec::new();
        for shard in self.shards.values() {
            let s = shard.read();
            for h in s.headers.values() {
                result.push(h.clone());
                if result.len() >= limit {
                    return Ok(result);
                }
            }
        }
        Ok(result)
    }

    fn delete_header(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        if let Some(shard) = self.shards.get(&bucket) {
            shard.write().headers.remove(&node_id);
        }
        Ok(())
    }

    fn load_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        {
            let mut loaded = self.loaded_partitions.write();
            if loaded.contains(&bucket) {
                return Err(GraphStoreError::PartitionAlreadyLoaded(bucket));
            }
            loaded.insert(bucket);
        }
        self.ensure_shard(bucket);
        Ok(())
    }

    fn unload_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        self.loaded_partitions.write().remove(&bucket);
        Ok(())
    }

    fn is_partition_loaded(&self, bucket: GoalBucketId) -> bool {
        self.loaded_partitions.read().contains(&bucket)
    }

    fn get_partition_stats(&self, bucket: GoalBucketId) -> Result<BucketInfo, GraphStoreError> {
        match self.shards.get(&bucket) {
            Some(shard) => {
                let s = shard.read();
                Ok(BucketInfo {
                    bucket_id: bucket,
                    node_count: s.nodes.len() as u64,
                    edge_count: s.edges.len() as u64,
                    size_bytes: 0,
                    last_modified: 0,
                })
            },
            None => Ok(BucketInfo {
                bucket_id: bucket,
                node_count: 0,
                edge_count: 0,
                size_bytes: 0,
                last_modified: 0,
            }),
        }
    }

    fn get_all_buckets(&self) -> Result<Vec<GoalBucketId>, GraphStoreError> {
        let mut buckets: Vec<_> = self.shards.keys().copied().collect();
        buckets.sort();
        Ok(buckets)
    }

    fn traverse_bfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();

        queue.push_back((start, 0));
        visited.insert(start);

        while let Some((node, depth)) = queue.pop_front() {
            result.push(node);
            if depth < max_depth {
                for neighbor in self.get_neighbors(bucket, node)? {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    fn traverse_dfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![(start, 0u32)];
        let mut result = Vec::new();

        while let Some((node, depth)) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }
            result.push(node);
            if depth < max_depth {
                for neighbor in self.get_neighbors(bucket, node)? {
                    if !visited.contains(&neighbor) {
                        stack.push((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    fn find_paths(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
        max_depth: u32,
    ) -> Result<Vec<GraphPath>, GraphStoreError> {
        let mut paths = Vec::new();
        let mut stack: Vec<(NodeId, Vec<NodeId>, u32)> = vec![(from, vec![from], 0)];

        while let Some((current, path, depth)) = stack.pop() {
            if current == to {
                let mut edges = Vec::new();
                let mut total_weight = 0.0;
                for w in path.windows(2) {
                    let outgoing = self.get_outgoing_edges(bucket, w[0])?;
                    if let Some(edge) = outgoing.into_iter().find(|e| e.target == w[1]) {
                        total_weight += edge.weight;
                        edges.push(edge);
                    }
                }
                let length = path.len() - 1;
                paths.push(GraphPath {
                    nodes: path,
                    edges,
                    total_weight,
                    length,
                });
                continue;
            }

            if depth >= max_depth {
                continue;
            }

            for neighbor in self.get_neighbors(bucket, current)? {
                if !path.contains(&neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    stack.push((neighbor, new_path, depth + 1));
                }
            }
        }

        Ok(paths)
    }

    fn get_subgraph(
        &self,
        bucket: GoalBucketId,
        center: NodeId,
        radius: u32,
    ) -> Result<Subgraph, GraphStoreError> {
        let node_ids = self.traverse_bfs(bucket, center, radius)?;
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let id_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();

        for &nid in &node_ids {
            if let Some(node) = self.get_node(bucket, nid)? {
                nodes.push(node);
            }
            for edge in self.get_outgoing_edges(bucket, nid)? {
                if id_set.contains(&edge.target) {
                    edges.push(edge);
                }
            }
        }

        Ok(Subgraph {
            center,
            nodes,
            edges,
            radius,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, NodeType};

    fn make_test_node(id: NodeId) -> GraphNode {
        GraphNode {
            id,
            node_type: NodeType::Event {
                event_id: id as u128,
                event_type: "test".to_string(),
                significance: 0.5,
            },
            created_at: 1000 + id,
            updated_at: 1000 + id,
            properties: std::collections::HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
        }
    }

    fn make_test_edge(id: EdgeId, source: NodeId, target: NodeId) -> GraphEdge {
        GraphEdge {
            id,
            source,
            target,
            edge_type: EdgeType::Causality {
                strength: 0.8,
                lag_ms: 100,
            },
            weight: 1.0,
            created_at: 2000,
            updated_at: 2000,
            valid_from: None,
            valid_until: None,
            observation_count: 1,
            confidence: 0.9,
            properties: std::collections::HashMap::new(),
            confidence_history: crate::tcell::TCell::Empty,
            weight_history: crate::tcell::TCell::Empty,
        }
    }

    #[test]
    fn test_add_and_get_node() {
        let mut store = InMemoryGraphStore::new();

        let node = make_test_node(100);
        store.add_node(42, node.clone()).unwrap();

        let retrieved = store.get_node(42, 100).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 100);
    }

    #[test]
    fn test_add_and_get_edge() {
        let mut store = InMemoryGraphStore::new();

        store.add_node(42, make_test_node(100)).unwrap();
        store.add_node(42, make_test_node(200)).unwrap();

        let edge = make_test_edge(1, 100, 200);
        store.add_edge(42, edge).unwrap();

        // Check neighbors
        let neighbors = store.get_neighbors(42, 100).unwrap();
        assert_eq!(neighbors, vec![200]);

        // Check predecessors
        let preds = store.get_predecessors(42, 200).unwrap();
        assert_eq!(preds, vec![100]);

        // Check get_edge by EdgeId
        let retrieved = store.get_edge(42, 1).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_delete_node_cleans_edges() {
        let mut store = InMemoryGraphStore::new();

        store.add_node(1, make_test_node(100)).unwrap();
        store.add_node(1, make_test_node(101)).unwrap();
        store.add_node(1, make_test_node(102)).unwrap();

        store.add_edge(1, make_test_edge(1, 100, 101)).unwrap();
        store.add_edge(1, make_test_edge(2, 102, 100)).unwrap();

        store.delete_node(1, 100).unwrap();

        // No dangling edges
        assert!(store.get_edge(1, 1).unwrap().is_none());
        assert!(store.get_edge(1, 2).unwrap().is_none());

        // No dangling adjacency
        let n102 = store.get_neighbors(1, 102).unwrap();
        assert!(!n102.contains(&100));

        let p101 = store.get_predecessors(1, 101).unwrap();
        assert!(!p101.contains(&100));
    }

    #[test]
    fn test_header_operations() {
        let mut store = InMemoryGraphStore::new();

        let node = make_test_node(100);
        let header = NodeHeader::from_node(&node, 42);
        store.store_header(42, header.clone()).unwrap();

        let headers = store.scan_headers(100).unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].node_id, 100);

        store.delete_header(42, 100).unwrap();
        let headers = store.scan_headers(100).unwrap();
        assert_eq!(headers.len(), 0);
    }

    #[test]
    fn test_traverse_bfs() {
        let mut store = InMemoryGraphStore::new();

        // 1 → 2 → 3, 2 → 4
        for i in 1..=4 {
            store.add_node(42, make_test_node(i)).unwrap();
        }

        store.add_edge(42, make_test_edge(1, 1, 2)).unwrap();
        store.add_edge(42, make_test_edge(2, 2, 3)).unwrap();
        store.add_edge(42, make_test_edge(3, 2, 4)).unwrap();

        let visited = store.traverse_bfs(42, 1, 2).unwrap();
        assert_eq!(visited.len(), 4);
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
        assert!(visited.contains(&4));
    }

    // ================================================================
    // ShardedGraphStore tests
    // ================================================================

    #[test]
    fn sharded_add_get_node() {
        let mut store = ShardedGraphStore::new();
        let node = make_test_node(100);
        store.add_node(1, node.clone()).unwrap();

        let retrieved = store.get_node(1, 100).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 100);

        // Different bucket → not found
        assert!(store.get_node(2, 100).unwrap().is_none());
    }

    #[test]
    fn sharded_add_get_edge() {
        let mut store = ShardedGraphStore::new();
        store.add_node(1, make_test_node(100)).unwrap();
        store.add_node(1, make_test_node(200)).unwrap();

        let edge = make_test_edge(1, 100, 200);
        store.add_edge(1, edge).unwrap();

        let neighbors = store.get_neighbors(1, 100).unwrap();
        assert_eq!(neighbors, vec![200]);

        let preds = store.get_predecessors(1, 200).unwrap();
        assert_eq!(preds, vec![100]);
    }

    #[test]
    fn sharded_cross_bucket_edge_rejected() {
        let mut store = ShardedGraphStore::new();
        store.add_node(1, make_test_node(100)).unwrap();
        store.add_node(2, make_test_node(200)).unwrap();

        // Edge spans buckets — should fail
        let edge = make_test_edge(1, 100, 200);
        let result = store.add_edge(1, edge);
        assert!(result.is_err());
    }

    #[test]
    fn sharded_delete_node() {
        let mut store = ShardedGraphStore::new();
        store.add_node(1, make_test_node(100)).unwrap();
        store.add_node(1, make_test_node(200)).unwrap();
        store.add_edge(1, make_test_edge(1, 100, 200)).unwrap();

        store.delete_node(1, 100).unwrap();
        assert!(store.get_node(1, 100).unwrap().is_none());
        assert!(store.get_edge(1, 1).unwrap().is_none());
    }

    #[test]
    fn sharded_outgoing_edges() {
        let mut store = ShardedGraphStore::new();
        store.add_node(1, make_test_node(100)).unwrap();
        store.add_node(1, make_test_node(200)).unwrap();
        store.add_edge(1, make_test_edge(1, 100, 200)).unwrap();

        let edges = store.get_outgoing_edges(1, 100).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, 200);
    }

    #[test]
    fn sharded_partition_ops() {
        let mut store = ShardedGraphStore::new();
        assert!(!store.is_partition_loaded(1));

        store.load_partition(1).unwrap();
        assert!(store.is_partition_loaded(1));

        // Double-load fails
        assert!(store.load_partition(1).is_err());

        store.unload_partition(1).unwrap();
        assert!(!store.is_partition_loaded(1));
    }

    #[test]
    fn sharded_traverse_bfs() {
        let mut store = ShardedGraphStore::new();
        for i in 1..=4 {
            store.add_node(1, make_test_node(i)).unwrap();
        }
        store.add_edge(1, make_test_edge(1, 1, 2)).unwrap();
        store.add_edge(1, make_test_edge(2, 2, 3)).unwrap();
        store.add_edge(1, make_test_edge(3, 2, 4)).unwrap();

        let visited = store.traverse_bfs(1, 1, 2).unwrap();
        assert_eq!(visited.len(), 4);
    }

    #[test]
    fn sharded_get_all_buckets() {
        let mut store = ShardedGraphStore::new();
        store.add_node(10, make_test_node(1)).unwrap();
        store.add_node(20, make_test_node(2)).unwrap();

        let buckets = store.get_all_buckets().unwrap();
        assert!(buckets.contains(&10));
        assert!(buckets.contains(&20));
    }

    #[test]
    fn sharded_has_node() {
        let mut store = ShardedGraphStore::new();
        store.add_node(1, make_test_node(100)).unwrap();
        assert!(store.has_node(1, 100).unwrap());
        assert!(!store.has_node(1, 999).unwrap());
        assert!(!store.has_node(99, 100).unwrap());
    }
}
