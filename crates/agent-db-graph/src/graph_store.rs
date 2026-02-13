// Graph persistence layer - Persistent storage for graph structure
//
// Provides trait-based abstraction for storing graph nodes and edges in persistent storage.
// Implementation uses goal-bucket partitioning for semantic sharding and efficient memory management.
//
// Types are unified with structures.rs — GraphNode/GraphEdge/NodeType/EdgeType are the
// single source of truth. NodeHeader provides a lightweight scoring record (~40 bytes)
// for streaming importance scoring without full deserialization.

use crate::structures::{EdgeId, GoalBucketId, NodeId};
use agent_db_core::types::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
// In-memory implementation for testing
// ============================================================================

/// In-memory implementation of GraphStore for testing.
///
/// Stores structures.rs types directly. Edges are keyed by EdgeId
/// with forward/reverse adjacency lists tracking EdgeIds.
pub struct InMemoryGraphStore {
    nodes: HashMap<(GoalBucketId, NodeId), GraphNode>,
    edges: HashMap<(GoalBucketId, EdgeId), GraphEdge>,
    forward_edges: HashMap<(GoalBucketId, NodeId), Vec<EdgeId>>,
    reverse_edges: HashMap<(GoalBucketId, NodeId), Vec<EdgeId>>,
    headers: HashMap<(GoalBucketId, NodeId), NodeHeader>,
    loaded_partitions: std::collections::HashSet<GoalBucketId>,
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            forward_edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            headers: HashMap::new(),
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
            observation_count: 1,
            confidence: 0.9,
            properties: std::collections::HashMap::new(),
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
}
