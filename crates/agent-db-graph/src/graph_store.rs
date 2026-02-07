// Graph persistence layer - Persistent storage for graph structure
//
// Provides trait-based abstraction for storing graph nodes and edges in persistent storage.
// Implementation uses goal-bucket partitioning for semantic sharding and efficient memory management.

use crate::structures::{GoalBucketId, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Graph node metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub id: NodeId,
    pub node_type: GraphNodeType,
    pub label: String,
    pub context_hash: u64,
    pub created_at: u64,
    pub properties: serde_json::Value,
}

/// Types of nodes in the event graph
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GraphNodeType {
    Event,
    Action,
    Observation,
    Cognitive,
    Communication,
    Learning,
    Context,
    Pattern,
}

/// Edge metadata connecting two nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: GraphEdgeType,
    pub weight: f32,
    pub confidence: f32,
    pub created_at: u64,
}

/// Types of edges in the event graph
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GraphEdgeType {
    Causality,   // A caused B
    Temporal,    // A happened before B
    Similarity,  // A is similar to B
    Containment, // A contains B (episode contains events)
    Reference,   // A references B (context reference)
    Transition,  // State transition A → B
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
/// Implementations should support:
/// - Node and edge CRUD operations
/// - Partition loading/unloading for memory management
/// - Efficient neighbor queries (forward and reverse)
/// - Graph traversal operations
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
    // Edge Operations
    // ============================================================================

    /// Add an edge between two nodes
    fn add_edge(&mut self, bucket: GoalBucketId, edge: GraphEdge) -> Result<(), GraphStoreError>;

    /// Get edge metadata
    fn get_edge(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError>;

    /// Delete an edge
    fn delete_edge(
        &mut self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
    ) -> Result<(), GraphStoreError>;

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

    #[error("Edge not found: bucket={0}, from={1}, to={2}")]
    EdgeNotFound(GoalBucketId, NodeId, NodeId),

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

/// In-memory implementation for testing
pub struct InMemoryGraphStore {
    nodes: HashMap<(GoalBucketId, NodeId), GraphNode>,
    forward_edges: HashMap<(GoalBucketId, NodeId), Vec<NodeId>>,
    reverse_edges: HashMap<(GoalBucketId, NodeId), Vec<NodeId>>,
    edge_metadata: HashMap<(GoalBucketId, NodeId, NodeId), GraphEdge>,
    loaded_partitions: std::collections::HashSet<GoalBucketId>,
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            edge_metadata: HashMap::new(),
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
        self.forward_edges.remove(&(bucket, node_id));
        self.reverse_edges.remove(&(bucket, node_id));
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
        // Add to forward adjacency
        self.forward_edges
            .entry((bucket, edge.from))
            .or_default()
            .push(edge.to);

        // Add to reverse adjacency
        self.reverse_edges
            .entry((bucket, edge.to))
            .or_default()
            .push(edge.from);

        // Store edge metadata
        self.edge_metadata
            .insert((bucket, edge.from, edge.to), edge);

        Ok(())
    }

    fn get_edge(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError> {
        Ok(self.edge_metadata.get(&(bucket, from, to)).cloned())
    }

    fn delete_edge(
        &mut self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
    ) -> Result<(), GraphStoreError> {
        // Remove from forward adjacency
        if let Some(neighbors) = self.forward_edges.get_mut(&(bucket, from)) {
            neighbors.retain(|&n| n != to);
        }

        // Remove from reverse adjacency
        if let Some(preds) = self.reverse_edges.get_mut(&(bucket, to)) {
            preds.retain(|&n| n != from);
        }

        // Remove metadata
        self.edge_metadata.remove(&(bucket, from, to));

        Ok(())
    }

    fn get_neighbors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        Ok(self
            .forward_edges
            .get(&(bucket, node_id))
            .cloned()
            .unwrap_or_default())
    }

    fn get_predecessors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        Ok(self
            .reverse_edges
            .get(&(bucket, node_id))
            .cloned()
            .unwrap_or_default())
    }

    fn get_outgoing_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let neighbors = self.get_neighbors(bucket, node_id)?;
        Ok(neighbors
            .iter()
            .filter_map(|&to| self.edge_metadata.get(&(bucket, node_id, to)).cloned())
            .collect())
    }

    fn get_incoming_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let preds = self.get_predecessors(bucket, node_id)?;
        Ok(preds
            .iter()
            .filter_map(|&from| self.edge_metadata.get(&(bucket, from, node_id)).cloned())
            .collect())
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
        let edge_count = self
            .edge_metadata
            .keys()
            .filter(|(b, _, _)| *b == bucket)
            .count() as u64;

        Ok(BucketInfo {
            bucket_id: bucket,
            node_count,
            edge_count,
            size_bytes: 0, // Not tracked in memory implementation
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
                // Found a path, construct GraphPath
                let mut edges = Vec::new();
                let mut total_weight = 0.0;

                for window in current_path.windows(2) {
                    if let Some(edge) = store.get_edge(bucket, window[0], window[1])? {
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
                if node_ids.contains(&edge.to) {
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

    #[test]
    fn test_add_and_get_node() {
        let mut store = InMemoryGraphStore::new();

        let node = GraphNode {
            id: 100,
            node_type: GraphNodeType::Event,
            label: "test_event".to_string(),
            context_hash: 12345,
            created_at: 1000,
            properties: serde_json::json!({"test": "value"}),
        };

        store.add_node(42, node.clone()).unwrap();

        let retrieved = store.get_node(42, 100).unwrap();
        assert_eq!(retrieved, Some(node));
    }

    #[test]
    fn test_add_and_get_edge() {
        let mut store = InMemoryGraphStore::new();

        // Add nodes first
        let node1 = GraphNode {
            id: 100,
            node_type: GraphNodeType::Event,
            label: "event1".to_string(),
            context_hash: 1,
            created_at: 1000,
            properties: serde_json::json!({}),
        };

        let node2 = GraphNode {
            id: 200,
            node_type: GraphNodeType::Event,
            label: "event2".to_string(),
            context_hash: 2,
            created_at: 2000,
            properties: serde_json::json!({}),
        };

        store.add_node(42, node1).unwrap();
        store.add_node(42, node2).unwrap();

        // Add edge
        let edge = GraphEdge {
            from: 100,
            to: 200,
            edge_type: GraphEdgeType::Causality,
            weight: 1.0,
            confidence: 0.95,
            created_at: 1500,
        };

        store.add_edge(42, edge.clone()).unwrap();

        // Check neighbors
        let neighbors = store.get_neighbors(42, 100).unwrap();
        assert_eq!(neighbors, vec![200]);

        // Check predecessors
        let preds = store.get_predecessors(42, 200).unwrap();
        assert_eq!(preds, vec![100]);
    }

    #[test]
    fn test_traverse_bfs() {
        let mut store = InMemoryGraphStore::new();

        // Create a simple graph: 1 → 2 → 3
        //                           ↓
        //                           4
        for i in 1..=4 {
            store
                .add_node(
                    42,
                    GraphNode {
                        id: i,
                        node_type: GraphNodeType::Event,
                        label: format!("node{}", i),
                        context_hash: i,
                        created_at: i * 1000,
                        properties: serde_json::json!({}),
                    },
                )
                .unwrap();
        }

        store
            .add_edge(
                42,
                GraphEdge {
                    from: 1,
                    to: 2,
                    edge_type: GraphEdgeType::Causality,
                    weight: 1.0,
                    confidence: 1.0,
                    created_at: 1000,
                },
            )
            .unwrap();

        store
            .add_edge(
                42,
                GraphEdge {
                    from: 2,
                    to: 3,
                    edge_type: GraphEdgeType::Causality,
                    weight: 1.0,
                    confidence: 1.0,
                    created_at: 2000,
                },
            )
            .unwrap();

        store
            .add_edge(
                42,
                GraphEdge {
                    from: 2,
                    to: 4,
                    edge_type: GraphEdgeType::Causality,
                    weight: 1.0,
                    confidence: 1.0,
                    created_at: 2500,
                },
            )
            .unwrap();

        let visited = store.traverse_bfs(42, 1, 2).unwrap();
        assert_eq!(visited.len(), 4); // Should visit all nodes within depth 2
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
        assert!(visited.contains(&4));
    }
}
