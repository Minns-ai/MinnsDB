// Persistent graph storage using redb
//
// Implements GraphStore trait with redb backend, using:
// - Hierarchical keys (inspired by Dgraph)
// - Delta-encoded adjacency lists for compression
// - Goal-bucket partitioning for semantic sharding
// - LRU partition loading for memory efficiency
//
// **Unified types**: Uses structures.rs GraphNode/GraphEdge directly.
// Edge key uses EdgeId for multi-edge support.
// NodeHeader stored under 0x06 prefix for fast streaming scans.

use crate::compression::CompressedAdjacencyList;
use crate::graph_store::{
    BucketInfo, GraphEdge, GraphNode, GraphPath, GraphStore, GraphStoreError, NodeHeader, Subgraph,
};
use crate::structures::{EdgeId, GoalBucketId, NodeId};
use agent_db_storage::{BatchOperation, RedbBackend};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Stack-allocated adjacency list for up to 8 edges (64 bytes inline).
/// Spills to the heap for high-degree nodes, same as Vec.
type AdjList = SmallVec<[EdgeId; 8]>;

// Table names for redb
const TABLE_GRAPH_NODES: &str = "graph_nodes";
const TABLE_GRAPH_ADJACENCY: &str = "graph_adjacency";
const TABLE_GRAPH_EDGES: &str = "graph_edges";
// ============================================================================
// Key Design (Hierarchical, inspired by Dgraph)
// ============================================================================

/// Key type prefixes
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum KeyType {
    NodeMeta = 0x01,         // Node metadata
    AdjacencyForward = 0x02, // Forward edges (A -> [B, C, D])
    AdjacencyReverse = 0x03, // Reverse edges (backlinks)
    EdgeMeta = 0x04,         // Edge metadata by EdgeId
    HeaderMeta = 0x06,       // NodeHeader for fast scoring
    DirEdgeOut = 0x07,       // Direction-encoded outgoing: [bucket][source][edge_id] → edge
    DirEdgeIn = 0x08,        // Direction-encoded incoming: [bucket][target][edge_id] → edge
}

/// Build hierarchical key: [TypeByte][GoalBucket(8)][NodeID(8)]
fn make_node_key(bucket: GoalBucketId, node_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::NodeMeta as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Build adjacency list key (forward edges)
fn make_adjacency_forward_key(bucket: GoalBucketId, node_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::AdjacencyForward as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Build reverse adjacency key (backlinks)
fn make_adjacency_reverse_key(bucket: GoalBucketId, node_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::AdjacencyReverse as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Build edge metadata key: [0x04][bucket:8][edge_id:8]
fn make_edge_key(bucket: GoalBucketId, edge_id: EdgeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::EdgeMeta as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&edge_id.to_be_bytes());
    key
}

/// Build header key: [0x06][bucket:8][node_id:8]
fn make_header_key(bucket: GoalBucketId, node_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::HeaderMeta as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Build direction-encoded outgoing edge key: [0x07][bucket:8][source:8][edge_id:8]
///
/// Enables prefix scan on `[0x07][bucket][source]` to find all outgoing edges
/// with full metadata, bypassing the adjacency list + individual edge lookup.
fn make_dir_out_key(bucket: GoalBucketId, source: NodeId, edge_id: EdgeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8 + 8);
    key.push(KeyType::DirEdgeOut as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&source.to_be_bytes());
    key.extend_from_slice(&edge_id.to_be_bytes());
    key
}

/// Build direction-encoded outgoing prefix: [0x07][bucket:8][source:8]
fn make_dir_out_prefix(bucket: GoalBucketId, source: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::DirEdgeOut as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&source.to_be_bytes());
    key
}

/// Build direction-encoded incoming edge key: [0x08][bucket:8][target:8][edge_id:8]
fn make_dir_in_key(bucket: GoalBucketId, target: NodeId, edge_id: EdgeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8 + 8);
    key.push(KeyType::DirEdgeIn as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&target.to_be_bytes());
    key.extend_from_slice(&edge_id.to_be_bytes());
    key
}

/// Build direction-encoded incoming prefix: [0x08][bucket:8][target:8]
fn make_dir_in_prefix(bucket: GoalBucketId, target: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8);
    key.push(KeyType::DirEdgeIn as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&target.to_be_bytes());
    key
}

// ============================================================================
// Small helpers
// ============================================================================

#[inline]
fn insert_sorted_unique(v: &mut AdjList, x: EdgeId) -> bool {
    match v.binary_search(&x) {
        Ok(_) => false,
        Err(i) => {
            v.insert(i, x);
            true
        },
    }
}

#[inline]
fn remove_sorted(v: &mut AdjList, x: EdgeId) -> bool {
    match v.binary_search(&x) {
        Ok(i) => {
            v.remove(i);
            true
        },
        Err(_) => false,
    }
}

#[inline]
fn json_bytes<T: serde::Serialize>(v: &T) -> Result<Vec<u8>, GraphStoreError> {
    serde_json::to_vec(v).map_err(|e| GraphStoreError::Serialization(e.to_string()))
}

#[inline]
fn msgpack_bytes<T: serde::Serialize>(v: &T) -> Result<Vec<u8>, GraphStoreError> {
    rmp_serde::to_vec(v).map_err(|e| GraphStoreError::Serialization(e.to_string()))
}

#[inline]
fn op_put(table: &str, key: Vec<u8>, value: Vec<u8>) -> BatchOperation {
    BatchOperation::Put {
        table_name: table.to_string(),
        key,
        value,
    }
}

#[inline]
fn op_del(table: &str, key: Vec<u8>) -> BatchOperation {
    BatchOperation::Delete {
        table_name: table.to_string(),
        key,
    }
}

// ============================================================================
// Partition Cache (for LRU management)
// ============================================================================

// Monotonic counter for deterministic LRU ordering.
static LRU_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_lru_tick() -> u64 {
    LRU_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Cached partition data (loaded into memory)
/// Uses structures.rs types directly.
#[derive(Debug)]
struct PartitionCache {
    nodes: HashMap<NodeId, GraphNode>,
    /// Forward adjacency: node_id -> sorted SmallVec of EdgeIds (inline ≤8)
    forward_edges: HashMap<NodeId, AdjList>,
    /// Reverse adjacency: node_id -> sorted SmallVec of EdgeIds (inline ≤8)
    reverse_edges: HashMap<NodeId, AdjList>,
    /// Edge metadata by EdgeId
    edge_metadata: HashMap<EdgeId, GraphEdge>,
    last_accessed: AtomicU64,
}

impl PartitionCache {
    fn new(_bucket_id: GoalBucketId) -> Self {
        Self {
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            edge_metadata: HashMap::new(),
            last_accessed: AtomicU64::new(next_lru_tick()),
        }
    }

    fn touch(&self) {
        self.last_accessed.store(next_lru_tick(), Ordering::Relaxed);
    }

    fn last_accessed(&self) -> u64 {
        self.last_accessed.load(Ordering::Relaxed)
    }

    /// Estimated memory footprint in bytes for this partition.
    /// Used by the weighted LRU eviction policy so large partitions
    /// are evicted before small ones when the memory budget is tight.
    fn estimated_bytes(&self) -> usize {
        // Average sizes including HashMap per-entry overhead (~64 bytes).
        const NODE_BYTES: usize = 256; // GraphNode + HashMap overhead
        const EDGE_BYTES: usize = 128; // GraphEdge + HashMap overhead
        const ADJ_ENTRY_BYTES: usize = 80; // HashMap entry + SmallVec inline (64+16)

        self.nodes.len() * NODE_BYTES
            + self.edge_metadata.len() * EDGE_BYTES
            + (self.forward_edges.len() + self.reverse_edges.len()) * ADJ_ENTRY_BYTES
    }
}

// ============================================================================
// RedbGraphStore Implementation
// ============================================================================

/// Default memory budget for the partition cache: 256 MiB.
const DEFAULT_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

/// Persistent graph store using redb backend with unified structures.rs types.
pub struct RedbGraphStore {
    backend: Arc<RedbBackend>,

    // Partition cache (weighted LRU)
    loaded_partitions: HashMap<GoalBucketId, PartitionCache>,
    /// Hard cap on loaded partitions (safety net).
    max_loaded_partitions: usize,
    /// Soft memory budget in bytes. When total cached bytes exceed this,
    /// the heaviest + stalest partitions are evicted.
    memory_budget_bytes: usize,
}

impl RedbGraphStore {
    /// Create a new RedbGraphStore with a default 256 MiB memory budget.
    pub fn new(backend: Arc<RedbBackend>, max_loaded_partitions: usize) -> Self {
        Self::with_memory_budget(backend, max_loaded_partitions, DEFAULT_MEMORY_BUDGET)
    }

    /// Create a new RedbGraphStore with an explicit memory budget.
    pub fn with_memory_budget(
        backend: Arc<RedbBackend>,
        max_loaded_partitions: usize,
        memory_budget_bytes: usize,
    ) -> Self {
        tracing::info!(
            "Initializing RedbGraphStore: max_partitions={}, memory_budget={}MiB",
            max_loaded_partitions,
            memory_budget_bytes / (1024 * 1024)
        );

        Self {
            backend,
            loaded_partitions: HashMap::new(),
            max_loaded_partitions,
            memory_budget_bytes,
        }
    }

    /// Total estimated bytes across all loaded partitions.
    fn total_cached_bytes(&self) -> usize {
        self.loaded_partitions
            .values()
            .map(|p| p.estimated_bytes())
            .sum()
    }

    /// Get the underlying redb backend.
    pub fn backend(&self) -> &Arc<RedbBackend> {
        &self.backend
    }

    /// Helper: get with JSON deserialization
    fn get_json<K, V>(&self, table: &str, key: K) -> Result<Option<V>, GraphStoreError>
    where
        K: AsRef<[u8]>,
        V: serde::de::DeserializeOwned,
    {
        match self
            .backend
            .get_raw(table, key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(bytes) => {
                let value = serde_json::from_slice(&bytes)
                    .map_err(|e| GraphStoreError::Serialization(e.to_string()))?;
                Ok(Some(value))
            },
            None => Ok(None),
        }
    }

    /// Helper: scan prefix with JSON deserialization
    fn scan_prefix_json<K, V>(
        &self,
        table: &str,
        prefix: K,
    ) -> Result<Vec<(Vec<u8>, V)>, GraphStoreError>
    where
        K: AsRef<[u8]>,
        V: serde::de::DeserializeOwned,
    {
        let raw_results = self
            .backend
            .scan_prefix_raw(table, prefix)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        let mut results = Vec::new();
        for (key, value_bytes) in raw_results {
            let value = serde_json::from_slice(&value_bytes)
                .map_err(|e| GraphStoreError::Serialization(e.to_string()))?;
            results.push((key, value));
        }
        Ok(results)
    }

    /// Ensure partition is loaded into memory
    fn ensure_partition_loaded(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        if self.loaded_partitions.contains_key(&bucket) {
            if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
                partition.touch();
            }
            return Ok(());
        }

        tracing::debug!("Loading partition {} from disk", bucket);
        self.load_partition_internal(bucket)?;

        // Evict until we're under both limits (hard partition cap + memory budget).
        while self.loaded_partitions.len() > self.max_loaded_partitions
            || self.total_cached_bytes() > self.memory_budget_bytes
        {
            // Never evict below 1 partition (the one we just loaded).
            if self.loaded_partitions.len() <= 1 {
                break;
            }
            self.evict_weighted_partition()?;
        }

        Ok(())
    }

    /// Load partition from disk into cache
    fn load_partition_internal(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        let mut partition = PartitionCache::new(bucket);

        // Load all nodes for this bucket (JSON-serialized structures::GraphNode)
        let node_prefix = {
            let mut p = vec![KeyType::NodeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        for (_, node) in
            self.scan_prefix_json::<Vec<u8>, GraphNode>(TABLE_GRAPH_NODES, node_prefix)?
        {
            partition.nodes.insert(node.id, node);
        }

        // Load forward adjacency lists (msgpack CompressedAdjacencyList storing EdgeIds)
        let fwd_prefix = {
            let mut p = vec![KeyType::AdjacencyForward as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        for (key, compressed) in self
            .backend
            .scan_prefix::<Vec<u8>, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, fwd_prefix)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            if key.len() != 17 {
                return Err(GraphStoreError::Storage(
                    "Invalid adjacency key length".to_string(),
                ));
            }

            let node_id = u64::from_be_bytes(
                key[9..17]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );

            partition
                .forward_edges
                .insert(node_id, SmallVec::from_vec(compressed.decompress()));
        }

        // Load reverse adjacency lists (msgpack CompressedAdjacencyList storing EdgeIds)
        let rev_prefix = {
            let mut p = vec![KeyType::AdjacencyReverse as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        for (key, compressed) in self
            .backend
            .scan_prefix::<Vec<u8>, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, rev_prefix)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            if key.len() != 17 {
                return Err(GraphStoreError::Storage(
                    "Invalid adjacency key length".to_string(),
                ));
            }

            let node_id = u64::from_be_bytes(
                key[9..17]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );

            partition
                .reverse_edges
                .insert(node_id, SmallVec::from_vec(compressed.decompress()));
        }

        // Load edge metadata (JSON-serialized structures::GraphEdge, keyed by EdgeId)
        let edge_prefix = {
            let mut p = vec![KeyType::EdgeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        for (key, edge) in
            self.scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, edge_prefix)?
        {
            if key.len() != 17 {
                return Err(GraphStoreError::Storage(
                    "Invalid edge key length".to_string(),
                ));
            }

            let edge_id = u64::from_be_bytes(
                key[9..17]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );

            partition.edge_metadata.insert(edge_id, edge);
        }

        tracing::info!(
            "Loaded partition {}: {} nodes, {} forward lists, {} reverse lists, {} edges",
            bucket,
            partition.nodes.len(),
            partition.forward_edges.len(),
            partition.reverse_edges.len(),
            partition.edge_metadata.len()
        );

        self.loaded_partitions.insert(bucket, partition);
        Ok(())
    }

    /// Evict the partition with the highest eviction score.
    ///
    /// Score = `staleness * memory_weight`, where:
    /// - `staleness` = `current_tick - last_accessed`  (higher = staler)
    /// - `memory_weight` = `estimated_bytes / 1024`    (larger partitions penalised)
    ///
    /// This keeps small, recently-used partitions in cache while aggressively
    /// evicting large, stale ones.
    fn evict_weighted_partition(&mut self) -> Result<(), GraphStoreError> {
        if self.loaded_partitions.is_empty() {
            return Ok(());
        }

        let current_tick = LRU_COUNTER.load(Ordering::Relaxed);

        let evict_bucket = self
            .loaded_partitions
            .iter()
            .map(|(&bucket_id, partition)| {
                let staleness = current_tick
                    .saturating_sub(partition.last_accessed())
                    .max(1);
                let mem_weight = (partition.estimated_bytes() / 1024).max(1) as u64;
                let score = staleness.saturating_mul(mem_weight);
                (bucket_id, score)
            })
            .max_by_key(|(_, score)| *score)
            .map(|(bucket_id, _)| bucket_id)
            .ok_or_else(|| GraphStoreError::Storage("No partitions to evict".to_string()))?;

        tracing::debug!(
            "Evicting partition {} (budget {:.1}MiB / {:.1}MiB)",
            evict_bucket,
            self.total_cached_bytes() as f64 / (1024.0 * 1024.0),
            self.memory_budget_bytes as f64 / (1024.0 * 1024.0),
        );
        self.loaded_partitions.remove(&evict_bucket);
        Ok(())
    }

    // ========================================================================
    // Pathfinding helpers for find_paths()
    // ========================================================================

    /// Depth-limited DFS to find all paths between two nodes within a loaded partition.
    ///
    /// Operates directly on the PartitionCache for zero-copy adjacency lookups.
    /// Collects up to `max_paths` complete paths with edge metadata and cost
    /// derived from `edge_cost()` (type-aware: strength, confidence, similarity).
    #[allow(clippy::too_many_arguments)]
    fn dfs_all_paths(
        partition: &PartitionCache,
        current: NodeId,
        target: NodeId,
        max_depth: u32,
        path_nodes: &mut Vec<NodeId>,
        path_edges: &mut Vec<GraphEdge>,
        visited: &mut HashSet<NodeId>,
        results: &mut Vec<GraphPath>,
        max_paths: usize,
    ) {
        if results.len() >= max_paths {
            return;
        }

        path_nodes.push(current);
        visited.insert(current);

        if current == target {
            let total_weight: f32 = path_edges.iter().map(crate::traversal::edge_cost).sum();
            results.push(GraphPath {
                nodes: path_nodes.clone(),
                edges: path_edges.clone(),
                total_weight,
                length: path_nodes.len() - 1,
            });
        } else if (path_nodes.len() as u32) <= max_depth {
            if let Some(edge_ids) = partition.forward_edges.get(&current) {
                for &eid in edge_ids {
                    if results.len() >= max_paths {
                        break;
                    }
                    if let Some(edge) = partition.edge_metadata.get(&eid) {
                        let neighbor = edge.target;
                        if !visited.contains(&neighbor) {
                            path_edges.push(edge.clone());
                            Self::dfs_all_paths(
                                partition, neighbor, target, max_depth, path_nodes, path_edges,
                                visited, results, max_paths,
                            );
                            path_edges.pop();
                        }
                    }
                }
            }
        }

        path_nodes.pop();
        visited.remove(&current);
    }

    /// Fallback `find_paths` when the partition is not memory-resident.
    ///
    /// Uses disk-based edge lookups via `get_outgoing_edges`. Slower than
    /// the cached path, but avoids loading the full partition just for a
    /// single pathfinding query.
    fn find_paths_unloaded(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
        max_depth: u32,
    ) -> Result<Vec<GraphPath>, GraphStoreError> {
        let mut results: Vec<GraphPath> = Vec::new();
        let mut path_nodes: Vec<NodeId> = Vec::new();
        let mut path_edges: Vec<GraphEdge> = Vec::new();
        let mut visited: HashSet<NodeId> = HashSet::new();

        self.dfs_all_paths_unloaded(
            bucket,
            from,
            to,
            max_depth,
            &mut path_nodes,
            &mut path_edges,
            &mut visited,
            &mut results,
            10,
        )?;

        results.sort_by(|a, b| a.total_weight.total_cmp(&b.total_weight));
        Ok(results)
    }

    /// Disk-backed DFS for unloaded partitions.
    ///
    /// Reads outgoing edges from storage on each step. Each call to
    /// `get_outgoing_edges` may hit the redb backend directly, so this
    /// path should only be used when partition loading is undesirable.
    #[allow(clippy::too_many_arguments)]
    fn dfs_all_paths_unloaded(
        &self,
        bucket: GoalBucketId,
        current: NodeId,
        target: NodeId,
        max_depth: u32,
        path_nodes: &mut Vec<NodeId>,
        path_edges: &mut Vec<GraphEdge>,
        visited: &mut HashSet<NodeId>,
        results: &mut Vec<GraphPath>,
        max_paths: usize,
    ) -> Result<(), GraphStoreError> {
        if results.len() >= max_paths {
            return Ok(());
        }

        path_nodes.push(current);
        visited.insert(current);

        if current == target {
            let total_weight: f32 = path_edges.iter().map(crate::traversal::edge_cost).sum();
            results.push(GraphPath {
                nodes: path_nodes.clone(),
                edges: path_edges.clone(),
                total_weight,
                length: path_nodes.len() - 1,
            });
        } else if (path_nodes.len() as u32) <= max_depth {
            let outgoing = self.get_outgoing_edges(bucket, current)?;
            for edge in outgoing {
                if results.len() >= max_paths {
                    break;
                }
                let neighbor = edge.target;
                if !visited.contains(&neighbor) {
                    path_edges.push(edge);
                    self.dfs_all_paths_unloaded(
                        bucket, neighbor, target, max_depth, path_nodes, path_edges, visited,
                        results, max_paths,
                    )?;
                    path_edges.pop();
                }
            }
        }

        path_nodes.pop();
        visited.remove(&current);
        Ok(())
    }

    /// Internal: apply delete_edge changes into the in-memory cache and append batch ops.
    fn delete_edge_in_batch(
        &mut self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
        ops: &mut Vec<BatchOperation>,
    ) -> Result<(), GraphStoreError> {
        let partition = self
            .loaded_partitions
            .get_mut(&bucket)
            .ok_or_else(|| GraphStoreError::Storage("Partition not loaded".to_string()))?;

        // Look up the edge to find source/target
        let edge = match partition.edge_metadata.remove(&edge_id) {
            Some(e) => e,
            None => return Ok(()), // Edge already gone
        };

        let from = edge.source;
        let to = edge.target;

        // Forward adjacency (from -> edge_ids)
        {
            let mut edge_ids = partition
                .forward_edges
                .get(&from)
                .cloned()
                .unwrap_or_default();
            let changed = remove_sorted(&mut edge_ids, edge_id);
            if changed {
                let fwd_key = make_adjacency_forward_key(bucket, from);

                if edge_ids.is_empty() {
                    ops.push(op_del(TABLE_GRAPH_ADJACENCY, fwd_key));
                    partition.forward_edges.remove(&from);
                } else {
                    let compressed = CompressedAdjacencyList::compress(&edge_ids);
                    ops.push(op_put(
                        TABLE_GRAPH_ADJACENCY,
                        fwd_key,
                        msgpack_bytes(&compressed)?,
                    ));
                    partition.forward_edges.insert(from, edge_ids);
                }
            }
        }

        // Reverse adjacency (to -> edge_ids)
        {
            let mut edge_ids = partition
                .reverse_edges
                .get(&to)
                .cloned()
                .unwrap_or_default();
            let changed = remove_sorted(&mut edge_ids, edge_id);
            if changed {
                let rev_key = make_adjacency_reverse_key(bucket, to);

                if edge_ids.is_empty() {
                    ops.push(op_del(TABLE_GRAPH_ADJACENCY, rev_key));
                    partition.reverse_edges.remove(&to);
                } else {
                    let compressed = CompressedAdjacencyList::compress(&edge_ids);
                    ops.push(op_put(
                        TABLE_GRAPH_ADJACENCY,
                        rev_key,
                        msgpack_bytes(&compressed)?,
                    ));
                    partition.reverse_edges.insert(to, edge_ids);
                }
            }
        }

        // Delete edge metadata
        ops.push(op_del(TABLE_GRAPH_EDGES, make_edge_key(bucket, edge_id)));

        // Delete direction-encoded keys
        ops.push(op_del(
            TABLE_GRAPH_EDGES,
            make_dir_out_key(bucket, from, edge_id),
        ));
        ops.push(op_del(
            TABLE_GRAPH_EDGES,
            make_dir_in_key(bucket, to, edge_id),
        ));

        Ok(())
    }
}

impl GraphStore for RedbGraphStore {
    // ========================================================================
    // Node Operations
    // ========================================================================

    fn add_node(&mut self, bucket: GoalBucketId, node: GraphNode) -> Result<(), GraphStoreError> {
        self.ensure_partition_loaded(bucket)?;

        let key = make_node_key(bucket, node.id);
        let ops = vec![op_put(TABLE_GRAPH_NODES, key, json_bytes(&node)?)];

        self.backend
            .write_batch(ops)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
            partition.nodes.insert(node.id, node);
            partition.touch();
        }

        Ok(())
    }

    fn get_node(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Option<GraphNode>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            partition.touch();
            return Ok(partition.nodes.get(&node_id).cloned());
        }

        let key = make_node_key(bucket, node_id);
        self.get_json(TABLE_GRAPH_NODES, &key)
    }

    fn delete_node(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        self.ensure_partition_loaded(bucket)?;

        // Snapshot incident edge IDs from cache
        let (in_edge_ids, out_edge_ids) = {
            let p = self
                .loaded_partitions
                .get(&bucket)
                .ok_or_else(|| GraphStoreError::Storage("Partition not loaded".to_string()))?;

            let in_eids = p.reverse_edges.get(&node_id).cloned().unwrap_or_default();
            let out_eids = p.forward_edges.get(&node_id).cloned().unwrap_or_default();
            (in_eids, out_eids)
        };

        let mut ops: Vec<BatchOperation> = Vec::new();

        // Remove incoming edges
        for edge_id in in_edge_ids {
            self.delete_edge_in_batch(bucket, edge_id, &mut ops)?;
        }

        // Remove outgoing edges
        for edge_id in out_edge_ids {
            self.delete_edge_in_batch(bucket, edge_id, &mut ops)?;
        }

        // Delete node record
        ops.push(op_del(TABLE_GRAPH_NODES, make_node_key(bucket, node_id)));

        // Delete header
        ops.push(op_del(TABLE_GRAPH_NODES, make_header_key(bucket, node_id)));

        // Delete node's own adjacency keys (should be absent after edge deletion, but safe)
        ops.push(op_del(
            TABLE_GRAPH_ADJACENCY,
            make_adjacency_forward_key(bucket, node_id),
        ));
        ops.push(op_del(
            TABLE_GRAPH_ADJACENCY,
            make_adjacency_reverse_key(bucket, node_id),
        ));

        self.backend
            .write_batch(ops)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        // Update cache
        if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
            partition.nodes.remove(&node_id);
            partition.forward_edges.remove(&node_id);
            partition.reverse_edges.remove(&node_id);
            partition.touch();
        }

        Ok(())
    }

    fn has_node(&self, bucket: GoalBucketId, node_id: NodeId) -> Result<bool, GraphStoreError> {
        Ok(self.get_node(bucket, node_id)?.is_some())
    }

    fn get_all_nodes(&self, bucket: GoalBucketId) -> Result<Vec<GraphNode>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(partition.nodes.values().cloned().collect());
        }

        let prefix = {
            let mut p = vec![KeyType::NodeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        let results = self.scan_prefix_json::<Vec<u8>, GraphNode>(TABLE_GRAPH_NODES, prefix)?;
        Ok(results.into_iter().map(|(_, node)| node).collect())
    }

    // ========================================================================
    // Edge Operations (EdgeId-based)
    // ========================================================================

    fn add_edge(&mut self, bucket: GoalBucketId, edge: GraphEdge) -> Result<(), GraphStoreError> {
        self.ensure_partition_loaded(bucket)?;

        let edge_id = edge.id;
        let source = edge.source;
        let target = edge.target;

        // Fetch current adjacency from cache
        let (mut fwd_edge_ids, mut rev_edge_ids) = {
            let p = self
                .loaded_partitions
                .get(&bucket)
                .ok_or_else(|| GraphStoreError::Storage("Partition not loaded".to_string()))?;
            (
                p.forward_edges.get(&source).cloned().unwrap_or_default(),
                p.reverse_edges.get(&target).cloned().unwrap_or_default(),
            )
        };

        let changed_fwd = insert_sorted_unique(&mut fwd_edge_ids, edge_id);
        let changed_rev = insert_sorted_unique(&mut rev_edge_ids, edge_id);

        let mut ops: Vec<BatchOperation> = Vec::with_capacity(3);

        if changed_fwd {
            let fwd_key = make_adjacency_forward_key(bucket, source);
            let compressed = CompressedAdjacencyList::compress(&fwd_edge_ids);
            ops.push(op_put(
                TABLE_GRAPH_ADJACENCY,
                fwd_key,
                msgpack_bytes(&compressed)?,
            ));
        }

        if changed_rev {
            let rev_key = make_adjacency_reverse_key(bucket, target);
            let compressed = CompressedAdjacencyList::compress(&rev_edge_ids);
            ops.push(op_put(
                TABLE_GRAPH_ADJACENCY,
                rev_key,
                msgpack_bytes(&compressed)?,
            ));
        }

        // Store edge metadata (JSON)
        let edge_bytes = json_bytes(&edge)?;
        ops.push(op_put(
            TABLE_GRAPH_EDGES,
            make_edge_key(bucket, edge_id),
            edge_bytes.clone(),
        ));

        // Direction-encoded keys for on-disk directional traversal
        ops.push(op_put(
            TABLE_GRAPH_EDGES,
            make_dir_out_key(bucket, source, edge_id),
            edge_bytes.clone(),
        ));
        ops.push(op_put(
            TABLE_GRAPH_EDGES,
            make_dir_in_key(bucket, target, edge_id),
            edge_bytes,
        ));

        self.backend
            .write_batch(ops)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        // Update cache
        if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
            partition.forward_edges.insert(source, fwd_edge_ids);
            partition.reverse_edges.insert(target, rev_edge_ids);
            partition.edge_metadata.insert(edge_id, edge);
            partition.touch();
        }

        Ok(())
    }

    fn get_edge(
        &self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(partition.edge_metadata.get(&edge_id).cloned());
        }

        let key = make_edge_key(bucket, edge_id);
        self.get_json(TABLE_GRAPH_EDGES, &key)
    }

    fn delete_edge(
        &mut self,
        bucket: GoalBucketId,
        edge_id: EdgeId,
    ) -> Result<(), GraphStoreError> {
        self.ensure_partition_loaded(bucket)?;

        let mut ops: Vec<BatchOperation> = Vec::with_capacity(3);
        self.delete_edge_in_batch(bucket, edge_id, &mut ops)?;

        self.backend
            .write_batch(ops)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
            partition.touch();
        }

        Ok(())
    }

    fn get_neighbors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            let edge_ids = partition
                .forward_edges
                .get(&node_id)
                .cloned()
                .unwrap_or_default();
            let mut neighbors: Vec<NodeId> = edge_ids
                .iter()
                .filter_map(|eid| partition.edge_metadata.get(eid).map(|e| e.target))
                .collect();
            neighbors.sort();
            neighbors.dedup();
            return Ok(neighbors);
        }

        // Direction-encoded prefix scan: single pass
        let prefix = make_dir_out_prefix(bucket, node_id);
        let results = self.scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, prefix)?;
        if !results.is_empty() {
            let mut neighbors: Vec<NodeId> =
                results.into_iter().map(|(_, edge)| edge.target).collect();
            neighbors.sort();
            neighbors.dedup();
            return Ok(neighbors);
        }

        // Fallback for data written before direction-encoded keys
        let key = make_adjacency_forward_key(bucket, node_id);
        match self
            .backend
            .get::<_, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(compressed) => {
                let edge_ids = compressed.decompress();
                let mut neighbors = Vec::new();
                for eid in edge_ids {
                    if let Some(edge) = self.get_edge(bucket, eid)? {
                        neighbors.push(edge.target);
                    }
                }
                neighbors.sort();
                neighbors.dedup();
                Ok(neighbors)
            },
            None => Ok(Vec::new()),
        }
    }

    fn get_predecessors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            let edge_ids = partition
                .reverse_edges
                .get(&node_id)
                .cloned()
                .unwrap_or_default();
            let mut preds: Vec<NodeId> = edge_ids
                .iter()
                .filter_map(|eid| partition.edge_metadata.get(eid).map(|e| e.source))
                .collect();
            preds.sort();
            preds.dedup();
            return Ok(preds);
        }

        // Direction-encoded prefix scan: single pass
        let prefix = make_dir_in_prefix(bucket, node_id);
        let results = self.scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, prefix)?;
        if !results.is_empty() {
            let mut preds: Vec<NodeId> = results.into_iter().map(|(_, edge)| edge.source).collect();
            preds.sort();
            preds.dedup();
            return Ok(preds);
        }

        // Fallback for data written before direction-encoded keys
        let key = make_adjacency_reverse_key(bucket, node_id);
        match self
            .backend
            .get::<_, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(compressed) => {
                let edge_ids = compressed.decompress();
                let mut preds = Vec::new();
                for eid in edge_ids {
                    if let Some(edge) = self.get_edge(bucket, eid)? {
                        preds.push(edge.source);
                    }
                }
                preds.sort();
                preds.dedup();
                Ok(preds)
            },
            None => Ok(Vec::new()),
        }
    }

    fn get_outgoing_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            let edge_ids = partition
                .forward_edges
                .get(&node_id)
                .cloned()
                .unwrap_or_default();
            return Ok(edge_ids
                .iter()
                .filter_map(|eid| partition.edge_metadata.get(eid).cloned())
                .collect());
        }

        // Direction-encoded prefix scan: single pass, no adjacency list lookup
        let prefix = make_dir_out_prefix(bucket, node_id);
        let results = self.scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, prefix)?;
        if !results.is_empty() {
            return Ok(results.into_iter().map(|(_, edge)| edge).collect());
        }

        // Fallback for data written before direction-encoded keys
        let key = make_adjacency_forward_key(bucket, node_id);
        match self
            .backend
            .get::<_, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(compressed) => {
                let edge_ids = compressed.decompress();
                let mut edges = Vec::new();
                for eid in edge_ids {
                    if let Some(edge) = self.get_edge(bucket, eid)? {
                        edges.push(edge);
                    }
                }
                Ok(edges)
            },
            None => Ok(Vec::new()),
        }
    }

    fn get_incoming_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            let edge_ids = partition
                .reverse_edges
                .get(&node_id)
                .cloned()
                .unwrap_or_default();
            return Ok(edge_ids
                .iter()
                .filter_map(|eid| partition.edge_metadata.get(eid).cloned())
                .collect());
        }

        // Direction-encoded prefix scan: single pass, no adjacency list lookup
        let prefix = make_dir_in_prefix(bucket, node_id);
        let results = self.scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, prefix)?;
        if !results.is_empty() {
            return Ok(results.into_iter().map(|(_, edge)| edge).collect());
        }

        // Fallback for data written before direction-encoded keys
        let key = make_adjacency_reverse_key(bucket, node_id);
        match self
            .backend
            .get::<_, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(compressed) => {
                let edge_ids = compressed.decompress();
                let mut edges = Vec::new();
                for eid in edge_ids {
                    if let Some(edge) = self.get_edge(bucket, eid)? {
                        edges.push(edge);
                    }
                }
                Ok(edges)
            },
            None => Ok(Vec::new()),
        }
    }

    // ========================================================================
    // Header Operations
    // ========================================================================

    fn store_header(
        &mut self,
        bucket: GoalBucketId,
        header: NodeHeader,
    ) -> Result<(), GraphStoreError> {
        let key = make_header_key(bucket, header.node_id);
        // Fixed 42-byte binary layout — ~4x faster to read than JSON during scans.
        let ops = vec![op_put(TABLE_GRAPH_NODES, key, header.to_bytes().to_vec())];

        self.backend
            .write_batch(ops)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        Ok(())
    }

    fn scan_headers(&self, limit: usize) -> Result<Vec<NodeHeader>, GraphStoreError> {
        let prefix = vec![KeyType::HeaderMeta as u8];
        let raw_results = self
            .backend
            .scan_prefix_raw(TABLE_GRAPH_NODES, prefix)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        let mut headers = Vec::with_capacity(raw_results.len().min(limit));
        for (_key, value) in raw_results {
            if headers.len() >= limit {
                break;
            }
            // Try compact binary first (42 bytes), fall back to JSON for legacy data.
            if let Some(h) = NodeHeader::from_bytes(&value) {
                headers.push(h);
            } else if let Ok(h) = serde_json::from_slice::<NodeHeader>(&value) {
                headers.push(h);
            }
        }
        Ok(headers)
    }

    fn delete_header(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        let key = make_header_key(bucket, node_id);
        let ops = vec![op_del(TABLE_GRAPH_NODES, key)];

        self.backend
            .write_batch(ops)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        Ok(())
    }

    // ========================================================================
    // Partition Operations
    // ========================================================================

    fn load_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        if self.loaded_partitions.contains_key(&bucket) {
            return Err(GraphStoreError::PartitionAlreadyLoaded(bucket));
        }

        self.load_partition_internal(bucket)
    }

    fn unload_partition(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        self.loaded_partitions.remove(&bucket);
        tracing::debug!("Unloaded partition {}", bucket);
        Ok(())
    }

    fn is_partition_loaded(&self, bucket: GoalBucketId) -> bool {
        self.loaded_partitions.contains_key(&bucket)
    }

    fn get_partition_stats(&self, bucket: GoalBucketId) -> Result<BucketInfo, GraphStoreError> {
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(BucketInfo {
                bucket_id: bucket,
                node_count: partition.nodes.len() as u64,
                edge_count: partition.edge_metadata.len() as u64,
                size_bytes: 0,
                last_modified: 0,
            });
        }

        let node_prefix = {
            let mut p = vec![KeyType::NodeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        let node_count = self
            .scan_prefix_json::<Vec<u8>, GraphNode>(TABLE_GRAPH_NODES, node_prefix)?
            .len() as u64;

        let edge_prefix = {
            let mut p = vec![KeyType::EdgeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        let edge_count = self
            .scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, edge_prefix)?
            .len() as u64;

        Ok(BucketInfo {
            bucket_id: bucket,
            node_count,
            edge_count,
            size_bytes: 0,
            last_modified: 0,
        })
    }

    fn get_all_buckets(&self) -> Result<Vec<GoalBucketId>, GraphStoreError> {
        let mut buckets = HashSet::new();

        let prefix = vec![KeyType::NodeMeta as u8];
        for (key, _) in self.scan_prefix_json::<Vec<u8>, GraphNode>(TABLE_GRAPH_NODES, prefix)? {
            if key.len() >= 9 {
                let bucket = u64::from_be_bytes(
                    key[1..9]
                        .try_into()
                        .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
                );
                buckets.insert(bucket);
            }
        }

        let mut result: Vec<_> = buckets.into_iter().collect();
        result.sort();
        Ok(result)
    }

    // ========================================================================
    // Traversal Operations
    // ========================================================================

    fn traverse_bfs(
        &self,
        bucket: GoalBucketId,
        start: NodeId,
        max_depth: u32,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
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
                    if visited.insert(neighbor) {
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
        let mut visited = HashSet::new();
        let mut result = Vec::new();

        fn dfs_helper<S: GraphStore>(
            store: &S,
            bucket: GoalBucketId,
            node: NodeId,
            depth: u32,
            max_depth: u32,
            visited: &mut HashSet<NodeId>,
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
        // Depth-limited DFS to find all paths (up to 10) within the partition.
        // Uses the partition cache for fast adjacency lookups.
        let partition = match self.loaded_partitions.get(&bucket) {
            Some(p) => {
                p.touch();
                p
            },
            None => {
                // Fall back to single Dijkstra-like BFS if partition not loaded
                return self.find_paths_unloaded(bucket, from, to, max_depth);
            },
        };

        let mut results: Vec<GraphPath> = Vec::new();
        let mut current_path_nodes: Vec<NodeId> = Vec::new();
        let mut current_path_edges: Vec<GraphEdge> = Vec::new();
        let mut visited: HashSet<NodeId> = HashSet::new();

        Self::dfs_all_paths(
            partition,
            from,
            to,
            max_depth,
            &mut current_path_nodes,
            &mut current_path_edges,
            &mut visited,
            &mut results,
            10, // max paths to return
        );

        // Sort by total weight (ascending = cheapest first)
        results.sort_by(|a, b| a.total_weight.total_cmp(&b.total_weight));
        Ok(results)
    }

    fn get_subgraph(
        &self,
        bucket: GoalBucketId,
        center: NodeId,
        radius: u32,
    ) -> Result<Subgraph, GraphStoreError> {
        let node_ids = self.traverse_bfs(bucket, center, radius)?;
        let node_set: HashSet<NodeId> = node_ids.iter().copied().collect();

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for &node_id in &node_ids {
            if let Some(node) = self.get_node(bucket, node_id)? {
                nodes.push(node);
            }

            for edge in self.get_outgoing_edges(bucket, node_id)? {
                if node_set.contains(&edge.target) {
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

    /// Optimized push-down filter scan.
    ///
    /// Scans the compact 42-byte `NodeHeader` index first, evaluating the
    /// filter against raw bytes without full deserialization. Only nodes
    /// whose headers pass the filter are loaded from the node table.
    ///
    /// For a partition with 10K nodes but only 50 matching, this reduces
    /// deserialization from 10K `GraphNode`s to 50.
    fn scan_nodes_filtered(
        &self,
        bucket: GoalBucketId,
        filter: &crate::graph_store::NodeFilter,
        limit: usize,
    ) -> Result<Vec<GraphNode>, GraphStoreError> {
        // Phase 1: scan header index with raw-byte filter evaluation
        let header_prefix = {
            let mut p = Vec::with_capacity(9);
            p.push(KeyType::HeaderMeta as u8);
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        let raw_headers = self
            .backend
            .scan_prefix_raw(TABLE_GRAPH_NODES, header_prefix)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        let mut matching_ids: Vec<NodeId> = Vec::new();

        for (_key, value) in &raw_headers {
            if matching_ids.len() >= limit {
                break;
            }
            // Evaluate filter against raw header bytes — zero-alloc fast path
            if filter.matches_bytes(value) {
                if let Some(header) = NodeHeader::from_bytes(value) {
                    matching_ids.push(header.node_id);
                }
            }
        }

        // Phase 2: load only matching full nodes
        let mut nodes = Vec::with_capacity(matching_ids.len());
        for node_id in matching_ids {
            if let Some(node) = self.get_node(bucket, node_id)? {
                nodes.push(node);
            }
        }

        Ok(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, NodeType};
    use agent_db_storage::{RedbBackend, RedbConfig};
    use tempfile::TempDir;

    fn create_test_store() -> (RedbGraphStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.redb");

        let config = RedbConfig {
            data_path: db_path,
            cache_size_bytes: 64 * 1024 * 1024,
            repair_on_open: false,
        };

        let backend = Arc::new(RedbBackend::open(config).unwrap());
        let store = RedbGraphStore::new(backend, 3);
        (store, temp_dir)
    }

    fn create_test_node(id: NodeId, _bucket: GoalBucketId) -> GraphNode {
        GraphNode {
            id,
            node_type: NodeType::Event {
                event_id: id as u128,
                event_type: format!("test_event_{}", id),
                significance: 0.5,
            },
            created_at: 1000 + id,
            updated_at: 1000 + id,
            properties: {
                let mut props = std::collections::HashMap::new();
                props.insert("test".to_string(), serde_json::json!("node"));
                props
            },
            degree: 0,
            embedding: Vec::new(),
        }
    }

    fn create_test_edge(id: EdgeId, from: NodeId, to: NodeId) -> GraphEdge {
        GraphEdge {
            id,
            source: from,
            target: to,
            edge_type: EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
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
    fn test_edge_batching_and_delete_node_cleanup() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        for id in [100u64, 101u64, 102u64] {
            store
                .add_node(bucket, create_test_node(id, bucket))
                .unwrap();
        }

        // 102 -> 100 (edge_id=1), 100 -> 101 (edge_id=2)
        store
            .add_edge(bucket, create_test_edge(1, 102, 100))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(2, 100, 101))
            .unwrap();

        store.delete_node(bucket, 100).unwrap();

        // No dangling adjacency references
        let n102 = store.get_neighbors(bucket, 102).unwrap();
        assert!(!n102.contains(&100));

        let p101 = store.get_predecessors(bucket, 101).unwrap();
        assert!(!p101.contains(&100));

        // No dangling edge metadata
        assert!(store.get_edge(bucket, 1).unwrap().is_none());
        assert!(store.get_edge(bucket, 2).unwrap().is_none());
    }

    #[test]
    fn test_node_crud_operations() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;
        let node = create_test_node(100, bucket);

        // Create
        store.add_node(bucket, node.clone()).unwrap();

        // Read
        let retrieved = store.get_node(bucket, 100).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 100);

        // Update (add same node with updated properties)
        let mut updated_node = node.clone();
        updated_node
            .properties
            .insert("test".to_string(), serde_json::json!("updated"));
        store.add_node(bucket, updated_node).unwrap();

        let retrieved = store.get_node(bucket, 100).unwrap();
        assert_eq!(
            retrieved.unwrap().properties.get("test"),
            Some(&serde_json::json!("updated"))
        );

        // Delete
        store.delete_node(bucket, 100).unwrap();
        let retrieved = store.get_node(bucket, 100).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_edge_operations() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        // Add nodes
        for i in 100..105 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
        }

        // Add edges: 100 -> 101 -> 102 -> 103 -> 104
        for (eid, i) in (100..104).enumerate() {
            store
                .add_edge(bucket, create_test_edge(eid as u64 + 1, i, i + 1))
                .unwrap();
        }

        // Test forward adjacency
        let neighbors = store.get_neighbors(bucket, 100).unwrap();
        assert_eq!(neighbors, vec![101]);

        // Test reverse adjacency
        let predecessors = store.get_predecessors(bucket, 101).unwrap();
        assert_eq!(predecessors, vec![100]);

        // Test getting edges
        let outgoing = store.get_outgoing_edges(bucket, 100).unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].target, 101);

        let incoming = store.get_incoming_edges(bucket, 101).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].source, 100);

        // Test edge deletion by EdgeId
        store.delete_edge(bucket, 1).unwrap(); // edge 100->101
        let neighbors = store.get_neighbors(bucket, 100).unwrap();
        assert_eq!(neighbors.len(), 0);

        let predecessors = store.get_predecessors(bucket, 101).unwrap();
        assert_eq!(predecessors.len(), 0);
    }

    #[test]
    fn test_partition_loading_and_unloading() {
        let (mut store, _temp) = create_test_store();

        for bucket in 1..=3 {
            for i in 0..5 {
                let node_id = (bucket * 100) + i;
                store
                    .add_node(bucket, create_test_node(node_id, bucket))
                    .unwrap();
            }
        }

        assert_eq!(store.loaded_partitions.len(), 3);

        store.unload_partition(1).unwrap();
        assert_eq!(store.loaded_partitions.len(), 2);
        assert!(!store.loaded_partitions.contains_key(&1));

        store.load_partition(1).unwrap();
        assert_eq!(store.loaded_partitions.len(), 3);
        assert!(store.loaded_partitions.contains_key(&1));

        let node = store.get_node(1, 100).unwrap();
        assert!(node.is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let (mut store, _temp) = create_test_store();

        for bucket in 1..=4 {
            for i in 0..3 {
                let node_id = (bucket * 100) + i;
                store
                    .add_node(bucket, create_test_node(node_id, bucket))
                    .unwrap();
            }
        }

        assert_eq!(store.loaded_partitions.len(), 3);

        store.get_node(2, 200).unwrap();

        store.add_node(5, create_test_node(500, 5)).unwrap();
        assert_eq!(store.loaded_partitions.len(), 3);
        assert!(store.loaded_partitions.contains_key(&2));
        assert!(store.loaded_partitions.contains_key(&5));
    }

    #[test]
    fn test_persistence_across_restarts() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.redb");

        {
            let config = RedbConfig {
                data_path: db_path.clone(),
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            };
            let backend = Arc::new(RedbBackend::open(config).unwrap());
            let mut store = RedbGraphStore::new(backend, 3);

            let bucket = 1;
            for i in 100..105 {
                store.add_node(bucket, create_test_node(i, bucket)).unwrap();
            }

            for (eid, i) in (100..104).enumerate() {
                store
                    .add_edge(bucket, create_test_edge(eid as u64 + 1, i, i + 1))
                    .unwrap();
            }

            store.unload_partition(bucket).unwrap();
        }

        {
            let config = RedbConfig {
                data_path: db_path,
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            };
            let backend = Arc::new(RedbBackend::open(config).unwrap());
            let store = RedbGraphStore::new(backend, 3);

            let node = store.get_node(1, 100).unwrap();
            assert!(node.is_some());

            let neighbors = store.get_neighbors(1, 100).unwrap();
            assert_eq!(neighbors, vec![101]);

            for i in 100..105 {
                let node = store.get_node(1, i).unwrap();
                assert!(node.is_some(), "Node {} should exist", i);
            }

            let edge = store.get_edge(1, 1).unwrap();
            assert!(edge.is_some());
        }
    }

    #[test]
    fn test_bfs_traversal() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        for i in 100..105 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
        }

        store
            .add_edge(bucket, create_test_edge(1, 100, 101))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(2, 100, 102))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(3, 101, 103))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(4, 102, 104))
            .unwrap();

        let result = store.traverse_bfs(bucket, 100, 1).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains(&100));
        assert!(result.contains(&101));
        assert!(result.contains(&102));

        let result = store.traverse_bfs(bucket, 100, 2).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_dfs_traversal() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        for i in 100..104 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
        }

        for (eid, i) in (100..103).enumerate() {
            store
                .add_edge(bucket, create_test_edge(eid as u64 + 1, i, i + 1))
                .unwrap();
        }

        let result = store.traverse_dfs(bucket, 100, 1).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 100);
        assert!(result.contains(&101));

        let result = store.traverse_dfs(bucket, 100, 3).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_subgraph_extraction() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        store
            .add_node(bucket, create_test_node(100, bucket))
            .unwrap();
        for (eid, i) in (101..105).enumerate() {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
            store
                .add_edge(bucket, create_test_edge(eid as u64 + 1, 100, i))
                .unwrap();
        }

        let subgraph = store.get_subgraph(bucket, 100, 1).unwrap();
        assert_eq!(subgraph.center, 100);
        assert_eq!(subgraph.radius, 1);
        assert_eq!(subgraph.nodes.len(), 5);
        assert_eq!(subgraph.edges.len(), 4);

        let subgraph = store.get_subgraph(bucket, 100, 0).unwrap();
        assert_eq!(subgraph.nodes.len(), 1);
        assert_eq!(subgraph.edges.len(), 0);
    }

    #[test]
    fn test_multi_bucket_operations() {
        let (mut store, _temp) = create_test_store();

        for bucket in 1..=3 {
            for i in 0..5 {
                let node_id = (bucket * 100) + i;
                store
                    .add_node(bucket, create_test_node(node_id, bucket))
                    .unwrap();
            }
        }

        for bucket in 1..=3 {
            for i in 0..5 {
                let node_id = (bucket * 100) + i;
                let node = store.get_node(bucket, node_id).unwrap();
                assert!(node.is_some());
            }
        }

        let node = store.get_node(2, 100).unwrap();
        assert!(node.is_none());

        let node = store.get_node(1, 200).unwrap();
        assert!(node.is_none());
    }

    #[test]
    fn test_empty_partition() {
        let (store, _temp) = create_test_store();

        let node = store.get_node(999, 100).unwrap();
        assert!(node.is_none());

        let neighbors = store.get_neighbors(999, 100).unwrap();
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_header_roundtrip() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        let node = create_test_node(100, bucket);
        let header = NodeHeader::from_node(&node, bucket);

        store.store_header(bucket, header.clone()).unwrap();

        let headers = store.scan_headers(100).unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].node_id, 100);
        assert_eq!(headers[0].goal_bucket, bucket);

        store.delete_header(bucket, 100).unwrap();
        let headers = store.scan_headers(100).unwrap();
        assert_eq!(headers.len(), 0);
    }
}
