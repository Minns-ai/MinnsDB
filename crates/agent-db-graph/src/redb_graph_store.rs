// Persistent graph storage using redb
//
// Implements GraphStore trait with redb backend, using:
// - Hierarchical keys (inspired by Dgraph)
// - Delta-encoded adjacency lists for compression
// - Goal-bucket partitioning for semantic sharding
// - LRU partition loading for memory efficiency

use crate::compression::CompressedAdjacencyList;
use crate::graph_store::{
    BucketInfo, GraphEdge, GraphNode, GraphPath, GraphStore, GraphStoreError, Subgraph,
};
use crate::structures::{GoalBucketId, NodeId};
use agent_db_storage::RedbBackend;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// Table names for redb
const TABLE_GRAPH_NODES: &str = "graph_nodes";
const TABLE_GRAPH_ADJACENCY: &str = "graph_adjacency";
const TABLE_GRAPH_EDGES: &str = "graph_edges";
#[allow(dead_code)]
const TABLE_GRAPH_BUCKETS: &str = "graph_buckets";

// ============================================================================
// Key Design (Hierarchical, inspired by Dgraph)
// ============================================================================

/// Key type prefixes
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum KeyType {
    NodeMeta = 0x01,         // Node metadata
    AdjacencyForward = 0x02, // Forward edges (A → [B, C, D])
    AdjacencyReverse = 0x03, // Reverse edges (backlinks)
    EdgeMeta = 0x04,         // Edge metadata
    #[allow(dead_code)]
    BucketCatalog = 0x05,    // Partition statistics
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

/// Build edge metadata key
fn make_edge_key(bucket: GoalBucketId, from: NodeId, to: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + 8 + 8 + 8);
    key.push(KeyType::EdgeMeta as u8);
    key.extend_from_slice(&bucket.to_be_bytes());
    key.extend_from_slice(&from.to_be_bytes());
    key.extend_from_slice(&to.to_be_bytes());
    key
}

// ============================================================================
// Partition Cache (for LRU management)
// ============================================================================

// Reference time for calculating relative timestamps (milliseconds since program start)
static START_TIME: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn get_timestamp_ms() -> u64 {
    let start = START_TIME.get_or_init(|| Instant::now());
    start.elapsed().as_millis() as u64
}

/// Cached partition data (loaded into memory)
#[derive(Debug)]
struct PartitionCache {
    nodes: HashMap<NodeId, GraphNode>,
    forward_edges: HashMap<NodeId, Vec<NodeId>>,
    reverse_edges: HashMap<NodeId, Vec<NodeId>>,
    edge_metadata: HashMap<(NodeId, NodeId), GraphEdge>,
    last_accessed_ms: AtomicU64, // Timestamp in milliseconds for LRU (thread-safe)
}

impl PartitionCache {
    fn new(_bucket_id: GoalBucketId) -> Self {
        Self {
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            edge_metadata: HashMap::new(),
            last_accessed_ms: AtomicU64::new(get_timestamp_ms()),
        }
    }

    fn touch(&self) {
        self.last_accessed_ms
            .store(get_timestamp_ms(), Ordering::Relaxed);
    }

    fn last_accessed(&self) -> u64 {
        self.last_accessed_ms.load(Ordering::Relaxed)
    }
}

// ============================================================================
// RedbGraphStore Implementation
// ============================================================================

/// Persistent graph store using redb backend
pub struct RedbGraphStore {
    backend: Arc<RedbBackend>,

    // Partition cache (LRU)
    loaded_partitions: HashMap<GoalBucketId, PartitionCache>,
    max_loaded_partitions: usize,
}

impl RedbGraphStore {
    /// Create a new RedbGraphStore
    pub fn new(backend: Arc<RedbBackend>, max_loaded_partitions: usize) -> Self {
        tracing::info!(
            "Initializing RedbGraphStore with max_loaded_partitions={}",
            max_loaded_partitions
        );

        Self {
            backend,
            loaded_partitions: HashMap::new(),
            max_loaded_partitions,
        }
    }

    /// Helper: put with JSON serialization (for types containing serde_json::Value)
    fn put_json<K, V>(&self, table: &str, key: K, value: &V) -> Result<(), GraphStoreError>
    where
        K: AsRef<[u8]>,
        V: serde::Serialize,
    {
        let json_bytes =
            serde_json::to_vec(value).map_err(|e| GraphStoreError::Serialization(e.to_string()))?;
        self.backend
            .put_raw(table, key, &json_bytes)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))
    }

    /// Helper: get with JSON deserialization (for types containing serde_json::Value)
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
            // Update LRU
            if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
                partition.touch();
            }
            return Ok(());
        }

        // Load partition from disk
        tracing::debug!("Loading partition {} from disk", bucket);
        self.load_partition_internal(bucket)?;

        // Evict LRU if needed
        if self.loaded_partitions.len() > self.max_loaded_partitions {
            self.evict_lru_partition()?;
        }

        Ok(())
    }

    /// Load partition from disk into cache
    fn load_partition_internal(&mut self, bucket: GoalBucketId) -> Result<(), GraphStoreError> {
        let mut partition = PartitionCache::new(bucket);

        // Load all nodes for this bucket
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

        // Load forward adjacency lists
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
            // Extract node_id from key (NodeId is u64 = 8 bytes at offset 9)
            let node_id = u64::from_be_bytes(
                key[9..17]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );

            partition
                .forward_edges
                .insert(node_id, compressed.decompress());
        }

        // Load reverse adjacency lists
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
            let node_id = u64::from_be_bytes(
                key[9..17]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );

            partition
                .reverse_edges
                .insert(node_id, compressed.decompress());
        }

        // Load edge metadata
        let edge_prefix = {
            let mut p = vec![KeyType::EdgeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        for (key, edge) in
            self.scan_prefix_json::<Vec<u8>, GraphEdge>(TABLE_GRAPH_EDGES, edge_prefix)?
        {
            // Extract from (offset 9-17), to (offset 17-25)
            let from = u64::from_be_bytes(
                key[9..17]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );
            let to = u64::from_be_bytes(
                key[17..25]
                    .try_into()
                    .map_err(|_| GraphStoreError::Storage("Invalid key format".to_string()))?,
            );

            partition.edge_metadata.insert((from, to), edge);
        }

        tracing::info!(
            "Loaded partition {}: {} nodes, {} forward edges, {} reverse edges",
            bucket,
            partition.nodes.len(),
            partition.forward_edges.len(),
            partition.reverse_edges.len()
        );

        self.loaded_partitions.insert(bucket, partition);
        Ok(())
    }

    /// Evict least recently used partition
    fn evict_lru_partition(&mut self) -> Result<(), GraphStoreError> {
        if self.loaded_partitions.is_empty() {
            return Ok(());
        }

        let lru_bucket = self
            .loaded_partitions
            .iter()
            .min_by_key(|(_, partition)| partition.last_accessed())
            .map(|(bucket_id, _)| *bucket_id)
            .ok_or_else(|| GraphStoreError::Storage("No partitions to evict".to_string()))?;

        tracing::debug!("Evicting LRU partition {}", lru_bucket);
        self.loaded_partitions.remove(&lru_bucket);

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
        self.put_json(TABLE_GRAPH_NODES, &key, &node)?;

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
        // Try cache first
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            partition.touch(); // Update LRU timestamp
            return Ok(partition.nodes.get(&node_id).cloned());
        }

        // Load from disk
        let key = make_node_key(bucket, node_id);
        self.get_json(TABLE_GRAPH_NODES, &key)
    }

    fn delete_node(
        &mut self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<(), GraphStoreError> {
        let key = make_node_key(bucket, node_id);
        self.backend
            .delete(TABLE_GRAPH_NODES, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        // Delete adjacency lists
        let fwd_key = make_adjacency_forward_key(bucket, node_id);
        let _ = self.backend.delete(TABLE_GRAPH_ADJACENCY, &fwd_key);

        let rev_key = make_adjacency_reverse_key(bucket, node_id);
        let _ = self.backend.delete(TABLE_GRAPH_ADJACENCY, &rev_key);

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
        // Try cache first
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(partition.nodes.values().cloned().collect());
        }

        // Load from disk
        let prefix = {
            let mut p = vec![KeyType::NodeMeta as u8];
            p.extend_from_slice(&bucket.to_be_bytes());
            p
        };

        let results = self.scan_prefix_json::<Vec<u8>, GraphNode>(TABLE_GRAPH_NODES, prefix)?;

        Ok(results.into_iter().map(|(_, node)| node).collect())
    }

    // ========================================================================
    // Edge Operations
    // ========================================================================

    fn add_edge(&mut self, bucket: GoalBucketId, edge: GraphEdge) -> Result<(), GraphStoreError> {
        self.ensure_partition_loaded(bucket)?;

        // 1. Add to forward adjacency
        let mut neighbors = if let Some(partition) = self.loaded_partitions.get(&bucket) {
            partition
                .forward_edges
                .get(&edge.from)
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        if !neighbors.contains(&edge.to) {
            neighbors.push(edge.to);
            neighbors.sort();
        }

        let compressed = CompressedAdjacencyList::compress(&neighbors);
        let fwd_key = make_adjacency_forward_key(bucket, edge.from);
        self.backend
            .put(TABLE_GRAPH_ADJACENCY, &fwd_key, &compressed)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        // 2. Add to reverse adjacency (backlinks)
        let mut preds = if let Some(partition) = self.loaded_partitions.get(&bucket) {
            partition
                .reverse_edges
                .get(&edge.to)
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        if !preds.contains(&edge.from) {
            preds.push(edge.from);
            preds.sort();
        }

        let compressed_preds = CompressedAdjacencyList::compress(&preds);
        let rev_key = make_adjacency_reverse_key(bucket, edge.to);
        self.backend
            .put(TABLE_GRAPH_ADJACENCY, &rev_key, &compressed_preds)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        // 3. Store edge metadata
        let edge_key = make_edge_key(bucket, edge.from, edge.to);
        self.put_json(TABLE_GRAPH_EDGES, &edge_key, &edge)?;

        // 4. Update cache
        if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
            partition.forward_edges.insert(edge.from, neighbors);
            partition.reverse_edges.insert(edge.to, preds);
            partition.edge_metadata.insert((edge.from, edge.to), edge);
            partition.touch();
        }

        Ok(())
    }

    fn get_edge(
        &self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
    ) -> Result<Option<GraphEdge>, GraphStoreError> {
        // Try cache first
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(partition.edge_metadata.get(&(from, to)).cloned());
        }

        // Load from disk
        let key = make_edge_key(bucket, from, to);
        self.get_json(TABLE_GRAPH_EDGES, &key)
    }

    fn delete_edge(
        &mut self,
        bucket: GoalBucketId,
        from: NodeId,
        to: NodeId,
    ) -> Result<(), GraphStoreError> {
        self.ensure_partition_loaded(bucket)?;

        // Update forward adjacency
        if let Some(partition) = self.loaded_partitions.get_mut(&bucket) {
            if let Some(neighbors) = partition.forward_edges.get_mut(&from) {
                neighbors.retain(|&n| n != to);

                let compressed = CompressedAdjacencyList::compress(neighbors);
                let fwd_key = make_adjacency_forward_key(bucket, from);
                self.backend
                    .put(TABLE_GRAPH_ADJACENCY, &fwd_key, &compressed)
                    .map_err(|e| GraphStoreError::Storage(e.to_string()))?;
            }

            // Update reverse adjacency
            if let Some(preds) = partition.reverse_edges.get_mut(&to) {
                preds.retain(|&p| p != from);

                let compressed = CompressedAdjacencyList::compress(preds);
                let rev_key = make_adjacency_reverse_key(bucket, to);
                self.backend
                    .put(TABLE_GRAPH_ADJACENCY, &rev_key, &compressed)
                    .map_err(|e| GraphStoreError::Storage(e.to_string()))?;
            }

            partition.edge_metadata.remove(&(from, to));
            partition.touch();
        }

        // Delete edge metadata
        let edge_key = make_edge_key(bucket, from, to);
        self.backend
            .delete(TABLE_GRAPH_EDGES, &edge_key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?;

        Ok(())
    }

    fn get_neighbors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        // Try cache first
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(partition
                .forward_edges
                .get(&node_id)
                .cloned()
                .unwrap_or_default());
        }

        // Load from disk
        let key = make_adjacency_forward_key(bucket, node_id);
        match self
            .backend
            .get::<_, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(compressed) => Ok(compressed.decompress()),
            None => Ok(Vec::new()),
        }
    }

    fn get_predecessors(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<NodeId>, GraphStoreError> {
        // Try cache first
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(partition
                .reverse_edges
                .get(&node_id)
                .cloned()
                .unwrap_or_default());
        }

        // Load from disk
        let key = make_adjacency_reverse_key(bucket, node_id);
        match self
            .backend
            .get::<_, CompressedAdjacencyList>(TABLE_GRAPH_ADJACENCY, &key)
            .map_err(|e| GraphStoreError::Storage(e.to_string()))?
        {
            Some(compressed) => Ok(compressed.decompress()),
            None => Ok(Vec::new()),
        }
    }

    fn get_outgoing_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let neighbors = self.get_neighbors(bucket, node_id)?;
        let mut edges = Vec::new();

        for to in neighbors {
            if let Some(edge) = self.get_edge(bucket, node_id, to)? {
                edges.push(edge);
            }
        }

        Ok(edges)
    }

    fn get_incoming_edges(
        &self,
        bucket: GoalBucketId,
        node_id: NodeId,
    ) -> Result<Vec<GraphEdge>, GraphStoreError> {
        let preds = self.get_predecessors(bucket, node_id)?;
        let mut edges = Vec::new();

        for from in preds {
            if let Some(edge) = self.get_edge(bucket, from, node_id)? {
                edges.push(edge);
            }
        }

        Ok(edges)
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
        // Try cache first
        if let Some(partition) = self.loaded_partitions.get(&bucket) {
            return Ok(BucketInfo {
                bucket_id: bucket,
                node_count: partition.nodes.len() as u64,
                edge_count: partition.edge_metadata.len() as u64,
                size_bytes: 0,
                last_modified: 0,
            });
        }

        // Count from disk
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
        _bucket: GoalBucketId,
        _from: NodeId,
        _to: NodeId,
        _max_depth: u32,
    ) -> Result<Vec<GraphPath>, GraphStoreError> {
        // TODO: Implement path finding algorithm
        Ok(Vec::new())
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
    use crate::graph_store::{GraphEdge, GraphEdgeType, GraphNode, GraphNodeType};
    use agent_db_storage::{RedbBackend, RedbConfig};
    use tempfile::TempDir;

    fn create_test_store() -> (RedbGraphStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.redb");

        let config = RedbConfig {
            data_path: db_path,
            cache_size_bytes: 64 * 1024 * 1024, // 64MB for tests
            repair_on_open: false,
        };

        let backend = Arc::new(RedbBackend::open(config).unwrap());
        let store = RedbGraphStore::new(backend, 3); // Max 3 partitions in cache
        (store, temp_dir)
    }

    fn create_test_node(id: NodeId, bucket: GoalBucketId) -> GraphNode {
        GraphNode {
            id,
            node_type: GraphNodeType::Action,
            label: format!("test_node_{}", id),
            context_hash: bucket,
            created_at: 1000 + id,
            properties: serde_json::json!({"test": "node", "bucket": bucket}),
        }
    }

    fn create_test_edge(from: NodeId, to: NodeId, _bucket: GoalBucketId) -> GraphEdge {
        GraphEdge {
            from,
            to,
            edge_type: GraphEdgeType::Temporal,
            weight: 1.0,
            confidence: 0.9,
            created_at: 2000,
        }
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
        assert_eq!(retrieved, Some(node.clone()));

        // Update (just add same node with updated properties)
        let mut updated_node = node.clone();
        updated_node.properties = serde_json::json!({"test": "updated"});
        store.add_node(bucket, updated_node.clone()).unwrap();

        let retrieved = store.get_node(bucket, 100).unwrap();
        assert_eq!(
            retrieved.unwrap().properties,
            serde_json::json!({"test": "updated"})
        );

        // Delete
        store.delete_node(bucket, 100).unwrap();
        let retrieved = store.get_node(bucket, 100).unwrap();
        assert_eq!(retrieved, None);
    }

    #[test]
    fn test_edge_operations_with_compression() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        // Add nodes
        for i in 100..105 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
        }

        // Add edges: 100 -> 101 -> 102 -> 103 -> 104
        for i in 100..104 {
            store
                .add_edge(bucket, create_test_edge(i, i + 1, bucket))
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
        assert_eq!(outgoing[0].to, 101);

        let incoming = store.get_incoming_edges(bucket, 101).unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].from, 100);

        // Test edge deletion
        store.delete_edge(bucket, 100, 101).unwrap();
        let neighbors = store.get_neighbors(bucket, 100).unwrap();
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_partition_loading_and_unloading() {
        let (mut store, _temp) = create_test_store();

        // Add nodes to different buckets
        for bucket in 1..=3 {
            for i in 0..5 {
                let node_id = (bucket * 100) + i;
                store
                    .add_node(bucket, create_test_node(node_id, bucket))
                    .unwrap();
            }
        }

        // All partitions should be loaded now
        assert_eq!(store.loaded_partitions.len(), 3);

        // Unload partition 1
        store.unload_partition(1).unwrap();
        assert_eq!(store.loaded_partitions.len(), 2);
        assert!(!store.loaded_partitions.contains_key(&1));

        // Reload partition 1
        store.load_partition(1).unwrap();
        assert_eq!(store.loaded_partitions.len(), 3);
        assert!(store.loaded_partitions.contains_key(&1));

        // Verify data is still there
        let node = store.get_node(1, 100).unwrap();
        assert!(node.is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let (mut store, _temp) = create_test_store();

        // Add nodes to 4 different buckets (max is 3)
        for bucket in 1..=4 {
            for i in 0..3 {
                let node_id = (bucket * 100) + i;
                store
                    .add_node(bucket, create_test_node(node_id, bucket))
                    .unwrap();
            }
        }

        // Should have evicted one partition (LRU)
        assert_eq!(store.loaded_partitions.len(), 3);

        // Access bucket 2 to make it recently used
        store.get_node(2, 200).unwrap();

        // Add to bucket 5, should evict bucket 1 or 3 (not 2 or 5)
        store.add_node(5, create_test_node(500, 5)).unwrap();
        assert_eq!(store.loaded_partitions.len(), 3);
        assert!(store.loaded_partitions.contains_key(&2)); // Should still be loaded
        assert!(store.loaded_partitions.contains_key(&5)); // Just added
    }

    #[test]
    fn test_persistence_across_restarts() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.redb");

        // Create store and add data
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

            for i in 100..104 {
                store
                    .add_edge(bucket, create_test_edge(i, i + 1, bucket))
                    .unwrap();
            }

            // Unload partition to force flush to disk
            store.unload_partition(bucket).unwrap();
        }

        // Reopen store and verify data persisted
        {
            let config = RedbConfig {
                data_path: db_path,
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            };
            let backend = Arc::new(RedbBackend::open(config).unwrap());
            let store = RedbGraphStore::new(backend, 3);

            // Load partition and verify
            let node = store.get_node(1, 100).unwrap();
            assert!(node.is_some());

            let neighbors = store.get_neighbors(1, 100).unwrap();
            assert_eq!(neighbors, vec![101]);

            // Verify all nodes persisted
            for i in 100..105 {
                let node = store.get_node(1, i).unwrap();
                assert!(node.is_some(), "Node {} should exist", i);
            }
        }
    }

    #[test]
    fn test_bfs_traversal() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        // Create a simple graph:
        //     100
        //    /   \
        //  101   102
        //   |     |
        //  103   104
        for i in 100..105 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
        }

        store
            .add_edge(bucket, create_test_edge(100, 101, bucket))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(100, 102, bucket))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(101, 103, bucket))
            .unwrap();
        store
            .add_edge(bucket, create_test_edge(102, 104, bucket))
            .unwrap();

        // BFS from root with max depth 1
        let result = store.traverse_bfs(bucket, 100, 1).unwrap();
        assert_eq!(result.len(), 3); // 100, 101, 102
        assert!(result.contains(&100));
        assert!(result.contains(&101));
        assert!(result.contains(&102));

        // BFS from root with max depth 2
        let result = store.traverse_bfs(bucket, 100, 2).unwrap();
        assert_eq!(result.len(), 5); // All nodes
        assert!(result.contains(&100));
        assert!(result.contains(&101));
        assert!(result.contains(&102));
        assert!(result.contains(&103));
        assert!(result.contains(&104));
    }

    #[test]
    fn test_dfs_traversal() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        // Create a linear chain: 100 -> 101 -> 102 -> 103
        for i in 100..104 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
        }

        for i in 100..103 {
            store
                .add_edge(bucket, create_test_edge(i, i + 1, bucket))
                .unwrap();
        }

        // DFS with max depth 1
        let result = store.traverse_dfs(bucket, 100, 1).unwrap();
        assert_eq!(result.len(), 2); // 100, 101
        assert_eq!(result[0], 100);
        assert!(result.contains(&101));

        // DFS with max depth 3
        let result = store.traverse_dfs(bucket, 100, 3).unwrap();
        assert_eq!(result.len(), 4); // All nodes
    }

    #[test]
    fn test_subgraph_extraction() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        // Create a star graph with center 100
        store
            .add_node(bucket, create_test_node(100, bucket))
            .unwrap();
        for i in 101..105 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
            store
                .add_edge(bucket, create_test_edge(100, i, bucket))
                .unwrap();
        }

        // Extract subgraph with radius 1
        let subgraph = store.get_subgraph(bucket, 100, 1).unwrap();
        assert_eq!(subgraph.center, 100);
        assert_eq!(subgraph.radius, 1);
        assert_eq!(subgraph.nodes.len(), 5); // Center + 4 neighbors
        assert_eq!(subgraph.edges.len(), 4); // 4 edges from center

        // Extract subgraph with radius 0 (just the center)
        let subgraph = store.get_subgraph(bucket, 100, 0).unwrap();
        assert_eq!(subgraph.nodes.len(), 1);
        assert_eq!(subgraph.edges.len(), 0);
    }

    #[test]
    fn test_multi_bucket_operations() {
        let (mut store, _temp) = create_test_store();

        // Add nodes to multiple buckets
        for bucket in 1..=3 {
            for i in 0..5 {
                let node_id = (bucket * 100) + i;
                store
                    .add_node(bucket, create_test_node(node_id, bucket))
                    .unwrap();
            }
        }

        // Verify each bucket has its data
        for bucket in 1..=3 {
            for i in 0..5 {
                let node_id = (bucket * 100) + i;
                let node = store.get_node(bucket, node_id).unwrap();
                assert!(
                    node.is_some(),
                    "Node {} in bucket {} should exist",
                    node_id,
                    bucket
                );
            }
        }

        // Verify buckets are isolated (node from bucket 1 shouldn't be in bucket 2)
        let node = store.get_node(2, 100).unwrap();
        assert!(node.is_none(), "Node 100 should not exist in bucket 2");

        let node = store.get_node(1, 200).unwrap();
        assert!(node.is_none(), "Node 200 should not exist in bucket 1");
    }

    #[test]
    fn test_concurrent_partition_access() {
        let (mut store, _temp) = create_test_store();

        // Add nodes to bucket 1
        for i in 100..110 {
            store.add_node(1, create_test_node(i, 1)).unwrap();
        }

        // Access same partition multiple times (should use cache)
        for _ in 0..10 {
            let node = store.get_node(1, 100).unwrap();
            assert!(node.is_some());
        }

        // Partition should still be loaded
        assert!(store.loaded_partitions.contains_key(&1));
    }

    #[test]
    fn test_empty_partition() {
        let (store, _temp) = create_test_store();

        // Try to get node from non-existent bucket
        let node = store.get_node(999, 100).unwrap();
        assert!(node.is_none());

        // Try to get neighbors from non-existent bucket
        let neighbors = store.get_neighbors(999, 100).unwrap();
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn test_compression_effectiveness() {
        let (mut store, _temp) = create_test_store();
        let bucket = 1;

        // Create a node with many sequential neighbors (good compression case)
        store
            .add_node(bucket, create_test_node(100, bucket))
            .unwrap();
        for i in 200..300 {
            store.add_node(bucket, create_test_node(i, bucket)).unwrap();
            store
                .add_edge(bucket, create_test_edge(100, i, bucket))
                .unwrap();
        }

        // Get neighbors (should be decompressed from delta encoding)
        let neighbors = store.get_neighbors(bucket, 100).unwrap();
        assert_eq!(neighbors.len(), 100);

        // Verify all neighbors are present
        for i in 200..300 {
            assert!(neighbors.contains(&i), "Should contain neighbor {}", i);
        }
    }
}
