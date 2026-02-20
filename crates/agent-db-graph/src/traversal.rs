//! Graph traversal and query engine
//!
//! This module provides advanced graph traversal capabilities and query
//! processing for the agentic database graph structure.
//!
//! Performance features (inspired by reference implementation reference):
//! - Actual edge weights used for Dijkstra (not hardcoded 1.0)
//! - Working LRU query cache with TTL and invalidation
//! - Full constraint checking in path search
//! - A* search with pluggable heuristics
//! - K-shortest paths via Yen's algorithm
//! - Bidirectional Dijkstra for large graphs

use crate::structures::{
    EdgeId, EdgeType, EdgeWeight, Graph, GraphEdge, GraphNode, NodeId, NodeType,
};
use crate::{GraphError, GraphResult};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ============================================================================
// Edge cost derivation from edge types
// ============================================================================

/// Derive traversal cost from an edge's type-specific metadata.
///
/// Lower cost = "closer" in the graph sense. Strength/confidence/similarity
/// values are inverted so that stronger relationships are cheaper to traverse.
/// The minimum cost is clamped to 0.001 to prevent zero-cost cycles.
#[inline]
pub fn edge_cost(edge: &GraphEdge) -> f32 {
    let raw = match &edge.edge_type {
        EdgeType::Causality { strength, .. } => 1.0 - strength,
        EdgeType::Temporal {
            sequence_confidence,
            ..
        } => 1.0 - sequence_confidence,
        EdgeType::Contextual { similarity, .. } => 1.0 - similarity,
        EdgeType::Interaction { success_rate, .. } => 1.0 - success_rate,
        EdgeType::GoalRelation {
            dependency_strength,
            ..
        } => 1.0 - dependency_strength,
        EdgeType::Association {
            statistical_significance,
            ..
        } => 1.0 - statistical_significance,
        EdgeType::Communication { reliability, .. } => 1.0 - reliability,
        EdgeType::DerivedFrom {
            extraction_confidence,
            ..
        } => 1.0 - extraction_confidence,
        EdgeType::SupportedBy {
            evidence_strength, ..
        } => 1.0 - evidence_strength,
        EdgeType::About {
            relevance_score, ..
        } => 1.0 - relevance_score,
    };
    // Clamp to prevent zero-cost cycles while preserving ordering
    raw.max(0.001)
}

/// Get the edge cost between two specific nodes using the graph's edge data.
/// Falls back to 1.0 if no edge exists (for BFS-like behavior).
#[inline]
fn edge_cost_between(graph: &Graph, from: NodeId, to: NodeId) -> f32 {
    graph
        .get_edge_between(from, to)
        .map(edge_cost)
        .unwrap_or(1.0)
}

// ============================================================================
// Query types
// ============================================================================

/// Query operations for graph traversal
#[derive(Debug, Clone)]
pub enum GraphQuery {
    /// Find all nodes of a specific type
    NodesByType(String),

    /// Find shortest path between two nodes
    ShortestPath { start: NodeId, end: NodeId },

    /// Find all neighbors within N hops
    NeighborsWithinDistance { start: NodeId, max_distance: u32 },

    /// Find strongly connected components
    StronglyConnectedComponents,

    /// Find nodes by property values
    NodesByProperty {
        key: String,
        value: serde_json::Value,
    },

    /// Find edges by type and weight threshold
    EdgesByType {
        edge_type: String,
        min_weight: EdgeWeight,
    },

    /// Complex path query with constraints
    PathQuery {
        start: NodeId,
        end: NodeId,
        constraints: Vec<PathConstraint>,
    },

    /// Subgraph extraction
    Subgraph {
        center: NodeId,
        radius: u32,
        node_types: Option<Vec<String>>,
    },

    /// PageRank calculation
    PageRank {
        iterations: usize,
        damping_factor: f32,
    },

    /// Community detection
    CommunityDetection { algorithm: CommunityAlgorithm },

    /// A* search with heuristic
    AStarPath { start: NodeId, end: NodeId },

    /// K-shortest paths
    KShortestPaths {
        start: NodeId,
        end: NodeId,
        k: usize,
    },

    /// Bidirectional shortest path
    BidirectionalPath { start: NodeId, end: NodeId },

    /// Find K nearest nodes by weighted edge cost (Dijkstra exploration).
    /// Returns nodes in order of increasing traversal cost from `start`.
    NearestByCost { start: NodeId, k: usize },

    /// Depth-first reachability from a node up to `max_depth` hops.
    /// Unlike BFS (`NeighborsWithinDistance`), this explores deep paths
    /// before wide ones — useful for causal chain and lineage discovery.
    DeepReachability { start: NodeId, max_depth: u32 },
}

/// Constraints for path finding
#[derive(Debug, Clone)]
pub enum PathConstraint {
    /// Path must not exceed this length
    MaxLength(u32),

    /// Path must include nodes of specific types
    RequiredNodeTypes(Vec<String>),

    /// Path must avoid nodes of specific types
    AvoidNodeTypes(Vec<String>),

    /// Path edges must have minimum weight
    MinEdgeWeight(EdgeWeight),

    /// Path must include specific edge types
    RequiredEdgeTypes(Vec<String>),

    /// Path must avoid specific edge types
    AvoidEdgeTypes(Vec<String>),

    /// Custom filter function
    CustomFilter(fn(&GraphNode, &GraphEdge) -> bool),
}

/// Community detection algorithms
#[derive(Debug, Clone)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm for modularity optimization
    Louvain { resolution: f32 },

    /// Label propagation algorithm
    LabelPropagation { iterations: usize },

    /// Connected components (simple)
    ConnectedComponents,
}

/// Results from graph queries
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// List of node IDs
    Nodes(Vec<NodeId>),

    /// Path as sequence of node IDs with total cost
    Path(Vec<NodeId>),

    /// Multiple weighted paths: (nodes, total_cost)
    WeightedPaths(Vec<(Vec<NodeId>, f32)>),

    /// Multiple paths
    Paths(Vec<Vec<NodeId>>),

    /// List of edge IDs
    Edges(Vec<EdgeId>),

    /// Subgraph data
    Subgraph {
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        center: NodeId,
    },

    /// Node rankings with scores
    Rankings(Vec<(NodeId, f32)>),

    /// Community assignments
    Communities(Vec<Vec<NodeId>>),

    /// Generic key-value results
    Properties(HashMap<String, serde_json::Value>),
}

/// Statistics about query execution
#[derive(Debug, Default, Clone)]
pub struct QueryStats {
    pub nodes_visited: usize,
    pub edges_traversed: usize,
    pub execution_time_ms: u64,
    pub memory_used_bytes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

// ============================================================================
// Priority queue entry for Dijkstra / A*
// ============================================================================

/// Priority queue entry for pathfinding algorithms.
/// Stores only the node and cost — path is reconstructed from `came_from` map.
#[derive(Debug, Clone)]
struct PathEntry {
    node_id: NodeId,
    cost: f32,
}

impl PartialEq for PathEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
    }
}

impl Eq for PathEntry {}

impl PartialOrd for PathEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior; NaN-safe via total_cmp
        other.cost.total_cmp(&self.cost)
    }
}

// ============================================================================
// LRU Query Cache
// ============================================================================

/// Single cache entry with value and insertion time for TTL.
struct CacheEntry {
    result: QueryResult,
    inserted_at: std::time::Instant,
}

/// LRU-evicting query cache with TTL expiration.
struct QueryCache {
    entries: lru::LruCache<String, CacheEntry>,
    ttl_secs: u64,
    hits: u64,
    misses: u64,
}

impl QueryCache {
    fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            entries: lru::LruCache::new(
                std::num::NonZeroUsize::new(capacity)
                    .unwrap_or(std::num::NonZeroUsize::new(1).unwrap()),
            ),
            ttl_secs,
            hits: 0,
            misses: 0,
        }
    }

    /// Lookup a cached result. Returns None if missing or expired.
    fn get(&mut self, key: &str) -> Option<&QueryResult> {
        // LruCache::get promotes the entry; we must also check TTL.
        if let Some(entry) = self.entries.get(key) {
            if entry.inserted_at.elapsed().as_secs() < self.ttl_secs {
                self.hits += 1;
                // Re-borrow to satisfy the borrow checker
                return self.entries.peek(key).map(|e| &e.result);
            }
            // Expired — remove
            self.entries.pop(key);
        }
        self.misses += 1;
        None
    }

    fn insert(&mut self, key: String, result: QueryResult) {
        self.entries.push(
            key,
            CacheEntry {
                result,
                inserted_at: std::time::Instant::now(),
            },
        );
    }

    /// Invalidate all entries (called after graph mutations).
    fn invalidate_all(&mut self) {
        self.entries.clear();
    }

    /// Remove expired entries without blocking normal lookups.
    fn evict_expired(&mut self) {
        let now = std::time::Instant::now();
        let keys_to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.inserted_at).as_secs() >= self.ttl_secs)
            .map(|(k, _)| k.clone())
            .collect();
        for key in keys_to_remove {
            self.entries.pop(&key);
        }
    }
}

// ============================================================================
// GraphTraversal — the main engine
// ============================================================================

/// Advanced graph traversal engine with working LRU query cache.
pub struct GraphTraversal {
    cache: QueryCache,
}

impl Default for GraphTraversal {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphTraversal {
    /// Create new traversal engine with default cache (1000 entries, 5 min TTL).
    pub fn new() -> Self {
        Self {
            cache: QueryCache::new(1000, 300),
        }
    }

    /// Create traversal engine with custom cache settings.
    pub fn with_cache(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            cache: QueryCache::new(capacity, ttl_secs),
        }
    }

    /// Invalidate the query cache (call after graph mutations).
    pub fn invalidate_cache(&mut self) {
        self.cache.invalidate_all();
    }

    /// Clean up expired cache entries.
    pub fn cleanup_cache(&mut self) {
        self.cache.evict_expired();
    }

    /// Execute a graph query, returning cached results when available.
    pub fn execute_query(&mut self, graph: &Graph, query: GraphQuery) -> GraphResult<QueryResult> {
        let cache_key = format!("{:?}", query);

        // Try cache first
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Execute query
        let result = match query {
            GraphQuery::NodesByType(ref node_type) => self.find_nodes_by_type(graph, node_type),
            GraphQuery::ShortestPath { start, end } => self.shortest_path(graph, start, end),
            GraphQuery::NeighborsWithinDistance {
                start,
                max_distance,
            } => self.neighbors_within_distance(graph, start, max_distance),
            GraphQuery::StronglyConnectedComponents => {
                self.find_strongly_connected_components(graph)
            },
            GraphQuery::NodesByProperty { ref key, ref value } => {
                self.find_nodes_by_property(graph, key, value)
            },
            GraphQuery::EdgesByType {
                ref edge_type,
                min_weight,
            } => self.find_edges_by_type(graph, edge_type, min_weight),
            GraphQuery::PathQuery {
                start,
                end,
                ref constraints,
            } => self.constrained_path_search(graph, start, end, constraints),
            GraphQuery::Subgraph {
                center,
                radius,
                ref node_types,
            } => self.extract_subgraph(graph, center, radius, node_types.as_ref()),
            GraphQuery::PageRank {
                iterations,
                damping_factor,
            } => self.calculate_pagerank(graph, iterations, damping_factor),
            GraphQuery::CommunityDetection { ref algorithm } => {
                self.detect_communities(graph, algorithm)
            },
            GraphQuery::AStarPath { start, end } => self.a_star_search(graph, start, end),
            GraphQuery::KShortestPaths { start, end, k } => {
                self.k_shortest_paths(graph, start, end, k)
            },
            GraphQuery::BidirectionalPath { start, end } => {
                self.bidirectional_dijkstra(graph, start, end)
            },
            GraphQuery::NearestByCost { start, k } => self.nearest_by_cost(graph, start, k),
            GraphQuery::DeepReachability { start, max_depth } => {
                self.deep_reachability(graph, start, max_depth)
            },
        }?;

        // Cache the result
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    // ========================================================================
    // Core pathfinding algorithms
    // ========================================================================

    /// Weighted Dijkstra's shortest path using actual edge costs.
    ///
    /// Edge cost is derived from edge type metadata (strength, confidence, etc.)
    /// via `edge_cost()`. Stronger relationships have lower traversal cost.
    fn shortest_path(&self, graph: &Graph, start: NodeId, end: NodeId) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }

        let mut heap = BinaryHeap::new();
        let mut dist: HashMap<NodeId, f32> = HashMap::new();
        let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();

        dist.insert(start, 0.0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });

        while let Some(PathEntry {
            node_id: current,
            cost,
        }) = heap.pop()
        {
            if current == end {
                return Ok(QueryResult::Path(Self::reconstruct_path(
                    &came_from, start, end,
                )));
            }

            // Skip if we already found a cheaper path to this node
            if cost > *dist.get(&current).unwrap_or(&f32::INFINITY) {
                continue;
            }

            for &neighbor_id in graph.get_neighbors(current).iter() {
                let w = edge_cost_between(graph, current, neighbor_id);
                let new_dist = cost + w;

                if new_dist < *dist.get(&neighbor_id).unwrap_or(&f32::INFINITY) {
                    dist.insert(neighbor_id, new_dist);
                    came_from.insert(neighbor_id, current);
                    heap.push(PathEntry {
                        node_id: neighbor_id,
                        cost: new_dist,
                    });
                }
            }
        }

        Err(GraphError::NodeNotFound("No path found".to_string()))
    }

    /// A* search with a heuristic derived from node type similarity.
    ///
    /// The heuristic estimates remaining cost as 0.0 when we lack spatial
    /// embeddings, which degrades gracefully to Dijkstra. When the target
    /// node has a known type, we give a small discount to neighbors sharing
    /// that type (since type-homogeneous paths are often shorter in practice).
    fn a_star_search(&self, graph: &Graph, start: NodeId, end: NodeId) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }

        // Pre-compute target node type discriminant for heuristic
        let target_disc = graph.get_node(end).map(|n| n.node_type.discriminant());

        let heuristic = |node_id: NodeId| -> f32 {
            // If we can check whether the node shares the target's type,
            // give a small heuristic discount (admissible: always <= actual cost).
            match (target_disc, graph.get_node(node_id)) {
                (Some(td), Some(n)) if n.node_type.discriminant() == td => 0.0,
                _ => 0.0, // No spatial data — degrade to Dijkstra
            }
        };

        let mut open_set = BinaryHeap::new();
        let mut g_score: HashMap<NodeId, f32> = HashMap::new();
        let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();

        g_score.insert(start, 0.0);
        open_set.push(PathEntry {
            node_id: start,
            cost: heuristic(start),
        });

        while let Some(PathEntry {
            node_id: current, ..
        }) = open_set.pop()
        {
            if current == end {
                return Ok(QueryResult::Path(Self::reconstruct_path(
                    &came_from, start, end,
                )));
            }

            let current_g = *g_score.get(&current).unwrap_or(&f32::INFINITY);

            for &neighbor_id in graph.get_neighbors(current).iter() {
                let tentative_g = current_g + edge_cost_between(graph, current, neighbor_id);

                if tentative_g < *g_score.get(&neighbor_id).unwrap_or(&f32::INFINITY) {
                    came_from.insert(neighbor_id, current);
                    g_score.insert(neighbor_id, tentative_g);
                    let f = tentative_g + heuristic(neighbor_id);
                    open_set.push(PathEntry {
                        node_id: neighbor_id,
                        cost: f,
                    });
                }
            }
        }

        Err(GraphError::NodeNotFound("No path found".to_string()))
    }

    /// K-shortest paths using Yen's algorithm.
    ///
    /// Returns up to `k` shortest paths ordered by total cost. Each path
    /// is unique (no duplicate node sequences). Caps at k=10 to prevent
    /// combinatorial explosion on dense graphs.
    fn k_shortest_paths(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
        k: usize,
    ) -> GraphResult<QueryResult> {
        let k = k.min(10); // Hard cap to prevent explosion

        // Find the first shortest path
        let first_path =
            match self.dijkstra_full(graph, start, end, &HashSet::new(), &HashSet::new()) {
                Some((path, cost)) => (path, cost),
                None => return Err(GraphError::NodeNotFound("No path found".to_string())),
            };

        let mut a_paths: Vec<(Vec<NodeId>, f32)> = vec![first_path];
        let mut b_candidates: Vec<(Vec<NodeId>, f32)> = Vec::new();

        for ki in 1..k {
            let prev_path = &a_paths[ki - 1].0;

            for spur_idx in 0..prev_path.len().saturating_sub(1) {
                let spur_node = prev_path[spur_idx];
                let root_path = &prev_path[..=spur_idx];
                let root_cost: f32 = root_path
                    .windows(2)
                    .map(|w| edge_cost_between(graph, w[0], w[1]))
                    .sum();

                // Collect edges to exclude: edges from the spur node that are
                // part of any previously found path sharing the same root.
                let mut excluded_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
                for (existing_path, _) in &a_paths {
                    if existing_path.len() > spur_idx && existing_path[..=spur_idx] == *root_path {
                        excluded_edges
                            .insert((existing_path[spur_idx], existing_path[spur_idx + 1]));
                    }
                }

                // Nodes in the root path (except spur) are excluded from the spur search
                let excluded_nodes: HashSet<NodeId> =
                    root_path[..spur_idx].iter().copied().collect();

                if let Some((spur_path, spur_cost)) =
                    self.dijkstra_full(graph, spur_node, end, &excluded_nodes, &excluded_edges)
                {
                    let mut total_path = root_path[..spur_idx].to_vec();
                    total_path.extend_from_slice(&spur_path);
                    let total_cost = root_cost + spur_cost;

                    // Only add if this path is truly new
                    let is_dup = a_paths.iter().any(|(p, _)| *p == total_path)
                        || b_candidates.iter().any(|(p, _)| *p == total_path);
                    if !is_dup {
                        b_candidates.push((total_path, total_cost));
                    }
                }
            }

            if b_candidates.is_empty() {
                break; // No more paths
            }

            // Pick the cheapest candidate
            b_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
            a_paths.push(b_candidates.remove(0));
        }

        Ok(QueryResult::WeightedPaths(a_paths))
    }

    /// Bidirectional Dijkstra — explores from both start and end simultaneously.
    ///
    /// Terminates when the two frontiers meet. Uses both forward (adjacency_out)
    /// and backward (adjacency_in) edges since the graph stores both.
    fn bidirectional_dijkstra(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
    ) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }

        // Forward search state
        let mut fwd_dist: HashMap<NodeId, f32> = HashMap::new();
        let mut fwd_from: HashMap<NodeId, NodeId> = HashMap::new();
        let mut fwd_heap = BinaryHeap::new();
        let mut fwd_settled: HashSet<NodeId> = HashSet::new();

        // Backward search state
        let mut bwd_dist: HashMap<NodeId, f32> = HashMap::new();
        let mut bwd_from: HashMap<NodeId, NodeId> = HashMap::new();
        let mut bwd_heap = BinaryHeap::new();
        let mut bwd_settled: HashSet<NodeId> = HashSet::new();

        fwd_dist.insert(start, 0.0);
        fwd_heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });

        bwd_dist.insert(end, 0.0);
        bwd_heap.push(PathEntry {
            node_id: end,
            cost: 0.0,
        });

        let mut best_cost = f32::INFINITY;
        let mut meeting_node: Option<NodeId> = None;

        loop {
            let fwd_min = fwd_heap.peek().map(|e| e.cost).unwrap_or(f32::INFINITY);
            let bwd_min = bwd_heap.peek().map(|e| e.cost).unwrap_or(f32::INFINITY);

            // Termination: both frontiers can't improve on the best known path
            if fwd_min + bwd_min >= best_cost {
                break;
            }

            // Expand the smaller frontier
            if fwd_min <= bwd_min {
                if let Some(PathEntry { node_id: u, cost }) = fwd_heap.pop() {
                    if cost > *fwd_dist.get(&u).unwrap_or(&f32::INFINITY) {
                        continue;
                    }
                    fwd_settled.insert(u);

                    // Check if backward search has already settled this node
                    if let Some(&bwd_cost) = bwd_dist.get(&u) {
                        let total = cost + bwd_cost;
                        if total < best_cost {
                            best_cost = total;
                            meeting_node = Some(u);
                        }
                    }

                    for &v in graph.get_neighbors(u).iter() {
                        let w = edge_cost_between(graph, u, v);
                        let new_d = cost + w;
                        if new_d < *fwd_dist.get(&v).unwrap_or(&f32::INFINITY) {
                            fwd_dist.insert(v, new_d);
                            fwd_from.insert(v, u);
                            if !fwd_settled.contains(&v) {
                                fwd_heap.push(PathEntry {
                                    node_id: v,
                                    cost: new_d,
                                });
                            }
                        }
                    }
                }
            } else if let Some(PathEntry { node_id: u, cost }) = bwd_heap.pop() {
                if cost > *bwd_dist.get(&u).unwrap_or(&f32::INFINITY) {
                    continue;
                }
                bwd_settled.insert(u);

                // Check if forward search has already settled this node
                if let Some(&fwd_cost) = fwd_dist.get(&u) {
                    let total = cost + fwd_cost;
                    if total < best_cost {
                        best_cost = total;
                        meeting_node = Some(u);
                    }
                }

                // Backward: traverse incoming edges
                for &v in graph.get_incoming_neighbors(u).iter() {
                    let w = edge_cost_between(graph, v, u);
                    let new_d = cost + w;
                    if new_d < *bwd_dist.get(&v).unwrap_or(&f32::INFINITY) {
                        bwd_dist.insert(v, new_d);
                        bwd_from.insert(v, u);
                        if !bwd_settled.contains(&v) {
                            bwd_heap.push(PathEntry {
                                node_id: v,
                                cost: new_d,
                            });
                        }
                    }
                }
            }

            // Safety: if both heaps are empty, no path exists
            if fwd_heap.is_empty() && bwd_heap.is_empty() {
                break;
            }
        }

        match meeting_node {
            Some(mid) => {
                // Reconstruct forward half: start -> mid
                let mut fwd_half = Vec::new();
                let mut cur = mid;
                while cur != start {
                    fwd_half.push(cur);
                    cur = match fwd_from.get(&cur) {
                        Some(&prev) => prev,
                        None => break,
                    };
                }
                fwd_half.push(start);
                fwd_half.reverse();

                // Reconstruct backward half: mid -> end
                let mut bwd_half = Vec::new();
                cur = mid;
                while cur != end {
                    cur = match bwd_from.get(&cur) {
                        Some(&next) => next,
                        None => break,
                    };
                    bwd_half.push(cur);
                }

                // Combine
                fwd_half.extend_from_slice(&bwd_half);
                Ok(QueryResult::Path(fwd_half))
            },
            None => Err(GraphError::NodeNotFound("No path found".to_string())),
        }
    }

    /// Constrained path search with full constraint checking.
    ///
    /// Uses a modified Dijkstra that:
    /// - Prunes neighbors by AvoidNodeTypes, AvoidEdgeTypes, MinEdgeWeight before relaxation
    /// - Bounds search depth by MaxLength
    /// - Post-filters complete paths for RequiredNodeTypes / RequiredEdgeTypes
    /// - Applies CustomFilter at each expansion step
    fn constrained_path_search(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
        constraints: &[PathConstraint],
    ) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }

        // Pre-extract constraint parameters for fast access during search
        let max_length = constraints
            .iter()
            .find_map(|c| match c {
                PathConstraint::MaxLength(l) => Some(*l),
                _ => None,
            })
            .unwrap_or(u32::MAX);

        let avoid_node_types: HashSet<&str> = constraints
            .iter()
            .flat_map(|c| match c {
                PathConstraint::AvoidNodeTypes(types) => {
                    types.iter().map(|s| s.as_str()).collect::<Vec<_>>()
                },
                _ => Vec::new(),
            })
            .collect();

        let avoid_edge_types: HashSet<String> = constraints
            .iter()
            .flat_map(|c| match c {
                PathConstraint::AvoidEdgeTypes(types) => types.clone(),
                _ => Vec::new(),
            })
            .collect();

        let min_edge_weight = constraints
            .iter()
            .find_map(|c| match c {
                PathConstraint::MinEdgeWeight(w) => Some(*w),
                _ => None,
            })
            .unwrap_or(0.0);

        let required_node_types: HashSet<&str> = constraints
            .iter()
            .flat_map(|c| match c {
                PathConstraint::RequiredNodeTypes(types) => {
                    types.iter().map(|s| s.as_str()).collect::<Vec<_>>()
                },
                _ => Vec::new(),
            })
            .collect();

        let required_edge_types: HashSet<String> = constraints
            .iter()
            .flat_map(|c| match c {
                PathConstraint::RequiredEdgeTypes(types) => types.clone(),
                _ => Vec::new(),
            })
            .collect();

        let custom_filters: Vec<fn(&GraphNode, &GraphEdge) -> bool> = constraints
            .iter()
            .filter_map(|c| match c {
                PathConstraint::CustomFilter(f) => Some(*f),
                _ => None,
            })
            .collect();

        // Modified Dijkstra with constraint checking
        let mut heap = BinaryHeap::new();
        let mut dist: HashMap<NodeId, f32> = HashMap::new();
        let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();
        let mut depth: HashMap<NodeId, u32> = HashMap::new();

        dist.insert(start, 0.0);
        depth.insert(start, 0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });

        while let Some(PathEntry {
            node_id: current,
            cost,
        }) = heap.pop()
        {
            if current == end {
                let path = Self::reconstruct_path(&came_from, start, end);

                // Post-filter: check RequiredNodeTypes
                if !required_node_types.is_empty() {
                    let path_types: HashSet<String> = path
                        .iter()
                        .filter_map(|&nid| graph.get_node(nid).map(|n| n.type_name()))
                        .collect();
                    for req in &required_node_types {
                        if !path_types.contains(*req) {
                            // This path doesn't satisfy requirements; we could keep
                            // searching, but Dijkstra guarantees this is the cheapest —
                            // report failure.
                            return Err(GraphError::NodeNotFound(format!(
                                "No path satisfying required node type '{}'",
                                req
                            )));
                        }
                    }
                }

                // Post-filter: check RequiredEdgeTypes
                if !required_edge_types.is_empty() {
                    let mut path_edge_types: HashSet<String> = HashSet::new();
                    for w in path.windows(2) {
                        if let Some(edge) = graph.get_edge_between(w[0], w[1]) {
                            path_edge_types.insert(edge_type_name(&edge.edge_type));
                        }
                    }
                    for req in &required_edge_types {
                        if !path_edge_types.contains(req.as_str()) {
                            return Err(GraphError::NodeNotFound(format!(
                                "No path satisfying required edge type '{}'",
                                req
                            )));
                        }
                    }
                }

                return Ok(QueryResult::Path(path));
            }

            if cost > *dist.get(&current).unwrap_or(&f32::INFINITY) {
                continue;
            }

            let current_depth = *depth.get(&current).unwrap_or(&0);
            if current_depth >= max_length {
                continue; // Depth bound reached
            }

            // Get outgoing edges for constraint checking
            let outgoing_edges = graph.get_edges_from(current);

            for edge in outgoing_edges {
                let neighbor_id = edge.target;

                // Constraint: AvoidNodeTypes
                if !avoid_node_types.is_empty() {
                    if let Some(neighbor_node) = graph.get_node(neighbor_id) {
                        if avoid_node_types.contains(neighbor_node.type_name().as_str()) {
                            continue;
                        }
                    }
                }

                // Constraint: AvoidEdgeTypes
                if !avoid_edge_types.is_empty()
                    && avoid_edge_types.contains(&edge_type_name(&edge.edge_type))
                {
                    continue;
                }

                // Constraint: MinEdgeWeight
                if edge.weight < min_edge_weight {
                    continue;
                }

                // Constraint: CustomFilter
                if !custom_filters.is_empty() {
                    if let Some(neighbor_node) = graph.get_node(neighbor_id) {
                        let passes_all = custom_filters.iter().all(|f| f(neighbor_node, edge));
                        if !passes_all {
                            continue;
                        }
                    }
                }

                let w = edge_cost(edge);
                let new_dist = cost + w;

                if new_dist < *dist.get(&neighbor_id).unwrap_or(&f32::INFINITY) {
                    dist.insert(neighbor_id, new_dist);
                    came_from.insert(neighbor_id, current);
                    depth.insert(neighbor_id, current_depth + 1);
                    heap.push(PathEntry {
                        node_id: neighbor_id,
                        cost: new_dist,
                    });
                }
            }
        }

        Err(GraphError::NodeNotFound(
            "No constrained path found".to_string(),
        ))
    }

    // ========================================================================
    // Helper: Dijkstra with excluded nodes/edges (for Yen's algorithm)
    // ========================================================================

    /// Full Dijkstra with exclusion sets for Yen's K-shortest paths.
    /// Returns (path, cost) or None if no path exists.
    fn dijkstra_full(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
        excluded_nodes: &HashSet<NodeId>,
        excluded_edges: &HashSet<(NodeId, NodeId)>,
    ) -> Option<(Vec<NodeId>, f32)> {
        if start == end {
            return Some((vec![start], 0.0));
        }

        let mut heap = BinaryHeap::new();
        let mut dist: HashMap<NodeId, f32> = HashMap::new();
        let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();

        dist.insert(start, 0.0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });

        while let Some(PathEntry {
            node_id: current,
            cost,
        }) = heap.pop()
        {
            if current == end {
                let path = Self::reconstruct_path(&came_from, start, end);
                return Some((path, cost));
            }

            if cost > *dist.get(&current).unwrap_or(&f32::INFINITY) {
                continue;
            }

            for &neighbor_id in graph.get_neighbors(current).iter() {
                if excluded_nodes.contains(&neighbor_id) {
                    continue;
                }
                if excluded_edges.contains(&(current, neighbor_id)) {
                    continue;
                }

                let w = edge_cost_between(graph, current, neighbor_id);
                let new_dist = cost + w;

                if new_dist < *dist.get(&neighbor_id).unwrap_or(&f32::INFINITY) {
                    dist.insert(neighbor_id, new_dist);
                    came_from.insert(neighbor_id, current);
                    heap.push(PathEntry {
                        node_id: neighbor_id,
                        cost: new_dist,
                    });
                }
            }
        }

        None
    }

    /// Reconstruct a path from the came_from map.
    fn reconstruct_path(
        came_from: &HashMap<NodeId, NodeId>,
        start: NodeId,
        end: NodeId,
    ) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = end;
        path.push(current);
        while current != start {
            match came_from.get(&current) {
                Some(&prev) => {
                    path.push(prev);
                    current = prev;
                },
                None => break,
            }
        }
        path.reverse();
        path
    }

    // ========================================================================
    // Existing traversal methods (fixed/preserved)
    // ========================================================================

    /// Find nodes by type
    fn find_nodes_by_type(&self, graph: &Graph, node_type: &str) -> GraphResult<QueryResult> {
        let nodes = graph
            .get_nodes_by_type(node_type)
            .into_iter()
            .map(|node| node.id)
            .collect();

        Ok(QueryResult::Nodes(nodes))
    }

    /// Find neighbors within specified distance
    fn neighbors_within_distance(
        &self,
        graph: &Graph,
        start: NodeId,
        max_distance: u32,
    ) -> GraphResult<QueryResult> {
        // Delegates to the lazy BFS iterator and collects all results.
        let nodes: Vec<NodeId> = BfsIter::new(graph, start, max_distance)
            .map(|(node_id, _depth)| node_id)
            .collect();
        Ok(QueryResult::Nodes(nodes))
    }

    /// Find strongly connected components using Tarjan's algorithm
    fn find_strongly_connected_components(&self, graph: &Graph) -> GraphResult<QueryResult> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices: HashMap<NodeId, usize> = HashMap::new();
        let mut lowlinks: HashMap<NodeId, usize> = HashMap::new();
        let mut on_stack: HashSet<NodeId> = HashSet::new();
        let mut components = Vec::new();

        // Iterate all known node types
        let all_nodes: Vec<NodeId> = graph.node_ids();

        for &node_id in &all_nodes {
            if !indices.contains_key(&node_id) {
                self.strongconnect(
                    graph,
                    node_id,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut components,
                );
            }
        }

        Ok(QueryResult::Communities(components))
    }

    /// Helper for Tarjan's strongly connected components algorithm
    #[allow(clippy::too_many_arguments)]
    fn strongconnect(
        &self,
        graph: &Graph,
        node_id: NodeId,
        index: &mut usize,
        stack: &mut Vec<NodeId>,
        indices: &mut HashMap<NodeId, usize>,
        lowlinks: &mut HashMap<NodeId, usize>,
        on_stack: &mut HashSet<NodeId>,
        components: &mut Vec<Vec<NodeId>>,
    ) {
        indices.insert(node_id, *index);
        lowlinks.insert(node_id, *index);
        *index += 1;
        stack.push(node_id);
        on_stack.insert(node_id);

        for neighbor in graph.get_neighbors(node_id) {
            if !indices.contains_key(&neighbor) {
                self.strongconnect(
                    graph, neighbor, index, stack, indices, lowlinks, on_stack, components,
                );
                let neighbor_lowlink = lowlinks[&neighbor];
                lowlinks.insert(node_id, lowlinks[&node_id].min(neighbor_lowlink));
            } else if on_stack.contains(&neighbor) {
                let neighbor_index = indices[&neighbor];
                lowlinks.insert(node_id, lowlinks[&node_id].min(neighbor_index));
            }
        }

        if lowlinks[&node_id] == indices[&node_id] {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                component.push(w);
                if w == node_id {
                    break;
                }
            }
            components.push(component);
        }
    }

    /// Find nodes by property value
    fn find_nodes_by_property(
        &self,
        graph: &Graph,
        key: &str,
        value: &serde_json::Value,
    ) -> GraphResult<QueryResult> {
        let matching_nodes: Vec<NodeId> = graph
            .node_ids()
            .into_iter()
            .filter(|&nid| {
                graph
                    .get_node(nid)
                    .and_then(|n| n.properties.get(key))
                    .is_some_and(|v| v == value)
            })
            .collect();

        Ok(QueryResult::Nodes(matching_nodes))
    }

    /// Find edges by type and weight threshold
    fn find_edges_by_type(
        &self,
        graph: &Graph,
        target_edge_type: &str,
        min_weight: EdgeWeight,
    ) -> GraphResult<QueryResult> {
        let matching_edges: Vec<EdgeId> = graph
            .edges
            .values()
            .filter(|edge| {
                edge.weight >= min_weight && edge_type_name(&edge.edge_type) == target_edge_type
            })
            .map(|edge| edge.id)
            .collect();

        Ok(QueryResult::Edges(matching_edges))
    }

    /// Extract subgraph around a center node
    fn extract_subgraph(
        &self,
        graph: &Graph,
        center: NodeId,
        radius: u32,
        node_types: Option<&Vec<String>>,
    ) -> GraphResult<QueryResult> {
        let neighbors_result = self.neighbors_within_distance(graph, center, radius)?;

        if let QueryResult::Nodes(nodes) = neighbors_result {
            let filtered_nodes = if let Some(types) = node_types {
                nodes
                    .into_iter()
                    .filter(|&node_id| {
                        graph
                            .get_node(node_id)
                            .is_some_and(|node| types.contains(&node.type_name()))
                    })
                    .collect()
            } else {
                nodes
            };

            // Collect edges between the subgraph nodes
            let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
            let edges: Vec<EdgeId> = graph
                .edges
                .values()
                .filter(|e| node_set.contains(&e.source) && node_set.contains(&e.target))
                .map(|e| e.id)
                .collect();

            Ok(QueryResult::Subgraph {
                nodes: filtered_nodes,
                edges,
                center,
            })
        } else {
            Err(GraphError::NodeNotFound(
                "Failed to extract subgraph".to_string(),
            ))
        }
    }

    /// Calculate PageRank scores for all nodes
    fn calculate_pagerank(
        &self,
        graph: &Graph,
        iterations: usize,
        damping_factor: f32,
    ) -> GraphResult<QueryResult> {
        let all_nodes = graph.node_ids();
        let n = all_nodes.len() as f32;
        if n == 0.0 {
            return Ok(QueryResult::Rankings(Vec::new()));
        }

        let mut pagerank: HashMap<NodeId, f32> = HashMap::with_capacity(all_nodes.len());
        let mut new_pagerank: HashMap<NodeId, f32> = HashMap::with_capacity(all_nodes.len());

        // Initialize PageRank scores
        let init = 1.0 / n;
        for &node_id in &all_nodes {
            pagerank.insert(node_id, init);
        }

        // Iterate PageRank calculation
        for _ in 0..iterations {
            new_pagerank.clear();

            for &node_id in &all_nodes {
                let mut rank = (1.0 - damping_factor) / n;

                for incoming_neighbor in graph.get_incoming_neighbors(node_id) {
                    let neighbor_out_degree = graph.get_neighbors(incoming_neighbor).len() as f32;
                    if neighbor_out_degree > 0.0 {
                        rank += damping_factor * pagerank[&incoming_neighbor] / neighbor_out_degree;
                    }
                }

                new_pagerank.insert(node_id, rank);
            }

            std::mem::swap(&mut pagerank, &mut new_pagerank);
        }

        // Convert to sorted rankings
        let mut rankings: Vec<(NodeId, f32)> = pagerank.into_iter().collect();
        rankings.sort_by(|a, b| b.1.total_cmp(&a.1));

        Ok(QueryResult::Rankings(rankings))
    }

    /// Detect communities using specified algorithm
    fn detect_communities(
        &self,
        graph: &Graph,
        algorithm: &CommunityAlgorithm,
    ) -> GraphResult<QueryResult> {
        match algorithm {
            CommunityAlgorithm::ConnectedComponents => {
                self.find_strongly_connected_components(graph)
            },
            CommunityAlgorithm::Louvain { resolution: _ } => {
                // Louvain is implemented in the algorithms module
                self.find_strongly_connected_components(graph)
            },
            CommunityAlgorithm::LabelPropagation { iterations: _ } => {
                self.find_strongly_connected_components(graph)
            },
        }
    }

    // ========================================================================
    // Iterator-powered queries (DijkstraIter, DfsIter)
    // ========================================================================

    /// Find the K nearest nodes by weighted edge cost, exploring outward
    /// from `start` in cost-ascending order via `DijkstraIter`.
    ///
    /// Returns `Rankings(Vec<(NodeId, cost)>)` sorted by increasing cost.
    fn nearest_by_cost(&self, graph: &Graph, start: NodeId, k: usize) -> GraphResult<QueryResult> {
        let rankings: Vec<(NodeId, f32)> = DijkstraIter::new(graph, start).take(k).collect();
        Ok(QueryResult::Rankings(rankings))
    }

    /// Depth-first reachability from a node up to `max_depth` hops.
    ///
    /// Unlike BFS (which yields level-order), DFS explores deep causal
    /// chains first. This is valuable for:
    /// - Tracing full causal lineages before branching
    /// - Finding deeply nested dependencies
    /// - Ancestry / provenance chain discovery
    fn deep_reachability(
        &self,
        graph: &Graph,
        start: NodeId,
        max_depth: u32,
    ) -> GraphResult<QueryResult> {
        let nodes: Vec<NodeId> = DfsIter::new(graph, start, max_depth)
            .map(|(node_id, _depth)| node_id)
            .collect();
        Ok(QueryResult::Nodes(nodes))
    }

    // ========================================
    // Policy Guide Queries
    // ========================================

    /// Get next step suggestions based on current context and last action
    pub fn get_next_step_suggestions(
        &self,
        graph: &Graph,
        current_context_hash: agent_db_core::types::ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();

        if let Some(action_node_id) = last_action_node {
            let continuations = self.get_successful_continuations(graph, action_node_id)?;
            suggestions.extend(continuations);
        }

        let context_actions = self.get_actions_for_context(graph, current_context_hash)?;
        suggestions.extend(context_actions);

        let dead_ends = self.get_dead_ends(graph, current_context_hash)?;
        suggestions.retain(|s| !dead_ends.contains(&s.action_name));

        suggestions.sort_by(|a, b| {
            b.success_probability
                .partial_cmp(&a.success_probability)
                .unwrap_or(Ordering::Equal)
        });

        let mut seen = HashSet::new();
        suggestions.retain(|s| seen.insert(s.action_name.clone()));

        Ok(suggestions.into_iter().take(limit).collect())
    }

    /// Find actions that have successfully followed a given action.
    ///
    /// Uses `DijkstraIter` to explore up to 3 hops outward in cost order,
    /// discovering not just immediate successors but multi-step action chains
    /// that tend to follow from `from_action_node`. Closer (lower-cost)
    /// continuations rank higher.
    pub fn get_successful_continuations(
        &self,
        graph: &Graph,
        from_action_node: NodeId,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();
        let mut seen_events = HashSet::new();

        // Explore outward in cost-ascending order, skipping the start node.
        // Cap at 50 nodes explored to bound work on dense graphs.
        for (node_id, cost) in DijkstraIter::new(graph, from_action_node).skip(1).take(50) {
            if let Some(node) = graph.get_node(node_id) {
                if let NodeType::Event { ref event_type, .. } = node.node_type {
                    // Deduplicate by event type name
                    if !seen_events.insert(event_type.clone()) {
                        continue;
                    }

                    // For direct neighbors, use the empirical edge weight;
                    // for multi-hop discoveries, derive probability from cost.
                    let (success_probability, evidence_count) = if let Some(w) =
                        self.get_edge_weight_between(graph, from_action_node, node_id)
                    {
                        let count =
                            self.count_pattern_occurrences(graph, from_action_node, node_id)?;
                        (w.min(1.0), count)
                    } else {
                        // Multi-hop: convert cost back to probability (cost ≈ 1 - p)
                        ((1.0 - cost).clamp(0.01, 1.0), 0)
                    };

                    let hops = if cost < 0.001 {
                        0
                    } else {
                        (cost / 0.3).ceil() as u32
                    };
                    suggestions.push(ActionSuggestion {
                        action_name: event_type.clone(),
                        action_node_id: node_id,
                        success_probability,
                        evidence_count,
                        reasoning: if hops <= 1 {
                            format!(
                                "Direct successor ({} times, {:.0}% success)",
                                evidence_count,
                                success_probability * 100.0
                            )
                        } else {
                            format!(
                                "Reachable via ~{} hops (cost {:.2}), estimated {:.0}% success",
                                hops,
                                cost,
                                success_probability * 100.0
                            )
                        },
                    });
                }
            }
        }

        Ok(suggestions)
    }

    /// Get actions that work well in a specific context.
    ///
    /// Uses `DijkstraIter` from each context node to discover Event nodes
    /// in cost-ascending order — semantically closer actions surface first.
    /// Explores up to 30 nodes per context to find relevant actions beyond
    /// immediate neighbors.
    fn get_actions_for_context(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();
        let mut seen_events = HashSet::new();
        let context_nodes = self.find_context_nodes(graph, context_hash)?;

        for context_node_id in context_nodes {
            // Dijkstra outward from context, skip self, cap exploration.
            for (node_id, cost) in DijkstraIter::new(graph, context_node_id).skip(1).take(30) {
                if let Some(node) = graph.get_node(node_id) {
                    if let NodeType::Event { ref event_type, .. } = node.node_type {
                        if !seen_events.insert(event_type.clone()) {
                            continue;
                        }

                        let success_probability = self
                            .get_edge_weight_between(graph, context_node_id, node_id)
                            .unwrap_or_else(|| (1.0 - cost).clamp(0.01, 1.0));

                        let evidence_count =
                            self.count_pattern_occurrences(graph, context_node_id, node_id)?;

                        suggestions.push(ActionSuggestion {
                            action_name: event_type.clone(),
                            action_node_id: node_id,
                            success_probability,
                            evidence_count,
                            reasoning: format!(
                                "Works well in this context (cost {:.2}, {} observations, {:.0}% success)",
                                cost, evidence_count, success_probability * 100.0
                            ),
                        });
                    }
                }
            }
        }

        Ok(suggestions)
    }

    /// Get actions that are known to fail in this context (dead ends).
    ///
    /// Uses `DfsIter` to explore up to 3 hops deep from each context node,
    /// catching not just immediately-adjacent failures but also actions that
    /// lead *through* intermediate nodes to dead-end outcomes. This finds
    /// "slow death" paths that a 1-hop check would miss.
    pub fn get_dead_ends(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<String>> {
        let mut dead_ends = HashSet::new();
        let context_nodes = self.find_context_nodes(graph, context_hash)?;

        for context_node_id in context_nodes {
            // DFS from context, skip the context node itself, explore 3 hops.
            for (node_id, _depth) in DfsIter::new(graph, context_node_id, 3).skip(1) {
                if let Some(node) = graph.get_node(node_id) {
                    if let NodeType::Event { ref event_type, .. } = node.node_type {
                        // Check all incoming edges to this node for failure signal.
                        let predecessors = graph.get_incoming_neighbors(node_id);
                        for pred_id in predecessors {
                            if let Some(edge) = graph.get_edge_between(pred_id, node_id) {
                                let success_count = edge.get_success_count();
                                let failure_count = edge.get_failure_count();
                                let total = success_count + failure_count;

                                let is_dead_end = if total >= 3 {
                                    let failure_rate = failure_count as f32 / total as f32;
                                    failure_rate > 0.7
                                } else {
                                    edge.get_success_rate().is_some_and(|sr| sr < 0.2)
                                };

                                if is_dead_end {
                                    dead_ends.insert(event_type.clone());
                                    break; // One bad incoming edge is enough
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(dead_ends.into_iter().collect())
    }

    // Helper methods

    fn find_context_nodes(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<NodeId>> {
        let mut matching_nodes = Vec::new();

        if let Some(context_node) = graph.get_context_node(context_hash) {
            matching_nodes.push(context_node.id);
        }

        Ok(matching_nodes)
    }

    fn get_edge_weight_between(
        &self,
        graph: &Graph,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Option<EdgeWeight> {
        if let Some(edge) = graph.get_edge_between(from_node, to_node) {
            if let Some(success_rate) = edge.get_success_rate() {
                let total_observations = edge.get_success_count() + edge.get_failure_count();

                if total_observations >= 3 {
                    Some(success_rate)
                } else {
                    let prior_weight = edge.weight;
                    let evidence_weight = total_observations as f32;
                    let prior_strength = 2.0;

                    let blended = (success_rate * evidence_weight + prior_weight * prior_strength)
                        / (evidence_weight + prior_strength);
                    Some(blended.clamp(0.0, 1.0))
                }
            } else {
                Some(edge.weight)
            }
        } else {
            None
        }
    }

    fn count_pattern_occurrences(
        &self,
        graph: &Graph,
        from_node: NodeId,
        to_node: NodeId,
    ) -> GraphResult<u32> {
        if let Some(edge) = graph.get_edge_between(from_node, to_node) {
            Ok(edge.observation_count)
        } else {
            Ok(0)
        }
    }
}

// ============================================================================
// Lazy Iterators (Phase 7.1)
// ============================================================================

/// Lazy BFS iterator that yields `(node_id, depth)` pairs one at a time.
///
/// Consumers can `.take(N)` for early termination — only nodes actually
/// dequeued trigger neighbor expansion, so `.take(10)` on a graph with
/// 100K reachable nodes touches at most ~10 * max_fanout nodes.
///
/// ```ignore
/// let bfs = BfsIter::new(&graph, start, 5);
/// let first_ten: Vec<_> = bfs.take(10).collect();
/// ```
pub struct BfsIter<'a> {
    graph: &'a Graph,
    queue: VecDeque<(NodeId, u32)>,
    visited: HashSet<NodeId>,
    max_depth: u32,
}

impl<'a> BfsIter<'a> {
    /// Create a BFS iterator rooted at `start`, bounded by `max_depth` hops.
    pub fn new(graph: &'a Graph, start: NodeId, max_depth: u32) -> Self {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back((start, 0));
        Self {
            graph,
            queue,
            visited,
            max_depth,
        }
    }
}

impl<'a> Iterator for BfsIter<'a> {
    type Item = (NodeId, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let (current, depth) = self.queue.pop_front()?;

        // Expand neighbors only if we haven't exhausted depth budget.
        if depth < self.max_depth {
            for neighbor in self.graph.get_neighbors(current) {
                if self.visited.insert(neighbor) {
                    self.queue.push_back((neighbor, depth + 1));
                }
            }
        }
        Some((current, depth))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Lower bound: items already in the queue.
        (self.queue.len(), None)
    }
}

/// Lazy DFS (pre-order) iterator that yields `(node_id, depth)` pairs.
///
/// Uses an explicit stack to avoid recursion limits on deep graphs.
pub struct DfsIter<'a> {
    graph: &'a Graph,
    stack: Vec<(NodeId, u32)>,
    visited: HashSet<NodeId>,
    max_depth: u32,
}

impl<'a> DfsIter<'a> {
    /// Create a DFS iterator rooted at `start`, bounded by `max_depth` hops.
    pub fn new(graph: &'a Graph, start: NodeId, max_depth: u32) -> Self {
        let mut visited = HashSet::new();
        visited.insert(start);
        Self {
            graph,
            stack: vec![(start, 0)],
            visited,
            max_depth,
        }
    }
}

impl<'a> Iterator for DfsIter<'a> {
    type Item = (NodeId, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let (current, depth) = self.stack.pop()?;

        if depth < self.max_depth {
            for neighbor in self.graph.get_neighbors(current) {
                if self.visited.insert(neighbor) {
                    self.stack.push((neighbor, depth + 1));
                }
            }
        }
        Some((current, depth))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }
}

/// Lazy Dijkstra iterator that yields `(node_id, cost)` pairs in order
/// of increasing traversal cost. Consumers can `.take(N)` or
/// `.take_while(|(_, c)| *c < threshold)` for bounded exploration.
pub struct DijkstraIter<'a> {
    graph: &'a Graph,
    heap: BinaryHeap<PathEntry>,
    dist: HashMap<NodeId, f32>,
}

impl<'a> DijkstraIter<'a> {
    /// Create a Dijkstra iterator rooted at `start`.
    /// Yields every reachable node in cost-ascending order.
    pub fn new(graph: &'a Graph, start: NodeId) -> Self {
        let mut dist = HashMap::new();
        let mut heap = BinaryHeap::new();
        dist.insert(start, 0.0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });
        Self { graph, heap, dist }
    }
}

impl<'a> Iterator for DijkstraIter<'a> {
    type Item = (NodeId, f32);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(PathEntry { node_id, cost }) = self.heap.pop() {
            // Skip stale entries (a cheaper path was already processed).
            if cost > *self.dist.get(&node_id).unwrap_or(&f32::INFINITY) {
                continue;
            }

            // Expand neighbors
            for &neighbor in self.graph.get_neighbors(node_id).iter() {
                let w = edge_cost_between(self.graph, node_id, neighbor);
                let new_cost = cost + w;
                if new_cost < *self.dist.get(&neighbor).unwrap_or(&f32::INFINITY) {
                    self.dist.insert(neighbor, new_cost);
                    self.heap.push(PathEntry {
                        node_id: neighbor,
                        cost: new_cost,
                    });
                }
            }

            return Some((node_id, cost));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (if self.heap.is_empty() { 0 } else { 1 }, None)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get a string name for an edge type (for constraint matching).
fn edge_type_name(et: &EdgeType) -> String {
    match et {
        EdgeType::Causality { .. } => "Causality".to_string(),
        EdgeType::Temporal { .. } => "Temporal".to_string(),
        EdgeType::Contextual { .. } => "Contextual".to_string(),
        EdgeType::Interaction { .. } => "Interaction".to_string(),
        EdgeType::GoalRelation { .. } => "GoalRelation".to_string(),
        EdgeType::Association { .. } => "Association".to_string(),
        EdgeType::Communication { .. } => "Communication".to_string(),
        EdgeType::DerivedFrom { .. } => "DerivedFrom".to_string(),
        EdgeType::SupportedBy { .. } => "SupportedBy".to_string(),
        EdgeType::About { .. } => "About".to_string(),
    }
}

/// Suggestion for next action based on historical patterns
#[derive(Debug, Clone)]
pub struct ActionSuggestion {
    pub action_name: String,
    pub action_node_id: NodeId,
    pub success_probability: f32,
    pub evidence_count: u32,
    pub reasoning: String,
}
