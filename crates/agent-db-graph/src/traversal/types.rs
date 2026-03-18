//! Query types, result types, and constraint definitions for graph traversal.

use crate::structures::{Depth, Direction, EdgeId, EdgeWeight, GraphEdge, GraphNode, NodeId};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::spec::TraversalRequest;

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

    /// Direction-aware traversal with depth specification.
    DirectedTraversal {
        start: NodeId,
        direction: Direction,
        depth: Depth,
    },

    /// Composable recursive traversal with filters, budgets, and instructions.
    RecursiveTraversal(TraversalRequest),
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

/// Compute a u64 cache key for a GraphQuery without allocating a Debug string.
/// Handles types that don't impl Hash (serde_json::Value, fn pointers) by
/// hashing their Debug representation bytes or raw address.
pub(crate) fn query_cache_key(query: &GraphQuery) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();

    // Discriminant first
    std::mem::discriminant(query).hash(&mut h);

    match query {
        GraphQuery::NodesByType(s) => s.hash(&mut h),
        GraphQuery::ShortestPath { start, end } => {
            start.hash(&mut h);
            end.hash(&mut h);
        },
        GraphQuery::NeighborsWithinDistance {
            start,
            max_distance,
        } => {
            start.hash(&mut h);
            max_distance.hash(&mut h);
        },
        GraphQuery::StronglyConnectedComponents => {},
        GraphQuery::NodesByProperty { key, value } => {
            key.hash(&mut h);
            // serde_json::Value doesn't impl Hash — hash its compact JSON bytes
            let json = serde_json::to_string(value).unwrap_or_default();
            json.hash(&mut h);
        },
        GraphQuery::EdgesByType {
            edge_type,
            min_weight,
        } => {
            edge_type.hash(&mut h);
            min_weight.to_bits().hash(&mut h);
        },
        GraphQuery::PathQuery {
            start,
            end,
            constraints,
        } => {
            start.hash(&mut h);
            end.hash(&mut h);
            for c in constraints {
                hash_path_constraint(c, &mut h);
            }
        },
        GraphQuery::Subgraph {
            center,
            radius,
            node_types,
        } => {
            center.hash(&mut h);
            radius.hash(&mut h);
            node_types.hash(&mut h);
        },
        GraphQuery::PageRank {
            iterations,
            damping_factor,
        } => {
            iterations.hash(&mut h);
            damping_factor.to_bits().hash(&mut h);
        },
        GraphQuery::CommunityDetection { algorithm } => {
            hash_community_algorithm(algorithm, &mut h);
        },
        GraphQuery::AStarPath { start, end } => {
            start.hash(&mut h);
            end.hash(&mut h);
        },
        GraphQuery::KShortestPaths { start, end, k } => {
            start.hash(&mut h);
            end.hash(&mut h);
            k.hash(&mut h);
        },
        GraphQuery::BidirectionalPath { start, end } => {
            start.hash(&mut h);
            end.hash(&mut h);
        },
        GraphQuery::NearestByCost { start, k } => {
            start.hash(&mut h);
            k.hash(&mut h);
        },
        GraphQuery::DeepReachability { start, max_depth } => {
            start.hash(&mut h);
            max_depth.hash(&mut h);
        },
        GraphQuery::DirectedTraversal {
            start,
            direction,
            depth,
        } => {
            start.hash(&mut h);
            direction.hash(&mut h);
            depth.hash(&mut h);
        },
        GraphQuery::RecursiveTraversal(ref req) => {
            0x01u8.hash(&mut h); // version byte
            req.start.hash(&mut h);
            req.direction.hash(&mut h);
            req.depth.hash(&mut h);
            req.instruction.hash(&mut h);
            // Sort filter lists for deterministic hashing
            let mut nf = req.node_filters.clone();
            nf.sort();
            nf.hash(&mut h);
            let mut ef = req.edge_filters.clone();
            ef.sort();
            ef.hash(&mut h);
            req.max_nodes_visited.hash(&mut h);
            req.max_edges_traversed.hash(&mut h);
            req.time_window.hash(&mut h);
        },
    }

    h.finish()
}

fn hash_path_constraint(c: &PathConstraint, h: &mut impl Hasher) {
    std::mem::discriminant(c).hash(h);
    match c {
        PathConstraint::MaxLength(n) => n.hash(h),
        PathConstraint::RequiredNodeTypes(v) => v.hash(h),
        PathConstraint::AvoidNodeTypes(v) => v.hash(h),
        PathConstraint::MinEdgeWeight(w) => w.to_bits().hash(h),
        PathConstraint::RequiredEdgeTypes(v) => v.hash(h),
        PathConstraint::AvoidEdgeTypes(v) => v.hash(h),
        PathConstraint::CustomFilter(f) => {
            // fn pointers are just addresses — hash the raw pointer
            (*f as usize).hash(h);
        },
    }
}

fn hash_community_algorithm(alg: &CommunityAlgorithm, h: &mut impl Hasher) {
    std::mem::discriminant(alg).hash(h);
    match alg {
        CommunityAlgorithm::Louvain { resolution } => resolution.to_bits().hash(h),
        CommunityAlgorithm::LabelPropagation { iterations } => iterations.hash(h),
        CommunityAlgorithm::ConnectedComponents => {},
    }
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

/// Suggestion for next action based on historical patterns
#[derive(Debug, Clone)]
pub struct ActionSuggestion {
    pub action_name: String,
    pub action_node_id: NodeId,
    pub success_probability: f32,
    pub evidence_count: u32,
    pub reasoning: String,
}
