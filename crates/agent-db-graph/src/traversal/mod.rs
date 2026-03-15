//! Graph traversal and query engine
//!
//! This module provides advanced graph traversal capabilities and query
//! processing for the agentic database graph structure.
//!
//! Performance features:
//! - Actual edge weights used for Dijkstra (not hardcoded 1.0)
//! - Working LRU query cache with TTL and invalidation
//! - Full constraint checking in path search
//! - A* search with pluggable heuristics
//! - K-shortest paths via Yen's algorithm
//! - Bidirectional Dijkstra for large graphs

mod algorithms;
mod cache;
mod directed;
pub mod edge_cost;
mod graph_ops;
mod helpers;
mod iterators;
pub mod spec;
mod streaming;
mod suggestions;
pub mod types;

#[cfg(test)]
mod tests;

// ── Re-exports ──

// Edge cost
pub use edge_cost::edge_cost;

// Types
pub use types::{
    ActionSuggestion, CommunityAlgorithm, GraphQuery, PathConstraint, QueryResult, QueryStats,
};

// Iterators
pub use iterators::{BfsIter, DfsIter, DijkstraIter};

// Directed iterators
pub use directed::{DirectedBfsIter, DirectedDfsIter, DirectedDijkstraIter};

// Spec types
pub use spec::{
    EdgeFilterExpr, EdgePredicate, Instruction, NodeFilterExpr, NodePredicate, TraversalRequest,
    TraversalSpec, execute_traversal,
};

// Streaming
pub use streaming::{CancelHandle, QueryContext, StreamingQuery};

// ============================================================================
// GraphTraversal — the main engine
// ============================================================================

use crate::structures::Graph;
use crate::GraphResult;

/// Advanced graph traversal engine with working LRU query cache.
pub struct GraphTraversal {
    cache: parking_lot::Mutex<cache::QueryCache>,
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
            cache: parking_lot::Mutex::new(cache::QueryCache::new(1000, 300)),
        }
    }

    /// Create traversal engine with custom cache settings.
    pub fn with_cache(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            cache: parking_lot::Mutex::new(cache::QueryCache::new(capacity, ttl_secs)),
        }
    }

    /// Invalidate the query cache (call after graph mutations).
    pub fn invalidate_cache(&self) {
        self.cache.lock().invalidate_all();
    }

    /// Clean up expired cache entries.
    pub fn cleanup_cache(&self) {
        self.cache.lock().evict_expired();
    }

    /// Execute a graph query, returning cached results when available.
    pub fn execute_query(&self, graph: &Graph, query: GraphQuery) -> GraphResult<QueryResult> {
        // Lock cache briefly for generation check + lookup
        let cache_key = types::query_cache_key(&query);
        {
            let mut cache = self.cache.lock();
            // Auto-invalidate cache if the graph has mutated since last query
            cache.check_generation(graph.generation());

            // Try cache first
            if let Some(cached) = cache.get(cache_key) {
                return Ok(cached.clone());
            }
        }
        // Cache lock released — execute the (potentially expensive) query

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
            GraphQuery::DirectedTraversal {
                start,
                direction,
                depth,
            } => {
                let spec = TraversalSpec {
                    start,
                    direction,
                    depth,
                    instruction: Instruction::Collect,
                    node_filter: None,
                    edge_filter: None,
                    max_nodes_visited: None,
                    max_edges_traversed: None,
                    time_window: None,
                };
                execute_traversal(graph, &spec)
            },
            GraphQuery::RecursiveTraversal(request) => {
                let spec = request.compile();
                execute_traversal(graph, &spec)
            },
        }?;

        // Re-acquire cache lock to insert result
        self.cache.lock().insert(cache_key, result.clone());

        Ok(result)
    }
}
