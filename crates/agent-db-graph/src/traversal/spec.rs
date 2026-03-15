//! Traversal specification types: TraversalSpec, TraversalRequest, Instruction,
//! filter expressions, and the execute_traversal dispatch.

use super::cache::PathEntry;
use super::edge_cost::edge_cost;
use super::helpers::edge_type_name;
use super::types::QueryResult;
use super::GraphTraversal;
use crate::structures::{
    Depth, Direction, Graph, GraphEdge, GraphNode, NodeId,
};
use crate::{GraphResult};
use ordered_float::OrderedFloat;
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::Arc;

/// Predicate for filtering nodes during traversal (closure-based, internal use).
pub type NodePredicate = Arc<dyn Fn(&GraphNode) -> bool + Send + Sync>;

/// Predicate for filtering edges during traversal (closure-based, internal use).
pub type EdgePredicate = Arc<dyn Fn(&GraphEdge) -> bool + Send + Sync>;

/// Serializable node filter expression for the public query API.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum NodeFilterExpr {
    /// Match nodes whose `type_name()` equals the given string.
    ByType(String),
    /// Match nodes created after the given timestamp.
    CreatedAfter(u64),
    /// Match nodes created before the given timestamp.
    CreatedBefore(u64),
    /// Match nodes with degree >= threshold.
    MinDegree(u32),
}

/// Serializable edge filter expression for the public query API.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum EdgeFilterExpr {
    /// Match edges whose type name equals the given string.
    ByType(String),
    /// Match edges with weight >= threshold.
    MinWeight(OrderedFloat<f32>),
    /// Match edges created after the given timestamp.
    CreatedAfter(u64),
    /// Match edges created before the given timestamp.
    CreatedBefore(u64),
}

/// What to do with traversed nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Instruction {
    /// Collect all reachable nodes.
    Collect,
    /// Find simple paths (no repeated nodes), stop after `max_paths`.
    Path { max_paths: usize },
    /// Find shortest path to target via directed Dijkstra.
    Shortest(NodeId),
}

/// Internal traversal specification (closures allowed).
pub struct TraversalSpec {
    pub start: NodeId,
    pub direction: Direction,
    pub depth: Depth,
    pub instruction: Instruction,
    pub node_filter: Option<NodePredicate>,
    pub edge_filter: Option<EdgePredicate>,
    pub max_nodes_visited: Option<u32>,
    pub max_edges_traversed: Option<u32>,
    pub time_window: Option<(u64, u64)>,
}

/// Serializable traversal request for the public query API.
#[derive(Debug, Clone)]
pub struct TraversalRequest {
    pub start: NodeId,
    pub direction: Direction,
    pub depth: Depth,
    pub instruction: Instruction,
    pub node_filters: Vec<NodeFilterExpr>,
    pub edge_filters: Vec<EdgeFilterExpr>,
    pub max_nodes_visited: Option<u32>,
    pub max_edges_traversed: Option<u32>,
    pub time_window: Option<(u64, u64)>,
}

impl TraversalRequest {
    /// Compile this request into an executable `TraversalSpec` by converting
    /// filter expressions into closures.
    pub fn compile(self) -> TraversalSpec {
        let node_filter = if self.node_filters.is_empty() {
            None
        } else {
            let filters = self.node_filters;
            Some(Arc::new(move |node: &GraphNode| {
                filters.iter().all(|f| match f {
                    NodeFilterExpr::ByType(t) => node.type_name() == t.as_str(),
                    NodeFilterExpr::CreatedAfter(ts) => node.created_at > *ts,
                    NodeFilterExpr::CreatedBefore(ts) => node.created_at < *ts,
                    NodeFilterExpr::MinDegree(d) => node.degree >= *d,
                })
            }) as NodePredicate)
        };

        let edge_filter = if self.edge_filters.is_empty() {
            None
        } else {
            let filters = self.edge_filters;
            Some(Arc::new(move |edge: &GraphEdge| {
                filters.iter().all(|f| match f {
                    EdgeFilterExpr::ByType(t) => edge_type_name(&edge.edge_type) == *t,
                    EdgeFilterExpr::MinWeight(w) => edge.weight >= w.into_inner(),
                    EdgeFilterExpr::CreatedAfter(ts) => edge.created_at > *ts,
                    EdgeFilterExpr::CreatedBefore(ts) => edge.created_at < *ts,
                })
            }) as EdgePredicate)
        };

        TraversalSpec {
            start: self.start,
            direction: self.direction,
            depth: self.depth,
            instruction: self.instruction,
            node_filter,
            edge_filter,
            max_nodes_visited: self.max_nodes_visited,
            max_edges_traversed: self.max_edges_traversed,
            time_window: self.time_window,
        }
    }
}

// ============================================================================
// execute_traversal — dispatch on Instruction
// ============================================================================

/// Execute a traversal specification against a graph.
pub fn execute_traversal(graph: &Graph, spec: &TraversalSpec) -> GraphResult<QueryResult> {
    spec.depth.validate()?;
    match &spec.instruction {
        Instruction::Collect => execute_collect(graph, spec),
        Instruction::Path { max_paths } => execute_paths(graph, spec, *max_paths),
        Instruction::Shortest(target) => execute_shortest(graph, spec, *target),
    }
}

/// Resolve the "other endpoint" of an edge relative to `current` and `direction`.
#[inline]
fn edge_neighbor(edge: &GraphEdge, current: NodeId, direction: Direction) -> NodeId {
    match direction {
        Direction::Out => edge.target,
        Direction::In => edge.source,
        Direction::Both => {
            if edge.source == current {
                edge.target
            } else {
                edge.source
            }
        },
    }
}

/// Check if an edge passes time-window + edge-filter + node-filter for its neighbor.
#[inline]
fn edge_passes_filters(
    graph: &Graph,
    edge: &GraphEdge,
    neighbor: NodeId,
    spec: &TraversalSpec,
) -> bool {
    // Time window on edge
    if let Some((tw_start, tw_end)) = spec.time_window {
        if edge.created_at < tw_start || edge.created_at > tw_end {
            return false;
        }
    }
    // Edge filter
    if let Some(ref ef) = spec.edge_filter {
        if !ef(edge) {
            return false;
        }
    }
    // Time window on neighbor node
    if let Some((tw_start, tw_end)) = spec.time_window {
        if let Some(node) = graph.get_node(neighbor) {
            if node.created_at < tw_start || node.created_at > tw_end {
                return false;
            }
        }
    }
    // Node filter
    if let Some(ref nf) = spec.node_filter {
        if let Some(node) = graph.get_node(neighbor) {
            if !nf(node) {
                return false;
            }
        }
    }
    true
}

/// `Collect` instruction: BFS with direction, depth, filters, budgets.
fn execute_collect(graph: &Graph, spec: &TraversalSpec) -> GraphResult<QueryResult> {
    let min_depth = spec.depth.min_depth();
    let max_depth = spec.depth.max_depth();

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();
    let mut nodes_visited: u32 = 0;
    let mut edges_traversed: u32 = 0;

    visited.insert(spec.start);
    queue.push_back((spec.start, 0u32));

    while let Some((current, depth)) = queue.pop_front() {
        nodes_visited += 1;
        if let Some(max) = spec.max_nodes_visited {
            if nodes_visited > max {
                break;
            }
        }

        if depth >= min_depth {
            result.push(current);
        }

        let should_expand = max_depth.is_none_or(|max| depth < max);
        if !should_expand {
            continue;
        }

        let edges = graph.edges_directed(current, spec.direction);
        for edge in edges {
            edges_traversed += 1;
            if let Some(max) = spec.max_edges_traversed {
                if edges_traversed > max {
                    break;
                }
            }

            let neighbor = edge_neighbor(edge, current, spec.direction);
            if !edge_passes_filters(graph, edge, neighbor, spec) {
                continue;
            }
            if visited.insert(neighbor) {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }

    Ok(QueryResult::Nodes(result))
}

/// `Path` instruction: DFS simple-path enumeration with budgets.
fn execute_paths(
    graph: &Graph,
    spec: &TraversalSpec,
    max_paths: usize,
) -> GraphResult<QueryResult> {
    let max_depth = spec.depth.max_depth();
    let mut paths: Vec<Vec<NodeId>> = Vec::new();

    // Stack entries: (node, path_so_far, depth)
    let mut stack: Vec<(NodeId, Vec<NodeId>, u32)> = Vec::new();
    stack.push((spec.start, vec![spec.start], 0));

    let mut nodes_visited: u32 = 0;
    // Bound the stack size to prevent OOM from exponential path explosion
    const MAX_STACK_SIZE: usize = 100_000;

    while let Some((current, path, depth)) = stack.pop() {
        if paths.len() >= max_paths {
            break;
        }

        nodes_visited += 1;
        if let Some(max) = spec.max_nodes_visited {
            if nodes_visited > max {
                break;
            }
        }

        let at_max = max_depth.is_some_and(|max| depth >= max);
        if at_max {
            if depth >= spec.depth.min_depth() {
                paths.push(path);
            }
            continue;
        }

        let edges = graph.edges_directed(current, spec.direction);
        let mut expanded = false;

        for edge in edges {
            // Check bounds BEFORE cloning the path
            if paths.len() >= max_paths || stack.len() >= MAX_STACK_SIZE {
                break;
            }

            let neighbor = edge_neighbor(edge, current, spec.direction);

            // Simple path: no repeated nodes
            if path.contains(&neighbor) {
                continue;
            }

            if !edge_passes_filters(graph, edge, neighbor, spec) {
                continue;
            }

            expanded = true;
            let mut new_path = path.clone();
            new_path.push(neighbor);
            stack.push((neighbor, new_path, depth + 1));
        }

        // Leaf node — record path if it meets min_depth
        if !expanded && depth >= spec.depth.min_depth() {
            paths.push(path);
        }
    }

    Ok(QueryResult::Paths(paths))
}

/// `Shortest` instruction: directed Dijkstra to a specific target.
fn execute_shortest(
    graph: &Graph,
    spec: &TraversalSpec,
    target: NodeId,
) -> GraphResult<QueryResult> {
    if spec.start == target {
        return Ok(QueryResult::Path(vec![spec.start]));
    }

    let mut heap = BinaryHeap::new();
    let mut dist: FxHashMap<NodeId, f32> = FxHashMap::default();
    let mut came_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();

    dist.insert(spec.start, 0.0);
    heap.push(PathEntry {
        node_id: spec.start,
        cost: 0.0,
    });

    let max_depth = spec.depth.max_depth();
    let mut depth_map: FxHashMap<NodeId, u32> = FxHashMap::default();
    depth_map.insert(spec.start, 0);

    while let Some(PathEntry {
        node_id: current,
        cost,
    }) = heap.pop()
    {
        if current == target {
            return Ok(QueryResult::Path(GraphTraversal::reconstruct_path(
                &came_from, spec.start, target,
            )));
        }

        if cost > *dist.get(&current).unwrap_or(&f32::INFINITY) {
            continue;
        }

        let current_depth = depth_map.get(&current).copied().unwrap_or(0);
        if max_depth.is_some_and(|max| current_depth >= max) {
            continue;
        }

        for edge in graph.edges_directed(current, spec.direction) {
            let neighbor = edge_neighbor(edge, current, spec.direction);

            if !edge_passes_filters(graph, edge, neighbor, spec) {
                continue;
            }

            let w = edge_cost(edge);
            let new_dist = cost + w;

            if new_dist < *dist.get(&neighbor).unwrap_or(&f32::INFINITY) {
                dist.insert(neighbor, new_dist);
                came_from.insert(neighbor, current);
                depth_map.insert(neighbor, current_depth + 1);
                heap.push(PathEntry {
                    node_id: neighbor,
                    cost: new_dist,
                });
            }
        }
    }

    Err(crate::GraphError::NodeNotFound(
        "No path found to target".to_string(),
    ))
}
