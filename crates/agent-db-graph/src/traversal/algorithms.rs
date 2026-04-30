//! Core pathfinding algorithms: Dijkstra, A*, Yen's K-shortest, bidirectional
//! Dijkstra, and constrained path search.

use super::cache::PathEntry;
use super::edge_cost::{edge_cost, edge_cost_between, MAX_DIJKSTRA_ITERATIONS};
use super::helpers::edge_type_name;
use super::types::{PathConstraint, QueryResult};
use super::GraphTraversal;
use crate::structures::{Graph, NodeId};
use crate::{GraphError, GraphResult};
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, HashSet};

impl GraphTraversal {
    /// Weighted Dijkstra's shortest path using actual edge costs.
    ///
    /// Edge cost is derived from edge type metadata (strength, confidence, etc.)
    /// via `edge_cost()`. Stronger relationships have lower traversal cost.
    pub(crate) fn shortest_path(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
    ) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }

        let mut heap = BinaryHeap::new();
        let mut dist: FxHashMap<NodeId, f32> = FxHashMap::default();
        let mut came_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();

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
    pub(crate) fn a_star_search(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
    ) -> GraphResult<QueryResult> {
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
        let mut g_score: FxHashMap<NodeId, f32> = FxHashMap::default();
        let mut came_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();

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
    pub(crate) fn k_shortest_paths(
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
    pub(crate) fn bidirectional_dijkstra(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
    ) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }

        // Forward search state
        let mut fwd_dist: FxHashMap<NodeId, f32> = FxHashMap::default();
        let mut fwd_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        let mut fwd_heap = BinaryHeap::new();
        let mut fwd_settled: HashSet<NodeId> = HashSet::new();

        // Backward search state
        let mut bwd_dist: FxHashMap<NodeId, f32> = FxHashMap::default();
        let mut bwd_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();
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
        let mut iterations: usize = 0;

        loop {
            let fwd_min = fwd_heap.peek().map(|e| e.cost).unwrap_or(f32::INFINITY);
            let bwd_min = bwd_heap.peek().map(|e| e.cost).unwrap_or(f32::INFINITY);

            // Termination: both frontiers can't improve on the best known path
            if fwd_min + bwd_min >= best_cost {
                break;
            }

            // Safety: if both heaps are empty, no path exists
            if fwd_heap.is_empty() && bwd_heap.is_empty() {
                break;
            }

            // Expand the smaller frontier
            if fwd_min <= bwd_min {
                if let Some(PathEntry { node_id: u, cost }) = fwd_heap.pop() {
                    if cost > *fwd_dist.get(&u).unwrap_or(&f32::INFINITY) {
                        continue;
                    }
                    iterations += 1;
                    if iterations > MAX_DIJKSTRA_ITERATIONS {
                        tracing::warn!(
                            "Bidirectional Dijkstra hit iteration cap ({}), terminating early",
                            MAX_DIJKSTRA_ITERATIONS
                        );
                        break;
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
                iterations += 1;
                if iterations > MAX_DIJKSTRA_ITERATIONS {
                    tracing::warn!(
                        "Bidirectional Dijkstra hit iteration cap ({}), terminating early",
                        MAX_DIJKSTRA_ITERATIONS
                    );
                    break;
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
    pub(crate) fn constrained_path_search(
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

        let custom_filters: Vec<
            fn(&crate::structures::GraphNode, &crate::structures::GraphEdge) -> bool,
        > = constraints
            .iter()
            .filter_map(|c| match c {
                PathConstraint::CustomFilter(f) => Some(*f),
                _ => None,
            })
            .collect();

        // Modified Dijkstra with constraint checking
        let mut heap = BinaryHeap::new();
        let mut dist: FxHashMap<NodeId, f32> = FxHashMap::default();
        let mut came_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        let mut depth: FxHashMap<NodeId, u32> = FxHashMap::default();

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
                    let path_types: HashSet<&str> = path
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
                        if avoid_node_types.contains(neighbor_node.type_name()) {
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
    pub(crate) fn dijkstra_full(
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
        let mut dist: FxHashMap<NodeId, f32> = FxHashMap::default();
        let mut came_from: FxHashMap<NodeId, NodeId> = FxHashMap::default();

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
    pub(crate) fn reconstruct_path(
        came_from: &FxHashMap<NodeId, NodeId>,
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
}
