//! Graph query methods: find nodes/edges, subgraph extraction, PageRank,
//! community detection, nearest-by-cost, deep reachability.

use super::edge_cost::{MAX_PAGERANK_ITERATIONS, MAX_PAGERANK_NODES, MAX_TRAVERSAL_EDGES, MAX_TRAVERSAL_NODES};
use super::helpers::edge_type_name;
use super::iterators::{BfsIter, DfsIter, DijkstraIter};
use super::types::{CommunityAlgorithm, QueryResult};
use super::GraphTraversal;
use crate::structures::{EdgeWeight, Graph, NodeId};
use crate::{GraphError, GraphResult};
use rustc_hash::FxHashMap;
use std::collections::HashSet;

impl GraphTraversal {
    /// Find nodes by type
    pub(crate) fn find_nodes_by_type(&self, graph: &Graph, node_type: &str) -> GraphResult<QueryResult> {
        let nodes = graph
            .get_nodes_by_type(node_type)
            .into_iter()
            .map(|node| node.id)
            .collect();

        Ok(QueryResult::Nodes(nodes))
    }

    /// Find neighbors within specified distance
    pub(crate) fn neighbors_within_distance(
        &self,
        graph: &Graph,
        start: NodeId,
        max_distance: u32,
    ) -> GraphResult<QueryResult> {
        // Delegates to the lazy BFS iterator. Bounded to prevent OOM on large components.
        let nodes: Vec<NodeId> = BfsIter::new(graph, start, max_distance)
            .take(MAX_TRAVERSAL_NODES)
            .map(|(node_id, _depth)| node_id)
            .collect();
        Ok(QueryResult::Nodes(nodes))
    }

    /// Find strongly connected components using Tarjan's algorithm.
    ///
    /// Uses an **iterative** implementation to avoid stack overflow on deep graphs.
    pub(crate) fn find_strongly_connected_components(&self, graph: &Graph) -> GraphResult<QueryResult> {
        let mut index: usize = 0;
        let mut scc_stack: Vec<NodeId> = Vec::new();
        let mut indices: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut lowlinks: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut on_stack: HashSet<NodeId> = HashSet::new();
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        let all_nodes: Vec<NodeId> = graph.node_ids();

        // Iterative Tarjan's using an explicit call stack.
        // Each frame stores: (node_id, neighbor_iterator_position, is_root_call)
        for &root in &all_nodes {
            if indices.contains_key(&root) {
                continue;
            }

            // call_stack entries: (node_id, Vec<neighbors>, next_neighbor_index)
            let mut call_stack: Vec<(NodeId, Vec<NodeId>, usize)> = Vec::new();

            // Initialize root
            indices.insert(root, index);
            lowlinks.insert(root, index);
            index += 1;
            scc_stack.push(root);
            on_stack.insert(root);

            let neighbors: Vec<NodeId> = graph.get_neighbors(root);
            call_stack.push((root, neighbors, 0));

            while let Some(frame) = call_stack.last_mut() {
                let node_id = frame.0;
                let ni_val = frame.2;

                if ni_val < frame.1.len() {
                    let neighbor = frame.1[ni_val];
                    frame.2 += 1;

                    match indices.entry(neighbor) {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            // "Recurse" — push new frame
                            e.insert(index);
                            lowlinks.insert(neighbor, index);
                            index += 1;
                            scc_stack.push(neighbor);
                            on_stack.insert(neighbor);

                            let next_neighbors = graph.get_neighbors(neighbor);
                            call_stack.push((neighbor, next_neighbors, 0));
                        },
                        std::collections::hash_map::Entry::Occupied(e) => {
                            if on_stack.contains(&neighbor) {
                                let ni_idx = *e.get();
                                if let Some(&nl) = lowlinks.get(&node_id) {
                                    lowlinks.insert(node_id, nl.min(ni_idx));
                                }
                            }
                        },
                    }
                } else {
                    // Done with all neighbors — "return" from this frame
                    let finished_node = node_id;
                    let finished_lowlink = lowlinks.get(&finished_node).copied().unwrap_or(0);
                    let finished_index = indices.get(&finished_node).copied().unwrap_or(0);

                    // Pop this frame
                    call_stack.pop();

                    // Update parent's lowlink (equivalent to post-recursion update)
                    if let Some(parent_frame) = call_stack.last() {
                        let parent_id = parent_frame.0;
                        if let Some(&parent_ll) = lowlinks.get(&parent_id) {
                            lowlinks.insert(parent_id, parent_ll.min(finished_lowlink));
                        }
                    }

                    // Check if this node is the root of an SCC
                    if finished_lowlink == finished_index {
                        let mut component = Vec::new();
                        loop {
                            let Some(w) = scc_stack.pop() else {
                                break;
                            };
                            on_stack.remove(&w);
                            component.push(w);
                            if w == finished_node {
                                break;
                            }
                        }
                        if !component.is_empty() {
                            components.push(component);
                        }
                    }
                }
            }
        }

        Ok(QueryResult::Communities(components))
    }

    /// Find nodes by property value
    pub(crate) fn find_nodes_by_property(
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
            .take(MAX_TRAVERSAL_NODES)
            .collect();

        Ok(QueryResult::Nodes(matching_nodes))
    }

    /// Find edges by type and weight threshold
    pub(crate) fn find_edges_by_type(
        &self,
        graph: &Graph,
        target_edge_type: &str,
        min_weight: EdgeWeight,
    ) -> GraphResult<QueryResult> {
        let matching_edges: Vec<crate::structures::EdgeId> = graph
            .edges
            .values()
            .filter(|edge| {
                edge.weight >= min_weight && edge_type_name(&edge.edge_type) == target_edge_type
            })
            .take(MAX_TRAVERSAL_EDGES)
            .map(|edge| edge.id)
            .collect();

        Ok(QueryResult::Edges(matching_edges))
    }

    /// Extract subgraph around a center node
    pub(crate) fn extract_subgraph(
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
                            .is_some_and(|node| types.iter().any(|t| t == node.type_name()))
                    })
                    .collect()
            } else {
                nodes
            };

            // Collect edges between the subgraph nodes (bounded to prevent OOM)
            let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
            let edges: Vec<crate::structures::EdgeId> = graph
                .edges
                .values()
                .filter(|e| node_set.contains(&e.source) && node_set.contains(&e.target))
                .take(MAX_TRAVERSAL_EDGES)
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
    pub(crate) fn calculate_pagerank(
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

        // Safety: refuse to run PageRank on very large graphs to prevent OOM
        if all_nodes.len() > MAX_PAGERANK_NODES {
            tracing::warn!(
                "PageRank skipped: {} nodes exceeds limit {}",
                all_nodes.len(),
                MAX_PAGERANK_NODES
            );
            return Ok(QueryResult::Rankings(Vec::new()));
        }

        let capped_iterations = iterations.min(MAX_PAGERANK_ITERATIONS);

        let mut pagerank: FxHashMap<NodeId, f32> =
            FxHashMap::with_capacity_and_hasher(all_nodes.len(), Default::default());
        let mut new_pagerank: FxHashMap<NodeId, f32> =
            FxHashMap::with_capacity_and_hasher(all_nodes.len(), Default::default());

        // Initialize PageRank scores
        let init = 1.0 / n;
        for &node_id in &all_nodes {
            pagerank.insert(node_id, init);
        }

        // Iterate PageRank calculation
        for _ in 0..capped_iterations {
            new_pagerank.clear();

            for &node_id in &all_nodes {
                let mut rank = (1.0 - damping_factor) / n;

                for incoming_neighbor in graph.get_incoming_neighbors(node_id) {
                    let neighbor_out_degree = graph.get_neighbors(incoming_neighbor).len() as f32;
                    if neighbor_out_degree > 0.0 {
                        // Safe lookup: use get() instead of [] to avoid panic
                        if let Some(&neighbor_rank) = pagerank.get(&incoming_neighbor) {
                            rank += damping_factor * neighbor_rank / neighbor_out_degree;
                        }
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
    pub(crate) fn detect_communities(
        &self,
        graph: &Graph,
        algorithm: &CommunityAlgorithm,
    ) -> GraphResult<QueryResult> {
        match algorithm {
            CommunityAlgorithm::ConnectedComponents => {
                self.find_strongly_connected_components(graph)
            },
            CommunityAlgorithm::Louvain { resolution } => {
                let config = crate::algorithms::LouvainConfig {
                    resolution: *resolution,
                    ..Default::default()
                };
                let louvain = crate::algorithms::LouvainAlgorithm::with_config(config);
                let result = louvain.detect_communities(graph)?;
                // Convert to Vec<Vec<NodeId>> for QueryResult::Communities
                let mut communities: Vec<Vec<NodeId>> = result.communities.into_values().collect();
                communities.sort_by_key(|b| std::cmp::Reverse(b.len())); // largest first
                Ok(QueryResult::Communities(communities))
            },
            CommunityAlgorithm::LabelPropagation { iterations } => {
                let config = crate::algorithms::LabelPropagationConfig {
                    max_iterations: *iterations,
                };
                let lp = crate::algorithms::LabelPropagationAlgorithm::with_config(config);
                let result = lp.detect_communities(graph)?;
                let mut communities: Vec<Vec<NodeId>> = result.communities.into_values().collect();
                communities.sort_by_key(|b| std::cmp::Reverse(b.len()));
                Ok(QueryResult::Communities(communities))
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
    pub(crate) fn nearest_by_cost(&self, graph: &Graph, start: NodeId, k: usize) -> GraphResult<QueryResult> {
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
    pub(crate) fn deep_reachability(
        &self,
        graph: &Graph,
        start: NodeId,
        max_depth: u32,
    ) -> GraphResult<QueryResult> {
        let nodes: Vec<NodeId> = DfsIter::new(graph, start, max_depth)
            .take(MAX_TRAVERSAL_NODES)
            .map(|(node_id, _depth)| node_id)
            .collect();
        Ok(QueryResult::Nodes(nodes))
    }
}
