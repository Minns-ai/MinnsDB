//! Parallel graph processing algorithms
//!
//! Uses rayon for data-parallel execution across available CPU cores.
//! Algorithms that are embarrassingly parallel (BFS from multiple sources,
//! PageRank iterations, degree/centrality computations, batch event processing)
//! use `.par_iter()` for automatic work-stealing across the rayon thread pool.
//!
//! Inherently sequential phases (Union-Find merge, BFS frontier expansion)
//! remain sequential to preserve correctness.

use crate::structures::{Graph, NodeId};
use crate::traversal::{edge_cost, BfsIter};
use crate::GraphResult;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Parallel graph algorithm implementations
pub struct ParallelGraphAlgorithms;

impl ParallelGraphAlgorithms {
    /// Create new parallel algorithm executor
    pub fn new() -> Self {
        Self
    }

    /// Parallel BFS from multiple starting nodes
    /// Useful for finding reachable nodes from multiple sources simultaneously
    pub fn parallel_multi_source_bfs(
        &self,
        graph: &Graph,
        starts: Vec<NodeId>,
        max_depth: u32,
    ) -> GraphResult<HashMap<NodeId, u32>> {
        // Run BFS from each start in parallel using the lazy BfsIter.
        // Each rayon task gets its own iterator — no shared mutable state.
        let results: Vec<HashMap<NodeId, u32>> = starts
            .par_iter()
            .map(|&start| BfsIter::new(graph, start, max_depth).collect())
            .collect();

        // Merge results: take minimum distance for each node
        let mut merged: HashMap<NodeId, u32> = HashMap::new();
        for result in results {
            for (node, dist) in result {
                merged
                    .entry(node)
                    .and_modify(|d| *d = (*d).min(dist))
                    .or_insert(dist);
            }
        }

        Ok(merged)
    }

    /// Parallel PageRank calculation
    /// Distributes node updates across threads
    pub fn parallel_pagerank(
        &self,
        graph: &Graph,
        damping_factor: f32,
        iterations: usize,
        tolerance: f32,
    ) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len() as f32;

        if n == 0.0 {
            return Ok(HashMap::new());
        }

        // Initialize with uniform distribution
        let mut pagerank: HashMap<NodeId, f32> = nodes.iter().map(|&id| (id, 1.0 / n)).collect();

        for _ in 0..iterations {
            // Parallel update of all nodes
            let new_values: Vec<(NodeId, f32)> = nodes
                .par_iter()
                .map(|&node_id| {
                    // Base rank (random jump probability)
                    let mut rank = (1.0 - damping_factor) / n;

                    // Add contribution from incoming neighbors
                    for incoming in graph.get_incoming_neighbors(node_id) {
                        let out_degree = graph.get_neighbors(incoming).len() as f32;
                        if out_degree > 0.0 {
                            rank += damping_factor * pagerank[&incoming] / out_degree;
                        }
                    }

                    (node_id, rank)
                })
                .collect();

            // Check convergence
            let max_diff = new_values
                .iter()
                .map(|(id, new_rank)| (new_rank - pagerank[id]).abs())
                .fold(0.0f32, f32::max);

            // Update pagerank
            for (node_id, rank) in new_values {
                pagerank.insert(node_id, rank);
            }

            if max_diff < tolerance {
                break;
            }
        }

        let sum: f32 = pagerank.values().sum();
        if sum > 0.0 {
            for value in pagerank.values_mut() {
                *value /= sum;
            }
        }

        Ok(pagerank)
    }

    /// Parallel degree centrality calculation
    pub fn parallel_degree_centrality(&self, graph: &Graph) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len() as f32;

        if n <= 1.0 {
            return Ok(HashMap::new());
        }

        let centrality: HashMap<NodeId, f32> = nodes
            .par_iter()
            .map(|&node_id| {
                let degree = graph.get_neighbors(node_id).len() as f32;
                let normalized = degree / (n - 1.0);
                (node_id, normalized)
            })
            .collect();

        Ok(centrality)
    }

    /// Parallel weighted shortest paths from one source to multiple targets.
    ///
    /// Uses Dijkstra's algorithm with actual edge costs derived from edge
    /// metadata (strength, confidence, similarity, etc.) so that stronger
    /// relationships produce shorter paths.  Path reconstruction for each
    /// target is parallelised with rayon.
    pub fn parallel_shortest_paths(
        &self,
        graph: &Graph,
        source: NodeId,
        targets: Vec<NodeId>,
    ) -> GraphResult<HashMap<NodeId, Vec<NodeId>>> {
        // --- single-source Dijkstra ------------------------------------------
        let mut dist: HashMap<NodeId, f32> = HashMap::new();
        let mut predecessors: HashMap<NodeId, NodeId> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source, 0.0);
        heap.push(DijkstraEntry {
            cost: 0.0,
            node: source,
        });

        while let Some(DijkstraEntry { cost, node }) = heap.pop() {
            // Skip if we already found a cheaper route
            if let Some(&best) = dist.get(&node) {
                if cost > best {
                    continue;
                }
            }

            for edge in graph.get_edges_from(node) {
                let next = edge.target;
                let next_cost = cost + edge_cost(edge);

                let is_better = match dist.get(&next) {
                    Some(&d) => next_cost < d,
                    None => true,
                };

                if is_better {
                    dist.insert(next, next_cost);
                    predecessors.insert(next, node);
                    heap.push(DijkstraEntry {
                        cost: next_cost,
                        node: next,
                    });
                }
            }
        }

        // --- parallel path reconstruction ------------------------------------
        let paths: HashMap<NodeId, Vec<NodeId>> = targets
            .par_iter()
            .filter_map(|&target| {
                if !dist.contains_key(&target) {
                    return None;
                }

                let mut path = vec![target];
                let mut current = target;

                while current != source {
                    match predecessors.get(&current) {
                        Some(&prev) => {
                            current = prev;
                            path.push(current);
                        },
                        None => return None,
                    }
                }

                path.reverse();
                Some((target, path))
            })
            .collect();

        Ok(paths)
    }

    /// Parallel connected components detection
    /// Uses Union-Find with parallel initialization
    pub fn parallel_connected_components(
        &self,
        graph: &Graph,
    ) -> GraphResult<HashMap<NodeId, u64>> {
        let nodes = graph.get_all_node_ids();

        // Initialize: each node in its own component
        let components: HashMap<NodeId, u64> =
            nodes.iter().map(|&node_id| (node_id, node_id)).collect();

        // Use sequential Union-Find for actual merging
        // (Parallel Union-Find is complex and requires lock-free data structures)
        let mut components = components;
        let mut changed = true;

        while changed {
            changed = false;

            for &node_id in &nodes {
                let node_component = components[&node_id];

                for neighbor in graph.get_neighbors(node_id) {
                    let neighbor_component = components[&neighbor];

                    if node_component != neighbor_component {
                        // Merge components: assign all nodes in larger component ID
                        let min_component = node_component.min(neighbor_component);
                        let max_component = node_component.max(neighbor_component);

                        for (&_id, comp) in components.iter_mut() {
                            if *comp == max_component {
                                *comp = min_component;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        Ok(components)
    }

    /// Parallel node property computation
    /// Compute custom property for each node in parallel
    pub fn parallel_node_computation<F>(
        &self,
        graph: &Graph,
        compute_fn: F,
    ) -> GraphResult<HashMap<NodeId, f32>>
    where
        F: Fn(&Graph, NodeId) -> f32 + Sync + Send,
    {
        let nodes = graph.get_all_node_ids();

        let results: HashMap<NodeId, f32> = nodes
            .par_iter()
            .map(|&node_id| {
                let value = compute_fn(graph, node_id);
                (node_id, value)
            })
            .collect();

        Ok(results)
    }

    /// Parallel batch event processing
    /// Process multiple events simultaneously (for EventGraphDB)
    pub fn parallel_event_batch_process<F>(
        &self,
        events: Vec<EventId>,
        process_fn: F,
    ) -> GraphResult<Vec<ProcessResult>>
    where
        F: Fn(EventId) -> ProcessResult + Sync + Send,
    {
        let results: Vec<ProcessResult> = events
            .par_iter()
            .map(|&event_id| process_fn(event_id))
            .collect();

        Ok(results)
    }

    /// Parallel community detection preparation
    /// Compute node degrees and edge weights in parallel
    pub fn parallel_community_prep(&self, graph: &Graph) -> GraphResult<CommunityPrepData> {
        let nodes = graph.get_all_node_ids();

        // Parallel degree calculation
        let degrees: HashMap<NodeId, f32> = nodes
            .par_iter()
            .map(|&node_id| {
                let degree = graph.get_node_degree(node_id);
                (node_id, degree)
            })
            .collect();

        // Parallel edge weight sum
        let total_weight: f32 = graph
            .get_all_edges()
            .par_iter()
            .map(|edge| edge.weight)
            .sum();

        Ok(CommunityPrepData {
            node_degrees: degrees,
            total_edge_weight: total_weight,
        })
    }
}

impl Default for ParallelGraphAlgorithms {
    fn default() -> Self {
        Self::new()
    }
}

/// Priority queue entry for Dijkstra's algorithm (min-heap via reversed Ord).
#[derive(Debug, Clone, PartialEq)]
struct DijkstraEntry {
    cost: f32,
    node: NodeId,
}

impl Eq for DijkstraEntry {}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Event ID type (for batch processing)
pub type EventId = u64;

/// Result of processing an event
#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub event_id: EventId,
    pub success: bool,
    pub nodes_created: u32,
    pub patterns_detected: u32,
}

/// Precomputed data for community detection
#[derive(Debug, Clone)]
pub struct CommunityPrepData {
    pub node_degrees: HashMap<NodeId, f32>,
    pub total_edge_weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    #[test]
    fn test_parallel_pagerank() {
        let mut graph = Graph::new();

        // Create a simple graph
        let n1 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 1,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n2 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 2,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n3 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 3,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        graph.add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Causality {
                strength: 0.9,
                lag_ms: 100,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n2,
            n3,
            EdgeType::Causality {
                strength: 0.9,
                lag_ms: 100,
            },
            0.9,
        ));

        let parallel_alg = ParallelGraphAlgorithms::new();
        let pagerank = parallel_alg
            .parallel_pagerank(&graph, 0.85, 50, 0.001)
            .unwrap();

        // Check that all nodes have PageRank
        assert!(pagerank.contains_key(&n1));
        assert!(pagerank.contains_key(&n2));
        assert!(pagerank.contains_key(&n3));

        // Sum should be approximately 1.0
        let sum: f32 = pagerank.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_parallel_degree_centrality() {
        let mut graph = Graph::new();

        let n1 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 1,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n2 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 2,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n3 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 3,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        // n1 connected to both n2 and n3
        graph.add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n1,
            n3,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));

        let parallel_alg = ParallelGraphAlgorithms::new();
        let centrality = parallel_alg.parallel_degree_centrality(&graph).unwrap();

        // n1 should have highest centrality
        assert!(centrality[&n1] > centrality[&n2]);
        assert!(centrality[&n1] > centrality[&n3]);
    }

    #[test]
    fn test_parallel_multi_source_bfs() {
        let mut graph = Graph::new();

        let n1 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 1,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n2 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 2,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n3 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 3,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n4 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 4,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        graph.add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n2,
            n3,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n3,
            n4,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));

        let parallel_alg = ParallelGraphAlgorithms::new();
        let distances = parallel_alg
            .parallel_multi_source_bfs(&graph, vec![n1], 10)
            .unwrap();

        // Check distances from n1
        assert_eq!(distances[&n1], 0);
        assert_eq!(distances[&n2], 1);
        assert_eq!(distances[&n3], 2);
        assert_eq!(distances[&n4], 3);
    }

    #[test]
    fn test_parallel_shortest_paths_dijkstra() {
        let mut graph = Graph::new();

        // Build a diamond graph:
        //
        //       n1
        //      /   \
        //    n2     n3
        //      \   /
        //       n4
        //
        // n1->n2 high strength (low cost)
        // n2->n4 high strength (low cost)
        // n1->n3 low strength  (high cost)
        // n3->n4 low strength  (high cost)
        //
        // Dijkstra should prefer n1->n2->n4 over n1->n3->n4.

        let n1 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 1,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n2 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 2,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n3 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 3,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n4 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 4,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        // Cheap route: n1 -> n2 -> n4  (strength 0.95 => cost 0.05 each)
        graph.add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Causality {
                strength: 0.95,
                lag_ms: 10,
            },
            0.95,
        ));
        graph.add_edge(GraphEdge::new(
            n2,
            n4,
            EdgeType::Causality {
                strength: 0.95,
                lag_ms: 10,
            },
            0.95,
        ));

        // Expensive route: n1 -> n3 -> n4  (strength 0.2 => cost 0.8 each)
        graph.add_edge(GraphEdge::new(
            n1,
            n3,
            EdgeType::Causality {
                strength: 0.2,
                lag_ms: 500,
            },
            0.2,
        ));
        graph.add_edge(GraphEdge::new(
            n3,
            n4,
            EdgeType::Causality {
                strength: 0.2,
                lag_ms: 500,
            },
            0.2,
        ));

        let alg = ParallelGraphAlgorithms::new();
        let paths = alg
            .parallel_shortest_paths(&graph, n1, vec![n4, n2, n3])
            .unwrap();

        // n1 -> n4 should go through n2 (cheaper), not n3
        let path_to_n4 = &paths[&n4];
        assert_eq!(path_to_n4, &vec![n1, n2, n4]);

        // n1 -> n2 is a direct edge
        let path_to_n2 = &paths[&n2];
        assert_eq!(path_to_n2, &vec![n1, n2]);

        // n1 -> n3 is a direct edge
        let path_to_n3 = &paths[&n3];
        assert_eq!(path_to_n3, &vec![n1, n3]);
    }
}
