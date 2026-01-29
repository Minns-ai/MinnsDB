//! Parallel graph processing algorithms
//!
//! Note: Parallel features require Rust 1.80+ (rayon dependency)
//! This module is currently compiled in sequential mode for compatibility.
//! To enable parallel processing, uncomment rayon in Cargo.toml and upgrade Rust.
//!
//! Target: 4-8x speedup on multi-core CPUs when enabled

use crate::structures::{Graph, NodeId};
use crate::GraphResult;
// use rayon::prelude::*;  // Requires Rust 1.80+
use std::collections::{HashMap, HashSet, VecDeque};

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
        // Run BFS from each start in parallel
        let results: Vec<HashMap<NodeId, u32>> = starts
            .iter()
            .map(|&start| {
                let mut distances = HashMap::new();
                let mut queue = VecDeque::new();
                let mut visited = HashSet::new();

                queue.push_back((start, 0));
                visited.insert(start);
                distances.insert(start, 0);

                while let Some((current, depth)) = queue.pop_front() {
                    if depth >= max_depth {
                        continue;
                    }

                    for neighbor in graph.get_neighbors(current) {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            distances.insert(neighbor, depth + 1);
                            queue.push_back((neighbor, depth + 1));
                        }
                    }
                }

                distances
            })
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
                .iter()
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
            .iter()
            .map(|&node_id| {
                let degree = graph.get_neighbors(node_id).len() as f32;
                let normalized = degree / (n - 1.0);
                (node_id, normalized)
            })
            .collect();

        Ok(centrality)
    }

    /// Parallel shortest paths from one source to multiple targets
    pub fn parallel_shortest_paths(
        &self,
        graph: &Graph,
        source: NodeId,
        targets: Vec<NodeId>,
    ) -> GraphResult<HashMap<NodeId, Vec<NodeId>>> {
        // First, run single-source shortest path to get distances
        let mut queue = VecDeque::new();
        let mut distances: HashMap<NodeId, u32> = HashMap::new();
        let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        queue.push_back(source);
        distances.insert(source, 0);

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[&current];

            for neighbor in graph.get_neighbors(current) {
                if !distances.contains_key(&neighbor) {
                    distances.insert(neighbor, current_dist + 1);
                    predecessors.insert(neighbor, vec![current]);
                    queue.push_back(neighbor);
                } else if distances[&neighbor] == current_dist + 1 {
                    predecessors.entry(neighbor).or_insert_with(Vec::new).push(current);
                }
            }
        }

        // Parallel path reconstruction for each target
        let paths: HashMap<NodeId, Vec<NodeId>> = targets
            .iter()
            .filter_map(|&target| {
                if !distances.contains_key(&target) {
                    return None;
                }

                // Reconstruct path
                let mut path = vec![target];
                let mut current = target;

                while current != source {
                    if let Some(preds) = predecessors.get(&current) {
                        if preds.is_empty() {
                            return None;
                        }
                        // Take first predecessor (one of potentially many shortest paths)
                        current = preds[0];
                        path.push(current);
                    } else {
                        return None;
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
        let components: HashMap<NodeId, u64> = nodes
            .iter()
            .map(|&node_id| (node_id, node_id as u64))
            .collect();

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
            .iter()
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
            .iter()
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
            .iter()
            .map(|&node_id| {
                let degree = graph.get_node_degree(node_id);
                (node_id, degree)
            })
            .collect();

        // Parallel edge weight sum
        let total_weight: f32 = graph.get_all_edges().iter().map(|edge| edge.weight).sum();

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
        let n1 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n2 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 2,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n3 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 3,
            event_type: "test".to_string(),
            significance: 0.5,
        }));

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

        let n1 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n2 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 2,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n3 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 3,
            event_type: "test".to_string(),
            significance: 0.5,
        }));

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

        let n1 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n2 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 2,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n3 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 3,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n4 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 4,
            event_type: "test".to_string(),
            significance: 0.5,
        }));

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
}
