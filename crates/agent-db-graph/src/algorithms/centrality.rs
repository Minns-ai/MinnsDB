//! Centrality measures for identifying important nodes
//!
//! Centrality identifies "important" nodes in the graph:
//! - High centrality actions = frequently successful patterns
//! - High centrality contexts = common situations
//! - High centrality events = key decision points

use crate::structures::{Graph, NodeId};
use crate::GraphResult;
use std::collections::{HashMap, HashSet, VecDeque};

/// Centrality algorithm implementations
pub struct CentralityMeasures;

impl CentralityMeasures {
    /// Create new centrality calculator
    pub fn new() -> Self {
        Self
    }

    /// Calculate degree centrality for all nodes
    /// Simple but effective: degree = number of connections
    pub fn degree_centrality(&self, graph: &Graph) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len() as f32;

        if n <= 1.0 {
            return Ok(HashMap::new());
        }

        let mut centrality = HashMap::new();

        for &node_id in &nodes {
            let degree = graph.get_neighbors(node_id).len() as f32;
            // Normalize by maximum possible degree (n-1)
            let normalized_degree = degree / (n - 1.0);
            centrality.insert(node_id, normalized_degree);
        }

        Ok(centrality)
    }

    /// Calculate betweenness centrality using Brandes' algorithm — O(V*E).
    ///
    /// Measures how often a node appears on shortest paths between other nodes.
    /// High betweenness = "bridge" or "gatekeeper" node.
    ///
    /// For each source *s* the algorithm:
    /// 1. Runs a BFS to compute shortest-path counts σ and predecessor lists.
    /// 2. Back-propagates a dependency value δ through the BFS tree.
    /// 3. Accumulates δ into the global betweenness scores.
    ///
    /// The result is normalized by (n-1)(n-2) for directed graphs so that
    /// values fall in [0, 1].
    pub fn betweenness_centrality(&self, graph: &Graph) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len();

        if n <= 2 {
            return Ok(nodes.iter().map(|&id| (id, 0.0)).collect());
        }

        // Accumulator: C_B[v] for every v
        let mut cb: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, 0.0)).collect();

        for &s in &nodes {
            // --- single-source shortest-path (BFS) ---
            // Stack of nodes in order of non-decreasing distance from s
            let mut stack: Vec<NodeId> = Vec::new();
            // Predecessors on shortest paths from s
            let mut pred: HashMap<NodeId, Vec<NodeId>> = nodes
                .iter()
                .map(|&v| (v, Vec::new()))
                .collect();
            // σ[t] = number of shortest paths from s to t
            let mut sigma: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
            *sigma.get_mut(&s).unwrap() = 1.0;
            // d[t] = distance from s to t (-1 = unseen)
            let mut dist: HashMap<NodeId, i64> = nodes.iter().map(|&v| (v, -1_i64)).collect();
            *dist.get_mut(&s).unwrap() = 0;

            let mut queue: VecDeque<NodeId> = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let dv = dist[&v];

                for w in graph.get_neighbors(v) {
                    // w found for the first time?
                    if dist[&w] < 0 {
                        queue.push_back(w);
                        *dist.get_mut(&w).unwrap() = dv + 1;
                    }
                    // shortest path to w via v?
                    if dist[&w] == dv + 1 {
                        *sigma.get_mut(&w).unwrap() += sigma[&v];
                        pred.get_mut(&w).unwrap().push(v);
                    }
                }
            }

            // --- dependency accumulation ---
            let mut delta: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();

            // Pop nodes from stack in reverse BFS order (furthest first)
            while let Some(w) = stack.pop() {
                let sw = sigma[&w];
                if sw == 0.0 {
                    continue;
                }
                let dw = delta[&w];
                for v in &pred[&w] {
                    let contribution = (sigma[v] / sw) * (1.0 + dw);
                    *delta.get_mut(v).unwrap() += contribution;
                }
                if w != s {
                    *cb.get_mut(&w).unwrap() += delta[&w];
                }
            }
        }

        // Normalize: for directed graphs the normalization factor is (n-1)(n-2)
        let normalization = ((n - 1) * (n - 2)) as f64;
        let mut result: HashMap<NodeId, f32> = HashMap::with_capacity(n);
        for (&node, &score) in &cb {
            let normalized = if normalization > 0.0 {
                score / normalization
            } else {
                0.0
            };
            result.insert(node, normalized as f32);
        }

        Ok(result)
    }

    /// Calculate closeness centrality
    /// Measures average distance to all other nodes
    /// High closeness = centrally located, can reach others quickly
    pub fn closeness_centrality(&self, graph: &Graph) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len();

        if n <= 1 {
            return Ok(HashMap::new());
        }

        let mut centrality = HashMap::new();

        for &node_id in &nodes {
            let mut total_distance = 0.0;
            let mut reachable_count = 0;

            // Calculate distance to all other nodes
            for &other_id in &nodes {
                if other_id == node_id {
                    continue;
                }

                if let Some(distance) = self.shortest_distance(graph, node_id, other_id)? {
                    total_distance += distance;
                    reachable_count += 1;
                }
            }

            // Closeness = (reachable - 1) / sum_of_distances
            // Higher value = more central
            let closeness = if total_distance > 0.0 {
                (reachable_count as f32) / total_distance
            } else {
                0.0
            };

            // Normalize by (n-1) to get value between 0 and 1
            let normalized = closeness / (n - 1) as f32;
            centrality.insert(node_id, normalized);
        }

        Ok(centrality)
    }

    /// Calculate eigenvector centrality
    /// Node is important if connected to other important nodes
    /// Like Google PageRank but for undirected graphs
    pub fn eigenvector_centrality(
        &self,
        graph: &Graph,
        max_iterations: usize,
        tolerance: f32,
    ) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(HashMap::new());
        }

        // Initialize with uniform values
        let mut centrality: HashMap<NodeId, f32> =
            nodes.iter().map(|&id| (id, 1.0 / n as f32)).collect();

        for _ in 0..max_iterations {
            let mut new_centrality = HashMap::new();
            let mut max_diff: f32 = 0.0;

            // For each node, new centrality = sum of neighbors' centralities
            for &node_id in &nodes {
                let neighbor_sum: f32 = graph
                    .get_neighbors(node_id)
                    .iter()
                    .map(|&neighbor| centrality.get(&neighbor).copied().unwrap_or(0.0))
                    .sum();

                new_centrality.insert(node_id, neighbor_sum);

                let diff = (neighbor_sum - centrality[&node_id]).abs();
                max_diff = max_diff.max(diff);
            }

            // Normalize so values sum to 1
            let sum: f32 = new_centrality.values().sum();
            if sum > 0.0 {
                for value in new_centrality.values_mut() {
                    *value /= sum;
                }
            }

            // Check convergence
            if max_diff < tolerance {
                return Ok(new_centrality);
            }

            centrality = new_centrality;
        }

        Ok(centrality)
    }

    /// Calculate PageRank (similar to eigenvector but with damping)
    /// Standard algorithm used by Google for web pages
    pub fn pagerank(
        &self,
        graph: &Graph,
        damping_factor: f32,
        max_iterations: usize,
        tolerance: f32,
    ) -> GraphResult<HashMap<NodeId, f32>> {
        let nodes = graph.get_all_node_ids();
        let n = nodes.len() as f32;

        if n == 0.0 {
            return Ok(HashMap::new());
        }

        // Initialize with uniform distribution
        let mut pagerank: HashMap<NodeId, f32> = nodes.iter().map(|&id| (id, 1.0 / n)).collect();

        for _ in 0..max_iterations {
            let mut new_pagerank = HashMap::new();
            let mut max_diff: f32 = 0.0;

            for &node_id in &nodes {
                // Base rank (random jump probability)
                let mut rank = (1.0 - damping_factor) / n;

                // Add contribution from incoming neighbors
                for incoming in graph.get_incoming_neighbors(node_id) {
                    let out_degree = graph.get_neighbors(incoming).len() as f32;
                    if out_degree > 0.0 {
                        rank += damping_factor * pagerank[&incoming] / out_degree;
                    }
                }

                new_pagerank.insert(node_id, rank);

                let diff = (rank - pagerank[&node_id]).abs();
                max_diff = max_diff.max(diff);
            }

            // Check convergence
            if max_diff < tolerance {
                return Ok(new_pagerank);
            }

            pagerank = new_pagerank;
        }

        Ok(pagerank)
    }

    /// Find shortest distance between two nodes (BFS)
    fn shortest_distance(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
    ) -> GraphResult<Option<f32>> {
        if start == end {
            return Ok(Some(0.0));
        }

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut distances: HashMap<NodeId, f32> = HashMap::new();

        queue.push_back(start);
        visited.insert(start);
        distances.insert(start, 0.0);

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[&current];

            for neighbor in graph.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    distances.insert(neighbor, current_dist + 1.0);
                    queue.push_back(neighbor);

                    if neighbor == end {
                        return Ok(Some(current_dist + 1.0));
                    }
                }
            }
        }

        Ok(None) // No path found
    }

    /// Get top N nodes by centrality
    pub fn top_nodes(centrality: &HashMap<NodeId, f32>, n: usize) -> Vec<(NodeId, f32)> {
        let mut sorted: Vec<_> = centrality.iter().map(|(&id, &score)| (id, score)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Calculate all centrality measures at once
    pub fn all_centralities(&self, graph: &Graph) -> GraphResult<AllCentralities> {
        Ok(AllCentralities {
            degree: self.degree_centrality(graph)?,
            betweenness: self.betweenness_centrality(graph)?,
            closeness: self.closeness_centrality(graph)?,
            eigenvector: self.eigenvector_centrality(graph, 100, 0.0001)?,
            pagerank: self.pagerank(graph, 0.85, 100, 0.0001)?,
        })
    }
}

impl Default for CentralityMeasures {
    fn default() -> Self {
        Self::new()
    }
}

/// All centrality measures for a graph
#[derive(Debug, Clone)]
pub struct AllCentralities {
    pub degree: HashMap<NodeId, f32>,
    pub betweenness: HashMap<NodeId, f32>,
    pub closeness: HashMap<NodeId, f32>,
    pub eigenvector: HashMap<NodeId, f32>,
    pub pagerank: HashMap<NodeId, f32>,
}

impl AllCentralities {
    /// Get combined score (average of all centralities)
    pub fn combined_score(&self, node_id: NodeId) -> f32 {
        let scores = [
            self.degree.get(&node_id).copied().unwrap_or(0.0),
            self.betweenness.get(&node_id).copied().unwrap_or(0.0),
            self.closeness.get(&node_id).copied().unwrap_or(0.0),
            self.eigenvector.get(&node_id).copied().unwrap_or(0.0),
            self.pagerank.get(&node_id).copied().unwrap_or(0.0),
        ];

        scores.iter().sum::<f32>() / scores.len() as f32
    }

    /// Get top N most central nodes across all measures
    pub fn top_combined(&self, n: usize) -> Vec<(NodeId, f32)> {
        let mut combined: HashMap<NodeId, f32> = HashMap::new();

        // Collect all node IDs
        let all_nodes: HashSet<NodeId> = self
            .degree
            .keys()
            .chain(self.betweenness.keys())
            .chain(self.closeness.keys())
            .chain(self.eigenvector.keys())
            .chain(self.pagerank.keys())
            .copied()
            .collect();

        // Calculate combined score for each node
        for &node_id in &all_nodes {
            combined.insert(node_id, self.combined_score(node_id));
        }

        CentralityMeasures::top_nodes(&combined, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    #[test]
    fn test_degree_centrality() {
        let mut graph = Graph::new();

        // Create star graph: node 1 connected to all others
        let n1 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 1,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        let mut other_nodes = Vec::new();
        for i in 2..=5 {
            let node = graph
                .add_node(GraphNode::new(NodeType::Event {
                    event_id: i,
                    event_type: "test".to_string(),
                    significance: 0.5,
                }))
                .unwrap();
            other_nodes.push(node);

            graph.add_edge(GraphEdge::new(
                n1,
                node,
                EdgeType::Temporal {
                    average_interval_ms: 100,
                    sequence_confidence: 0.9,
                },
                0.9,
            ));
        }

        let centrality_calc = CentralityMeasures::new();
        let centrality = centrality_calc.degree_centrality(&graph).unwrap();

        // Node 1 should have highest degree centrality (connected to 4 others)
        assert!(centrality[&n1] > centrality[&other_nodes[0]]);
        assert!(centrality[&n1] > 0.5);
    }

    #[test]
    fn test_brandes_betweenness_line_graph() {
        // Line graph: A -> B -> C -> D
        // B and C are the only intermediate nodes — they should have highest betweenness.
        let mut graph = Graph::new();
        let a = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        let b = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 2, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        let c = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 3, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        let d = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 4, event_type: "t".into(), significance: 0.5,
        })).unwrap();

        let edge = |s, t| GraphEdge::new(s, t, EdgeType::Causality { strength: 1.0, lag_ms: 10 }, 1.0);
        graph.add_edge(edge(a, b));
        graph.add_edge(edge(b, c));
        graph.add_edge(edge(c, d));

        let calc = CentralityMeasures::new();
        let bc = calc.betweenness_centrality(&graph).unwrap();

        // A and D are endpoints — zero betweenness
        assert_eq!(bc[&a], 0.0);
        assert_eq!(bc[&d], 0.0);

        // B and C are intermediaries — positive betweenness
        assert!(bc[&b] > 0.0, "B should have positive betweenness, got {}", bc[&b]);
        assert!(bc[&c] > 0.0, "C should have positive betweenness, got {}", bc[&c]);
    }

    #[test]
    fn test_brandes_betweenness_star_graph() {
        // Star: center connected to 4 leaves (bidirectional edges).
        // Center sits on ALL shortest paths between leaves.
        let mut graph = Graph::new();
        let center = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 100, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        let mut leaves = Vec::new();
        for i in 1..=4 {
            let leaf = graph.add_node(GraphNode::new(NodeType::Event {
                event_id: i, event_type: "t".into(), significance: 0.5,
            })).unwrap();
            leaves.push(leaf);
            let edge = |s, t| GraphEdge::new(s, t, EdgeType::Causality { strength: 1.0, lag_ms: 10 }, 1.0);
            graph.add_edge(edge(center, leaf));
            graph.add_edge(edge(leaf, center));
        }

        let calc = CentralityMeasures::new();
        let bc = calc.betweenness_centrality(&graph).unwrap();

        // Center should have highest betweenness
        for leaf in &leaves {
            assert!(
                bc[&center] > bc[leaf],
                "Center betweenness {} should exceed leaf {} betweenness {}",
                bc[&center], leaf, bc[leaf]
            );
        }

        // Leaves have zero betweenness (no paths go through them)
        for leaf in &leaves {
            assert_eq!(bc[leaf], 0.0, "Leaf {} should have zero betweenness", leaf);
        }
    }

    #[test]
    fn test_brandes_betweenness_empty_and_small() {
        let calc = CentralityMeasures::new();

        // Empty graph
        let graph = Graph::new();
        let bc = calc.betweenness_centrality(&graph).unwrap();
        assert!(bc.is_empty());

        // Single node
        let mut graph = Graph::new();
        let _n = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        let bc = calc.betweenness_centrality(&graph).unwrap();
        assert_eq!(bc.len(), 1);
        assert_eq!(*bc.values().next().unwrap(), 0.0);

        // Two nodes
        let mut graph = Graph::new();
        let n1 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        let n2 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 2, event_type: "t".into(), significance: 0.5,
        })).unwrap();
        graph.add_edge(GraphEdge::new(n1, n2, EdgeType::Causality { strength: 1.0, lag_ms: 10 }, 1.0));
        let bc = calc.betweenness_centrality(&graph).unwrap();
        assert_eq!(bc[&n1], 0.0);
        assert_eq!(bc[&n2], 0.0);
    }

    #[test]
    fn test_pagerank() {
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

        // n1 -> n2, n1 -> n3, n2 -> n3
        // n3 should have highest PageRank (receives links from both)
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
            n1,
            n3,
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

        let centrality_calc = CentralityMeasures::new();
        let pagerank = centrality_calc.pagerank(&graph, 0.85, 100, 0.0001).unwrap();

        // n3 should have highest PageRank
        assert!(pagerank[&n3] > pagerank[&n1]);
        assert!(pagerank[&n3] > pagerank[&n2]);
    }
}
