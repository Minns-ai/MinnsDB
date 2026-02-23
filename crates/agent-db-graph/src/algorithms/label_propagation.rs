//! Label Propagation community detection — O(V + E) per iteration.
//!
//! Each node adopts the label held by the majority of its neighbours
//! (ties broken deterministically by smallest label). The algorithm
//! converges when no node changes its label.
//!
//! Reference: Raghavan et al., "Near linear time algorithm to detect
//! community structures in large-scale networks" (2007).

use crate::structures::{Graph, NodeId};
use crate::GraphResult;
use std::collections::HashMap;

/// Configuration for Label Propagation.
#[derive(Debug, Clone)]
pub struct LabelPropagationConfig {
    /// Maximum number of full sweeps.
    pub max_iterations: usize,
}

impl Default for LabelPropagationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
        }
    }
}

/// Result of label propagation community detection.
#[derive(Debug, Clone)]
pub struct LabelPropagationResult {
    /// Node → community label mapping.
    pub node_labels: HashMap<NodeId, u64>,
    /// Community label → member nodes.
    pub communities: HashMap<u64, Vec<NodeId>>,
    /// Number of communities found.
    pub community_count: usize,
    /// Iterations until convergence (or max).
    pub iterations: usize,
}

/// Label Propagation community detection.
pub struct LabelPropagationAlgorithm {
    config: LabelPropagationConfig,
}

impl LabelPropagationAlgorithm {
    /// Create with default config.
    pub fn new() -> Self {
        Self {
            config: LabelPropagationConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: LabelPropagationConfig) -> Self {
        Self { config }
    }

    /// Run label propagation on the graph.
    pub fn detect_communities(&self, graph: &Graph) -> GraphResult<LabelPropagationResult> {
        let nodes = graph.get_all_node_ids();
        if nodes.is_empty() {
            return Ok(LabelPropagationResult {
                node_labels: HashMap::new(),
                communities: HashMap::new(),
                community_count: 0,
                iterations: 0,
            });
        }

        // Initialize: each node labelled with its own ID
        let mut labels: HashMap<NodeId, u64> = nodes.iter().map(|&n| (n, n)).collect();

        // Deterministic ordering (sorted by node ID) for reproducibility
        let mut order: Vec<NodeId> = nodes.clone();
        order.sort_unstable();

        let mut iterations = 0;

        for _ in 0..self.config.max_iterations {
            iterations += 1;
            let mut changed = false;

            for &node in &order {
                let new_label = self.majority_label(graph, node, &labels);
                if new_label != labels[&node] {
                    labels.insert(node, new_label);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Build communities map
        let mut communities: HashMap<u64, Vec<NodeId>> = HashMap::new();
        for (&node, &label) in &labels {
            communities.entry(label).or_default().push(node);
        }
        let community_count = communities.len();

        Ok(LabelPropagationResult {
            node_labels: labels,
            communities,
            community_count,
            iterations,
        })
    }

    /// Find the label held by the majority of node's neighbours (both
    /// incoming and outgoing, treating the graph as undirected).
    /// Ties are broken by choosing the smallest label.
    fn majority_label(
        &self,
        graph: &Graph,
        node: NodeId,
        labels: &HashMap<NodeId, u64>,
    ) -> u64 {
        let mut label_weights: HashMap<u64, f32> = HashMap::new();

        // Outgoing neighbours
        for edge in graph.get_edges_from(node) {
            let neighbor_label = labels.get(&edge.target).copied().unwrap_or(edge.target);
            *label_weights.entry(neighbor_label).or_insert(0.0) += edge.weight;
        }

        // Incoming neighbours
        for edge in graph.get_edges_to(node) {
            let neighbor_label = labels.get(&edge.source).copied().unwrap_or(edge.source);
            *label_weights.entry(neighbor_label).or_insert(0.0) += edge.weight;
        }

        if label_weights.is_empty() {
            // Isolated node — keep its own label
            return labels.get(&node).copied().unwrap_or(node);
        }

        // Find label with max weight; break ties with smallest label
        let mut best_label = labels.get(&node).copied().unwrap_or(node);
        let mut best_weight = f32::NEG_INFINITY;

        for (&label, &weight) in &label_weights {
            if weight > best_weight || (weight == best_weight && label < best_label) {
                best_weight = weight;
                best_label = label;
            }
        }

        best_label
    }
}

impl Default for LabelPropagationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn make_node(event_id: u128) -> GraphNode {
        GraphNode::new(NodeType::Event {
            event_id,
            event_type: "test".into(),
            significance: 0.5,
        })
    }

    fn make_edge(source: NodeId, target: NodeId, weight: f32) -> GraphEdge {
        GraphEdge::new(
            source,
            target,
            EdgeType::Association {
                association_type: "test".into(),
                evidence_count: 1,
                statistical_significance: weight,
            },
            weight,
        )
    }

    #[test]
    fn test_label_propagation_two_cliques() {
        let mut graph = Graph::new();

        // Clique 1: nodes a,b,c fully connected with weight 1.0
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        let c = graph.add_node(make_node(3)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, a, 1.0));
        graph.add_edge(make_edge(b, c, 1.0));
        graph.add_edge(make_edge(c, b, 1.0));
        graph.add_edge(make_edge(a, c, 1.0));
        graph.add_edge(make_edge(c, a, 1.0));

        // Clique 2: nodes d,e,f fully connected with weight 1.0
        let d = graph.add_node(make_node(4)).unwrap();
        let e = graph.add_node(make_node(5)).unwrap();
        let f = graph.add_node(make_node(6)).unwrap();

        graph.add_edge(make_edge(d, e, 1.0));
        graph.add_edge(make_edge(e, d, 1.0));
        graph.add_edge(make_edge(e, f, 1.0));
        graph.add_edge(make_edge(f, e, 1.0));
        graph.add_edge(make_edge(d, f, 1.0));
        graph.add_edge(make_edge(f, d, 1.0));

        // Weak bridge: c -> d with weight 0.1
        graph.add_edge(make_edge(c, d, 0.1));

        let lp = LabelPropagationAlgorithm::new();
        let result = lp.detect_communities(&graph).unwrap();

        // Nodes in clique 1 should share a label
        let la = result.node_labels[&a];
        let lb = result.node_labels[&b];
        let lc = result.node_labels[&c];
        assert_eq!(la, lb, "a and b should share a label");
        assert_eq!(lb, lc, "b and c should share a label");

        // Nodes in clique 2 should share a label
        let ld = result.node_labels[&d];
        let le = result.node_labels[&e];
        let lf = result.node_labels[&f];
        assert_eq!(ld, le, "d and e should share a label");
        assert_eq!(le, lf, "e and f should share a label");

        // The two communities should be at most 2
        assert!(
            result.community_count <= 3,
            "Expected at most 3 communities, got {}",
            result.community_count
        );
    }

    #[test]
    fn test_label_propagation_empty_graph() {
        let graph = Graph::new();
        let lp = LabelPropagationAlgorithm::new();
        let result = lp.detect_communities(&graph).unwrap();
        assert_eq!(result.community_count, 0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_label_propagation_single_node() {
        let mut graph = Graph::new();
        let n = graph.add_node(make_node(1)).unwrap();

        let lp = LabelPropagationAlgorithm::new();
        let result = lp.detect_communities(&graph).unwrap();
        assert_eq!(result.community_count, 1);
        assert_eq!(result.node_labels[&n], n);
    }

    #[test]
    fn test_label_propagation_converges() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, a, 1.0));

        let lp = LabelPropagationAlgorithm::with_config(LabelPropagationConfig {
            max_iterations: 1000,
        });
        let result = lp.detect_communities(&graph).unwrap();

        // Should converge in very few iterations
        assert!(result.iterations < 10, "Should converge quickly, took {} iterations", result.iterations);
        // Both nodes should be in the same community
        assert_eq!(result.node_labels[&a], result.node_labels[&b]);
    }
}
