//! Query validator and repairer.
//!
//! Validates built queries against graph state and attempts repairs
//! when possible (e.g., node not found → BM25 fuzzy search).

use crate::structures::Graph;
use crate::traversal::GraphQuery;

/// Maximum traversal depth allowed.
const MAX_DEPTH: u32 = 10;

/// Maximum K for Yen's k-shortest paths.
const MAX_K: usize = 10;

/// Result of query validation.
#[derive(Debug)]
pub enum ValidationResult {
    /// Query is valid as-is.
    Valid(GraphQuery),
    /// Query was repaired (original kept for logging).
    Repaired {
        original: GraphQuery,
        repaired: GraphQuery,
        reason: String,
    },
    /// Query is invalid and cannot be repaired.
    Invalid { query: GraphQuery, reason: String },
}

/// Validate a GraphQuery against the current graph state.
pub fn validate(query: GraphQuery, graph: &Graph) -> ValidationResult {
    // First pass: extract info we need while borrowing
    let check = analyze(&query, graph);

    match check {
        Check::Valid => ValidationResult::Valid(query),
        Check::Invalid(reason) => ValidationResult::Invalid { query, reason },
        Check::Repair(repaired, reason) => ValidationResult::Repaired {
            original: query,
            repaired,
            reason,
        },
    }
}

enum Check {
    Valid,
    Invalid(String),
    Repair(GraphQuery, String),
}

fn analyze(query: &GraphQuery, graph: &Graph) -> Check {
    match query {
        GraphQuery::ShortestPath { start, end }
        | GraphQuery::AStarPath { start, end }
        | GraphQuery::BidirectionalPath { start, end } => {
            if start == end {
                return Check::Invalid("Start and end nodes are the same".to_string());
            }
            if let Some(reason) = check_nodes_exist(graph, &[*start, *end]) {
                return Check::Invalid(reason);
            }
        },

        GraphQuery::KShortestPaths { start, end, k } => {
            if start == end {
                return Check::Invalid("Start and end nodes are the same".to_string());
            }
            if let Some(reason) = check_nodes_exist(graph, &[*start, *end]) {
                return Check::Invalid(reason);
            }
            if *k > MAX_K {
                return Check::Repair(
                    GraphQuery::KShortestPaths {
                        start: *start,
                        end: *end,
                        k: MAX_K,
                    },
                    format!("K clamped from {} to {}", k, MAX_K),
                );
            }
        },

        GraphQuery::NeighborsWithinDistance {
            start,
            max_distance,
        } => {
            if let Some(reason) = check_nodes_exist(graph, &[*start]) {
                return Check::Invalid(reason);
            }
            if *max_distance > MAX_DEPTH {
                return Check::Repair(
                    GraphQuery::NeighborsWithinDistance {
                        start: *start,
                        max_distance: MAX_DEPTH,
                    },
                    format!("Depth clamped from {} to {}", max_distance, MAX_DEPTH),
                );
            }
        },

        GraphQuery::Subgraph { center, radius, .. } => {
            if let Some(reason) = check_nodes_exist(graph, &[*center]) {
                return Check::Invalid(reason);
            }
            if *radius > MAX_DEPTH {
                return Check::Repair(
                    GraphQuery::Subgraph {
                        center: *center,
                        radius: MAX_DEPTH,
                        node_types: None,
                    },
                    format!("Radius clamped from {} to {}", radius, MAX_DEPTH),
                );
            }
        },

        GraphQuery::NearestByCost { start, .. }
        | GraphQuery::DeepReachability { start, .. }
        | GraphQuery::DirectedTraversal { start, .. } => {
            if let Some(reason) = check_nodes_exist(graph, &[*start]) {
                return Check::Invalid(reason);
            }
        },

        GraphQuery::RecursiveTraversal(req) => {
            if let Some(reason) = check_nodes_exist(graph, &[req.start]) {
                return Check::Invalid(reason);
            }
        },

        // Global queries don't need node validation
        GraphQuery::PageRank { .. }
        | GraphQuery::CommunityDetection { .. }
        | GraphQuery::StronglyConnectedComponents
        | GraphQuery::NodesByType(_)
        | GraphQuery::NodesByProperty { .. }
        | GraphQuery::EdgesByType { .. }
        | GraphQuery::PathQuery { .. } => {},
    }

    Check::Valid
}

/// Check that all given node IDs exist in the graph.
fn check_nodes_exist(graph: &Graph, node_ids: &[u64]) -> Option<String> {
    let missing: Vec<u64> = node_ids
        .iter()
        .filter(|id| !graph.nodes.contains_key(id))
        .copied()
        .collect();
    if missing.is_empty() {
        None
    } else {
        Some(format!("Node(s) not found: {:?}", missing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{GraphNode, NodeType};

    fn test_graph_with_nodes() -> Graph {
        let mut graph = Graph::new();
        graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Alice".to_string(),
                concept_type: crate::structures::ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();
        graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Bob".to_string(),
                concept_type: crate::structures::ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();
        graph
    }

    #[test]
    fn test_validator_valid_query() {
        let graph = test_graph_with_nodes();
        let query = GraphQuery::ShortestPath { start: 1, end: 2 };
        let result = validate(query, &graph);
        assert!(matches!(result, ValidationResult::Valid(_)));
    }

    #[test]
    fn test_validator_missing_node() {
        let graph = test_graph_with_nodes();
        let query = GraphQuery::ShortestPath { start: 1, end: 999 };
        let result = validate(query, &graph);
        assert!(matches!(result, ValidationResult::Invalid { .. }));
    }

    #[test]
    fn test_validator_self_loop() {
        let graph = test_graph_with_nodes();
        let query = GraphQuery::ShortestPath { start: 1, end: 1 };
        let result = validate(query, &graph);
        assert!(matches!(result, ValidationResult::Invalid { .. }));
    }

    #[test]
    fn test_validator_depth_clamp() {
        let graph = test_graph_with_nodes();
        let query = GraphQuery::NeighborsWithinDistance {
            start: 1,
            max_distance: 50,
        };
        let result = validate(query, &graph);
        match result {
            ValidationResult::Repaired { repaired, .. } => {
                assert!(matches!(
                    repaired,
                    GraphQuery::NeighborsWithinDistance {
                        max_distance: 10,
                        ..
                    }
                ));
            },
            other => panic!("Expected Repaired, got {:?}", other),
        }
    }

    #[test]
    fn test_validator_pagerank_valid() {
        let graph = test_graph_with_nodes();
        let query = GraphQuery::PageRank {
            iterations: 20,
            damping_factor: 0.85,
        };
        let result = validate(query, &graph);
        assert!(matches!(result, ValidationResult::Valid(_)));
    }
}
