//! Result formatter — converts QueryResult into human-readable answers.

use super::entity::ResolvedEntity;
use super::intent::QueryIntent;
use crate::structures::{Graph, NodeId};
use crate::traversal::QueryResult;

/// Format a QueryResult into a human-readable answer string.
pub fn format_result(
    _question: &str,
    _intent: &QueryIntent,
    result: &QueryResult,
    _resolved: &[ResolvedEntity],
    graph: &Graph,
) -> String {
    match result {
        QueryResult::Nodes(ids) => format_nodes(ids, graph),
        QueryResult::Path(ids) => format_path(ids, graph),
        QueryResult::WeightedPaths(paths) => format_weighted_paths(paths, graph),
        QueryResult::Paths(paths) => format_paths(paths, graph),
        QueryResult::Edges(ids) => {
            format!("Found {} edges: {:?}", ids.len(), &ids[..ids.len().min(10)])
        },
        QueryResult::Subgraph {
            nodes,
            edges,
            center,
        } => {
            let center_label = node_label(*center, graph);
            format!(
                "Subgraph around {}: {} nodes, {} edges",
                center_label,
                nodes.len(),
                edges.len()
            )
        },
        QueryResult::Rankings(rankings) => format_rankings(rankings, graph),
        QueryResult::Communities(communities) => {
            format!(
                "Found {} communities (largest: {} nodes)",
                communities.len(),
                communities.iter().map(|c| c.len()).max().unwrap_or(0)
            )
        },
        QueryResult::Properties(props) => {
            // If the handler set a human-readable "answer" field, use it directly
            if let Some(answer) = props.get("answer").and_then(|v| v.as_str()) {
                answer.to_string()
            } else {
                format!(
                    "Properties: {}",
                    serde_json::to_string_pretty(props).unwrap_or_default()
                )
            }
        },
    }
}

fn format_nodes(ids: &[NodeId], graph: &Graph) -> String {
    if ids.is_empty() {
        return "No nodes found.".to_string();
    }
    let count = ids.len();
    let labels: Vec<String> = ids
        .iter()
        .take(10)
        .map(|&id| {
            let label = node_label(id, graph);
            let type_name = graph.get_node(id).map(|n| n.type_name()).unwrap_or("?");
            format!("{} ({})", label, type_name)
        })
        .collect();
    let suffix = if count > 10 {
        format!(" ... and {} more", count - 10)
    } else {
        String::new()
    };
    format!("Found {} nodes: {}{}", count, labels.join(", "), suffix)
}

fn format_path(ids: &[NodeId], graph: &Graph) -> String {
    if ids.is_empty() {
        return "No path found.".to_string();
    }
    let labels: Vec<String> = ids.iter().map(|&id| node_label(id, graph)).collect();
    format!("Path ({} hops): {}", ids.len() - 1, labels.join(" -> "))
}

fn format_weighted_paths(paths: &[(Vec<NodeId>, f32)], graph: &Graph) -> String {
    if paths.is_empty() {
        return "No paths found.".to_string();
    }
    let mut lines = Vec::new();
    for (i, (ids, cost)) in paths.iter().enumerate().take(5) {
        let labels: Vec<String> = ids.iter().map(|&id| node_label(id, graph)).collect();
        lines.push(format!(
            "  {}. {} (cost: {:.2})",
            i + 1,
            labels.join(" -> "),
            cost
        ));
    }
    let suffix = if paths.len() > 5 {
        format!("\n  ... and {} more paths", paths.len() - 5)
    } else {
        String::new()
    };
    format!(
        "{} paths found:\n{}{}",
        paths.len(),
        lines.join("\n"),
        suffix
    )
}

fn format_paths(paths: &[Vec<NodeId>], graph: &Graph) -> String {
    if paths.is_empty() {
        return "No paths found.".to_string();
    }
    let mut lines = Vec::new();
    for (i, ids) in paths.iter().enumerate().take(5) {
        let labels: Vec<String> = ids.iter().map(|&id| node_label(id, graph)).collect();
        lines.push(format!("  {}. {}", i + 1, labels.join(" -> ")));
    }
    format!("{} paths found:\n{}", paths.len(), lines.join("\n"))
}

fn format_rankings(rankings: &[(NodeId, f32)], graph: &Graph) -> String {
    if rankings.is_empty() {
        return "No rankings computed.".to_string();
    }
    let count = rankings.len();
    let lines: Vec<String> = rankings
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, &(id, score))| {
            let label = node_label(id, graph);
            let type_name = graph.get_node(id).map(|n| n.type_name()).unwrap_or("?");
            format!(
                "  {}. {} ({}) — score: {:.4}",
                i + 1,
                label,
                type_name,
                score
            )
        })
        .collect();
    let suffix = if count > 10 {
        format!("\n  ... and {} more", count - 10)
    } else {
        String::new()
    };
    format!(
        "Top {} of {} ranked nodes:\n{}{}",
        lines.len(),
        count,
        lines.join("\n"),
        suffix
    )
}

/// Get a human-readable label for a node, falling back to its ID.
fn node_label(id: NodeId, graph: &Graph) -> String {
    graph
        .get_node(id)
        .map(|n| n.label())
        .unwrap_or_else(|| format!("#{}", id))
}

/// Count the result items in a QueryResult.
pub fn result_count(result: &QueryResult) -> usize {
    match result {
        QueryResult::Nodes(ids) => ids.len(),
        QueryResult::Path(ids) => ids.len(),
        QueryResult::WeightedPaths(paths) => paths.len(),
        QueryResult::Paths(paths) => paths.len(),
        QueryResult::Edges(ids) => ids.len(),
        QueryResult::Subgraph { nodes, .. } => nodes.len(),
        QueryResult::Rankings(rankings) => rankings.len(),
        QueryResult::Communities(c) => c.len(),
        QueryResult::Properties(p) => p.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{GraphNode, NodeType};

    fn test_graph() -> Graph {
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
    fn test_format_path_result() {
        let graph = test_graph();
        let result = QueryResult::Path(vec![1, 2]);
        let formatted = format_result("", &QueryIntent::Unknown, &result, &[], &graph);
        assert!(formatted.contains("Alice"));
        assert!(formatted.contains("Bob"));
        assert!(formatted.contains("->"));
    }

    #[test]
    fn test_format_nodes_result() {
        let graph = test_graph();
        let result = QueryResult::Nodes(vec![1, 2]);
        let formatted = format_result("", &QueryIntent::Unknown, &result, &[], &graph);
        assert!(formatted.contains("Found 2 nodes"));
        assert!(formatted.contains("Alice"));
    }

    #[test]
    fn test_format_rankings_result() {
        let graph = test_graph();
        let result = QueryResult::Rankings(vec![(1, 0.95), (2, 0.87)]);
        let formatted = format_result("", &QueryIntent::Unknown, &result, &[], &graph);
        assert!(formatted.contains("Alice"));
        assert!(formatted.contains("0.95"));
    }

    #[test]
    fn test_format_empty_results() {
        let graph = test_graph();
        let result = QueryResult::Nodes(vec![]);
        let formatted = format_result("", &QueryIntent::Unknown, &result, &[], &graph);
        assert!(formatted.contains("No nodes found"));
    }
}
