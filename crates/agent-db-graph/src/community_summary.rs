//! Community summaries via LLM-generated descriptions.
//!
//! After Louvain/LP community detection, each community's nodes and edges
//! are summarized into a `CommunitySummary` using an LLM call.

use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::structures::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;

/// LLM-generated summary for a detected graph community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunitySummary {
    pub community_id: u64,
    pub summary: String,
    pub key_entities: Vec<String>,
    pub node_count: usize,
    pub generated_at: u64,
    pub token_estimate: u32,
}

/// Configuration for community summary generation.
#[derive(Debug, Clone)]
pub struct CommunitySummaryConfig {
    /// Minimum community size to generate a summary for.
    pub min_community_size: usize,
    /// Maximum number of communities to summarize.
    pub max_communities_to_summarize: usize,
    /// Maximum entity labels to include in the LLM prompt.
    pub max_entities_per_prompt: usize,
    /// LLM temperature.
    pub temperature: f32,
    /// Maximum tokens for the LLM response.
    pub max_tokens: u32,
}

impl Default for CommunitySummaryConfig {
    fn default() -> Self {
        Self {
            min_community_size: 3,
            max_communities_to_summarize: 50,
            max_entities_per_prompt: 20,
            temperature: 0.3,
            max_tokens: 256,
        }
    }
}

const SYSTEM_PROMPT: &str = concat!(
    "Summarize this graph community. The community consists of related nodes and their connections.\n",
    "Output strict JSON with exactly two fields:\n",
    "- \"summary\": a 1-3 sentence description of what this community represents\n",
    "- \"key_entities\": an array of the most important entity names (max 5)\n",
    "No markdown fences, no explanation, no other fields.",
);

/// Parsed LLM response for community summary.
#[derive(Debug, Deserialize)]
struct SummaryResponse {
    summary: String,
    key_entities: Vec<String>,
}

/// Build the user prompt describing a community's nodes and edges.
pub fn build_community_prompt(
    node_labels: &[String],
    edge_descriptions: &[String],
    max_entities: usize,
) -> String {
    let mut prompt = String::with_capacity(1024);
    prompt.push_str("Community nodes:\n");
    for (i, label) in node_labels.iter().take(max_entities).enumerate() {
        let _ = writeln!(prompt, "{}. {}", i + 1, label);
    }
    if node_labels.len() > max_entities {
        let _ = writeln!(prompt, "... and {} more", node_labels.len() - max_entities);
    }
    if !edge_descriptions.is_empty() {
        prompt.push_str("\nRelationships:\n");
        for desc in edge_descriptions.iter().take(max_entities) {
            let _ = writeln!(prompt, "- {}", desc);
        }
    }
    prompt
}

/// Generate a summary for a single community using an LLM call.
///
/// Returns `None` if the LLM call fails or the response cannot be parsed (fail-open).
pub async fn generate_community_summary(
    client: &dyn LlmClient,
    community_id: u64,
    node_labels: &[String],
    edge_descriptions: &[String],
    config: &CommunitySummaryConfig,
) -> Option<CommunitySummary> {
    let user_prompt = build_community_prompt(
        node_labels,
        edge_descriptions,
        config.max_entities_per_prompt,
    );

    let request = LlmRequest {
        system_prompt: SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        json_mode: true,
    };

    let response =
        tokio::time::timeout(std::time::Duration::from_secs(15), client.complete(request))
            .await
            .ok()?
            .ok()?;

    let value = parse_json_from_llm(&response.content)?;
    let parsed: SummaryResponse = serde_json::from_value(value).ok()?;

    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    Some(CommunitySummary {
        community_id,
        summary: parsed.summary,
        key_entities: parsed.key_entities,
        node_count: node_labels.len(),
        generated_at: now_nanos,
        token_estimate: response.tokens_used,
    })
}

/// Extract community data from a graph: maps community_id → list of (node_id, label).
///
/// Returns owned data so the graph lock can be released before LLM calls.
pub fn extract_community_data(graph: &Graph) -> HashMap<u64, Vec<(u64, String)>> {
    let mut communities: HashMap<u64, Vec<(u64, String)>> = HashMap::new();
    for (&node_id, node) in &graph.nodes {
        if let Some(cid) = node.properties.get("community_id").and_then(|v| v.as_u64()) {
            let label = node.label().to_string();
            communities.entry(cid).or_default().push((node_id, label));
        }
    }
    communities
}

/// Extract edge descriptions for nodes within a community.
pub fn extract_edge_descriptions(graph: &Graph, node_ids: &[(u64, String)]) -> Vec<String> {
    // Build HashMap for O(1) label lookups instead of O(n) linear search
    let label_map: HashMap<u64, &str> = node_ids.iter().map(|(id, l)| (*id, l.as_str())).collect();
    let mut descriptions = Vec::new();
    for edge in graph.edges.values() {
        if let (Some(src_label), Some(tgt_label)) =
            (label_map.get(&edge.source), label_map.get(&edge.target))
        {
            let edge_type = format!("{:?}", edge.edge_type);
            descriptions.push(format!("{} --[{}]--> {}", src_label, edge_type, tgt_label));
            if descriptions.len() >= 30 {
                break;
            }
        }
    }
    descriptions
}

/// Generate summaries for all communities above the minimum size.
///
/// Communities are sorted by size (descending) and capped at `max_communities_to_summarize`.
pub async fn generate_all_summaries(
    client: &dyn LlmClient,
    graph: &Graph,
    config: &CommunitySummaryConfig,
) -> HashMap<u64, CommunitySummary> {
    let communities = extract_community_data(graph);

    // Sort by size descending, filter by min size, cap at max
    let mut sorted: Vec<(u64, Vec<(u64, String)>)> = communities
        .into_iter()
        .filter(|(_, members)| members.len() >= config.min_community_size)
        .collect();
    sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    sorted.truncate(config.max_communities_to_summarize);

    // Extract edge descriptions before releasing graph reference
    let community_data: Vec<(u64, Vec<String>, Vec<String>)> = sorted
        .iter()
        .map(|(cid, members)| {
            let labels: Vec<String> = members.iter().map(|(_, l)| l.clone()).collect();
            let edges = extract_edge_descriptions(graph, members);
            (*cid, labels, edges)
        })
        .collect();

    let mut result = HashMap::new();
    for (cid, labels, edges) in &community_data {
        if let Some(summary) = generate_community_summary(client, *cid, labels, edges, config).await
        {
            result.insert(*cid, summary);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_community_prompt_basic() {
        let labels = vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ];
        let edges = vec!["Alice --[knows]--> Bob".to_string()];
        let prompt = build_community_prompt(&labels, &edges, 20);
        assert!(prompt.contains("Alice"));
        assert!(prompt.contains("Bob"));
        assert!(prompt.contains("Charlie"));
        assert!(prompt.contains("knows"));
    }

    #[test]
    fn test_build_community_prompt_truncation() {
        let labels: Vec<String> = (0..30).map(|i| format!("Node{}", i)).collect();
        let prompt = build_community_prompt(&labels, &[], 5);
        assert!(prompt.contains("Node0"));
        assert!(prompt.contains("Node4"));
        assert!(prompt.contains("... and 25 more"));
        assert!(!prompt.contains("Node5"));
    }

    #[test]
    fn test_default_config() {
        let config = CommunitySummaryConfig::default();
        assert_eq!(config.min_community_size, 3);
        assert_eq!(config.max_communities_to_summarize, 50);
        assert_eq!(config.max_entities_per_prompt, 20);
    }

    #[test]
    fn test_extract_community_data_empty_graph() {
        let graph = Graph::new();
        let data = extract_community_data(&graph);
        assert!(data.is_empty());
    }

    #[test]
    fn test_extract_community_data_with_communities() {
        use crate::structures::{ConceptType, GraphNode, NodeType};
        let mut graph = Graph::new();

        let mut node1 = GraphNode::new(NodeType::Concept {
            concept_name: "Alice".to_string(),
            concept_type: ConceptType::Person,
            confidence: 1.0,
        });
        node1
            .properties
            .insert("community_id".to_string(), serde_json::json!(1));
        let _id1 = graph.add_node(node1).unwrap();

        let mut node2 = GraphNode::new(NodeType::Concept {
            concept_name: "Bob".to_string(),
            concept_type: ConceptType::Person,
            confidence: 1.0,
        });
        node2
            .properties
            .insert("community_id".to_string(), serde_json::json!(1));
        let _id2 = graph.add_node(node2).unwrap();

        let mut node3 = GraphNode::new(NodeType::Concept {
            concept_name: "Charlie".to_string(),
            concept_type: ConceptType::Person,
            confidence: 1.0,
        });
        node3
            .properties
            .insert("community_id".to_string(), serde_json::json!(2));
        let _id3 = graph.add_node(node3).unwrap();

        let data = extract_community_data(&graph);
        assert_eq!(data.len(), 2);
        assert_eq!(data[&1].len(), 2);
        assert_eq!(data[&2].len(), 1);
    }

    #[test]
    fn test_parse_summary_response() {
        let json = r#"{"summary": "A group of people", "key_entities": ["Alice", "Bob"]}"#;
        let value = parse_json_from_llm(json).unwrap();
        let parsed: SummaryResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.summary, "A group of people");
        assert_eq!(parsed.key_entities, vec!["Alice", "Bob"]);
    }

    #[test]
    fn test_parse_summary_response_empty_entities() {
        let json = r#"{"summary": "An isolated cluster", "key_entities": []}"#;
        let value = parse_json_from_llm(json).unwrap();
        let parsed: SummaryResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.summary, "An isolated cluster");
        assert!(parsed.key_entities.is_empty());
    }
}
