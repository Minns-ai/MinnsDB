//! Unified NLQ response formatting.
//!
//! Formats results from multiple retrieval sources (BM25, memory, claims,
//! graph entities) into a single human-readable answer.

use crate::memory::Memory;
use crate::structures::Graph;

/// Format unified retrieval results into a human-readable answer.
///
/// Combines ranked node results with memory matches into sections.
pub fn format_unified_results(
    fused: &[(u64, f32)],
    _question: &str,
    graph: &Graph,
    memories: &[Memory],
) -> String {
    if fused.is_empty() && memories.is_empty() {
        return "No results found.".to_string();
    }

    let mut parts = Vec::new();

    // Section 1: Ranked graph results
    if !fused.is_empty() {
        let limit = fused.len().min(10);
        let mut section = format!("Found {} relevant results:", fused.len());
        for &(node_id, score) in fused.iter().take(limit) {
            let label = graph
                .get_node(node_id)
                .map(|n| n.label())
                .unwrap_or_else(|| "(unknown)".to_string());
            section.push_str(&format!("\n  - {} (score: {:.3})", label, score));
        }
        if fused.len() > limit {
            section.push_str(&format!("\n  ... and {} more", fused.len() - limit));
        }
        parts.push(section);
    }

    // Section 2: Related memories
    if !memories.is_empty() {
        let limit = memories.len().min(5);
        let mut section = String::from("Related memories:");
        for mem in memories.iter().take(limit) {
            let summary = if mem.summary.len() > 120 {
                format!("{}...", &mem.summary[..120])
            } else {
                mem.summary.clone()
            };
            section.push_str(&format!("\n  - {}", summary));
        }
        parts.push(section);
    }

    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeOutcome;
    use crate::structures::{ConceptType, GraphNode, NodeType};

    /// Create a minimal test Memory with only the summary populated.
    fn test_memory(id: u64, summary: &str) -> Memory {
        Memory {
            id,
            agent_id: 0,
            session_id: 0,
            episode_id: 0,
            summary: summary.to_string(),
            takeaway: String::new(),
            causal_note: String::new(),
            summary_embedding: vec![],
            tier: crate::memory::MemoryTier::Semantic,
            consolidated_from: vec![],
            schema_id: None,
            consolidation_status: crate::memory::ConsolidationStatus::Active,
            context: Default::default(),
            key_events: vec![],
            strength: 1.0,
            relevance_score: 1.0,
            formed_at: 0,
            last_accessed: 0,
            access_count: 0,
            outcome: EpisodeOutcome::Partial,
            memory_type: crate::memory::MemoryType::Episodic { significance: 1.0 },
            metadata: Default::default(),
            expires_at: None,
        }
    }

    #[test]
    fn test_format_empty() {
        let graph = Graph::new();
        let result = format_unified_results(&[], "test?", &graph, &[]);
        assert_eq!(result, "No results found.");
    }

    #[test]
    fn test_format_with_nodes() {
        let mut graph = Graph::new();
        let id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Alice".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        let fused = vec![(id, 0.95)];
        let result = format_unified_results(&fused, "tell me about Alice", &graph, &[]);
        assert!(result.contains("Found 1 relevant results:"));
        assert!(result.contains("Alice"));
        assert!(result.contains("0.950"));
    }

    #[test]
    fn test_format_with_memories() {
        let graph = Graph::new();
        let mem = test_memory(1, "Alice prefers coffee over tea");
        let result = format_unified_results(&[], "preferences?", &graph, &[mem]);
        assert!(result.contains("Related memories:"));
        assert!(result.contains("Alice prefers coffee"));
    }

    #[test]
    fn test_format_mixed_sources() {
        let mut graph = Graph::new();
        let id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Bob".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        let fused = vec![(id, 0.8)];
        let mem = test_memory(1, "Bob lives in NYC");
        let result = format_unified_results(&fused, "about Bob", &graph, &[mem]);
        assert!(result.contains("Found 1 relevant results:"));
        assert!(result.contains("Related memories:"));
        assert!(result.contains("Bob"));
    }
}
