//! Unified NLQ response formatting.
//!
//! Formats results from multiple retrieval sources (BM25, memory, claims,
//! graph entities) into a single human-readable answer.

use crate::conversation::graph_projection;
use crate::memory::Memory;
use crate::structures::{Graph, NodeType};
use std::collections::HashSet;

/// Minimum RRF fused score to include a result.
/// With RRF k=60, a rank-1 result in a single source scores ~0.016.
/// Set low enough to include single-source top results.
const MIN_FUSED_SCORE: f32 = 0.005;

/// Check if text is operational noise that should be filtered from answers.
fn is_operational_noise(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("success rate:")
        || lower.contains("approach needs improvement")
        || lower.starts_with("for goal bucket")
        || lower.starts_with("effective approach for goal bucket")
        || lower.starts_with("session complete")
        || lower.starts_with("session start")
        || lower.starts_with("session_")
}

/// Format unified retrieval results into a human-readable answer.
///
/// Extracts actual content from graph nodes (claims, concepts, events) and
/// memory summaries instead of showing raw IDs. Filters low-score noise.
/// `superseded_targets` contains lowercase entity names that have been
/// temporally superseded — claims mentioning them are filtered out.
pub fn format_unified_results(
    fused: &[(u64, f32)],
    _question: &str,
    graph: &Graph,
    memories: &[Memory],
    superseded_targets: &std::collections::HashSet<String>,
) -> String {
    // Extract meaningful content from fused results
    let mut content_items: Vec<ContentItem> = Vec::new();
    let mut seen_texts: HashSet<String> = HashSet::new();

    // Collect entity state facts from graph edges — only for entities in fused results,
    // and only facts whose attribute/value are relevant to the question.
    let mut state_facts: Vec<String> = Vec::new();
    let mut seen_entities: HashSet<String> = HashSet::new();
    let question_lower = _question.to_lowercase();

    // Seed entities from fused results (NOT hardcoded "user")
    let mut seed_entities: Vec<String> = Vec::new();
    for &(node_id, score) in fused {
        if score < MIN_FUSED_SCORE {
            continue;
        }
        if let Some(node) = graph.get_node(node_id) {
            if let NodeType::Concept { concept_name, .. } = &node.node_type {
                seed_entities.push(concept_name.to_string());
            }
        }
    }

    // Also add "user" but only if the question references the user (first person)
    if question_lower.contains(" i ")
        || question_lower.starts_with("i ")
        || question_lower.contains(" my ")
        || question_lower.contains(" me ")
        || question_lower.starts_with("my ")
        || question_lower.starts_with("what do you")
        || question_lower.starts_with("where do")
        || question_lower.starts_with("where am")
        || question_lower.contains("should i")
    {
        seed_entities.push("user".to_string());
    }

    for entity_name in &seed_entities {
        if !seen_entities.insert(entity_name.to_lowercase()) {
            continue;
        }
        // Use successor-state projection for authoritative current state
        let projected = graph_projection::project_entity_state(graph, entity_name, u64::MAX, None);
        for slot in projected.slots.values() {
            let assoc = &slot.association_type;
            let value = slot.value.as_deref().unwrap_or(&slot.target_name);

            // Skip noisy facts
            let trimmed = value.trim();
            if trimmed.len() < 3
                || trimmed.len() > 80
                || trimmed.eq_ignore_ascii_case("it")
                || trimmed.eq_ignore_ascii_case("true")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("yes")
                || trimmed.eq_ignore_ascii_case("no")
            {
                continue;
            }

            let attr = assoc.split(':').nth(1).unwrap_or(assoc).to_lowercase();

            // Format state facts as natural language
            let text = if assoc.starts_with("state:") {
                format!("Current {}: {}", attr, value)
            } else if assoc.starts_with("preference:") {
                format!("{} prefers: {}", entity_name, value)
            } else if assoc.starts_with("relationship:") {
                let rel = assoc.strip_prefix("relationship:").unwrap_or(assoc);
                format!("{} {} {}", entity_name, rel, slot.target_name)
            } else {
                format!("{}: {}", assoc, value)
            };
            let key = text.to_lowercase();
            if seen_texts.insert(key) {
                state_facts.push(text);
            }

            // Cap state facts at 5 per query
            if state_facts.len() >= 5 {
                break;
            }
        }
    }

    // Build memory lookup for resolving Memory graph nodes → summaries
    let memory_by_id: std::collections::HashMap<u64, &Memory> =
        memories.iter().map(|m| (m.id, m)).collect();

    // NLQ returns claims and entity relationships from the graph.
    // Memory nodes are only included when memories are explicitly requested.
    for &(node_id, score) in fused {
        if score < MIN_FUSED_SCORE {
            continue;
        }

        let Some(node) = graph.get_node(node_id) else {
            continue;
        };

        let item = match &node.node_type {
            NodeType::Claim { claim_text, .. } => {
                // Skip claims that reference superseded entities (temporal filter)
                if !superseded_targets.is_empty() {
                    let text_lower = claim_text.to_lowercase();
                    let is_stale = superseded_targets
                        .iter()
                        .any(|t| t.len() >= 3 && text_lower.contains(t.as_str()));
                    if is_stale {
                        continue;
                    }
                }
                let text = truncate(claim_text, 150);
                ContentItem {
                    kind: "claim",
                    text,
                    score,
                }
            },
            NodeType::Concept {
                concept_name,
                concept_type,
                ..
            } => ContentItem {
                kind: "concept",
                text: format!("{} ({:?})", concept_name, concept_type),
                score,
            },
            // Memory nodes: resolve to summary text when memories are included
            NodeType::Memory { memory_id, .. } if !memories.is_empty() => {
                if let Some(mem) = memory_by_id.get(memory_id) {
                    ContentItem {
                        kind: "memory",
                        text: truncate(&mem.summary, 150),
                        score,
                    }
                } else {
                    continue;
                }
            },
            // Skip Strategy, Event, Goal, Result, and Memory when not requested
            _ => continue,
        };

        // Deduplicate by text content and filter operational noise
        if item.text.is_empty() || is_operational_noise(&item.text) {
            continue;
        }
        let dedup_key = item.text.trim_end_matches('.').to_lowercase();
        if seen_texts.insert(dedup_key) {
            content_items.push(item);
        }
    }

    // Build output
    if content_items.is_empty() && state_facts.is_empty() && memories.is_empty() {
        return "No relevant information found.".to_string();
    }

    let mut parts = Vec::new();

    // Entity state facts from graph edges (highest priority, clearly labeled as current)
    if !state_facts.is_empty() {
        let mut section = vec!["Current state:".to_string()];
        section.extend(state_facts.iter().take(5).map(|f| format!("- {}", f)));
        parts.push(section.join("\n"));
    }

    // State timeline: show progression of state changes with (superseded)/(CURRENT) labels
    // This gives the LLM explicit temporal context to prevent hallucination from old state.
    for entity_name in &seed_entities {
        if let Some(timeline) = graph_projection::build_entity_timeline_summary(graph, entity_name)
        {
            // Only include if there are superseded entries (i.e., state has changed over time)
            if timeline.contains("(superseded)") {
                parts.push(timeline);
            }
        }
    }

    // Claims and entity results (capped at 10 for focused context)
    if !content_items.is_empty() {
        let limit = content_items.len().min(10);
        let mut section = vec!["Known facts:".to_string()];
        for item in content_items.iter().take(limit) {
            section.push(format!("- {}", item.text));
        }
        if content_items.len() > limit {
            section.push(format!(
                "  ... and {} more results",
                content_items.len() - limit
            ));
        }
        parts.push(section.join("\n"));
    }

    // Related memories (only when explicitly included via parameter)
    if !memories.is_empty() {
        let limit = memories.len().min(5);
        let mut section = vec!["Related context:".to_string()];
        for mem in memories.iter().take(limit) {
            let summary = truncate(&mem.summary, 120);
            let dedup_key = summary.trim_end_matches('.').to_lowercase();
            if seen_texts.insert(dedup_key) {
                section.push(format!("- {}", summary));
            }
        }
        if section.len() > 1 {
            parts.push(section.join("\n"));
        }
    }

    parts.join("\n\n")
}

struct ContentItem {
    #[allow(dead_code)]
    kind: &'static str,
    text: String,
    #[allow(dead_code)]
    score: f32,
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.len() <= max {
        s.to_string()
    } else {
        // Find a valid UTF-8 boundary at or before `max`
        let boundary = s
            .char_indices()
            .take_while(|&(i, _)| i <= max)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}...", &s[..boundary])
    }
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
        let result =
            format_unified_results(&[], "test?", &graph, &[], &std::collections::HashSet::new());
        assert_eq!(result, "No relevant information found.");
    }

    #[test]
    fn test_format_with_concept_nodes() {
        let mut graph = Graph::new();
        let id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Alice".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        let fused = vec![(id, 0.95)];
        let result = format_unified_results(
            &fused,
            "tell me about Alice",
            &graph,
            &[],
            &std::collections::HashSet::new(),
        );
        assert!(result.contains("Alice"));
        assert!(result.contains("Person"));
    }

    #[test]
    fn test_format_with_claim_nodes() {
        let mut graph = Graph::new();
        let id = graph
            .add_node(GraphNode::new(NodeType::Claim {
                claim_id: 1,
                claim_text: "Alice prefers dark roast coffee".to_string(),
                confidence: 0.9,
                source_event_id: 1,
            }))
            .unwrap();

        let fused = vec![(id, 0.8)];
        let result = format_unified_results(
            &fused,
            "what does Alice like?",
            &graph,
            &[],
            &std::collections::HashSet::new(),
        );
        assert!(
            result.contains("Alice prefers dark roast coffee"),
            "Should show claim text, got: {}",
            result
        );
    }

    #[test]
    fn test_format_filters_low_scores() {
        let mut graph = Graph::new();
        let id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Noise".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        // Score below MIN_FUSED_SCORE (0.005)
        let fused = vec![(id, 0.004)];
        let result = format_unified_results(
            &fused,
            "anything?",
            &graph,
            &[],
            &std::collections::HashSet::new(),
        );
        assert_eq!(result, "No relevant information found.");
    }

    #[test]
    fn test_format_with_memories() {
        let graph = Graph::new();
        let mem = test_memory(1, "Alice prefers coffee over tea");
        let result = format_unified_results(
            &[],
            "preferences?",
            &graph,
            &[mem],
            &std::collections::HashSet::new(),
        );
        assert!(result.contains("Related context:"));
        assert!(result.contains("Alice prefers coffee"));
    }

    #[test]
    fn test_format_deduplicates() {
        let mut graph = Graph::new();
        let id = graph
            .add_node(GraphNode::new(NodeType::Claim {
                claim_id: 1,
                claim_text: "Alice likes coffee".to_string(),
                confidence: 0.9,
                source_event_id: 1,
            }))
            .unwrap();

        // Same content via fused results AND memory
        let fused = vec![(id, 0.8)];
        let mem = test_memory(1, "Alice likes coffee");
        let result = format_unified_results(
            &fused,
            "preferences?",
            &graph,
            &[mem],
            &std::collections::HashSet::new(),
        );
        // Should not show duplicate content
        let count = result.matches("Alice likes coffee").count();
        assert_eq!(
            count, 1,
            "Content should appear exactly once, got: {}",
            result
        );
    }

    #[test]
    fn test_format_memory_node_shows_summary() {
        let mut graph = Graph::new();
        // Add a Memory graph node
        let node_id = graph
            .add_node(GraphNode::new(NodeType::Memory {
                memory_id: 42,
                agent_id: 1,
                session_id: 1,
            }))
            .unwrap();

        // Create the actual Memory with a useful summary
        let mem = test_memory(42, "Customer reported late delivery on order #500");

        let fused = vec![(node_id, 0.5)];
        let result = format_unified_results(
            &fused,
            "late delivery?",
            &graph,
            &[mem],
            &std::collections::HashSet::new(),
        );
        assert!(
            result.contains("Customer reported late delivery"),
            "Should show memory summary, not 'Memory 42'. Got: {}",
            result
        );
        assert!(
            !result.contains("Memory 42"),
            "Should not show raw memory ID. Got: {}",
            result
        );
    }
}
