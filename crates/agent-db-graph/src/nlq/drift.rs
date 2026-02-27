//! DRIFT-style adaptive search pipeline.
//!
//! 3-phase pipeline:
//! 1. **Primer** — Score community summaries against query, generate follow-up queries via LLM
//! 2. **Follow-up** — Execute follow-up queries across BM25/memory/claims, deduplicate
//! 3. **Synthesis** — LLM synthesizes a comprehensive answer from all retrieved context

use crate::community_summary::CommunitySummary;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

/// Configuration for DRIFT search.
#[derive(Debug, Clone)]
pub struct DriftConfig {
    /// Maximum communities to use in the primer phase.
    pub max_primer_communities: usize,
    /// Maximum results per follow-up query.
    pub max_followup_results: usize,
    /// Maximum follow-up queries to generate.
    pub max_followup_queries: usize,
    /// LLM temperature for synthesis.
    pub synthesis_temperature: f32,
    /// Maximum tokens for synthesis response.
    pub synthesis_max_tokens: u32,
    /// Overall timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            max_primer_communities: 5,
            max_followup_results: 10,
            max_followup_queries: 3,
            synthesis_temperature: 0.3,
            synthesis_max_tokens: 512,
            timeout_secs: 30,
        }
    }
}

/// Result of a DRIFT search.
#[derive(Debug, Clone)]
pub struct DriftResult {
    pub answer: String,
    pub primer_communities_used: Vec<u64>,
    pub followup_queries: Vec<String>,
    pub total_items_retrieved: usize,
}

/// Parsed LLM response for follow-up query generation.
#[derive(Debug, Deserialize)]
struct FollowUpResponse {
    queries: Vec<String>,
}

/// Phase 1: Score community summaries against the query and generate follow-up queries.
///
/// Uses keyword overlap (BM25-like) to rank communities, then asks the LLM
/// to generate targeted follow-up queries based on the top communities.
pub async fn drift_primer(
    client: &dyn LlmClient,
    question: &str,
    summaries: &HashMap<u64, CommunitySummary>,
    config: &DriftConfig,
) -> (Vec<u64>, Vec<String>) {
    if summaries.is_empty() {
        return (vec![], vec![question.to_string()]);
    }

    // Score each community by keyword overlap
    let query_tokens_owned: HashSet<String> = question
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .map(|s| s.to_string())
        .collect();

    let mut scored: Vec<(u64, f32)> = summaries
        .iter()
        .map(|(&cid, summary)| {
            let summary_lower = summary.summary.to_lowercase();
            let entity_text = summary.key_entities.join(" ").to_lowercase();
            let combined = format!("{} {}", summary_lower, entity_text);

            let overlap: usize = query_tokens_owned
                .iter()
                .filter(|t| combined.contains(t.as_str()))
                .count();

            let score = if query_tokens_owned.is_empty() {
                0.0
            } else {
                overlap as f32 / query_tokens_owned.len() as f32
            };
            (cid, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(config.max_primer_communities);

    let top_community_ids: Vec<u64> = scored.iter().map(|(cid, _)| *cid).collect();

    // Build context from top communities
    let mut context_parts = Vec::new();
    for &cid in &top_community_ids {
        if let Some(summary) = summaries.get(&cid) {
            context_parts.push(format!(
                "Community {}: {} (entities: {})",
                cid,
                summary.summary,
                summary.key_entities.join(", ")
            ));
        }
    }

    if context_parts.is_empty() {
        return (top_community_ids, vec![question.to_string()]);
    }

    // Ask LLM for follow-up queries
    let system = format!(
        "Given community summaries and a user question, generate up to {} targeted follow-up search queries \
         that would help answer the question comprehensively. Output strict JSON: {{\"queries\": [...]}}. \
         No markdown fences.",
        config.max_followup_queries
    );
    let user = format!(
        "Communities:\n{}\n\nUser question: {}",
        context_parts.join("\n"),
        question
    );

    let request = LlmRequest {
        system_prompt: system,
        user_prompt: user,
        temperature: 0.0,
        max_tokens: 256,
        json_mode: true,
    };

    let followups =
        match tokio::time::timeout(std::time::Duration::from_secs(10), client.complete(request))
            .await
        {
            Ok(Ok(response)) => {
                if let Some(value) = parse_json_from_llm(&response.content) {
                    if let Ok(parsed) = serde_json::from_value::<FollowUpResponse>(value) {
                        let mut queries: Vec<String> = parsed
                            .queries
                            .into_iter()
                            .filter(|q| !q.trim().is_empty())
                            .take(config.max_followup_queries)
                            .collect();
                        // Always include the original question
                        if !queries.iter().any(|q| q == question) {
                            queries.insert(0, question.to_string());
                        }
                        queries
                    } else {
                        vec![question.to_string()]
                    }
                } else {
                    vec![question.to_string()]
                }
            },
            _ => vec![question.to_string()],
        };

    (top_community_ids, followups)
}

/// Phase 2: Execute follow-up queries and collect deduplicated results.
///
/// The caller provides a search function that returns `Vec<(u64, f32)>` for each query.
/// Results are deduplicated by node_id, keeping the highest score.
pub fn drift_followup_merge(results_per_query: &[Vec<(u64, f32)>]) -> Vec<(u64, f32)> {
    let mut best_scores: HashMap<u64, f32> = HashMap::new();
    for results in results_per_query {
        for &(id, score) in results {
            let entry = best_scores.entry(id).or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }
    }
    let mut merged: Vec<(u64, f32)> = best_scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// Phase 3: Synthesize a comprehensive answer from community context and retrieved snippets.
///
/// Returns the synthesized text, or falls back to a formatted version of the raw snippets.
pub async fn drift_synthesis(
    client: &dyn LlmClient,
    question: &str,
    community_context: &[String],
    retrieved_snippets: &[String],
    config: &DriftConfig,
) -> String {
    let mut context = String::with_capacity(4096);

    if !community_context.is_empty() {
        context.push_str("Community summaries:\n");
        for c in community_context {
            context.push_str(&format!("- {}\n", c));
        }
        context.push('\n');
    }

    if !retrieved_snippets.is_empty() {
        context.push_str("Retrieved information:\n");
        for (i, snippet) in retrieved_snippets.iter().take(20).enumerate() {
            context.push_str(&format!("{}. {}\n", i + 1, snippet));
        }
    }

    if context.is_empty() {
        return format!("No information found for: {}", question);
    }

    // Truncate context to ~12K chars
    if context.len() > 12_000 {
        context.truncate(12_000);
        context.push_str("\n... (truncated)");
    }

    let system = "Answer the user's question comprehensively based on the provided context. \
                  Be specific and reference the information given. If the context doesn't fully \
                  answer the question, say what is known and what is missing.";

    let user = format!("Context:\n{}\n\nQuestion: {}", context, question);

    let request = LlmRequest {
        system_prompt: system.to_string(),
        user_prompt: user,
        temperature: config.synthesis_temperature,
        max_tokens: config.synthesis_max_tokens,
        json_mode: false,
    };

    match tokio::time::timeout(
        std::time::Duration::from_secs(config.timeout_secs),
        client.complete(request),
    )
    .await
    {
        Ok(Ok(response)) if !response.content.trim().is_empty() => response.content,
        _ => {
            // Fail-open: format raw snippets
            let mut fallback = format!("Information related to: {}\n\n", question);
            for snippet in retrieved_snippets.iter().take(10) {
                fallback.push_str(&format!("- {}\n", snippet));
            }
            fallback
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DriftConfig::default();
        assert_eq!(config.max_primer_communities, 5);
        assert_eq!(config.max_followup_queries, 3);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_followup_merge_dedup() {
        let q1 = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let q2 = vec![(2, 0.95), (4, 0.6)];
        let merged = drift_followup_merge(&[q1, q2]);

        let score_for = |id: u64| merged.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
        assert_eq!(score_for(1), Some(0.9));
        assert_eq!(score_for(2), Some(0.95)); // highest wins
        assert_eq!(score_for(3), Some(0.7));
        assert_eq!(score_for(4), Some(0.6));
    }

    #[test]
    fn test_followup_merge_empty() {
        let merged = drift_followup_merge(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_followup_merge_sorted_by_score() {
        let q1 = vec![(1, 0.5), (2, 0.9)];
        let merged = drift_followup_merge(&[q1]);
        assert_eq!(merged[0].0, 2);
        assert_eq!(merged[1].0, 1);
    }

    #[test]
    fn test_primer_empty_summaries() {
        // Synchronous test for the empty case
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            // With empty summaries, should return original question
            let summaries = HashMap::new();
            let config = DriftConfig::default();
            // We can't easily test with a real LLM client, but the empty path doesn't call LLM
            let (communities, queries) = drift_primer_no_llm("test question?", &summaries, &config);
            assert!(communities.is_empty());
            assert_eq!(queries, vec!["test question?"]);
        });
    }

    /// Test-only version of primer that doesn't require LLM (for scoring logic).
    fn drift_primer_no_llm(
        question: &str,
        summaries: &HashMap<u64, CommunitySummary>,
        config: &DriftConfig,
    ) -> (Vec<u64>, Vec<String>) {
        if summaries.is_empty() {
            return (vec![], vec![question.to_string()]);
        }

        let query_tokens: HashSet<String> = question
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() >= 3)
            .map(|s| s.to_string())
            .collect();

        let mut scored: Vec<(u64, f32)> = summaries
            .iter()
            .map(|(&cid, summary)| {
                let combined = format!(
                    "{} {}",
                    summary.summary.to_lowercase(),
                    summary.key_entities.join(" ").to_lowercase()
                );
                let overlap: usize = query_tokens
                    .iter()
                    .filter(|t| combined.contains(t.as_str()))
                    .count();
                let score = if query_tokens.is_empty() {
                    0.0
                } else {
                    overlap as f32 / query_tokens.len() as f32
                };
                (cid, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(config.max_primer_communities);

        let ids: Vec<u64> = scored.iter().map(|(cid, _)| *cid).collect();
        (ids, vec![question.to_string()])
    }

    #[test]
    fn test_primer_scoring() {
        let mut summaries = HashMap::new();
        summaries.insert(
            1,
            CommunitySummary {
                community_id: 1,
                summary: "Alice and Bob work together on projects".to_string(),
                key_entities: vec!["Alice".to_string(), "Bob".to_string()],
                node_count: 5,
                generated_at: 0,
                token_estimate: 0,
            },
        );
        summaries.insert(
            2,
            CommunitySummary {
                community_id: 2,
                summary: "Charlie maintains database systems".to_string(),
                key_entities: vec!["Charlie".to_string()],
                node_count: 3,
                generated_at: 0,
                token_estimate: 0,
            },
        );

        let config = DriftConfig::default();
        let (ids, _queries) = drift_primer_no_llm("Tell me about Alice", &summaries, &config);

        // Community 1 should score higher (contains "Alice")
        assert!(!ids.is_empty());
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_primer_caps_at_max() {
        let mut summaries = HashMap::new();
        for i in 0..20 {
            summaries.insert(
                i,
                CommunitySummary {
                    community_id: i,
                    summary: format!("Community {} about topic{}", i, i),
                    key_entities: vec![format!("Entity{}", i)],
                    node_count: 5,
                    generated_at: 0,
                    token_estimate: 0,
                },
            );
        }

        let config = DriftConfig {
            max_primer_communities: 3,
            ..Default::default()
        };
        let (ids, _) = drift_primer_no_llm("topic5 topic10", &summaries, &config);
        assert!(ids.len() <= 3);
    }

    #[test]
    fn test_parse_followup_response() {
        let json = r#"{"queries": ["What is Alice's role?", "How does Bob contribute?"]}"#;
        let value = parse_json_from_llm(json).unwrap();
        let parsed: FollowUpResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.queries.len(), 2);
    }

    #[test]
    fn test_parse_followup_response_empty() {
        let json = r#"{"queries": []}"#;
        let value = parse_json_from_llm(json).unwrap();
        let parsed: FollowUpResponse = serde_json::from_value(value).unwrap();
        assert!(parsed.queries.is_empty());
    }
}
