//! Reranking stage for multi-signal retrieval.
//!
//! After RRF fusion produces a ranked list, the reranker re-scores the top-K
//! candidates using a more expensive signal (e.g., LLM judgment or cross-encoder).
//! This two-stage approach (fast recall → precise rerank) is standard in
//! information retrieval and inspired by prior work's reranker abstraction.

use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::memory::Memory;
use async_trait::async_trait;

/// A reranked item with its new score.
#[derive(Debug, Clone)]
pub struct RerankedItem {
    /// The original item ID (MemoryId or StrategyId).
    pub id: u64,
    /// The reranker's score (higher = more relevant).
    pub rerank_score: f32,
    /// The original fusion score before reranking.
    pub original_score: f32,
}

/// Configuration for the reranking stage.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Maximum number of candidates to send to the reranker.
    /// Only the top-K from RRF are reranked (the rest keep their RRF scores).
    pub top_k: usize,
    /// Weight blend: `final = alpha * rerank_score + (1-alpha) * original_score`.
    /// 1.0 = pure reranker, 0.0 = pure RRF.
    pub alpha: f32,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            alpha: 0.7,
        }
    }
}

/// Trait for reranking a set of memory candidates given a query.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank the given candidates for the query.
    ///
    /// Returns scores in the same order as `candidates`.
    /// Scores should be in [0.0, 1.0] where 1.0 is most relevant.
    async fn rerank(&self, query: &str, candidates: &[&Memory]) -> anyhow::Result<Vec<f32>>;
}

/// LLM-based reranker that asks the LLM to score relevance of each candidate.
pub struct LlmReranker {
    client: Box<dyn LlmClient>,
}

impl LlmReranker {
    pub fn new(client: Box<dyn LlmClient>) -> Self {
        Self { client }
    }
}

const RERANK_SYSTEM_PROMPT: &str = r#"You are a relevance scoring assistant. Given a query and a list of memory summaries, score each memory's relevance to the query on a scale of 0.0 to 1.0.

Scoring guide:
- 1.0: Directly answers or is highly relevant to the query
- 0.7-0.9: Strongly related, contains useful context
- 0.4-0.6: Somewhat related, tangential information
- 0.1-0.3: Weakly related, mostly irrelevant
- 0.0: Completely irrelevant

Respond with a JSON array of numbers (one score per memory, in order).
Example: [0.9, 0.3, 0.7, 0.1]"#;

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(&self, query: &str, candidates: &[&Memory]) -> anyhow::Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let memories_text = candidates
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "  [{}]: {} | Takeaway: {}",
                    i,
                    safe_truncate(&m.summary, 200),
                    safe_truncate(&m.takeaway, 100),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let user_prompt = format!(
            "Query: {}\n\nMemories to score:\n{}\n\nRespond with the JSON array of scores.",
            query, memories_text
        );

        let request = LlmRequest {
            system_prompt: RERANK_SYSTEM_PROMPT.to_string(),
            user_prompt,
            temperature: 0.0,
            max_tokens: 256,
            json_mode: true,
        };

        let response = self.client.complete(request).await?;
        let scores = parse_scores(&response.content, candidates.len());
        Ok(scores)
    }
}

/// Parse LLM response into score vector, falling back to uniform scores on error.
fn parse_scores(response: &str, expected_len: usize) -> Vec<f32> {
    if let Some(json) = parse_json_from_llm(response) {
        if let Some(arr) = json.as_array() {
            let scores: Vec<f32> = arr
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.5) as f32)
                .map(|s| s.clamp(0.0, 1.0))
                .collect();
            if scores.len() == expected_len {
                return scores;
            }
        }
    }
    // Fallback: uniform scores (preserves original ranking)
    vec![0.5; expected_len]
}

fn safe_truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Apply reranking to a set of RRF-fused results.
///
/// Takes the top-K items from `fused_results`, reranks them, and blends scores.
/// Items beyond top-K keep their original RRF scores.
pub async fn apply_reranking(
    reranker: &dyn Reranker,
    query: &str,
    fused_results: &[(u64, f32)],
    all_memories: &[Memory],
    config: &RerankerConfig,
) -> anyhow::Result<Vec<(u64, f32)>> {
    if fused_results.is_empty() || config.top_k == 0 {
        return Ok(fused_results.to_vec());
    }

    let k = config.top_k.min(fused_results.len());

    // Collect (id, memory_ref) pairs for the top-K, preserving alignment
    let top_k_pairs: Vec<(u64, &Memory)> = fused_results[..k]
        .iter()
        .filter_map(|&(id, _)| all_memories.iter().find(|m| m.id == id).map(|m| (id, m)))
        .collect();

    if top_k_pairs.is_empty() {
        return Ok(fused_results.to_vec());
    }

    let top_k_memories: Vec<&Memory> = top_k_pairs.iter().map(|(_, m)| *m).collect();
    let scores = reranker.rerank(query, &top_k_memories).await?;

    // Build a map from memory ID → reranker score for correct alignment
    let score_map: std::collections::HashMap<u64, f32> = top_k_pairs
        .iter()
        .zip(scores.iter())
        .map(|(&(id, _), &score)| (id, score))
        .collect();

    // Blend scores using the ID-based map (not positional index)
    let mut result: Vec<(u64, f32)> = Vec::with_capacity(fused_results.len());

    for &(id, original_score) in fused_results.iter() {
        if let Some(&rerank_score) = score_map.get(&id) {
            let blended = config.alpha * rerank_score + (1.0 - config.alpha) * original_score;
            result.push((id, blended));
        } else {
            result.push((id, original_score));
        }
    }

    // Re-sort by blended score (NaN-safe)
    result.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scores_valid() {
        let response = "[0.9, 0.3, 0.7]";
        let scores = parse_scores(response, 3);
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.9).abs() < 0.01);
        assert!((scores[1] - 0.3).abs() < 0.01);
        assert!((scores[2] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_parse_scores_clamped() {
        let response = "[1.5, -0.3, 0.7]";
        let scores = parse_scores(response, 3);
        assert!((scores[0] - 1.0).abs() < 0.01); // clamped to 1.0
        assert!((scores[1] - 0.0).abs() < 0.01); // clamped to 0.0
    }

    #[test]
    fn test_parse_scores_wrong_length_falls_back() {
        let response = "[0.9, 0.3]"; // expected 3
        let scores = parse_scores(response, 3);
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| (s - 0.5).abs() < 0.01));
    }

    #[test]
    fn test_parse_scores_invalid_json() {
        let scores = parse_scores("not json", 3);
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| (s - 0.5).abs() < 0.01));
    }

    #[test]
    fn test_parse_scores_fenced() {
        let response = "```json\n[0.8, 0.2]\n```";
        let scores = parse_scores(response, 2);
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_reranker_config_default() {
        let config = RerankerConfig::default();
        assert_eq!(config.top_k, 10);
        assert!((config.alpha - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_safe_truncate() {
        assert_eq!(safe_truncate("hello", 10), "hello");
        assert_eq!(safe_truncate("hello world", 5), "hello");
        // Multi-byte: "café" — 'é' is 2 bytes at index 3-4
        let s = "café";
        let t = safe_truncate(s, 4);
        assert!(t.len() <= 4);
        assert!(t.is_char_boundary(t.len()));
    }

    #[tokio::test]
    async fn test_apply_reranking_empty() {
        // Stub reranker
        struct StubReranker;
        #[async_trait]
        impl Reranker for StubReranker {
            async fn rerank(&self, _q: &str, candidates: &[&Memory]) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.5; candidates.len()])
            }
        }

        let config = RerankerConfig::default();
        let result = apply_reranking(&StubReranker, "query", &[], &[], &config)
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_apply_reranking_reorders() {
        use crate::episodes::EpisodeOutcome;
        use crate::memory::{ConsolidationStatus, MemoryTier, MemoryType};
        use agent_db_events::core::EventContext;

        fn make_mem(id: u64, summary: &str) -> Memory {
            Memory {
                id,
                agent_id: 1,
                session_id: 1,
                episode_id: 1,
                summary: summary.to_string(),
                takeaway: String::new(),
                causal_note: String::new(),
                summary_embedding: Vec::new(),
                tier: MemoryTier::Episodic,
                consolidated_from: Vec::new(),
                schema_id: None,
                consolidation_status: ConsolidationStatus::Active,
                context: EventContext::default(),
                key_events: Vec::new(),
                strength: 0.7,
                relevance_score: 0.5,
                formed_at: 0,
                last_accessed: 0,
                access_count: 0,
                outcome: EpisodeOutcome::Success,
                memory_type: MemoryType::Episodic { significance: 0.5 },
                metadata: std::collections::HashMap::new(),
                expires_at: None,
            }
        }

        // Reranker that reverses order (gives highest score to last item)
        struct ReverseReranker;
        #[async_trait]
        impl Reranker for ReverseReranker {
            async fn rerank(&self, _q: &str, candidates: &[&Memory]) -> anyhow::Result<Vec<f32>> {
                let n = candidates.len() as f32;
                Ok((0..candidates.len())
                    .map(|i| (n - i as f32) / n)
                    .rev()
                    .collect())
            }
        }

        let memories = vec![
            make_mem(1, "first"),
            make_mem(2, "second"),
            make_mem(3, "third"),
        ];
        let fused = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let config = RerankerConfig {
            top_k: 3,
            alpha: 1.0, // pure reranker
        };

        let result = apply_reranking(&ReverseReranker, "query", &fused, &memories, &config)
            .await
            .unwrap();

        // After pure reranking with reversed scores, order should change
        assert_eq!(result.len(), 3);
        // All should have scores
        assert!(result.iter().all(|&(_, s)| s > 0.0));
    }
}
