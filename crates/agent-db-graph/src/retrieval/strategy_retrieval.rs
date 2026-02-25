//! Multi-signal retrieval pipeline for strategies.
//!
//! Pure functions: accepts candidates + available signals, returns ranked IDs.

use super::fusion::multi_list_rrf;
use super::temporal::temporal_decay_score;
use crate::indexing::Bm25Index;
use crate::strategies::{Strategy, StrategyId};
use agent_db_core::types::Timestamp;
use agent_db_core::utils::cosine_similarity;
use std::collections::HashMap;

/// Configuration for strategy retrieval scoring.
#[derive(Debug, Clone)]
pub struct StrategyRetrievalConfig {
    /// RRF smoothing constant.
    pub rrf_k: f32,
    /// Half-life in hours for temporal decay.
    pub temporal_half_life_hours: f32,
    /// Minimum cosine similarity to include in semantic signal.
    pub min_semantic_similarity: f32,
    /// Maximum candidates per signal list before RRF.
    pub per_signal_limit: usize,
}

impl Default for StrategyRetrievalConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            temporal_half_life_hours: 72.0,
            min_semantic_similarity: 0.3,
            per_signal_limit: 50,
        }
    }
}

/// Query parameters for strategy retrieval.
#[derive(Debug, Clone)]
pub struct StrategyRetrievalQuery {
    /// Free-text query for BM25 keyword search. Empty → BM25 skipped.
    pub query_text: String,
    /// Query embedding for semantic search. Empty → semantic skipped.
    pub query_embedding: Vec<f32>,
    /// Anchor node in graph for PPR proximity signal.
    pub anchor_node: Option<u64>,
    /// Current timestamp (for temporal decay). Uses system time if None.
    pub now: Option<Timestamp>,
    /// Maximum results to return.
    pub limit: usize,
}

/// Stateless retrieval pipeline for strategies.
pub struct StrategyRetrievalPipeline;

impl StrategyRetrievalPipeline {
    /// Score and rank strategy candidates using all available signals.
    ///
    /// # Arguments
    /// * `candidates` — full Strategy objects to score
    /// * `jaccard_scored` — optional pre-computed Jaccard scores (strategy_id, score)
    ///   from the existing find_similar_strategies path
    /// * `query` — retrieval query parameters
    /// * `config` — scoring configuration
    /// * `bm25` — optional BM25 index (keyed by strategy_id as NodeId)
    /// * `ppr_scores` — optional PPR scores from graph (node_id → score)
    /// * `strategy_to_node` — mapping from StrategyId to graph NodeId
    ///
    /// Returns `(StrategyId, fused_score)` sorted descending.
    pub fn retrieve(
        candidates: &[Strategy],
        jaccard_scored: Option<&[(StrategyId, f32)]>,
        query: &StrategyRetrievalQuery,
        config: &StrategyRetrievalConfig,
        bm25: Option<&Bm25Index>,
        ppr_scores: Option<&HashMap<u64, f64>>,
        strategy_to_node: Option<&rustc_hash::FxHashMap<u64, u64>>,
    ) -> Vec<(StrategyId, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let now = query
            .now
            .unwrap_or_else(agent_db_core::types::current_timestamp);
        let limit = config.per_signal_limit;

        let mut ranked_lists: Vec<Vec<(StrategyId, f32)>> = Vec::new();

        // Signal 1: Semantic (cosine on summary_embedding)
        if !query.query_embedding.is_empty() {
            let mut semantic: Vec<(StrategyId, f32)> = candidates
                .iter()
                .filter(|s| !s.summary_embedding.is_empty())
                .map(|s| {
                    let sim = cosine_similarity(&query.query_embedding, &s.summary_embedding);
                    (s.id, sim)
                })
                .filter(|&(_, sim)| sim >= config.min_semantic_similarity)
                .collect();
            semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            semantic.truncate(limit);
            if !semantic.is_empty() {
                ranked_lists.push(semantic);
            }
        }

        // Signal 2: BM25 keyword search
        if !query.query_text.is_empty() {
            if let Some(index) = bm25 {
                let bm25_results = index.search(&query.query_text, limit);
                let candidate_ids: std::collections::HashSet<StrategyId> =
                    candidates.iter().map(|s| s.id).collect();
                let bm25_filtered: Vec<(StrategyId, f32)> = bm25_results
                    .into_iter()
                    .filter(|&(id, _)| candidate_ids.contains(&id))
                    .collect();
                if !bm25_filtered.is_empty() {
                    ranked_lists.push(bm25_filtered);
                }
            }
        }

        // Signal 3: Existing Jaccard similarity (goals/tools/results)
        if let Some(jaccard) = jaccard_scored {
            let mut jac: Vec<(StrategyId, f32)> = jaccard
                .iter()
                .filter(|&&(_, score)| score > 0.0)
                .copied()
                .collect();
            jac.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            jac.truncate(limit);
            if !jac.is_empty() {
                ranked_lists.push(jac);
            }
        }

        // Signal 4: Temporal recency
        {
            let mut temporal: Vec<(StrategyId, f32)> = candidates
                .iter()
                .map(|s| {
                    let ts = s.last_used.max(s.created_at);
                    let score = temporal_decay_score(ts, now, config.temporal_half_life_hours);
                    (s.id, score)
                })
                .filter(|&(_, s)| s > 1e-6)
                .collect();
            temporal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            temporal.truncate(limit);
            if !temporal.is_empty() {
                ranked_lists.push(temporal);
            }
        }

        // Signal 5: Graph proximity (PPR)
        if let (Some(ppr), Some(s2n)) = (ppr_scores, strategy_to_node) {
            let mut proximity: Vec<(StrategyId, f32)> = candidates
                .iter()
                .filter_map(|s| {
                    let node_id = s2n.get(&s.id)?;
                    let &score = ppr.get(node_id)?;
                    if score > 1e-9 {
                        Some((s.id, score as f32))
                    } else {
                        None
                    }
                })
                .collect();
            proximity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            proximity.truncate(limit);
            if !proximity.is_empty() {
                ranked_lists.push(proximity);
            }
        }

        // Signal 6: Quality × Confidence
        {
            let mut quality: Vec<(StrategyId, f32)> = candidates
                .iter()
                .map(|s| {
                    let score = s.quality_score * s.confidence;
                    (s.id, score)
                })
                .filter(|&(_, s)| s > 1e-6)
                .collect();
            quality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            quality.truncate(limit);
            if !quality.is_empty() {
                ranked_lists.push(quality);
            }
        }

        // Fuse all signals via RRF
        let mut fused = multi_list_rrf(&ranked_lists, config.rrf_k);

        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(query.limit);
        fused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeOutcome;
    use crate::strategies::StrategyType;

    fn make_strategy(id: StrategyId, quality: f32, confidence: f32, hours_ago: u64) -> Strategy {
        let nanos_per_hour = 3_600_000_000_000u64;
        let now = 1000 * nanos_per_hour;
        Strategy {
            id,
            name: format!("Strategy {}", id),
            summary: format!("Strategy {} summary", id),
            when_to_use: String::new(),
            when_not_to_use: String::new(),
            failure_modes: Vec::new(),
            playbook: Vec::new(),
            counterfactual: String::new(),
            supersedes: Vec::new(),
            applicable_domains: Vec::new(),
            lineage_depth: 0,
            summary_embedding: Vec::new(),
            agent_id: 1,
            reasoning_steps: Vec::new(),
            context_patterns: Vec::new(),
            success_indicators: Vec::new(),
            failure_patterns: Vec::new(),
            quality_score: quality,
            success_count: 5,
            failure_count: 1,
            support_count: 6,
            strategy_type: StrategyType::Positive,
            precondition: String::new(),
            action_hint: String::new(),
            expected_success: 0.8,
            expected_cost: 1.0,
            expected_value: 0.5,
            confidence,
            contradictions: Vec::new(),
            goal_bucket_id: 0,
            behavior_signature: String::new(),
            source_episodes: Vec::new(),
            created_at: now - hours_ago * nanos_per_hour,
            last_used: now - hours_ago * nanos_per_hour,
            metadata: std::collections::HashMap::new(),
            self_judged_quality: None,
            source_outcomes: vec![EpisodeOutcome::Success],
            version: 1,
            parent_strategy: None,
        }
    }

    #[test]
    fn test_strategy_retrieval_fuses_jaccard_and_quality() {
        let s1 = make_strategy(1, 0.9, 0.9, 10); // high quality
        let s2 = make_strategy(2, 0.3, 0.3, 10); // low quality

        let candidates = vec![s1, s2];
        let jaccard = vec![(2u64, 0.95), (1u64, 0.1)]; // s2 wins Jaccard

        let query = StrategyRetrievalQuery {
            query_text: String::new(),
            query_embedding: Vec::new(),
            anchor_node: None,
            now: Some(1000 * 3_600_000_000_000),
            limit: 10,
        };
        let config = StrategyRetrievalConfig::default();
        let result = StrategyRetrievalPipeline::retrieve(
            &candidates,
            Some(&jaccard),
            &query,
            &config,
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 2);
        // s1 has high quality×confidence, s2 wins Jaccard — both should appear
        // The exact ranking depends on RRF weights, but both should be present
        let ids: Vec<u64> = result.iter().map(|r| r.0).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_strategy_retrieval_empty_candidates() {
        let query = StrategyRetrievalQuery {
            query_text: String::new(),
            query_embedding: Vec::new(),
            anchor_node: None,
            now: None,
            limit: 10,
        };
        let result = StrategyRetrievalPipeline::retrieve(
            &[],
            None,
            &query,
            &StrategyRetrievalConfig::default(),
            None,
            None,
            None,
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_strategy_retrieval_semantic_signal() {
        let mut s1 = make_strategy(1, 0.5, 0.5, 10);
        s1.summary_embedding = vec![1.0, 0.0, 0.0];
        let mut s2 = make_strategy(2, 0.5, 0.5, 10);
        s2.summary_embedding = vec![0.0, 1.0, 0.0];

        let query = StrategyRetrievalQuery {
            query_text: String::new(),
            query_embedding: vec![1.0, 0.0, 0.0], // Matches s1
            anchor_node: None,
            now: Some(1000 * 3_600_000_000_000),
            limit: 10,
        };
        let config = StrategyRetrievalConfig::default();
        let result =
            StrategyRetrievalPipeline::retrieve(&[s1, s2], None, &query, &config, None, None, None);
        assert!(!result.is_empty());
        assert_eq!(
            result[0].0, 1,
            "Semantically matching strategy should rank first"
        );
    }
}
