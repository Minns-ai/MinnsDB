//! Multi-signal retrieval pipeline for memories.
//!
//! Pure functions: accepts candidates + available signals, returns ranked IDs.
//! Signals are independently optional; the pipeline fuses whatever is available.

use super::fusion::multi_list_rrf;
use super::temporal::{
    importance_modulated_decay_score, temporal_decay_score, ImportanceDecayConfig,
    ImportanceDecayParams,
};
use crate::indexing::Bm25Index;
use crate::memory::{calculate_context_similarity, Memory, MemoryId, MemoryTier};
use agent_db_core::types::{AgentId, SessionId, Timestamp};
use agent_db_core::utils::cosine_similarity;
use agent_db_events::core::EventContext;
use std::collections::HashMap;

/// Configuration for memory retrieval scoring.
#[derive(Debug, Clone)]
pub struct MemoryRetrievalConfig {
    /// RRF smoothing constant.
    pub rrf_k: f32,
    /// Half-life in hours for temporal decay.
    pub temporal_half_life_hours: f32,
    /// Minimum cosine similarity to include in semantic signal.
    pub min_semantic_similarity: f32,
    /// Maximum candidates per signal list before RRF.
    pub per_signal_limit: usize,
    /// Tier boost for Schema memories (added to fused score).
    pub tier_boost_schema: f32,
    /// Tier boost for Semantic memories.
    pub tier_boost_semantic: f32,
    /// Enable importance-modulated decay (baseline system-inspired).
    /// When true, Signal 4 uses importance to slow decay for frequently-accessed,
    /// highly-relevant memories.
    pub enable_importance_decay: bool,
    /// Configuration for importance-modulated decay weights.
    pub importance_decay_config: ImportanceDecayConfig,
}

impl Default for MemoryRetrievalConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            temporal_half_life_hours: 24.0,
            min_semantic_similarity: 0.3,
            per_signal_limit: 50,
            tier_boost_schema: 0.3,
            tier_boost_semantic: 0.15,
            enable_importance_decay: true,
            importance_decay_config: ImportanceDecayConfig::default(),
        }
    }
}

/// Query parameters for memory retrieval.
#[derive(Debug, Clone)]
pub struct MemoryRetrievalQuery {
    /// Free-text query for BM25 keyword search. Empty → BM25 skipped.
    pub query_text: String,
    /// Query embedding for semantic search. Empty → semantic skipped.
    pub query_embedding: Vec<f32>,
    /// Context snapshot for context-similarity signal.
    pub context: Option<EventContext>,
    /// Anchor node in graph for PPR proximity signal.
    pub anchor_node: Option<u64>,
    /// Filter: only return memories for this agent.
    pub agent_id: Option<AgentId>,
    /// Filter: only return memories for this session.
    pub session_id: Option<SessionId>,
    /// Current timestamp (for temporal decay). Uses system time if None.
    pub now: Option<Timestamp>,
    /// Maximum results to return.
    pub limit: usize,
}

/// Stateless retrieval pipeline for memories.
pub struct MemoryRetrievalPipeline;

impl MemoryRetrievalPipeline {
    /// Score and rank memory candidates using all available signals.
    ///
    /// # Arguments
    /// * `candidates` — full Memory objects to score
    /// * `query` — retrieval query parameters
    /// * `config` — scoring configuration
    /// * `bm25` — optional BM25 index (keyed by memory_id as NodeId)
    /// * `ppr_scores` — optional PPR scores from graph (node_id → score)
    /// * `memory_to_node` — mapping from MemoryId to graph NodeId (for PPR lookup)
    ///
    /// Returns `(MemoryId, fused_score)` sorted descending.
    pub fn retrieve(
        candidates: &[Memory],
        query: &MemoryRetrievalQuery,
        config: &MemoryRetrievalConfig,
        bm25: Option<&Bm25Index>,
        ppr_scores: Option<&HashMap<u64, f64>>,
        memory_to_node: Option<&rustc_hash::FxHashMap<u64, u64>>,
    ) -> Vec<(MemoryId, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Apply agent/session filters
        let filtered: Vec<&Memory> = candidates
            .iter()
            .filter(|m| {
                if let Some(aid) = query.agent_id {
                    if m.agent_id != aid {
                        return false;
                    }
                }
                if let Some(sid) = query.session_id {
                    if m.session_id != sid {
                        return false;
                    }
                }
                true
            })
            .collect();

        if filtered.is_empty() {
            return Vec::new();
        }

        let now = query
            .now
            .unwrap_or_else(agent_db_core::types::current_timestamp);
        let limit = config.per_signal_limit;

        let mut ranked_lists: Vec<Vec<(MemoryId, f32)>> = Vec::new();

        // Signal 1: Semantic (cosine on summary_embedding)
        if !query.query_embedding.is_empty() {
            let mut semantic: Vec<(MemoryId, f32)> = filtered
                .iter()
                .filter(|m| !m.summary_embedding.is_empty())
                .map(|m| {
                    let sim = cosine_similarity(&query.query_embedding, &m.summary_embedding);
                    (m.id, sim)
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
                // BM25 index is keyed by memory_id (cast to NodeId). Filter to our candidate set.
                let candidate_ids: std::collections::HashSet<MemoryId> =
                    filtered.iter().map(|m| m.id).collect();
                let bm25_filtered: Vec<(MemoryId, f32)> = bm25_results
                    .into_iter()
                    .filter(|&(id, _)| candidate_ids.contains(&id))
                    .collect();
                if !bm25_filtered.is_empty() {
                    ranked_lists.push(bm25_filtered);
                }
            }
        }

        // Signal 3: Context similarity
        if let Some(ref ctx) = query.context {
            let mut ctx_scores: Vec<(MemoryId, f32)> = filtered
                .iter()
                .map(|m| {
                    let sim = calculate_context_similarity(ctx, &m.context);
                    (m.id, sim)
                })
                .filter(|&(_, sim)| sim > 0.0)
                .collect();
            ctx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ctx_scores.truncate(limit);
            if !ctx_scores.is_empty() {
                ranked_lists.push(ctx_scores);
            }
        }

        // Signal 4: Temporal recency (with optional importance modulation)
        {
            // Pre-compute max access count for normalization
            let max_access = filtered
                .iter()
                .map(|m| m.access_count)
                .max()
                .unwrap_or(1)
                .max(1) as f32;

            let mut temporal: Vec<(MemoryId, f32)> = filtered
                .iter()
                .map(|m| {
                    let score = if config.enable_importance_decay {
                        // Compute per-memory semantic relevance for importance
                        let relevance = if !query.query_embedding.is_empty()
                            && !m.summary_embedding.is_empty()
                        {
                            cosine_similarity(&query.query_embedding, &m.summary_embedding).max(0.0)
                        } else {
                            0.0
                        };
                        let params = ImportanceDecayParams {
                            access_frequency: m.access_count as f32 / max_access,
                            relevance,
                            strength: m.strength,
                        };
                        importance_modulated_decay_score(
                            m.formed_at,
                            now,
                            config.temporal_half_life_hours,
                            &params,
                            &config.importance_decay_config,
                        )
                    } else {
                        temporal_decay_score(m.formed_at, now, config.temporal_half_life_hours)
                    };
                    (m.id, score)
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
        if let (Some(ppr), Some(m2n)) = (ppr_scores, memory_to_node) {
            let mut proximity: Vec<(MemoryId, f32)> = filtered
                .iter()
                .filter_map(|m| {
                    let node_id = m2n.get(&m.id)?;
                    let &score = ppr.get(node_id)?;
                    if score > 1e-9 {
                        Some((m.id, score as f32))
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

        // Signal 6: Access frequency (normalized)
        {
            let max_access = filtered
                .iter()
                .map(|m| m.access_count)
                .max()
                .unwrap_or(1)
                .max(1) as f32;
            let mut access: Vec<(MemoryId, f32)> = filtered
                .iter()
                .map(|m| (m.id, m.access_count as f32 / max_access))
                .collect();
            access.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            access.truncate(limit);
            if !access.is_empty() {
                ranked_lists.push(access);
            }
        }

        // Fuse all signals via RRF
        let mut fused = multi_list_rrf(&ranked_lists, config.rrf_k);

        // Post-fusion tier boost
        let tier_map: HashMap<MemoryId, MemoryTier> =
            filtered.iter().map(|m| (m.id, m.tier.clone())).collect();
        for item in &mut fused {
            match tier_map.get(&item.0) {
                Some(MemoryTier::Schema) => item.1 += config.tier_boost_schema,
                Some(MemoryTier::Semantic) => item.1 += config.tier_boost_semantic,
                _ => {},
            }
        }

        // Re-sort after tier boost
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(query.limit);
        fused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeOutcome;
    use crate::memory::MemoryType;
    use agent_db_events::core::EventContext;

    fn make_memory(id: MemoryId, tier: MemoryTier, formed_hours_ago: u64) -> Memory {
        let nanos_per_hour = 3_600_000_000_000u64;
        let now = 1000 * nanos_per_hour;
        Memory {
            id,
            agent_id: 1,
            session_id: 1,
            episode_id: id,
            summary: format!("Memory {}", id),
            takeaway: String::new(),
            causal_note: String::new(),
            summary_embedding: Vec::new(),
            tier,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: crate::memory::ConsolidationStatus::default(),
            context: EventContext::default(),
            key_events: Vec::new(),
            strength: 0.8,
            relevance_score: 0.5,
            formed_at: now - formed_hours_ago * nanos_per_hour,
            last_accessed: now,
            access_count: 1,
            outcome: EpisodeOutcome::Success,
            memory_type: MemoryType::Episodic { significance: 0.5 },
            metadata: std::collections::HashMap::new(),
            expires_at: None,
        }
    }

    #[test]
    fn test_memory_retrieval_no_embeddings() {
        let candidates = vec![
            make_memory(1, MemoryTier::Episodic, 1),
            make_memory(2, MemoryTier::Episodic, 48),
            make_memory(3, MemoryTier::Episodic, 100),
        ];
        let query = MemoryRetrievalQuery {
            query_text: String::new(),
            query_embedding: Vec::new(),
            context: None,
            anchor_node: None,
            agent_id: None,
            session_id: None,
            now: Some(1000 * 3_600_000_000_000),
            limit: 10,
        };
        let config = MemoryRetrievalConfig::default();
        let result =
            MemoryRetrievalPipeline::retrieve(&candidates, &query, &config, None, None, None);
        // Should still return results via temporal + access signals
        assert!(!result.is_empty());
        // Most recent should rank highest (temporal decay)
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn test_memory_retrieval_with_embeddings() {
        let mut m1 = make_memory(1, MemoryTier::Episodic, 10);
        m1.summary_embedding = vec![1.0, 0.0, 0.0];
        let mut m2 = make_memory(2, MemoryTier::Episodic, 10);
        m2.summary_embedding = vec![0.0, 1.0, 0.0];

        let candidates = vec![m1, m2];
        let query = MemoryRetrievalQuery {
            query_text: String::new(),
            query_embedding: vec![1.0, 0.0, 0.0], // Matches m1
            context: None,
            anchor_node: None,
            agent_id: None,
            session_id: None,
            now: Some(1000 * 3_600_000_000_000),
            limit: 10,
        };
        let config = MemoryRetrievalConfig::default();
        let result =
            MemoryRetrievalPipeline::retrieve(&candidates, &query, &config, None, None, None);
        assert!(!result.is_empty());
        // m1 should rank first due to perfect cosine match
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn test_memory_retrieval_bm25() {
        let mut m1 = make_memory(1, MemoryTier::Episodic, 10);
        m1.summary = "authentication login failed".to_string();
        let mut m2 = make_memory(2, MemoryTier::Episodic, 10);
        m2.summary = "database migration completed".to_string();

        // Build BM25 index keyed by memory_id
        let mut bm25 = Bm25Index::new();
        bm25.index_document(
            1,
            &format!("{} {} {}", m1.summary, m1.takeaway, m1.causal_note),
        );
        bm25.index_document(
            2,
            &format!("{} {} {}", m2.summary, m2.takeaway, m2.causal_note),
        );

        let candidates = vec![m1, m2];
        let query = MemoryRetrievalQuery {
            query_text: "authentication login".to_string(),
            query_embedding: Vec::new(),
            context: None,
            anchor_node: None,
            agent_id: None,
            session_id: None,
            now: Some(1000 * 3_600_000_000_000),
            limit: 10,
        };
        let config = MemoryRetrievalConfig::default();
        let result = MemoryRetrievalPipeline::retrieve(
            &candidates,
            &query,
            &config,
            Some(&bm25),
            None,
            None,
        );
        assert!(!result.is_empty());
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn test_tier_boost() {
        // Create memories at the same age so temporal is equal
        let m1 = make_memory(1, MemoryTier::Episodic, 10);
        let m2 = make_memory(2, MemoryTier::Schema, 10);
        let m3 = make_memory(3, MemoryTier::Semantic, 10);

        let candidates = vec![m1, m2, m3];
        let query = MemoryRetrievalQuery {
            query_text: String::new(),
            query_embedding: Vec::new(),
            context: None,
            anchor_node: None,
            agent_id: None,
            session_id: None,
            now: Some(1000 * 3_600_000_000_000),
            limit: 10,
        };
        let config = MemoryRetrievalConfig::default();
        let result =
            MemoryRetrievalPipeline::retrieve(&candidates, &query, &config, None, None, None);

        // Schema should be boosted highest
        assert_eq!(result[0].0, 2, "Schema memory should rank first");
        assert_eq!(result[1].0, 3, "Semantic memory should rank second");
    }
}
