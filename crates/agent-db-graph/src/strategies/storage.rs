// crates/agent-db-graph/src/strategies/storage.rs
//
// Strategy storage, removal, pruning, merging, and loading from persistence.

use crate::episodes::EpisodeId;
use agent_db_core::types::{current_timestamp, AgentId};
use std::collections::HashSet;

use super::extractor::StrategyExtractor;
use super::types::*;

impl StrategyExtractor {
    pub(crate) fn store_strategy(
        &mut self,
        mut strategy: Strategy,
        context_patterns: &[ContextPattern],
        goal_bucket_id: u64,
    ) -> StrategyId {
        let signature = strategy
            .metadata
            .get("strategy_signature")
            .cloned()
            .unwrap_or_else(|| self.compute_strategy_signature(&strategy));

        if let Some(existing_id) = self.strategy_signature_index.get(&signature).copied() {
            if let Some(existing) = self.strategies.get_mut(&existing_id) {
                let new_support = existing.support_count + strategy.support_count;
                existing.support_count = new_support;
                existing.success_count += strategy.success_count;
                existing.failure_count += strategy.failure_count;
                existing.last_used = current_timestamp();
                existing
                    .source_episodes
                    .append(&mut strategy.source_episodes);
                existing
                    .source_outcomes
                    .append(&mut strategy.source_outcomes);

                let expected_success = (existing.success_count as f32 + self.config.alpha)
                    / (existing.support_count as f32 + self.config.alpha + self.config.beta);
                existing.expected_success = expected_success;
                existing.expected_value = match existing.strategy_type {
                    StrategyType::Positive => expected_success,
                    StrategyType::Constraint => -expected_success,
                };
                existing.confidence = 1.0 - (-((existing.support_count as f32) / 3.0)).exp();

                if existing.reasoning_steps.is_empty() && !strategy.reasoning_steps.is_empty() {
                    existing.reasoning_steps = strategy.reasoning_steps;
                }

                return existing_id;
            }
        }

        // Fuzzy dedup: check goal bucket peers for Jaccard similarity on action_hint
        if let Some(bucket_ids) = self.goal_bucket_index.get(&goal_bucket_id) {
            let mut best_peer: Option<StrategyId> = None;
            let mut best_jaccard: f32 = 0.0;
            for &peer_id in bucket_ids {
                if let Some(peer) = self.strategies.get(&peer_id) {
                    if peer.strategy_type != strategy.strategy_type {
                        continue;
                    }
                    if peer.agent_id != strategy.agent_id {
                        continue;
                    }
                    let jaccard = word_jaccard(&peer.action_hint, &strategy.action_hint);
                    if jaccard >= 0.60 && jaccard > best_jaccard {
                        best_jaccard = jaccard;
                        best_peer = Some(peer_id);
                    }
                }
            }
            if let Some(peer_id) = best_peer {
                if let Some(existing) = self.strategies.get_mut(&peer_id) {
                    existing.support_count += strategy.support_count;
                    existing.success_count += strategy.success_count;
                    existing.failure_count += strategy.failure_count;
                    existing.last_used = current_timestamp();
                    existing
                        .source_episodes
                        .append(&mut strategy.source_episodes);
                    existing
                        .source_outcomes
                        .append(&mut strategy.source_outcomes);

                    let expected_success = (existing.success_count as f32 + self.config.alpha)
                        / (existing.support_count as f32 + self.config.alpha + self.config.beta);
                    existing.expected_success = expected_success;
                    existing.expected_value = match existing.strategy_type {
                        StrategyType::Positive => expected_success,
                        StrategyType::Constraint => -expected_success,
                    };
                    existing.confidence = 1.0 - (-((existing.support_count as f32) / 3.0)).exp();

                    if existing.reasoning_steps.is_empty() && !strategy.reasoning_steps.is_empty() {
                        existing.reasoning_steps = strategy.reasoning_steps;
                    }

                    tracing::debug!(
                        "Fuzzy dedup merged into peer={} jaccard={:.3}",
                        peer_id,
                        best_jaccard
                    );
                    return peer_id;
                }
            }
        }

        let strategy_id = strategy.id;
        self.strategy_signature_index.insert(signature, strategy_id);
        let agent_id = strategy.agent_id;
        self.strategies.insert(strategy_id, strategy);

        self.agent_strategies
            .entry(agent_id)
            .or_default()
            .push(strategy_id);

        for pattern in context_patterns {
            if let Some(context_hash) = self.pattern_to_hash(pattern) {
                self.context_index
                    .entry(context_hash)
                    .or_default()
                    .push(strategy_id);
            }
        }

        self.goal_bucket_index
            .entry(goal_bucket_id)
            .or_default()
            .push(strategy_id);

        if let Some(sig) = self
            .strategies
            .get(&strategy_id)
            .and_then(|s| s.metadata.get("behavior_signature"))
            .cloned()
        {
            self.behavior_index
                .entry(sig)
                .or_default()
                .push(strategy_id);
        }

        strategy_id
    }

    /// Remove a strategy and clean all indexes.
    pub(crate) fn remove_strategy(&mut self, id: StrategyId) {
        if let Some(strategy) = self.strategies.remove(&id) {
            if let Some(ids) = self.agent_strategies.get_mut(&strategy.agent_id) {
                ids.retain(|sid| *sid != id);
            }
            if let Some(ids) = self.goal_bucket_index.get_mut(&strategy.goal_bucket_id) {
                ids.retain(|sid| *sid != id);
            }
            if let Some(ids) = self.behavior_index.get_mut(&strategy.behavior_signature) {
                ids.retain(|sid| *sid != id);
            }
            self.strategy_signature_index.retain(|_, v| *v != id);
            self.episode_index.retain(|_, v| *v != id);
        }
    }

    /// Prune weak / stale strategies that no longer contribute value.
    ///
    /// Returns the IDs of strategies that were removed.
    pub fn prune_weak_strategies(
        &mut self,
        min_confidence: f32,
        min_support: u32,
        max_stale_hours: f32,
    ) -> Vec<StrategyId> {
        let now = current_timestamp();
        let hour_ns = 3_600_000_000_000u64;

        let to_remove: Vec<StrategyId> = self
            .strategies
            .values()
            .filter(|s| {
                let hours_since_use = (now.saturating_sub(s.last_used) / hour_ns) as f32;

                // Remove if BOTH low confidence and low support
                let weak = s.confidence < min_confidence && s.support_count < min_support;
                // Remove if stale AND weak
                let stale_and_weak =
                    hours_since_use > max_stale_hours && s.support_count < min_support;

                weak || stale_and_weak
            })
            .map(|s| s.id)
            .collect();

        for id in &to_remove {
            if let Some(strategy) = self.strategies.remove(id) {
                // Clean indexes
                if let Some(ids) = self.agent_strategies.get_mut(&strategy.agent_id) {
                    ids.retain(|sid| sid != id);
                }
                if let Some(ids) = self.goal_bucket_index.get_mut(&strategy.goal_bucket_id) {
                    ids.retain(|sid| sid != id);
                }
                if let Some(ids) = self.behavior_index.get_mut(&strategy.behavior_signature) {
                    ids.retain(|sid| sid != id);
                }
                // Remove from signature index
                self.strategy_signature_index.retain(|_, v| v != id);
                // Remove from episode index
                self.episode_index.retain(|_, v| v != id);
            }
        }

        if !to_remove.is_empty() {
            // Clean accumulator maps: collect surviving (agent_id, goal_bucket_id) pairs
            let surviving_keys: std::collections::HashSet<(AgentId, u64)> = self
                .strategies
                .values()
                .map(|s| (s.agent_id, s.goal_bucket_id))
                .collect();

            // Remove entries from context_counts whose (agent_id, goal_bucket_id) is gone
            self.context_counts.retain(|&(agent_id, bucket_id, _), _| {
                surviving_keys.contains(&(agent_id, bucket_id))
            });

            // Remove entries from goal_bucket_counts whose key is gone
            self.goal_bucket_counts
                .retain(|key, _| surviving_keys.contains(key));

            // Remove entries from motif_stats_by_bucket whose key is gone
            self.motif_stats_by_bucket
                .retain(|key, _| surviving_keys.contains(key));

            // Remove entries from episode_cache_by_bucket whose key is gone
            self.episode_cache_by_bucket
                .retain(|key, _| surviving_keys.contains(key));

            // Cap episode_outcomes at 10,000 entries (trim oldest by episode_id)
            if self.episode_outcomes.len() > 10_000 {
                let mut ids: Vec<EpisodeId> = self.episode_outcomes.keys().copied().collect();
                ids.sort_unstable();
                let cutoff = ids[ids.len() - 10_000];
                self.episode_outcomes.retain(|&id, _| id >= cutoff);
            }

            // Clean context_index: remove pruned strategy IDs from vectors, drop empty vectors
            self.context_index.retain(|_, ids| {
                ids.retain(|id| self.strategies.contains_key(id));
                !ids.is_empty()
            });

            tracing::info!(
                "Strategy pruning removed {} weak/stale strategies",
                to_remove.len()
            );
        }

        to_remove
    }

    /// Merge near-duplicate strategies within the same goal bucket.
    ///
    /// Two strategies are "near-duplicates" if they share the same agent, goal bucket,
    /// strategy type, AND have overlapping behavior signatures in the behavior_index.
    /// The weaker strategy is merged into the stronger one (support counts transferred).
    ///
    /// Returns the number of strategies merged (removed).
    pub fn merge_similar_strategies(&mut self) -> usize {
        use rustc_hash::FxHashMap;

        let mut merged = 0usize;
        let mut to_merge: Vec<(StrategyId, StrategyId)> = Vec::new(); // (victim, survivor)

        // Group strategies by (agent_id, goal_bucket_id, strategy_type)
        let mut groups: FxHashMap<(AgentId, u64, StrategyType), Vec<StrategyId>> =
            FxHashMap::default();
        for s in self.strategies.values() {
            groups
                .entry((s.agent_id, s.goal_bucket_id, s.strategy_type))
                .or_default()
                .push(s.id);
        }

        for ids in groups.values() {
            if ids.len() < 2 {
                continue;
            }
            // Compare all pairs within the group
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let id_a = ids[i];
                    let id_b = ids[j];
                    let (a, b) = match (self.strategies.get(&id_a), self.strategies.get(&id_b)) {
                        (Some(a), Some(b)) => (a, b),
                        _ => continue,
                    };

                    // Word-level Jaccard on action_hint
                    let jaccard = word_jaccard(&a.action_hint, &b.action_hint);

                    if jaccard >= 0.70 {
                        // Merge weaker into stronger
                        let (victim, survivor) = if a.quality_score >= b.quality_score {
                            (id_b, id_a)
                        } else {
                            (id_a, id_b)
                        };
                        to_merge.push((victim, survivor));
                    }
                }
            }
        }

        // Deduplicate merge pairs (a victim can only be merged once)
        let mut already_merged = HashSet::new();
        for (victim, survivor) in to_merge {
            if already_merged.contains(&victim) || already_merged.contains(&survivor) {
                continue;
            }
            already_merged.insert(victim);

            // Transfer counts
            if let Some(victim_strategy) = self.strategies.remove(&victim) {
                if let Some(survivor_strategy) = self.strategies.get_mut(&survivor) {
                    survivor_strategy.support_count += victim_strategy.support_count;
                    survivor_strategy.success_count += victim_strategy.success_count;
                    survivor_strategy.failure_count += victim_strategy.failure_count;

                    // Record supersession
                    survivor_strategy.supersedes.push(victim);

                    // Recalculate confidence
                    let total = survivor_strategy.support_count as f32;
                    survivor_strategy.confidence = 1.0 - (-(total / 3.0)).exp();
                    survivor_strategy.expected_success =
                        (survivor_strategy.success_count as f32 + 1.0) / (total + 2.0);
                }

                // Clean indexes for victim
                if let Some(ids) = self.agent_strategies.get_mut(&victim_strategy.agent_id) {
                    ids.retain(|sid| *sid != victim);
                }
                if let Some(ids) = self
                    .goal_bucket_index
                    .get_mut(&victim_strategy.goal_bucket_id)
                {
                    ids.retain(|sid| *sid != victim);
                }
                if let Some(ids) = self
                    .behavior_index
                    .get_mut(&victim_strategy.behavior_signature)
                {
                    ids.retain(|sid| *sid != victim);
                }
                self.strategy_signature_index.retain(|_, v| *v != victim);
                self.episode_index.retain(|_, v| *v != victim);

                merged += 1;
            }
        }

        if merged > 0 {
            tracing::info!(
                "Strategy merge: combined {} near-duplicate strategies",
                merged
            );
        }
        merged
    }

    /// Insert a strategy loaded from persistent storage
    ///
    /// This is used to restore strategies from disk without going through
    /// the extraction process. Used during initialization.
    pub fn insert_loaded_strategy(&mut self, strategy: Strategy) -> Result<(), crate::GraphError> {
        let strategy_id = strategy.id;
        let agent_id = strategy.agent_id;
        let goal_bucket_id = strategy.goal_bucket_id;
        let behavior_signature = strategy.behavior_signature.clone();
        let strategy_signature = strategy
            .metadata
            .get("strategy_signature")
            .cloned()
            .unwrap_or_else(|| self.compute_strategy_signature(&strategy));

        // Update next_strategy_id if needed
        if strategy_id >= self.next_strategy_id {
            self.next_strategy_id = strategy_id + 1;
        }

        // Store strategy
        self.strategies.insert(strategy_id, strategy);

        // Index by agent
        self.agent_strategies
            .entry(agent_id)
            .or_default()
            .push(strategy_id);

        // Index by goal bucket
        self.goal_bucket_index
            .entry(goal_bucket_id)
            .or_default()
            .push(strategy_id);

        // Index by behavior signature
        self.behavior_index
            .entry(behavior_signature)
            .or_default()
            .push(strategy_id);

        // Index by strategy signature for dedup
        self.strategy_signature_index
            .insert(strategy_signature, strategy_id);

        Ok(())
    }
}
