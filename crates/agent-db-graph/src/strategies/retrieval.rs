// crates/agent-db-graph/src/strategies/retrieval.rs
//
// Strategy retrieval, similarity search, outcome correction, and statistics.

use crate::episodes::{EpisodeId, EpisodeOutcome};
use agent_db_core::types::{AgentId, ContextHash};
use std::collections::HashSet;

use super::extractor::StrategyExtractor;
use super::types::*;

impl StrategyExtractor {
    /// Retrieve strategies applicable to a context
    pub fn get_strategies_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
    ) -> Vec<&Strategy> {
        self.context_index
            .get(&context_hash)
            .map(|ids| {
                let mut strategies: Vec<&Strategy> = ids
                    .iter()
                    .filter_map(|id| self.strategies.get(id))
                    .collect();

                // Sort by quality score
                strategies.sort_by(|a, b| {
                    b.quality_score
                        .partial_cmp(&a.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                strategies.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn apply_episode_outcome_correction(
        &mut self,
        episode_id: EpisodeId,
        strategy_id: StrategyId,
        new_outcome: &EpisodeOutcome,
    ) {
        let prev_outcome = self.episode_outcomes.get(&episode_id).cloned();
        if prev_outcome.as_ref() == Some(new_outcome) {
            return;
        }

        let Some(strategy) = self.strategies.get_mut(&strategy_id) else {
            return;
        };

        let (prev_success, prev_failure) = outcome_to_counts(prev_outcome.as_ref());
        let (new_success, new_failure) = outcome_to_counts(Some(new_outcome));

        if prev_success > 0 {
            strategy.success_count = strategy.success_count.saturating_sub(prev_success);
        }
        if prev_failure > 0 {
            strategy.failure_count = strategy.failure_count.saturating_sub(prev_failure);
        }
        strategy.success_count = strategy.success_count.saturating_add(new_success);
        strategy.failure_count = strategy.failure_count.saturating_add(new_failure);

        let total = strategy.success_count + strategy.failure_count;
        if total > 0 {
            strategy.quality_score = strategy.success_count as f32 / total as f32;
        }
        strategy.support_count = total;
        strategy.expected_success = (strategy.success_count as f32 + 1.0) / (total as f32 + 2.0);
        strategy.confidence = 1.0 - (-((total as f32) / 3.0)).exp();

        self.episode_outcomes
            .insert(episode_id, new_outcome.clone());

        tracing::info!(
            "Strategy updated from episode correction strategy_id={} episode_id={} success_count={} failure_count={}",
            strategy_id,
            episode_id,
            strategy.success_count,
            strategy.failure_count
        );
    }

    /// Find strategies similar to a query signature
    pub fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)> {
        tracing::info!(
            "Strategy similarity query goals={} tools={} results={} context_hash={:?} agent_id={:?} min_score={:.3} limit={}",
            query.goal_ids.len(),
            query.tool_names.len(),
            query.result_types.len(),
            query.context_hash,
            query.agent_id,
            query.min_score,
            query.limit
        );
        let goal_bucket_id = compute_goal_bucket_id_from_ids(&query.goal_ids);

        let candidate_ids: Vec<StrategyId> = if let Some(context_hash) = query.context_hash {
            self.context_index
                .get(&context_hash)
                .cloned()
                .unwrap_or_default()
        } else if goal_bucket_id != 0 {
            self.goal_bucket_index
                .get(&goal_bucket_id)
                .cloned()
                .unwrap_or_default()
        } else {
            self.strategies.keys().copied().collect()
        };

        let query_goals: HashSet<u64> = query.goal_ids.iter().copied().collect();
        let query_tools: HashSet<String> = query.tool_names.iter().cloned().collect();
        let query_results: HashSet<String> = query.result_types.iter().cloned().collect();

        let goal_weight = if query_goals.is_empty() { 0.0 } else { 0.5 };
        let tool_weight = if query_tools.is_empty() { 0.0 } else { 0.3 };
        let result_weight = if query_results.is_empty() { 0.0 } else { 0.2 };
        let weight_sum = goal_weight + tool_weight + result_weight;

        let mut scored: Vec<(Strategy, f32)> = candidate_ids
            .into_iter()
            .filter_map(|id| self.strategies.get(&id))
            .filter(|strategy| {
                if let Some(agent_id) = query.agent_id {
                    strategy.agent_id == agent_id
                } else {
                    true
                }
            })
            .map(|strategy| {
                let (goal_ids, tool_names, result_types) = self.parse_graph_signature(strategy);
                let score = if weight_sum == 0.0 {
                    0.0
                } else {
                    let goals_score = Self::jaccard_u64(&query_goals, &goal_ids);
                    let tools_score = Self::jaccard_string(&query_tools, &tool_names);
                    let results_score = Self::jaccard_string(&query_results, &result_types);
                    (goals_score * goal_weight
                        + tools_score * tool_weight
                        + results_score * result_weight)
                        / weight_sum
                };
                (strategy.clone(), score)
            })
            .filter(|(_, score)| *score >= query.min_score)
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        {
            let result = scored.into_iter().take(query.limit).collect::<Vec<_>>();
            tracing::info!("Strategy similarity results={}", result.len());
            result
        }
    }

    /// Get a strategy by ID
    pub fn get_strategy(&self, strategy_id: StrategyId) -> Option<&Strategy> {
        self.strategies.get(&strategy_id)
    }

    pub(crate) fn parse_graph_signature(
        &self,
        strategy: &Strategy,
    ) -> (HashSet<u64>, HashSet<String>, HashSet<String>) {
        let goal_ids = strategy
            .metadata
            .get("goal_ids")
            .and_then(|value| serde_json::from_str::<Vec<u64>>(value).ok())
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        let tool_names = strategy
            .metadata
            .get("tool_names")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        let result_types = strategy
            .metadata
            .get("result_types")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        (goal_ids, tool_names, result_types)
    }

    /// Get all strategies for an agent
    pub fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<&Strategy> {
        self.agent_strategies
            .get(&agent_id)
            .map(|ids| {
                let mut strategies: Vec<&Strategy> = ids
                    .iter()
                    .filter_map(|id| self.strategies.get(id))
                    .collect();

                // Sort by quality score
                strategies.sort_by(|a, b| {
                    b.quality_score
                        .partial_cmp(&a.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                strategies.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    /// Get strategy statistics
    pub fn get_stats(&self) -> StrategyStats {
        StrategyStats {
            total_strategies: self.strategies.len(),
            high_quality_strategies: self
                .strategies
                .values()
                .filter(|s| s.quality_score > 0.8)
                .count(),
            agents_with_strategies: self.agent_strategies.len(),
            average_quality: if !self.strategies.is_empty() {
                self.strategies
                    .values()
                    .map(|s| s.quality_score)
                    .sum::<f32>()
                    / self.strategies.len() as f32
            } else {
                0.0
            },
        }
    }

    /// List all strategies (used by maintenance / pruning).
    pub fn list_all_strategies(&self) -> Vec<&Strategy> {
        self.strategies.values().collect()
    }
}
