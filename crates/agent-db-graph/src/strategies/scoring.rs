// crates/agent-db-graph/src/strategies/scoring.rs
//
// Eligibility scoring, redundancy estimation, calibrated metrics, and Q-value updates.

use crate::episodes::{Episode, EpisodeOutcome};
use crate::GraphResult;
use agent_db_core::types::current_timestamp;
use agent_db_events::core::Event;

use super::extractor::StrategyExtractor;
use super::types::*;

impl StrategyExtractor {
    pub(crate) fn calculate_eligibility_score(
        &self,
        episode: &Episode,
        behavior_signature: &str,
        action_hint: &str,
        goal_bucket_id: u64,
        events: &[Event],
    ) -> f32 {
        let context_count = *self
            .context_counts
            .get(&(episode.agent_id, goal_bucket_id, episode.context_signature))
            .unwrap_or(&0);
        let novelty = 1.0 / (1.0 + context_count as f32);

        let outcome_utility = match episode
            .outcome
            .clone()
            .unwrap_or(EpisodeOutcome::Interrupted)
        {
            EpisodeOutcome::Success => 1.0,
            EpisodeOutcome::Partial => 0.7,
            EpisodeOutcome::Failure => 0.8,
            EpisodeOutcome::Interrupted => 0.4,
        };

        let difficulty = ((events.len() as f32) / 10.0).min(1.0);

        let bucket_count = *self
            .goal_bucket_counts
            .get(&(episode.agent_id, goal_bucket_id))
            .unwrap_or(&0);
        let reuse_potential = if bucket_count == 0 {
            0.0
        } else {
            (bucket_count as f32 / 10.0).min(1.0)
        };

        let redundancy = self.estimate_redundancy(goal_bucket_id, behavior_signature, action_hint);

        let score = self.config.w_novelty * novelty
            + self.config.w_outcome_utility * outcome_utility
            + self.config.w_difficulty * difficulty
            + self.config.w_reuse_potential * reuse_potential
            - self.config.w_redundancy * redundancy;

        score.clamp(0.0, 1.0)
    }

    /// Estimate how redundant a candidate strategy is relative to existing bucket peers.
    ///
    /// Returns a continuous 0.0–1.0 score: 1.0 for exact behavior_signature match,
    /// otherwise the highest word-level Jaccard similarity on action_hint found.
    pub(crate) fn estimate_redundancy(
        &self,
        goal_bucket_id: u64,
        behavior_signature: &str,
        action_hint: &str,
    ) -> f32 {
        let mut max_similarity: f32 = 0.0;
        if let Some(ids) = self.goal_bucket_index.get(&goal_bucket_id) {
            for id in ids {
                if let Some(strategy) = self.strategies.get(id) {
                    // Exact signature match → maximum redundancy
                    if strategy.behavior_signature == behavior_signature {
                        return 1.0;
                    }
                    // Fuzzy match on action_hint
                    let jaccard = word_jaccard(&strategy.action_hint, action_hint);
                    if jaccard > max_similarity {
                        max_similarity = jaccard;
                    }
                }
            }
        }
        max_similarity
    }

    pub(crate) fn derive_calibrated_metrics(
        &self,
        strategy_type: &StrategyType,
        outcome: &EpisodeOutcome,
        support_count: u32,
    ) -> (f32, f32, f32) {
        let expected_success = match outcome {
            EpisodeOutcome::Success => 0.8,
            EpisodeOutcome::Partial => 0.6,
            EpisodeOutcome::Failure => 0.4,
            EpisodeOutcome::Interrupted => 0.4,
        };
        let expected_value = match strategy_type {
            StrategyType::Positive => expected_success,
            StrategyType::Constraint => -expected_success,
        };
        let confidence = 1.0 - (-((support_count as f32) / 3.0)).exp();
        (expected_success, expected_value, confidence)
    }

    /// Update strategy based on new usage outcome.
    ///
    /// Keeps existing counters and raw ratio (`quality_score`) for backward compat.
    /// Adds EMA Q-value in metadata and piecewise-blended `confidence` that
    /// reflects both sample size and outcome quality.
    pub fn update_strategy_outcome(
        &mut self,
        strategy_id: StrategyId,
        success: bool,
    ) -> GraphResult<()> {
        if let Some(strategy) = self.strategies.get_mut(&strategy_id) {
            // Lossless counters (keep existing)
            if success {
                strategy.success_count += 1;
            } else {
                strategy.failure_count += 1;
            }

            // Raw win ratio (keep for backward compat)
            let total = strategy.success_count + strategy.failure_count;
            if total > 0 {
                strategy.quality_score = strategy.success_count as f32 / total as f32;
            }
            strategy.support_count = total;

            // Bayesian expected success (keep for small N)
            strategy.expected_success =
                (strategy.success_count as f32 + 1.0) / (total as f32 + 2.0);

            // Update EMA Q-value in metadata: Q = Q + α(r − Q)
            let r = if success { 1.0_f32 } else { 0.0 };
            let q_old: f32 = strategy
                .metadata
                .get(META_Q_VALUE)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5);
            let q_new = q_old + Q_ALPHA * (r - q_old);
            strategy
                .metadata
                .insert(META_Q_VALUE.to_string(), format!("{:.6}", q_new));

            // Store lifetime counters in metadata for audit
            let pos_key = if success {
                META_POSITIVE_OUTCOMES
            } else {
                META_NEGATIVE_OUTCOMES
            };
            let current_count: u32 = strategy
                .metadata
                .get(pos_key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            strategy
                .metadata
                .insert(pos_key.to_string(), (current_count + 1).to_string());

            // Piecewise score: Bayesian for small N, EMA for large N
            let piecewise_score = if total < Q_KICK_IN {
                strategy.expected_success // Bayesian
            } else {
                q_new // EMA Q-value
            };

            // Confidence = sample-size factor × piecewise outcome quality
            let sample_confidence = 1.0 - (-((total as f32) / 3.0)).exp();
            strategy.confidence = sample_confidence * piecewise_score;

            strategy.last_used = current_timestamp();
        }

        Ok(())
    }
}
