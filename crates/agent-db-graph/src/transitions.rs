//! Transition model (Markov/MDP layer) for procedural memory.

use std::collections::HashMap;

use crate::contracts::{AbstractTrace, AbstractTransition};
use crate::episodes::EpisodeId;

#[derive(Debug, Clone)]
pub struct TransitionModelConfig {
    pub prior_success: f32,
    pub prior_failure: f32,
}

impl Default for TransitionModelConfig {
    fn default() -> Self {
        Self {
            prior_success: 1.0,
            prior_failure: 3.0,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TransitionKey {
    state: String,
    action: String,
    next_state: String,
}

impl From<&AbstractTransition> for TransitionKey {
    fn from(value: &AbstractTransition) -> Self {
        Self {
            state: value.state.clone(),
            action: value.action.clone(),
            next_state: value.next_state.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransitionStats {
    pub count: u64,
    pub success_count: u64,
    pub failure_count: u64,
}

impl TransitionStats {
    pub fn posterior_success(&self, config: &TransitionModelConfig) -> f32 {
        let alpha = config.prior_success + self.success_count as f32;
        let beta = config.prior_failure + self.failure_count as f32;
        (alpha / (alpha + beta)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Default)]
pub struct TransitionModel {
    config: TransitionModelConfig,
    buckets: HashMap<u64, HashMap<TransitionKey, TransitionStats>>,
    episode_transitions: HashMap<EpisodeId, Vec<TransitionKey>>,
    episode_outcomes: HashMap<EpisodeId, bool>,
    episode_goal_bucket: HashMap<EpisodeId, u64>,
}

impl TransitionModel {
    pub fn new(config: TransitionModelConfig) -> Self {
        Self {
            config,
            buckets: HashMap::new(),
            episode_transitions: HashMap::new(),
            episode_outcomes: HashMap::new(),
            episode_goal_bucket: HashMap::new(),
        }
    }

    pub fn update_from_trace(
        &mut self,
        goal_bucket_id: u64,
        trace: &AbstractTrace,
        episode_id: EpisodeId,
        success: bool,
    ) {
        let previous_goal_bucket = self.episode_goal_bucket.get(&episode_id).copied();
        let previous_outcome = self.episode_outcomes.get(&episode_id).copied();

        if let (Some(prev_bucket), Some(prev_outcome), Some(prev_transitions)) = (
            previous_goal_bucket,
            previous_outcome,
            self.episode_transitions.get(&episode_id),
        ) {
            if prev_bucket == goal_bucket_id && prev_outcome == success {
                return;
            }
            if let Some(bucket) = self.buckets.get_mut(&prev_bucket) {
                for key in prev_transitions {
                    if let Some(stats) = bucket.get_mut(key) {
                        stats.count = stats.count.saturating_sub(1);
                        if prev_outcome {
                            stats.success_count = stats.success_count.saturating_sub(1);
                        } else {
                            stats.failure_count = stats.failure_count.saturating_sub(1);
                        }
                    }
                }
            }
        }

        let bucket = self.buckets.entry(goal_bucket_id).or_default();
        let keys: Vec<TransitionKey> = trace.transitions.iter().map(TransitionKey::from).collect();

        for key in &keys {
            let stats = bucket.entry(key.clone()).or_insert(TransitionStats {
                count: 0,
                success_count: 0,
                failure_count: 0,
            });
            stats.count += 1;
            if success {
                stats.success_count += 1;
            } else {
                stats.failure_count += 1;
            }
        }

        self.episode_transitions.insert(episode_id, keys);
        self.episode_outcomes.insert(episode_id, success);
        self.episode_goal_bucket.insert(episode_id, goal_bucket_id);
    }

    pub fn get_stats(
        &self,
        goal_bucket_id: u64,
        transition: &AbstractTransition,
    ) -> Option<TransitionStats> {
        self.buckets
            .get(&goal_bucket_id)
            .and_then(|bucket| bucket.get(&TransitionKey::from(transition)).cloned())
    }

    pub fn config(&self) -> &TransitionModelConfig {
        &self.config
    }
}
