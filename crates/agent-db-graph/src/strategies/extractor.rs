// crates/agent-db-graph/src/strategies/extractor.rs
//
// StrategyExtractor struct definition and the main extract_from_episode pipeline.

use crate::episodes::{Episode, EpisodeId, EpisodeOutcome};
use crate::GraphResult;
use agent_db_core::types::{current_timestamp, AgentId, ContextHash};
use agent_db_events::core::Event;
use rustc_hash::FxHashMap;
use serde_json::json;
use std::collections::HashMap;

use super::synthesis::{
    build_playbook, synthesize_counterfactual, synthesize_failure_modes,
    synthesize_strategy_summary, synthesize_when_not_to_use, synthesize_when_to_use,
};
use super::types::*;

/// Strategy extraction engine
pub struct StrategyExtractor {
    /// All extracted strategies
    pub(crate) strategies: FxHashMap<StrategyId, Strategy>,

    /// Strategy index by agent
    pub(crate) agent_strategies: FxHashMap<AgentId, Vec<StrategyId>>,

    /// Strategy index by context hash
    pub(crate) context_index: FxHashMap<ContextHash, Vec<StrategyId>>,

    /// Strategy index by goal bucket
    pub(crate) goal_bucket_index: FxHashMap<u64, Vec<StrategyId>>,

    /// Strategy index by behavior signature
    pub(crate) behavior_index: HashMap<String, Vec<StrategyId>>,

    /// Context counts for novelty estimation
    pub(crate) context_counts: FxHashMap<(AgentId, u64, ContextHash), u32>,

    /// Goal bucket occurrence counts (per agent)
    pub(crate) goal_bucket_counts: FxHashMap<(AgentId, u64), u32>,

    /// Motif stats by goal bucket (per agent)
    pub(crate) motif_stats_by_bucket: FxHashMap<(AgentId, u64), HashMap<String, MotifStats>>,

    /// Episode cache for validation (per agent + goal bucket)
    pub(crate) episode_cache_by_bucket: FxHashMap<(AgentId, u64), Vec<EpisodeMotifRecord>>,

    /// Strategy signature index to prevent duplicates
    pub(crate) strategy_signature_index: HashMap<String, StrategyId>,

    /// Episode to strategy index (idempotency)
    pub(crate) episode_index: FxHashMap<EpisodeId, StrategyId>,

    /// Episode outcome tracking for corrections
    pub(crate) episode_outcomes: FxHashMap<EpisodeId, EpisodeOutcome>,

    /// Configuration
    pub(crate) config: StrategyExtractionConfig,

    /// Next strategy ID
    pub(crate) next_strategy_id: StrategyId,
}

impl StrategyExtractor {
    /// Create a new strategy extractor
    pub fn new(config: StrategyExtractionConfig) -> Self {
        Self {
            strategies: FxHashMap::default(),
            agent_strategies: FxHashMap::default(),
            context_index: FxHashMap::default(),
            goal_bucket_index: FxHashMap::default(),
            behavior_index: HashMap::new(),
            context_counts: FxHashMap::default(),
            goal_bucket_counts: FxHashMap::default(),
            motif_stats_by_bucket: FxHashMap::default(),
            episode_cache_by_bucket: FxHashMap::default(),
            strategy_signature_index: HashMap::new(),
            episode_index: FxHashMap::default(),
            episode_outcomes: FxHashMap::default(),
            config,
            next_strategy_id: 1,
        }
    }

    /// Extract strategies from a successful episode
    ///
    /// Analyzes the episode's events to identify reusable patterns and strategies
    pub fn extract_from_episode(
        &mut self,
        episode: &Episode,
        events: &[Event],
    ) -> GraphResult<Option<StrategyUpsert>> {
        if let Some(existing_id) = self.episode_index.get(&episode.id).copied() {
            let new_outcome = episode
                .outcome
                .clone()
                .unwrap_or(EpisodeOutcome::Interrupted);
            self.apply_episode_outcome_correction(episode.id, existing_id, &new_outcome);
            return Ok(Some(StrategyUpsert {
                id: existing_id,
                is_new: false,
            }));
        }

        // Early exit: trivial episodes with too few events
        if events.len() < self.config.min_episode_events {
            tracing::info!(
                "Strategy extraction rejected episode_id={} events={} min={}",
                episode.id,
                events.len(),
                self.config.min_episode_events
            );
            return Ok(None);
        }

        let outcome = episode
            .outcome
            .clone()
            .unwrap_or(EpisodeOutcome::Interrupted);

        // Defense-in-depth: skip low-signal outcomes early
        if matches!(
            outcome,
            EpisodeOutcome::Partial | EpisodeOutcome::Interrupted
        ) {
            tracing::info!(
                "Strategy extraction skipped episode_id={} outcome={:?} (low signal)",
                episode.id,
                outcome
            );
            return Ok(None);
        }

        let strategy_type = match outcome {
            EpisodeOutcome::Success => StrategyType::Positive,
            EpisodeOutcome::Failure => StrategyType::Constraint,
            EpisodeOutcome::Partial => StrategyType::Positive,
            EpisodeOutcome::Interrupted => StrategyType::Constraint,
        };

        let goal_bucket_id = self.derive_goal_bucket_id(episode);
        let behavior_signature = self.compute_behavior_signature(events);

        // Extract action_hint early so eligibility can use it for fuzzy redundancy
        let (precondition, action_hint, expected_cost) =
            self.extract_behavior_skeleton(events, &strategy_type, goal_bucket_id);

        let eligibility_score = self.calculate_eligibility_score(
            episode,
            &behavior_signature,
            &action_hint,
            goal_bucket_id,
            events,
        );
        if eligibility_score < self.config.eligibility_threshold {
            tracing::info!(
                "Strategy extraction rejected episode_id={} eligibility={:.3} min={:.3}",
                episode.id,
                eligibility_score,
                self.config.eligibility_threshold
            );
            return Ok(None);
        }

        // Per-bucket cap: if bucket is full, only allow if quality exceeds weakest
        let quality_with_prediction_early =
            episode.significance * (1.0 + episode.prediction_error * 0.3);
        if let Some(bucket_ids) = self.goal_bucket_index.get(&goal_bucket_id) {
            if bucket_ids.len() >= self.config.max_strategies_per_bucket {
                let weakest_quality = bucket_ids
                    .iter()
                    .filter_map(|id| self.strategies.get(id))
                    .map(|s| s.quality_score)
                    .fold(f32::INFINITY, f32::min);
                if quality_with_prediction_early <= weakest_quality {
                    tracing::info!(
                        "Strategy extraction rejected episode_id={} bucket full ({}) quality={:.3} <= weakest={:.3}",
                        episode.id,
                        bucket_ids.len(),
                        quality_with_prediction_early,
                        weakest_quality
                    );
                    return Ok(None);
                }
            }
        }

        // Extract reasoning traces from cognitive events
        let reasoning_steps = self.extract_reasoning_steps(events)?;

        if reasoning_steps.is_empty() {
            tracing::info!(
                "Strategy extraction note episode_id={} (no reasoning steps)",
                episode.id
            );
        }

        // Extract context patterns
        let context_patterns = self.extract_context_patterns(events)?;

        // Identify success indicators
        let success_indicators = self.identify_success_indicators(events)?;

        // Phase 1 Feature K: Extract failure patterns from failed events
        let failure_patterns = self.extract_failure_patterns(events, &episode.outcome)?;

        // Phase 1: Initialize quality with prediction-error weighting
        let quality_with_prediction = episode.significance * (1.0 + episode.prediction_error * 0.3);

        // Quality floor: reject low-quality strategies
        if quality_with_prediction < self.config.min_quality_score {
            tracing::info!(
                "Strategy extraction rejected episode_id={} quality={:.3} min={:.3}",
                episode.id,
                quality_with_prediction,
                self.config.min_quality_score
            );
            return Ok(None);
        }

        // Create strategy
        let strategy_id = self.next_strategy_id;
        self.next_strategy_id += 1;

        // precondition, action_hint, expected_cost already extracted above
        let (expected_success, expected_value, confidence) =
            self.derive_calibrated_metrics(&strategy_type, &outcome, events.len() as u32);
        // Generate natural language summary
        let summary = synthesize_strategy_summary(
            &strategy_type,
            &outcome,
            episode,
            events,
            &success_indicators,
            &failure_patterns,
        );

        // Generate 10x/100x fields
        let when_to_use = synthesize_when_to_use(&strategy_type, episode, events);
        let when_not_to_use = synthesize_when_not_to_use(&strategy_type, episode, events);
        let failure_mode_hints = synthesize_failure_modes(events);
        let playbook = build_playbook(events, &strategy_type);
        let counterfactual = synthesize_counterfactual(&outcome, events);

        // Detect applicable domains from goals + context
        let applicable_domains: Vec<String> = episode
            .context
            .active_goals
            .iter()
            .filter_map(|g| {
                if g.description.is_empty() {
                    None
                } else {
                    // Extract domain keyword from goal description
                    Some(
                        g.description
                            .split_whitespace()
                            .take(4)
                            .collect::<Vec<_>>()
                            .join(" "),
                    )
                }
            })
            .collect();

        let mut strategy = Strategy {
            id: strategy_id,
            name: match strategy_type {
                StrategyType::Positive => {
                    format!("strategy_{}_ep_{}", episode.agent_id, episode.id)
                },
                StrategyType::Constraint => {
                    format!("constraint_{}_ep_{}", episode.agent_id, episode.id)
                },
            },
            summary,
            when_to_use,
            when_not_to_use,
            failure_modes: failure_mode_hints,
            playbook,
            counterfactual,
            supersedes: Vec::new(),
            applicable_domains,
            lineage_depth: 0,
            summary_embedding: Vec::new(),
            agent_id: episode.agent_id,
            reasoning_steps,
            context_patterns: context_patterns.clone(),
            success_indicators,
            failure_patterns,
            quality_score: quality_with_prediction.min(1.0),
            success_count: if matches!(outcome, EpisodeOutcome::Success) {
                1
            } else {
                0
            },
            failure_count: if matches!(outcome, EpisodeOutcome::Failure) {
                1
            } else {
                0
            },
            support_count: 1,
            strategy_type,
            precondition,
            action_hint,
            expected_success,
            expected_cost,
            expected_value,
            confidence,
            contradictions: Vec::new(),
            goal_bucket_id,
            behavior_signature: behavior_signature.clone(),
            source_episodes: vec![episode.clone()],
            created_at: current_timestamp(),
            last_used: current_timestamp(),
            metadata: HashMap::new(),
            // Phase 1 fields
            self_judged_quality: episode.self_judged_quality,
            source_outcomes: vec![outcome.clone()],
            version: 1,
            parent_strategy: None,
        };

        let (goal_ids, tool_names, result_types) = self.extract_graph_signature(episode, events);
        strategy
            .metadata
            .insert("goal_ids".to_string(), json!(goal_ids).to_string());
        strategy
            .metadata
            .insert("tool_names".to_string(), json!(tool_names).to_string());
        strategy
            .metadata
            .insert("result_types".to_string(), json!(result_types).to_string());
        strategy
            .metadata
            .insert("goal_bucket_id".to_string(), goal_bucket_id.to_string());
        strategy
            .metadata
            .insert("behavior_signature".to_string(), behavior_signature.clone());
        strategy.metadata.insert(
            "strategy_type".to_string(),
            format!("{:?}", strategy.strategy_type),
        );
        let strategy_signature = self.compute_strategy_signature(&strategy);
        strategy
            .metadata
            .insert("strategy_signature".to_string(), strategy_signature.clone());

        let stored_id = self.store_strategy(strategy, &context_patterns, goal_bucket_id);
        self.episode_index.insert(episode.id, stored_id);
        self.episode_outcomes.insert(episode.id, outcome.clone());

        // Per-bucket cap enforcement: evict weakest if over limit after insertion
        if let Some(bucket_ids) = self.goal_bucket_index.get(&goal_bucket_id) {
            if bucket_ids.len() > self.config.max_strategies_per_bucket {
                // Find the weakest strategy in this bucket (excluding the one we just stored)
                let weakest = bucket_ids
                    .iter()
                    .filter(|&&id| id != stored_id)
                    .filter_map(|&id| self.strategies.get(&id).map(|s| (id, s.quality_score)))
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((victim_id, _)) = weakest {
                    self.remove_strategy(victim_id);
                    tracing::debug!(
                        "Per-bucket cap: evicted strategy {} from bucket {}",
                        victim_id,
                        goal_bucket_id
                    );
                }
            }
        }

        *self
            .context_counts
            .entry((episode.agent_id, goal_bucket_id, episode.context_signature))
            .or_insert(0) += 1;
        let bucket_count = {
            let bucket = self
                .goal_bucket_counts
                .entry((episode.agent_id, goal_bucket_id))
                .or_insert(0);
            *bucket += 1;
            *bucket
        };

        self.update_motif_stats(episode.agent_id, goal_bucket_id, outcome.clone(), events);
        if self.should_distill(bucket_count) {
            self.run_contrastive_distiller(episode.agent_id, goal_bucket_id);
        }

        tracing::info!(
            "Strategy stored id={} episode_id={} agent_id={} quality={:.3}",
            stored_id,
            episode.id,
            episode.agent_id,
            quality_with_prediction.min(1.0)
        );

        Ok(Some(StrategyUpsert {
            id: stored_id,
            is_new: true,
        }))
    }
}
