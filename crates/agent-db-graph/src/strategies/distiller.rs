// crates/agent-db-graph/src/strategies/distiller.rs
//
// Contrastive distiller, motif extraction, holdout validation, and drift detection.

use crate::episodes::EpisodeOutcome;
use agent_db_core::types::{current_timestamp, AgentId};
use agent_db_events::core::{Event, EventType};
use std::collections::{HashMap, HashSet};

use super::extractor::StrategyExtractor;
use super::types::*;

impl StrategyExtractor {
    pub(crate) fn should_distill(&self, bucket_count: u32) -> bool {
        bucket_count.is_multiple_of(self.config.distill_every)
    }

    pub(crate) fn update_motif_stats(
        &mut self,
        agent_id: AgentId,
        goal_bucket_id: u64,
        outcome: EpisodeOutcome,
        events: &[Event],
    ) {
        let motifs = self.extract_motifs(events);
        let bucket_stats = self
            .motif_stats_by_bucket
            .entry((agent_id, goal_bucket_id))
            .or_default();

        for motif in motifs.iter() {
            let stats = bucket_stats.entry(motif.clone()).or_default();
            match outcome {
                EpisodeOutcome::Success | EpisodeOutcome::Partial => stats.success_count += 1,
                EpisodeOutcome::Failure | EpisodeOutcome::Interrupted => stats.failure_count += 1,
            }
        }

        let cache = self
            .episode_cache_by_bucket
            .entry((agent_id, goal_bucket_id))
            .or_default();
        cache.push(EpisodeMotifRecord { outcome, motifs });
        if cache.len() > self.config.cache_max {
            cache.remove(0);
        }
    }

    pub(crate) fn run_contrastive_distiller(&mut self, agent_id: AgentId, goal_bucket_id: u64) {
        let bucket_stats = match self.motif_stats_by_bucket.get(&(agent_id, goal_bucket_id)) {
            Some(stats) => stats.clone(),
            None => return,
        };
        let cache = match self
            .episode_cache_by_bucket
            .get(&(agent_id, goal_bucket_id))
        {
            Some(records) => records.clone(),
            None => return,
        };

        let (success_total, failure_total) = self.count_outcomes(&cache);
        if success_total == 0 && failure_total == 0 {
            return;
        }

        let baseline_success = (success_total as f32 + self.config.alpha)
            / (success_total as f32 + failure_total as f32 + self.config.alpha + self.config.beta);
        let baseline_failure = 1.0 - baseline_success;

        for (motif, stats) in bucket_stats {
            let s = stats.success_count as f32;
            let f = stats.failure_count as f32;
            let success_total_f = success_total as f32;
            let failure_total_f = failure_total as f32;

            let p_s =
                (s + self.config.alpha) / (success_total_f + self.config.alpha + self.config.beta);
            let p_f =
                (f + self.config.alpha) / (failure_total_f + self.config.alpha + self.config.beta);

            let lift = Self::log_odds(p_s) - Self::log_odds(p_f);
            let uplift = p_s - baseline_success;
            let failure_uplift = p_f - baseline_failure;

            let strategy_type = if lift >= self.config.min_lift && uplift >= self.config.min_uplift
            {
                StrategyType::Positive
            } else if lift <= -self.config.min_lift && failure_uplift >= self.config.min_uplift {
                StrategyType::Constraint
            } else {
                continue;
            };

            let support = match strategy_type {
                StrategyType::Positive => stats.success_count,
                StrategyType::Constraint => stats.failure_count,
            };

            if strategy_type == StrategyType::Positive && support < self.config.min_support_success
            {
                continue;
            }
            if strategy_type == StrategyType::Constraint
                && support < self.config.min_support_failure
            {
                continue;
            }

            if !self.validate_candidate(&cache, &motif, strategy_type, baseline_success) {
                continue;
            }

            let precondition = format!("goal_bucket={} motif={}", goal_bucket_id, motif);
            let action_hint = match strategy_type {
                StrategyType::Positive => format!("prefer motif: {}", motif),
                StrategyType::Constraint => format!("avoid motif: {}", motif),
            };
            let expected_success = p_s;
            let expected_value = match strategy_type {
                StrategyType::Positive => uplift,
                StrategyType::Constraint => -failure_uplift,
            };
            let confidence = 1.0 - (-((support as f32) / 3.0)).exp();

            let strategy_id = self.next_strategy_id;
            self.next_strategy_id += 1;

            // Build summary for distiller-generated strategy
            let distiller_summary = match strategy_type {
                StrategyType::Positive => format!(
                    "DO this when applicable. Prefer motif pattern '{}' in goal_bucket {}. Success rate {:.0}%, uplift {:.0}%, supported by {} episodes. Confidence {:.0}%.",
                    motif, goal_bucket_id, p_s * 100.0, uplift * 100.0, support, confidence * 100.0
                ),
                StrategyType::Constraint => format!(
                    "AVOID this pattern. Motif '{}' in goal_bucket {} correlates with failure. Failure uplift {:.0}%, supported by {} episodes. Confidence {:.0}%.",
                    motif, goal_bucket_id, failure_uplift * 100.0, support, confidence * 100.0
                ),
            };

            // Generate 10x fields for distiller-generated strategies
            let distiller_when_to_use = match strategy_type {
                StrategyType::Positive => format!(
                    "Use when goal bucket is {} and the motif '{}' is applicable. Best when success rate > {:.0}%.",
                    goal_bucket_id, motif, p_s * 100.0
                ),
                StrategyType::Constraint => format!(
                    "Avoid when goal bucket is {} and the motif '{}' appears. Failure correlation is strong ({:.0}%).",
                    goal_bucket_id, motif, failure_uplift * 100.0
                ),
            };
            let distiller_when_not_to_use = match strategy_type {
                StrategyType::Positive => format!(
                    "Do not use when context significantly differs from goal_bucket {} or when contradictions were observed.",
                    goal_bucket_id
                ),
                StrategyType::Constraint => format!(
                    "Safe to ignore when success rate in similar contexts is already > {:.0}% without this motif.",
                    p_s * 100.0
                ),
            };

            let mut strategy = Strategy {
                id: strategy_id,
                name: match strategy_type {
                    StrategyType::Positive => format!("strategy_{}_motif", goal_bucket_id),
                    StrategyType::Constraint => format!("constraint_{}_motif", goal_bucket_id),
                },
                summary: distiller_summary,
                when_to_use: distiller_when_to_use,
                when_not_to_use: distiller_when_not_to_use,
                failure_modes: Vec::new(),
                playbook: Vec::new(), // Distiller strategies are motif-level, no step playbook
                counterfactual: String::new(),
                supersedes: Vec::new(),
                applicable_domains: Vec::new(),
                lineage_depth: 0,
                summary_embedding: Vec::new(),
                agent_id,
                reasoning_steps: Vec::new(),
                context_patterns: Vec::new(),
                success_indicators: Vec::new(),
                failure_patterns: vec![motif.clone()],
                quality_score: (expected_success).min(1.0),
                success_count: if strategy_type == StrategyType::Positive {
                    support
                } else {
                    0
                },
                failure_count: if strategy_type == StrategyType::Constraint {
                    support
                } else {
                    0
                },
                support_count: support,
                strategy_type,
                precondition,
                action_hint,
                expected_success,
                expected_cost: 1.0,
                expected_value,
                confidence,
                contradictions: Vec::new(),
                goal_bucket_id,
                behavior_signature: motif.clone(),
                source_episodes: Vec::new(),
                created_at: current_timestamp(),
                last_used: current_timestamp(),
                metadata: HashMap::new(),
                self_judged_quality: None,
                source_outcomes: Vec::new(),
                version: 1,
                parent_strategy: None,
            };

            strategy.metadata.insert(
                "strategy_signature".to_string(),
                self.compute_strategy_signature(&strategy),
            );
            strategy
                .metadata
                .insert("behavior_signature".to_string(), motif.clone());
            strategy
                .metadata
                .insert("goal_bucket_id".to_string(), goal_bucket_id.to_string());
            strategy.metadata.insert(
                "strategy_type".to_string(),
                format!("{:?}", strategy.strategy_type),
            );

            let _ = self.store_strategy(strategy, &[], goal_bucket_id);
        }
    }

    pub(crate) fn validate_candidate(
        &self,
        cache: &[EpisodeMotifRecord],
        motif: &str,
        strategy_type: StrategyType,
        baseline_success: f32,
    ) -> bool {
        if cache.len() < self.config.holdout_size {
            return false;
        }

        let holdout = &cache[cache.len().saturating_sub(self.config.holdout_size)..];
        let mut matches = Vec::new();
        for record in holdout {
            if record.motifs.contains(motif) {
                matches.push(record.outcome.clone());
            }
        }

        if matches.len() < self.config.min_holdout_coverage as usize {
            return false;
        }

        let success_matches = matches
            .iter()
            .filter(|o| matches!(o, EpisodeOutcome::Success))
            .count();
        let failure_matches = matches
            .iter()
            .filter(|o| matches!(o, EpisodeOutcome::Failure))
            .count();

        let precision = success_matches as f32 / matches.len().max(1) as f32;
        let failure_rate = failure_matches as f32 / matches.len().max(1) as f32;

        let baseline_failure = 1.0 - baseline_success;
        let passes_precision = match strategy_type {
            StrategyType::Positive => precision >= baseline_success + self.config.min_uplift,
            StrategyType::Constraint => failure_rate >= baseline_failure + self.config.min_uplift,
        };

        if !passes_precision {
            return false;
        }

        let mid = matches.len() / 2;
        if mid > 0 {
            let (first, second) = matches.split_at(mid);
            let first_success = first
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Success))
                .count();
            let second_success = second
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Success))
                .count();
            let first_failure = first
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Failure))
                .count();
            let second_failure = second
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Failure))
                .count();

            match strategy_type {
                StrategyType::Positive => {
                    let first_rate = first_success as f32 / first.len().max(1) as f32;
                    let second_rate = second_success as f32 / second.len().max(1) as f32;
                    if first_rate - second_rate > self.config.drift_max_drop {
                        return false;
                    }
                },
                StrategyType::Constraint => {
                    let first_rate = first_failure as f32 / first.len().max(1) as f32;
                    let second_rate = second_failure as f32 / second.len().max(1) as f32;
                    if first_rate - second_rate > self.config.drift_max_drop {
                        return false;
                    }
                },
            }
        }

        true
    }

    pub(crate) fn extract_motifs(&self, events: &[Event]) -> HashSet<String> {
        let mut motifs = HashSet::new();
        let tokens = self.build_behavior_skeleton(events);

        for i in 0..tokens.len().saturating_sub(1) {
            let left = tokens[i].clone();
            let right = tokens[i + 1].clone();
            motifs.insert(Self::motif_key(
                MotifClass::Transition,
                format!("{}->{}", left, right),
            ));
        }

        let anchors = self.find_anchor_indices(events);
        for anchor in anchors {
            let start = anchor.saturating_sub(self.config.motif_window_k);
            let end = (anchor + self.config.motif_window_k).min(tokens.len().saturating_sub(1));
            let window = tokens[start..=end].join(">");
            motifs.insert(Self::motif_key(MotifClass::Anchor, window));
        }

        let action_tokens = self.build_action_tokens(events);
        for n in 3..=6 {
            if action_tokens.len() < n {
                continue;
            }
            for i in 0..=action_tokens.len() - n {
                let seq = action_tokens[i..i + n].join(">");
                motifs.insert(Self::motif_key(MotifClass::Macro, seq));
            }
        }

        motifs
    }

    pub(crate) fn find_anchor_indices(&self, events: &[Event]) -> Vec<usize> {
        let mut anchors = Vec::new();
        for (idx, event) in events.iter().enumerate() {
            match &event.event_type {
                EventType::Action {
                    action_name,
                    outcome,
                    ..
                } => {
                    if action_name == "user_feedback" {
                        anchors.push(idx);
                    }
                    match outcome {
                        agent_db_events::core::ActionOutcome::Success { .. } => anchors.push(idx),
                        agent_db_events::core::ActionOutcome::Failure { .. } => anchors.push(idx),
                        _ => {},
                    }
                },
                EventType::Observation { data, .. } => {
                    let text = data.to_string().to_lowercase();
                    if text.contains("error")
                        || text.contains("failed")
                        || text.contains("exception")
                    {
                        anchors.push(idx);
                    }
                },
                _ => {},
            }
        }
        anchors
    }

    pub(crate) fn build_action_tokens(&self, events: &[Event]) -> Vec<String> {
        let mut tokens = Vec::new();
        for event in events {
            if let EventType::Action { action_name, .. } = &event.event_type {
                let tool = self
                    .extract_tool_from_metadata(event)
                    .map(|t| format!(":{}", t))
                    .unwrap_or_default();
                tokens.push(format!("{}{}", action_name, tool));
            }
        }
        tokens
    }

    pub(crate) fn motif_key(class: MotifClass, token: String) -> String {
        format!("{:?}::{}", class, token)
    }

    pub(crate) fn count_outcomes(&self, records: &[EpisodeMotifRecord]) -> (u32, u32) {
        let mut success = 0;
        let mut failure = 0;
        for record in records {
            match record.outcome {
                EpisodeOutcome::Success | EpisodeOutcome::Partial => success += 1,
                EpisodeOutcome::Failure | EpisodeOutcome::Interrupted => failure += 1,
            }
        }
        (success, failure)
    }

    pub(crate) fn log_odds(p: f32) -> f32 {
        let clamped = p.clamp(0.001, 0.999);
        (clamped / (1.0 - clamped)).ln()
    }
}
