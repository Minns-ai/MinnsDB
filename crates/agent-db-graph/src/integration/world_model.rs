//! World model feature extraction, training tuple assembly, and scoring wrappers.
//!
//! This module contains all world-model-specific logic, keeping the other
//! integration files minimally changed.

use agent_db_events::core::{ActionOutcome, Event, EventType};
use agent_db_world_model::{
    CriticReport, EventFeatures, MemoryFeatures, PolicyFeatures, PredictionErrorReport,
    StrategyFeatures, TrainingTuple,
};

use crate::episodes::{Episode, EpisodeOutcome};
use crate::memory::Memory;
use crate::strategies::Strategy;

use super::*;

// ─────────────────────────── FNV-1a hash ───────────────────────────

/// FNV-1a hash for string → u64 (deterministic, fast).
fn hash_str(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ─────────────────────────── Feature extractors ───────────────────────────

/// Extract EventFeatures from a raw Event (no episode context — for per-event scoring).
pub(super) fn extract_event_features_raw(event: &Event) -> EventFeatures {
    let (event_type_hash, action_name_hash, outcome_success, duration_ns) = match &event.event_type
    {
        EventType::Action {
            action_name,
            outcome,
            duration_ns,
            ..
        } => {
            let success = match outcome {
                ActionOutcome::Success { .. } => 1.0,
                ActionOutcome::Failure { .. } => 0.0,
                ActionOutcome::Partial { .. } => 0.5,
            };
            (
                hash_str("Action"),
                hash_str(action_name),
                success,
                *duration_ns as f64,
            )
        },
        EventType::Observation {
            observation_type, ..
        } => (
            hash_str("Observation"),
            hash_str(observation_type),
            0.5,
            0.0,
        ),
        EventType::Cognitive { .. } => (hash_str("Cognitive"), 0, 0.5, 0.0),
        EventType::Communication { .. } => (hash_str("Communication"), 0, 0.5, 0.0),
        EventType::Learning { .. } => (hash_str("Learning"), 0, 0.5, 0.0),
        EventType::Context { .. } => (hash_str("Context"), 0, 0.5, 0.0),
        EventType::Conversation { .. } => (hash_str("Conversation"), 0, 0.5, 0.0),
        EventType::CodeReview { .. } => (hash_str("CodeReview"), 0, 0.5, 0.0),
        EventType::CodeFile { .. } => (hash_str("CodeFile"), 0, 0.5, 0.0),
    };

    EventFeatures {
        event_type_hash,
        action_name_hash,
        context_fingerprint: event.context.fingerprint,
        outcome_success,
        significance: 0.5, // no episode-level significance available
        temporal_delta_ns: 0.0,
        duration_ns,
    }
}

/// Extract EventFeatures from an Event within an episode context.
///
/// Picks the most representative data from the episode:
/// - outcome_success from episode outcome
/// - significance from episode significance
/// - temporal_delta_ns = median inter-event gap (approximated as total / count)
/// - duration_ns = total episode duration
pub(super) fn extract_event_features(event: &Event, episode: &Episode) -> EventFeatures {
    let mut features = extract_event_features_raw(event);

    // Override with episode-level information
    features.outcome_success = match &episode.outcome {
        Some(EpisodeOutcome::Success) => 1.0,
        Some(EpisodeOutcome::Failure) => 0.0,
        Some(EpisodeOutcome::Partial) | Some(EpisodeOutcome::Interrupted) => 0.5,
        None => 0.5,
    };
    features.significance = episode.significance;

    // Total episode duration
    if let Some(end_ts) = episode.end_timestamp {
        features.duration_ns = end_ts.saturating_sub(episode.start_timestamp) as f64;
        // Approximate median inter-event gap as total / count
        let count = episode.events.len().max(1) as f64;
        features.temporal_delta_ns = features.duration_ns / count;
    }

    features
}

/// Extract MemoryFeatures from a Memory, or return defaults if None.
pub(super) fn extract_memory_features(
    memory: Option<&Memory>,
    episode: &Episode,
) -> MemoryFeatures {
    match memory {
        Some(m) => {
            let tier = match m.tier {
                crate::memory::MemoryTier::Episodic => 0,
                crate::memory::MemoryTier::Semantic => 1,
                crate::memory::MemoryTier::Schema => 2,
            };
            MemoryFeatures {
                tier,
                strength: m.strength,
                access_count: m.access_count,
                context_fingerprint: m.context.fingerprint,
                goal_bucket_id: m.context.goal_bucket_id,
            }
        },
        None => MemoryFeatures {
            tier: 0,
            strength: 0.5,
            access_count: 1,
            context_fingerprint: episode.context.fingerprint,
            goal_bucket_id: episode.context.goal_bucket_id,
        },
    }
}

/// Extract StrategyFeatures from a Strategy, or return defaults if None.
pub(super) fn extract_strategy_features(strategy: Option<&Strategy>) -> StrategyFeatures {
    match strategy {
        Some(s) => StrategyFeatures {
            quality_score: s.quality_score,
            expected_success: s.expected_success,
            expected_value: s.expected_value,
            confidence: s.confidence,
            goal_bucket_id: s.goal_bucket_id,
            behavior_signature_hash: hash_str(&s.behavior_signature),
        },
        None => StrategyFeatures {
            quality_score: 0.5,
            expected_success: 0.5,
            expected_value: 0.5,
            confidence: 0.0,
            goal_bucket_id: 0,
            behavior_signature_hash: 0,
        },
    }
}

/// Extract PolicyFeatures from an Episode's context.
pub(super) fn extract_policy_features(episode: &Episode) -> PolicyFeatures {
    PolicyFeatures {
        goal_count: episode.context.active_goals.len() as u32,
        top_goal_priority: episode
            .context
            .active_goals
            .iter()
            .map(|g| g.priority)
            .fold(0.5f32, f32::max),
        resource_cpu_percent: 0.0,
        resource_memory_bytes: 0,
        context_fingerprint: episode.context.fingerprint,
    }
}

// ─────────────────────── Training tuple assembly ──────────────────────

/// Assemble a positive training tuple from a completed episode + matched memory + strategy.
/// Returns None if episode has < 3 events (too noisy per assembly rules).
pub(super) fn assemble_training_tuple(
    episode: &Episode,
    events: &[Event],
    memory: Option<&Memory>,
    strategy: Option<&Strategy>,
) -> Option<TrainingTuple> {
    if events.len() < 3 {
        return None;
    }

    // Pick the most significant event: first Action event, or last event
    let representative_event = events
        .iter()
        .find(|e| matches!(e.event_type, EventType::Action { .. }))
        .unwrap_or_else(|| events.last().unwrap());

    let event_features = extract_event_features(representative_event, episode);
    let memory_features = extract_memory_features(memory, episode);
    let strategy_features = extract_strategy_features(strategy);
    let policy_features = extract_policy_features(episode);

    Some(TrainingTuple {
        event_features,
        memory_features,
        strategy_features,
        policy_features,
        is_positive: true,
        weight: episode.salience_score,
    })
}

// ─────────────────────── Top-down scoring wrappers ──────────────────────

impl GraphEngine {
    /// Score a strategy candidate (top-down). Returns None if world model disabled.
    pub async fn wm_score_strategy(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
    ) -> Option<CriticReport> {
        let wm = self.world_model.as_ref()?;
        let guard = wm.read().await;
        Some(guard.score_strategy(policy, strategy))
    }

    /// Score a full configuration (top-down). Returns None if world model disabled.
    pub async fn wm_score(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
        memory: &MemoryFeatures,
        event: &EventFeatures,
    ) -> Option<CriticReport> {
        let wm = self.world_model.as_ref()?;
        let guard = wm.read().await;
        Some(guard.score(policy, strategy, memory, event))
    }

    /// Compute prediction error (bottom-up). Returns None if world model disabled.
    pub async fn wm_prediction_error(
        &self,
        event: &EventFeatures,
        memory: &MemoryFeatures,
        strategy: &StrategyFeatures,
        policy: &PolicyFeatures,
    ) -> Option<PredictionErrorReport> {
        let wm = self.world_model.as_ref()?;
        let guard = wm.read().await;
        Some(guard.prediction_error(event, memory, strategy, policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_core::types::ContextHash;
    use agent_db_events::core::{ActionOutcome, EventContext};

    fn make_event(action_name: &str, success: bool) -> Event {
        Event {
            id: 1,
            timestamp: 1_000_000_000,
            agent_id: 1,
            agent_type: "test".to_string(),
            session_id: 100,
            event_type: EventType::Action {
                action_name: action_name.to_string(),
                parameters: serde_json::json!({}),
                outcome: if success {
                    ActionOutcome::Success {
                        result: serde_json::json!({}),
                    }
                } else {
                    ActionOutcome::Failure {
                        error: "err".to_string(),
                        error_code: 1,
                    }
                },
                duration_ns: 500_000_000,
            },
            causality_chain: vec![],
            context: EventContext::default(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        }
    }

    fn make_episode(num_events: usize) -> (Episode, Vec<Event>) {
        let events: Vec<Event> = (0..num_events)
            .map(|i| {
                let mut e = make_event("test_action", true);
                e.id = i as u128;
                e.timestamp = (i as u64 + 1) * 1_000_000_000;
                e
            })
            .collect();

        let episode = Episode {
            id: 1,
            episode_version: 1,
            agent_id: 1,
            start_event: 0,
            end_event: Some(events.last().unwrap().id),
            events: events.iter().map(|e| e.id).collect(),
            session_id: 100,
            context_signature: 12345 as ContextHash,
            context: EventContext::default(),
            outcome: Some(EpisodeOutcome::Success),
            start_timestamp: 1_000_000_000,
            end_timestamp: Some(num_events as u64 * 1_000_000_000),
            significance: 0.8,
            prediction_error: 0.3,
            self_judged_quality: None,
            salience_score: 0.75,
            last_event_timestamp: Some(num_events as u64 * 1_000_000_000),
            consecutive_outcome_count: 0,
        };

        (episode, events)
    }

    #[test]
    fn test_hash_str_deterministic() {
        assert_eq!(hash_str("Action"), hash_str("Action"));
        assert_ne!(hash_str("Action"), hash_str("Observation"));
    }

    #[test]
    fn test_extract_event_features_raw() {
        let event = make_event("deploy", true);
        let features = extract_event_features_raw(&event);
        assert_eq!(features.event_type_hash, hash_str("Action"));
        assert_eq!(features.action_name_hash, hash_str("deploy"));
        assert_eq!(features.outcome_success, 1.0);
        assert_eq!(features.duration_ns, 500_000_000.0);
    }

    #[test]
    fn test_extract_event_features_with_episode() {
        let (episode, events) = make_episode(5);
        let features = extract_event_features(&events[0], &episode);
        assert_eq!(features.outcome_success, 1.0); // Success episode
        assert_eq!(features.significance, 0.8);
        assert!(features.duration_ns > 0.0);
    }

    #[test]
    fn test_extract_default_memory_features() {
        let (episode, _) = make_episode(3);
        let features = extract_memory_features(None, &episode);
        assert_eq!(features.tier, 0);
        assert_eq!(features.strength, 0.5);
        assert_eq!(features.access_count, 1);
    }

    #[test]
    fn test_extract_default_strategy_features() {
        let features = extract_strategy_features(None);
        assert_eq!(features.quality_score, 0.5);
        assert_eq!(features.confidence, 0.0);
    }

    #[test]
    fn test_extract_policy_features() {
        let (episode, _) = make_episode(3);
        let features = extract_policy_features(&episode);
        assert_eq!(features.goal_count, 0); // default context has no goals
        assert_eq!(features.top_goal_priority, 0.5); // fold default
    }

    #[test]
    fn test_assemble_training_tuple_skips_short_episodes() {
        let (episode, events) = make_episode(2); // < 3 events
        let result = assemble_training_tuple(&episode, &events, None, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_assemble_training_tuple_success() {
        let (episode, events) = make_episode(5);
        let result = assemble_training_tuple(&episode, &events, None, None);
        assert!(result.is_some());
        let tuple = result.unwrap();
        assert!(tuple.is_positive);
        assert_eq!(tuple.weight, 0.75); // salience_score
        assert_eq!(tuple.event_features.outcome_success, 1.0);
    }
}
