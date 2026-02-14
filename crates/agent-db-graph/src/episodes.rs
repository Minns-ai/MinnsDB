// crates/agent-db-graph/src/episodes.rs
//
// Episode Detection Module
//
// Automatically detects episode boundaries from event streams for memory formation.
// Episodes are meaningful sequences of events that form coherent units of experience.

use crate::structures::Graph;
use agent_db_core::types::{AgentId, ContextHash, EventId, SessionId, Timestamp};
use agent_db_events::core::{
    ActionOutcome, CognitiveType, Event, EventContext, EventType, MetadataValue,
};

use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for an episode
pub type EpisodeId = u64;

/// Episode represents a coherent sequence of events forming a memorable experience
#[derive(Debug, Clone)]
pub struct Episode {
    /// Unique episode identifier
    pub id: EpisodeId,

    /// Episode version (incremented on late corrections)
    pub episode_version: u32,

    /// Agent that experienced this episode
    pub agent_id: AgentId,

    /// First event in the episode
    pub start_event: EventId,

    /// Last event in the episode (None if still active)
    pub end_event: Option<EventId>,

    /// All events in this episode
    pub events: Vec<EventId>,

    /// Session identifier for this episode
    pub session_id: SessionId,

    /// Context signature for this episode
    pub context_signature: ContextHash,

    /// Most recent context snapshot in this episode
    pub context: EventContext,

    /// Overall outcome of the episode
    pub outcome: Option<EpisodeOutcome>,

    /// Timestamp when episode started
    pub start_timestamp: Timestamp,

    /// Timestamp when episode ended (None if still active)
    pub end_timestamp: Option<Timestamp>,

    /// Significance score (0.0 to 1.0)
    pub significance: f32,

    // ========== Phase 1 Upgrades ==========
    /// Prediction error: difference between expected and actual outcome (0.0 to 1.0)
    /// Higher values indicate surprising outcomes that should be weighted more in learning
    pub prediction_error: f32,

    /// Self-judged quality score provided by the agent (0.0 to 1.0)
    /// None if agent hasn't provided self-assessment
    pub self_judged_quality: Option<f32>,

    /// Salience score: combination of surprise, outcome importance, and goal relevance (0.0 to 1.0)
    /// Used to prioritize which episodes to replay and consolidate into semantic memory
    pub salience_score: f32,

    /// Timestamp of the last event added to this episode
    /// Used for time-gap episode end detection
    pub last_event_timestamp: Option<Timestamp>,

    /// Count of consecutive outcome events (Success/Failure) at the tail of the episode
    /// Resets to 0 when a non-outcome event is added
    pub consecutive_outcome_count: u32,
}

/// Outcome of an episode
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EpisodeOutcome {
    /// Goal was achieved successfully
    Success,

    /// Goal was not achieved
    Failure,

    /// Partial achievement or mixed results
    Partial,

    /// Episode interrupted or abandoned
    Interrupted,
}

/// Configuration for episode detection
#[derive(Debug, Clone)]
pub struct EpisodeDetectorConfig {
    /// Minimum significance threshold for creating episodes (0.0 to 1.0)
    pub min_significance_threshold: f32,

    /// Context similarity threshold for detecting context shifts
    pub context_shift_threshold: f32,

    /// Maximum time gap before considering episode ended (nanoseconds)
    pub max_time_gap_ns: u64,

    /// Minimum events in an episode
    pub min_events_per_episode: usize,

    /// Periodic consolidation interval (nanoseconds)
    pub consolidation_interval_ns: u64,

    /// Late event correction window (nanoseconds)
    pub late_event_window_ns: u64,

    /// Maximum completed episodes to keep in memory (ring-buffer cap)
    pub max_completed_episodes: usize,

    /// Cognitive event types that can start a new episode
    pub episode_start_types: Vec<CognitiveType>,

    /// Minimum events before an outcome (Success/Failure) can end the episode
    pub min_events_before_outcome_end: usize,

    /// Significance threshold that alone qualifies an episode for storage (OR gate)
    pub high_significance_override: f32,
}

impl Default for EpisodeDetectorConfig {
    fn default() -> Self {
        Self {
            min_significance_threshold: 0.25,
            context_shift_threshold: 0.4,
            max_time_gap_ns: 3_600_000_000_000, // 1 hour
            min_events_per_episode: 2,
            consolidation_interval_ns: 3_600_000_000_000, // 1 hour
            late_event_window_ns: 5_000_000_000,          // 5 seconds
            max_completed_episodes: 5_000,
            episode_start_types: vec![
                CognitiveType::GoalFormation,
                CognitiveType::Planning,
                CognitiveType::LearningUpdate,
            ],
            min_events_before_outcome_end: 2,
            high_significance_override: 0.5,
        }
    }
}

/// Episode detector automatically identifies episode boundaries from event streams
pub struct EpisodeDetector {
    /// Active episodes by agent
    active_episodes: HashMap<AgentId, Episode>,

    /// Completed episodes
    completed_episodes: Vec<Episode>,

    /// Reference to graph for relationship analysis
    #[allow(dead_code)]
    graph: Arc<Graph>,

    /// Configuration
    config: EpisodeDetectorConfig,

    /// Next episode ID
    next_episode_id: EpisodeId,

    /// Last consolidation timestamp by agent
    last_consolidation: HashMap<AgentId, Timestamp>,

    // ========== Novelty Tracking (for significance calculation) ==========
    /// Track seen context hashes for novelty detection
    seen_contexts: HashMap<ContextHash, u32>, // hash -> count

    /// Track seen event types for novelty detection
    seen_event_types: HashMap<String, u32>, // event_type_name -> count
}

#[derive(Debug, Clone, Copy)]
pub enum EpisodeUpdate {
    Completed(EpisodeId),
    Corrected(EpisodeId),
}

impl EpisodeDetector {
    /// Create a new episode detector
    pub fn new(graph: Arc<Graph>, config: EpisodeDetectorConfig) -> Self {
        Self {
            active_episodes: HashMap::new(),
            completed_episodes: Vec::new(),
            graph,
            config,
            next_episode_id: 1,
            last_consolidation: HashMap::new(),
            seen_contexts: HashMap::new(),
            seen_event_types: HashMap::new(),
        }
    }

    /// Process a new event and detect episode boundaries
    ///
    /// Returns the episode ID if the event triggers episode formation
    pub fn process_event(&mut self, event: &Event) -> Option<EpisodeUpdate> {
        let agent_id = event.agent_id;
        tracing::info!(
            "EpisodeDetector process_event agent_id={} event_id={}",
            agent_id,
            event.id
        );

        // Track event for novelty detection (do this first, before calculating significance)
        self.track_event_for_novelty(event);

        // Calculate significance for this event before mutably borrowing episode
        let event_significance = self.calculate_significance(event);

        // If no active episode, try to apply late correction to a recently completed episode
        if !self.active_episodes.contains_key(&agent_id) {
            if let Some(episode_id) = self.apply_late_event_correction(event, event_significance) {
                return Some(EpisodeUpdate::Corrected(episode_id));
            }
        }

        // Add event to active episode if exists and update significance
        if let Some(episode) = self.active_episodes.get_mut(&agent_id) {
            episode.events.push(event.id);

            // Update episode significance incrementally as events are added
            let weighted_avg = (episode.significance * (episode.events.len() - 1) as f32
                + event_significance)
                / episode.events.len() as f32;
            let max_significance = episode.significance.max(event_significance);
            episode.significance = (max_significance * 0.7 + weighted_avg * 0.3).min(1.0);

            // Track latest context snapshot for memory formation
            episode.context = event.context.clone();

            // Track last event timestamp for time-gap detection
            episode.last_event_timestamp = Some(event.timestamp);

            // Track consecutive outcome events for smarter episode end conditions
            let is_outcome = matches!(
                &event.event_type,
                EventType::Action {
                    outcome: ActionOutcome::Success { .. } | ActionOutcome::Failure { .. },
                    ..
                }
            );
            if is_outcome {
                episode.consecutive_outcome_count += 1;
            } else {
                episode.consecutive_outcome_count = 0;
            }
        }

        // Check if this event should end the episode
        let should_end = if let Some(episode) = self.active_episodes.get(&agent_id) {
            self.is_episode_end(event, episode)
        } else {
            false
        };

        if should_end {
            tracing::info!(
                "EpisodeDetector ending episode agent_id={} event_id={}",
                agent_id,
                event.id
            );
            let completed = self.complete_episode(agent_id, event);
            return Some(EpisodeUpdate::Completed(completed.id));
        }

        // Check if this event should start a new episode
        if self.is_episode_start(event) {
            tracing::info!(
                "EpisodeDetector starting new episode agent_id={} event_id={}",
                agent_id,
                event.id
            );
            self.start_episode(event);
        }

        // Check for periodic consolidation
        if self.should_consolidate(agent_id, event.timestamp) {
            if let Some(episode_id) = self.consolidate_agent_episode(agent_id, event.timestamp) {
                return Some(EpisodeUpdate::Completed(episode_id));
            }
        }

        None
    }

    /// Check if an event should start a new episode
    fn is_episode_start(&self, event: &Event) -> bool {
        let agent_id = event.agent_id;

        // No active episode exists
        if !self.active_episodes.contains_key(&agent_id) {
            return true;
        }

        // Check for episode start triggers using configurable types
        match &event.event_type {
            EventType::Cognitive { process_type, .. } => {
                self.config.episode_start_types.contains(process_type)
            },
            _ => false,
        }
    }

    /// Check if an event should end the current episode
    fn is_episode_end(&self, event: &Event, episode: &Episode) -> bool {
        if Self::is_feedback_event(event) {
            return true;
        }

        // Check for goal completion — but only if the episode has enough events
        // This prevents premature truncation of multi-step retry patterns
        if let EventType::Action {
            outcome: ActionOutcome::Success { .. } | ActionOutcome::Failure { .. },
            ..
        } = &event.event_type
        {
            let enough_events = episode.events.len() >= self.config.min_events_before_outcome_end;
            let consecutive_outcome = self.is_consecutive_outcome_event(episode, event);
            if enough_events || consecutive_outcome {
                return true;
            }
        }

        // Check for significant context shift
        if self.has_context_shift(event, episode) {
            return true;
        }

        // Check for time gap
        if let Some(last_event_time) = self.get_last_event_timestamp(episode) {
            let time_gap = event.timestamp.saturating_sub(last_event_time);
            if time_gap > self.config.max_time_gap_ns {
                return true;
            }
        }

        false
    }

    /// Check if the previous event in the episode was also an outcome event
    /// (prevents infinite episodes by ending on second consecutive outcome)
    fn is_consecutive_outcome_event(&self, episode: &Episode, _current_event: &Event) -> bool {
        // The current event is an outcome event (checked by caller).
        // consecutive_outcome_count was already incremented in process_event().
        // If count >= 2, this is at least the second consecutive outcome.
        episode.consecutive_outcome_count >= 2
    }

    /// Check if there's a significant context shift
    fn has_context_shift(&self, event: &Event, episode: &Episode) -> bool {
        let similarity = self.calculate_context_similarity(&episode.context, &event.context);
        similarity < self.config.context_shift_threshold
    }

    fn is_feedback_event(event: &Event) -> bool {
        if let EventType::Action { action_name, .. } = &event.event_type {
            if action_name == "user_feedback" {
                return true;
            }
        }
        event.metadata.contains_key("feedback")
    }

    fn calculate_context_similarity(&self, a: &EventContext, b: &EventContext) -> f32 {
        if let (Some(embed_a), Some(embed_b)) = (&a.embeddings, &b.embeddings) {
            return Self::cosine_similarity(embed_a, embed_b);
        }

        let goals_a: std::collections::HashSet<u64> = a.active_goals.iter().map(|g| g.id).collect();
        let goals_b: std::collections::HashSet<u64> = b.active_goals.iter().map(|g| g.id).collect();
        let goal_score = Self::jaccard_u64(&goals_a, &goals_b);

        let keys_a: std::collections::HashSet<String> =
            a.environment.variables.keys().cloned().collect();
        let keys_b: std::collections::HashSet<String> =
            b.environment.variables.keys().cloned().collect();
        let env_score = Self::jaccard_string(&keys_a, &keys_b);

        (goal_score * 0.7 + env_score * 0.3).clamp(0.0, 1.0)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(0.0, 1.0)
    }

    fn jaccard_u64(a: &std::collections::HashSet<u64>, b: &std::collections::HashSet<u64>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    fn jaccard_string(
        a: &std::collections::HashSet<String>,
        b: &std::collections::HashSet<String>,
    ) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Start a new episode
    fn start_episode(&mut self, event: &Event) {
        let episode_id = self.next_episode_id;
        self.next_episode_id += 1;

        let episode = Episode {
            id: episode_id,
            episode_version: 1,
            agent_id: event.agent_id,
            start_event: event.id,
            end_event: None,
            events: vec![event.id],
            session_id: event.session_id,
            context_signature: event.context.fingerprint,
            context: event.context.clone(),
            outcome: None,
            start_timestamp: event.timestamp,
            end_timestamp: None,
            significance: self.calculate_significance(event),
            // Phase 1 fields - initialized with defaults, updated when episode completes
            prediction_error: 0.0,
            self_judged_quality: None,
            salience_score: 0.0,
            last_event_timestamp: Some(event.timestamp),
            consecutive_outcome_count: 0,
        };

        // Bootstrap consolidation timer on first episode for this agent
        self.last_consolidation
            .entry(event.agent_id)
            .or_insert(event.timestamp);

        self.active_episodes.insert(event.agent_id, episode);
    }

    /// Complete the current episode for an agent
    fn complete_episode(&mut self, agent_id: AgentId, end_event: &Event) -> Episode {
        let mut episode = self
            .active_episodes
            .remove(&agent_id)
            .unwrap_or_else(|| {
                tracing::error!(
                    "Episode not found for agent_id={} during complete_episode - creating emergency episode",
                    agent_id
                );
                // Create emergency episode to avoid panic
                Episode {
                    id: self.next_episode_id,
                    episode_version: 1,
                    agent_id,
                    session_id: end_event.session_id,
                    start_event: end_event.id,
                    end_event: None,
                    events: vec![end_event.id],
                    context_signature: end_event.context.fingerprint,
                    start_timestamp: end_event.timestamp,
                    end_timestamp: None,
                    outcome: None,
                    significance: 0.0,
                    prediction_error: 0.0,
                    self_judged_quality: None,
                    salience_score: 0.0,
                    context: end_event.context.clone(),
                    last_event_timestamp: Some(end_event.timestamp),
                    consecutive_outcome_count: 0,
                }
            });

        episode.end_event = Some(end_event.id);
        episode.end_timestamp = Some(end_event.timestamp);
        episode.outcome = Some(self.determine_outcome(end_event));

        // Phase 1: Calculate prediction error and salience
        episode.prediction_error = self.calculate_prediction_error(&episode);
        episode.salience_score = self.calculate_salience(&episode);

        if Self::is_feedback_event(end_event)
            && episode.significance < self.config.min_significance_threshold
        {
            episode.significance = self.config.min_significance_threshold;
        }

        // Store if: enough events OR high significance OR feedback
        let enough_events = episode.events.len() >= self.config.min_events_per_episode;
        let high_significance = episode.significance >= self.config.high_significance_override;
        let has_feedback = Self::is_feedback_event(end_event);
        if enough_events || high_significance || has_feedback {
            tracing::info!(
                "EpisodeDetector completed episode_id={} agent_id={} events={} significance={:.3} outcome={:?}",
                episode.id,
                episode.agent_id,
                episode.events.len(),
                episode.significance,
                episode.outcome
            );
            self.completed_episodes.push(episode.clone());
            self.enforce_completed_episodes_cap();
        } else {
            tracing::info!(
                "EpisodeDetector discarded episode_id={} events={} significance={:.3} min_events={} min_sig={:.3}",
                episode.id,
                episode.events.len(),
                episode.significance,
                self.config.min_events_per_episode,
                self.config.min_significance_threshold
            );
        }

        episode
    }

    /// Determine episode outcome from the final event
    fn determine_outcome(&self, event: &Event) -> EpisodeOutcome {
        match &event.event_type {
            EventType::Action { outcome, .. } => match outcome {
                ActionOutcome::Success { .. } => EpisodeOutcome::Success,
                ActionOutcome::Failure { .. } => EpisodeOutcome::Failure,
                ActionOutcome::Partial { .. } => EpisodeOutcome::Partial,
            },
            _ => EpisodeOutcome::Interrupted,
        }
    }

    fn apply_late_event_correction(
        &mut self,
        event: &Event,
        event_significance: f32,
    ) -> Option<EpisodeId> {
        let window = self.config.late_event_window_ns;
        if window == 0 {
            return None;
        }

        let mut target_index: Option<usize> = None;
        for (idx, episode) in self.completed_episodes.iter().enumerate().rev() {
            if episode.agent_id != event.agent_id || episode.session_id != event.session_id {
                continue;
            }
            let Some(end_ts) = episode.end_timestamp else {
                continue;
            };
            let delta = event.timestamp.abs_diff(end_ts);
            if delta <= window {
                target_index = Some(idx);
                break;
            }
        }

        let idx = target_index?;

        let outcome_from_event = self.determine_outcome(event);

        {
            let episode = &mut self.completed_episodes[idx];
            if episode.events.contains(&event.id) {
                return Some(episode.id);
            }

            episode.episode_version = episode.episode_version.saturating_add(1);

            let previous_count = episode.events.len() as f32;
            episode.events.push(event.id);
            let weighted_avg = (episode.significance * previous_count + event_significance)
                / (previous_count + 1.0);
            let max_significance = episode.significance.max(event_significance);
            episode.significance = (max_significance * 0.7 + weighted_avg * 0.3).min(1.0);
            episode.context = event.context.clone();

            if let Some(end_ts) = episode.end_timestamp {
                if event.timestamp >= end_ts {
                    episode.end_timestamp = Some(event.timestamp);
                    episode.end_event = Some(event.id);
                }
            }

            if matches!(episode.outcome, Some(EpisodeOutcome::Interrupted) | None) {
                if let EventType::Action { outcome, .. } = &event.event_type {
                    match outcome {
                        ActionOutcome::Success { .. }
                        | ActionOutcome::Failure { .. }
                        | ActionOutcome::Partial { .. } => {
                            episode.outcome = Some(outcome_from_event.clone());
                        },
                    }
                }
            }

            if Self::is_feedback_event(event)
                && episode.significance < self.config.min_significance_threshold
            {
                episode.significance = self.config.min_significance_threshold;
            }
            tracing::info!(
            "EpisodeDetector applied late correction episode_id={} event_id={} events={} significance={:.3}",
            episode.id,
            event.id,
            episode.events.len(),
            episode.significance
        );
        }

        let (prediction_error, salience_score) = {
            let episode = &self.completed_episodes[idx];
            (
                self.calculate_prediction_error(episode),
                self.calculate_salience(episode),
            )
        };

        {
            let episode = &mut self.completed_episodes[idx];
            episode.prediction_error = prediction_error;
            episode.salience_score = salience_score;
            Some(episode.id)
        }
    }

    /// Calculate comprehensive significance score for an event
    ///
    /// Combines multiple signals to determine how memorable/important this event is:
    /// - Manual override from event metadata (highest priority)
    /// - Goal relevance (number of active goals)
    /// - Causal chain length (deeper chains = more significant)
    /// - Event duration/effort (longer actions = more significant)
    /// - Novelty (rare contexts or event types = more significant)
    /// - Event type importance (goals, successes, failures)
    ///
    /// All signals are weighted and normalized to [0.0, 1.0]
    fn calculate_significance(&self, event: &Event) -> f32 {
        // 1) Manual override: if metadata contains "significance", use it directly
        if let Some(sig_value) = event.metadata.get("significance") {
            if let MetadataValue::String(sig_str) = sig_value {
                if let Ok(sig_float) = sig_str.parse::<f32>() {
                    return sig_float.clamp(0.0, 1.0);
                }
            } else if let MetadataValue::Float(sig_float) = sig_value {
                return (*sig_float as f32).clamp(0.0, 1.0);
            }
        }

        // Base significance
        let mut significance: f32 = 0.3;

        // 2) Goal relevance: more active goals = more significant context
        let goal_count = event.context.active_goals.len();
        let goal_relevance = (goal_count as f32 * 0.15).min(0.25);
        significance += goal_relevance;

        // 3) Causal chain length: deeper chains indicate more complex reasoning
        let chain_length = event.causality_chain.len();
        let chain_significance = (chain_length as f32 * 0.05).min(0.20);
        significance += chain_significance;

        // 4) Event duration/effort: check metadata for duration or use event-specific duration
        let duration_significance = if let Some(duration_value) = event.metadata.get("duration_ns")
        {
            // Custom duration from metadata
            let duration_ns = match duration_value {
                MetadataValue::String(s) => s.parse::<u64>().ok(),
                MetadataValue::Integer(i) => Some(*i as u64),
                _ => None,
            };
            if let Some(ns) = duration_ns {
                // Scale: 1 second = 0.05, 10 seconds = 0.15, cap at 0.15
                ((ns as f32 / 1_000_000_000.0) * 0.015).min(0.15)
            } else {
                0.0
            }
        } else if let EventType::Action { duration_ns, .. } = &event.event_type {
            // Use action duration
            ((*duration_ns as f32 / 1_000_000_000.0) * 0.015).min(0.15)
        } else {
            0.0
        };
        significance += duration_significance;

        // 5) Novelty: rare contexts or event types are more memorable
        let context_novelty = self.calculate_context_novelty(event.context.fingerprint);
        let event_type_novelty = self.calculate_event_type_novelty(event);
        let novelty_significance = ((context_novelty + event_type_novelty) / 2.0 * 0.20).min(0.20);
        significance += novelty_significance;

        // 6) Event type importance (baseline behavior)
        match &event.event_type {
            EventType::Cognitive { process_type, .. } => match process_type {
                CognitiveType::GoalFormation => significance += 0.15,
                CognitiveType::Planning => significance += 0.1,
                CognitiveType::Reasoning => significance += 0.08,
                CognitiveType::LearningUpdate => significance += 0.12,
                CognitiveType::MemoryRetrieval => {},
            },
            EventType::Action { outcome, .. } => match outcome {
                ActionOutcome::Success { .. } => significance += 0.10,
                ActionOutcome::Failure { .. } => significance += 0.12,
                _ => {},
            },
            EventType::Communication { .. } => significance += 0.05,
            EventType::Observation { confidence, .. } => {
                if *confidence > 0.7 {
                    significance += 0.08;
                }
            },
            EventType::Learning { .. } => significance += 0.1,
            EventType::Context { .. } => {},
        }

        // Normalize to [0.0, 1.0]
        significance.min(1.0)
    }

    /// Calculate novelty for a context hash (0.0 = common, 1.0 = brand new)
    fn calculate_context_novelty(&self, context_hash: ContextHash) -> f32 {
        match self.seen_contexts.get(&context_hash) {
            None => 1.0, // Never seen before
            Some(&count) => {
                // Novelty decays as we see it more: 1/(1 + log(count))
                let novelty = 1.0 / (1.0 + (count as f32).log2());
                novelty.clamp(0.0, 1.0)
            },
        }
    }

    /// Calculate novelty for an event type (0.0 = common, 1.0 = brand new)
    fn calculate_event_type_novelty(&self, event: &Event) -> f32 {
        let event_type_name = match &event.event_type {
            EventType::Action { action_name, .. } => format!("Action:{}", action_name),
            EventType::Cognitive { process_type, .. } => format!("Cognitive:{:?}", process_type),
            EventType::Communication { message_type, .. } => {
                format!("Communication:{:?}", message_type)
            },
            EventType::Observation {
                observation_type, ..
            } => format!("Observation:{}", observation_type),
            EventType::Learning { .. } => "Learning".to_string(),
            EventType::Context { context_type, .. } => format!("Context:{}", context_type),
        };

        match self.seen_event_types.get(&event_type_name) {
            None => 1.0, // Never seen before
            Some(&count) => {
                let novelty = 1.0 / (1.0 + (count as f32).log2());
                novelty.clamp(0.0, 1.0)
            },
        }
    }

    /// Update episode significance when a new event is added
    ///
    /// This is called incrementally as events are added to an active episode.
    /// The significance can increase as more goal-relevant events occur,
    /// longer causal chains develop, or important outcomes happen.
    #[allow(dead_code)]
    fn update_episode_significance(&mut self, episode: &mut Episode, event: &Event) {
        // Calculate significance for this new event
        let event_significance = self.calculate_significance(event);

        // Update episode significance by taking the maximum of:
        // 1. Current episode significance
        // 2. New event significance
        // 3. Weighted average that emphasizes significant events

        // Weight towards higher significance (episodes become more significant, rarely less)
        let weighted_avg = (episode.significance * episode.events.len() as f32
            + event_significance)
            / (episode.events.len() as f32 + 1.0);

        let max_significance = episode.significance.max(event_significance);

        // Blend: 70% max (keeps highest peaks), 30% weighted average (smooth growth)
        episode.significance = (max_significance * 0.7 + weighted_avg * 0.3).min(1.0);
    }

    /// Track an event for novelty detection
    fn track_event_for_novelty(&mut self, event: &Event) {
        // Track context hash
        *self
            .seen_contexts
            .entry(event.context.fingerprint)
            .or_insert(0) += 1;

        // Track event type
        let event_type_name = match &event.event_type {
            EventType::Action { action_name, .. } => format!("Action:{}", action_name),
            EventType::Cognitive { process_type, .. } => format!("Cognitive:{:?}", process_type),
            EventType::Communication { message_type, .. } => {
                format!("Communication:{:?}", message_type)
            },
            EventType::Observation {
                observation_type, ..
            } => format!("Observation:{}", observation_type),
            EventType::Learning { .. } => "Learning".to_string(),
            EventType::Context { context_type, .. } => format!("Context:{}", context_type),
        };
        *self.seen_event_types.entry(event_type_name).or_insert(0) += 1;
    }

    /// Phase 1 Feature A: Calculate prediction error for an episode
    /// Measures how surprising the outcome was compared to expectations
    /// Higher values (closer to 1.0) indicate unexpected outcomes requiring stronger learning
    fn calculate_prediction_error(&self, episode: &Episode) -> f32 {
        // Get actual outcome value (1.0 = success, 0.0 = failure, 0.5 = partial/interrupted)
        let actual_outcome = match episode.outcome {
            Some(EpisodeOutcome::Success) => 1.0,
            Some(EpisodeOutcome::Failure) => 0.0,
            Some(EpisodeOutcome::Partial) => 0.5,
            Some(EpisodeOutcome::Interrupted) => 0.3,
            None => 0.5, // Unknown outcome, medium surprise
        };

        // Expected outcome based on episode significance and length
        // High significance + many events => expect success
        // Low significance + few events => expect partial/failure
        let episode_quality =
            (episode.significance * 0.7) + ((episode.events.len() as f32 / 10.0).min(1.0) * 0.3);
        let expected_outcome = episode_quality.min(0.9); // Cap expectation at 0.9

        // Prediction error is the absolute difference
        (actual_outcome - expected_outcome).abs()
    }

    /// Phase 1 Feature A: Calculate salience score for an episode
    /// Combines surprise (prediction error), outcome importance, and goal relevance
    /// Used to prioritize which episodes to replay and consolidate
    fn calculate_salience(&self, episode: &Episode) -> f32 {
        let prediction_error = episode.prediction_error;
        let significance = episode.significance;

        // Outcome importance: Success and Failure are more important than Partial/Interrupted
        let outcome_importance = match episode.outcome {
            Some(EpisodeOutcome::Success) => 0.9,
            Some(EpisodeOutcome::Failure) => 0.8, // Failures are highly salient for learning
            Some(EpisodeOutcome::Partial) => 0.5,
            Some(EpisodeOutcome::Interrupted) => 0.3,
            None => 0.4,
        };

        // Goal relevance: Higher significance implies more goal-directed behavior
        let goal_relevance = significance;

        // Weighted combination: emphasize surprise and outcome importance
        let salience =
            (prediction_error * 0.4) + (outcome_importance * 0.4) + (goal_relevance * 0.2);

        salience.min(1.0)
    }

    /// Get the timestamp of the last event in an episode
    fn get_last_event_timestamp(&self, episode: &Episode) -> Option<Timestamp> {
        episode.last_event_timestamp
    }

    /// Check if periodic consolidation should happen
    fn should_consolidate(&self, agent_id: AgentId, current_time: Timestamp) -> bool {
        if let Some(&last_time) = self.last_consolidation.get(&agent_id) {
            current_time.saturating_sub(last_time) >= self.config.consolidation_interval_ns
        } else {
            // No consolidation record yet — bootstrap will be set in start_episode
            false
        }
    }

    /// Consolidate active episode for an agent
    fn consolidate_agent_episode(
        &mut self,
        agent_id: AgentId,
        current_time: Timestamp,
    ) -> Option<EpisodeId> {
        self.last_consolidation.insert(agent_id, current_time);

        if let Some(episode) = self.active_episodes.get(&agent_id) {
            // Don't consolidate if episode is too short
            if episode.events.len() < self.config.min_events_per_episode {
                return None;
            }

            // Create a synthetic end event for consolidation
            // In practice, we'd use the last actual event
            let episode_id = episode.id;

            if let Some(mut episode) = self.active_episodes.remove(&agent_id) {
                episode.end_timestamp = Some(current_time);
                episode.outcome = Some(EpisodeOutcome::Interrupted);

                if episode.significance >= self.config.min_significance_threshold {
                    self.completed_episodes.push(episode);
                    self.enforce_completed_episodes_cap();
                    return Some(episode_id);
                }
            }
        }

        None
    }

    /// Get all completed episodes
    pub fn get_completed_episodes(&self) -> &[Episode] {
        &self.completed_episodes
    }

    /// Get active episode for an agent
    pub fn get_active_episode(&self, agent_id: AgentId) -> Option<&Episode> {
        self.active_episodes.get(&agent_id)
    }

    /// Get a specific completed episode by ID
    pub fn get_episode(&self, episode_id: EpisodeId) -> Option<&Episode> {
        self.completed_episodes.iter().find(|e| e.id == episode_id)
    }

    /// Get all episodes for a specific agent
    pub fn get_agent_episodes(&self, agent_id: AgentId) -> Vec<&Episode> {
        self.completed_episodes
            .iter()
            .filter(|e| e.agent_id == agent_id)
            .collect()
    }

    /// Enforce the completed_episodes cap by draining oldest entries
    fn enforce_completed_episodes_cap(&mut self) {
        let cap = self.config.max_completed_episodes;
        if self.completed_episodes.len() > cap {
            let excess = self.completed_episodes.len() - cap;
            self.completed_episodes.drain(..excess);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_core::types::current_timestamp;
    use agent_db_events::core::{CognitiveType, EventType};
    use agent_db_events::{
        ComputationalResources, EnvironmentState, Goal, ResourceState, TemporalContext,
    };

    fn create_test_event(agent_id: AgentId, event_type: EventType) -> Event {
        Event {
            id: 1,
            timestamp: current_timestamp(),
            agent_id,
            agent_type: "test".to_string(),
            session_id: 1,
            event_type,
            causality_chain: vec![],
            context: EventContext {
                environment: EnvironmentState {
                    variables: HashMap::new(),
                    spatial: None,
                    temporal: TemporalContext {
                        time_of_day: None,
                        deadlines: vec![],
                        patterns: vec![],
                    },
                },
                active_goals: vec![],
                resources: ResourceState {
                    computational: ComputationalResources {
                        cpu_percent: 50.0,
                        memory_bytes: 1024 * 1024 * 1024,
                        storage_bytes: 10 * 1024 * 1024 * 1024,
                        network_bandwidth: 1000 * 1000,
                    },
                    external: HashMap::new(),
                },
                fingerprint: 12345,
                goal_bucket_id: 0,
                embeddings: None,
            },
            metadata: HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        }
    }

    #[test]
    fn test_episode_start_on_goal_formation() {
        let graph = Arc::new(Graph::new());
        let mut detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        let event = create_test_event(
            1,
            EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: serde_json::json!({}),
                output: serde_json::json!({}),
                reasoning_trace: vec![],
            },
        );

        assert!(detector.is_episode_start(&event));
        detector.process_event(&event);

        assert!(detector.get_active_episode(1).is_some());
    }

    #[test]
    fn test_episode_end_on_success() {
        let graph = Arc::new(Graph::new());
        let mut detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // Start episode
        let start_event = create_test_event(
            1,
            EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: serde_json::json!({}),
                output: serde_json::json!({}),
                reasoning_trace: vec![],
            },
        );
        detector.process_event(&start_event);

        // End episode with success
        let end_event = create_test_event(
            1,
            EventType::Action {
                action_name: "test_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!({}),
                },
                duration_ns: 1000,
            },
        );

        let episode_update = detector.process_event(&end_event);
        assert!(matches!(episode_update, Some(EpisodeUpdate::Completed(_))));
        assert!(detector.get_active_episode(1).is_none());
    }

    // ========== Enhanced Significance Calculation Tests ==========

    #[test]
    fn test_significance_increases_with_goal_events() {
        let graph = Arc::new(Graph::new());
        let mut detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // Start episode with no goals
        let mut start_event = create_test_event(
            1,
            EventType::Action {
                action_name: "initial_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );
        start_event.context.active_goals = vec![]; // No goals initially

        detector.process_event(&start_event);
        let initial_significance = detector
            .get_active_episode(1)
            .map(|ep| ep.significance)
            .unwrap_or(0.0);

        // Add event with multiple goals
        let mut goal_event = create_test_event(
            1,
            EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: serde_json::json!({}),
                output: serde_json::json!({}),
                reasoning_trace: vec![],
            },
        );
        goal_event.id = 2;
        goal_event.context.active_goals = vec![
            Goal {
                id: 1,
                description: "goal1".to_string(),
                priority: 1.0,
                deadline: None,
                progress: 0.0,
                subgoals: vec![],
            },
            Goal {
                id: 2,
                description: "goal2".to_string(),
                priority: 1.0,
                deadline: None,
                progress: 0.0,
                subgoals: vec![],
            },
        ];

        detector.process_event(&goal_event);
        let updated_significance = detector
            .get_active_episode(1)
            .map(|ep| ep.significance)
            .unwrap_or(0.0);

        // Significance should increase with goal-relevant events
        assert!(
            updated_significance > initial_significance,
            "Significance should increase from {:.3} to {:.3} when goals are present",
            initial_significance,
            updated_significance
        );
    }

    #[test]
    fn test_manual_significance_override() {
        let graph = Arc::new(Graph::new());
        let detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // Create event with manual significance override
        let mut event = create_test_event(
            1,
            EventType::Action {
                action_name: "test_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );
        event.metadata.insert(
            "significance".to_string(),
            MetadataValue::String("0.95".to_string()),
        );

        let significance = detector.calculate_significance(&event);

        // Should use the override value
        assert_eq!(
            significance, 0.95,
            "Manual override should set significance to 0.95, got {:.3}",
            significance
        );
    }

    #[test]
    fn test_manual_significance_override_clamped() {
        let graph = Arc::new(Graph::new());
        let detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // Create event with out-of-range override
        let mut event = create_test_event(
            1,
            EventType::Action {
                action_name: "test_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );
        event.metadata.insert(
            "significance".to_string(),
            MetadataValue::String("1.5".to_string()),
        );

        let significance = detector.calculate_significance(&event);

        // Should be clamped to 1.0
        assert_eq!(
            significance, 1.0,
            "Override should be clamped to 1.0, got {:.3}",
            significance
        );
    }

    #[test]
    fn test_longer_causality_increases_significance() {
        let graph = Arc::new(Graph::new());
        let detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // Event with no causal chain
        let mut event_no_chain = create_test_event(
            1,
            EventType::Action {
                action_name: "simple_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );
        event_no_chain.causality_chain = vec![];
        let sig_no_chain = detector.calculate_significance(&event_no_chain);

        // Event with longer causal chain
        let mut event_with_chain = create_test_event(
            1,
            EventType::Action {
                action_name: "complex_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );
        event_with_chain.causality_chain = vec![1, 2, 3, 4, 5]; // 5 steps
        let sig_with_chain = detector.calculate_significance(&event_with_chain);

        // Longer causal chain should increase significance
        assert!(
            sig_with_chain > sig_no_chain,
            "Longer causality chain should increase significance from {:.3} to {:.3}",
            sig_no_chain,
            sig_with_chain
        );
    }

    #[test]
    fn test_novelty_increases_significance() {
        let graph = Arc::new(Graph::new());
        let mut detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // First event (novel)
        let event1 = create_test_event(
            1,
            EventType::Action {
                action_name: "novel_action".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );

        // Calculate significance before tracking (should be high due to novelty)
        let sig_novel = detector.calculate_significance(&event1);

        // Track the event multiple times to reduce novelty
        for _ in 0..10 {
            detector.track_event_for_novelty(&event1);
        }

        // Same event after being seen multiple times
        let sig_familiar = detector.calculate_significance(&event1);

        // Novel events should have higher significance than familiar ones
        assert!(
            sig_novel > sig_familiar,
            "Novel event significance ({:.3}) should be higher than familiar ({:.3})",
            sig_novel,
            sig_familiar
        );
    }

    #[test]
    fn test_context_novelty_decay() {
        let graph = Arc::new(Graph::new());
        let mut detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        let context_hash = 12345u64;

        // First time seeing this context (brand new)
        let novelty_first = detector.calculate_context_novelty(context_hash);
        assert_eq!(
            novelty_first, 1.0,
            "Brand new context should have novelty 1.0"
        );

        // Track it a few times to get past the log2 edge case
        for _ in 0..5 {
            *detector.seen_contexts.entry(context_hash).or_insert(0) += 1;
        }
        let novelty_second = detector.calculate_context_novelty(context_hash);

        // Track it many more times
        for _ in 0..20 {
            *detector.seen_contexts.entry(context_hash).or_insert(0) += 1;
        }
        let novelty_familiar = detector.calculate_context_novelty(context_hash);

        // Novelty should decay with repeated exposure
        assert!(
            novelty_first > novelty_second,
            "Novelty should decrease after first exposure: {:.3} -> {:.3}",
            novelty_first,
            novelty_second
        );
        assert!(
            novelty_second > novelty_familiar,
            "Novelty should continue decreasing: {:.3} -> {:.3}",
            novelty_second,
            novelty_familiar
        );
    }

    #[test]
    fn test_episode_significance_grows_incrementally() {
        let graph = Arc::new(Graph::new());
        let mut detector = EpisodeDetector::new(graph, EpisodeDetectorConfig::default());

        // Start episode with basic event
        let start_event = create_test_event(
            1,
            EventType::Action {
                action_name: "start".to_string(),
                parameters: serde_json::json!({}),
                outcome: ActionOutcome::Partial {
                    result: serde_json::json!({}),
                    issues: vec![],
                },
                duration_ns: 1000,
            },
        );
        detector.process_event(&start_event);
        let sig_after_start = detector.get_active_episode(1).unwrap().significance;

        // Add a goal formation event (high significance)
        let mut goal_event = create_test_event(
            1,
            EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: serde_json::json!({}),
                output: serde_json::json!({}),
                reasoning_trace: vec![],
            },
        );
        goal_event.id = 2;
        goal_event.context.active_goals = vec![Goal {
            id: 1,
            description: "goal1".to_string(),
            priority: 1.0,
            deadline: None,
            progress: 0.0,
            subgoals: vec![],
        }];
        detector.process_event(&goal_event);
        let sig_after_goal = detector.get_active_episode(1).unwrap().significance;

        // Significance should have increased
        assert!(
            sig_after_goal > sig_after_start,
            "Episode significance should grow from {:.3} to {:.3} as important events are added",
            sig_after_start,
            sig_after_goal
        );
    }
}
