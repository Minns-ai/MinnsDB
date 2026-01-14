// crates/agent-db-graph/src/episodes.rs
//
// Episode Detection Module
//
// Automatically detects episode boundaries from event streams for memory formation.
// Episodes are meaningful sequences of events that form coherent units of experience.

use crate::structures::{Graph, NodeId};
use agent_db_core::types::{AgentId, EventId, Timestamp, ContextHash};
use agent_db_events::core::{Event, EventType, ActionOutcome, CognitiveType, EventContext};
use agent_db_events::{EnvironmentState, TemporalContext, ResourceState, ComputationalResources};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for an episode
pub type EpisodeId = u64;

/// Episode represents a coherent sequence of events forming a memorable experience
#[derive(Debug, Clone)]
pub struct Episode {
    /// Unique episode identifier
    pub id: EpisodeId,

    /// Agent that experienced this episode
    pub agent_id: AgentId,

    /// First event in the episode
    pub start_event: EventId,

    /// Last event in the episode (None if still active)
    pub end_event: Option<EventId>,

    /// All events in this episode
    pub events: Vec<EventId>,

    /// Context signature for this episode
    pub context_signature: ContextHash,

    /// Overall outcome of the episode
    pub outcome: Option<EpisodeOutcome>,

    /// Timestamp when episode started
    pub start_timestamp: Timestamp,

    /// Timestamp when episode ended (None if still active)
    pub end_timestamp: Option<Timestamp>,

    /// Significance score (0.0 to 1.0)
    pub significance: f32,
}

/// Outcome of an episode
#[derive(Debug, Clone, PartialEq)]
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
}

impl Default for EpisodeDetectorConfig {
    fn default() -> Self {
        Self {
            min_significance_threshold: 0.3,
            context_shift_threshold: 0.4,
            max_time_gap_ns: 3_600_000_000_000, // 1 hour
            min_events_per_episode: 3,
            consolidation_interval_ns: 3_600_000_000_000, // 1 hour
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
    graph: Arc<Graph>,

    /// Configuration
    config: EpisodeDetectorConfig,

    /// Next episode ID
    next_episode_id: EpisodeId,

    /// Last consolidation timestamp by agent
    last_consolidation: HashMap<AgentId, Timestamp>,
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
        }
    }

    /// Process a new event and detect episode boundaries
    ///
    /// Returns the episode ID if the event triggers episode formation
    pub fn process_event(&mut self, event: &Event) -> Option<EpisodeId> {
        let agent_id = event.agent_id;

        // Check if this event should start a new episode
        if self.is_episode_start(event) {
            self.start_episode(event);
        }

        // Add event to active episode if exists
        let should_end = if let Some(episode) = self.active_episodes.get_mut(&agent_id) {
            episode.events.push(event.id);

            // Check if this event should end the episode
            // Clone episode data needed for the check
            let episode_clone = episode.clone();
            drop(episode); // Drop the mutable borrow

            self.is_episode_end(event, &episode_clone)
        } else {
            false
        };

        if should_end {
            let completed = self.complete_episode(agent_id, event);
            return Some(completed.id);
        }

        // Check for periodic consolidation
        if self.should_consolidate(agent_id, event.timestamp) {
            if let Some(episode_id) = self.consolidate_agent_episode(agent_id, event.timestamp) {
                return Some(episode_id);
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

        // Check for episode start triggers
        match &event.event_type {
            EventType::Cognitive { process_type, .. } => {
                matches!(process_type, CognitiveType::GoalFormation)
            }
            _ => {
                // Root event (empty causality chain)
                event.causality_chain.is_empty()
            }
        }
    }

    /// Check if an event should end the current episode
    fn is_episode_end(&self, event: &Event, episode: &Episode) -> bool {
        // Check for goal completion
        if let EventType::Action { outcome, .. } = &event.event_type {
            match outcome {
                ActionOutcome::Success { .. } | ActionOutcome::Failure { .. } => {
                    return true;
                }
                _ => {}
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

    /// Check if there's a significant context shift
    fn has_context_shift(&self, event: &Event, episode: &Episode) -> bool {
        // Compare context fingerprints
        let current_hash = event.context.fingerprint;
        let episode_hash = episode.context_signature;

        if current_hash != episode_hash {
            // Context has changed significantly
            return true;
        }

        false
    }

    /// Start a new episode
    fn start_episode(&mut self, event: &Event) {
        let episode_id = self.next_episode_id;
        self.next_episode_id += 1;

        let episode = Episode {
            id: episode_id,
            agent_id: event.agent_id,
            start_event: event.id,
            end_event: None,
            events: vec![event.id],
            context_signature: event.context.fingerprint,
            outcome: None,
            start_timestamp: event.timestamp,
            end_timestamp: None,
            significance: self.calculate_significance(event),
        };

        self.active_episodes.insert(event.agent_id, episode);
    }

    /// Complete the current episode for an agent
    fn complete_episode(&mut self, agent_id: AgentId, end_event: &Event) -> Episode {
        let mut episode = self.active_episodes.remove(&agent_id).expect("Episode must exist");

        episode.end_event = Some(end_event.id);
        episode.end_timestamp = Some(end_event.timestamp);
        episode.outcome = Some(self.determine_outcome(end_event));

        // Only store if meets minimum criteria
        if episode.events.len() >= self.config.min_events_per_episode
            && episode.significance >= self.config.min_significance_threshold
        {
            self.completed_episodes.push(episode.clone());
        }

        episode
    }

    /// Determine episode outcome from the final event
    fn determine_outcome(&self, event: &Event) -> EpisodeOutcome {
        match &event.event_type {
            EventType::Action { outcome, .. } => {
                match outcome {
                    ActionOutcome::Success { .. } => EpisodeOutcome::Success,
                    ActionOutcome::Failure { .. } => EpisodeOutcome::Failure,
                    ActionOutcome::Partial { .. } => EpisodeOutcome::Partial,
                }
            }
            _ => EpisodeOutcome::Interrupted,
        }
    }

    /// Calculate significance score for an event
    fn calculate_significance(&self, event: &Event) -> f32 {
        let mut significance: f32 = 0.5; // Base significance

        // Increase significance for goal-related events
        if let EventType::Cognitive { process_type, .. } = &event.event_type {
            if matches!(process_type, CognitiveType::GoalFormation) {
                significance += 0.3;
            }
        }

        // Increase significance for successful actions
        if let EventType::Action { outcome, .. } = &event.event_type {
            match outcome {
                ActionOutcome::Success { .. } => significance += 0.2,
                ActionOutcome::Failure { .. } => significance += 0.1,
                _ => {}
            }
        }

        significance.min(1.0)
    }

    /// Get the timestamp of the last event in an episode
    fn get_last_event_timestamp(&self, episode: &Episode) -> Option<Timestamp> {
        // In a real implementation, we'd look up the last event's timestamp
        // For now, return None as we'd need access to the event storage
        None
    }

    /// Check if periodic consolidation should happen
    fn should_consolidate(&self, agent_id: AgentId, current_time: Timestamp) -> bool {
        if let Some(&last_time) = self.last_consolidation.get(&agent_id) {
            current_time.saturating_sub(last_time) >= self.config.consolidation_interval_ns
        } else {
            false
        }
    }

    /// Consolidate active episode for an agent
    fn consolidate_agent_episode(&mut self, agent_id: AgentId, current_time: Timestamp) -> Option<EpisodeId> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_events::core::{EventType, CognitiveType};
    use agent_db_core::types::current_timestamp;

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
                embeddings: None,
            },
            metadata: HashMap::new(),
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

        let episode_id = detector.process_event(&end_event);
        assert!(episode_id.is_some());
        assert!(detector.get_active_episode(1).is_none());
    }
}
