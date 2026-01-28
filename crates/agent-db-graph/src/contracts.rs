//! Learning output contracts produced by the graph engine.
//!
//! These structs are the boundary between the graph engine and
//! the memory/strategy stores.

use agent_db_core::types::{AgentId, ContextHash, EventId, SessionId, Timestamp};
use agent_db_events::core::{Event, EventType};

use crate::episodes::{Episode, EpisodeOutcome};

#[derive(Debug, Clone)]
pub struct OutcomeSignal {
    pub episode_id: u64,
    pub episode_version: u32,
    pub signal_seq: u32,
    pub source_event_id: Option<EventId>,
    pub outcome: EpisodeOutcome,
    pub emitted_at: Timestamp,
}

#[derive(Debug, Clone)]
pub struct EpisodeRecord {
    pub episode_id: u64,
    pub episode_version: u32,
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub start_event_id: EventId,
    pub end_event_id: Option<EventId>,
    pub context_signature: ContextHash,
    pub goal_bucket_id: u64,
    pub behavior_signature: String,
    pub outcome: EpisodeOutcome,
    pub significance: f32,
    pub salience_score: f32,
}

#[derive(Debug, Clone)]
pub struct AbstractTransition {
    pub state: String,
    pub action: String,
    pub next_state: String,
}

#[derive(Debug, Clone)]
pub struct AbstractTrace {
    pub states: Vec<String>,
    pub actions: Vec<String>,
    pub transitions: Vec<AbstractTransition>,
}

#[derive(Debug, Clone)]
pub struct ContextFeatures {
    pub context_hash: ContextHash,
    pub goal_bucket_id: u64,
    pub goal_ids: Vec<u64>,
    pub env_keys: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LearningOutputs {
    pub episode_record: EpisodeRecord,
    pub abstract_trace: AbstractTrace,
    pub context_features: ContextFeatures,
    pub outcome_signals: Vec<OutcomeSignal>,
}

pub fn build_episode_record(episode: &Episode, events: &[Event]) -> EpisodeRecord {
    let goal_bucket_id = derive_goal_bucket_id(episode);
    let behavior_signature = compute_behavior_signature(events);
    EpisodeRecord {
        episode_id: episode.id,
        episode_version: episode.episode_version,
        agent_id: episode.agent_id,
        session_id: episode.session_id,
        start_event_id: episode.start_event,
        end_event_id: episode.end_event,
        context_signature: episode.context_signature,
        goal_bucket_id,
        behavior_signature,
        outcome: episode
            .outcome
            .clone()
            .unwrap_or(EpisodeOutcome::Interrupted),
        significance: episode.significance,
        salience_score: episode.salience_score,
    }
}

pub fn build_learning_outputs(episode: &Episode, events: &[Event]) -> LearningOutputs {
    let episode_record = build_episode_record(episode, events);
    let abstract_trace = build_abstract_trace(events);
    let context_features = build_context_features(episode);
    let outcome = episode
        .outcome
        .clone()
        .unwrap_or(EpisodeOutcome::Interrupted);
    let outcome_signals = vec![OutcomeSignal {
        episode_id: episode.id,
        episode_version: 1,
        signal_seq: 1,
        source_event_id: episode.end_event,
        outcome,
        emitted_at: episode.end_timestamp.unwrap_or(episode.start_timestamp),
    }];

    LearningOutputs {
        episode_record,
        abstract_trace,
        context_features,
        outcome_signals,
    }
}

fn build_context_features(episode: &Episode) -> ContextFeatures {
    let goal_ids = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.id)
        .collect::<Vec<_>>();
    let env_keys = episode
        .context
        .environment
        .variables
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let mut sorted_goals = goal_ids.clone();
    sorted_goals.sort();
    let goal_bucket_id = sorted_goals.first().copied().unwrap_or(0);

    ContextFeatures {
        context_hash: episode.context_signature,
        goal_bucket_id,
        goal_ids,
        env_keys,
    }
}

fn derive_goal_bucket_id(episode: &Episode) -> u64 {
    let mut goals: Vec<u64> = episode
        .context
        .active_goals
        .iter()
        .map(|goal| goal.id)
        .collect();
    goals.sort();
    goals.first().copied().unwrap_or(0)
}

fn compute_behavior_signature(events: &[Event]) -> String {
    let skeleton = build_behavior_skeleton(events);
    let joined = skeleton.join(">");
    format!("{:x}", hash_str(&joined))
}

fn build_behavior_skeleton(events: &[Event]) -> Vec<String> {
    let mut skeleton = Vec::new();
    for event in events {
        match &event.event_type {
            EventType::Observation { .. } => skeleton.push("Observe".to_string()),
            EventType::Cognitive { process_type, .. } => {
                skeleton.push(format!("Think:{:?}", process_type));
            },
            EventType::Action { action_name, .. } => {
                skeleton.push(format!("Act:{}", action_name));
            },
            EventType::Communication { .. } => skeleton.push("Communicate".to_string()),
            EventType::Learning { .. } => skeleton.push("Learn".to_string()),
            EventType::Context { context_type, .. } => {
                skeleton.push(format!("Context:{}", context_type));
            },
        }
    }
    skeleton
}

fn build_abstract_trace(events: &[Event]) -> AbstractTrace {
    let states = build_behavior_skeleton(events);
    let mut actions = Vec::new();
    let mut transitions = Vec::new();

    for window in events.windows(2) {
        let from_state = state_from_event(&window[0]);
        let to_state = state_from_event(&window[1]);
        let action = action_from_event(&window[1]);
        actions.push(action.clone());
        transitions.push(AbstractTransition {
            state: from_state,
            action,
            next_state: to_state,
        });
    }

    AbstractTrace {
        states,
        actions,
        transitions,
    }
}

fn state_from_event(event: &Event) -> String {
    match &event.event_type {
        EventType::Observation { .. } => "Observe".to_string(),
        EventType::Cognitive { process_type, .. } => format!("Think:{:?}", process_type),
        EventType::Action { .. } => "Act".to_string(),
        EventType::Communication { .. } => "Communicate".to_string(),
        EventType::Learning { .. } => "Learn".to_string(),
        EventType::Context { context_type, .. } => format!("Context:{}", context_type),
    }
}

fn action_from_event(event: &Event) -> String {
    match &event.event_type {
        EventType::Action { action_name, .. } => action_name.clone(),
        EventType::Cognitive { process_type, .. } => format!("Think:{:?}", process_type),
        EventType::Observation {
            observation_type, ..
        } => format!("Observe:{observation_type}"),
        EventType::Communication { message_type, .. } => format!("Comm:{message_type}"),
        EventType::Learning { .. } => "Learn".to_string(),
        EventType::Context { context_type, .. } => format!("Context:{}", context_type),
    }
}

fn hash_str(value: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
