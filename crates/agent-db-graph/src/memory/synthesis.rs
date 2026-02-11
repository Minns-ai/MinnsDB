// crates/agent-db-graph/src/memory/synthesis.rs
//
// Natural-language synthesis helpers for memory formation.

use crate::episodes::{Episode, EpisodeOutcome};
use agent_db_events::core::{ActionOutcome, Event, EventType};

/// Truncate a string to `max_len` chars, appending "…" if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}…", &s[..max_len])
    } else {
        s.to_string()
    }
}

/// Truncate a `serde_json::Value` to a readable string of at most `max_len` chars.
fn truncate_value(v: &serde_json::Value, max_len: usize) -> String {
    // For strings, extract the inner string to avoid extra quotes
    let raw = match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    truncate_str(&raw, max_len)
}

/// Synthesize a natural language summary from an episode and its events.
///
/// The summary is designed to be directly usable by an LLM for retrieval-augmented
/// generation — it describes *what happened*, *what was done*, and *how it ended*
/// in plain English.
pub fn synthesize_memory_summary(episode: &Episode, events: &[Event]) -> String {
    let mut parts: Vec<String> = Vec::new();

    // 1. Goal context
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    if !goals.is_empty() {
        if goals.len() == 1 {
            parts.push(format!("Goal: {}", goals[0]));
        } else {
            parts.push(format!("Goals: {}", goals.join("; ")));
        }
    }

    // 2. Environment context (user, intent, key vars)
    let env_vars = &episode.context.environment.variables;
    let mut env_parts = Vec::new();
    if let Some(user) = env_vars.get("user_id").or_else(|| env_vars.get("user")) {
        env_parts.push(format!("user={}", user));
    }
    if let Some(intent) = env_vars
        .get("intent_type")
        .or_else(|| env_vars.get("intent"))
    {
        env_parts.push(format!("intent={}", intent));
    }
    // Include up to 3 other notable variables
    for (k, v) in env_vars.iter() {
        if env_parts.len() >= 5 {
            break;
        }
        if k != "user_id" && k != "user" && k != "intent_type" && k != "intent" {
            env_parts.push(format!("{}={}", k, v));
        }
    }
    if !env_parts.is_empty() {
        parts.push(format!("Context: {}", env_parts.join(", ")));
    }

    // 3. Walk events and extract narrative
    let mut actions: Vec<String> = Vec::new();
    let mut observations: Vec<String> = Vec::new();
    let mut context_texts: Vec<String> = Vec::new();
    let mut communications: Vec<String> = Vec::new();

    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                outcome,
                ..
            } => {
                let outcome_str = match outcome {
                    ActionOutcome::Success { result } => {
                        format!("succeeded: {}", truncate_value(result, 120))
                    }
                    ActionOutcome::Failure { error, .. } => {
                        format!("failed: {}", truncate_str(error, 120))
                    }
                    ActionOutcome::Partial { result, issues } => {
                        format!(
                            "partial: {} (issues: {:?})",
                            truncate_value(result, 80),
                            issues
                        )
                    }
                };
                actions.push(format!("'{}' {}", action_name, outcome_str));
            }
            EventType::Observation {
                observation_type,
                data,
                source,
                confidence,
                ..
            } => {
                observations.push(format!(
                    "[{}] from '{}' (conf {:.0}%): {}",
                    observation_type,
                    source,
                    confidence * 100.0,
                    truncate_value(data, 150)
                ));
            }
            EventType::Context { text, context_type, .. } => {
                context_texts.push(format!("[{}] {}", context_type, truncate_str(text, 200)));
            }
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                communications.push(format!(
                    "{} agent {} -> agent {}: {}",
                    message_type,
                    sender,
                    recipient,
                    truncate_value(content, 150)
                ));
            }
            _ => {} // Cognitive & Learning are internal machinery, skip for narrative
        }
    }

    if !context_texts.is_empty() {
        // Limit to first 3 to keep summary tight
        let shown: Vec<&str> = context_texts.iter().take(3).map(|s| s.as_str()).collect();
        parts.push(format!("Input: {}", shown.join(" | ")));
    }
    if !actions.is_empty() {
        let shown: Vec<&str> = actions.iter().take(5).map(|s| s.as_str()).collect();
        parts.push(format!("Actions: {}", shown.join("; ")));
    }
    if !observations.is_empty() {
        let shown: Vec<&str> = observations.iter().take(3).map(|s| s.as_str()).collect();
        parts.push(format!("Observations: {}", shown.join("; ")));
    }
    if !communications.is_empty() {
        let shown: Vec<&str> = communications.iter().take(2).map(|s| s.as_str()).collect();
        parts.push(format!("Comms: {}", shown.join("; ")));
    }

    // 4. Outcome
    let outcome_str = match &episode.outcome {
        Some(EpisodeOutcome::Success) => "Outcome: Success",
        Some(EpisodeOutcome::Failure) => "Outcome: Failure",
        Some(EpisodeOutcome::Partial) => "Outcome: Partial success",
        Some(EpisodeOutcome::Interrupted) | None => "Outcome: Interrupted/unknown",
    };
    parts.push(outcome_str.to_string());

    // 5. Stats
    parts.push(format!(
        "({} events, significance {:.0}%)",
        events.len(),
        episode.significance * 100.0
    ));

    if parts.is_empty() {
        format!(
            "Episode {} for agent {} — no detailed events available.",
            episode.id, episode.agent_id
        )
    } else {
        parts.join(". ")
    }
}

/// Synthesize a causal explanation for why the episode succeeded or failed.
pub fn synthesize_causal_note(episode: &Episode, events: &[Event]) -> String {
    let outcome = episode
        .outcome
        .clone()
        .unwrap_or(EpisodeOutcome::Interrupted);

    let mut causes: Vec<String> = Vec::new();

    // Analyze action outcomes for causal signal
    let mut successes = 0u32;
    let mut failures = 0u32;
    let mut last_failure_error = String::new();
    let mut last_success_action = String::new();

    for event in events {
        if let EventType::Action {
            action_name,
            outcome: action_out,
            ..
        } = &event.event_type
        {
            match action_out {
                ActionOutcome::Success { .. } => {
                    successes += 1;
                    last_success_action = action_name.clone();
                }
                ActionOutcome::Failure { error, .. } => {
                    failures += 1;
                    last_failure_error = error.clone();
                }
                ActionOutcome::Partial { issues, .. } => {
                    causes.push(format!(
                        "Action '{}' partially succeeded with issues: {:?}",
                        action_name, issues
                    ));
                }
            }
        }
    }

    match outcome {
        EpisodeOutcome::Success => {
            if failures == 0 {
                causes.push(format!(
                    "All {} actions succeeded cleanly (last: '{}')",
                    successes, last_success_action
                ));
            } else {
                causes.push(format!(
                    "Recovered from {} failure(s) — final action '{}' succeeded",
                    failures, last_success_action
                ));
            }
            // Goal context
            for goal in &episode.context.active_goals {
                if goal.progress >= 0.8 && !goal.description.is_empty() {
                    causes.push(format!(
                        "Goal '{}' reached {:.0}% progress",
                        goal.description,
                        goal.progress * 100.0
                    ));
                }
            }
        }
        EpisodeOutcome::Failure => {
            if !last_failure_error.is_empty() {
                causes.push(format!("Failed because: {}", truncate_str(&last_failure_error, 200)));
            } else {
                causes.push("Episode ended in failure without a clear action error".to_string());
            }
            if successes > 0 {
                causes.push(format!(
                    "{} action(s) succeeded before the failure occurred",
                    successes
                ));
            }
        }
        EpisodeOutcome::Partial => {
            causes.push(format!(
                "Partial: {} action(s) succeeded, {} failed",
                successes, failures
            ));
        }
        EpisodeOutcome::Interrupted => {
            causes.push("Episode was interrupted before completion".to_string());
        }
    }

    if episode.prediction_error > 0.3 {
        causes.push(format!(
            "This was a surprising outcome (prediction error {:.0}%)",
            episode.prediction_error * 100.0
        ));
    }

    if causes.is_empty() {
        "No causal signal extracted from events.".to_string()
    } else {
        causes.join(". ")
    }
}

/// Synthesize the single most important lesson from this episode.
pub fn synthesize_takeaway(episode: &Episode, events: &[Event]) -> String {
    let outcome = episode
        .outcome
        .clone()
        .unwrap_or(EpisodeOutcome::Interrupted);

    // Find the pivotal action (last success for successful episodes, last failure for failed)
    let mut pivotal_action: Option<(&str, &ActionOutcome)> = None;
    for event in events.iter().rev() {
        if let EventType::Action {
            action_name,
            outcome: action_out,
            ..
        } = &event.event_type
        {
            match (&outcome, action_out) {
                (EpisodeOutcome::Success, ActionOutcome::Success { .. }) => {
                    pivotal_action = Some((action_name, action_out));
                    break;
                }
                (EpisodeOutcome::Failure, ActionOutcome::Failure { .. }) => {
                    pivotal_action = Some((action_name, action_out));
                    break;
                }
                _ => {}
            }
        }
    }

    // Build takeaway
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    let goal_str = if goals.is_empty() {
        "this task".to_string()
    } else {
        goals[0].to_string()
    };

    match (&outcome, pivotal_action) {
        (EpisodeOutcome::Success, Some((action, ActionOutcome::Success { result }))) => format!(
            "For '{}': action '{}' was the key step that led to success (result: {}).",
            goal_str,
            action,
            truncate_value(result, 100)
        ),
        (EpisodeOutcome::Failure, Some((action, ActionOutcome::Failure { error, .. }))) => format!(
            "For '{}': action '{}' caused failure — {}. Avoid this in similar contexts.",
            goal_str,
            action,
            truncate_str(error, 100)
        ),
        (EpisodeOutcome::Success, _) => format!(
            "Successfully completed '{}' with {} actions and significance {:.0}%.",
            goal_str,
            events.len(),
            episode.significance * 100.0
        ),
        (EpisodeOutcome::Failure, _) => {
            format!("Failed '{}' — review approach for this context to avoid repeating.", goal_str)
        }
        (EpisodeOutcome::Partial, _) => format!(
            "Partially completed '{}' — some actions succeeded, others need improvement.",
            goal_str
        ),
        (EpisodeOutcome::Interrupted, _) => {
            format!("'{}' was interrupted — retry when context is stable.", goal_str)
        }
    }
}
