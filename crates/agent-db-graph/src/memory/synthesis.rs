// crates/agent-db-graph/src/memory/synthesis.rs
//
// Natural-language synthesis helpers for memory formation.

use crate::episodes::{Episode, EpisodeOutcome};
use crate::event_content::{
    extract_action_description, extract_cognitive_summary, extract_communication_summary,
    extract_context_summary, extract_observation_summary,
};
use agent_db_events::core::{ActionOutcome, Event, EventType};

/// Truncate a string to `max_len` chars, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
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
    let mut cognitive_items: Vec<String> = Vec::new();

    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                parameters,
                outcome,
                ..
            } => {
                actions.push(extract_action_description(action_name, parameters, outcome));
            },
            EventType::Observation {
                observation_type,
                data,
                source,
                confidence,
                ..
            } => {
                observations.push(extract_observation_summary(
                    observation_type,
                    data,
                    *confidence,
                    source,
                ));
            },
            EventType::Context {
                text, context_type, ..
            } => {
                context_texts.push(extract_context_summary(text, context_type));
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                communications.push(extract_communication_summary(
                    message_type,
                    *sender,
                    *recipient,
                    content,
                ));
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                cognitive_items.push(extract_cognitive_summary(
                    process_type,
                    input,
                    output,
                    reasoning_trace,
                ));
            },
            EventType::Conversation {
                speaker,
                content,
                category,
            } => {
                // Include conversation messages in memory synthesis
                let truncated = truncate_str(content, 120);
                match category.as_str() {
                    "transaction" => {
                        actions.push(format!("{}: {}", speaker, truncated));
                    },
                    "state_change" => {
                        context_texts.push(format!("{}: {}", speaker, truncated));
                    },
                    "relationship" => {
                        context_texts.push(format!("{}: {}", speaker, truncated));
                    },
                    "preference" => {
                        context_texts.push(format!("{}: {}", speaker, truncated));
                    },
                    _ => {
                        communications.push(format!("{}: {}", speaker, truncated));
                    },
                }
            },
            EventType::CodeReview {
                body, file_path, ..
            } => {
                let truncated = truncate_str(body, 120);
                let ctx = file_path.as_deref().unwrap_or("unknown file");
                context_texts.push(format!("Code review on {}: {}", ctx, truncated));
            },
            EventType::CodeFile {
                file_path,
                language,
                ..
            } => {
                let lang = language.as_deref().unwrap_or("unknown");
                context_texts.push(format!("Code file: {} ({})", file_path, lang));
            },
            _ => {},
        }
    }

    if !context_texts.is_empty() {
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
    if !cognitive_items.is_empty() {
        let shown: Vec<&str> = cognitive_items.iter().take(3).map(|s| s.as_str()).collect();
        parts.push(format!("Reasoning: {}", shown.join("; ")));
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
    let mut cognitive_reasoning: Option<String> = None;

    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                outcome: action_out,
                ..
            } => match action_out {
                ActionOutcome::Success { .. } => {
                    successes += 1;
                    last_success_action = action_name.clone();
                },
                ActionOutcome::Failure { error, .. } => {
                    failures += 1;
                    last_failure_error = error.clone();
                },
                ActionOutcome::Partial { issues, .. } => {
                    causes.push(format!(
                        "Action '{}' partially succeeded with issues: {:?}",
                        action_name, issues
                    ));
                },
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                // Capture cognitive reasoning — often explains *why* actions were taken
                cognitive_reasoning = Some(extract_cognitive_summary(
                    process_type,
                    input,
                    output,
                    reasoning_trace,
                ));
            },
            EventType::Conversation {
                speaker,
                content,
                category,
            } => {
                // Conversation events count as successful actions for causal analysis
                if category == "transaction" {
                    successes += 1;
                    last_success_action = format!("{}:{}", speaker, truncate_str(content, 60));
                }
            },
            _ => {},
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
        },
        EpisodeOutcome::Failure => {
            if !last_failure_error.is_empty() {
                causes.push(format!(
                    "Failed because: {}",
                    truncate_str(&last_failure_error, 200)
                ));
            } else {
                causes.push("Episode ended in failure without a clear action error".to_string());
            }
            if successes > 0 {
                causes.push(format!(
                    "{} action(s) succeeded before the failure occurred",
                    successes
                ));
            }
        },
        EpisodeOutcome::Partial => {
            causes.push(format!(
                "Partial: {} action(s) succeeded, {} failed",
                successes, failures
            ));
        },
        EpisodeOutcome::Interrupted => {
            causes.push("Episode was interrupted before completion".to_string());
        },
    }

    // Include cognitive reasoning if it helps explain the cause
    if let Some(reasoning) = cognitive_reasoning {
        causes.push(format!("Agent reasoning: {}", reasoning));
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
    let mut pivotal_action: Option<(&str, &serde_json::Value, &ActionOutcome)> = None;
    for event in events.iter().rev() {
        if let EventType::Action {
            action_name,
            parameters,
            outcome: action_out,
            ..
        } = &event.event_type
        {
            match (&outcome, action_out) {
                (EpisodeOutcome::Success, ActionOutcome::Success { .. }) => {
                    pivotal_action = Some((action_name, parameters, action_out));
                    break;
                },
                (EpisodeOutcome::Failure, ActionOutcome::Failure { .. }) => {
                    pivotal_action = Some((action_name, parameters, action_out));
                    break;
                },
                _ => {},
            }
        }
    }

    // For conversation-only episodes, extract a conversation-based takeaway
    if pivotal_action.is_none() {
        let conv_events: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let EventType::Conversation {
                    speaker,
                    content,
                    category,
                } = &e.event_type
                {
                    Some((speaker.as_str(), content.as_str(), category.as_str()))
                } else {
                    None
                }
            })
            .collect();

        if !conv_events.is_empty() {
            // Summarize conversation: count categories, pick most relevant
            let tx_count = conv_events
                .iter()
                .filter(|(_, _, c)| *c == "transaction")
                .count();
            let rel_count = conv_events
                .iter()
                .filter(|(_, _, c)| *c == "relationship")
                .count();
            let pref_count = conv_events
                .iter()
                .filter(|(_, _, c)| *c == "preference")
                .count();

            let mut summary_parts = Vec::new();
            if tx_count > 0 {
                summary_parts.push(format!("{} transactions", tx_count));
            }
            if rel_count > 0 {
                summary_parts.push(format!("{} relationships", rel_count));
            }
            if pref_count > 0 {
                summary_parts.push(format!("{} preferences", pref_count));
            }

            if !summary_parts.is_empty() {
                return format!(
                    "Conversation session captured: {}. {} messages processed.",
                    summary_parts.join(", "),
                    conv_events.len()
                );
            } else {
                return format!(
                    "Conversation session with {} messages processed.",
                    conv_events.len()
                );
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
        (EpisodeOutcome::Success, Some((action_name, parameters, outcome))) => {
            let desc = extract_action_description(action_name, parameters, outcome);
            format!(
                "For '{}': {} was the key step that led to success.",
                goal_str, desc
            )
        },
        (EpisodeOutcome::Failure, Some((action_name, parameters, outcome))) => {
            let desc = extract_action_description(action_name, parameters, outcome);
            format!(
                "For '{}': {} — avoid this in similar contexts.",
                goal_str, desc
            )
        },
        (EpisodeOutcome::Success, _) => format!(
            "Successfully completed '{}' with {} actions and significance {:.0}%.",
            goal_str,
            events.len(),
            episode.significance * 100.0
        ),
        (EpisodeOutcome::Failure, _) => {
            format!(
                "Failed '{}' — review approach for this context to avoid repeating.",
                goal_str
            )
        },
        (EpisodeOutcome::Partial, _) => format!(
            "Partially completed '{}' — some actions succeeded, others need improvement.",
            goal_str
        ),
        (EpisodeOutcome::Interrupted, _) => {
            format!(
                "'{}' was interrupted — retry when context is stable.",
                goal_str
            )
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_core::types::{AgentId, SessionId, Timestamp};
    use agent_db_events::core::{CognitiveType, EventContext};
    use serde_json::json;

    fn make_episode(outcome: EpisodeOutcome, significance: f32) -> Episode {
        Episode {
            id: 1,
            episode_version: 1,
            agent_id: AgentId::from(1u64),
            start_event: 1u128,
            end_event: Some(2u128),
            events: vec![1u128, 2u128],
            session_id: SessionId::from(1u64),
            context_signature: 0,
            context: EventContext::default(),
            outcome: Some(outcome),
            start_timestamp: Timestamp::from(1000u64),
            end_timestamp: Some(Timestamp::from(2000u64)),
            significance,
            prediction_error: 0.0,
            self_judged_quality: None,
            salience_score: 0.5,
            last_event_timestamp: Some(Timestamp::from(2000u64)),
            consecutive_outcome_count: 0,
        }
    }

    fn make_event(event_type: EventType) -> Event {
        Event {
            id: 1u128,
            timestamp: Timestamp::from(1000u64),
            agent_id: AgentId::from(1u64),
            agent_type: "test".to_string(),
            session_id: SessionId::from(1u64),
            event_type,
            causality_chain: vec![],
            context: EventContext::default(),
            metadata: Default::default(),
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        }
    }

    #[test]
    fn test_cognitive_events_included() {
        let episode = make_episode(EpisodeOutcome::Success, 0.8);
        let events = vec![
            make_event(EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                input: json!("analyze deployment options"),
                output: json!("use blue-green strategy"),
                reasoning_trace: vec!["compared options".to_string()],
            }),
            make_event(EventType::Action {
                action_name: "deploy".to_string(),
                parameters: json!({}),
                outcome: ActionOutcome::Success {
                    result: json!({"text": "deployed"}),
                },
                duration_ns: 1000,
            }),
        ];

        let summary = synthesize_memory_summary(&episode, &events);
        // Cognitive events should now appear in memory summary
        assert!(
            summary.contains("Reasoning"),
            "Summary should contain cognitive reasoning, got: {}",
            summary
        );
        assert!(summary.contains("blue-green"));
    }

    #[test]
    fn test_action_result_not_raw_json() {
        let episode = make_episode(EpisodeOutcome::Success, 0.8);
        let events = vec![make_event(EventType::Action {
            action_name: "create_user".to_string(),
            parameters: json!({"name": "Alice"}),
            outcome: ActionOutcome::Success {
                result: json!({"created": true}),
            },
            duration_ns: 500,
        })];

        let summary = synthesize_memory_summary(&episode, &events);
        // Should NOT contain raw JSON like {"created":true}
        assert!(
            !summary.contains(r#"{"created":true}"#),
            "Summary should humanize JSON, not dump raw: {}",
            summary
        );
        // Should contain the humanized action description
        assert!(summary.contains("Create User"));
    }

    #[test]
    fn test_causal_note_includes_cognitive() {
        let episode = make_episode(EpisodeOutcome::Success, 0.8);
        let events = vec![
            make_event(EventType::Cognitive {
                process_type: CognitiveType::Planning,
                input: json!("plan approach"),
                output: json!("decided to use caching"),
                reasoning_trace: vec![],
            }),
            make_event(EventType::Action {
                action_name: "deploy".to_string(),
                parameters: json!({}),
                outcome: ActionOutcome::Success {
                    result: json!("ok"),
                },
                duration_ns: 1000,
            }),
        ];

        let causal = synthesize_causal_note(&episode, &events);
        assert!(
            causal.contains("Agent reasoning"),
            "Causal note should include cognitive reasoning, got: {}",
            causal
        );
    }
}
