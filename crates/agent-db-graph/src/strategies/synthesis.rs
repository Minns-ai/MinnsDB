// crates/agent-db-graph/src/strategies/synthesis.rs
//
// Natural language synthesis functions for strategy summaries, playbooks,
// when-to-use, when-not-to-use, failure modes, and counterfactuals.

use crate::episodes::{Episode, EpisodeOutcome};
use crate::event_content::{
    extract_action_description, extract_cognitive_summary, extract_communication_summary,
    extract_context_summary, extract_observation_summary,
};
use agent_db_events::core::{ActionOutcome, Event, EventType};

use super::types::*;

/// Truncate a string to `max_len` bytes (on a char boundary), appending "..." if truncated.
pub(crate) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    } else {
        s.to_string()
    }
}

/// Synthesize a natural language strategy summary from an episode and its events.
///
/// The summary describes *when* to use this strategy, *what steps* to follow,
/// *what success looks like*, and *what to avoid* — all in plain English an LLM can use.
pub fn synthesize_strategy_summary(
    strategy_type: &StrategyType,
    outcome: &EpisodeOutcome,
    episode: &Episode,
    events: &[Event],
    success_indicators: &[String],
    failure_patterns: &[String],
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // 1. Strategy type framing — derive from goals + actions, not generic templates
    let goal_desc: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    match strategy_type {
        StrategyType::Positive => {
            if !goal_desc.is_empty() {
                parts.push(format!("Strategy for achieving: {}", goal_desc.join("; ")));
            } else {
                parts.push("Proven approach from a successful episode.".to_string());
            }
        },
        StrategyType::Constraint => {
            if !goal_desc.is_empty() {
                parts.push(format!(
                    "Constraint: avoid this pattern when pursuing: {}",
                    goal_desc.join("; ")
                ));
            } else {
                parts.push("Constraint: avoid this pattern (led to failure).".to_string());
            }
        },
    }

    // 2. When — goals and context
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    if !goals.is_empty() {
        parts.push(format!("When: {}", goals.join("; ")));
    }

    let env_vars = &episode.context.environment.variables;
    let mut context_hints = Vec::new();
    if let Some(intent) = env_vars
        .get("intent_type")
        .or_else(|| env_vars.get("intent"))
    {
        context_hints.push(format!("intent={}", intent));
    }
    for (k, v) in env_vars.iter().take(3) {
        if k != "intent_type" && k != "intent" && k != "user_id" && k != "user" {
            context_hints.push(format!("{}={}", k, v));
        }
    }
    if !context_hints.is_empty() {
        parts.push(format!("Context: {}", context_hints.join(", ")));
    }

    // 3. Steps — walk events and produce a human-readable sequence
    let mut steps: Vec<String> = Vec::new();
    let mut step_num = 1u32;
    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                parameters,
                outcome: action_outcome,
                ..
            } => {
                let desc = extract_action_description(action_name, parameters, action_outcome);
                steps.push(format!("{}. {}", step_num, desc));
                step_num += 1;
            },
            EventType::Context {
                context_type, text, ..
            } => {
                steps.push(format!(
                    "{}. Receive {}",
                    step_num,
                    extract_context_summary(text, context_type)
                ));
                step_num += 1;
            },
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                steps.push(format!(
                    "{}. Observe {}",
                    step_num,
                    extract_observation_summary(observation_type, data, *confidence, source)
                ));
                step_num += 1;
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                steps.push(format!(
                    "{}. {}",
                    step_num,
                    extract_cognitive_summary(process_type, input, output, reasoning_trace)
                ));
                step_num += 1;
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                steps.push(format!(
                    "{}. {}",
                    step_num,
                    extract_communication_summary(message_type, *sender, *recipient, content,)
                ));
                step_num += 1;
            },
            _ => {},
        }
    }
    if !steps.is_empty() {
        // Limit to first 8 steps to keep summary digestible
        let shown: Vec<&str> = steps.iter().take(8).map(|s| s.as_str()).collect();
        let suffix = if steps.len() > 8 {
            format!(" ... ({} more steps)", steps.len() - 8)
        } else {
            String::new()
        };
        parts.push(format!("Steps: {}{}", shown.join(" → "), suffix));
    }

    // 4. Success indicators
    if !success_indicators.is_empty() {
        let shown: Vec<&str> = success_indicators
            .iter()
            .take(3)
            .map(|s| s.as_str())
            .collect();
        parts.push(format!("Success looks like: {}", shown.join("; ")));
    }

    // 5. Failure patterns
    if !failure_patterns.is_empty() {
        let shown: Vec<&str> = failure_patterns
            .iter()
            .take(3)
            .map(|s| s.as_str())
            .collect();
        parts.push(format!("Avoid: {}", shown.join("; ")));
    }

    // 6. Outcome + stats
    let outcome_label = match outcome {
        EpisodeOutcome::Success => "succeeded",
        EpisodeOutcome::Failure => "failed",
        EpisodeOutcome::Partial => "partially succeeded",
        EpisodeOutcome::Interrupted => "was interrupted",
    };
    parts.push(format!(
        "This episode {} ({} events, significance {:.0}%).",
        outcome_label,
        events.len(),
        episode.significance * 100.0
    ));

    parts.join(" ")
}

// ============================================================================
// 10x / 100x Synthesis Functions
// ============================================================================

/// Generate "when to use" in natural language.
pub(crate) fn synthesize_when_to_use(
    strategy_type: &StrategyType,
    episode: &Episode,
    events: &[Event],
) -> String {
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();

    let env_hints: Vec<String> = episode
        .context
        .environment
        .variables
        .iter()
        .filter(|(k, _)| k != &"user_id" && k != &"user")
        .take(3)
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();

    // Find top cognitive reasoning if available
    let cognitive_hint: Option<String> = events.iter().find_map(|e| {
        if let EventType::Cognitive {
            process_type,
            input,
            output,
            reasoning_trace,
        } = &e.event_type
        {
            Some(extract_cognitive_summary(
                process_type,
                input,
                output,
                reasoning_trace,
            ))
        } else {
            None
        }
    });

    match strategy_type {
        StrategyType::Positive => {
            let mut parts = Vec::new();
            if !goals.is_empty() {
                parts.push(format!("Use when the goal is: {}", goals.join("; ")));
            }
            if !env_hints.is_empty() {
                parts.push(format!("Context matches: {}", env_hints.join(", ")));
            }
            if let Some(reasoning) = cognitive_hint {
                parts.push(format!("Agent reasoning: {}", reasoning));
            }
            if parts.is_empty() {
                parts.push("Use when facing a similar task context.".to_string());
            }
            parts.join(". ")
        },
        StrategyType::Constraint => {
            let mut parts = Vec::new();
            if !goals.is_empty() {
                parts.push(format!("Watch out when the goal is: {}", goals.join("; ")));
            }
            if let Some(reasoning) = cognitive_hint {
                parts.push(format!(
                    "Agent reasoning that led to failure: {}",
                    reasoning
                ));
            }
            parts.push(
                "Applies when the agent is about to repeat a pattern that previously failed"
                    .to_string(),
            );
            parts.join(". ")
        },
    }
}

/// Generate "when NOT to use" in natural language.
pub(crate) fn synthesize_when_not_to_use(
    strategy_type: &StrategyType,
    episode: &Episode,
    events: &[Event],
) -> String {
    match strategy_type {
        StrategyType::Positive => {
            let mut reasons = Vec::new();
            // Identify specific failure points and their context
            for event in events {
                if let EventType::Action {
                    action_name,
                    parameters,
                    outcome: ActionOutcome::Failure { error, .. },
                    ..
                } = &event.event_type
                {
                    let desc = extract_action_description(
                        action_name,
                        parameters,
                        &ActionOutcome::Failure {
                            error: error.clone(),
                            error_code: 0,
                        },
                    );
                    reasons.push(format!(
                        "Fragile when: {}",
                        truncate_str(&desc, 120)
                    ));
                }
            }
            if episode.significance < 0.3 {
                reasons.push("Low-significance episode — may not generalize to other contexts".to_string());
            }
            let goals: Vec<&str> = episode
                .context
                .active_goals
                .iter()
                .map(|g| g.description.as_str())
                .filter(|d| !d.is_empty())
                .collect();
            if !goals.is_empty() {
                reasons.push(format!(
                    "Do not use when goals differ significantly from: {}",
                    goals.join("; ")
                ));
            } else {
                reasons.push(
                    "Do not use when the context significantly differs from the original episode"
                        .to_string(),
                );
            }
            reasons.join(". ")
        },
        StrategyType::Constraint => {
            "Safe to ignore when a newer strategy explicitly supersedes this constraint, or when the failure pattern is no longer applicable to the current context."
                .to_string()
        },
    }
}

/// Extract known failure modes from event data.
pub(crate) fn synthesize_failure_modes(events: &[Event]) -> Vec<String> {
    let mut modes = Vec::new();
    for event in events {
        if let EventType::Action {
            action_name,
            parameters,
            outcome: outcome @ ActionOutcome::Failure { .. },
            ..
        } = &event.event_type
        {
            modes.push(extract_action_description(action_name, parameters, outcome));
        }
        if let EventType::Action {
            action_name,
            parameters,
            outcome: outcome @ ActionOutcome::Partial { .. },
            ..
        } = &event.event_type
        {
            modes.push(extract_action_description(action_name, parameters, outcome));
        }
    }
    modes.truncate(5); // Keep top 5
    modes
}

/// Build an executable playbook from the event sequence (100x).
pub(crate) fn build_playbook(events: &[Event], strategy_type: &StrategyType) -> Vec<PlaybookStep> {
    let mut steps = Vec::new();
    let mut step_num = 1u32;

    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                parameters,
                outcome: action_outcome,
                ..
            } => {
                let mut branches = Vec::new();
                let mut recovery = String::new();
                let action_desc =
                    extract_action_description(action_name, parameters, action_outcome);

                match action_outcome {
                    ActionOutcome::Failure { error, .. } => {
                        recovery = format!(
                            "On failure ({}): retry or use alternative approach",
                            truncate_str(error, 80)
                        );
                    },
                    ActionOutcome::Partial { issues, .. } => {
                        recovery = format!(
                            "On partial success: address {:?}",
                            issues.iter().take(2).cloned().collect::<Vec<_>>()
                        );
                    },
                    ActionOutcome::Success { .. } => {
                        // For constraint strategies, add a branch to avoid this action
                        if *strategy_type == StrategyType::Constraint {
                            branches.push(PlaybookBranch {
                                condition: "If this pattern appears".to_string(),
                                action: format!("Skip '{}' and use alternative", action_name),
                                next_step_id: None,
                            });
                        }
                    },
                }

                steps.push(PlaybookStep {
                    step: step_num,
                    action: action_desc,
                    condition: String::new(),
                    skip_if: String::new(),
                    branches,
                    recovery,
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Context {
                context_type, text, ..
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: format!("Receive {}", extract_context_summary(text, context_type)),
                    condition: String::new(),
                    skip_if: "No input available".to_string(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: format!(
                        "Observe {}",
                        extract_observation_summary(observation_type, data, *confidence, source)
                    ),
                    condition: String::new(),
                    skip_if: String::new(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: extract_cognitive_summary(process_type, input, output, reasoning_trace),
                    condition: String::new(),
                    skip_if: String::new(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: extract_communication_summary(
                        message_type,
                        *sender,
                        *recipient,
                        content,
                    ),
                    condition: String::new(),
                    skip_if: String::new(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            _ => {},
        }
    }

    // Cap at 12 steps
    steps.truncate(12);
    steps
}

/// Generate a counterfactual: what would have happened differently.
pub(crate) fn synthesize_counterfactual(outcome: &EpisodeOutcome, events: &[Event]) -> String {
    // Find failure points with descriptive context
    let failures: Vec<String> = events
        .iter()
        .filter_map(|e| {
            if let EventType::Action {
                action_name,
                parameters,
                outcome: outcome @ ActionOutcome::Failure { .. },
                ..
            } = &e.event_type
            {
                Some(extract_action_description(action_name, parameters, outcome))
            } else {
                None
            }
        })
        .collect();

    // Find cognitive reasoning that explains what was tried
    let reasoning: Option<String> = events.iter().find_map(|e| {
        if let EventType::Cognitive {
            process_type,
            input,
            output,
            reasoning_trace,
        } = &e.event_type
        {
            Some(extract_cognitive_summary(
                process_type,
                input,
                output,
                reasoning_trace,
            ))
        } else {
            None
        }
    });

    let mut result = match outcome {
        EpisodeOutcome::Success => {
            if failures.is_empty() {
                "All actions succeeded; no obvious alternative path needed.".to_string()
            } else {
                format!(
                    "Despite failures ({}), the episode recovered. Skipping those steps could have been faster.",
                    failures.iter().take(2).cloned().collect::<Vec<_>>().join("; ")
                )
            }
        },
        EpisodeOutcome::Failure => {
            if failures.is_empty() {
                "Episode failed without clear action errors; the approach may need rethinking."
                    .to_string()
            } else {
                format!(
                    "If these had been handled differently: {}. Retry, alternative, or skip could lead to success.",
                    failures.iter().take(2).cloned().collect::<Vec<_>>().join("; ")
                )
            }
        },
        EpisodeOutcome::Partial => {
            "A more conservative approach (smaller steps, more validation) could have improved completeness.".to_string()
        },
        EpisodeOutcome::Interrupted => {
            "Ensuring preconditions and resources before starting would reduce interruption risk.".to_string()
        },
    };

    if let Some(reasoning) = reasoning {
        result.push_str(&format!(" Agent's reasoning: {}", reasoning));
    }

    result
}
