// crates/agent-db-graph/src/event_content.rs
//
// Shared utility functions that extract human-readable content from event data.
// Used by both synthesis (memory/strategy) and LLM refinement pipelines.

use agent_db_events::core::{ActionOutcome, CognitiveType, Event, EventType};

/// Humanize an action name: "cognitive_plan" -> "Cognitive Plan", "api_call" -> "API Call"
fn humanize_action_name(name: &str) -> String {
    name.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    format!("{}{}", upper, chars.as_str())
                },
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract a readable value from a JSON object, preferring known semantic keys.
/// For objects: extracts known key fields (text, message, content, result, error, query, prompt, description).
/// Falls back to compact key=value pairs, not raw JSON dumps.
pub fn humanize_json_value(value: &serde_json::Value, max_len: usize) -> String {
    match value {
        serde_json::Value::Null => "none".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => truncate_at_boundary(s, max_len),
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                return "[]".to_string();
            }
            let items: Vec<String> = arr
                .iter()
                .take(3)
                .map(|v| humanize_json_value(v, max_len / 3))
                .collect();
            let suffix = if arr.len() > 3 {
                format!(" (+{} more)", arr.len() - 3)
            } else {
                String::new()
            };
            format!("[{}{}]", items.join(", "), suffix)
        },
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                return "{}".to_string();
            }
            // Try known semantic keys first
            const SEMANTIC_KEYS: &[&str] = &[
                "text",
                "message",
                "content",
                "result",
                "error",
                "query",
                "prompt",
                "description",
                "answer",
                "response",
                "output",
                "summary",
            ];
            for key in SEMANTIC_KEYS {
                if let Some(v) = map.get(*key) {
                    let extracted = humanize_json_value(v, max_len);
                    if !extracted.is_empty() && extracted != "none" {
                        return extracted;
                    }
                }
            }
            // Fall back to compact key=value pairs
            let pairs: Vec<String> = map
                .iter()
                .take(4)
                .map(|(k, v)| {
                    let val = humanize_json_value(v, 40);
                    format!("{}={}", k, val)
                })
                .collect();
            let result = pairs.join(", ");
            truncate_at_boundary(&result, max_len)
        },
    }
}

/// Truncate a string intelligently at sentence or word boundaries.
fn truncate_at_boundary(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    // Try to break at a sentence boundary
    let slice = &s[..max_len];
    if let Some(pos) = slice.rfind(". ") {
        return format!("{}.", &s[..pos]);
    }
    // Try word boundary
    if let Some(pos) = slice.rfind(' ') {
        return format!("{}...", &s[..pos]);
    }
    format!("{}...", slice)
}

/// Extract a human-readable description of an action and its outcome.
pub fn extract_action_description(
    action_name: &str,
    parameters: &serde_json::Value,
    outcome: &ActionOutcome,
) -> String {
    let name = humanize_action_name(action_name);

    // Extract key parameter info
    let param_hint = humanize_json_value(parameters, 80);
    let param_str = if param_hint == "{}" || param_hint == "none" {
        String::new()
    } else {
        format!(" ({})", param_hint)
    };

    match outcome {
        ActionOutcome::Success { result } => {
            let result_str = humanize_json_value(result, 100);
            format!("{}{} -> {}", name, param_str, result_str)
        },
        ActionOutcome::Failure { error, .. } => {
            format!(
                "{}{} FAILED: {}",
                name,
                param_str,
                truncate_at_boundary(error, 100)
            )
        },
        ActionOutcome::Partial { result, issues } => {
            let result_str = humanize_json_value(result, 80);
            let issue_str = issues
                .iter()
                .take(2)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
            format!(
                "{}{} partially succeeded: {} (issues: {})",
                name, param_str, result_str, issue_str
            )
        },
    }
}

/// Extract a human-readable summary from a Cognitive event.
pub fn extract_cognitive_summary(
    process_type: &CognitiveType,
    input: &serde_json::Value,
    output: &serde_json::Value,
    reasoning_trace: &[String],
) -> String {
    let type_label = match process_type {
        CognitiveType::GoalFormation => "Goal Formation",
        CognitiveType::Planning => "Planning",
        CognitiveType::Reasoning => "Reasoning",
        CognitiveType::MemoryRetrieval => "Memory Retrieval",
        CognitiveType::LearningUpdate => "Learning Update",
    };

    let input_str = humanize_json_value(input, 80);
    let output_str = humanize_json_value(output, 80);

    let mut result = if input_str == "none" || input_str == "{}" {
        format!("[{}] -> {}", type_label, output_str)
    } else {
        format!("[{}] '{}' -> '{}'", type_label, input_str, output_str)
    };

    // Include first 2 reasoning steps if present
    if !reasoning_trace.is_empty() {
        let steps: Vec<&str> = reasoning_trace.iter().take(2).map(|s| s.as_str()).collect();
        result.push_str(&format!(" (reasoning: {})", steps.join(" -> ")));
    }

    result
}

/// Extract a human-readable summary from an Observation event.
pub fn extract_observation_summary(
    observation_type: &str,
    data: &serde_json::Value,
    confidence: f32,
    source: &str,
) -> String {
    let data_str = humanize_json_value(data, 120);
    format!(
        "[{}] from '{}' ({:.0}% confidence): {}",
        observation_type,
        source,
        confidence * 100.0,
        data_str
    )
}

/// Extract a human-readable summary from a Context event.
pub fn extract_context_summary(text: &str, context_type: &str) -> String {
    format!("[{}] {}", context_type, truncate_at_boundary(text, 200))
}

/// Extract a human-readable summary from a Communication event.
pub fn extract_communication_summary(
    message_type: &str,
    sender: u64,
    recipient: u64,
    content: &serde_json::Value,
) -> String {
    let content_str = humanize_json_value(content, 150);
    format!(
        "{} agent {} -> agent {}: {}",
        message_type, sender, recipient, content_str
    )
}

/// Build a full multi-line narrative from a set of events.
/// This is designed for LLM refinement — includes ALL event types with rich detail.
pub fn build_event_narrative(events: &[Event]) -> String {
    let mut lines: Vec<String> = Vec::new();

    for (i, event) in events.iter().enumerate() {
        let line = match &event.event_type {
            EventType::Action {
                action_name,
                parameters,
                outcome,
                ..
            } => {
                let desc = extract_action_description(action_name, parameters, outcome);
                format!("{}. Action: {}", i + 1, desc)
            },
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                let desc = extract_observation_summary(observation_type, data, *confidence, source);
                format!("{}. Observation: {}", i + 1, desc)
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                let desc = extract_cognitive_summary(process_type, input, output, reasoning_trace);
                format!("{}. Cognitive: {}", i + 1, desc)
            },
            EventType::Context {
                text, context_type, ..
            } => {
                let desc = extract_context_summary(text, context_type);
                format!("{}. Context: {}", i + 1, desc)
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                let desc = extract_communication_summary(
                    message_type,
                    *sender,
                    *recipient,
                    content,
                );
                format!("{}. Communication: {}", i + 1, desc)
            },
            EventType::Learning { event } => {
                format!("{}. Learning: {:?}", i + 1, event)
            },
        };
        lines.push(line);
    }

    if lines.is_empty() {
        "No events recorded.".to_string()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_humanize_json_value_extracts_known_keys() {
        let obj = json!({"text": "hello world", "id": 42});
        assert_eq!(humanize_json_value(&obj, 200), "hello world");

        let obj2 = json!({"message": "deployment succeeded"});
        assert_eq!(humanize_json_value(&obj2, 200), "deployment succeeded");
    }

    #[test]
    fn test_humanize_json_value_falls_back() {
        let obj = json!({"status_code": 200, "latency_ms": 42});
        let result = humanize_json_value(&obj, 200);
        assert!(result.contains("status_code="));
        assert!(result.contains("latency_ms="));
        // Should NOT be raw JSON like {"status_code":200,...}
        assert!(!result.starts_with('{'));
    }

    #[test]
    fn test_humanize_json_value_string() {
        let val = json!("simple string");
        assert_eq!(humanize_json_value(&val, 200), "simple string");
    }

    #[test]
    fn test_humanize_json_value_truncates() {
        let long = "a".repeat(300);
        let val = json!(long);
        let result = humanize_json_value(&val, 50);
        assert!(result.len() <= 55); // Allow for "..." suffix
    }

    #[test]
    fn test_extract_action_description_success() {
        let params = json!({"query": "SELECT * FROM users"});
        let outcome = ActionOutcome::Success {
            result: json!({"text": "found 42 users"}),
        };
        let desc = extract_action_description("database_query", &params, &outcome);
        assert!(desc.contains("Database Query"));
        assert!(desc.contains("SELECT * FROM users"));
        assert!(desc.contains("found 42 users"));
    }

    #[test]
    fn test_extract_action_description_failure() {
        let params = json!({});
        let outcome = ActionOutcome::Failure {
            error: "connection timeout".to_string(),
            error_code: 408,
        };
        let desc = extract_action_description("api_call", &params, &outcome);
        assert!(desc.contains("Api Call"));
        assert!(desc.contains("FAILED"));
        assert!(desc.contains("connection timeout"));
    }

    #[test]
    fn test_extract_cognitive_summary() {
        let input = json!("how to deploy");
        let output = json!("use blue-green strategy");
        let trace = vec![
            "analyzed deployment options".to_string(),
            "compared blue-green vs rolling".to_string(),
            "selected blue-green for safety".to_string(),
        ];
        let desc = extract_cognitive_summary(&CognitiveType::Reasoning, &input, &output, &trace);
        assert!(desc.contains("[Reasoning]"));
        assert!(desc.contains("how to deploy"));
        assert!(desc.contains("blue-green strategy"));
        assert!(desc.contains("reasoning:"));
        // Only first 2 trace steps
        assert!(desc.contains("analyzed deployment options"));
        assert!(desc.contains("compared blue-green vs rolling"));
        assert!(!desc.contains("selected blue-green for safety"));
    }

    #[test]
    fn test_extract_observation_summary() {
        let data = json!({"result": "transaction approved"});
        let desc = extract_observation_summary("api_response", &data, 0.95, "payment-service");
        assert!(desc.contains("[api_response]"));
        assert!(desc.contains("payment-service"));
        assert!(desc.contains("95%"));
        assert!(desc.contains("transaction approved"));
    }

    #[test]
    fn test_build_event_narrative_all_types() {
        use agent_db_core::types::{AgentId, SessionId, Timestamp};
        use agent_db_events::core::{Event, EventContext, EventType, LearningEvent};

        let base_event = || Event {
            id: 1u128.into(),
            timestamp: Timestamp::from(1000u64),
            agent_id: AgentId::from(1u64),
            agent_type: "test".to_string(),
            session_id: SessionId::from(1u64),
            event_type: EventType::Context {
                text: "placeholder".to_string(),
                context_type: "test".to_string(),
                language: None,
            },
            causality_chain: vec![],
            context: EventContext::default(),
            metadata: Default::default(),
            context_size_bytes: 0,
            segment_pointer: None,
        };

        let events = vec![
            Event {
                event_type: EventType::Action {
                    action_name: "api_call".to_string(),
                    parameters: json!({"url": "https://example.com"}),
                    outcome: ActionOutcome::Success {
                        result: json!({"text": "ok"}),
                    },
                    duration_ns: 1000,
                },
                ..base_event()
            },
            Event {
                event_type: EventType::Observation {
                    observation_type: "metric".to_string(),
                    data: json!({"message": "latency normal"}),
                    confidence: 0.9,
                    source: "monitor".to_string(),
                },
                ..base_event()
            },
            Event {
                event_type: EventType::Cognitive {
                    process_type: CognitiveType::Planning,
                    input: json!("plan next step"),
                    output: json!("execute backup"),
                    reasoning_trace: vec!["step 1".to_string()],
                },
                ..base_event()
            },
            Event {
                event_type: EventType::Context {
                    text: "User asked for help".to_string(),
                    context_type: "conversation".to_string(),
                    language: None,
                },
                ..base_event()
            },
            Event {
                event_type: EventType::Communication {
                    message_type: "request".to_string(),
                    sender: AgentId::from(1u64),
                    recipient: AgentId::from(2u64),
                    content: json!({"text": "need assistance"}),
                },
                ..base_event()
            },
            Event {
                event_type: EventType::Learning {
                    event: LearningEvent::Outcome {
                        query_id: "q1".to_string(),
                        success: true,
                    },
                },
                ..base_event()
            },
        ];

        let narrative = build_event_narrative(&events);
        assert!(narrative.contains("1. Action:"));
        assert!(narrative.contains("2. Observation:"));
        assert!(narrative.contains("3. Cognitive:"));
        assert!(narrative.contains("4. Context:"));
        assert!(narrative.contains("5. Communication:"));
        assert!(narrative.contains("6. Learning:"));
        // Should have 6 lines
        assert_eq!(narrative.lines().count(), 6);
    }

    #[test]
    fn test_humanize_action_name() {
        assert_eq!(humanize_action_name("cognitive_plan"), "Cognitive Plan");
        assert_eq!(humanize_action_name("api_call"), "Api Call");
        assert_eq!(humanize_action_name("search"), "Search");
    }
}
