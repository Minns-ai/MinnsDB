// crates/agent-db-graph/src/strategies/signatures.rs
//
// Graph signature extraction, behavior skeleton, tool extraction, and hashing.

use crate::episodes::Episode;
use agent_db_events::core::{Event, EventType, MetadataValue};
use std::collections::HashSet;

use super::extractor::StrategyExtractor;
use super::types::*;

impl StrategyExtractor {
    pub(crate) fn extract_graph_signature(
        &self,
        episode: &Episode,
        events: &[Event],
    ) -> (Vec<u64>, Vec<String>, Vec<String>) {
        let goal_ids = episode
            .context
            .active_goals
            .iter()
            .map(|goal| goal.id)
            .collect::<Vec<_>>();

        let tool_names = self.extract_tool_names(events);
        let result_types = self.extract_result_types(events);

        (goal_ids, tool_names, result_types)
    }

    pub(crate) fn extract_tool_names(&self, events: &[Event]) -> Vec<String> {
        let mut tools: HashSet<String> = HashSet::new();

        for event in events {
            for key in ["tool", "tool_name", "tools", "tool_used"] {
                if let Some(value) = event.metadata.get(key) {
                    self.collect_tools_from_metadata(value, &mut tools);
                }
            }

            if let EventType::Action { parameters, .. } = &event.event_type {
                self.collect_tools_from_json(parameters, &mut tools);
            }
        }

        let mut list: Vec<String> = tools.into_iter().collect();
        list.sort();
        list
    }

    pub(crate) fn collect_tools_from_metadata(&self, value: &MetadataValue, tools: &mut HashSet<String>) {
        match value {
            MetadataValue::String(name) => {
                if !name.trim().is_empty() {
                    tools.insert(name.trim().to_string());
                }
            },
            MetadataValue::Json(json) => {
                self.collect_tools_from_json(json, tools);
            },
            _ => {},
        }
    }

    pub(crate) fn collect_tools_from_json(&self, value: &serde_json::Value, tools: &mut HashSet<String>) {
        match value {
            serde_json::Value::String(name) => {
                if !name.trim().is_empty() {
                    tools.insert(name.trim().to_string());
                }
            },
            serde_json::Value::Array(items) => {
                for item in items {
                    self.collect_tools_from_json(item, tools);
                }
            },
            serde_json::Value::Object(map) => {
                for key in ["tool", "tool_name", "tools", "tool_used"] {
                    if let Some(value) = map.get(key) {
                        self.collect_tools_from_json(value, tools);
                    }
                }
            },
            _ => {},
        }
    }

    pub(crate) fn extract_result_types(&self, events: &[Event]) -> Vec<String> {
        let mut types: HashSet<String> = HashSet::new();

        for event in events {
            match &event.event_type {
                EventType::Action { outcome, .. } => match outcome {
                    agent_db_events::core::ActionOutcome::Success { .. } => {
                        types.insert("action_success".to_string());
                    },
                    agent_db_events::core::ActionOutcome::Failure { .. } => {
                        types.insert("action_failure".to_string());
                    },
                    agent_db_events::core::ActionOutcome::Partial { .. } => {
                        types.insert("action_partial".to_string());
                    },
                },
                EventType::Observation { .. } => {
                    types.insert("observation".to_string());
                },
                EventType::Cognitive { .. } => {
                    types.insert("cognitive_output".to_string());
                },
                EventType::Communication { .. } => {
                    types.insert("communication".to_string());
                },
                EventType::Learning { .. } => {
                    types.insert("learning_telemetry".to_string());
                },
                EventType::Context { .. } => {
                    types.insert("context".to_string());
                },
                EventType::Conversation { .. } => {
                    types.insert("conversation".to_string());
                },
                EventType::CodeReview { .. } => {
                    types.insert("code_review".to_string());
                },
                EventType::CodeFile { .. } => {
                    types.insert("code_file".to_string());
                },
            }
        }

        let mut list: Vec<String> = types.into_iter().collect();
        list.sort();
        list
    }

    /// Convert context pattern to hash for indexing
    pub(crate) fn pattern_to_hash(&self, pattern: &ContextPattern) -> Option<u64> {
        // Simplified hashing - in production would use proper hash function
        pattern.task_type.as_ref().map(|t| t.len() as u64)
    }

    pub(crate) fn derive_goal_bucket_id(&self, episode: &Episode) -> u64 {
        // Prefer the pre-computed goal_bucket_id from EventContext when available.
        if episode.context.goal_bucket_id != 0 {
            return episode.context.goal_bucket_id;
        }
        let goal_ids: Vec<u64> = episode
            .context
            .active_goals
            .iter()
            .map(|goal| goal.id)
            .collect();
        compute_goal_bucket_id_from_ids(&goal_ids)
    }

    pub(crate) fn compute_behavior_signature(&self, events: &[Event]) -> String {
        let skeleton = self.build_behavior_skeleton(events);
        let joined = skeleton.join(">");
        format!("{:x}", self.hash_str(&joined))
    }

    pub(crate) fn build_behavior_skeleton(&self, events: &[Event]) -> Vec<String> {
        let mut skeleton = Vec::new();
        for event in events {
            match &event.event_type {
                EventType::Observation { .. } => skeleton.push("Observe".to_string()),
                EventType::Cognitive { process_type, .. } => {
                    skeleton.push(format!("Think:{:?}", process_type));
                },
                EventType::Action { action_name, .. } => {
                    let tool = self
                        .extract_tool_from_metadata(event)
                        .map(|t| format!(":{}", t))
                        .unwrap_or_default();
                    skeleton.push(format!("Act:{}{}", action_name, tool));
                },
                EventType::Communication { .. } => skeleton.push("Communicate".to_string()),
                EventType::Learning { .. } => skeleton.push("Learn".to_string()),
                EventType::Context { context_type, .. } => {
                    skeleton.push(format!("Context:{}", context_type));
                },
                EventType::Conversation { speaker, .. } => {
                    skeleton.push(format!("Conv:{}", speaker));
                },
                EventType::CodeReview { .. } => skeleton.push("CodeReview".to_string()),
                EventType::CodeFile { .. } => skeleton.push("CodeFile".to_string()),
            }
        }
        skeleton
    }

    pub(crate) fn extract_behavior_skeleton(
        &self,
        events: &[Event],
        strategy_type: &StrategyType,
        goal_bucket_id: u64,
    ) -> (String, String, f32) {
        let skeleton = self.build_behavior_skeleton(events);
        let action_hint = match strategy_type {
            StrategyType::Positive => format!("repeat sequence: {}", skeleton.join(" > ")),
            StrategyType::Constraint => format!("avoid sequence: {}", skeleton.join(" > ")),
        };
        let precondition = format!("goal_bucket={} contexts={}", goal_bucket_id, skeleton.len());
        let expected_cost = (events.len() as f32).min(50.0);
        (precondition, action_hint, expected_cost)
    }

    pub(crate) fn compute_strategy_signature(&self, strategy: &Strategy) -> String {
        let raw = format!(
            "{}|{}|{:?}",
            strategy.precondition, strategy.action_hint, strategy.strategy_type
        );
        format!("{:x}", self.hash_str(&raw))
    }

    pub(crate) fn extract_tool_from_metadata(&self, event: &Event) -> Option<String> {
        for key in ["tool_name", "tool", "tool_used"] {
            if let Some(value) = event.metadata.get(key) {
                if let Some(tool) = self.metadata_to_string(value) {
                    return Some(tool);
                }
            }
        }
        None
    }

    pub(crate) fn metadata_to_string(&self, value: &MetadataValue) -> Option<String> {
        match value {
            MetadataValue::String(s) => Some(s.clone()),
            MetadataValue::Integer(i) => Some(i.to_string()),
            MetadataValue::Float(f) => Some(format!("{}", f)),
            _ => None,
        }
    }

    pub(crate) fn hash_str(&self, value: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn jaccard_u64(a: &HashSet<u64>, b: &HashSet<u64>) -> f32 {
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

    pub(crate) fn jaccard_string(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
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
}
