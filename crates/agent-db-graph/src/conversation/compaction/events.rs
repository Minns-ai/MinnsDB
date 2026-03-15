//! Conversion of compaction results into pipeline-ready Events.

use super::types::*;
use crate::memory_classifier::{ClassifiedOperation, MemoryAction};
use agent_db_events::core::{ActionOutcome, CognitiveType, EventContext, EventType, MetadataValue};
use std::collections::HashMap;

/// Convert a `CompactionResponse` into pipeline-ready `Event` structs.
pub fn compaction_to_events(
    response: &CompactionResponse,
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    base_ts: u64,
) -> Vec<agent_db_events::Event> {
    let mut events = Vec::new();
    let mut ts_offset: u64 = 0;

    let next_ts = |offset: &mut u64| -> u64 {
        let ts = base_ts + *offset * 1_000;
        *offset += 1;
        ts
    };

    // Facts → Events (state-aware: predicate determines edge type)
    for fact in &response.facts {
        let mut metadata = HashMap::new();
        metadata.insert(
            "compaction_fact".to_string(),
            MetadataValue::String("true".to_string()),
        );
        metadata.insert(
            "case_id".to_string(),
            MetadataValue::String(case_id.to_string()),
        );

        // Include entity + new_state + predicate so the pipeline creates
        // graph edges directly from LLM-extracted facts.
        metadata.insert(
            "entity".to_string(),
            MetadataValue::String(fact.subject.clone()),
        );
        metadata.insert(
            "new_state".to_string(),
            MetadataValue::String(fact.object.clone()),
        );
        metadata.insert(
            "attribute".to_string(),
            MetadataValue::String(fact.predicate.clone()),
        );
        metadata.insert(
            "entity_state".to_string(),
            MetadataValue::String("true".to_string()),
        );
        // Category for semantic supersession grouping
        if let Some(cat) = &fact.category {
            metadata.insert("category".to_string(), MetadataValue::String(cat.clone()));
        }
        // Conditional dependency — this fact is only valid while the condition holds
        if let Some(dep) = &fact.depends_on {
            metadata.insert("depends_on".to_string(), MetadataValue::String(dep.clone()));
        }
        // Explicit state update marker
        if fact.is_update == Some(true) {
            metadata.insert(
                "is_update".to_string(),
                MetadataValue::String("true".to_string()),
            );
        }

        let evt = agent_db_events::Event {
            id: agent_db_core::types::generate_event_id(),
            timestamp: next_ts(&mut ts_offset),
            agent_id,
            agent_type: "conversation_compaction".to_string(),
            session_id,
            event_type: EventType::Observation {
                observation_type: "extracted_fact".to_string(),
                data: serde_json::json!({
                    "statement": fact.statement,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                }),
                confidence: fact.confidence,
                source: "conversation_compaction".to_string(),
            },
            causality_chain: Vec::new(),
            context: EventContext::default(),
            metadata,
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        };
        events.push(evt);
    }

    // Goals → Cognitive/GoalFormation events
    for goal in &response.goals {
        let mut metadata = HashMap::new();
        metadata.insert(
            "compaction_goal".to_string(),
            MetadataValue::String("true".to_string()),
        );
        metadata.insert(
            "case_id".to_string(),
            MetadataValue::String(case_id.to_string()),
        );

        let evt = agent_db_events::Event {
            id: agent_db_core::types::generate_event_id(),
            timestamp: next_ts(&mut ts_offset),
            agent_id,
            agent_type: "conversation_compaction".to_string(),
            session_id,
            event_type: EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: serde_json::json!({
                    "description": goal.description,
                    "owner": goal.owner,
                }),
                output: serde_json::json!({
                    "status": goal.status,
                }),
                reasoning_trace: vec!["LLM conversation compaction".to_string()],
            },
            causality_chain: Vec::new(),
            context: EventContext::default(),
            metadata,
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        };
        events.push(evt);
    }

    // Procedural steps → Action events
    if let Some(ref summary) = response.procedural_summary {
        for step in &summary.steps {
            let mut metadata = HashMap::new();
            metadata.insert(
                "compaction_step".to_string(),
                MetadataValue::String("true".to_string()),
            );
            metadata.insert(
                "case_id".to_string(),
                MetadataValue::String(case_id.to_string()),
            );

            let outcome = map_step_outcome(&step.outcome);

            let evt = agent_db_events::Event {
                id: agent_db_core::types::generate_event_id(),
                timestamp: next_ts(&mut ts_offset),
                agent_id,
                agent_type: "conversation_compaction".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: format!("step_{}", step.step_number),
                    parameters: serde_json::json!({
                        "action": step.action,
                        "result": step.result,
                    }),
                    outcome,
                    duration_ns: 0,
                },
                causality_chain: Vec::new(),
                context: EventContext::default(),
                metadata,
                context_size_bytes: 0,
                segment_pointer: None,
                is_code: false,
            };
            events.push(evt);
        }
    }

    events
}

/// Map a step outcome string to an `ActionOutcome`.
pub(crate) fn map_step_outcome(outcome: &str) -> ActionOutcome {
    match outcome {
        "success" => ActionOutcome::Success {
            result: serde_json::json!({"status": "success"}),
        },
        "failure" => ActionOutcome::Failure {
            error: "step failed".to_string(),
            error_code: 1,
        },
        "partial" => ActionOutcome::Partial {
            result: serde_json::json!({"status": "partial"}),
            issues: vec!["partial completion".to_string()],
        },
        _ => ActionOutcome::Success {
            result: serde_json::json!({"status": "pending"}),
        },
    }
}

// ────────── Goal Dedup Helpers ──────────

/// Filter goals by their classification operations.
///
/// Returns `(approved_goals, dedup_count)` where approved goals are those
/// classified as ADD or UPDATE. Goals classified as DELETE or NONE are filtered out.
pub fn filter_goals_by_classification(
    goals: &[ExtractedGoal],
    goal_ops: &[ClassifiedOperation],
) -> (Vec<ExtractedGoal>, usize) {
    let mut approved = Vec::new();
    let mut dedup_count = 0usize;

    for (i, goal) in goals.iter().enumerate() {
        let action = goal_ops
            .get(i)
            .map(|op| op.action)
            .unwrap_or(MemoryAction::Add); // fallback: keep

        match action {
            MemoryAction::Add | MemoryAction::Update => {
                approved.push(goal.clone());
            },
            MemoryAction::Delete | MemoryAction::None => {
                dedup_count += 1;
            },
        }
    }

    (approved, dedup_count)
}
