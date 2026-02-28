//! LLM-driven conversation compaction.
//!
//! Runs AFTER the existing rule-based pipeline to extract:
//! - **Facts**: Cross-message inferences the rule-based classifier misses
//! - **Goals**: User objectives/intentions embedded in conversation flow
//! - **Procedural summary**: Structured session summary with steps and outcomes
//!
//! Extracted data is converted into Events (Observation, Cognitive, Action)
//! and a procedural Memory, then fed back through the pipeline.

use crate::conversation::types::ConversationIngest;
use crate::episodes::EpisodeOutcome;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::memory::{Memory, MemoryTier, MemoryType};
use crate::memory_audit::MutationActor;
use crate::memory_classifier::{
    classify_memory_updates, resolve_target, ClassifiedOperation, MemoryAction,
};
use agent_db_events::core::{ActionOutcome, CognitiveType, EventContext, EventType, MetadataValue};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ────────── Rolling Summary ──────────

/// A rolling, incrementally updated summary of an ongoing conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRollingSummary {
    /// Conversation case identifier.
    pub case_id: String,
    /// Current summary text.
    pub summary: String,
    /// Last update timestamp (nanoseconds since epoch).
    pub last_updated: u64,
    /// How many turns have been summarized so far.
    pub turn_count: u32,
    /// Rough token estimate of the current summary.
    pub token_estimate: u32,
}

const SUMMARY_UPDATE_SYSTEM_PROMPT: &str = r#"You are a conversation summarizer. Given an existing summary and new messages, produce an UPDATED summary that captures all key information.

Rules:
- Preserve all important facts, preferences, goals, and decisions from the existing summary
- Integrate new information from the recent messages
- Keep the summary concise (under 500 words)
- Focus on: facts, preferences, goals, decisions, relationships, state changes
- Drop: greetings, filler, repetition
- Output ONLY the updated summary text, no JSON"#;

/// Update the rolling summary with new conversation messages.
///
/// - First call (no existing summary): "Summarize this conversation"
/// - Subsequent calls: "Existing summary + new messages → updated summary"
///
/// Returns `None` on failure (fail-open).
pub async fn update_rolling_summary(
    llm: &dyn LlmClient,
    existing_summary: Option<&str>,
    new_messages: &[crate::conversation::types::ConversationMessage],
) -> Option<String> {
    if new_messages.is_empty() {
        return existing_summary.map(|s| s.to_string());
    }

    let mut messages_text = String::new();
    for msg in new_messages {
        messages_text.push_str(&msg.role);
        messages_text.push_str(": ");
        messages_text.push_str(&msg.content);
        messages_text.push('\n');
    }

    let user_prompt = if let Some(summary) = existing_summary {
        format!(
            "Existing summary:\n{}\n\nNew messages:\n{}\nProduce updated summary.",
            summary, messages_text
        )
    } else {
        format!("Summarize this conversation:\n{}", messages_text)
    };

    let request = LlmRequest {
        system_prompt: SUMMARY_UPDATE_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 1024,
        json_mode: false,
    };

    let response = llm.complete(request).await.ok()?;
    let text = response.content.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Format transcript using rolling summary + last N messages.
///
/// Instead of the full transcript, this uses the summary as context
/// and appends only the most recent messages for detail.
pub fn format_with_summary(
    summary: &ConversationRollingSummary,
    data: &ConversationIngest,
    recent_count: usize,
) -> String {
    let mut buf = String::new();
    buf.push_str("[Rolling Summary]\n");
    buf.push_str(&summary.summary);
    buf.push_str("\n\n[Recent Messages]\n");

    // Collect all messages in order
    let all_messages: Vec<&crate::conversation::types::ConversationMessage> = data
        .sessions
        .iter()
        .flat_map(|s| s.messages.iter())
        .collect();

    // Take the last `recent_count` messages
    let start = all_messages.len().saturating_sub(recent_count);
    for msg in &all_messages[start..] {
        buf.push_str(&msg.role);
        buf.push_str(": ");
        buf.push_str(&msg.content);
        buf.push('\n');
    }

    buf
}

// ────────── Types ──────────

/// LLM extraction response (deserialized from JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResponse {
    pub facts: Vec<ExtractedFact>,
    pub goals: Vec<ExtractedGoal>,
    pub procedural_summary: Option<ProceduralSummary>,
}

/// A single extracted fact (cross-message inference).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub statement: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// A user goal/intention detected in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedGoal {
    pub description: String,
    pub status: String,
    pub owner: String,
}

/// Structured procedural summary of the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralSummary {
    pub objective: String,
    pub progress_status: String,
    pub steps: Vec<ProceduralStep>,
    pub overall_summary: String,
    pub takeaway: String,
}

/// A single step in a procedural summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralStep {
    pub step_number: u32,
    pub action: String,
    pub result: String,
    pub outcome: String,
}

/// Result of the compaction process (returned for logging/response).
#[derive(Debug, Clone, Default, Serialize)]
pub struct CompactionResult {
    pub facts_extracted: usize,
    pub goals_extracted: usize,
    pub goals_deduplicated: usize,
    pub procedural_steps_extracted: usize,
    pub procedural_memory_created: bool,
    pub procedural_memory_id: Option<u64>,
    pub memories_updated: usize,
    pub memories_deleted: usize,
    pub playbooks_extracted: usize,
    pub llm_success: bool,
    pub tokens_used: u32,
}

/// Retrospective playbook for a single goal extracted from conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalPlaybook {
    pub goal_description: String,
    pub what_worked: Vec<String>,
    pub what_didnt_work: Vec<String>,
    pub lessons_learned: Vec<String>,
    #[serde(default)]
    pub steps_taken: Vec<String>,
    #[serde(default = "default_playbook_confidence")]
    pub confidence: f32,
}

fn default_playbook_confidence() -> f32 {
    0.5
}

/// LLM response containing per-goal playbooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookExtractionResponse {
    pub playbooks: Vec<GoalPlaybook>,
}

// ────────── System Prompt ──────────

const COMPACTION_SYSTEM_PROMPT: &str = r#"You are an information extraction system. Given a conversation transcript, extract:

1. "facts": Atomic statements (cross-message inferences, implicit knowledge).
   Each: { "statement", "subject", "predicate", "object", "confidence": 0.0-1.0 }

2. "goals": User objectives/intentions detected.
   Each: { "description", "status": "active"|"completed"|"abandoned", "owner" }

3. "procedural_summary": Structured session summary, or null if no procedural content.
   { "objective", "progress_status": "completed"|"in_progress"|"blocked"|"abandoned",
     "steps": [{ "step_number", "action", "result", "outcome": "success"|"failure"|"partial"|"pending" }],
     "overall_summary", "takeaway" }

Rules:
- Look for cross-message inferences: facts that only become apparent when combining information from multiple messages (e.g., user says "I live in NYC" in one message and asks about "nearby restaurants" later → user wants restaurants in NYC)
- Detect implicit goals from questions: if the user asks "How do I deploy to AWS?", the implicit goal is "Deploy application to AWS"
- Detect commercial/purchase intent: phrases like "I want to order", "I need to buy", "place an order for" indicate purchase goals
- Focus on relationships, preferences, states, and implicit knowledge
- For goals, look for intent/desire/objective expressions including "I would like to", "I need to", "Can you help me"
- Output ONLY valid JSON

Example output:
{
  "facts": [
    {"statement": "User lives in New York", "subject": "User", "predicate": "lives_in", "object": "New York", "confidence": 0.9}
  ],
  "goals": [
    {"description": "Order blue jeans online", "status": "active", "owner": "user"}
  ],
  "procedural_summary": {
    "objective": "Set up development environment",
    "progress_status": "completed",
    "steps": [
      {"step_number": 1, "action": "Install Node.js", "result": "Installed v18.0", "outcome": "success"},
      {"step_number": 2, "action": "Configure ESLint", "result": "Config file created", "outcome": "success"}
    ],
    "overall_summary": "Successfully set up dev environment with Node.js and ESLint",
    "takeaway": "Use nvm for Node.js version management"
  }
}"#;

// ────────── Transcript Formatting ──────────

/// Maximum transcript length sent to the LLM (chars). Keeps the tail.
const MAX_TRANSCRIPT_CHARS: usize = 16_000;

/// Format a `ConversationIngest` into a text transcript for the LLM.
pub fn format_transcript(data: &ConversationIngest) -> String {
    let mut buf = String::new();
    for session in &data.sessions {
        for msg in &session.messages {
            buf.push_str(&msg.role);
            buf.push_str(": ");
            buf.push_str(&msg.content);
            buf.push('\n');
        }
    }

    if buf.len() > MAX_TRANSCRIPT_CHARS {
        let start = buf.len() - MAX_TRANSCRIPT_CHARS;
        // Find the next newline after the cut point to avoid splitting a line
        let adjusted = buf[start..]
            .find('\n')
            .map(|p| start + p + 1)
            .unwrap_or(start);
        buf = buf[adjusted..].to_string();
    }

    buf
}

// ────────── LLM Extraction ──────────

/// Call the LLM to extract facts, goals, and procedural summary from a transcript.
///
/// Returns `None` on any failure (fail-open).
pub async fn extract_compaction(
    llm: &dyn LlmClient,
    data: &ConversationIngest,
) -> Option<CompactionResponse> {
    let transcript = format_transcript(data);
    extract_compaction_from_transcript(llm, &transcript).await
}

/// Extract compaction from a pre-formatted transcript string.
///
/// Returns `None` on any failure (fail-open).
pub async fn extract_compaction_from_transcript(
    llm: &dyn LlmClient,
    transcript: &str,
) -> Option<CompactionResponse> {
    if transcript.is_empty() {
        return None;
    }

    let request = LlmRequest {
        system_prompt: COMPACTION_SYSTEM_PROMPT.to_string(),
        user_prompt: transcript.to_string(),
        temperature: 0.0,
        max_tokens: 2048,
        json_mode: true,
    };

    let response = llm.complete(request).await.ok()?;
    let value = parse_json_from_llm(&response.content)?;
    serde_json::from_value::<CompactionResponse>(value).ok()
}

// ────────── Playbook Extraction ──────────

const PLAYBOOK_SYSTEM_PROMPT: &str = r#"You are a retrospective analysis system. Given a conversation transcript and goals, extract a playbook for each goal:

1. "what_worked": Actions/approaches that succeeded
2. "what_didnt_work": Actions/approaches that failed or were abandoned
3. "lessons_learned": Key takeaways for future attempts
4. "steps_taken": Brief ordered list of steps actually taken
5. "confidence": 0.0-1.0

If prior playbook experience is provided, use it to compare approaches and note what was
done differently this time. Reference prior lessons when relevant.

Output: { "playbooks": [ { "goal_description", "what_worked", "what_didnt_work", "lessons_learned", "steps_taken", "confidence" } ] }
Rules: One playbook per goal. Empty arrays if goal was barely discussed. Be specific, not generic. Output ONLY valid JSON"#;

/// Call the LLM to extract retrospective playbooks for each goal.
///
/// Returns `None` on any failure (fail-open).
pub async fn extract_playbooks(
    llm: &dyn LlmClient,
    transcript: &str,
    goals: &[ExtractedGoal],
) -> Option<PlaybookExtractionResponse> {
    if goals.is_empty() || transcript.is_empty() {
        return None;
    }

    let mut goal_list = String::new();
    for (i, goal) in goals.iter().enumerate() {
        goal_list.push_str(&format!("{}. {}\n", i + 1, goal.description));
    }

    let user_prompt = format!(
        "Transcript:\n{}\n\nGoals to analyze:\n{}",
        transcript, goal_list
    );

    let request = LlmRequest {
        system_prompt: PLAYBOOK_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 2048,
        json_mode: true,
    };

    let response = llm.complete(request).await.ok()?;
    let value = parse_json_from_llm(&response.content)?;
    serde_json::from_value::<PlaybookExtractionResponse>(value).ok()
}

/// Attach extracted playbooks to GoalStore entries and graph node properties.
async fn attach_playbooks(
    engine: &crate::integration::GraphEngine,
    playbooks: &[GoalPlaybook],
    _case_id: &str,
) {
    for playbook in playbooks {
        // 1. Find matching goal in GoalStore via BM25
        let goal_store = engine.goal_store.read().await;
        let similar = goal_store.find_similar(&playbook.goal_description, 1);
        let goal_id = match similar.first() {
            Some((id, _score)) => *id,
            None => continue,
        };
        drop(goal_store);

        // 2. Attach playbook to GoalEntry
        let mut goal_store = engine.goal_store.write().await;
        goal_store.attach_playbook(goal_id, playbook.clone());
        drop(goal_store);

        // 3. Set properties on the Goal graph node
        let mut inference = engine.inference.write().await;
        let graph = inference.graph_mut();
        if let Some(&node_id) = graph.goal_index.get(&goal_id) {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.properties.insert(
                    "playbook_what_worked".to_string(),
                    serde_json::json!(playbook.what_worked),
                );
                node.properties.insert(
                    "playbook_what_didnt_work".to_string(),
                    serde_json::json!(playbook.what_didnt_work),
                );
                node.properties.insert(
                    "playbook_lessons_learned".to_string(),
                    serde_json::json!(playbook.lessons_learned),
                );
                node.properties.insert(
                    "playbook_steps_taken".to_string(),
                    serde_json::json!(playbook.steps_taken),
                );
                node.properties.insert(
                    "playbook_confidence".to_string(),
                    serde_json::json!(playbook.confidence),
                );
            }
        }
    }
}

// ────────── Event Conversion ──────────

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

    // Facts → Observation events
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
fn map_step_outcome(outcome: &str) -> ActionOutcome {
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

// ────────── Procedural Memory ──────────

/// Build a procedural `Memory` from a `ProceduralSummary`.
pub fn build_procedural_memory(
    summary: &ProceduralSummary,
    agent_id: u64,
    session_id: u64,
    episode_id: u64,
) -> Memory {
    let outcome = map_progress_to_outcome(&summary.progress_status);

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "compaction".to_string());
    metadata.insert("objective".to_string(), summary.objective.clone());
    metadata.insert(
        "progress_status".to_string(),
        summary.progress_status.clone(),
    );

    Memory {
        id: 0, // Will be assigned by the store
        agent_id,
        session_id,
        episode_id,
        summary: summary.overall_summary.clone(),
        takeaway: summary.takeaway.clone(),
        causal_note: format!(
            "Objective: {}. Status: {}",
            summary.objective, summary.progress_status
        ),
        summary_embedding: Vec::new(),
        tier: MemoryTier::Episodic,
        consolidated_from: Vec::new(),
        schema_id: None,
        consolidation_status: crate::memory::ConsolidationStatus::Active,
        context: EventContext::default(),
        key_events: Vec::new(),
        strength: 0.8,
        relevance_score: 0.8,
        formed_at: agent_db_core::types::current_timestamp(),
        last_accessed: agent_db_core::types::current_timestamp(),
        access_count: 0,
        outcome,
        memory_type: MemoryType::Episodic { significance: 0.8 },
        metadata,
        expires_at: None,
    }
}

/// Map a progress_status string to an `EpisodeOutcome`.
pub fn map_progress_to_outcome(status: &str) -> EpisodeOutcome {
    match status {
        "completed" => EpisodeOutcome::Success,
        "blocked" | "abandoned" => EpisodeOutcome::Failure,
        "in_progress" => EpisodeOutcome::Partial,
        _ => EpisodeOutcome::Partial,
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

// ────────── Procedural Memory Helpers ──────────

/// Handle procedural memory with a classification op.
#[allow(clippy::too_many_arguments)]
async fn handle_procedural_memory(
    engine: &crate::integration::GraphEngine,
    summary: &ProceduralSummary,
    proc_op: &ClassifiedOperation,
    similar_refs: &[&Memory],
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    result: &mut CompactionResult,
) {
    let episode_id = procedural_episode_id(case_id);

    match proc_op.action {
        MemoryAction::Add => {
            store_new_procedural_memory(
                engine, summary, case_id, agent_id, session_id, episode_id, result,
            )
            .await;
        },
        MemoryAction::Update => {
            if let Some(target_id) = resolve_target(proc_op, similar_refs) {
                let store = engine.memory_store.read().await;
                let existing = store.get_memory(target_id);
                drop(store);

                if let Some(existing_mem) = existing {
                    let old_summary = existing_mem.summary.clone();
                    let old_takeaway = existing_mem.takeaway.clone();

                    let new_summary_text = proc_op
                        .new_text
                        .as_deref()
                        .unwrap_or(&summary.overall_summary);
                    let mut updated = existing_mem;
                    updated.summary = new_summary_text.to_string();
                    updated.takeaway = summary.takeaway.clone();
                    updated.last_accessed = agent_db_core::types::current_timestamp();

                    let text = format!(
                        "{} {} {}",
                        updated.summary, updated.takeaway, updated.causal_note
                    );

                    let mut store = engine.memory_store.write().await;
                    store.store_consolidated_memory(updated);
                    drop(store);

                    let mut bm25 = engine.memory_bm25_index.write().await;
                    bm25.index_document(target_id, &text);
                    drop(bm25);

                    result.memories_updated += 1;

                    let mut audit = engine.memory_audit_log.write().await;
                    audit.record_update(
                        target_id,
                        &old_summary,
                        new_summary_text,
                        &old_takeaway,
                        &summary.takeaway,
                        MutationActor::LlmClassifier,
                        Some(format!("Compaction UPDATE for case {}", case_id)),
                    );
                }
            }
        },
        MemoryAction::Delete => {
            if let Some(target_id) = resolve_target(proc_op, similar_refs) {
                let store = engine.memory_store.read().await;
                let existing = store.get_memory(target_id);
                drop(store);

                if let Some(existing_mem) = existing {
                    let old_summary = existing_mem.summary.clone();
                    let old_takeaway = existing_mem.takeaway.clone();

                    let mut store = engine.memory_store.write().await;
                    store.delete_memories_batch(vec![target_id]);
                    drop(store);

                    result.memories_deleted += 1;

                    let mut audit = engine.memory_audit_log.write().await;
                    audit.record_delete(
                        target_id,
                        &old_summary,
                        &old_takeaway,
                        MutationActor::LlmClassifier,
                        Some(format!("Compaction DELETE for case {}", case_id)),
                    );
                }
            }
        },
        MemoryAction::None => {
            // Skip — already captured
        },
    }
}

/// Fallback: unconditionally create a new procedural memory (fail-open).
async fn handle_procedural_memory_fallback(
    engine: &crate::integration::GraphEngine,
    summary: &ProceduralSummary,
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    result: &mut CompactionResult,
) {
    let episode_id = procedural_episode_id(case_id);
    store_new_procedural_memory(
        engine, summary, case_id, agent_id, session_id, episode_id, result,
    )
    .await;

    // Record audit trail as fallback
    let actual_id = procedural_memory_id(case_id);
    let mut audit = engine.memory_audit_log.write().await;
    audit.record_add(
        actual_id,
        &summary.overall_summary,
        &summary.takeaway,
        MutationActor::ConversationBridge,
        Some(format!(
            "Compaction procedural memory for case {} (classifier fallback)",
            case_id
        )),
    );
}

/// Store a new procedural memory and record its audit trail.
async fn store_new_procedural_memory(
    engine: &crate::integration::GraphEngine,
    summary: &ProceduralSummary,
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    episode_id: u64,
    result: &mut CompactionResult,
) {
    let memory = build_procedural_memory(summary, agent_id, session_id, episode_id);
    let actual_id = procedural_memory_id(case_id);

    let mut mem_with_id = memory;
    mem_with_id.id = actual_id;

    let text = format!(
        "{} {} {}",
        mem_with_id.summary, mem_with_id.takeaway, mem_with_id.causal_note
    );

    let mut store = engine.memory_store.write().await;
    store.store_consolidated_memory(mem_with_id);
    drop(store);

    let mut bm25 = engine.memory_bm25_index.write().await;
    bm25.index_document(actual_id, &text);
    drop(bm25);

    result.procedural_memory_created = true;
    result.procedural_memory_id = Some(actual_id);

    let mut audit = engine.memory_audit_log.write().await;
    audit.record_add(
        actual_id,
        &summary.overall_summary,
        &summary.takeaway,
        MutationActor::LlmClassifier,
        Some(format!("Compaction ADD for case {}", case_id)),
    );
}

/// Compute a deterministic episode ID for procedural memory.
fn procedural_episode_id(case_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{}:procedural", case_id).hash(&mut hasher);
    hasher.finish()
}

/// Compute a deterministic memory ID for procedural memory.
fn procedural_memory_id(case_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{}:procedural_memory", case_id).hash(&mut hasher);
    hasher.finish()
}

// ────────── Top-Level Entry Point ──────────

/// Run LLM-driven compaction on a conversation ingest.
///
/// This is the top-level entry point called after the rule-based pipeline.
/// It extracts facts, goals, and a procedural summary via a single LLM call,
/// then feeds the results back through the event pipeline and stores a
/// procedural memory.
///
/// Fail-open: returns an empty `CompactionResult` on any failure.
pub async fn run_compaction(
    engine: &crate::integration::GraphEngine,
    data: &ConversationIngest,
    case_id: &str,
) -> CompactionResult {
    let mut result = CompactionResult::default();

    // Need an LLM client
    let llm = match engine.unified_llm_client() {
        Some(client) => Arc::clone(client),
        None => return result,
    };

    // Derive agent_id and session_id using the same formula as ingest_to_events
    let agent_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        case_id.hash(&mut hasher);
        hasher.finish() | 0x8000_0000_0000_0000
    };

    let session_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{}:compaction", case_id).hash(&mut hasher);
        hasher.finish()
    };

    let base_ts: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // If rolling summary exists, use it + last N messages instead of full transcript
    let transcript = {
        let summaries = engine.conversation_summaries.read().await;
        if let Some(summary) = summaries.get(case_id) {
            format_with_summary(summary, data, engine.config.rolling_summary_recent_messages)
        } else {
            format_transcript(data)
        }
    };

    // Enrich transcript with community context if enabled
    let enriched_transcript = if engine.config.enable_context_enrichment {
        let summaries = engine.community_summaries.read().await;
        if summaries.is_empty() {
            transcript
        } else {
            let topic_slice = &transcript[..transcript.len().min(500)];
            let ctx = crate::context_enrichment::community_context_for_topic(
                topic_slice,
                &summaries,
                &engine.config.enrichment_config,
            );
            if ctx.is_empty() {
                transcript
            } else {
                format!("{}\n\n[Knowledge Context]\n{}", transcript, ctx)
            }
        }
    } else {
        transcript
    };

    // LLM extraction with 30-second timeout
    let extraction = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        extract_compaction_from_transcript(llm.as_ref(), &enriched_transcript),
    )
    .await;

    let response = match extraction {
        Ok(Some(r)) => r,
        _ => return result,
    };

    result.llm_success = true;
    result.facts_extracted = response.facts.len();
    result.goals_extracted = response.goals.len();
    result.procedural_steps_extracted = response
        .procedural_summary
        .as_ref()
        .map(|s| s.steps.len())
        .unwrap_or(0);

    // ── Fast goal dedup via GoalStore (no LLM needed) ──
    let mut fast_dedup_count = 0usize;
    let pre_filtered_goals: Vec<ExtractedGoal> = {
        let mut goal_store = engine.goal_store.write().await;
        let mut kept = Vec::new();
        for goal in &response.goals {
            match goal_store.store_or_dedup(&goal.description, &goal.status, &goal.owner, case_id) {
                crate::goal_store::GoalDedupDecision::NewGoal => {
                    kept.push(goal.clone());
                },
                crate::goal_store::GoalDedupDecision::Duplicate { .. } => {
                    fast_dedup_count += 1;
                },
                crate::goal_store::GoalDedupDecision::StatusUpdate {
                    existing_id,
                    new_status,
                } => {
                    goal_store.update_status(existing_id, new_status);
                    fast_dedup_count += 1;
                },
            }
        }
        kept
    };
    result.goals_deduplicated += fast_dedup_count;

    // Use pre-filtered goals for the rest of the pipeline
    let response = CompactionResponse {
        facts: response.facts,
        goals: pre_filtered_goals,
        procedural_summary: response.procedural_summary,
    };

    let has_goals = !response.goals.is_empty();
    let has_summary = response.procedural_summary.is_some();

    if has_goals || has_summary {
        // Collect classifiable items: goal descriptions + summary (if present)
        let goal_count = response.goals.len();
        let mut classifiable: Vec<String> = response
            .goals
            .iter()
            .map(|g| g.description.clone())
            .collect();
        if let Some(ref summary) = response.procedural_summary {
            classifiable.push(summary.overall_summary.clone());
        }
        let classifiable_refs: Vec<&str> = classifiable.iter().map(|s| s.as_str()).collect();

        // BM25 search: union of hits across all items (HashSet dedup)
        let similar_memories = {
            let bm25 = engine.memory_bm25_index.read().await;
            let mut seen_ids = HashSet::new();
            let mut all_memories = Vec::new();

            for item in &classifiable_refs {
                let hits = bm25.search(item, 10);
                for (id, _score) in &hits {
                    if seen_ids.insert(*id) {
                        // new ID
                    }
                }
            }
            drop(bm25);

            let store = engine.memory_store.read().await;
            for id in &seen_ids {
                if let Some(mem) = store.get_memory(*id) {
                    all_memories.push(mem);
                }
            }
            all_memories
        };

        let similar_refs: Vec<&Memory> = similar_memories.iter().collect();

        // Build community context for classifier enrichment
        let classifier_ctx = if engine.config.enable_context_enrichment {
            let summaries = engine.community_summaries.read().await;
            if summaries.is_empty() {
                None
            } else {
                let topic = classifiable.join(" ");
                let ctx = crate::context_enrichment::community_context_for_topic(
                    &topic[..topic.len().min(500)],
                    &summaries,
                    &engine.config.enrichment_config,
                );
                if ctx.is_empty() {
                    None
                } else {
                    Some(ctx)
                }
            }
        } else {
            None
        };

        // Single classify_memory_updates() call (batched)
        let classification = classify_memory_updates(
            llm.as_ref(),
            &classifiable_refs,
            &similar_refs,
            classifier_ctx.as_deref(),
        )
        .await;

        match classification {
            Ok(class_result) => {
                // Split results: goal_ops[0..goal_count] + proc_ops[goal_count..]
                let goal_ops =
                    &class_result.operations[..goal_count.min(class_result.operations.len())];
                let proc_ops = if class_result.operations.len() > goal_count {
                    &class_result.operations[goal_count..]
                } else {
                    &[]
                };

                // Filter goals by classification
                let (approved_goals, dedup_count) =
                    filter_goals_by_classification(&response.goals, goal_ops);
                result.goals_deduplicated = dedup_count;

                // Handle goal UPDATE audit trails
                for (i, op) in goal_ops.iter().enumerate() {
                    if op.action == MemoryAction::Update {
                        if let Some(target_id) = resolve_target(op, &similar_refs) {
                            let store = engine.memory_store.read().await;
                            let existing = store.get_memory(target_id);
                            drop(store);

                            if let Some(existing_mem) = existing {
                                let goal_desc = response
                                    .goals
                                    .get(i)
                                    .map(|g| g.description.as_str())
                                    .unwrap_or("");
                                let mut audit = engine.memory_audit_log.write().await;
                                audit.record_update(
                                    target_id,
                                    &existing_mem.summary,
                                    goal_desc,
                                    &existing_mem.takeaway,
                                    goal_desc,
                                    MutationActor::LlmClassifier,
                                    Some(format!("Compaction goal UPDATE for case {}", case_id)),
                                );
                            }
                        }
                    } else if op.action == MemoryAction::Delete {
                        if let Some(target_id) = resolve_target(op, &similar_refs) {
                            let store = engine.memory_store.read().await;
                            let existing = store.get_memory(target_id);
                            drop(store);

                            if let Some(existing_mem) = existing {
                                let old_summary = existing_mem.summary.clone();
                                let old_takeaway = existing_mem.takeaway.clone();

                                let mut store = engine.memory_store.write().await;
                                store.delete_memories_batch(vec![target_id]);
                                drop(store);

                                result.memories_deleted += 1;

                                let mut audit = engine.memory_audit_log.write().await;
                                audit.record_delete(
                                    target_id,
                                    &old_summary,
                                    &old_takeaway,
                                    MutationActor::LlmClassifier,
                                    Some(format!("Compaction goal DELETE for case {}", case_id)),
                                );
                            }
                        }
                    }
                }

                // Build filtered response with approved goals only
                let filtered_response = CompactionResponse {
                    facts: response.facts.clone(),
                    goals: approved_goals,
                    procedural_summary: response.procedural_summary.clone(),
                };

                // Convert to events and process through pipeline
                let events = compaction_to_events(
                    &filtered_response,
                    case_id,
                    agent_id,
                    session_id,
                    base_ts,
                );
                for event in events {
                    if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                        tracing::debug!("Compaction event pipeline error: {}", e);
                    }
                }

                // Handle procedural memory using proc_ops
                if let Some(ref summary) = response.procedural_summary {
                    if let Some(proc_op) = proc_ops.first() {
                        handle_procedural_memory(
                            engine,
                            summary,
                            proc_op,
                            &similar_refs,
                            case_id,
                            agent_id,
                            session_id,
                            &mut result,
                        )
                        .await;
                    } else {
                        // No proc op returned — fail-open: create unconditionally
                        handle_procedural_memory_fallback(
                            engine,
                            summary,
                            case_id,
                            agent_id,
                            session_id,
                            &mut result,
                        )
                        .await;
                    }
                }
            },
            Err(_) => {
                // Fallback: create all events + unconditional procedural memory (fail-open)
                let events =
                    compaction_to_events(&response, case_id, agent_id, session_id, base_ts);
                for event in events {
                    if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                        tracing::debug!("Compaction event pipeline error: {}", e);
                    }
                }

                if let Some(ref summary) = response.procedural_summary {
                    handle_procedural_memory_fallback(
                        engine,
                        summary,
                        case_id,
                        agent_id,
                        session_id,
                        &mut result,
                    )
                    .await;
                }
            },
        }
    } else {
        // Only facts/steps — no goals or summary to classify
        let events = compaction_to_events(&response, case_id, agent_id, session_id, base_ts);
        for event in events {
            if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                tracing::debug!("Compaction event pipeline error: {}", e);
            }
        }
    }

    // ── Playbook extraction (separate LLM call, fail-open, 30s timeout) ──
    if !response.goals.is_empty() {
        let transcript = format_transcript(data);

        // Enrich with prior playbook experience if enabled
        let enriched_pb_transcript = if engine.config.enable_context_enrichment {
            let goal_store = engine.goal_store.read().await;
            let mut existing = Vec::new();
            for goal in &response.goals {
                for (id, _score) in goal_store.find_similar(&goal.description, 3) {
                    if let Some(entry) = goal_store.get(id) {
                        if let Some(ref pb) = entry.playbook {
                            existing.push((entry.description.clone(), pb.clone()));
                        }
                    }
                }
            }
            drop(goal_store);
            if existing.is_empty() {
                transcript.clone()
            } else {
                let ctx = crate::context_enrichment::build_playbook_context(
                    &existing,
                    engine.config.enrichment_config.max_similar_playbooks,
                );
                format!("{}\n\n[Prior Playbook Experience]\n{}", transcript, ctx)
            }
        } else {
            transcript.clone()
        };

        if let Ok(Some(pb_response)) = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            extract_playbooks(llm.as_ref(), &enriched_pb_transcript, &response.goals),
        )
        .await
        {
            result.playbooks_extracted = pb_response.playbooks.len();
            attach_playbooks(engine, &pb_response.playbooks, case_id).await;
        }
    }

    result
}

// ────────── Tests ──────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::types::{ConversationMessage, ConversationSession};

    fn make_ingest(messages: Vec<(&str, &str)>) -> ConversationIngest {
        ConversationIngest {
            case_id: Some("test_compaction".to_string()),
            sessions: vec![ConversationSession {
                session_id: "s1".to_string(),
                topic: None,
                messages: messages
                    .into_iter()
                    .map(|(role, content)| ConversationMessage {
                        role: role.to_string(),
                        content: content.to_string(),
                    })
                    .collect(),
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
        }
    }

    // 1. test_format_transcript
    #[test]
    fn test_format_transcript() {
        let data = make_ingest(vec![
            ("user", "Hello, I want to plan a trip"),
            ("assistant", "Sure! Where would you like to go?"),
            ("user", "I want to visit Japan in April"),
        ]);

        let transcript = format_transcript(&data);
        assert!(transcript.contains("user: Hello, I want to plan a trip\n"));
        assert!(transcript.contains("assistant: Sure! Where would you like to go?\n"));
        assert!(transcript.contains("user: I want to visit Japan in April\n"));
    }

    #[test]
    fn test_format_transcript_truncation() {
        // Create a very long transcript that exceeds MAX_TRANSCRIPT_CHARS
        let long_msg = "x".repeat(20_000);
        let data = make_ingest(vec![("user", &long_msg)]);

        let transcript = format_transcript(&data);
        assert!(transcript.len() <= MAX_TRANSCRIPT_CHARS);
    }

    // 2. test_parse_compaction_response
    #[test]
    fn test_parse_compaction_response() {
        let json = r#"{
            "facts": [
                {
                    "statement": "Alice lives in Paris",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Paris",
                    "confidence": 0.9
                }
            ],
            "goals": [
                {
                    "description": "Plan a trip to Japan",
                    "status": "active",
                    "owner": "user"
                }
            ],
            "procedural_summary": {
                "objective": "Plan vacation",
                "progress_status": "in_progress",
                "steps": [
                    {
                        "step_number": 1,
                        "action": "Choose destination",
                        "result": "Selected Japan",
                        "outcome": "success"
                    }
                ],
                "overall_summary": "User is planning a trip to Japan",
                "takeaway": "User prefers travel to Asia"
            }
        }"#;

        let response: CompactionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.facts.len(), 1);
        assert_eq!(response.facts[0].subject, "Alice");
        assert_eq!(response.goals.len(), 1);
        assert_eq!(response.goals[0].status, "active");
        assert!(response.procedural_summary.is_some());
        let summary = response.procedural_summary.unwrap();
        assert_eq!(summary.steps.len(), 1);
        assert_eq!(summary.progress_status, "in_progress");
    }

    // 3. test_parse_compaction_response_minimal
    #[test]
    fn test_parse_compaction_response_minimal() {
        let json = r#"{
            "facts": [],
            "goals": [],
            "procedural_summary": null
        }"#;

        let response: CompactionResponse = serde_json::from_str(json).unwrap();
        assert!(response.facts.is_empty());
        assert!(response.goals.is_empty());
        assert!(response.procedural_summary.is_none());
    }

    // 4. test_parse_compaction_response_fenced
    #[test]
    fn test_parse_compaction_response_fenced() {
        let fenced = r#"```json
{
    "facts": [{"statement": "Bob is tall", "subject": "Bob", "predicate": "is", "object": "tall", "confidence": 0.8}],
    "goals": [],
    "procedural_summary": null
}
```"#;

        let value = parse_json_from_llm(fenced).unwrap();
        let response: CompactionResponse = serde_json::from_value(value).unwrap();
        assert_eq!(response.facts.len(), 1);
        assert_eq!(response.facts[0].statement, "Bob is tall");
    }

    // 5. test_compaction_to_events_facts
    #[test]
    fn test_compaction_to_events_facts() {
        let response = CompactionResponse {
            facts: vec![
                ExtractedFact {
                    statement: "Alice lives in Paris".to_string(),
                    subject: "Alice".to_string(),
                    predicate: "lives_in".to_string(),
                    object: "Paris".to_string(),
                    confidence: 0.9,
                },
                ExtractedFact {
                    statement: "Bob works at Google".to_string(),
                    subject: "Bob".to_string(),
                    predicate: "works_at".to_string(),
                    object: "Google".to_string(),
                    confidence: 0.85,
                },
                ExtractedFact {
                    statement: "Alice and Bob are friends".to_string(),
                    subject: "Alice".to_string(),
                    predicate: "friends_with".to_string(),
                    object: "Bob".to_string(),
                    confidence: 0.75,
                },
            ],
            goals: vec![],
            procedural_summary: None,
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert_eq!(events.len(), 3);

        for evt in &events {
            assert_eq!(evt.agent_type, "conversation_compaction");
            match &evt.event_type {
                EventType::Observation {
                    observation_type,
                    source,
                    ..
                } => {
                    assert_eq!(observation_type, "extracted_fact");
                    assert_eq!(source, "conversation_compaction");
                },
                other => panic!("Expected Observation, got {:?}", other),
            }
            assert!(evt.metadata.contains_key("compaction_fact"));
        }
    }

    // 6. test_compaction_to_events_goals
    #[test]
    fn test_compaction_to_events_goals() {
        let response = CompactionResponse {
            facts: vec![],
            goals: vec![
                ExtractedGoal {
                    description: "Plan trip to Japan".to_string(),
                    status: "active".to_string(),
                    owner: "user".to_string(),
                },
                ExtractedGoal {
                    description: "Learn Rust".to_string(),
                    status: "completed".to_string(),
                    owner: "user".to_string(),
                },
            ],
            procedural_summary: None,
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert_eq!(events.len(), 2);

        for evt in &events {
            match &evt.event_type {
                EventType::Cognitive {
                    process_type,
                    reasoning_trace,
                    ..
                } => {
                    assert_eq!(*process_type, CognitiveType::GoalFormation);
                    assert_eq!(reasoning_trace[0], "LLM conversation compaction");
                },
                other => panic!("Expected Cognitive, got {:?}", other),
            }
            assert!(evt.metadata.contains_key("compaction_goal"));
        }
    }

    // 7. test_compaction_to_events_steps
    #[test]
    fn test_compaction_to_events_steps() {
        let response = CompactionResponse {
            facts: vec![],
            goals: vec![],
            procedural_summary: Some(ProceduralSummary {
                objective: "Deploy app".to_string(),
                progress_status: "completed".to_string(),
                steps: vec![
                    ProceduralStep {
                        step_number: 1,
                        action: "Build Docker image".to_string(),
                        result: "Image built successfully".to_string(),
                        outcome: "success".to_string(),
                    },
                    ProceduralStep {
                        step_number: 2,
                        action: "Push to registry".to_string(),
                        result: "Push failed due to auth".to_string(),
                        outcome: "failure".to_string(),
                    },
                    ProceduralStep {
                        step_number: 3,
                        action: "Retry with credentials".to_string(),
                        result: "Push succeeded".to_string(),
                        outcome: "success".to_string(),
                    },
                ],
                overall_summary: "Deployed app after auth fix".to_string(),
                takeaway: "Always verify registry credentials".to_string(),
            }),
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert_eq!(events.len(), 3);

        // Check outcome mapping
        match &events[0].event_type {
            EventType::Action {
                action_name,
                outcome,
                ..
            } => {
                assert_eq!(action_name, "step_1");
                assert!(matches!(outcome, ActionOutcome::Success { .. }));
            },
            other => panic!("Expected Action, got {:?}", other),
        }

        match &events[1].event_type {
            EventType::Action { outcome, .. } => {
                assert!(matches!(outcome, ActionOutcome::Failure { .. }));
            },
            other => panic!("Expected Action, got {:?}", other),
        }

        for evt in &events {
            assert!(evt.metadata.contains_key("compaction_step"));
        }
    }

    // 8. test_compaction_to_events_mixed
    #[test]
    fn test_compaction_to_events_mixed() {
        let response = CompactionResponse {
            facts: vec![
                ExtractedFact {
                    statement: "F1".to_string(),
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                    confidence: 0.9,
                },
                ExtractedFact {
                    statement: "F2".to_string(),
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                    confidence: 0.8,
                },
            ],
            goals: vec![ExtractedGoal {
                description: "G1".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            }],
            procedural_summary: Some(ProceduralSummary {
                objective: "Obj".to_string(),
                progress_status: "completed".to_string(),
                steps: vec![ProceduralStep {
                    step_number: 1,
                    action: "A".to_string(),
                    result: "R".to_string(),
                    outcome: "success".to_string(),
                }],
                overall_summary: "S".to_string(),
                takeaway: "T".to_string(),
            }),
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        // 2 facts + 1 goal + 1 step = 4
        assert_eq!(events.len(), 4);

        // Timestamps must be monotonically increasing
        for window in events.windows(2) {
            assert!(
                window[1].timestamp > window[0].timestamp,
                "Timestamps not monotonic: {} <= {}",
                window[1].timestamp,
                window[0].timestamp
            );
        }
    }

    // 9. test_compaction_to_events_empty
    #[test]
    fn test_compaction_to_events_empty() {
        let response = CompactionResponse {
            facts: vec![],
            goals: vec![],
            procedural_summary: None,
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert!(events.is_empty());
    }

    // 10. test_build_procedural_memory
    #[test]
    fn test_build_procedural_memory() {
        let summary = ProceduralSummary {
            objective: "Deploy app".to_string(),
            progress_status: "completed".to_string(),
            steps: vec![],
            overall_summary: "Successfully deployed the application".to_string(),
            takeaway: "Always test in staging first".to_string(),
        };

        let memory = build_procedural_memory(&summary, 100, 200, 300);

        assert_eq!(memory.summary, "Successfully deployed the application");
        assert_eq!(memory.takeaway, "Always test in staging first");
        assert_eq!(
            memory.causal_note,
            "Objective: Deploy app. Status: completed"
        );
        assert_eq!(memory.tier, MemoryTier::Episodic);
        assert_eq!(memory.strength, 0.8);
        assert_eq!(memory.outcome, EpisodeOutcome::Success);
        assert_eq!(memory.agent_id, 100);
        assert_eq!(memory.session_id, 200);
        assert_eq!(memory.episode_id, 300);
        assert_eq!(
            memory.metadata.get("source").map(|s| s.as_str()),
            Some("compaction")
        );
        assert_eq!(
            memory.metadata.get("objective").map(|s| s.as_str()),
            Some("Deploy app")
        );
        assert_eq!(
            memory.metadata.get("progress_status").map(|s| s.as_str()),
            Some("completed")
        );
    }

    // 11. test_build_procedural_memory_outcome_mapping
    #[test]
    fn test_build_procedural_memory_outcome_mapping() {
        assert_eq!(
            map_progress_to_outcome("completed"),
            EpisodeOutcome::Success
        );
        assert_eq!(map_progress_to_outcome("blocked"), EpisodeOutcome::Failure);
        assert_eq!(
            map_progress_to_outcome("abandoned"),
            EpisodeOutcome::Failure
        );
        assert_eq!(
            map_progress_to_outcome("in_progress"),
            EpisodeOutcome::Partial
        );
        assert_eq!(
            map_progress_to_outcome("unknown_status"),
            EpisodeOutcome::Partial
        );
    }

    // 12. test_compaction_result_has_classifier_fields
    #[test]
    fn test_compaction_result_has_classifier_fields() {
        let result = CompactionResult::default();
        assert_eq!(result.memories_updated, 0);
        assert_eq!(result.memories_deleted, 0);
        assert_eq!(result.facts_extracted, 0);
        assert!(!result.procedural_memory_created);
        assert!(!result.llm_success);
    }

    // 13. test_goals_deduplicated_field_default
    #[test]
    fn test_goals_deduplicated_field_default() {
        let result = CompactionResult::default();
        assert_eq!(result.goals_deduplicated, 0);
    }

    // 14. test_filter_goals_add_keeps
    #[test]
    fn test_filter_goals_add_keeps() {
        use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

        let goals = vec![ExtractedGoal {
            description: "Visit Japan".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        }];
        let ops = vec![ClassifiedOperation {
            action: MemoryAction::Add,
            target_index: None,
            new_text: Some("Visit Japan".to_string()),
            fact_text: "Visit Japan".to_string(),
        }];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        assert_eq!(approved.len(), 1);
        assert_eq!(approved[0].description, "Visit Japan");
        assert_eq!(dedup, 0);
    }

    // 15. test_filter_goals_none_filters
    #[test]
    fn test_filter_goals_none_filters() {
        use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

        let goals = vec![ExtractedGoal {
            description: "Visit Japan".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        }];
        let ops = vec![ClassifiedOperation {
            action: MemoryAction::None,
            target_index: None,
            new_text: None,
            fact_text: "Visit Japan".to_string(),
        }];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        assert!(approved.is_empty());
        assert_eq!(dedup, 1);
    }

    // 16. test_filter_goals_mixed
    #[test]
    fn test_filter_goals_mixed() {
        use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

        let goals = vec![
            ExtractedGoal {
                description: "Visit Japan".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Learn Rust".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Buy groceries".to_string(),
                status: "completed".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Read a book".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
        ];
        let ops = vec![
            ClassifiedOperation {
                action: MemoryAction::Add,
                target_index: None,
                new_text: Some("Visit Japan".to_string()),
                fact_text: "Visit Japan".to_string(),
            },
            ClassifiedOperation {
                action: MemoryAction::None,
                target_index: None,
                new_text: None,
                fact_text: "Learn Rust".to_string(),
            },
            ClassifiedOperation {
                action: MemoryAction::Delete,
                target_index: Some(0),
                new_text: None,
                fact_text: "Buy groceries".to_string(),
            },
            ClassifiedOperation {
                action: MemoryAction::Update,
                target_index: Some(1),
                new_text: Some("Read a book regularly".to_string()),
                fact_text: "Read a book".to_string(),
            },
        ];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        // ADD + UPDATE kept, NONE + DELETE filtered
        assert_eq!(approved.len(), 2);
        assert_eq!(approved[0].description, "Visit Japan");
        assert_eq!(approved[1].description, "Read a book");
        assert_eq!(dedup, 2);
    }

    // 17. test_compaction_to_events_with_filtered_goals
    #[test]
    fn test_compaction_to_events_with_filtered_goals() {
        // An empty goals vec produces only fact+step events (no Cognitive)
        let response = CompactionResponse {
            facts: vec![ExtractedFact {
                statement: "F1".to_string(),
                subject: "s".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
                confidence: 0.9,
            }],
            goals: vec![], // all goals filtered out
            procedural_summary: Some(ProceduralSummary {
                objective: "Obj".to_string(),
                progress_status: "completed".to_string(),
                steps: vec![ProceduralStep {
                    step_number: 1,
                    action: "A".to_string(),
                    result: "R".to_string(),
                    outcome: "success".to_string(),
                }],
                overall_summary: "S".to_string(),
                takeaway: "T".to_string(),
            }),
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        // 1 fact + 0 goals + 1 step = 2
        assert_eq!(events.len(), 2);

        // No Cognitive events
        for evt in &events {
            assert!(
                !matches!(&evt.event_type, EventType::Cognitive { .. }),
                "Expected no Cognitive events when goals are filtered"
            );
        }
    }

    // 18. test_fallback_keeps_all_goals
    #[test]
    fn test_fallback_keeps_all_goals() {
        // When no classification is provided (more goals than ops), extras default to ADD
        use crate::memory_classifier::ClassifiedOperation;

        let goals = vec![
            ExtractedGoal {
                description: "G1".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "G2".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "G3".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
        ];
        // Empty ops — all goals should pass through (fallback to Add)
        let ops: Vec<ClassifiedOperation> = vec![];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        assert_eq!(approved.len(), 3);
        assert_eq!(dedup, 0);
    }

    // 19. test_fast_goal_dedup_filters_duplicates
    #[test]
    fn test_fast_goal_dedup_filters_duplicates() {
        use crate::goal_store::{GoalDedupDecision, GoalStore};

        let mut store = GoalStore::new();

        // First time: all goals are new
        let goals = vec![
            ExtractedGoal {
                description: "Plan trip to Japan".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Learn Rust programming".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
        ];

        let mut new_goals = Vec::new();
        let mut dedup_count = 0usize;
        for goal in &goals {
            match store.store_or_dedup(&goal.description, &goal.status, &goal.owner, "case1") {
                GoalDedupDecision::NewGoal => new_goals.push(goal.clone()),
                GoalDedupDecision::Duplicate { .. } => dedup_count += 1,
                GoalDedupDecision::StatusUpdate {
                    existing_id,
                    new_status,
                } => {
                    store.update_status(existing_id, new_status);
                    dedup_count += 1;
                },
            }
        }
        assert_eq!(new_goals.len(), 2);
        assert_eq!(dedup_count, 0);

        // Second time: same goals → all duplicates
        let mut new_goals2 = Vec::new();
        let mut dedup_count2 = 0usize;
        for goal in &goals {
            match store.store_or_dedup(&goal.description, &goal.status, &goal.owner, "case2") {
                GoalDedupDecision::NewGoal => new_goals2.push(goal.clone()),
                GoalDedupDecision::Duplicate { .. } => dedup_count2 += 1,
                GoalDedupDecision::StatusUpdate {
                    existing_id,
                    new_status,
                } => {
                    store.update_status(existing_id, new_status);
                    dedup_count2 += 1;
                },
            }
        }
        assert!(new_goals2.is_empty(), "All goals should be deduplicated");
        assert_eq!(dedup_count2, 2);
    }

    // 20. test_goal_playbook_serde
    #[test]
    fn test_goal_playbook_serde() {
        let playbook = GoalPlaybook {
            goal_description: "Deploy the app".to_string(),
            what_worked: vec!["Docker build".to_string()],
            what_didnt_work: vec!["Manual deployment".to_string()],
            lessons_learned: vec!["Always use CI/CD".to_string()],
            steps_taken: vec![
                "Build".to_string(),
                "Push".to_string(),
                "Deploy".to_string(),
            ],
            confidence: 0.85,
        };

        let json = serde_json::to_string(&playbook).unwrap();
        let roundtrip: GoalPlaybook = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.goal_description, "Deploy the app");
        assert_eq!(roundtrip.what_worked.len(), 1);
        assert_eq!(roundtrip.what_didnt_work.len(), 1);
        assert_eq!(roundtrip.lessons_learned.len(), 1);
        assert_eq!(roundtrip.steps_taken.len(), 3);
        assert!((roundtrip.confidence - 0.85).abs() < f32::EPSILON);
    }

    // 21. test_playbook_extraction_response_serde
    #[test]
    fn test_playbook_extraction_response_serde() {
        let json = r#"{
            "playbooks": [
                {
                    "goal_description": "Plan trip",
                    "what_worked": ["Booked flights early"],
                    "what_didnt_work": ["Waited too long for hotel"],
                    "lessons_learned": ["Book everything 2 months ahead"],
                    "steps_taken": ["Research", "Book flights", "Find hotel"],
                    "confidence": 0.9
                }
            ]
        }"#;

        let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.playbooks.len(), 1);
        assert_eq!(response.playbooks[0].goal_description, "Plan trip");
        assert_eq!(response.playbooks[0].what_worked[0], "Booked flights early");
    }

    // 22. test_playbook_extraction_response_empty
    #[test]
    fn test_playbook_extraction_response_empty() {
        let json = r#"{"playbooks": []}"#;
        let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
        assert!(response.playbooks.is_empty());
    }

    // 23. test_playbook_extraction_response_partial
    #[test]
    fn test_playbook_extraction_response_partial() {
        // Missing optional fields (steps_taken, confidence) should use defaults
        let json = r#"{
            "playbooks": [
                {
                    "goal_description": "Learn Rust",
                    "what_worked": ["Read the book"],
                    "what_didnt_work": [],
                    "lessons_learned": ["Practice daily"]
                }
            ]
        }"#;

        let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.playbooks.len(), 1);
        let pb = &response.playbooks[0];
        assert!(
            pb.steps_taken.is_empty(),
            "steps_taken should default to empty"
        );
        assert!(
            (pb.confidence - 0.5).abs() < f32::EPSILON,
            "confidence should default to 0.5"
        );
    }

    // 24. test_compaction_result_playbooks_field
    #[test]
    fn test_compaction_result_playbooks_field() {
        let result = CompactionResult::default();
        assert_eq!(result.playbooks_extracted, 0);
    }

    // ── Rolling Summary Tests ──

    // 25. test_conversation_rolling_summary_serde
    #[test]
    fn test_conversation_rolling_summary_serde() {
        let summary = ConversationRollingSummary {
            case_id: "case_123".to_string(),
            summary: "User wants to plan a trip to Japan".to_string(),
            last_updated: 1_000_000_000,
            turn_count: 5,
            token_estimate: 42,
        };

        let json = serde_json::to_string(&summary).unwrap();
        let roundtrip: ConversationRollingSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.case_id, "case_123");
        assert_eq!(roundtrip.summary, "User wants to plan a trip to Japan");
        assert_eq!(roundtrip.last_updated, 1_000_000_000);
        assert_eq!(roundtrip.turn_count, 5);
        assert_eq!(roundtrip.token_estimate, 42);
    }

    // 26. test_format_with_summary
    #[test]
    fn test_format_with_summary() {
        let summary = ConversationRollingSummary {
            case_id: "test".to_string(),
            summary: "User is planning a trip to Japan in April.".to_string(),
            last_updated: 0,
            turn_count: 3,
            token_estimate: 10,
        };
        let data = make_ingest(vec![
            ("user", "I want to visit Tokyo"),
            ("assistant", "Great choice!"),
            ("user", "What about Kyoto?"),
            ("assistant", "Kyoto is wonderful too"),
            ("user", "Let's add Osaka"),
        ]);

        let result = format_with_summary(&summary, &data, 2);
        assert!(result.contains("[Rolling Summary]"));
        assert!(result.contains("User is planning a trip to Japan in April."));
        assert!(result.contains("[Recent Messages]"));
        // Only last 2 messages
        assert!(result.contains("assistant: Kyoto is wonderful too"));
        assert!(result.contains("user: Let's add Osaka"));
        // Earlier messages should NOT be present
        assert!(!result.contains("I want to visit Tokyo"));
    }

    // 27. test_format_with_summary_few_messages
    #[test]
    fn test_format_with_summary_few_messages() {
        let summary = ConversationRollingSummary {
            case_id: "test".to_string(),
            summary: "Summary here.".to_string(),
            last_updated: 0,
            turn_count: 1,
            token_estimate: 5,
        };
        let data = make_ingest(vec![("user", "Hello"), ("assistant", "Hi there")]);

        // recent_count > actual messages: should include all
        let result = format_with_summary(&summary, &data, 10);
        assert!(result.contains("user: Hello"));
        assert!(result.contains("assistant: Hi there"));
    }

    // 28. test_update_rolling_summary_first_call
    #[tokio::test]
    async fn test_update_rolling_summary_first_call() {
        use crate::llm_client::LlmResponse;

        // Create a simple mock that returns a summary
        struct SimpleLlm;
        #[async_trait::async_trait]
        impl LlmClient for SimpleLlm {
            async fn complete(
                &self,
                req: crate::llm_client::LlmRequest,
            ) -> anyhow::Result<LlmResponse> {
                // First call has no existing summary → prompt starts with "Summarize"
                assert!(req.user_prompt.contains("Summarize this conversation"));
                Ok(LlmResponse {
                    content: "User discussed trip plans to Japan.".to_string(),
                    tokens_used: 10,
                })
            }
            fn model_name(&self) -> &str {
                "test"
            }
        }

        let messages = vec![crate::conversation::types::ConversationMessage {
            role: "user".to_string(),
            content: "I want to go to Japan".to_string(),
        }];

        let result = update_rolling_summary(&SimpleLlm, None, &messages).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "User discussed trip plans to Japan.");
    }
}
