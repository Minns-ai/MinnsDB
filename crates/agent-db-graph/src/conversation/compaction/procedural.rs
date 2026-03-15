//! Procedural memory creation, updates, and playbook attachment.

use super::types::*;
use crate::episodes::EpisodeOutcome;
use crate::memory::{Memory, MemoryTier, MemoryType};
use crate::memory_audit::MutationActor;
use crate::memory_classifier::{resolve_target, ClassifiedOperation, MemoryAction};
use agent_db_events::core::EventContext;
use std::collections::HashMap;

// ────────── Procedural Memory Construction ──────────

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

// ────────── Procedural Memory Handling ──────────

/// Handle procedural memory with a classification op.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_procedural_memory(
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
pub(crate) async fn handle_procedural_memory_fallback(
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

// ────────── Playbook Attachment ──────────

/// Attach extracted playbooks to GoalStore entries and graph node properties.
pub(crate) async fn attach_playbooks(
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

// ────────── ID Helpers ──────────

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
