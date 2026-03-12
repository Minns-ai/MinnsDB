//! Memory mutation audit trail.
//!
//! Tracks every ADD, UPDATE, and DELETE operation on memories with
//! old→new transitions, actor identification, and timestamps.
//! Adapted to our in-memory + redb persistence model.

use crate::memory::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The type of mutation that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryMutationType {
    /// New memory created (from episode formation, consolidation, or LLM classification).
    Add,
    /// Existing memory updated (refinement, consolidation merge, LLM update decision).
    Update,
    /// Memory deleted or archived (consolidation archive, LLM delete decision, decay).
    Delete,
}

/// Who or what triggered the mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationActor {
    /// Pipeline-driven (episode formation, consolidation, decay).
    Pipeline,
    /// LLM-driven update classification.
    LlmClassifier,
    /// LLM refinement of summary/takeaway/causal_note.
    Refinement,
    /// Manual API call.
    Api,
    /// Conversation ingestion bridge.
    ConversationBridge,
}

/// A single audit entry recording one mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAuditEntry {
    /// Auto-incrementing entry ID.
    pub id: u64,
    /// The memory that was mutated.
    pub memory_id: MemoryId,
    /// Type of mutation.
    pub mutation_type: MemoryMutationType,
    /// Snapshot of the memory summary BEFORE the mutation (None for Add).
    pub old_summary: Option<String>,
    /// Snapshot of the memory summary AFTER the mutation (None for Delete).
    pub new_summary: Option<String>,
    /// Old takeaway (None for Add).
    pub old_takeaway: Option<String>,
    /// New takeaway (None for Delete).
    pub new_takeaway: Option<String>,
    /// Who triggered this mutation.
    pub actor: MutationActor,
    /// Unix timestamp in nanoseconds.
    pub timestamp: u64,
    /// Optional reason / context for the mutation.
    pub reason: Option<String>,
}

/// Append-only audit log for memory mutations.
#[derive(Debug)]
pub struct MemoryAuditLog {
    entries: Vec<MemoryAuditEntry>,
    next_id: u64,
    /// Index: memory_id → entry indices for fast per-memory history lookup.
    memory_index: HashMap<MemoryId, Vec<usize>>,
}

impl MemoryAuditLog {
    /// Create a new empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            memory_index: HashMap::new(),
        }
    }

    /// Record a new mutation.
    pub fn record(&mut self, mut entry: MemoryAuditEntry) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        entry.id = id;

        let idx = self.entries.len();
        let memory_id = entry.memory_id;
        self.entries.push(entry);

        self.memory_index.entry(memory_id).or_default().push(idx);

        id
    }

    /// Record an ADD mutation.
    pub fn record_add(
        &mut self,
        memory_id: MemoryId,
        new_summary: &str,
        new_takeaway: &str,
        actor: MutationActor,
        reason: Option<String>,
    ) -> u64 {
        self.record(MemoryAuditEntry {
            id: 0,
            memory_id,
            mutation_type: MemoryMutationType::Add,
            old_summary: None,
            new_summary: Some(new_summary.to_string()),
            old_takeaway: None,
            new_takeaway: Some(new_takeaway.to_string()),
            actor,
            timestamp: agent_db_core::types::current_timestamp(),
            reason,
        })
    }

    /// Record an UPDATE mutation.
    #[allow(clippy::too_many_arguments)]
    pub fn record_update(
        &mut self,
        memory_id: MemoryId,
        old_summary: &str,
        new_summary: &str,
        old_takeaway: &str,
        new_takeaway: &str,
        actor: MutationActor,
        reason: Option<String>,
    ) -> u64 {
        self.record(MemoryAuditEntry {
            id: 0,
            memory_id,
            mutation_type: MemoryMutationType::Update,
            old_summary: Some(old_summary.to_string()),
            new_summary: Some(new_summary.to_string()),
            old_takeaway: Some(old_takeaway.to_string()),
            new_takeaway: Some(new_takeaway.to_string()),
            actor,
            timestamp: agent_db_core::types::current_timestamp(),
            reason,
        })
    }

    /// Record a DELETE mutation.
    pub fn record_delete(
        &mut self,
        memory_id: MemoryId,
        old_summary: &str,
        old_takeaway: &str,
        actor: MutationActor,
        reason: Option<String>,
    ) -> u64 {
        self.record(MemoryAuditEntry {
            id: 0,
            memory_id,
            mutation_type: MemoryMutationType::Delete,
            old_summary: Some(old_summary.to_string()),
            new_summary: None,
            old_takeaway: Some(old_takeaway.to_string()),
            new_takeaway: None,
            actor,
            timestamp: agent_db_core::types::current_timestamp(),
            reason,
        })
    }

    /// Get the full history for a specific memory.
    pub fn history_for(&self, memory_id: MemoryId) -> Vec<&MemoryAuditEntry> {
        self.memory_index
            .get(&memory_id)
            .map(|indices| indices.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Get the N most recent entries across all memories.
    pub fn recent(&self, limit: usize) -> Vec<&MemoryAuditEntry> {
        self.entries.iter().rev().take(limit).collect()
    }

    /// Total number of audit entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count mutations of a specific type.
    pub fn count_by_type(&self, mutation_type: MemoryMutationType) -> usize {
        self.entries
            .iter()
            .filter(|e| e.mutation_type == mutation_type)
            .count()
    }

    /// All entries (for serialization/export).
    pub fn all_entries(&self) -> &[MemoryAuditEntry] {
        &self.entries
    }
}

impl Default for MemoryAuditLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_record_add() {
        let mut log = MemoryAuditLog::new();
        let id = log.record_add(
            42,
            "Agent performed login",
            "Always authenticate first",
            MutationActor::Pipeline,
            None,
        );
        assert_eq!(id, 1);
        assert_eq!(log.len(), 1);

        let history = log.history_for(42);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].mutation_type, MemoryMutationType::Add);
        assert_eq!(
            history[0].new_summary.as_deref(),
            Some("Agent performed login")
        );
        assert!(history[0].old_summary.is_none());
    }

    #[test]
    fn test_audit_log_record_update() {
        let mut log = MemoryAuditLog::new();
        log.record_add(
            42,
            "old text",
            "old takeaway",
            MutationActor::Pipeline,
            None,
        );
        log.record_update(
            42,
            "old text",
            "refined text",
            "old takeaway",
            "refined takeaway",
            MutationActor::Refinement,
            Some("LLM refinement improved clarity".to_string()),
        );

        let history = log.history_for(42);
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].mutation_type, MemoryMutationType::Update);
        assert_eq!(history[1].old_summary.as_deref(), Some("old text"));
        assert_eq!(history[1].new_summary.as_deref(), Some("refined text"));
        assert_eq!(history[1].actor, MutationActor::Refinement);
    }

    #[test]
    fn test_audit_log_record_delete() {
        let mut log = MemoryAuditLog::new();
        log.record_add(42, "memory", "takeaway", MutationActor::Pipeline, None);
        log.record_delete(
            42,
            "memory",
            "takeaway",
            MutationActor::LlmClassifier,
            Some("Superseded by newer memory".to_string()),
        );

        assert_eq!(log.len(), 2);
        assert_eq!(log.count_by_type(MemoryMutationType::Delete), 1);
    }

    #[test]
    fn test_audit_log_recent() {
        let mut log = MemoryAuditLog::new();
        for i in 1..=10 {
            log.record_add(
                i,
                &format!("memory {}", i),
                "t",
                MutationActor::Pipeline,
                None,
            );
        }

        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent first
        assert_eq!(recent[0].memory_id, 10);
        assert_eq!(recent[1].memory_id, 9);
        assert_eq!(recent[2].memory_id, 8);
    }

    #[test]
    fn test_audit_log_empty_history() {
        let log = MemoryAuditLog::new();
        assert!(log.history_for(999).is_empty());
        assert!(log.is_empty());
    }

    #[test]
    fn test_audit_log_multi_memory_index() {
        let mut log = MemoryAuditLog::new();
        log.record_add(1, "m1", "t1", MutationActor::Pipeline, None);
        log.record_add(2, "m2", "t2", MutationActor::Pipeline, None);
        log.record_update(
            1,
            "m1",
            "m1v2",
            "t1",
            "t1v2",
            MutationActor::Refinement,
            None,
        );

        assert_eq!(log.history_for(1).len(), 2);
        assert_eq!(log.history_for(2).len(), 1);
    }
}
