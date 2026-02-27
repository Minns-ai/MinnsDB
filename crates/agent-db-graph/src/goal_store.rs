//! In-memory goal store with BM25-backed fast deduplication.
//!
//! Mirrors the claim dedup pattern (`check_claim_dedup()` in maintenance.rs)
//! but is lightweight: no redb persistence, no vector embeddings.
//! Goals are rare enough that in-memory BM25 + Jaro-Winkler similarity suffices.

use crate::conversation::compaction::GoalPlaybook;
use crate::indexing::Bm25Index;
use crate::structures::GoalStatus;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Similarity threshold for deduplication (same as entity fuzzy matching).
const DEDUP_SIMILARITY_THRESHOLD: f64 = 0.85;

/// Maximum BM25 candidates to consider for dedup.
const DEDUP_TOP_K: usize = 3;

/// A stored goal entry.
#[derive(Debug, Clone)]
pub struct GoalEntry {
    /// Deterministic hash of the description.
    pub id: u64,
    /// Goal description text.
    pub description: String,
    /// Current goal status.
    pub status: GoalStatus,
    /// Who owns this goal (e.g. "user", "assistant").
    pub owner: String,
    /// Timestamp (nanos) when first created.
    pub created_at: u64,
    /// Timestamp (nanos) when last updated.
    pub updated_at: u64,
    /// How many times this goal was re-detected.
    pub support_count: u32,
    /// Conversation case IDs that mentioned this goal.
    pub case_ids: Vec<String>,
    /// Retrospective playbook attached after compaction.
    pub playbook: Option<GoalPlaybook>,
}

/// Result of the deduplication check.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalDedupDecision {
    /// No similar goal found — store as new.
    NewGoal,
    /// Existing goal is essentially the same (same or similar status).
    Duplicate { existing_id: u64, similarity: f64 },
    /// Same goal detected with a different status — update the existing entry.
    StatusUpdate {
        existing_id: u64,
        new_status: GoalStatus,
    },
}

/// In-memory goal store with BM25 dedup.
pub struct GoalStore {
    goals: HashMap<u64, GoalEntry>,
    bm25_index: Bm25Index,
}

impl Default for GoalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalStore {
    /// Create an empty goal store.
    pub fn new() -> Self {
        Self {
            goals: HashMap::new(),
            bm25_index: Bm25Index::new(),
        }
    }

    /// Store a goal or deduplicate against existing goals.
    ///
    /// 1. BM25 search for similar goals (top 3).
    /// 2. For each hit, compute Jaro-Winkler similarity on descriptions.
    /// 3. If similarity >= 0.85:
    ///    - Same status → `Duplicate` (increment support_count).
    ///    - Different status → `StatusUpdate`.
    /// 4. No match → `NewGoal` (inserted into store + BM25 index).
    pub fn store_or_dedup(
        &mut self,
        description: &str,
        status: &str,
        owner: &str,
        case_id: &str,
    ) -> GoalDedupDecision {
        let new_status = parse_goal_status(status);
        let desc_lower = description.to_lowercase();

        // BM25 candidate search
        let candidates = self.bm25_index.search(description, DEDUP_TOP_K);

        for (candidate_id, _bm25_score) in &candidates {
            if let Some(existing) = self.goals.get_mut(candidate_id) {
                let existing_lower = existing.description.to_lowercase();
                let sim = strsim::jaro_winkler(&desc_lower, &existing_lower);

                if sim >= DEDUP_SIMILARITY_THRESHOLD {
                    let eid = existing.id;

                    if existing.status == new_status {
                        // Same status → duplicate
                        existing.support_count += 1;
                        existing.updated_at = now_nanos();
                        if !existing.case_ids.contains(&case_id.to_string()) {
                            existing.case_ids.push(case_id.to_string());
                        }
                        return GoalDedupDecision::Duplicate {
                            existing_id: eid,
                            similarity: sim,
                        };
                    } else {
                        // Different status → status update
                        return GoalDedupDecision::StatusUpdate {
                            existing_id: eid,
                            new_status,
                        };
                    }
                }
            }
        }

        // No match — insert as new goal
        let id = goal_id_hash(description);
        let entry = GoalEntry {
            id,
            description: description.to_string(),
            status: new_status,
            owner: owner.to_string(),
            created_at: now_nanos(),
            updated_at: now_nanos(),
            support_count: 1,
            case_ids: vec![case_id.to_string()],
            playbook: None,
        };
        self.goals.insert(id, entry);
        self.bm25_index.index_document(id, description);

        GoalDedupDecision::NewGoal
    }

    /// Update the status of an existing goal.
    pub fn update_status(&mut self, id: u64, status: GoalStatus) {
        if let Some(entry) = self.goals.get_mut(&id) {
            entry.status = status;
            entry.updated_at = now_nanos();
        }
    }

    /// Get a goal by ID.
    pub fn get(&self, id: u64) -> Option<&GoalEntry> {
        self.goals.get(&id)
    }

    /// Get all active goals, ordered by most recently updated, limited to `limit`.
    pub fn get_all_active(&self, limit: usize) -> Vec<&GoalEntry> {
        let mut active: Vec<&GoalEntry> = self
            .goals
            .values()
            .filter(|e| e.status == GoalStatus::Active)
            .collect();
        active.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        active.truncate(limit);
        active
    }

    /// Number of goals currently stored.
    pub fn len(&self) -> usize {
        self.goals.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.goals.is_empty()
    }

    /// BM25 search for similar goals (returns (id, score) pairs).
    pub fn find_similar(&self, description: &str, top_k: usize) -> Vec<(u64, f32)> {
        self.bm25_index.search(description, top_k)
    }

    /// Attach a retrospective playbook to an existing goal.
    ///
    /// No-op if the goal ID does not exist.
    pub fn attach_playbook(&mut self, id: u64, playbook: GoalPlaybook) {
        if let Some(entry) = self.goals.get_mut(&id) {
            entry.playbook = Some(playbook);
            entry.updated_at = now_nanos();
        }
    }
}

// ────────── Helpers ──────────

/// Deterministic goal ID from description text.
fn goal_id_hash(description: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    description.to_lowercase().hash(&mut hasher);
    hasher.finish()
}

/// Parse a status string into `GoalStatus`.
fn parse_goal_status(s: &str) -> GoalStatus {
    match s.to_lowercase().as_str() {
        "active" => GoalStatus::Active,
        "completed" => GoalStatus::Completed,
        "failed" => GoalStatus::Failed,
        "paused" => GoalStatus::Paused,
        "abandoned" => GoalStatus::Failed, // treat abandoned as failed
        _ => GoalStatus::Active,           // default to active
    }
}

/// Current timestamp in nanoseconds.
fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ────────── Tests ──────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_goal_stored() {
        let mut store = GoalStore::new();
        let decision = store.store_or_dedup("Plan trip to Japan", "active", "user", "case1");
        assert_eq!(decision, GoalDedupDecision::NewGoal);
        assert_eq!(store.len(), 1);

        let id = goal_id_hash("Plan trip to Japan");
        let entry = store.get(id).unwrap();
        assert_eq!(entry.description, "Plan trip to Japan");
        assert_eq!(entry.status, GoalStatus::Active);
        assert_eq!(entry.owner, "user");
        assert_eq!(entry.support_count, 1);
        assert_eq!(entry.case_ids, vec!["case1".to_string()]);
    }

    #[test]
    fn test_duplicate_detection() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Plan trip to Japan", "active", "user", "case1");

        let decision = store.store_or_dedup("Plan trip to Japan", "active", "user", "case2");
        match decision {
            GoalDedupDecision::Duplicate { similarity, .. } => {
                assert!(similarity >= 0.85);
            },
            other => panic!("Expected Duplicate, got {:?}", other),
        }
        // Should NOT have created a new entry
        assert_eq!(store.len(), 1);

        // Support count should be incremented
        let id = goal_id_hash("Plan trip to Japan");
        let entry = store.get(id).unwrap();
        assert_eq!(entry.support_count, 2);
        assert!(entry.case_ids.contains(&"case2".to_string()));
    }

    #[test]
    fn test_similar_goals_deduplicated() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Plan trip to Japan", "active", "user", "case1");

        // Slightly different wording — should still match
        let decision = store.store_or_dedup("Plan a trip to Japan", "active", "user", "case2");
        match decision {
            GoalDedupDecision::Duplicate { similarity, .. } => {
                assert!(similarity >= 0.85, "Similarity was {}", similarity);
            },
            other => panic!("Expected Duplicate, got {:?}", other),
        }
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_status_update_detection() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Plan trip to Japan", "active", "user", "case1");

        let decision = store.store_or_dedup("Plan trip to Japan", "completed", "user", "case2");
        match decision {
            GoalDedupDecision::StatusUpdate { new_status, .. } => {
                assert_eq!(new_status, GoalStatus::Completed);
            },
            other => panic!("Expected StatusUpdate, got {:?}", other),
        }
        // No new entry created (caller should call update_status)
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_different_goals_not_deduplicated() {
        let mut store = GoalStore::new();
        let d1 = store.store_or_dedup("Learn Rust programming", "active", "user", "case1");
        let d2 = store.store_or_dedup("Buy groceries for dinner", "active", "user", "case1");
        assert_eq!(d1, GoalDedupDecision::NewGoal);
        assert_eq!(d2, GoalDedupDecision::NewGoal);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_support_count_incremented() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Learn Rust", "active", "user", "case1");
        store.store_or_dedup("Learn Rust", "active", "user", "case2");
        store.store_or_dedup("Learn Rust", "active", "user", "case3");

        let id = goal_id_hash("Learn Rust");
        let entry = store.get(id).unwrap();
        assert_eq!(entry.support_count, 3);
        assert_eq!(entry.case_ids.len(), 3);
    }

    #[test]
    fn test_bm25_search() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Plan trip to Japan", "active", "user", "case1");
        store.store_or_dedup("Learn Rust programming language", "active", "user", "case1");
        store.store_or_dedup("Buy groceries for dinner", "active", "user", "case1");

        let results = store.find_similar("trip to Japan", 3);
        assert!(!results.is_empty(), "BM25 should find matching goals");

        // First result should be the Japan goal
        let japan_id = goal_id_hash("Plan trip to Japan");
        assert_eq!(results[0].0, japan_id);
    }

    #[test]
    fn test_update_status() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Learn Rust", "active", "user", "case1");

        let id = goal_id_hash("Learn Rust");
        store.update_status(id, GoalStatus::Completed);

        let entry = store.get(id).unwrap();
        assert_eq!(entry.status, GoalStatus::Completed);
    }

    #[test]
    fn test_get_all_active() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Learn Rust programming", "active", "user", "c1");
        store.store_or_dedup("Finish reading the novel", "completed", "user", "c1");
        store.store_or_dedup("Plan trip to Japan", "active", "user", "c1");

        let active = store.get_all_active(10);
        assert_eq!(active.len(), 2);
        for entry in &active {
            assert_eq!(entry.status, GoalStatus::Active);
        }
    }

    #[test]
    fn test_parse_goal_status_variants() {
        assert_eq!(parse_goal_status("active"), GoalStatus::Active);
        assert_eq!(parse_goal_status("Active"), GoalStatus::Active);
        assert_eq!(parse_goal_status("completed"), GoalStatus::Completed);
        assert_eq!(parse_goal_status("failed"), GoalStatus::Failed);
        assert_eq!(parse_goal_status("paused"), GoalStatus::Paused);
        assert_eq!(parse_goal_status("abandoned"), GoalStatus::Failed);
        assert_eq!(parse_goal_status("unknown"), GoalStatus::Active);
    }

    #[test]
    fn test_is_empty() {
        let store = GoalStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn test_duplicate_case_id_not_added_twice() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Learn Rust", "active", "user", "case1");
        store.store_or_dedup("Learn Rust", "active", "user", "case1");

        let id = goal_id_hash("Learn Rust");
        let entry = store.get(id).unwrap();
        // case_id "case1" should only appear once
        assert_eq!(entry.case_ids.iter().filter(|c| *c == "case1").count(), 1);
    }

    #[test]
    fn test_attach_playbook_to_goal() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Deploy the application", "active", "user", "case1");

        let id = goal_id_hash("Deploy the application");
        let playbook = GoalPlaybook {
            goal_description: "Deploy the application".to_string(),
            what_worked: vec!["Docker build".to_string()],
            what_didnt_work: vec!["Manual deploy".to_string()],
            lessons_learned: vec!["Use CI/CD".to_string()],
            steps_taken: vec!["Build".to_string(), "Push".to_string()],
            confidence: 0.9,
        };

        store.attach_playbook(id, playbook);

        let entry = store.get(id).unwrap();
        assert!(entry.playbook.is_some());
        let pb = entry.playbook.as_ref().unwrap();
        assert_eq!(pb.goal_description, "Deploy the application");
        assert_eq!(pb.what_worked.len(), 1);
        assert!((pb.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attach_playbook_nonexistent_goal() {
        let mut store = GoalStore::new();
        let playbook = GoalPlaybook {
            goal_description: "Nonexistent".to_string(),
            what_worked: vec![],
            what_didnt_work: vec![],
            lessons_learned: vec![],
            steps_taken: vec![],
            confidence: 0.5,
        };

        // Should not panic
        store.attach_playbook(999999, playbook);
        assert!(store.is_empty());
    }

    #[test]
    fn test_goal_entry_playbook_default_none() {
        let mut store = GoalStore::new();
        store.store_or_dedup("Learn Rust", "active", "user", "case1");

        let id = goal_id_hash("Learn Rust");
        let entry = store.get(id).unwrap();
        assert!(entry.playbook.is_none());
    }
}
