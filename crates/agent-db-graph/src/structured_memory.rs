//! Dynamic structured memory templates for EventGraphDB.
//!
//! Provides typed memory structures (ledgers, trees, state machines, preference lists)
//! with O(1) current-state lookup and append-only ledger balances.
//!
//! Keys are built from stable `NodeId` pairs, not display names, to avoid
//! alias/case/rename collisions.

use crate::structures::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// How a structured memory was created.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryProvenance {
    /// Created via direct API upsert.
    Manual,
    /// Auto-extracted from the event/episode pipeline.
    EpisodePipeline,
    /// Created via a natural-language query mutation.
    NlqUpsert,
}

// ---------------------------------------------------------------------------
// Canonical key builders (Section A of the plan)
// ---------------------------------------------------------------------------

/// Build a canonical ledger key from two `NodeId`s (sorted, stable).
pub fn ledger_key(id_a: NodeId, id_b: NodeId) -> String {
    let (lo, hi) = if id_a <= id_b {
        (id_a, id_b)
    } else {
        (id_b, id_a)
    };
    format!("ledger:{}:{}", lo, hi)
}

/// Build a canonical state-machine key from an entity `NodeId`.
pub fn state_key(entity_id: NodeId) -> String {
    format!("state:{}", entity_id)
}

/// Build a canonical preference-list key from an entity `NodeId` and category.
pub fn prefs_key(entity_id: NodeId, category: &str) -> String {
    format!("prefs:{}:{}", entity_id, category.to_lowercase())
}

/// Build a canonical tree key from a root `NodeId`.
pub fn tree_key(root_id: NodeId) -> String {
    format!("tree:{}", root_id)
}

// ---------------------------------------------------------------------------
// Sub-types
// ---------------------------------------------------------------------------

/// A single ledger entry (append-only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub timestamp: u64,
    pub amount: f64,
    pub description: String,
    pub direction: LedgerDirection,
}

/// Direction of a ledger entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LedgerDirection {
    Credit,
    Debit,
}

/// A single state transition record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from: String,
    pub to: String,
    pub timestamp: u64,
    pub trigger: String,
}

/// A single preference item with optional score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceItem {
    pub name: String,
    pub rank: usize,
    pub score: Option<f64>,
}

// ---------------------------------------------------------------------------
// MemoryTemplate enum
// ---------------------------------------------------------------------------

/// Memory structure template — the agent selects the right one per entity/domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTemplate {
    /// Append-only ledger with running balance (debts, transactions).
    Ledger {
        entity_pair: (String, String),
        entries: Vec<LedgerEntry>,
        balance: f64,
        provenance: MemoryProvenance,
    },
    /// Hierarchical parent-child structure (org charts, file systems).
    Tree {
        root: String,
        children: HashMap<String, Vec<String>>,
        provenance: MemoryProvenance,
    },
    /// Entity with named states and transition history.
    StateMachine {
        entity: String,
        current_state: String,
        history: Vec<StateTransition>,
        provenance: MemoryProvenance,
    },
    /// Ordered ranking with optional scores.
    PreferenceList {
        entity: String,
        ranked_items: Vec<PreferenceItem>,
        provenance: MemoryProvenance,
    },
}

// ---------------------------------------------------------------------------
// StructuredMemoryStore
// ---------------------------------------------------------------------------

/// In-memory store for structured memories, keyed by canonical domain keys.
///
/// Domain key examples: `"ledger:3:5"`, `"state:42"`, `"prefs:7:food"`.
#[derive(Debug, Default)]
pub struct StructuredMemoryStore {
    memories: HashMap<String, MemoryTemplate>,
}

impl StructuredMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Upsert a structured memory by key.
    pub fn upsert(&mut self, key: &str, template: MemoryTemplate) {
        self.memories.insert(key.to_string(), template);
    }

    /// Get a structured memory by key.
    pub fn get(&self, key: &str) -> Option<&MemoryTemplate> {
        self.memories.get(key)
    }

    /// List all keys matching a prefix.
    pub fn list_keys(&self, prefix: &str) -> Vec<&str> {
        self.memories
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.as_str())
            .collect()
    }

    /// Remove a structured memory by key.
    pub fn remove(&mut self, key: &str) -> Option<MemoryTemplate> {
        self.memories.remove(key)
    }

    /// Total number of stored templates.
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    // =======================================================================
    // Ledger mutation
    // =======================================================================

    /// Append a ledger entry and recompute balance from the full entry fold.
    ///
    /// Returns the new balance or an error if the key doesn't point to a Ledger
    /// or the amount is not finite.
    pub fn ledger_append(&mut self, key: &str, entry: LedgerEntry) -> Result<f64, String> {
        if !entry.amount.is_finite() {
            return Err(format!("Invalid amount: {} (must be finite)", entry.amount));
        }

        let template = self
            .memories
            .get_mut(key)
            .ok_or_else(|| format!("Key not found: {}", key))?;

        match template {
            MemoryTemplate::Ledger {
                entries, balance, ..
            } => {
                entries.push(entry);
                // Full fold recompute (Section H — no incremental mutation)
                *balance = entries.iter().fold(0.0, |acc, e| match e.direction {
                    LedgerDirection::Credit => acc + e.amount,
                    LedgerDirection::Debit => acc - e.amount,
                });
                Ok(*balance)
            },
            _ => Err(format!("Key '{}' is not a Ledger", key)),
        }
    }

    /// Get ledger balance (O(1) — stored, verified by fold in tests).
    pub fn ledger_balance(&self, key: &str) -> Option<f64> {
        match self.memories.get(key)? {
            MemoryTemplate::Ledger { balance, .. } => Some(*balance),
            _ => None,
        }
    }

    // =======================================================================
    // State machine mutation
    // =======================================================================

    /// Transition a state machine and record history.
    pub fn state_transition(
        &mut self,
        key: &str,
        new_state: &str,
        trigger: &str,
        timestamp: u64,
    ) -> Result<(), String> {
        let template = self
            .memories
            .get_mut(key)
            .ok_or_else(|| format!("Key not found: {}", key))?;

        match template {
            MemoryTemplate::StateMachine {
                current_state,
                history,
                ..
            } => {
                let transition = StateTransition {
                    from: current_state.clone(),
                    to: new_state.to_string(),
                    timestamp,
                    trigger: trigger.to_string(),
                };
                history.push(transition);
                *current_state = new_state.to_string();
                Ok(())
            },
            _ => Err(format!("Key '{}' is not a StateMachine", key)),
        }
    }

    /// Get current state (O(1)).
    pub fn state_current(&self, key: &str) -> Option<&str> {
        match self.memories.get(key)? {
            MemoryTemplate::StateMachine { current_state, .. } => Some(current_state.as_str()),
            _ => None,
        }
    }

    // =======================================================================
    // Preference list mutation
    // =======================================================================

    /// Update or insert a preference item, re-sorting by rank.
    pub fn preference_update(
        &mut self,
        key: &str,
        item: &str,
        rank: usize,
        score: Option<f64>,
    ) -> Result<(), String> {
        let template = self
            .memories
            .get_mut(key)
            .ok_or_else(|| format!("Key not found: {}", key))?;

        match template {
            MemoryTemplate::PreferenceList { ranked_items, .. } => {
                // Remove existing item if present
                ranked_items.retain(|p| p.name != item);
                ranked_items.push(PreferenceItem {
                    name: item.to_string(),
                    rank,
                    score,
                });
                // Sort by rank ascending
                ranked_items.sort_by_key(|p| p.rank);
                Ok(())
            },
            _ => Err(format!("Key '{}' is not a PreferenceList", key)),
        }
    }

    // =======================================================================
    // Tree mutation
    // =======================================================================

    /// Add a child to a tree node.
    pub fn tree_add_child(&mut self, key: &str, parent: &str, child: &str) -> Result<(), String> {
        let template = self
            .memories
            .get_mut(key)
            .ok_or_else(|| format!("Key not found: {}", key))?;

        match template {
            MemoryTemplate::Tree { children, .. } => {
                let child_list = children.entry(parent.to_string()).or_default();
                // Prevent duplicate children
                if !child_list.contains(&child.to_string()) {
                    child_list.push(child.to_string());
                }
                Ok(())
            },
            _ => Err(format!("Key '{}' is not a Tree", key)),
        }
    }

    /// Get children of a tree node.
    pub fn tree_children(&self, key: &str, parent: &str) -> Option<&Vec<String>> {
        match self.memories.get(key)? {
            MemoryTemplate::Tree { children, .. } => children.get(parent),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Key builder tests =====

    #[test]
    fn test_ledger_key_canonical_sorted() {
        assert_eq!(ledger_key(5, 3), ledger_key(3, 5));
        assert_eq!(ledger_key(3, 5), "ledger:3:5");
    }

    #[test]
    fn test_state_key_stable() {
        assert_eq!(state_key(42), "state:42");
    }

    #[test]
    fn test_prefs_key_lowercased() {
        assert_eq!(prefs_key(7, "Food"), "prefs:7:food");
        assert_eq!(prefs_key(7, "FOOD"), "prefs:7:food");
    }

    #[test]
    fn test_tree_key() {
        assert_eq!(tree_key(10), "tree:10");
    }

    // ===== Ledger tests =====

    #[test]
    fn test_ledger_append_and_balance() {
        let mut store = StructuredMemoryStore::new();
        let key = "ledger:1:2";
        store.upsert(
            key,
            MemoryTemplate::Ledger {
                entity_pair: ("Alice".into(), "Bob".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::Manual,
            },
        );

        // Credit 50
        let bal = store
            .ledger_append(
                key,
                LedgerEntry {
                    timestamp: 100,
                    amount: 50.0,
                    description: "loan".into(),
                    direction: LedgerDirection::Credit,
                },
            )
            .unwrap();
        assert!((bal - 50.0).abs() < f64::EPSILON);

        // Debit 20
        let bal = store
            .ledger_append(
                key,
                LedgerEntry {
                    timestamp: 200,
                    amount: 20.0,
                    description: "repay".into(),
                    direction: LedgerDirection::Debit,
                },
            )
            .unwrap();
        assert!((bal - 30.0).abs() < f64::EPSILON);

        // O(1) balance lookup
        assert_eq!(store.ledger_balance(key), Some(30.0));
    }

    #[test]
    fn test_ledger_balance_fold_consistency() {
        let mut store = StructuredMemoryStore::new();
        let key = "ledger:10:20";
        store.upsert(
            key,
            MemoryTemplate::Ledger {
                entity_pair: ("A".into(), "B".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );

        // Multiple entries
        for i in 0..10 {
            let dir = if i % 3 == 0 {
                LedgerDirection::Debit
            } else {
                LedgerDirection::Credit
            };
            store
                .ledger_append(
                    key,
                    LedgerEntry {
                        timestamp: i as u64,
                        amount: (i + 1) as f64 * 10.0,
                        description: format!("tx_{}", i),
                        direction: dir,
                    },
                )
                .unwrap();
        }

        // Verify fold matches stored balance (Section H)
        let stored_balance = store.ledger_balance(key).unwrap();
        if let Some(MemoryTemplate::Ledger { entries, .. }) = store.get(key) {
            let fold_balance = entries.iter().fold(0.0, |acc, e| match e.direction {
                LedgerDirection::Credit => acc + e.amount,
                LedgerDirection::Debit => acc - e.amount,
            });
            assert!(
                (stored_balance - fold_balance).abs() < f64::EPSILON,
                "Balance {} != fold {}",
                stored_balance,
                fold_balance
            );
        }
    }

    #[test]
    fn test_ledger_append_wrong_type() {
        let mut store = StructuredMemoryStore::new();
        store.upsert(
            "state:1",
            MemoryTemplate::StateMachine {
                entity: "pkg".into(),
                current_state: "pending".into(),
                history: vec![],
                provenance: MemoryProvenance::Manual,
            },
        );
        assert!(store
            .ledger_append(
                "state:1",
                LedgerEntry {
                    timestamp: 1,
                    amount: 10.0,
                    description: "x".into(),
                    direction: LedgerDirection::Credit,
                }
            )
            .is_err());
    }

    // ===== State machine tests =====

    #[test]
    fn test_state_machine_transitions() {
        let mut store = StructuredMemoryStore::new();
        let key = "state:42";
        store.upsert(
            key,
            MemoryTemplate::StateMachine {
                entity: "package_123".into(),
                current_state: "warehouse".into(),
                history: vec![],
                provenance: MemoryProvenance::Manual,
            },
        );

        assert_eq!(store.state_current(key), Some("warehouse"));

        store
            .state_transition(key, "in_transit", "shipped", 1000)
            .unwrap();
        assert_eq!(store.state_current(key), Some("in_transit"));

        store
            .state_transition(key, "delivered", "arrived", 2000)
            .unwrap();
        assert_eq!(store.state_current(key), Some("delivered"));

        // Verify history
        if let Some(MemoryTemplate::StateMachine { history, .. }) = store.get(key) {
            assert_eq!(history.len(), 2);
            assert_eq!(history[0].from, "warehouse");
            assert_eq!(history[0].to, "in_transit");
            assert_eq!(history[1].from, "in_transit");
            assert_eq!(history[1].to, "delivered");
        }
    }

    #[test]
    fn test_state_current_o1() {
        let mut store = StructuredMemoryStore::new();
        let key = "state:99";
        store.upsert(
            key,
            MemoryTemplate::StateMachine {
                entity: "entity_x".into(),
                current_state: "init".into(),
                history: vec![],
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        // O(1) lookup, no traversal needed
        assert_eq!(store.state_current(key), Some("init"));
    }

    // ===== Tree tests =====

    #[test]
    fn test_tree_add_children() {
        let mut store = StructuredMemoryStore::new();
        let key = "tree:1";
        store.upsert(
            key,
            MemoryTemplate::Tree {
                root: "CEO".into(),
                children: HashMap::new(),
                provenance: MemoryProvenance::Manual,
            },
        );

        store.tree_add_child(key, "CEO", "VP_Eng").unwrap();
        store.tree_add_child(key, "CEO", "VP_Sales").unwrap();
        store.tree_add_child(key, "VP_Eng", "Dev_Lead").unwrap();

        assert_eq!(
            store.tree_children(key, "CEO"),
            Some(&vec!["VP_Eng".to_string(), "VP_Sales".to_string()])
        );
        assert_eq!(
            store.tree_children(key, "VP_Eng"),
            Some(&vec!["Dev_Lead".to_string()])
        );
        assert_eq!(store.tree_children(key, "VP_Sales"), None);
    }

    // ===== Preference list tests =====

    #[test]
    fn test_preference_update_and_ordering() {
        let mut store = StructuredMemoryStore::new();
        let key = "prefs:5:food";
        store.upsert(
            key,
            MemoryTemplate::PreferenceList {
                entity: "Alice".into(),
                ranked_items: vec![],
                provenance: MemoryProvenance::Manual,
            },
        );

        store.preference_update(key, "Pizza", 1, Some(9.5)).unwrap();
        store.preference_update(key, "Sushi", 2, Some(8.0)).unwrap();
        store.preference_update(key, "Tacos", 3, None).unwrap();

        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(key) {
            assert_eq!(ranked_items.len(), 3);
            assert_eq!(ranked_items[0].name, "Pizza");
            assert_eq!(ranked_items[0].rank, 1);
            assert_eq!(ranked_items[1].name, "Sushi");
            assert_eq!(ranked_items[2].name, "Tacos");
        }

        // Update existing item rank
        store
            .preference_update(key, "Sushi", 0, Some(10.0))
            .unwrap();
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(key) {
            assert_eq!(ranked_items[0].name, "Sushi");
            assert_eq!(ranked_items[0].rank, 0);
        }
    }

    // ===== Provenance tests =====

    #[test]
    fn test_provenance_tracking() {
        let mut store = StructuredMemoryStore::new();

        store.upsert(
            "ledger:1:2",
            MemoryTemplate::Ledger {
                entity_pair: ("A".into(), "B".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::Manual,
            },
        );

        store.upsert(
            "state:3",
            MemoryTemplate::StateMachine {
                entity: "pkg".into(),
                current_state: "new".into(),
                history: vec![],
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );

        store.upsert(
            "prefs:4:music",
            MemoryTemplate::PreferenceList {
                entity: "Bob".into(),
                ranked_items: vec![],
                provenance: MemoryProvenance::NlqUpsert,
            },
        );

        match store.get("ledger:1:2").unwrap() {
            MemoryTemplate::Ledger { provenance, .. } => {
                assert_eq!(*provenance, MemoryProvenance::Manual);
            },
            _ => panic!("Expected Ledger"),
        }
        match store.get("state:3").unwrap() {
            MemoryTemplate::StateMachine { provenance, .. } => {
                assert_eq!(*provenance, MemoryProvenance::EpisodePipeline);
            },
            _ => panic!("Expected StateMachine"),
        }
        match store.get("prefs:4:music").unwrap() {
            MemoryTemplate::PreferenceList { provenance, .. } => {
                assert_eq!(*provenance, MemoryProvenance::NlqUpsert);
            },
            _ => panic!("Expected PreferenceList"),
        }
    }

    // ===== CRUD tests =====

    #[test]
    fn test_list_keys_prefix() {
        let mut store = StructuredMemoryStore::new();
        store.upsert(
            "ledger:1:2",
            MemoryTemplate::Ledger {
                entity_pair: ("A".into(), "B".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::Manual,
            },
        );
        store.upsert(
            "ledger:3:4",
            MemoryTemplate::Ledger {
                entity_pair: ("C".into(), "D".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::Manual,
            },
        );
        store.upsert(
            "state:5",
            MemoryTemplate::StateMachine {
                entity: "x".into(),
                current_state: "new".into(),
                history: vec![],
                provenance: MemoryProvenance::Manual,
            },
        );

        let ledger_keys = store.list_keys("ledger:");
        assert_eq!(ledger_keys.len(), 2);
        let state_keys = store.list_keys("state:");
        assert_eq!(state_keys.len(), 1);
        let all_keys = store.list_keys("");
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_remove() {
        let mut store = StructuredMemoryStore::new();
        store.upsert(
            "state:1",
            MemoryTemplate::StateMachine {
                entity: "x".into(),
                current_state: "a".into(),
                history: vec![],
                provenance: MemoryProvenance::Manual,
            },
        );
        assert!(store.get("state:1").is_some());
        let removed = store.remove("state:1");
        assert!(removed.is_some());
        assert!(store.get("state:1").is_none());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut store = StructuredMemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.upsert(
            "state:1",
            MemoryTemplate::StateMachine {
                entity: "x".into(),
                current_state: "a".into(),
                history: vec![],
                provenance: MemoryProvenance::Manual,
            },
        );
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_ledger_rejects_nan() {
        let mut store = StructuredMemoryStore::new();
        let key = "ledger:1:2";
        store.upsert(
            key,
            MemoryTemplate::Ledger {
                entity_pair: ("A".into(), "B".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::Manual,
            },
        );

        let result = store.ledger_append(
            key,
            LedgerEntry {
                timestamp: 1,
                amount: f64::NAN,
                description: "bad".into(),
                direction: LedgerDirection::Credit,
            },
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be finite"));

        let result = store.ledger_append(
            key,
            LedgerEntry {
                timestamp: 2,
                amount: f64::INFINITY,
                description: "bad".into(),
                direction: LedgerDirection::Credit,
            },
        );
        assert!(result.is_err());

        // Balance should still be 0 (no entries accepted)
        assert_eq!(store.ledger_balance(key), Some(0.0));
    }

    #[test]
    fn test_tree_no_duplicate_children() {
        let mut store = StructuredMemoryStore::new();
        let key = "tree:1";
        store.upsert(
            key,
            MemoryTemplate::Tree {
                root: "CEO".into(),
                children: HashMap::new(),
                provenance: MemoryProvenance::Manual,
            },
        );

        store.tree_add_child(key, "CEO", "VP_Eng").unwrap();
        store.tree_add_child(key, "CEO", "VP_Eng").unwrap(); // duplicate

        let children = store.tree_children(key, "CEO").unwrap();
        assert_eq!(children.len(), 1, "Duplicate child should be rejected");
    }
}
