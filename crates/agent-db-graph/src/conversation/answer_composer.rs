//! Memory context gathering for conversation query responses.
//!
//! Collects relevant structured memory backing a conversation query result,
//! returning serializable context entries alongside the computed answer.

use super::nlq_ext::ConversationQueryType;
use super::types::NameRegistry;
use crate::structured_memory::{MemoryTemplate, StructuredMemoryStore};
use serde::Serialize;

// ---------------------------------------------------------------------------
// Serializable summary types
// ---------------------------------------------------------------------------

/// Summary of a single ledger entry (lightweight, serializable).
#[derive(Debug, Clone, Serialize)]
pub struct LedgerEntrySummary {
    pub timestamp: u64,
    pub amount: f64,
    pub description: String,
    pub direction: String,
}

/// Summary of a single preference item (lightweight, serializable).
#[derive(Debug, Clone, Serialize)]
pub struct PreferenceItemSummary {
    pub name: String,
    pub rank: usize,
    pub score: Option<f64>,
}

/// Summary of a state transition (lightweight, serializable).
#[derive(Debug, Clone, Serialize)]
pub struct StateTransitionSummary {
    pub from: String,
    pub to: String,
    pub timestamp: u64,
    pub trigger: String,
}

/// Summary of an episodic/semantic/schema memory (lightweight, serializable).
#[derive(Debug, Clone, Serialize)]
pub struct MemorySummary {
    pub id: u64,
    pub summary: String,
    pub takeaway: String,
    pub tier: String,
}

impl MemorySummary {
    /// Create from a `Memory` reference.
    pub fn from_memory(m: &crate::memory::Memory) -> Self {
        Self {
            id: m.id,
            summary: m.summary.clone(),
            takeaway: m.takeaway.clone(),
            tier: format!("{:?}", m.tier),
        }
    }
}

/// Summary of a strategy (lightweight, serializable).
#[derive(Debug, Clone, Serialize)]
pub struct StrategySummary {
    pub id: u64,
    pub name: String,
    pub summary: String,
    pub when_to_use: String,
}

impl StrategySummary {
    /// Create from a `Strategy` reference.
    pub fn from_strategy(s: &crate::strategies::Strategy) -> Self {
        Self {
            id: s.id,
            name: s.name.clone(),
            summary: s.summary.clone(),
            when_to_use: s.when_to_use.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryContextEntry
// ---------------------------------------------------------------------------

/// A piece of structured memory relevant to a conversation query.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum MemoryContextEntry {
    /// Ledger between two entities with entries and balance.
    Ledger {
        key: String,
        entity_a: String,
        entity_b: String,
        balance: f64,
        entries: Vec<LedgerEntrySummary>,
    },
    /// State machine for an entity attribute.
    State {
        key: String,
        entity: String,
        current_value: String,
        history_len: usize,
        recent_transitions: Vec<StateTransitionSummary>,
    },
    /// Preference list for an entity in a category.
    Preference {
        key: String,
        entity: String,
        category: String,
        items: Vec<PreferenceItemSummary>,
    },
    /// Relationship path between entities.
    Relationship {
        relation_type: String,
        path: Option<Vec<String>>,
    },
}

// ---------------------------------------------------------------------------
// Context gathering
// ---------------------------------------------------------------------------

/// Gather structured memory relevant to a conversation query.
///
/// Returns a list of `MemoryContextEntry` values that back the computed answer.
/// This is a sync function — runs under an existing read lock.
pub fn gather_memory_context(
    query: &ConversationQueryType,
    store: &StructuredMemoryStore,
    registry: &NameRegistry,
) -> Vec<MemoryContextEntry> {
    match query {
        ConversationQueryType::Numeric { .. } => gather_numeric(store),
        ConversationQueryType::State { entity, .. } => {
            gather_state(entity.as_deref(), store, registry)
        },
        ConversationQueryType::EntitySummary { entity } => {
            gather_entity_summary(entity, store, registry)
        },
        ConversationQueryType::Preference { entity, category } => {
            gather_preference(entity.as_deref(), category.as_deref(), store, registry)
        },
        ConversationQueryType::RelationshipPath { from, to, relation } => {
            gather_relationship(from, to, relation.as_deref(), store)
        },
    }
}

/// Gather all ledgers for numeric queries.
fn gather_numeric(store: &StructuredMemoryStore) -> Vec<MemoryContextEntry> {
    let mut entries = Vec::new();

    for key in store.list_keys("ledger:") {
        if let Some(MemoryTemplate::Ledger {
            entity_pair,
            entries: ledger_entries,
            balance,
            ..
        }) = store.get(key)
        {
            entries.push(MemoryContextEntry::Ledger {
                key: key.to_string(),
                entity_a: entity_pair.0.clone(),
                entity_b: entity_pair.1.clone(),
                balance: *balance,
                entries: ledger_entries
                    .iter()
                    .map(|e| LedgerEntrySummary {
                        timestamp: e.timestamp,
                        amount: e.amount,
                        description: e.description.clone(),
                        direction: format!("{:?}", e.direction),
                    })
                    .collect(),
            });
        }
    }

    entries
}

/// Gather state machines and related preference data for a target entity.
fn gather_state(
    entity: Option<&str>,
    store: &StructuredMemoryStore,
    registry: &NameRegistry,
) -> Vec<MemoryContextEntry> {
    let entity_name = entity.unwrap_or("user");
    let entity_id = match resolve_entity_id(entity_name, registry) {
        Some(id) => id,
        None => return Vec::new(),
    };

    let mut entries = Vec::new();

    // State machines for the entity
    gather_state_machines(store, entity_id, entity_name, &mut entries);

    // All preference categories for this entity (generic — no hardcoded category names)
    for key in store.list_keys(&format!("prefs:{}:", entity_id)) {
        gather_pref_list(
            store,
            entity_id,
            entity_name,
            key.rsplit(':').next().unwrap_or("general"),
            &mut entries,
        );
    }

    entries
}

/// Gather all data types for an entity summary.
fn gather_entity_summary(
    entity: &str,
    store: &StructuredMemoryStore,
    registry: &NameRegistry,
) -> Vec<MemoryContextEntry> {
    let entity_id = match resolve_entity_id(entity, registry) {
        Some(id) => id,
        None => return Vec::new(),
    };

    let mut entries = Vec::new();

    // 1. State machines
    gather_state_machines(store, entity_id, entity, &mut entries);

    // 2. All preference lists
    for key in store.list_keys(&format!("prefs:{}:", entity_id)) {
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(key) {
            let category = key.rsplit(':').next().unwrap_or("general").to_string();
            entries.push(MemoryContextEntry::Preference {
                key: key.to_string(),
                entity: entity.to_string(),
                category,
                items: ranked_items
                    .iter()
                    .map(|i| PreferenceItemSummary {
                        name: i.name.clone(),
                        rank: i.rank,
                        score: i.score,
                    })
                    .collect(),
            });
        }
    }

    // 3. Ledger involvement
    for key in store.list_keys("ledger:") {
        if let Some(MemoryTemplate::Ledger {
            entity_pair,
            entries: ledger_entries,
            balance,
            ..
        }) = store.get(key)
        {
            if entity_pair.0 == entity || entity_pair.1 == entity {
                entries.push(MemoryContextEntry::Ledger {
                    key: key.to_string(),
                    entity_a: entity_pair.0.clone(),
                    entity_b: entity_pair.1.clone(),
                    balance: *balance,
                    entries: ledger_entries
                        .iter()
                        .map(|e| LedgerEntrySummary {
                            timestamp: e.timestamp,
                            amount: e.amount,
                            description: e.description.clone(),
                            direction: format!("{:?}", e.direction),
                        })
                        .collect(),
                });
            }
        }
    }

    entries
}

/// Gather preference data for a preference query.
fn gather_preference(
    entity: Option<&str>,
    category: Option<&str>,
    store: &StructuredMemoryStore,
    registry: &NameRegistry,
) -> Vec<MemoryContextEntry> {
    let entity_name = entity.unwrap_or("user");
    let entity_id = match resolve_entity_id(entity_name, registry) {
        Some(id) => id,
        None => return Vec::new(),
    };

    let mut entries = Vec::new();

    if let Some(cat) = category {
        // Specific category
        gather_pref_list(store, entity_id, entity_name, cat, &mut entries);
    } else {
        // All preference lists for this entity
        for key in store.list_keys(&format!("prefs:{}:", entity_id)) {
            if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(key) {
                let cat = key.rsplit(':').next().unwrap_or("general").to_string();
                entries.push(MemoryContextEntry::Preference {
                    key: key.to_string(),
                    entity: entity_name.to_string(),
                    category: cat,
                    items: ranked_items
                        .iter()
                        .map(|i| PreferenceItemSummary {
                            name: i.name.clone(),
                            rank: i.rank,
                            score: i.score,
                        })
                        .collect(),
                });
            }
        }
    }

    // Also include facts if they exist
    gather_pref_list(store, entity_id, entity_name, "facts", &mut entries);

    entries
}

/// Gather relationship path data via graph projection.
fn gather_relationship(
    _from: &str,
    _to: &str,
    relation: Option<&str>,
    _store: &StructuredMemoryStore,
) -> Vec<MemoryContextEntry> {
    let rel_type = relation.unwrap_or("colleague");
    // Graph-based path finding is done at query time via graph_projection.
    // This function returns an empty path since it has no graph reference;
    // callers with graph access should use graph_projection directly.
    vec![MemoryContextEntry::Relationship {
        relation_type: rel_type.to_string(),
        path: None,
    }]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve an entity name to its registry ID, with "user" fallback.
fn resolve_entity_id(name: &str, registry: &NameRegistry) -> Option<u64> {
    registry
        .id_for_name(name)
        .or_else(|| registry.id_for_name("user"))
}

/// Gather all state machines for an entity and append to `out`.
fn gather_state_machines(
    store: &StructuredMemoryStore,
    entity_id: u64,
    entity_name: &str,
    out: &mut Vec<MemoryContextEntry>,
) {
    for key in store.list_keys(&format!("state:{}:", entity_id)) {
        if let Some(MemoryTemplate::StateMachine {
            current_state,
            history,
            ..
        }) = store.get(key)
        {
            let recent: Vec<StateTransitionSummary> = history
                .iter()
                .rev()
                .take(5)
                .map(|t| StateTransitionSummary {
                    from: t.from.clone(),
                    to: t.to.clone(),
                    timestamp: t.timestamp,
                    trigger: t.trigger.clone(),
                })
                .collect();

            out.push(MemoryContextEntry::State {
                key: key.to_string(),
                entity: entity_name.to_string(),
                current_value: current_state.clone(),
                history_len: history.len(),
                recent_transitions: recent,
            });
        }
    }
}

/// Gather a single preference list by category and append to `out`.
fn gather_pref_list(
    store: &StructuredMemoryStore,
    entity_id: u64,
    entity_name: &str,
    category: &str,
    out: &mut Vec<MemoryContextEntry>,
) {
    let key = format!("prefs:{}:{}", entity_id, category);
    if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&key) {
        out.push(MemoryContextEntry::Preference {
            key,
            entity: entity_name.to_string(),
            category: category.to_string(),
            items: ranked_items
                .iter()
                .map(|i| PreferenceItemSummary {
                    name: i.name.clone(),
                    rank: i.rank,
                    score: i.score,
                })
                .collect(),
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::nlq_ext::NumericOp;
    use crate::structured_memory::{
        LedgerDirection, LedgerEntry, MemoryProvenance, MemoryTemplate, PreferenceItem,
    };
    fn make_registry() -> NameRegistry {
        let mut r = NameRegistry::new();
        r.get_or_create("Alice"); // id=1
        r.get_or_create("Bob"); // id=2
        r.get_or_create("Charlie"); // id=3
        r
    }

    fn make_ledger_store() -> StructuredMemoryStore {
        let mut store = StructuredMemoryStore::new();

        store.upsert(
            "ledger:1:2",
            MemoryTemplate::Ledger {
                entity_pair: ("Alice".into(), "Bob".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        let _ = store.ledger_append(
            "ledger:1:2",
            LedgerEntry {
                timestamp: 100,
                amount: 50.0,
                description: "dinner".into(),
                direction: LedgerDirection::Credit,
            },
        );
        let _ = store.ledger_append(
            "ledger:1:2",
            LedgerEntry {
                timestamp: 200,
                amount: 20.0,
                description: "taxi".into(),
                direction: LedgerDirection::Debit,
            },
        );

        store.upsert(
            "ledger:2:3",
            MemoryTemplate::Ledger {
                entity_pair: ("Bob".into(), "Charlie".into()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        let _ = store.ledger_append(
            "ledger:2:3",
            LedgerEntry {
                timestamp: 300,
                amount: 60.0,
                description: "concert".into(),
                direction: LedgerDirection::Credit,
            },
        );

        store
    }

    #[test]
    fn test_gather_context_numeric() {
        let store = make_ledger_store();
        let registry = make_registry();
        let query = ConversationQueryType::Numeric {
            op: NumericOp::TransferMinimize,
        };

        let ctx = gather_memory_context(&query, &store, &registry);

        // Should return all ledgers
        let ledger_count = ctx
            .iter()
            .filter(|e| matches!(e, MemoryContextEntry::Ledger { .. }))
            .count();
        assert_eq!(ledger_count, 2);

        // Check first ledger has entries
        if let MemoryContextEntry::Ledger { entries, .. } = &ctx[0] {
            assert!(!entries.is_empty());
        } else {
            panic!("Expected Ledger entry");
        }
    }

    #[test]
    fn test_gather_context_state() {
        let mut store = StructuredMemoryStore::new();
        let registry = make_registry();

        // Add state machine for Alice (id=1)
        store.upsert(
            "state:1:location",
            MemoryTemplate::StateMachine {
                entity: "Alice".into(),
                current_state: "Paris".into(),
                history: vec![],
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        let _ = store.state_transition("state:1:location", "London", "traveled", 100);

        // Add facts for Alice
        store.upsert(
            "prefs:1:facts",
            MemoryTemplate::PreferenceList {
                entity: "Alice".into(),
                ranked_items: vec![PreferenceItem {
                    name: "Runs every morning".into(),
                    rank: 1,
                    score: None,
                }],
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );

        let query = ConversationQueryType::State {
            entity: Some("Alice".into()),
            attribute: None,
        };
        let ctx = gather_memory_context(&query, &store, &registry);

        let state_count = ctx
            .iter()
            .filter(|e| matches!(e, MemoryContextEntry::State { .. }))
            .count();
        let pref_count = ctx
            .iter()
            .filter(|e| matches!(e, MemoryContextEntry::Preference { .. }))
            .count();

        assert_eq!(state_count, 1, "Should have 1 state machine");
        assert!(pref_count >= 1, "Should have at least facts preference");

        // Verify state has correct current value
        if let MemoryContextEntry::State {
            current_value,
            history_len,
            ..
        } = &ctx[0]
        {
            assert_eq!(current_value, "London");
            assert_eq!(*history_len, 1);
        }
    }

    #[test]
    fn test_gather_context_entity_summary() {
        let mut store = make_ledger_store();
        let registry = make_registry();

        // Add state machine for Alice
        store.upsert(
            "state:1:location",
            MemoryTemplate::StateMachine {
                entity: "Alice".into(),
                current_state: "Berlin".into(),
                history: vec![],
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );

        // Add preferences for Alice
        store.upsert(
            "prefs:1:food",
            MemoryTemplate::PreferenceList {
                entity: "Alice".into(),
                ranked_items: vec![PreferenceItem {
                    name: "Pizza".into(),
                    rank: 1,
                    score: Some(9.0),
                }],
                provenance: MemoryProvenance::Manual,
            },
        );

        let query = ConversationQueryType::EntitySummary {
            entity: "Alice".into(),
        };
        let ctx = gather_memory_context(&query, &store, &registry);

        // Should have state + preference + ledger(s) involving Alice
        let has_state = ctx
            .iter()
            .any(|e| matches!(e, MemoryContextEntry::State { .. }));
        let has_pref = ctx
            .iter()
            .any(|e| matches!(e, MemoryContextEntry::Preference { .. }));
        let has_ledger = ctx
            .iter()
            .any(|e| matches!(e, MemoryContextEntry::Ledger { .. }));

        assert!(has_state, "Should include state machines");
        assert!(has_pref, "Should include preferences");
        assert!(has_ledger, "Should include ledgers involving Alice");
    }

    #[test]
    fn test_gather_context_preference() {
        let mut store = StructuredMemoryStore::new();
        let registry = make_registry();

        store.upsert(
            "prefs:1:music",
            MemoryTemplate::PreferenceList {
                entity: "Alice".into(),
                ranked_items: vec![
                    PreferenceItem {
                        name: "Jazz".into(),
                        rank: 1,
                        score: Some(9.0),
                    },
                    PreferenceItem {
                        name: "Rock".into(),
                        rank: 2,
                        score: Some(7.0),
                    },
                ],
                provenance: MemoryProvenance::Manual,
            },
        );

        let query = ConversationQueryType::Preference {
            entity: Some("Alice".into()),
            category: Some("music".into()),
        };
        let ctx = gather_memory_context(&query, &store, &registry);

        let pref_count = ctx
            .iter()
            .filter(|e| matches!(e, MemoryContextEntry::Preference { .. }))
            .count();
        assert!(
            pref_count >= 1,
            "Should have at least the music preference list"
        );

        // Verify items
        let music = ctx.iter().find(
            |e| matches!(e, MemoryContextEntry::Preference { category, .. } if category == "music"),
        );
        assert!(music.is_some());
        if let Some(MemoryContextEntry::Preference { items, .. }) = music {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].name, "Jazz");
        }
    }

    #[test]
    fn test_gather_context_relationship() {
        let store = StructuredMemoryStore::new();
        let _registry = make_registry();

        // Relationship path finding is now a graph projection (no stored tree).
        // gather_relationship returns path: None without a graph reference;
        // actual path resolution happens at query time via graph_projection.
        let query = ConversationQueryType::RelationshipPath {
            from: "Alice".into(),
            to: "Bob".into(),
            relation: Some("colleague".into()),
        };
        let ctx = gather_memory_context(&query, &store, &_registry);

        assert_eq!(ctx.len(), 1);
        if let MemoryContextEntry::Relationship {
            relation_type,
            path,
        } = &ctx[0]
        {
            assert_eq!(relation_type, "colleague");
            // Path is None since graph projection needs a Graph reference
            assert!(path.is_none());
        } else {
            panic!("Expected Relationship entry");
        }
    }

    #[test]
    fn test_gather_context_empty_store() {
        let store = StructuredMemoryStore::new();
        let registry = NameRegistry::new();

        let query = ConversationQueryType::Numeric {
            op: NumericOp::NetBalance,
        };
        let ctx = gather_memory_context(&query, &store, &registry);
        assert!(ctx.is_empty());

        let query = ConversationQueryType::State {
            entity: Some("Nobody".into()),
            attribute: None,
        };
        let ctx = gather_memory_context(&query, &store, &registry);
        assert!(ctx.is_empty());
    }
}
