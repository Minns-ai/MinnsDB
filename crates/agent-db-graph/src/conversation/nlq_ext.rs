//! NLQ extensions for conversation-aware queries.
//!
//! Adds new query types: NumericQuery, StateQuery, EntitySummaryQuery,
//! PreferenceQuery, and RelationshipPath. These operate on structured
//! memory populated by the conversation bridge.

use super::graph_projection;
use super::numeric_reasoning;
use super::types::NameRegistry;
use crate::structured_memory::{MemoryTemplate, StructuredMemoryStore};
use crate::structures::Graph;

// ---------------------------------------------------------------------------
// Query intent types (conversation-specific)
// ---------------------------------------------------------------------------

/// Numeric operation type for ledger queries.
#[derive(Debug, Clone)]
pub enum NumericOp {
    /// "who owes whom", "calculate balance"
    NetBalance,
    /// "total spent", "how much did X pay"
    Sum,
    /// "minimum transfers", "simplify debts", "settle"
    TransferMinimize,
}

/// Conversation-aware query type.
#[derive(Debug, Clone)]
pub enum ConversationQueryType {
    /// Numeric reasoning over ledgers
    Numeric { op: NumericOp },
    /// "where is X", "current state of X"
    State {
        entity: Option<String>,
        attribute: Option<String>,
    },
    /// "who is X", "tell me about X"
    EntitySummary { entity: String },
    /// "what does X like", "recommend"
    Preference {
        entity: Option<String>,
        category: Option<String>,
    },
    /// "are X and Y related through colleagues"
    RelationshipPath {
        from: String,
        to: String,
        relation: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

/// Execute a conversation-specific query against the structured memory store,
/// with optional graph projection (tries graph first, falls back to store).
pub fn execute_conversation_query(
    query: &ConversationQueryType,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
    question: &str,
) -> String {
    execute_conversation_query_with_graph(query, store, name_registry, question, None)
}

/// Execute a conversation-specific query, trying graph projections first when
/// a graph reference is provided.
pub fn execute_conversation_query_with_graph(
    query: &ConversationQueryType,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
    question: &str,
    graph: Option<&Graph>,
) -> String {
    match query {
        ConversationQueryType::Numeric { op } => execute_numeric(op, store, graph),
        ConversationQueryType::State { entity, attribute } => execute_state(
            entity.as_deref(),
            attribute.as_deref(),
            store,
            name_registry,
            question,
            graph,
        ),
        ConversationQueryType::EntitySummary { entity } => {
            execute_entity_summary(entity, store, name_registry, graph)
        },
        ConversationQueryType::Preference { entity, category } => execute_preference(
            entity.as_deref(),
            category.as_deref(),
            store,
            name_registry,
            graph,
        ),
        ConversationQueryType::RelationshipPath { from, to, relation } => {
            let rel = relation.as_deref().unwrap_or("colleague");
            execute_relationship_path(from, to, rel, store, graph)
        },
    }
}

fn execute_numeric(op: &NumericOp, store: &StructuredMemoryStore, graph: Option<&Graph>) -> String {
    // Try graph projection first, fall back to store
    let balances = if let Some(g) = graph {
        let gb = graph_projection::compute_net_balances_from_graph(g);
        if gb.is_empty() {
            numeric_reasoning::compute_net_balances(store)
        } else {
            gb
        }
    } else {
        numeric_reasoning::compute_net_balances(store)
    };

    match op {
        NumericOp::NetBalance => {
            if balances.is_empty() {
                return "No financial transactions recorded.".to_string();
            }
            numeric_reasoning::format_balances(&balances)
        },
        NumericOp::Sum => {
            let total: f64 = store
                .list_keys("ledger:")
                .iter()
                .filter_map(|key| numeric_reasoning::ledger_sum(store, key))
                .sum();
            format!("Total across all ledgers: {:.2}", total)
        },
        NumericOp::TransferMinimize => {
            if balances.is_empty() {
                return "No debts to settle.".to_string();
            }
            let transfers = numeric_reasoning::minimize_transfers(&balances, "EUR");
            numeric_reasoning::format_transfers(&transfers)
        },
    }
}

fn execute_state(
    entity: Option<&str>,
    _attribute: Option<&str>,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
    question: &str,
    graph: Option<&Graph>,
) -> String {
    let entity_name = entity.unwrap_or("user");
    // Try registry lookup; if missing, use None so graph projection can still work.
    let entity_id = name_registry
        .id_for_name(entity_name)
        .or_else(|| name_registry.id_for_name("user"));

    let lower_q = question.to_lowercase();

    // Extract temporal cues and question keywords for relevance matching
    let temporal_cues = extract_temporal_cues(&lower_q);
    let question_words: Vec<&str> = lower_q
        .split_whitespace()
        .filter(|w| w.len() > 3 && !is_stop_word(w))
        .collect();

    // Get current location: graph projection is the single source of truth.
    // Falls back to store only if no graph is provided and registry has the entity.
    let current_location = graph
        .and_then(|g| graph_projection::state_current_from_graph(g, entity_name, "location"))
        .or_else(|| {
            // Fallback: try store only when graph projection returned nothing
            entity_id.and_then(|eid| numeric_reasoning::state_current(store, eid, "location"))
        });
    let current_loc_lower = current_location.as_ref().map(|s| s.to_lowercase());

    // Collect facts from graph via successor-state projection
    let graph_entity_facts: Vec<(String, f32)> = if let Some(g) = graph {
        let projected = graph_projection::project_entity_state(g, entity_name, u64::MAX, None);
        if !projected.repair_hints.is_empty() {
            tracing::warn!(
                "project_entity_state: {} repair hints for '{}'",
                projected.repair_hints.len(),
                entity_name
            );
        }
        projected
            .slots
            .values()
            .map(|slot| {
                let fact_text = format!("{}: {}", slot.association_type, slot.target_name);
                let item_lower = fact_text.to_lowercase();
                let mut score: f32 = 1.0; // Base score for graph facts

                // Boost for temporal cue match
                if temporal_cues.iter().any(|c| item_lower.contains(c)) {
                    score += 2.0;
                }
                for w in &question_words {
                    if item_lower.contains(w) {
                        score += 1.0;
                    }
                }
                if let Some(ref loc) = current_loc_lower {
                    if loc.split(',').any(|part| {
                        let p = part.trim();
                        !p.is_empty() && item_lower.contains(p)
                    }) {
                        score += 3.0;
                    }
                }
                (fact_text, score)
            })
            .collect()
    } else {
        vec![]
    };

    // Also collect from store (supplementary, only if registry has the entity)
    let store_facts: Vec<(String, f32)> = if let Some(eid) = entity_id {
        let facts_key = format!("prefs:{}:facts", eid);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
            ranked_items
                .iter()
                .map(|item| {
                    let item_lower = item.name.to_lowercase();
                    let mut score: f32 = 0.0;

                    if temporal_cues.iter().any(|c| item_lower.contains(c)) {
                        score += 2.0;
                    }
                    for w in &question_words {
                        if item_lower.contains(w) {
                            score += 1.0;
                        }
                    }
                    if let Some(ref loc) = current_loc_lower {
                        if loc.split(',').any(|part| {
                            let p = part.trim();
                            !p.is_empty() && item_lower.contains(p)
                        }) {
                            score += 3.0;
                        }
                    }

                    (item.name.clone(), score)
                })
                .collect()
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    // Merge graph facts (priority) with store facts
    let mut scored_facts = graph_entity_facts;
    scored_facts.extend(store_facts);

    // Sort by relevance score descending
    let mut sorted_facts = scored_facts;
    sorted_facts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build response: location context + top relevant facts + landmarks
    let mut parts = Vec::new();

    // Current location
    if let Some(ref loc) = current_location {
        parts.push(format!("Current location: {}", loc));
    }

    // Current status: graph projection is the single source of truth.
    let current_status = graph
        .and_then(|g| graph_projection::state_current_from_graph(g, entity_name, "status"))
        .or_else(|| entity_id.and_then(|eid| numeric_reasoning::state_current(store, eid, "status")));
    if let Some(status) = current_status {
        parts.push(format!("Current status: {}", status));
    }

    // Top relevant facts (prefer scored items over raw dump)
    let top_facts: Vec<_> = sorted_facts
        .iter()
        .filter(|(_, s)| *s > 0.0)
        .take(5)
        .collect();
    if !top_facts.is_empty() {
        for (fact, _) in &top_facts {
            parts.push(fact.clone());
        }
    } else {
        // No scored matches — show all facts (up to 5)
        for (fact, _) in sorted_facts.iter().take(5) {
            parts.push(fact.clone());
        }
    }

    // Landmarks (only if registry has the entity)
    if let Some(eid) = entity_id {
        let landmarks_key = format!("prefs:{}:landmarks", eid);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) =
            store.get(&landmarks_key)
        {
            for item in ranked_items.iter().take(3) {
                parts.push(format!("Near: {}", item.name));
            }
        }
    }

    if parts.is_empty() {
        format!("No state information found for '{}'.", entity_name)
    } else {
        parts.join("\n")
    }
}

/// Extract temporal cues from a question.
fn extract_temporal_cues(lower: &str) -> Vec<String> {
    let mut cues = Vec::new();

    let temporal_patterns = [
        ("every morning", "every morning"),
        ("every evening", "every evening"),
        ("every day", "every day"),
        ("every week", "every week"),
        ("saturday morning", "saturday"),
        ("saturday", "saturday"),
        ("sunday morning", "sunday"),
        ("sunday", "sunday"),
        ("morning", "morning"),
        ("evening", "evening"),
    ];

    for (pattern, cue) in &temporal_patterns {
        if lower.contains(pattern) {
            cues.push(cue.to_string());
        }
    }

    cues
}

fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "what"
            | "where"
            | "when"
            | "which"
            | "does"
            | "should"
            | "have"
            | "this"
            | "that"
            | "your"
            | "based"
            | "live"
            | "usually"
            | "currently"
            | "from"
            | "with"
            | "about"
            | "tell"
            | "like"
            | "know"
    )
}

fn execute_entity_summary(
    entity: &str,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
    graph: Option<&Graph>,
) -> String {
    // Try registry lookup; if missing, use None so graph projection can still work.
    let entity_id = name_registry
        .id_for_name(entity)
        .or_else(|| name_registry.id_for_name("user"));

    let mut sections = Vec::new();

    // 1. Current states (max 2) — try graph first, fall back to store
    for attr in &["location", "status"] {
        let val = graph
            .and_then(|g| graph_projection::state_current_from_graph(g, entity, attr))
            .or_else(|| {
                entity_id.and_then(|eid| numeric_reasoning::state_current(store, eid, attr))
            });
        if let Some(v) = val {
            sections.push(format!("Current {}: {}", attr, v));
        }
    }

    // Include graph-projected entity facts via successor-state projection
    if let Some(g) = graph {
        let projected = graph_projection::project_entity_state(g, entity, u64::MAX, None);
        if !projected.repair_hints.is_empty() {
            tracing::warn!(
                "project_entity_state: {} repair hints for '{}'",
                projected.repair_hints.len(),
                entity
            );
        }
        for slot in projected.slots.values().take(3) {
            if !slot.association_type.starts_with("state:") {
                sections.push(format!("{}: {}", slot.association_type, slot.target_name));
            }
        }
    }

    // Store-based supplementary data (only if registry has the entity)
    if let Some(eid) = entity_id {
        // 2. Facts/routines (max 3)
        let facts_key = format!("prefs:{}:facts", eid);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
            for item in ranked_items.iter().take(3) {
                sections.push(item.name.clone());
            }
        }

        // 3. Landmarks (max 3)
        let landmarks_key = format!("prefs:{}:landmarks", eid);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) =
            store.get(&landmarks_key)
        {
            for item in ranked_items.iter().take(3) {
                sections.push(format!("Near: {}", item.name));
            }
        }

        // 4. Preferences (max 3 per category, max 2 categories)
        let pref_keys = store.list_keys(&format!("prefs:{}:", eid));
        let mut cat_count = 0;
        for key in pref_keys {
            if key.ends_with(":facts") || key.ends_with(":landmarks") {
                continue;
            }
            if cat_count >= 2 {
                break;
            }
            if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(key) {
                let items: Vec<String> = ranked_items
                    .iter()
                    .take(3)
                    .map(|i| i.name.clone())
                    .collect();
                if !items.is_empty() {
                    let category = key.rsplit(':').next().unwrap_or("items");
                    sections.push(format!("Likes ({}): {}", category, items.join(", ")));
                    cat_count += 1;
                }
            }
        }
    }

    if sections.is_empty() {
        format!("No information found for '{}'.", entity)
    } else {
        sections.join("\n")
    }
}

fn execute_preference(
    entity: Option<&str>,
    category: Option<&str>,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
    graph: Option<&Graph>,
) -> String {
    let entity_name = entity.unwrap_or("user");
    // Try registry lookup; if missing, use None so graph projection can still work.
    let entity_id = name_registry
        .id_for_name(entity_name)
        .or_else(|| name_registry.id_for_name("user"));

    if let Some(cat) = category {
        // Try graph projection first, using PPR for larger graphs
        let prefs = if let Some(g) = graph {
            let gp = graph_projection::rank_preferences_with_ppr(g, entity_name, cat);
            if gp.is_empty() {
                // Fallback to store only if registry has the entity
                entity_id
                    .map(|eid| numeric_reasoning::rank_preferences(store, eid, cat))
                    .unwrap_or_default()
            } else {
                gp
            }
        } else {
            entity_id
                .map(|eid| numeric_reasoning::rank_preferences(store, eid, cat))
                .unwrap_or_default()
        };
        if prefs.is_empty() {
            return format!("No {} preferences found.", cat);
        }
        let items: Vec<String> = prefs
            .iter()
            .map(|(name, score)| format!("{} ({:.1})", name, score))
            .collect();
        format!("{} preferences: {}", cat, items.join(", "))
    } else {
        // All preferences (store-based, only if registry has the entity)
        let mut all = Vec::new();
        if let Some(eid) = entity_id {
            let pref_keys = store.list_keys(&format!("prefs:{}:", eid));
            for key in pref_keys {
                if key.ends_with(":facts") || key.ends_with(":landmarks") {
                    continue;
                }
                if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(key) {
                    let cat = key.rsplit(':').next().unwrap_or("general");
                    let items: Vec<String> = ranked_items.iter().map(|i| i.name.clone()).collect();
                    if !items.is_empty() {
                        all.push(format!("{}: {}", cat, items.join(", ")));
                    }
                }
            }
        }
        if all.is_empty() {
            "No preferences found.".to_string()
        } else {
            all.join("\n")
        }
    }
}

fn execute_relationship_path(
    from: &str,
    to: &str,
    relation_type: &str,
    _store: &StructuredMemoryStore,
    graph: Option<&Graph>,
) -> String {
    if let Some(g) = graph {
        // Try exact relation type first
        if let Some(path) =
            graph_projection::find_relationship_path_from_graph(g, from, to, relation_type)
        {
            let chain = numeric_reasoning::format_path(&path, relation_type);
            return format!(
                "Yes, {} and {} are connected through {} relations.\nPath: {}",
                from, to, relation_type, chain
            );
        }
        // Fallback: try any relationship-like edge
        if let Some(path) = graph_projection::find_any_relationship_path_from_graph(g, from, to) {
            let chain = numeric_reasoning::format_path(&path, relation_type);
            return format!("Yes, {} and {} are connected.\nPath: {}", from, to, chain);
        }
    }

    format!(
        "No {} connection found between {} and {}.",
        relation_type, from, to
    )
}
