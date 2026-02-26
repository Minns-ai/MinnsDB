//! NLQ extensions for conversation-aware queries.
//!
//! Adds new query types: NumericQuery, StateQuery, EntitySummaryQuery,
//! PreferenceQuery, and RelationshipPath. These operate on structured
//! memory populated by the conversation bridge.

use super::numeric_reasoning;
use super::types::NameRegistry;
use crate::structured_memory::{MemoryTemplate, StructuredMemoryStore};

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
// Intent classification for conversation queries
// ---------------------------------------------------------------------------

/// Try to classify a question as a conversation-specific query.
///
/// Returns `None` if the question doesn't match any conversation-specific pattern.
pub fn classify_conversation_query(question: &str) -> Option<ConversationQueryType> {
    let lower = question.to_lowercase();

    // Numeric queries
    if is_numeric_query(&lower) {
        let op = if lower.contains("settle")
            || lower.contains("minimum transfer")
            || lower.contains("simplif")
            || lower.contains("how to pay")
        {
            NumericOp::TransferMinimize
        } else if lower.contains("total") || lower.contains("sum") || lower.contains("how much") {
            NumericOp::Sum
        } else {
            NumericOp::NetBalance
        };
        return Some(ConversationQueryType::Numeric { op });
    }

    // Relationship path queries
    if is_relationship_query(&lower) {
        let (from, to) = extract_pair_names(question);
        if !from.is_empty() && !to.is_empty() {
            return Some(ConversationQueryType::RelationshipPath {
                from,
                to,
                relation: Some("colleague".to_string()),
            });
        }
    }

    // State queries
    if is_state_query(&lower) {
        let entity = extract_entity_from_question(question);
        return Some(ConversationQueryType::State {
            entity,
            attribute: None,
        });
    }

    // Entity summary
    if is_entity_summary_query(&lower) {
        let entity = extract_entity_from_question(question).unwrap_or_else(|| "user".to_string());
        return Some(ConversationQueryType::EntitySummary { entity });
    }

    // Preference queries
    if is_preference_query(&lower) {
        let entity = extract_entity_from_question(question);
        let category = extract_category_from_question(&lower);
        return Some(ConversationQueryType::Preference { entity, category });
    }

    tracing::debug!(
        question,
        "classify_conversation_query: no conversation-specific pattern matched"
    );
    None
}

fn is_numeric_query(lower: &str) -> bool {
    let patterns = [
        "owes",
        "owe",
        "balance",
        "settle",
        "total spent",
        "how much did",
        "how much does",
        "how much was",
        "minimum transfer",
        "simplif",
        "debt",
        "who pays",
        "who needs to pay",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

fn is_relationship_query(lower: &str) -> bool {
    let patterns = [
        "related to each other",
        "related through",
        "connected through",
        "colleagues relation",
        "path between",
        "are .* and .* related",
        "know each other",
    ];
    patterns.iter().any(|p| {
        if p.contains(".*") {
            // Simple glob: just check both parts exist
            let parts: Vec<&str> = p.split(".*").collect();
            parts.iter().all(|part| lower.contains(part))
        } else {
            lower.contains(p)
        }
    })
}

fn is_state_query(lower: &str) -> bool {
    let patterns = [
        "where is",
        "where do",
        "where does",
        "where am i",
        "current status",
        "current state",
        "state of",
        "location of",
        "what should i do",
        "what do you have every",
        "what do i have every",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

fn is_entity_summary_query(lower: &str) -> bool {
    let patterns = [
        "who is",
        "tell me about",
        "what do you know about",
        "describe",
        "summary of",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

fn is_preference_query(lower: &str) -> bool {
    let patterns = [
        "recommend",
        "suggest",
        "favorite",
        "favourite",
        "what do i like",
        "what does",
        "do i like",
        "preference",
        "which .* do i like",
        "rank",
        "rating",
    ];
    patterns.iter().any(|p| {
        if p.contains(".*") {
            let parts: Vec<&str> = p.split(".*").collect();
            parts.iter().all(|part| lower.contains(part))
        } else {
            lower.contains(p)
        }
    })
}

// ---------------------------------------------------------------------------
// Name extraction from questions
// ---------------------------------------------------------------------------

/// Extract a pair of names from a relationship question.
fn extract_pair_names(question: &str) -> (String, String) {
    let lower = question.to_lowercase();

    // Pattern: "Are X and Y related..."
    if lower.starts_with("are ") {
        if let Some(and_pos) = lower.find(" and ") {
            let from_str = &question[4..and_pos].trim();
            // Find end of second name (before "related", "connected", etc.)
            let after_and = &question[and_pos + 5..];
            let end_markers = [" related", " connected", " colleagues", " at work"];
            let mut end = after_and.len();
            for marker in &end_markers {
                if let Some(p) = after_and.to_lowercase().find(marker) {
                    end = end.min(p);
                }
            }
            let to_str = after_and[..end].trim();
            return (from_str.to_string(), to_str.to_string());
        }
    }

    // Pattern: "path between X and Y"
    if let Some(between_pos) = lower.find("between ") {
        let after = &question[between_pos + 8..];
        if let Some(and_pos) = after.to_lowercase().find(" and ") {
            let from_str = after[..and_pos].trim();
            let to_str = after[and_pos + 5..]
                .trim()
                .trim_end_matches('?')
                .trim_end_matches('.');
            return (from_str.to_string(), to_str.to_string());
        }
    }

    (String::new(), String::new())
}

/// Extract a single entity name from a question.
fn extract_entity_from_question(question: &str) -> Option<String> {
    let lower = question.to_lowercase();

    // "who is X" / "tell me about X" / "where is X"
    let prefixes = [
        "who is ",
        "tell me about ",
        "what do you know about ",
        "where is ",
        "where does ",
        "describe ",
        "summary of ",
    ];

    for prefix in &prefixes {
        if let Some(pos) = lower.find(prefix) {
            let after = &question[pos + prefix.len()..];
            let name = after
                .trim()
                .trim_end_matches('?')
                .trim_end_matches('.')
                .trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    None
}

/// Extract a category from a preference question.
fn extract_category_from_question(lower: &str) -> Option<String> {
    let categories = [
        "art",
        "music",
        "movie",
        "film",
        "book",
        "food",
        "sport",
        "series",
        "game",
        "boardgame",
    ];
    for cat in &categories {
        if lower.contains(cat) {
            return Some(cat.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

/// Execute a conversation-specific query against the structured memory store.
pub fn execute_conversation_query(
    query: &ConversationQueryType,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
    question: &str,
) -> String {
    match query {
        ConversationQueryType::Numeric { op } => execute_numeric(op, store),
        ConversationQueryType::State { entity, attribute } => execute_state(
            entity.as_deref(),
            attribute.as_deref(),
            store,
            name_registry,
            question,
        ),
        ConversationQueryType::EntitySummary { entity } => {
            execute_entity_summary(entity, store, name_registry)
        },
        ConversationQueryType::Preference { entity, category } => {
            execute_preference(entity.as_deref(), category.as_deref(), store, name_registry)
        },
        ConversationQueryType::RelationshipPath { from, to, relation } => {
            let rel = relation.as_deref().unwrap_or("colleague");
            execute_relationship_path(from, to, rel, store)
        },
    }
}

fn execute_numeric(op: &NumericOp, store: &StructuredMemoryStore) -> String {
    let balances = numeric_reasoning::compute_net_balances(store);

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
) -> String {
    let entity_name = entity.unwrap_or("user");
    let entity_id = match name_registry.id_for_name(entity_name) {
        Some(id) => id,
        None => {
            // Try "user" as fallback
            match name_registry.id_for_name("user") {
                Some(id) => id,
                None => {
                    tracing::debug!(
                        entity = entity_name,
                        "No state information found for entity"
                    );
                    return format!("No state information found for '{}'.", entity_name);
                },
            }
        },
    };

    let lower_q = question.to_lowercase();

    // Extract temporal cues from question
    let temporal_cues = extract_temporal_cues(&lower_q);

    // If question targets a specific routine/temporal slot, try to answer directly
    if !temporal_cues.is_empty() {
        let facts_key = format!("prefs:{}:facts", entity_id);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
            for cue in &temporal_cues {
                // Look for matching routine
                for item in ranked_items {
                    let item_lower = item.name.to_lowercase();
                    if item_lower.contains(cue) {
                        return item.name.clone();
                    }
                }
            }
        }
    }

    // If question asks about a specific activity at a place, check activities
    if lower_q.contains("bakery") || lower_q.contains("cafe") || lower_q.contains("restaurant") {
        let facts_key = format!("prefs:{}:facts", entity_id);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
            for item in ranked_items {
                let item_lower = item.name.to_lowercase();
                // Match place name in question against stored facts
                if lower_q.contains("bakery") && item_lower.contains("bakery") {
                    return item.name.clone();
                }
                if lower_q.contains("cafe") && item_lower.contains("cafe") {
                    return item.name.clone();
                }
            }
        }
    }

    // If question asks "what should I do" with temporal hint, compose answer
    if lower_q.contains("what should i do") || lower_q.contains("what do") {
        let mut suggestions = Vec::new();

        // Check routines matching temporal cues
        let facts_key = format!("prefs:{}:facts", entity_id);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
            for item in ranked_items {
                let item_lower = item.name.to_lowercase();
                let matches_cue = temporal_cues.iter().any(|c| item_lower.contains(c))
                    || temporal_cues.is_empty();
                if matches_cue {
                    suggestions.push(item.name.clone());
                }
            }
        }

        // Add current location context
        if let Some(loc) = numeric_reasoning::state_current(store, entity_id, "location") {
            if suggestions.is_empty() {
                suggestions.push(format!("You are currently in {}.", loc));
            }
        }

        // Add nearby landmarks
        let landmarks_key = format!("prefs:{}:landmarks", entity_id);
        if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&landmarks_key)
        {
            for item in ranked_items.iter().take(2) {
                suggestions.push(format!("Near: {}", item.name));
            }
        }

        if !suggestions.is_empty() {
            return suggestions.join("\n");
        }
    }

    // Default: dump all state info
    let mut parts = Vec::new();

    // Current location
    if let Some(loc) = numeric_reasoning::state_current(store, entity_id, "location") {
        parts.push(format!("Current location: {}", loc));
    }

    // Current status
    if let Some(status) = numeric_reasoning::state_current(store, entity_id, "status") {
        parts.push(format!("Current status: {}", status));
    }

    // Facts (routines, activities)
    let facts_key = format!("prefs:{}:facts", entity_id);
    if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
        for item in ranked_items.iter().take(5) {
            parts.push(item.name.clone());
        }
    }

    // Landmarks
    let landmarks_key = format!("prefs:{}:landmarks", entity_id);
    if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&landmarks_key) {
        for item in ranked_items.iter().take(3) {
            parts.push(format!("Near: {}", item.name));
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

fn execute_entity_summary(
    entity: &str,
    store: &StructuredMemoryStore,
    name_registry: &NameRegistry,
) -> String {
    let entity_id = match name_registry.id_for_name(entity) {
        Some(id) => id,
        None => match name_registry.id_for_name("user") {
            Some(id) => id,
            None => {
                tracing::debug!(entity, "No information found for entity");
                return format!("No information found for '{}'.", entity);
            },
        },
    };

    let mut sections = Vec::new();

    // 1. Current states (max 2)
    for attr in &["location", "status"] {
        if let Some(val) = numeric_reasoning::state_current(store, entity_id, attr) {
            sections.push(format!("Current {}: {}", attr, val));
        }
    }

    // 2. Facts/routines (max 3)
    let facts_key = format!("prefs:{}:facts", entity_id);
    if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&facts_key) {
        for item in ranked_items.iter().take(3) {
            sections.push(item.name.clone());
        }
    }

    // 3. Landmarks (max 3)
    let landmarks_key = format!("prefs:{}:landmarks", entity_id);
    if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&landmarks_key) {
        for item in ranked_items.iter().take(3) {
            sections.push(format!("Near: {}", item.name));
        }
    }

    // 4. Preferences (max 3 per category, max 2 categories)
    let pref_keys = store.list_keys(&format!("prefs:{}:", entity_id));
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
) -> String {
    let entity_name = entity.unwrap_or("user");
    let entity_id = match name_registry.id_for_name(entity_name) {
        Some(id) => id,
        None => match name_registry.id_for_name("user") {
            Some(id) => id,
            None => return "No preferences found.".to_string(),
        },
    };

    if let Some(cat) = category {
        let prefs = numeric_reasoning::rank_preferences(store, entity_id, cat);
        if prefs.is_empty() {
            return format!("No {} preferences found.", cat);
        }
        let items: Vec<String> = prefs
            .iter()
            .map(|(name, score)| format!("{} ({:.1})", name, score))
            .collect();
        format!("{} preferences: {}", cat, items.join(", "))
    } else {
        // All preferences
        let pref_keys = store.list_keys(&format!("prefs:{}:", entity_id));
        let mut all = Vec::new();
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
    store: &StructuredMemoryStore,
) -> String {
    match numeric_reasoning::find_relationship_path(store, from, to, relation_type) {
        Some(path) => {
            let chain = numeric_reasoning::format_path(&path, relation_type);
            format!(
                "Yes, {} and {} are connected through {} relations.\nPath: {}",
                from, to, relation_type, chain
            )
        },
        None => format!(
            "No {} connection found between {} and {}.",
            relation_type, from, to
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_owes_query() {
        let q = classify_conversation_query("Who owes whom?");
        assert!(q.is_some());
        assert!(matches!(
            q.unwrap(),
            ConversationQueryType::Numeric {
                op: NumericOp::NetBalance
            }
        ));
    }

    #[test]
    fn classify_settle_query() {
        let q = classify_conversation_query("How to settle the debts?");
        assert!(q.is_some());
        assert!(matches!(
            q.unwrap(),
            ConversationQueryType::Numeric {
                op: NumericOp::TransferMinimize
            }
        ));
    }

    #[test]
    fn classify_relationship_query() {
        let q = classify_conversation_query(
            "Are Brenda Nguyen and Johnny Fisher related to each other at work through colleagues relations?",
        );
        assert!(q.is_some());
        if let Some(ConversationQueryType::RelationshipPath { from, to, .. }) = q {
            assert_eq!(from, "Brenda Nguyen");
            assert_eq!(to, "Johnny Fisher");
        } else {
            panic!("Expected RelationshipPath");
        }
    }

    #[test]
    fn classify_state_query() {
        let q = classify_conversation_query("What should I do this Saturday morning?");
        assert!(q.is_some());
        assert!(matches!(q.unwrap(), ConversationQueryType::State { .. }));
    }

    #[test]
    fn classify_preference_query() {
        let q = classify_conversation_query("What art do I like?");
        assert!(q.is_some());
        assert!(matches!(
            q.unwrap(),
            ConversationQueryType::Preference { .. }
        ));
    }

    #[test]
    fn extract_pair_names_from_question() {
        let (from, to) =
            extract_pair_names("Are Brenda Nguyen and Johnny Fisher related to each other?");
        assert_eq!(from, "Brenda Nguyen");
        assert_eq!(to, "Johnny Fisher");
    }
}
