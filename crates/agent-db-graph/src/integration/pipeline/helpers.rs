// crates/agent-db-graph/src/integration/pipeline/helpers.rs
//
// Entity resolution helpers: metadata extraction, name normalization,
// fuzzy matching, and concept node creation.

use super::*;
use crate::structures::ConceptType;
use std::collections::HashSet;

/// Extract a string value from event metadata by key.
pub(super) fn metadata_str(
    metadata: &std::collections::HashMap<String, agent_db_events::core::MetadataValue>,
    key: &str,
) -> Option<String> {
    metadata.get(key).and_then(|v| match v {
        agent_db_events::core::MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    })
}

// ────────── Entity Resolution Helpers ──────────

/// Normalize an entity name for consistent matching.
///
/// Lowercases, trims, strips leading articles, replaces underscores with spaces.
pub(crate) fn normalize_entity_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let trimmed = lower.trim();
    // Strip leading articles
    let stripped = trimmed
        .strip_prefix("the ")
        .or_else(|| trimmed.strip_prefix("a "))
        .or_else(|| trimmed.strip_prefix("an "))
        .unwrap_or(trimmed);
    // Replace underscores with spaces
    stripped.replace('_', " ").trim().to_string()
}

/// Fast fuzzy entity match: checks if two normalized entity names refer to the same entity.
///
/// Returns true if:
/// - Exact match
/// - One is a substring of the other
/// - They share >80% of words
fn fuzzy_entity_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.contains(b) || b.contains(a) {
        return true;
    }
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();
    let intersection = words_a.intersection(&words_b).count();
    let max_len = words_a.len().max(words_b.len());
    if max_len > 0 && intersection as f32 / max_len as f32 > 0.8 {
        return true;
    }
    false
}

/// Resolve an entity to an existing concept node or create a new one.
///
/// Resolution order:
/// 1. Exact match on normalized name (instant)
/// 2. Fuzzy string match across existing concepts (fast scan)
/// 3. Create new node with normalized name
pub(crate) fn resolve_or_create_entity(
    graph: &mut Graph,
    raw_name: &str,
    concept_type: ConceptType,
    group_id: &str,
) -> Option<NodeId> {
    let normalized = normalize_entity_name(raw_name);

    // 1. Exact match on normalized name
    if let Some(&nid) = graph.concept_index.get(&*normalized) {
        return Some(nid);
    }

    // Also try raw name (handles already-normalized names)
    if let Some(&nid) = graph.concept_index.get(raw_name) {
        return Some(nid);
    }

    // 2. Fuzzy match against existing concept names
    let match_found = graph
        .concept_index
        .iter()
        .find(|(existing_name, _)| fuzzy_entity_match(&normalized, existing_name))
        .map(|(_, &nid)| nid);

    if let Some(existing_id) = match_found {
        // Add normalized name as alias
        let interned = graph.interner.intern(&normalized);
        graph.concept_index.insert(interned, existing_id);
        return Some(existing_id);
    }

    // 3. No match — create new node with normalized name
    let mut node = GraphNode::new(NodeType::Concept {
        concept_name: normalized.clone(),
        concept_type,
        confidence: 0.7,
    });
    node.group_id = group_id.to_string();
    match graph.add_node(node) {
        Ok(nid) => {
            let interned = graph.interner.intern(&normalized);
            graph.concept_index.insert(interned, nid);
            // Also index the raw name if different
            if raw_name != normalized {
                let raw_interned = graph.interner.intern(raw_name);
                graph.concept_index.insert(raw_interned, nid);
            }
            tracing::debug!(
                "Entity resolution: created concept node '{}' (raw='{}') id={}",
                normalized,
                raw_name,
                nid
            );
            Some(nid)
        },
        Err(e) => {
            tracing::warn!(
                "Entity resolution: failed to create concept node '{}': {}",
                normalized,
                e
            );
            None
        },
    }
}

/// Derive a sub-key from an object string for multi-valued categories.
/// Takes the first 3 words with 4+ chars, lowercased and joined with underscores.
pub(crate) fn derive_sub_key(object: &str) -> String {
    object
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 4)
        .take(3)
        .collect::<Vec<_>>()
        .join("_")
}
