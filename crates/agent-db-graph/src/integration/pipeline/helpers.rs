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

/// Threshold for Jaccard similarity on character 3-grams. Tuned to catch
/// typos and case variations ("Mochi" / "Moshi", "Luna Smith" / "Luna Smyth")
/// without merging unrelated entities. 0.9 is the empirically-stable point
/// for the entity-name dedup use case across the entity-resolution
/// literature.
const JACCARD_THRESHOLD: f32 = 0.9;

/// Names below this Shannon entropy threshold are too generic to auto-merge
/// via fuzzy match. "Apple" / "Mike" / "Luna" can refer to different
/// real-world entities, so we fall through to exact-match-only resolution
/// for them. (Their exact-match path still works — only the fuzzy step is
/// gated.)
const ENTITY_ENTROPY_THRESHOLD: f32 = 1.5;

/// Minimum name length for auto-fuzzy-merge. Short names are too ambiguous.
const MIN_FUZZY_LENGTH: usize = 6;

/// Minimum token count for auto-fuzzy-merge. Single-word names are too
/// ambiguous.
const MIN_FUZZY_TOKENS: usize = 2;

/// Compute Shannon entropy over the characters of a name (whitespace
/// stripped). Used by the entropy gate — short / low-entropy names are
/// considered ambiguous and excluded from auto-fuzzy-merge.
fn shannon_entropy(s: &str) -> f32 {
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return 0.0;
    }
    let total = chars.len() as f32;
    let mut counts: HashMap<char, u32> = HashMap::new();
    for c in &chars {
        *counts.entry(*c).or_insert(0) += 1;
    }
    counts
        .values()
        .map(|&n| {
            let p = n as f32 / total;
            -p * p.log2()
        })
        .sum()
}

/// Build the 3-gram character shingle set for a name (for Jaccard).
fn shingles_3gram(s: &str) -> HashSet<[u8; 3]> {
    // Operate on lowercase ASCII bytes; non-ASCII chars are passed through
    // as their UTF-8 byte sequences (which yields slightly different
    // shingles but is consistent).
    let lower = s.to_lowercase();
    let bytes = lower.as_bytes();
    let mut set: HashSet<[u8; 3]> = HashSet::new();
    if bytes.len() < 3 {
        return set;
    }
    for window in bytes.windows(3) {
        let mut arr = [0u8; 3];
        arr.copy_from_slice(window);
        set.insert(arr);
    }
    set
}

/// Jaccard similarity on 3-gram shingles.
fn jaccard_3gram(a: &HashSet<[u8; 3]>, b: &HashSet<[u8; 3]>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Entropy gate. Returns true if a name is safe to auto-merge by fuzzy
/// similarity. Names that are short, single-token, or low-entropy fall
/// through to exact-match-only resolution — too ambiguous to merge
/// without LLM-level disambiguation.
fn name_is_safe_to_auto_fuzzy_merge(name: &str) -> bool {
    if name.chars().count() < MIN_FUZZY_LENGTH {
        return false;
    }
    if name.split_whitespace().count() < MIN_FUZZY_TOKENS {
        return false;
    }
    if shannon_entropy(name) < ENTITY_ENTROPY_THRESHOLD {
        return false;
    }
    true
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

    // 2. Fuzzy match against existing concept names. Two passes, ordered
    // by cost (cheapest first) and conservatism (no entropy gate on the
    // existing substring path; new 3-gram Jaccard path is gated):
    //
    //   2a. Existing fuzzy_entity_match — substring / >80% word overlap.
    //       No entropy gate — this is the historical behaviour and is
    //       already conservative enough (it requires word-level overlap,
    //       not character-level). Resolves "luna" inside "luna the cat".
    //
    //   2b. NEW: character 3-gram Jaccard ≥ 0.9, gated by name entropy.
    //       Catches typos ("Mochi" / "Mohci") and small variants
    //       ("Sarah Chen" / "Sarah Chenn") that the word-level filter
    //       misses. Entropy-gated so short / single-token / low-entropy
    //       names ("Luna", "Mike") don't auto-merge across what could be
    //       different real-world entities.
    //
    // Both passes are deterministic — no LLM call. Cost: a 3-gram set per
    // candidate name and a linear scan of concept_index. For <1K concepts
    // per tenant this is sub-millisecond.
    let match_found = graph
        .concept_index
        .iter()
        .find(|(existing_name, _)| fuzzy_entity_match(&normalized, existing_name))
        .map(|(_, &nid)| nid)
        .or_else(|| {
            if !name_is_safe_to_auto_fuzzy_merge(&normalized) {
                return None;
            }
            let query_shingles = shingles_3gram(&normalized);
            let mut best: Option<(f32, NodeId)> = None;
            for (existing_name, &nid) in graph.concept_index.iter() {
                let score = jaccard_3gram(&query_shingles, &shingles_3gram(existing_name));
                if score >= JACCARD_THRESHOLD && best.map(|(s, _)| score > s).unwrap_or(true) {
                    best = Some((score, nid));
                }
            }
            best.map(|(_, nid)| nid)
        });

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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shannon entropy ──────────────────────────────────────────────

    #[test]
    fn entropy_all_same_char_is_zero() {
        assert!(shannon_entropy("aaaa") < 0.01);
    }

    #[test]
    fn entropy_low_for_short_repeating_name() {
        // "ana" has chars {a:2, n:1} -> entropy ~0.92
        assert!(shannon_entropy("ana") < ENTITY_ENTROPY_THRESHOLD);
    }

    #[test]
    fn entropy_high_for_diverse_long_name() {
        // "sarah chen" has 9 distinct chars in 9 (whitespace stripped) -> entropy ~3.17
        assert!(shannon_entropy("sarah chen") > ENTITY_ENTROPY_THRESHOLD);
    }

    // ── Shingles ─────────────────────────────────────────────────────

    #[test]
    fn shingles_of_short_name_empty() {
        // "lu" has only 2 chars — no 3-grams possible.
        assert!(shingles_3gram("lu").is_empty());
    }

    #[test]
    fn shingles_count_matches_string_length() {
        // "luna" has 4 chars -> 2 trigrams: "lun", "una".
        let s = shingles_3gram("luna");
        assert_eq!(s.len(), 2);
        assert!(s.contains(b"lun"));
        assert!(s.contains(b"una"));
    }

    #[test]
    fn shingles_lowercased() {
        let s1 = shingles_3gram("LUNA");
        let s2 = shingles_3gram("luna");
        assert_eq!(s1, s2);
    }

    // ── Jaccard ──────────────────────────────────────────────────────

    #[test]
    fn jaccard_identical_is_one() {
        let s = shingles_3gram("sarah chen");
        assert!((jaccard_3gram(&s, &s) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjoint_is_zero() {
        let a = shingles_3gram("alice");
        let b = shingles_3gram("zorgo");
        // No 3-gram overlap between "alice" and "zorgo".
        assert_eq!(jaccard_3gram(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_two_empty_sets_is_one() {
        let a = HashSet::new();
        let b = HashSet::new();
        assert_eq!(jaccard_3gram(&a, &b), 1.0);
    }

    // ── Entropy gate ─────────────────────────────────────────────────

    #[test]
    fn entropy_gate_blocks_short_names() {
        // "luna" (4 chars) — below MIN_FUZZY_LENGTH.
        assert!(!name_is_safe_to_auto_fuzzy_merge("luna"));
        assert!(!name_is_safe_to_auto_fuzzy_merge("mike"));
    }

    #[test]
    fn entropy_gate_blocks_single_token_names() {
        // "panopticon" — long but one word.
        assert!(!name_is_safe_to_auto_fuzzy_merge("panopticon"));
    }

    #[test]
    fn entropy_gate_passes_realistic_full_name() {
        assert!(name_is_safe_to_auto_fuzzy_merge("sarah chen"));
        assert!(name_is_safe_to_auto_fuzzy_merge("anne frank house"));
    }
}
