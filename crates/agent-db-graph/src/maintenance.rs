// crates/agent-db-graph/src/maintenance.rs
//
// Background Maintenance Engine
//
// Runs periodic housekeeping tasks:
//   1. Memory decay  — age out stale memories below forget_threshold
//   2. Strategy pruning — remove low-quality / stale strategies
//   3. Graph node cap — enforce per-bucket node limits (oldest Event nodes evicted first)
//
// Spawned as a background tokio task via `GraphEngine::start_maintenance_loop`.

/// Configuration for the periodic maintenance loop.
#[derive(Debug, Clone)]
pub struct MaintenanceConfig {
    /// Interval between maintenance passes (seconds). 0 = disabled.
    pub interval_secs: u64,

    // ── Strategy pruning thresholds ──
    /// Minimum confidence to survive pruning (strategies below this AND below
    /// `strategy_min_support` are removed).
    pub strategy_min_confidence: f32,

    /// Minimum support count to survive pruning.
    pub strategy_min_support: u32,

    /// Strategies not used within this many hours (and below support threshold) are pruned.
    pub strategy_max_stale_hours: f32,

    // ── Claim dedup thresholds (used at insertion time, not in the loop) ──
    /// Cosine similarity above which two claims are considered duplicates.
    pub claim_dedup_threshold: f32,

    /// Cosine similarity above which we check for contradiction.
    pub claim_contradiction_threshold: f32,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            interval_secs: 300, // 5 minutes
            strategy_min_confidence: 0.15,
            strategy_min_support: 1,
            strategy_max_stale_hours: 72.0, // 3 days
            claim_dedup_threshold: 0.92,
            claim_contradiction_threshold: 0.85,
        }
    }
}

/// Result of a single maintenance pass.
#[derive(Debug, Clone, Default)]
pub struct MaintenanceResult {
    pub memories_decayed: bool,
    pub strategies_pruned: usize,
}

// ── Negation detection for claim contradiction ─────────────────────────────

/// Simple heuristic: returns `true` when one claim contains a negation that the
/// other does not, indicating they are likely contradictory.
///
/// Examples:
///   "I like Adidas shoes" vs "I do not like Adidas shoes" → true
///   "I like Adidas shoes" vs "I like Nike shoes"           → false
pub fn is_contradiction(claim_a: &str, claim_b: &str) -> bool {
    const NEGATION_WORDS: &[&str] = &[
        "not",
        "no",
        "don't",
        "doesn't",
        "never",
        "neither",
        "nor",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "couldn't",
        "shouldn't",
        "cannot",
        "can't",
        "haven't",
        "hasn't",
        "hadn't",
        "didn't",
        "dislike",
        "hate",
        "refuse",
        "reject",
    ];

    let a_lower = claim_a.to_lowercase();
    let b_lower = claim_b.to_lowercase();

    let a_words: Vec<&str> = a_lower.split_whitespace().collect();
    let b_words: Vec<&str> = b_lower.split_whitespace().collect();

    let a_has_neg = NEGATION_WORDS.iter().any(|neg| {
        a_words
            .iter()
            .any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *neg)
    });
    let b_has_neg = NEGATION_WORDS.iter().any(|neg| {
        b_words
            .iter()
            .any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *neg)
    });

    // Contradiction when exactly one side is negated
    a_has_neg != b_has_neg
}

/// Dedup decision for a new claim relative to an existing one.
#[derive(Debug, Clone)]
pub enum ClaimDedupDecision {
    /// No similar claim found — store normally.
    NewClaim,
    /// Near-duplicate found — increment support on existing, skip storing new.
    Duplicate { existing_id: u64, similarity: f32 },
    /// Contradiction found — store new claim and mark the old one as Superseded.
    Contradiction { existing_id: u64, similarity: f32 },
}

/// Run dedup check against existing claims via the vector index.
///
/// Returns the decision: store, merge, or supersede.
pub fn check_claim_dedup(
    claim_store: &crate::claims::store::ClaimStore,
    new_claim_text: &str,
    new_embedding: &[f32],
    config: &MaintenanceConfig,
) -> ClaimDedupDecision {
    if new_embedding.is_empty() {
        return ClaimDedupDecision::NewClaim;
    }

    // Find the most similar existing claim
    let similar =
        match claim_store.find_similar(new_embedding, 1, config.claim_contradiction_threshold) {
            Ok(results) => results,
            Err(_) => return ClaimDedupDecision::NewClaim,
        };

    let Some((existing_id, similarity)) = similar.first().copied() else {
        return ClaimDedupDecision::NewClaim;
    };

    // Load existing claim text
    let existing_text = match claim_store.get(existing_id) {
        Ok(Some(c)) => c.claim_text,
        _ => return ClaimDedupDecision::NewClaim,
    };

    // Check for contradiction (negation flip)
    if is_contradiction(new_claim_text, &existing_text) {
        return ClaimDedupDecision::Contradiction {
            existing_id,
            similarity,
        };
    }

    // If above the stricter dedup threshold → duplicate
    if similarity >= config.claim_dedup_threshold {
        return ClaimDedupDecision::Duplicate {
            existing_id,
            similarity,
        };
    }

    // Similar but not duplicate and not contradiction — treat as new
    ClaimDedupDecision::NewClaim
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contradiction_detection() {
        assert!(is_contradiction(
            "I like Adidas shoes",
            "I do not like Adidas shoes"
        ));
        assert!(is_contradiction(
            "The API is stable",
            "The API isn't stable"
        ));
        assert!(is_contradiction("I hate running", "I like running"));
        // Same polarity
        assert!(!is_contradiction(
            "I like Adidas shoes",
            "I like Nike shoes"
        ));
        // Both negative
        assert!(!is_contradiction(
            "I don't like Adidas",
            "I don't like Nike"
        ));
    }

    #[test]
    fn test_dedup_decision_new_when_empty_embedding() {
        let config = MaintenanceConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::claims::store::ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let decision = check_claim_dedup(&store, "some text", &[], &config);
        assert!(matches!(decision, ClaimDedupDecision::NewClaim));
    }

    #[test]
    fn test_dedup_detects_duplicate() {
        let config = MaintenanceConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::claims::store::ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        // Store an existing claim
        let existing = crate::claims::types::DerivedClaim::new(
            1,
            "I like Adidas shoes".to_string(),
            vec![crate::claims::types::EvidenceSpan::new(0, 5, "I lik")],
            0.9,
            vec![1.0, 0.0, 0.0],
            100,
            None,
            None,
            None,
            None,
        );
        store.store(&existing).unwrap();

        // Query with nearly identical embedding
        let decision = check_claim_dedup(
            &store,
            "I like Adidas sneakers",
            &[0.99, 0.01, 0.0],
            &config,
        );
        assert!(matches!(decision, ClaimDedupDecision::Duplicate { .. }));
    }

    #[test]
    fn test_dedup_detects_contradiction() {
        let config = MaintenanceConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::claims::store::ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        // Store an existing claim
        let existing = crate::claims::types::DerivedClaim::new(
            1,
            "I like Adidas shoes".to_string(),
            vec![crate::claims::types::EvidenceSpan::new(0, 5, "I lik")],
            0.9,
            vec![1.0, 0.0, 0.0],
            100,
            None,
            None,
            None,
            None,
        );
        store.store(&existing).unwrap();

        // Query with negated claim but similar embedding
        let decision = check_claim_dedup(
            &store,
            "I do not like Adidas shoes",
            &[0.98, 0.02, 0.0],
            &config,
        );
        assert!(matches!(decision, ClaimDedupDecision::Contradiction { .. }));
    }
}
