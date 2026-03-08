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

    // ── Claim store caps (enforced in the maintenance loop) ──
    /// Maximum embeddings to keep in the in-memory vector index (0 = unlimited).
    pub max_vector_index_size: usize,

    /// Whether to purge inactive (Dormant/Rejected/Superseded) claims from disk.
    pub purge_inactive_claims: bool,

    /// Maximum days to retain soft-deleted edges before permanent removal.
    /// 0 = never purge. Default: 0.
    pub invalidated_edge_retention_days: u64,
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
            max_vector_index_size: 50_000,
            purge_inactive_claims: true,
            invalidated_edge_retention_days: 0,
        }
    }
}

/// Result of a single maintenance pass.
#[derive(Debug, Clone, Default)]
pub struct MaintenanceResult {
    pub memories_decayed: bool,
    pub strategies_pruned: usize,
    /// Number of graph nodes merged during pruning.
    pub graph_nodes_merged: usize,
    /// Number of graph nodes deleted during pruning.
    pub graph_nodes_deleted: usize,
    /// Number of graph node headers scanned during pruning.
    pub graph_headers_scanned: usize,
    /// Whether graph pruning stopped early due to budget exhaustion.
    pub graph_pruning_stopped_early: bool,
    /// Number of transition episodes cleaned up.
    pub transition_episodes_cleaned: usize,
    /// Number of weak transitions pruned.
    pub transition_entries_pruned_pass: bool,
    /// Number of memories removed due to TTL expiration.
    pub memories_expired: usize,
    /// Number of claims moved to Dormant due to TTL expiration.
    pub claims_expired: usize,
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

/// Run dedup and conflict check against existing claims via semantic search + LLM.
///
/// Uses the same conflict detection approach for both claims and graph edges:
/// 1. Semantic search for similar existing claims
/// 2. If similarity >= dedup threshold → Duplicate (merge)
/// 3. If similarity >= contradiction threshold → ask LLM if it's a contradiction
/// 4. LLM decides conflict → Contradiction (supersede old claim)
///
/// Returns the decision: store, merge, or supersede.
pub async fn check_claim_dedup(
    claim_store: &crate::claims::store::ClaimStore,
    new_claim_text: &str,
    new_embedding: &[f32],
    config: &MaintenanceConfig,
    llm: &dyn crate::llm_client::LlmClient,
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

    // If above the stricter dedup threshold → duplicate
    if similarity >= config.claim_dedup_threshold {
        return ClaimDedupDecision::Duplicate {
            existing_id,
            similarity,
        };
    }

    // Above contradiction threshold but below dedup → ask LLM if it's a conflict
    let is_conflict = llm_conflict_check(llm, new_claim_text, &existing_text).await;
    if is_conflict {
        return ClaimDedupDecision::Contradiction {
            existing_id,
            similarity,
        };
    }

    ClaimDedupDecision::NewClaim
}

/// Ask the LLM whether a new fact contradicts/supersedes an existing fact.
///
/// This is the unified conflict detection function used by both claims and
/// graph edges. Returns true if the new fact invalidates the old one.
pub async fn llm_conflict_check(
    llm: &dyn crate::llm_client::LlmClient,
    new_fact: &str,
    existing_fact: &str,
) -> bool {
    let prompt = format!(
        "Does the NEW fact contradict or supersede the EXISTING fact? \
        Answer only \"yes\" or \"no\".\n\n\
        EXISTING: {}\nNEW: {}\n\n\
        \"yes\" means: the new fact makes the existing fact outdated or false \
        (e.g. \"lives in NYC\" supersedes \"lives in Lisbon\").\n\
        \"no\" means: both facts can be true simultaneously \
        (e.g. \"works with Alice\" and \"works with Bob\" are both valid).",
        existing_fact, new_fact
    );

    let request = crate::llm_client::LlmRequest {
        system_prompt: "You detect factual contradictions. Answer only yes or no.".to_string(),
        user_prompt: prompt,
        temperature: 0.0,
        max_tokens: 5,
        json_mode: false,
    };

    match tokio::time::timeout(std::time::Duration::from_secs(8), llm.complete(request)).await {
        Ok(Ok(response)) => {
            let answer = response.content.trim().to_lowercase();
            answer.starts_with("yes")
        },
        _ => false, // On error/timeout, don't invalidate
    }
}

// ── Soft-delete edge purging ────────────────────────────────────────────────

/// Permanently remove soft-deleted edges older than `retention_days`.
///
/// Returns the number of edges permanently removed.
pub fn purge_old_invalidated_edges(
    graph: &mut crate::structures::Graph,
    retention_days: u64,
) -> usize {
    if retention_days == 0 {
        return 0;
    }

    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let cutoff_nanos = now_nanos.saturating_sub(retention_days * 86_400 * 1_000_000_000);

    // Collect edge IDs to purge
    let to_purge: Vec<crate::structures::EdgeId> = graph
        .edges
        .values()
        .filter(|e| {
            !e.is_valid()
                && e.invalidated_at()
                    .map(|ts| ts < cutoff_nanos)
                    .unwrap_or(false)
        })
        .map(|e| e.id)
        .collect();

    let count = to_purge.len();

    for edge_id in to_purge {
        if let Some(edge) = graph.edges.remove(edge_id) {
            // Remove from adjacency lists
            if let Some(out_list) = graph.adjacency_out.get_mut(edge.source) {
                out_list.retain(|eid| *eid != edge_id);
            }
            if let Some(in_list) = graph.adjacency_in.get_mut(edge.target) {
                in_list.retain(|eid| *eid != edge_id);
            }
            // Update degree
            if let Some(source) = graph.nodes.get_mut(edge.source) {
                source.degree = source.degree.saturating_sub(1);
                graph.total_degree = graph.total_degree.saturating_sub(1);
            }
            if let Some(target) = graph.nodes.get_mut(edge.target) {
                target.degree = target.degree.saturating_sub(1);
                graph.total_degree = graph.total_degree.saturating_sub(1);
            }
            graph.deleted_edges.insert(edge_id);
            graph.dirty_edges.remove(&edge_id);
        }
    }

    if count > 0 {
        graph.generation += 1;
        graph.adjacency_dirty = true;
    }

    count
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock LLM that always answers "yes" (conflict) for testing.
    struct MockConflictLlm;

    #[async_trait::async_trait]
    impl crate::llm_client::LlmClient for MockConflictLlm {
        async fn complete(
            &self,
            _request: crate::llm_client::LlmRequest,
        ) -> anyhow::Result<crate::llm_client::LlmResponse> {
            Ok(crate::llm_client::LlmResponse {
                content: "yes".to_string(),
                tokens_used: 1,
            })
        }
        fn model_name(&self) -> &str {
            "mock-conflict"
        }
    }

    /// Mock LLM that always answers "no" (no conflict) for testing.
    struct MockNoConflictLlm;

    #[async_trait::async_trait]
    impl crate::llm_client::LlmClient for MockNoConflictLlm {
        async fn complete(
            &self,
            _request: crate::llm_client::LlmRequest,
        ) -> anyhow::Result<crate::llm_client::LlmResponse> {
            Ok(crate::llm_client::LlmResponse {
                content: "no".to_string(),
                tokens_used: 1,
            })
        }
        fn model_name(&self) -> &str {
            "mock-no-conflict"
        }
    }

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

    #[tokio::test]
    async fn test_dedup_decision_new_when_empty_embedding() {
        let config = MaintenanceConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::claims::store::ClaimStore::new(dir.path().join("claims.redb")).unwrap();
        let llm = MockNoConflictLlm;

        let decision = check_claim_dedup(&store, "some text", &[], &config, &llm).await;
        assert!(matches!(decision, ClaimDedupDecision::NewClaim));
    }

    #[tokio::test]
    async fn test_dedup_detects_duplicate() {
        let config = MaintenanceConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::claims::store::ClaimStore::new(dir.path().join("claims.redb")).unwrap();
        let llm = MockNoConflictLlm;

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
            &llm,
        )
        .await;
        assert!(matches!(decision, ClaimDedupDecision::Duplicate { .. }));
    }

    #[tokio::test]
    async fn test_dedup_detects_contradiction() {
        let config = MaintenanceConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::claims::store::ClaimStore::new(dir.path().join("claims.redb")).unwrap();
        let llm = MockConflictLlm; // answers "yes" to conflict check

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

        // Query with embedding that gives cosine similarity ~0.88 vs [1,0,0]
        // (above contradiction threshold 0.85, below dedup threshold 0.92)
        let decision = check_claim_dedup(
            &store,
            "I do not like Adidas shoes",
            &[0.88, 0.475, 0.0],
            &config,
            &llm,
        )
        .await;
        assert!(matches!(decision, ClaimDedupDecision::Contradiction { .. }));
    }
}
