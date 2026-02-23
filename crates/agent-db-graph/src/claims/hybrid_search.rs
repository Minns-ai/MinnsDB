//! Hybrid BM25 + vector search for claims via RRF fusion

use super::store::ClaimStore;
use super::types::ClaimId;
use crate::indexing::{FusionStrategy, SearchMode};
use anyhow::Result;

/// Configuration for hybrid claim search
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// How to combine keyword and semantic results
    pub mode: SearchMode,
    /// Fusion strategy (only used when mode = Hybrid)
    pub fusion: FusionStrategy,
    /// Minimum vector similarity threshold
    pub min_similarity: f32,
    /// How many results to fetch from each index before fusion
    pub per_index_limit: usize,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            mode: SearchMode::Hybrid,
            fusion: FusionStrategy::default(),
            min_similarity: 0.3,
            per_index_limit: 50,
        }
    }
}

/// Hybrid search combining BM25 keyword search and vector similarity
pub struct HybridClaimSearch;

impl HybridClaimSearch {
    /// Search claims using hybrid BM25 + vector search with rank fusion.
    ///
    /// Returns `(ClaimId, fused_score)` pairs sorted by descending score.
    pub fn search(
        query_text: &str,
        query_embedding: &[f32],
        claim_store: &ClaimStore,
        top_k: usize,
        config: &HybridSearchConfig,
    ) -> Result<Vec<(ClaimId, f32)>> {
        // Apply pending index updates for consistency
        claim_store.apply_pending();

        match config.mode {
            SearchMode::Keyword => {
                let bm25 = claim_store.bm25_index().read();
                let mut results = bm25.search(query_text, config.per_index_limit);
                results.truncate(top_k);
                Ok(results)
            },
            SearchMode::Semantic => {
                let results =
                    claim_store.find_similar(query_embedding, top_k, config.min_similarity)?;
                Ok(results)
            },
            SearchMode::Hybrid => {
                // BM25 keyword leg
                let keyword_results = {
                    let bm25 = claim_store.bm25_index().read();
                    bm25.search(query_text, config.per_index_limit)
                };

                // Vector similarity leg
                let semantic_results = claim_store.find_similar(
                    query_embedding,
                    config.per_index_limit,
                    config.min_similarity,
                )?;

                // Fuse via RRF (or weighted)
                let mut fused = config.fusion.fuse(keyword_results, semantic_results);
                fused.truncate(top_k);
                Ok(fused)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::types::{DerivedClaim, EvidenceSpan};
    use tempfile::tempdir;

    fn make_claim(id: ClaimId, text: &str, embedding: Vec<f32>) -> DerivedClaim {
        let evidence = vec![EvidenceSpan::new(0, 4, "test")];
        let mut c = DerivedClaim::new(
            id,
            text.to_string(),
            evidence,
            0.9,
            embedding,
            100,
            None,
            None,
            None,
            None,
        );
        c.status = crate::claims::types::ClaimStatus::Active;
        c
    }

    #[test]
    fn test_hybrid_search_finds_both_keyword_and_semantic() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        // Claims with distinct keywords and embeddings
        // "rust compiler" — keyword match only (embedding along x)
        store
            .store(&make_claim(
                1,
                "The rust compiler is fast and safe",
                vec![1.0, 0.0, 0.0],
            ))
            .unwrap();
        // "memory safety" — semantic match only (embedding along y, close to query)
        store
            .store(&make_claim(
                2,
                "Memory safety guarantees prevent bugs",
                vec![0.0, 0.95, 0.05],
            ))
            .unwrap();
        // "rust memory" — both keyword and semantic match
        store
            .store(&make_claim(
                3,
                "Rust memory model ensures thread safety",
                vec![0.5, 0.8, 0.0],
            ))
            .unwrap();
        // Unrelated claim
        store
            .store(&make_claim(
                4,
                "Python web framework django",
                vec![0.0, 0.0, 1.0],
            ))
            .unwrap();

        let config = HybridSearchConfig::default();

        // Query: text has "rust", embedding close to y-axis
        let query_embedding = vec![0.0, 1.0, 0.0];
        let results =
            HybridClaimSearch::search("rust programming", &query_embedding, &store, 10, &config)
                .unwrap();

        // Claim 3 should be top (matches both keyword "rust" and semantic y-axis)
        assert!(!results.is_empty());
        let ids: Vec<ClaimId> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&3), "claim 3 (both) should appear");
        assert!(ids.contains(&1), "claim 1 (keyword 'rust') should appear");
        assert!(ids.contains(&2), "claim 2 (semantic) should appear");
        // Claim 4 should be missing or very low
        assert!(
            !ids.contains(&4) || results.iter().find(|r| r.0 == 4).unwrap().1 < 0.01,
            "claim 4 should not appear"
        );
    }

    #[test]
    fn test_keyword_only_mode() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        store
            .store(&make_claim(1, "The quick brown fox", vec![1.0, 0.0, 0.0]))
            .unwrap();
        store
            .store(&make_claim(2, "A lazy dog sleeps", vec![0.0, 1.0, 0.0]))
            .unwrap();

        let config = HybridSearchConfig {
            mode: SearchMode::Keyword,
            ..Default::default()
        };

        let results =
            HybridClaimSearch::search("quick fox", &[0.0, 1.0, 0.0], &store, 10, &config).unwrap();
        // Only claim 1 has "quick" and "fox"
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_semantic_only_mode() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        store
            .store(&make_claim(1, "Alpha", vec![1.0, 0.0, 0.0]))
            .unwrap();
        store
            .store(&make_claim(2, "Beta", vec![0.0, 1.0, 0.0]))
            .unwrap();

        let config = HybridSearchConfig {
            mode: SearchMode::Semantic,
            min_similarity: 0.5,
            ..Default::default()
        };

        let results =
            HybridClaimSearch::search("irrelevant", &[1.0, 0.0, 0.0], &store, 10, &config).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
