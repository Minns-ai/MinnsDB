//! Hybrid BM25 + vector search for claims via RRF fusion.

use super::store::ClaimStore;
use super::types::ClaimId;
use crate::indexing::{FusionStrategy, SearchMode};
use anyhow::Result;

/// Configuration for hybrid claim search.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// How to combine keyword and semantic results.
    pub mode: SearchMode,
    /// Fusion strategy (only used when mode = Hybrid).
    pub fusion: FusionStrategy,
    /// Minimum vector similarity threshold.
    pub min_similarity: f32,
    /// How many results to fetch from each index before fusion.
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

/// Hybrid search combining BM25 keyword search and vector similarity.
pub struct HybridClaimSearch;

impl HybridClaimSearch {
    /// Search claims using hybrid BM25 + vector search with rank fusion.
    ///
    /// Returns `(ClaimId, fused_score)` pairs sorted by descending score.
    ///
    /// The keyword leg runs locally against the in-memory BM25 index. The
    /// semantic leg goes to the configured
    /// [`VectorStore`](minns_vectors::VectorStore) (Qdrant in production).
    pub async fn search(
        query_text: &str,
        query_embedding: &[f32],
        claim_store: &ClaimStore,
        top_k: usize,
        config: &HybridSearchConfig,
    ) -> Result<Vec<(ClaimId, f32)>> {
        match config.mode {
            SearchMode::Keyword => {
                let bm25 = claim_store.bm25_index().read();
                let mut results = bm25.search(query_text, config.per_index_limit);
                results.truncate(top_k);
                Ok(results)
            },
            SearchMode::Semantic => {
                let results = claim_store
                    .find_similar(query_embedding, top_k, config.min_similarity)
                    .await?;
                Ok(results)
            },
            SearchMode::Hybrid => {
                // BM25 keyword leg (sync, in-memory)
                let keyword_results = {
                    let bm25 = claim_store.bm25_index().read();
                    bm25.search(query_text, config.per_index_limit)
                };

                // Vector similarity leg (async, Qdrant)
                let semantic_results = claim_store
                    .find_similar(
                        query_embedding,
                        config.per_index_limit,
                        config.min_similarity,
                    )
                    .await?;

                let mut fused = config.fusion.fuse(keyword_results, semantic_results);
                fused.truncate(top_k);
                Ok(fused)
            },
        }
    }
}

// Hybrid search behaviour is covered by `tests/claims_integration.rs`, which
// drives both BM25 and the Qdrant-backed vector store via the shared
// testcontainers fixture.
