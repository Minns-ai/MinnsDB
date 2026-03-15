// crates/agent-db-graph/src/integration/queries/search.rs
//
// BM25 full-text search, hybrid semantic search, and code entity search.

use super::*;

impl GraphEngine {
    /// Search nodes using BM25 full-text search across all indexes
    /// (graph nodes, memories, strategies).
    ///
    /// Returns a list of (NodeId, score) tuples ranked by relevance
    pub async fn search_bm25(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
        let inference = self.inference.read().await;
        let mut results = inference.graph().bm25_index.search(query, limit);

        // Also search memory and strategy BM25 indexes
        {
            let mem_idx = self.memory_bm25_index.read().await;
            let mem_hits = mem_idx.search(query, limit);
            for (id, score) in mem_hits {
                // Memory IDs are u64; look up the corresponding graph node
                if let Some(&node_id) = inference.graph().memory_index.get(&id) {
                    results.push((node_id, score));
                } else {
                    results.push((id, score));
                }
            }
        }
        {
            let strat_idx = self.strategy_bm25_index.read().await;
            let strat_hits = strat_idx.search(query, limit);
            for (id, score) in strat_hits {
                if let Some(&node_id) = inference.graph().strategy_index.get(&id) {
                    results.push((node_id, score));
                } else {
                    results.push((id, score));
                }
            }
        }

        // Also search claim store BM25 index
        if let Some(ref store) = self.claim_store {
            store.apply_pending();
            let claim_bm25 = store.bm25_index().read();
            let claim_hits = claim_bm25.search(query, limit);
            drop(claim_bm25);
            for (claim_id, score) in claim_hits {
                if let Some(&node_id) = inference.graph().claim_index.get(&claim_id) {
                    results.push((node_id, score));
                }
            }
        }

        // Deduplicate by node_id, keeping highest score
        results.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        results.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = b.1.max(a.1); // keep highest score in b
                true
            } else {
                false
            }
        });

        // Re-sort by score descending and truncate to limit
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);
        results
    }

    /// Get a node by ID for search results
    pub async fn get_node(&self, node_id: u64) -> Option<crate::structures::GraphNode> {
        let inference = self.inference.read().await;
        inference.graph().nodes.get(node_id).cloned()
    }

    /// Search claims using hybrid BM25 + semantic search.
    ///
    /// Gracefully degrades to BM25-only when no embedding client is configured.
    /// Returns a list of (NodeId, score) tuples for claims ranked by relevance.
    pub async fn search_claims_semantic(
        &self,
        query: &str,
        limit: usize,
        min_similarity: f32,
    ) -> crate::GraphResult<Vec<(u64, f32)>> {
        let claim_store = match &self.claim_store {
            Some(s) => s,
            None => return Ok(vec![]), // No claim store available
        };

        // Generate embedding if client is available; degrade to keyword-only otherwise
        let query_embedding = if let Some(ref embedding_client) = self.embedding_client {
            let request = crate::claims::EmbeddingRequest {
                text: query.to_string(),
                context: None,
            };

            match embedding_client.embed(request).await {
                Ok(response) => Some(response.embedding),
                Err(e) => {
                    tracing::info!(
                        "Embedding generation failed, falling back to keyword-only: {}",
                        e
                    );
                    None
                },
            }
        } else {
            None
        };

        let search_mode = if query_embedding.is_some() {
            crate::indexing::SearchMode::Hybrid
        } else {
            crate::indexing::SearchMode::Keyword
        };

        let hybrid_config = crate::claims::hybrid_search::HybridSearchConfig {
            mode: search_mode,
            min_similarity,
            ..Default::default()
        };

        let empty_embedding: Vec<f32> = Vec::new();
        let search_embedding = query_embedding.as_deref().unwrap_or(&empty_embedding);

        let similar_claims = crate::claims::hybrid_search::HybridClaimSearch::search(
            query,
            search_embedding,
            claim_store,
            limit,
            &hybrid_config,
        )
        .map_err(|e| {
            crate::GraphError::OperationError(format!("Failed to search claims: {}", e))
        })?;

        // Convert claim IDs to node IDs
        let inference = self.inference.read().await;
        let graph = inference.graph();

        let mut results = Vec::new();
        for (claim_id, similarity) in similar_claims {
            if let Some(&node_id) = graph.claim_index.get(&claim_id) {
                results.push((node_id, similarity));
            }
        }

        Ok(results)
    }

    /// Search code entities in the graph by filtering Concept nodes with code metadata.
    pub async fn search_code_entities(
        &self,
        name_pattern: Option<&str>,
        kind: Option<&str>,
        language: Option<&str>,
        file_pattern: Option<&str>,
        limit: usize,
    ) -> Vec<crate::code_graph::CodeEntityMatch> {
        let inference = self.inference.read().await;
        crate::code_graph::search_code_entities_in_graph(
            inference.graph(),
            name_pattern,
            kind,
            language,
            file_pattern,
            limit,
        )
    }
}
