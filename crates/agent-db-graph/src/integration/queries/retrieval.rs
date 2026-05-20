// crates/agent-db-graph/src/integration/queries/retrieval.rs
//
// Memory and strategy retrieval methods (hierarchical, multi-signal, by-context).

use super::*;

impl GraphEngine {
    /// Retrieve memories using hierarchical search (Schema > Semantic > Episodic)
    pub async fn retrieve_memories_hierarchical(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
    ) -> Vec<Memory> {
        let mut store = self.memory_store.write().await;
        store.retrieve_hierarchical(context, limit, min_similarity, agent_id)
    }

    /// Manually trigger a consolidation pass
    pub async fn run_consolidation(&self) -> crate::consolidation::ConsolidationResult {
        // Infer goal labels for goalless buckets if LLM is available
        // Extract data under read lock, drop it, then make LLM call without holding lock
        let all_memories = {
            let store = self.memory_store.read().await;
            store.list_all_memories()
        };
        let goal_overrides = if let Some(ref llm) = self.unified_llm_client {
            crate::consolidation::infer_goal_labels(llm.as_ref(), &all_memories).await
        } else {
            std::collections::HashMap::new()
        };

        // Pre-fetch embeddings for every memory that claims to have one. The
        // consolidation engine is pure and synchronous; doing the Qdrant
        // fetch here keeps it that way. One RPC for the whole snapshot.
        let memory_vectors = {
            let ids: Vec<u128> = all_memories
                .iter()
                .filter(|m| m.has_summary_embedding)
                .map(|m| m.id as u128)
                .collect();
            if ids.is_empty() {
                std::collections::HashMap::new()
            } else {
                match self.vectors.memories.fetch(&ids).await {
                    Ok(points) => points
                        .into_iter()
                        .zip(ids.iter())
                        .filter_map(|(p, id)| p.map(|pt| (*id as u64, pt.vector)))
                        .collect(),
                    Err(e) => {
                        tracing::warn!("Memory vector prefetch for consolidation failed: {e}");
                        std::collections::HashMap::new()
                    },
                }
            }
        };

        let mut store = self.memory_store.write().await;
        let mut engine = self.consolidation_engine.write().await;
        engine.run_consolidation(store.as_mut(), &goal_overrides, &memory_vectors)
    }

    /// Get all memories for an agent
    pub async fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory> {
        self.memory_store
            .read()
            .await
            .get_agent_memories(agent_id, limit)
    }

    /// Retrieve memories by context similarity
    pub async fn retrieve_memories_by_context(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
    ) -> Vec<Memory> {
        self.memory_store
            .write()
            .await
            .retrieve_by_context(context, limit)
    }

    /// Retrieve memories by context similarity with optional filtering
    pub async fn retrieve_memories_by_context_similar(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory> {
        self.memory_store.write().await.retrieve_by_context_similar(
            context,
            limit,
            min_similarity,
            agent_id,
            session_id,
        )
    }

    /// Get all strategies for an agent
    pub async fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy> {
        let extractor = self.strategy_store.read().await;
        extractor.get_agent_strategies(agent_id, limit)
    }

    /// Get strategies applicable to a context
    pub async fn get_strategies_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
    ) -> Vec<Strategy> {
        let extractor = self.strategy_store.read().await;
        extractor.get_strategies_for_context(context_hash, limit)
    }

    /// Find strategies similar to a graph signature
    pub async fn get_similar_strategies(
        &self,
        query: StrategySimilarityQuery,
    ) -> Vec<(Strategy, f32)> {
        let extractor = self.strategy_store.read().await;
        extractor.find_similar_strategies(query)
    }

    /// Get all completed episodes
    pub async fn get_completed_episodes(&self) -> Vec<Episode> {
        self.episode_detector
            .read()
            .await
            .get_completed_episodes()
            .to_vec()
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        self.memory_store.read().await.get_stats()
    }

    /// Get strategy statistics
    pub async fn get_strategy_stats(&self) -> StrategyStats {
        self.strategy_store.read().await.get_stats()
    }

    /// Retrieve memories using multi-signal fusion (semantic + BM25 + context +
    /// temporal + PPR + access frequency), with tier boosts.
    ///
    /// This is the recommended retrieval method when multiple signals are
    /// available. Falls back gracefully when signals are missing.
    pub async fn retrieve_memories_multi_signal(
        &self,
        query: crate::retrieval::MemoryRetrievalQuery,
        config: Option<crate::retrieval::MemoryRetrievalConfig>,
    ) -> Vec<Memory> {
        let config = config.unwrap_or_default();

        // Load candidates from store
        let candidates = {
            let store = self.memory_store.read().await;
            store.list_all_memories()
        };

        if candidates.is_empty() {
            return Vec::new();
        }

        // Compute PPR if anchor node provided
        let ppr_scores = if let Some(anchor) = query.anchor_node {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            self.random_walker.personalized_pagerank(graph, anchor).ok()
        } else {
            None
        };

        // Get memory→node mapping
        let memory_to_node = {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            graph.memory_index.clone()
        };

        // Pre-compute Signal 1 (semantic) scores via Qdrant when an embedding
        // is supplied. One Qdrant `search` RPC, top_k = per_signal_limit, so
        // the pipeline itself stays pure and synchronous.
        let semantic_scores = if !query.query_embedding.is_empty() {
            let q = minns_vectors::Query::builder(query.query_embedding.clone())
                .top_k(config.per_signal_limit)
                .min_score(config.min_semantic_similarity)
                .build();
            match self.vectors.memories.search(&q).await {
                Ok(hits) => {
                    let map: std::collections::HashMap<crate::memory::MemoryId, f32> =
                        hits.into_iter().map(|h| (h.id as u64, h.score)).collect();
                    if map.is_empty() {
                        None
                    } else {
                        Some(map)
                    }
                },
                Err(e) => {
                    tracing::warn!("Memory vector search failed: {e}");
                    None
                },
            }
        } else {
            None
        };

        // Run the pipeline
        let bm25 = self.memory_bm25_index.read().await;
        let ranked = crate::retrieval::MemoryRetrievalPipeline::retrieve(
            &candidates,
            &query,
            &config,
            Some(&*bm25),
            semantic_scores.as_ref(),
            ppr_scores.as_ref(),
            Some(&memory_to_node),
        );

        // Resolve IDs to Memory objects
        let limit = query.limit;
        let store = self.memory_store.read().await;
        let mut results = Vec::with_capacity(ranked.len().min(limit));
        for (memory_id, _score) in ranked.into_iter().take(limit) {
            if let Some(mem) = store.get_memory(memory_id) {
                results.push(mem);
            }
        }
        results
    }

    /// Retrieve strategies using multi-signal fusion (semantic + BM25 + Jaccard +
    /// temporal + PPR + quality×confidence).
    ///
    /// Falls back gracefully when signals are missing.
    pub async fn retrieve_strategies_multi_signal(
        &self,
        query: crate::retrieval::StrategyRetrievalQuery,
        config: Option<crate::retrieval::StrategyRetrievalConfig>,
    ) -> Vec<Strategy> {
        let config = config.unwrap_or_default();

        // Load candidates from store
        let candidates = {
            let store = self.strategy_store.read().await;
            store.list_all_strategies()
        };

        if candidates.is_empty() {
            return Vec::new();
        }

        // Compute PPR if anchor node provided
        let ppr_scores = if let Some(anchor) = query.anchor_node {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            self.random_walker.personalized_pagerank(graph, anchor).ok()
        } else {
            None
        };

        // Get strategy→node mapping
        let strategy_to_node = {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            graph.strategy_index.clone()
        };

        // Run the pipeline (no pre-computed Jaccard for now — pass None)
        let bm25 = self.strategy_bm25_index.read().await;
        let ranked = crate::retrieval::StrategyRetrievalPipeline::retrieve(
            &candidates,
            None,
            &query,
            &config,
            Some(&*bm25),
            ppr_scores.as_ref(),
            Some(&strategy_to_node),
        );

        // Resolve IDs to Strategy objects
        let limit = query.limit;
        let store = self.strategy_store.read().await;
        let mut results = Vec::with_capacity(ranked.len().min(limit));
        for (strategy_id, _score) in ranked.into_iter().take(limit) {
            if let Some(strat) = store.get_strategy(strategy_id) {
                results.push(strat);
            }
        }
        results
    }

    /// Backfill pending memory summary embeddings.
    ///
    /// Scans up to `batch_size` memories whose `has_summary_embedding` flag
    /// is `false`, embeds the existing summary text in one batch RPC, then
    /// upserts the points to `vectors.memories` and flips the flags. The
    /// LLM refinement step is intentionally skipped — refinement runs on
    /// new episodes; this path exists to catch memories whose vector was
    /// lost (e.g. after an export/import upgrade between Qdrant clusters).
    ///
    /// Returns the number of memories that successfully got vectors. A
    /// returned value smaller than `batch_size` plus an empty
    /// candidate list (see [`Self::has_pending_memory_embeddings`]) means
    /// the backfill is drained.
    pub async fn process_pending_memory_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<usize, crate::GraphError> {
        use minns_vectors::{Payload, Point};

        let embedding_client = match &self.embedding_client {
            Some(c) => c.clone(),
            None => {
                return Err(crate::GraphError::InvalidOperation(
                    "Embedding client not initialized".to_string(),
                ));
            },
        };

        // Snapshot candidates under a read lock.
        let candidates: Vec<Memory> = {
            let store = self.memory_store.read().await;
            store
                .list_all_memories()
                .into_iter()
                .filter(|m| !m.has_summary_embedding && !m.summary.trim().is_empty())
                .take(batch_size)
                .collect()
        };

        if candidates.is_empty() {
            return Ok(0);
        }

        // One batch embedding RPC for the whole batch.
        let requests: Vec<crate::claims::EmbeddingRequest> = candidates
            .iter()
            .map(|m| crate::claims::EmbeddingRequest {
                text: m.summary.clone(),
                context: None,
            })
            .collect();

        let responses = embedding_client
            .embed_batch(requests)
            .await
            .map_err(|e| crate::GraphError::OperationError(format!("Batch embed failed: {}", e)))?;

        // Build the point batch. Anything with an empty embedding is
        // skipped — those memories stay in the pending set.
        let mut points = Vec::with_capacity(responses.len());
        let mut ids_to_flag: Vec<crate::memory::MemoryId> = Vec::with_capacity(responses.len());
        for (memory, response) in candidates.iter().zip(responses.into_iter()) {
            if response.embedding.is_empty() {
                continue;
            }
            points.push(Point::new(
                memory.id as u128,
                response.embedding,
                Payload::EMPTY,
            ));
            ids_to_flag.push(memory.id);
        }

        if points.is_empty() {
            return Ok(0);
        }

        // Upsert first. On failure no flag flips; the next pass retries.
        self.vectors.memories.upsert(points).await.map_err(|e| {
            crate::GraphError::OperationError(format!("Memory vector batch upsert failed: {}", e))
        })?;

        // Flip flags. Each iteration is a tiny redb write; consider
        // batching if this becomes a bottleneck on very large backfills.
        let mut store = self.memory_store.write().await;
        let mut updated = 0usize;
        for mid in &ids_to_flag {
            if let Some(mut memory) = store.get_memory(*mid) {
                memory.has_summary_embedding = true;
                memory.context.embeddings = None;
                store.store_consolidated_memory(memory);
                updated += 1;
            }
        }

        Ok(updated)
    }

    /// Whether any memory has `has_summary_embedding == false`. Used by
    /// the post-upgrade backfill loop to decide when to stop polling.
    pub async fn has_pending_memory_embeddings(&self) -> bool {
        let store = self.memory_store.read().await;
        store
            .list_all_memories()
            .iter()
            .any(|m| !m.has_summary_embedding && !m.summary.trim().is_empty())
    }
}
