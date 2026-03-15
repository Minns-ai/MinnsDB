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
        let goal_overrides = if let Some(ref llm) = self.unified_llm_client {
            let all_memories = {
                let store = self.memory_store.read().await;
                store.list_all_memories()
            };
            crate::consolidation::infer_goal_labels(llm.as_ref(), &all_memories).await
        } else {
            std::collections::HashMap::new()
        };

        let mut store = self.memory_store.write().await;
        let mut engine = self.consolidation_engine.write().await;
        engine.run_consolidation(store.as_mut(), &goal_overrides)
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

        // Run the pipeline
        let bm25 = self.memory_bm25_index.read().await;
        let ranked = crate::retrieval::MemoryRetrievalPipeline::retrieve(
            &candidates,
            &query,
            &config,
            Some(&*bm25),
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
}
