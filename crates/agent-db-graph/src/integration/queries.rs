use super::*;

impl GraphEngine {
    /// Get reference to claim store (if semantic memory is enabled)
    pub fn claim_store(&self) -> Option<&Arc<crate::claims::ClaimStore>> {
        self.claim_store.as_ref()
    }

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
        let mut store = self.memory_store.write().await;
        let mut engine = self.consolidation_engine.write().await;
        engine.run_consolidation(store.as_mut())
    }

    /// Execute a graph query
    pub async fn execute_query(&self, query: GraphQuery) -> GraphResult<QueryResult> {
        let start_time = std::time::Instant::now();

        // Get read access to the graph through inference engine
        {
            let _inference = self.inference.read().await;
            // We need a way to get a reference to the graph
            // For now, we'll execute queries directly through traversal
        }

        let result = {
            let inference = self.inference.read().await;
            let mut traversal = self.traversal.write().await;
            traversal.execute_query(inference.graph(), query)?
        };

        // Update query statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_queries_executed += 1;

            let _query_time = start_time.elapsed().as_millis() as f64;
            // Would update query timing statistics here
        }

        Ok(result)
    }

    /// Get current graph statistics
    pub async fn get_graph_stats(&self) -> GraphStats {
        let inference = self.inference.read().await;
        inference.graph().stats().clone()
    }

    /// Search nodes using BM25 full-text search
    ///
    /// Returns a list of (NodeId, score) tuples ranked by relevance
    pub async fn search_bm25(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
        let inference = self.inference.read().await;
        inference.graph().bm25_index.search(query, limit)
    }

    /// Get a node by ID for search results
    pub async fn get_node(&self, node_id: u64) -> Option<crate::structures::GraphNode> {
        let inference = self.inference.read().await;
        inference.graph().nodes.get(&node_id).cloned()
    }

    /// Search claims by semantic similarity for hybrid search
    ///
    /// Returns a list of (NodeId, score) tuples for claims ranked by semantic similarity
    pub async fn search_claims_semantic(
        &self,
        query: &str,
        limit: usize,
        min_similarity: f32,
    ) -> crate::GraphResult<Vec<(u64, f32)>> {
        // Check if embedding client is available
        let embedding_client = match &self.embedding_client {
            Some(c) => c,
            None => return Ok(vec![]), // No semantic search available
        };

        let claim_store = match &self.claim_store {
            Some(s) => s,
            None => return Ok(vec![]), // No claim store available
        };

        // Generate embedding for query
        let request = crate::claims::EmbeddingRequest {
            text: query.to_string(),
            context: None,
        };

        let response = embedding_client.embed(request).await.map_err(|e| {
            crate::GraphError::OperationError(format!("Failed to generate query embedding: {}", e))
        })?;

        // Search for similar claims
        let similar_claims = claim_store
            .find_similar(&response.embedding, limit, min_similarity)
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

    /// Get scoped inference statistics
    pub async fn get_scoped_inference_stats(&self) -> crate::scoped_inference::ScopeStatistics {
        self.scoped_inference.get_scope_statistics().await
    }

    /// Query events in a specific scope
    pub async fn query_events_in_scope(
        &self,
        scope: &crate::scoped_inference::InferenceScope,
        query: crate::scoped_inference::ScopeQuery,
    ) -> GraphResult<crate::scoped_inference::ScopeQueryResult> {
        self.scoped_inference.query_scope(scope, query).await
    }

    /// Get cross-scope relationships
    pub async fn get_cross_scope_relationships(
        &self,
    ) -> crate::scoped_inference::CrossScopeInsights {
        self.scoped_inference.get_cross_scope_insights().await
    }

    /// Get policy guide suggestions for what action to take next
    ///
    /// **Returns:** Action suggestions ranked by success probability and centrality
    pub async fn get_next_action_suggestions(
        &self,
        context_hash: ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let inference = self.inference.read().await;
        let graph = inference.graph();
        let traversal = self.traversal.read().await;

        let mut suggestions = traversal.get_next_step_suggestions(
            graph,
            context_hash,
            last_action_node,
            limit * 2,
        )?;

        // Calculate centrality scores for ranking
        let centrality_scores = self.centrality.all_centralities(graph)?;

        // Re-rank suggestions using combined score: success_probability * centrality
        for suggestion in &mut suggestions {
            let combined_score = centrality_scores.combined_score(suggestion.action_node_id);

            // Blend success probability (60%) with centrality importance (40%)
            let original_prob = suggestion.success_probability;
            suggestion.success_probability = (original_prob * 0.6) + (combined_score * 0.4);
        }

        // Re-sort by updated success probability
        suggestions.sort_by(|a, b| {
            b.success_probability
                .partial_cmp(&a.success_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // PPR-based proximity boost: nodes closer to last action get a small bonus
        if let Some(source_node) = last_action_node {
            if let Ok(ppr_scores) = self.random_walker.personalized_pagerank(graph, source_node) {
                // Rank-based normalization: sort PPR scores for the candidate set,
                // map to [0.0, 1.0] rank percentiles
                let n = suggestions.len();
                if n > 0 {
                    let mut indexed_scores: Vec<(usize, f64)> = suggestions
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let score = ppr_scores.get(&s.action_node_id).copied().unwrap_or(0.0);
                            (i, score)
                        })
                        .collect();
                    indexed_scores
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut rank_scores = vec![0.0f64; n];
                    for (rank, &(idx, _)) in indexed_scores.iter().enumerate() {
                        rank_scores[idx] = rank as f64 / (n.max(2) - 1) as f64;
                    }

                    for (i, suggestion) in suggestions.iter_mut().enumerate() {
                        suggestion.success_probability =
                            suggestion.success_probability * 0.9 + rank_scores[i] as f32 * 0.1;
                    }

                    suggestions.sort_by(|a, b| {
                        b.success_probability
                            .partial_cmp(&a.success_probability)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // World model reranking (ScoringAndReranking or Full mode only)
        if matches!(
            self.config.effective_world_model_mode(),
            WorldModelMode::ScoringAndReranking | WorldModelMode::Full
        ) {
            if let Some(ref wm) = self.world_model {
                let wm_guard = wm.read().await;
                if wm_guard.energy_stats().is_warmed_up {
                    let policy = agent_db_world_model::PolicyFeatures {
                        goal_count: 1,
                        top_goal_priority: 0.8,
                        resource_cpu_percent: 0.0,
                        resource_memory_bytes: 0,
                        context_fingerprint: context_hash,
                    };
                    let strategy = world_model::extract_strategy_features(None);
                    let report = wm_guard.score_strategy(&policy, &strategy);
                    let wm_score = (-report.total_energy).clamp(0.0, 1.0);

                    for suggestion in &mut suggestions {
                        // Blend: 80% existing score + 20% world model compatibility
                        suggestion.success_probability =
                            suggestion.success_probability * 0.8 + wm_score * 0.2;
                    }

                    // Re-sort after blending
                    suggestions.sort_by(|a, b| {
                        b.success_probability
                            .partial_cmp(&a.success_probability)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // Return top-k after centrality ranking
        Ok(suggestions.into_iter().take(limit).collect())
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

    /// Get reinforcement learning statistics
    pub async fn get_reinforcement_stats(&self) -> crate::inference::ReinforcementStats {
        self.inference.read().await.get_reinforcement_stats()
    }

    /// Manually update strategy outcome (for external feedback)
    pub async fn update_strategy_outcome(
        &self,
        strategy_id: StrategyId,
        success: bool,
    ) -> GraphResult<()> {
        self.strategy_store
            .write()
            .await
            .update_strategy_outcome(strategy_id, success)
    }

    /// Force memory decay (for testing or periodic cleanup)
    pub async fn decay_memories(&self) {
        self.memory_store.write().await.apply_decay();
    }

    /// Get the most recent events from the in-memory event store
    pub async fn get_recent_events(&self, limit: usize) -> Vec<Event> {
        let store = self.event_store.read().await;
        let mut events: Vec<Event> = store.values().cloned().collect();
        events.sort_by_key(|event| std::cmp::Reverse(event.timestamp));
        events.into_iter().take(limit).collect()
    }
}
