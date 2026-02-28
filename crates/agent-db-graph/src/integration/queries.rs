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
        // Get read access to the graph through inference engine
        {
            let _inference = self.inference.read().await;
            // We need a way to get a reference to the graph
            // For now, we'll execute queries directly through traversal
        }

        let result = {
            let inference = self.inference.read().await;
            self.traversal.execute_query(inference.graph(), query)?
        };

        // Update query statistics
        self.stats
            .total_queries_executed
            .fetch_add(1, AtomicOrdering::Relaxed);

        Ok(result)
    }

    /// Get current graph statistics
    pub async fn get_graph_stats(&self) -> GraphStats {
        let inference = self.inference.read().await;
        inference.graph().stats().clone()
    }

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

        // Deduplicate by node_id, keeping highest score
        results.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        results.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = b.1.max(a.1); // keep highest score in b
                true
            } else {
                false
            }
        });

        // Re-sort by score descending and truncate to limit
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
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

        let mut suggestions = self.traversal.get_next_step_suggestions(
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

    /// Get the most recent events from the in-memory event store
    pub async fn get_recent_events(&self, limit: usize) -> Vec<Event> {
        let store = self.event_store.read().await;
        let mut events: Vec<Event> = store.values().cloned().collect();
        events.sort_by_key(|event| std::cmp::Reverse(event.timestamp));
        events.into_iter().take(limit).collect()
    }

    /// Get a reference to the structured memory store (for server handlers).
    pub fn structured_memory(
        &self,
    ) -> &Arc<RwLock<crate::structured_memory::StructuredMemoryStore>> {
        &self.structured_memory
    }

    /// Get a reference to the memory audit log.
    pub fn memory_audit_log(&self) -> &Arc<RwLock<crate::memory_audit::MemoryAuditLog>> {
        &self.memory_audit_log
    }

    /// Get a reference to the goal store.
    pub fn goal_store(&self) -> &Arc<RwLock<crate::goal_store::GoalStore>> {
        &self.goal_store
    }

    /// Get a reference to the persistent conversation states (for incremental ingestion).
    pub fn conversation_states(
        &self,
    ) -> &tokio::sync::Mutex<HashMap<String, crate::conversation::ConversationState>> {
        &self.conversation_states
    }

    /// Get a reference to the unified LLM client (for conversation handlers).
    pub fn unified_llm_client(&self) -> Option<&Arc<dyn crate::llm_client::LlmClient>> {
        self.unified_llm_client.as_ref()
    }

    /// Get the rolling conversation summary for a case_id (if one exists).
    pub async fn conversation_summary(
        &self,
        case_id: &str,
    ) -> Option<crate::conversation::compaction::ConversationRollingSummary> {
        let summaries = self.conversation_summaries.read().await;
        summaries.get(case_id).cloned()
    }

    /// Set (or update) the rolling conversation summary for a case_id.
    pub async fn set_conversation_summary(
        &self,
        summary: crate::conversation::compaction::ConversationRollingSummary,
    ) {
        let mut summaries = self.conversation_summaries.write().await;
        summaries.insert(summary.case_id.clone(), summary);
    }

    /// Get a snapshot of all community summaries.
    pub async fn community_summaries(
        &self,
    ) -> HashMap<u64, crate::community_summary::CommunitySummary> {
        self.community_summaries.read().await.clone()
    }

    /// Replace all community summaries.
    pub async fn set_community_summaries(
        &self,
        summaries: HashMap<u64, crate::community_summary::CommunitySummary>,
    ) {
        *self.community_summaries.write().await = summaries;
    }

    /// Pre-create Concept nodes for conversation participants.
    ///
    /// For each name: checks `concept_index`, creates a Concept node if missing
    /// (ConceptType::Person, confidence 0.8). Returns name→NodeId mapping.
    pub async fn ensure_conversation_participants(
        &self,
        participants: &[String],
    ) -> GraphResult<Vec<(String, NodeId)>> {
        use crate::structures::{ConceptType, GraphNode, NodeType};

        let mut inference = self.inference.write().await;
        let mut mappings = Vec::new();

        for name in participants {
            if name.is_empty() {
                continue;
            }
            if let Some(&existing_id) = inference.graph().concept_index.get(name.as_str()) {
                mappings.push((name.clone(), existing_id));
            } else {
                let node = GraphNode::new(NodeType::Concept {
                    concept_name: name.clone(),
                    concept_type: ConceptType::Person,
                    confidence: 0.8,
                });
                let node_id = inference.graph_mut().add_node(node)?;
                let interned = inference.graph_mut().interner.intern(name);
                inference
                    .graph_mut()
                    .concept_index
                    .insert(interned, node_id);
                mappings.push((name.clone(), node_id));
                tracing::debug!(
                    "Pre-created Concept node for participant '{}' (id={})",
                    name,
                    node_id
                );
            }
        }

        Ok(mappings)
    }

    /// Execute a natural language query against the graph.
    ///
    /// Translates the question into a `GraphQuery`, executes it, and returns
    /// a human-readable answer along with raw results and metadata.
    ///
    /// Supports pagination and conversational context via `session_id`.
    pub async fn natural_language_query(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        let start = std::time::Instant::now();

        // Resolve follow-up questions using conversational context
        let effective_question = if let Some(sid) = session_id {
            let contexts = self.nlq_contexts.lock().await;
            if let Some(ctx) = contexts.get(sid) {
                crate::nlq::resolve_followup(question, ctx).unwrap_or_else(|| question.to_string())
            } else {
                question.to_string()
            }
        } else {
            question.to_string()
        };

        // LLM hint classifier: run rule-based + LLM in parallel, merge results
        let intent_override = if self.config.enable_nlq_hint {
            if let Some(ref llm_client) = self.unified_llm_client {
                let rule_based = crate::nlq::intent::classify_intent_full(&effective_question);
                let llm_result = tokio::time::timeout(
                    Duration::from_secs(5),
                    crate::nlq::llm_hint::classify_with_llm(
                        llm_client.as_ref(),
                        &effective_question,
                    ),
                )
                .await;
                let llm_hint = match llm_result {
                    Ok(Ok(hint)) => hint,
                    Ok(Err(e)) => {
                        tracing::warn!("NLQ hint classifier error: {}", e);
                        None
                    },
                    Err(_) => {
                        tracing::warn!("NLQ hint classifier timed out (5s)");
                        None
                    },
                };
                let merged = crate::nlq::llm_hint::merge_classification(rule_based, llm_hint);
                if merged.llm_overrode {
                    tracing::info!(
                        "NLQ hint classifier overrode rule-based intent to {:?}",
                        crate::nlq::intent::intent_display_name(&merged.intent.intent),
                    );
                }
                Some(merged.intent)
            } else {
                None
            }
        } else {
            None
        };

        // Check if this should route through the unified pipeline
        let is_full_pipeline = self.config.enable_unified_nlq
            && match &intent_override {
                Some(ci) => matches!(
                    ci.intent,
                    crate::nlq::intent::QueryIntent::KnowledgeQuery
                        | crate::nlq::intent::QueryIntent::SimilaritySearch
                        | crate::nlq::intent::QueryIntent::Unknown
                ),
                None => {
                    // Also check rule-based classification for routing
                    let probe = crate::nlq::intent::classify_intent(&effective_question);
                    matches!(
                        probe,
                        crate::nlq::intent::QueryIntent::KnowledgeQuery
                            | crate::nlq::intent::QueryIntent::Unknown
                    )
                },
            };

        if is_full_pipeline {
            return self
                .execute_unified_query(&effective_question, pagination, session_id)
                .await;
        }

        let response = {
            let inference = self.inference.read().await;
            let sm_guard = self.structured_memory.read().await;
            self.nlq_pipeline.execute_with_hint(
                &effective_question,
                inference.graph(),
                &self.traversal,
                pagination,
                Some(&sm_guard),
                intent_override,
            )?
        };

        // Push exchange to conversational context
        if let Some(sid) = session_id {
            let mut contexts = self.nlq_contexts.lock().await;
            let ctx = contexts
                .entry(sid.to_string())
                .or_insert_with(crate::nlq::ConversationContext::new);
            ctx.push(crate::nlq::ConversationExchange {
                question: effective_question.clone(),
                intent: crate::nlq::intent::intent_display_name(&response.intent).to_string(),
                entities: response
                    .entities_resolved
                    .iter()
                    .map(|e| e.mention.text.clone())
                    .collect(),
                timestamp: crate::nlq::now_millis(),
            });

            // LRU eviction: cap at 1000 sessions to prevent unbounded growth
            const MAX_NLQ_SESSIONS: usize = 1000;
            if contexts.len() > MAX_NLQ_SESSIONS {
                if let Some(oldest_key) = contexts
                    .iter()
                    .min_by_key(|(_, ctx)| ctx.last_activity())
                    .map(|(k, _)| k.clone())
                {
                    contexts.remove(&oldest_key);
                }
            }
        }

        // Log feedback
        let feedback = crate::nlq::feedback::NlqFeedback {
            question: question.to_string(),
            intent: crate::nlq::intent::intent_display_name(&response.intent).to_string(),
            entities_found: response.entities_resolved.len(),
            template_used: Some(format!("{:?}", response.query_used)),
            query_built: true,
            validation_result: "Valid".to_string(),
            execution_success: true,
            result_count: crate::nlq::formatter::result_count(&response.result),
            confidence: response.confidence,
            execution_time_ms: start.elapsed().as_millis() as u64,
        };
        crate::nlq::feedback::log_nlq_feedback(&feedback);

        Ok(response)
    }

    /// Execute a unified multi-source query (BM25 + memory + claims + graph + optional DRIFT).
    ///
    /// Triggered for KnowledgeQuery, SimilaritySearch, and Unknown intents when
    /// `enable_unified_nlq` is true.
    async fn execute_unified_query(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        let start = std::time::Instant::now();
        let mut explanation = vec!["Unified NLQ pipeline activated".to_string()];
        let mut ranked_lists: Vec<Vec<(u64, f32)>> = Vec::new();

        // 1. BM25 search (always available)
        let bm25 = self.search_bm25(question, 20).await;
        if !bm25.is_empty() {
            explanation.push(format!("BM25: {} results", bm25.len()));
            ranked_lists.push(bm25);
        }

        // Generate query embedding if embedding client is available (reused for memory + claims)
        let query_embedding = if let Some(ref ec) = self.embedding_client {
            match ec
                .embed(crate::claims::EmbeddingRequest {
                    text: question.to_string(),
                    context: None,
                })
                .await
            {
                Ok(resp) => resp.embedding,
                Err(_) => vec![], // fail-open: fall back to BM25-only
            }
        } else {
            vec![]
        };

        // 2. Memory retrieval (with semantic signal when embedding is available)
        let memories = self
            .retrieve_memories_multi_signal(
                crate::retrieval::MemoryRetrievalQuery {
                    query_text: question.to_string(),
                    query_embedding: query_embedding.clone(),
                    context: None,
                    anchor_node: None,
                    agent_id: None,
                    session_id: None,
                    now: None,
                    limit: 20,
                },
                None,
            )
            .await;
        if !memories.is_empty() {
            explanation.push(format!("Memory retrieval: {} results", memories.len()));
            // Convert to ranked list using position-based scoring
            let memory_ranked: Vec<(u64, f32)> = {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                memories
                    .iter()
                    .enumerate()
                    .map(|(rank, mem)| {
                        let node_id = graph.memory_index.get(&mem.id).copied().unwrap_or(mem.id);
                        (node_id, 1.0 / (rank as f32 + 1.0))
                    })
                    .collect()
            };
            ranked_lists.push(memory_ranked);
        }

        // 3. Claim search (hybrid BM25 + semantic via RRF, falls back to BM25-only if no embedding)
        if let Some(ref store) = self.claim_store {
            {
                let hybrid_config = if query_embedding.is_empty() {
                    // No embedding available — use keyword-only mode
                    crate::claims::hybrid_search::HybridSearchConfig {
                        mode: crate::indexing::SearchMode::Keyword,
                        ..Default::default()
                    }
                } else {
                    crate::claims::hybrid_search::HybridSearchConfig::default()
                };
                if let Ok(claims) = crate::claims::hybrid_search::HybridClaimSearch::search(
                    question,
                    &query_embedding,
                    store,
                    20,
                    &hybrid_config,
                ) {
                    if !claims.is_empty() {
                        // Convert claim IDs to node IDs
                        let claim_node_ids: Vec<(u64, f32)> = {
                            let inference = self.inference.read().await;
                            let graph = inference.graph();
                            claims
                                .iter()
                                .filter_map(|&(claim_id, score)| {
                                    graph.claim_index.get(&claim_id).map(|&nid| (nid, score))
                                })
                                .collect()
                        };
                        if !claim_node_ids.is_empty() {
                            explanation
                                .push(format!("Claims (hybrid): {} results", claim_node_ids.len()));
                            ranked_lists.push(claim_node_ids);
                        }
                    }
                }
            }
        }

        // 4. Graph entity BFS (resolve entities → 1-hop neighbors)
        {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            let nlq_pipeline = &self.nlq_pipeline;
            let intent = crate::nlq::intent::QueryIntent::KnowledgeQuery;
            let mentions = nlq_pipeline
                .entity_resolver()
                .extract_mentions(question, &intent);
            let resolved = nlq_pipeline.entity_resolver().resolve(&mentions, graph);
            if !resolved.is_empty() {
                let mut entity_hits: Vec<(u64, f32)> = Vec::new();
                for entity in &resolved {
                    entity_hits.push((entity.node_id, 1.0));
                    // 1-hop neighbors (both directions: outgoing + incoming)
                    // Incoming edges capture claims linked via ABOUT edges (Claim→Concept)
                    for &neighbor_id in graph
                        .neighbors_directed(entity.node_id, crate::structures::Direction::Both)
                        .iter()
                    {
                        entity_hits.push((neighbor_id, 0.5));
                    }
                }
                if !entity_hits.is_empty() {
                    explanation.push(format!(
                        "Entity resolution: {} entities, {} neighbors",
                        resolved.len(),
                        entity_hits.len() - resolved.len()
                    ));
                    ranked_lists.push(entity_hits);
                }
            }
        }

        // 5. Fuse via RRF
        let fused = crate::retrieval::multi_list_rrf(&ranked_lists, 60.0);

        // 6. Format answer from top results
        let answer = {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            crate::nlq::unified::format_unified_results(&fused, question, graph, &memories)
        };

        // 7. Optionally run DRIFT for synthesized answer
        // Clone summaries snapshot to avoid holding read lock during LLM calls.
        let (final_answer, drift_explanation) = if self.config.enable_drift_search {
            if let Some(ref llm_client) = self.unified_llm_client {
                let summaries_snapshot = {
                    let guard = self.community_summaries.read().await;
                    if guard.is_empty() {
                        None
                    } else {
                        Some(guard.clone())
                    }
                }; // read lock released here
                if let Some(summaries) = summaries_snapshot {
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(self.config.drift_config.timeout_secs),
                        self.run_drift_search(question, &summaries, &fused, llm_client.as_ref()),
                    )
                    .await
                    {
                        Ok(drift_result) => {
                            let explain = format!(
                                "DRIFT: {} communities, {} follow-ups, {} items",
                                drift_result.primer_communities_used.len(),
                                drift_result.followup_queries.len(),
                                drift_result.total_items_retrieved,
                            );
                            (drift_result.answer, Some(explain))
                        },
                        Err(_) => {
                            tracing::warn!("DRIFT search timed out, using standard results");
                            (answer, None)
                        },
                    }
                } else {
                    (answer, None)
                }
            } else {
                (answer, None)
            }
        } else {
            (answer, None)
        };

        if let Some(drift_exp) = drift_explanation {
            explanation.push(drift_exp);
        }

        // Apply pagination to fused results
        let limit = pagination.limit.unwrap_or(20);
        let offset = pagination.offset.unwrap_or(0);
        let total_count = fused.len();
        let paginated: Vec<(u64, f32)> = fused.into_iter().skip(offset).take(limit).collect();

        // Push exchange to conversational context
        if let Some(sid) = session_id {
            let mut contexts = self.nlq_contexts.lock().await;
            let ctx = contexts
                .entry(sid.to_string())
                .or_insert_with(crate::nlq::ConversationContext::new);
            ctx.push(crate::nlq::ConversationExchange {
                question: question.to_string(),
                intent: "KnowledgeQuery".to_string(),
                entities: vec![],
                timestamp: crate::nlq::now_millis(),
            });
        }

        Ok(crate::nlq::NlqResponse {
            answer: final_answer,
            query_used: crate::traversal::GraphQuery::PageRank {
                iterations: 0,
                damping_factor: 0.0,
            },
            result: crate::traversal::QueryResult::Rankings(paginated),
            intent: crate::nlq::intent::QueryIntent::KnowledgeQuery,
            confidence: 0.8,
            execution_time_ms: start.elapsed().as_millis() as u64,
            explanation,
            total_count,
            entities_resolved: vec![],
        })
    }

    /// Run DRIFT search: primer → follow-up → synthesis.
    async fn run_drift_search(
        &self,
        question: &str,
        community_summaries: &HashMap<u64, crate::community_summary::CommunitySummary>,
        initial_fused: &[(u64, f32)],
        llm_client: &dyn crate::llm_client::LlmClient,
    ) -> crate::nlq::drift::DriftResult {
        let config = &self.config.drift_config;

        // Phase 1: Primer — score communities + generate follow-up queries
        let (primer_communities, followup_queries) =
            crate::nlq::drift::drift_primer(llm_client, question, community_summaries, config)
                .await;

        // Phase 2: Follow-up — execute each query and collect results
        let mut results_per_query = Vec::new();
        for q in &followup_queries {
            let bm25 = self.search_bm25(q, config.max_followup_results).await;
            results_per_query.push(bm25);
        }
        let merged = crate::nlq::drift::drift_followup_merge(&results_per_query);

        // Collect text snippets for synthesis
        let mut community_context = Vec::new();
        for &cid in &primer_communities {
            if let Some(summary) = community_summaries.get(&cid) {
                community_context.push(format!(
                    "{} (entities: {})",
                    summary.summary,
                    summary.key_entities.join(", ")
                ));
            }
        }

        let mut retrieved_snippets = Vec::new();
        {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            let mut seen = std::collections::HashSet::new();
            // Add snippets from initial fused + DRIFT follow-up results
            let combined_iter = initial_fused.iter().take(10).chain(merged.iter().take(10));
            for &(node_id, _score) in combined_iter {
                let label = graph
                    .get_node(node_id)
                    .map(|n| n.label())
                    .unwrap_or_else(|| format!("Node {}", node_id));
                if seen.insert(label.clone()) {
                    retrieved_snippets.push(label);
                }
            }
        }

        let total_items = merged.len();

        // Phase 3: Synthesis
        let answer = crate::nlq::drift::drift_synthesis(
            llm_client,
            question,
            &community_context,
            &retrieved_snippets,
            config,
        )
        .await;

        crate::nlq::drift::DriftResult {
            answer,
            primer_communities_used: primer_communities,
            followup_queries,
            total_items_retrieved: total_items,
        }
    }
}
