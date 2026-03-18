// crates/agent-db-graph/src/integration/queries/accessors.rs
//
// Simple getters, setters, and utility methods on GraphEngine:
// claim_store, execute_query, get_graph_stats, scoped inference,
// reinforcement stats, decay, recent events, structured memory accessors,
// conversation summaries, buffer flush, community summaries, and participants.

use super::*;

impl GraphEngine {
    /// Get reference to claim store (if semantic memory is enabled)
    pub fn claim_store(&self) -> Option<&Arc<crate::claims::ClaimStore>> {
        self.claim_store.as_ref()
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

    /// Get a reference to the structured memory store (for server handlers).
    pub fn structured_memory(
        &self,
    ) -> &Arc<RwLock<crate::structured_memory::StructuredMemoryStore>> {
        &self.structured_memory
    }

    /// Get a reference to the inference engine (for graph access in server handlers).
    pub fn inference(&self) -> &Arc<RwLock<GraphInference>> {
        &self.inference
    }

    /// Get a reference to the ontology registry.
    pub fn ontology(&self) -> &Arc<crate::ontology::OntologyRegistry> {
        &self.ontology
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
        // Evict if too many conversation summaries tracked
        if summaries.len() > 1000 {
            let keys: Vec<String> = summaries
                .keys()
                .take(summaries.len() / 2)
                .cloned()
                .collect();
            for k in keys {
                summaries.remove(&k);
            }
        }
        summaries.insert(summary.case_id.clone(), summary);
    }

    /// Flush the message buffer for a case_id, running compaction on the
    /// buffered batch with the rolling summary as prior context.
    /// Returns `Some(CompactionResult)` if compaction ran, `None` if the
    /// buffer was empty or compaction is disabled.
    pub async fn flush_message_buffer(
        self: &std::sync::Arc<Self>,
        case_id: &str,
    ) -> Option<crate::conversation::compaction::CompactionResult> {
        if !self.config.enable_conversation_compaction {
            return None;
        }

        // Drain the buffer from conversation state
        let (messages, session_id) = {
            let mut states = self.conversation_states.lock().await;
            let state = states.get_mut(case_id)?;
            if state.message_buffer.is_empty() {
                return None;
            }
            let msgs = std::mem::take(&mut state.message_buffer);
            let sid = state.buffer_session_id.take().unwrap_or_default();
            state.buffer_first_timestamp = None;
            (msgs, sid)
        };

        tracing::info!(
            "Flushing message buffer for case_id={}: {} messages",
            case_id,
            messages.len(),
        );

        // Build a ConversationIngest from the buffered messages
        let ingest_data = crate::conversation::ConversationIngest {
            case_id: Some(case_id.to_string()),
            sessions: vec![crate::conversation::ConversationSession {
                session_id,
                topic: None,
                messages,
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
            group_id: Default::default(),
            metadata: Default::default(),
        };

        // Fetch rolling summary for prior context
        let prior_summary = self.conversation_summary(case_id).await;
        let prior_text = prior_summary.as_ref().map(|s| s.summary.as_str());

        // Run compaction with context
        let cr = crate::conversation::compaction::run_compaction_with_context(
            self,
            &ingest_data,
            case_id,
            prior_text,
        )
        .await;

        tracing::info!(
            "Buffer flush compaction case_id={}: facts={} goals={} playbooks={}",
            case_id,
            cr.facts_extracted,
            cr.goals_extracted,
            cr.playbooks_extracted,
        );

        Some(cr)
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
}
