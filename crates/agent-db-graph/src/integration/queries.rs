use super::*;

// ────────── Answer Synthesis ──────────

const SYNTHESIS_SYSTEM_PROMPT: &str = r#"You are answering a question about a user based on stored knowledge. Answer the question directly and concisely using ONLY the information provided in the context.

Rules:
- Give a direct, focused answer — not a list of facts
- Use only information present in the context
- The "Current state" section contains AUTHORITATIVE facts about what is true RIGHT NOW
- Current state facts (location, routine, preferences, relationships) are ALWAYS sufficient to answer questions about the user's current life, activities, and recommendations
- When asked "what should I do", "what to do this weekend", etc., use the user's current routine and location to answer directly
- Write naturally, as if you personally know the user
- Do NOT add speculation or information not in the context
- Only say "I don't have enough information" if the context has NO relevant facts at all
- For preference/recommendation questions: Compare categories, count positive vs negative, identify patterns, and make a clear recommendation with reasoning. Cite specific items.
- For "what do X have in common" questions: Identify shared themes across items.

CRITICAL — TEMPORAL STATE RULES:
- The "Current state" section is the AUTHORITATIVE ground truth. It shows what is true RIGHT NOW.
- If the current state says "location: Vancouver", the user is in Vancouver. Period.
- ALL other facts must be CONSISTENT with the current state. Discard any known facts about activities, routines, landmarks, or places from a DIFFERENT or superseded state.
- NEVER reference activities, places, or routines from superseded locations/states unless the question explicitly asks about history."#;

/// Use the LLM to synthesize a focused answer from retrieved context.
///
/// Takes the question and the raw retrieval context (facts, claims, entity state)
/// and produces a direct, concise answer instead of a bullet-point dump.
async fn synthesize_answer(
    llm: &dyn crate::llm_client::LlmClient,
    question: &str,
    context: &str,
) -> anyhow::Result<String> {
    let user_prompt = format!("Context:\n{}\n\nQuestion: {}", context, question);
    tracing::info!("NLQ synthesis context:\n{}", context);

    let request = crate::llm_client::LlmRequest {
        system_prompt: SYNTHESIS_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 256,
        json_mode: false,
    };

    let response = llm.complete(request).await?;
    let answer = response.content.trim().to_string();
    if answer.is_empty() {
        anyhow::bail!("Empty LLM response");
    }
    Ok(answer)
}

/// Temporal validity filter: remove results linked to superseded `state:*` edges.
///
/// Returns the set of superseded target names for downstream filtering.
///
/// For multi-transition scenarios (2tr, 3tr, 4tr+), this is the critical gate
/// that prevents historical facts from polluting the synthesis context.
/// Superseded results are zeroed out (removed), not just soft-demoted.
fn apply_temporal_validity_filter(
    results: &mut Vec<(u64, f32)>,
    graph: &crate::structures::Graph,
) -> std::collections::HashSet<String> {
    use crate::conversation::graph_projection::concept_name_of;
    use crate::structures::{EdgeType, NodeType};

    // Phase 1: Collect superseded state target names.
    let mut superseded_targets: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    // Any structured "category:predicate" edge is potentially stateful.
    // The LLM generates arbitrary categories so we accept all of them —
    // supersession (valid_until) already distinguishes current vs old.
    let is_stateful = |assoc: &str| -> bool { assoc.contains(':') };

    for (_eid, edge) in graph.edges.iter() {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if is_stateful(association_type) && edge.valid_until.is_some() {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    superseded_targets.insert(name.to_lowercase());
                }
            }
        }
    }

    // Phase 1d: depends_on cascade — edges whose depends_on property references
    // a superseded target should also have their targets added to superseded_targets.
    // E.g., edge with depends_on: "User lives in Tokyo" → if "tokyo" is in
    // superseded_targets, add this edge's target too.
    if !superseded_targets.is_empty() {
        let mut dep_cascade_targets: Vec<String> = Vec::new();
        for (_eid, edge) in graph.edges.iter() {
            if edge.valid_until.is_some() {
                continue; // already superseded
            }
            if let Some(serde_json::Value::String(dep)) = edge.properties.get("depends_on") {
                let dep_lower = dep.to_lowercase();
                let is_dep_stale = superseded_targets
                    .iter()
                    .any(|t| dep_lower.contains(t.as_str()));
                if is_dep_stale {
                    if let Some(name) = concept_name_of(graph, edge.target) {
                        dep_cascade_targets.push(name.to_lowercase());
                    }
                }
            }
        }
        for t in dep_cascade_targets {
            superseded_targets.insert(t);
        }
    }

    // Phase 1c: Also collect current targets so we never accidentally filter them
    let mut current_targets: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_eid, edge) in graph.edges.iter() {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if is_stateful(association_type) && edge.valid_until.is_none() {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    current_targets.insert(name.to_lowercase());
                }
            }
        }
    }
    // Remove anything that's also a current target (safety: don't filter current state)
    for ct in &current_targets {
        superseded_targets.remove(ct);
    }

    // Phase 2: Zero out nodes that are direct targets of superseded state edges
    for (node_id, score) in results.iter_mut() {
        for edge in graph.get_edges_to(*node_id) {
            if let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            {
                if association_type.starts_with("state:") {
                    if edge.valid_until.is_some() {
                        *score = 0.0; // Hard remove — superseded state target
                        break;
                    }
                    // Check if a newer edge exists (implicit supersession)
                    let edge_vf = edge.valid_from.unwrap_or(0);
                    let source = edge.source;
                    let assoc = association_type.clone();

                    let is_superseded = graph.get_edges_from(source).iter().any(|other| {
                        if let EdgeType::Association {
                            association_type: ref at,
                            ..
                        } = other.edge_type
                        {
                            at == &assoc
                                && other.target != *node_id
                                && other.valid_from.unwrap_or(0) > edge_vf
                        } else {
                            false
                        }
                    });

                    if is_superseded {
                        *score = 0.0; // Hard remove
                        break;
                    }
                }
            }
        }
    }

    // Phase 3: Zero out Claim nodes that mention superseded targets
    if !superseded_targets.is_empty() {
        for (node_id, score) in results.iter_mut() {
            if *score == 0.0 {
                continue; // Already removed
            }
            if let Some(n) = graph.get_node(*node_id) {
                if let NodeType::Claim { claim_text, .. } = &n.node_type {
                    let text_lower = claim_text.to_lowercase();
                    for target in &superseded_targets {
                        if target.len() >= 3 && text_lower.contains(target.as_str()) {
                            *score = 0.0; // Hard remove — references superseded entity
                            break;
                        }
                    }
                }
            }
        }
    }

    // Remove zeroed entries and re-sort
    results.retain(|(_, score)| *score > 0.0);
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    superseded_targets
}

/// Filter claims whose state_anchor metadata disagrees with the current projected state.
///
/// Claims ingested while the user was in Amsterdam will have
/// `state_anchor:location:lives_in = Amsterdam`. If the current state is Vancouver,
/// those claims are zeroed out. Claims without state anchors pass through.
fn apply_state_anchor_filter(
    results: &mut Vec<(u64, f32)>,
    graph: &crate::structures::Graph,
    claim_store: &crate::claims::ClaimStore,
    ontology: &crate::ontology::OntologyRegistry,
) {
    use crate::conversation::graph_projection;
    use crate::structures::NodeType;

    // Project current state for "user" entity (the primary entity in personal knowledge graphs)
    let projected = graph_projection::project_entity_state(graph, "user", u64::MAX, Some(ontology));
    if projected.slots.is_empty() {
        return; // No state to filter against
    }

    // Build map of current state values: "location:lives_in" → "Vancouver"
    let mut current_state: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for slot in projected.slots.values() {
        let category = slot.association_type.split(':').next().unwrap_or("");
        if ontology.is_single_valued(category) {
            let val = slot.value.as_deref().unwrap_or(&slot.target_name);
            current_state.insert(
                format!("state_anchor:{}", slot.association_type),
                val.to_lowercase(),
            );
        }
    }

    if current_state.is_empty() {
        return;
    }

    for (node_id, score) in results.iter_mut() {
        if *score == 0.0 {
            continue;
        }
        // Only filter Claim nodes
        let claim_id = if let Some(node) = graph.get_node(*node_id) {
            if let NodeType::Claim { claim_id, .. } = &node.node_type {
                *claim_id
            } else {
                continue;
            }
        } else {
            continue;
        };

        // Look up claim metadata from store
        if let Ok(Some(claim)) = claim_store.get(claim_id) {
            for (anchor_key, anchor_val) in &claim.metadata {
                if !anchor_key.starts_with("state_anchor:") {
                    continue;
                }
                // Check if current state has a different value for this anchor
                if let Some(current_val) = current_state.get(anchor_key) {
                    if *current_val != anchor_val.to_lowercase() {
                        // Claim was anchored to a different state — filter it out
                        tracing::debug!(
                            "State anchor filter: claim {} anchored to {}={}, current={}",
                            claim_id,
                            anchor_key,
                            anchor_val,
                            current_val
                        );
                        *score = 0.0;
                        break;
                    }
                }
            }
        }
    }

    // Remove zeroed entries
    results.retain(|(_, score)| *score > 0.0);
}

/// Build a dynamic projection from graph edges based on LLM classification.
///
/// Maps LLM hints to `ConversationQueryType` variants and delegates to the
/// unified conversation query path (`execute_conversation_query_with_graph`).
/// This ensures a single, complete execution path for all structured queries.
fn build_dynamic_projection(
    graph: &crate::structures::Graph,
    hint: &crate::nlq::llm_hint::LlmHintResponse,
    question: &str,
    store: &crate::structured_memory::StructuredMemoryStore,
    name_registry: &crate::conversation::types::NameRegistry,
) -> Option<String> {
    use crate::conversation::nlq_ext::{
        execute_conversation_query_with_graph, ConversationQueryType, NumericOp,
    };
    use crate::nlq::llm_hint::{IntentHint, StructureHint};

    let entities = extract_entity_names_from_question(graph, question);
    let primary_entity = entities.first().cloned();

    let query_type = match (&hint.structure_hint, &hint.intent_hint) {
        (StructureHint::Ledger, IntentHint::Balance | IntentHint::AggregateBalance) => {
            ConversationQueryType::Numeric {
                op: NumericOp::NetBalance,
            }
        },
        (StructureHint::StateMachine, IntentHint::CurrentState) => ConversationQueryType::State {
            entity: primary_entity,
            attribute: None,
        },
        (StructureHint::PreferenceList, IntentHint::Ranking) => ConversationQueryType::Preference {
            entity: primary_entity,
            category: None,
        },
        (_, IntentHint::Path) if entities.len() >= 2 => ConversationQueryType::RelationshipPath {
            from: entities[0].clone(),
            to: entities[1].clone(),
            relation: None,
        },
        (StructureHint::GenericGraph, IntentHint::Neighbors | IntentHint::Subgraph) => {
            match primary_entity {
                Some(e) => ConversationQueryType::EntitySummary { entity: e },
                None => return None,
            }
        },
        _ => return None,
    };

    let result = execute_conversation_query_with_graph(
        &query_type,
        store,
        name_registry,
        question,
        Some(graph),
    );
    if result.is_empty() || result.contains("No ") {
        None
    } else {
        Some(result)
    }
}

/// Extract entity names mentioned in a question by checking against graph concept index.
///
/// Also resolves implicit relational references (e.g., "my neighbor" → look for
/// `relationship:neighbor` edges from user → return the target entity name).
fn extract_entity_names_from_question(
    graph: &crate::structures::Graph,
    question: &str,
) -> Vec<String> {
    let lower = question.to_lowercase();
    let mut found = Vec::new();

    // Always include "user" for first-person queries
    let has_first_person = lower.contains(" i ")
        || lower.contains("my ")
        || lower.starts_with("i ")
        || lower.contains(" me");
    if has_first_person {
        found.push("user".to_string());
    }

    // Check all concept names against the question
    for concept_name in graph.concept_index.keys() {
        if lower.contains(&concept_name.to_lowercase()) {
            found.push(concept_name.to_string());
        }
    }

    // Resolve implicit relational references from the "user" node.
    // E.g., "my neighbor" → look for `relationship:neighbor` edges from user.
    if has_first_person {
        if let Some(&user_nid) = graph.concept_index.get("user") {
            // Common relational words to check
            let relational_words = [
                "neighbor",
                "neighbour",
                "boss",
                "manager",
                "sister",
                "brother",
                "mother",
                "father",
                "friend",
                "partner",
                "colleague",
                "coworker",
                "wife",
                "husband",
                "child",
                "doctor",
                "teacher",
            ];
            for rel_word in &relational_words {
                if lower.contains(rel_word) {
                    // Look for relationship edges from user matching this word
                    for edge in graph.get_edges_from(user_nid) {
                        if let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &edge.edge_type
                        {
                            // Check if edge predicate (after ':') matches the relational word
                            let rel_suffix = association_type.split(':').nth(1).unwrap_or("");
                            if rel_suffix.to_lowercase().contains(rel_word) {
                                if let Some(name) =
                                    crate::conversation::graph_projection::concept_name_of(
                                        graph,
                                        edge.target,
                                    )
                                {
                                    if !found.contains(&name) {
                                        found.push(name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    found
}

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

        // Infer goal labels for goalless buckets if LLM is available
        let goal_overrides = if let Some(ref llm) = self.unified_llm_client {
            let all_memories = store.list_all_memories();
            crate::consolidation::infer_goal_labels(llm.as_ref(), &all_memories).await
        } else {
            std::collections::HashMap::new()
        };

        let mut engine = self.consolidation_engine.write().await;
        engine.run_consolidation(store.as_mut(), &goal_overrides)
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
                    indexed_scores.sort_by(|a, b| a.1.total_cmp(&b.1));

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

    /// Get a reference to the inference engine (for graph access in server handlers).
    pub fn inference(&self) -> &Arc<RwLock<GraphInference>> {
        &self.inference
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

    /// Execute a natural language query against the graph.
    ///
    /// Routes all queries through the unified multi-source pipeline (BM25 +
    /// memory + claims + graph entity resolution with RRF fusion).
    ///
    /// Supports pagination and conversational context via `session_id`.
    pub async fn natural_language_query(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        self.natural_language_query_with_options(question, pagination, session_id, false)
            .await
    }

    /// Execute a natural language query with options.
    ///
    /// When `include_memories` is true, memory retrieval is included in the
    /// fusion pipeline and memory summaries appear in the answer.
    pub async fn natural_language_query_with_options(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
        include_memories: bool,
    ) -> GraphResult<crate::nlq::NlqResponse> {
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

        self.execute_unified_query(
            &effective_question,
            pagination,
            session_id,
            include_memories,
        )
        .await
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
        include_memories: bool,
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

        // 1b. Node vector search (if embedding client available)
        // Generates query embedding early so it can be reused for memory + claims below.

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

        // 1c. Hybrid node vector search: fuse BM25 node results with vector similarity
        if !query_embedding.is_empty() {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            let vector_hits = graph.node_vector_index.search(&query_embedding, 20, 0.3);
            if !vector_hits.is_empty() {
                explanation.push(format!("Node vector: {} results", vector_hits.len()));
                ranked_lists.push(vector_hits);
            }

            // 1d. Edge/triplet vector search: find edges whose "subject predicate object"
            // text is semantically similar to the query. Resolve hits to source+target nodes.
            let edge_hits = graph.edge_vector_index.search(&query_embedding, 10, 0.4);
            if !edge_hits.is_empty() {
                let mut triplet_node_hits: Vec<(u64, f32)> = Vec::new();
                for &(edge_id, sim) in &edge_hits {
                    if let Some(edge) = graph.edges.get(edge_id) {
                        // Both source and target nodes are relevant
                        triplet_node_hits.push((edge.source, sim));
                        triplet_node_hits.push((edge.target, sim * 0.8)); // target slightly lower
                    }
                }
                if !triplet_node_hits.is_empty() {
                    explanation.push(format!(
                        "Triplet vector: {} edges -> {} nodes",
                        edge_hits.len(),
                        triplet_node_hits.len()
                    ));
                    ranked_lists.push(triplet_node_hits);
                }
            }
        }

        // 2. Memory retrieval — only when explicitly requested via include_memories.
        // By default NLQ returns only claims and graph-edge facts.
        // Memories are also available separately via the /memory endpoint.
        let memories: Vec<crate::memory::Memory> = if include_memories {
            let mem_query = crate::MemoryRetrievalQuery {
                query_text: question.to_string(),
                query_embedding: query_embedding.clone(),
                context: None,
                anchor_node: None,
                agent_id: None,
                session_id: None,
                now: None,
                limit: 5,
            };
            let retrieved = self.retrieve_memories_multi_signal(mem_query, None).await;
            if !retrieved.is_empty() {
                explanation.push(format!("Memories: {} results", retrieved.len()));
            }
            retrieved
        } else {
            Vec::new()
        };

        // 3. Claim search (always hybrid BM25 + semantic via RRF)
        if let Some(ref store) = self.claim_store {
            {
                let hybrid_config = crate::claims::hybrid_search::HybridSearchConfig::default();
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
        // Only include Claim and Concept neighbors — skip Memory/Strategy/Event nodes.
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
                    // 1-hop neighbors — only Claims and Concepts (fact-bearing nodes)
                    for &neighbor_id in graph
                        .neighbors_directed(entity.node_id, crate::structures::Direction::Both)
                        .iter()
                    {
                        if let Some(node) = graph.get_node(neighbor_id) {
                            match &node.node_type {
                                crate::structures::NodeType::Claim { .. }
                                | crate::structures::NodeType::Concept { .. } => {
                                    entity_hits.push((neighbor_id, 0.5));
                                },
                                _ => {}, // Skip Memory, Strategy, Event, etc.
                            }
                        }
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
        let mut fused = crate::retrieval::multi_list_rrf(&ranked_lists, 60.0);

        // 5a. Classify query intent + temporal frame (LLM, fallback to rule-based).
        // Must run BEFORE temporal filter so we know whether to apply it.
        let (llm_hint, temporal_frame) = if let Some(ref llm_client) = self.unified_llm_client {
            match tokio::time::timeout(
                std::time::Duration::from_secs(3),
                crate::nlq::llm_hint::classify_with_llm(llm_client.as_ref(), question),
            )
            .await
            {
                Ok(Ok(Some(hint))) => {
                    let frame = hint.temporal_frame.clone();
                    (Some(hint), frame)
                },
                Ok(Ok(None)) => (None, crate::nlq::llm_hint::detect_temporal_frame(question)),
                Ok(Err(e)) => {
                    tracing::debug!("LLM classification failed (non-fatal): {}", e);
                    (None, crate::nlq::llm_hint::detect_temporal_frame(question))
                },
                Err(_) => {
                    tracing::debug!("LLM classification timed out");
                    (None, crate::nlq::llm_hint::detect_temporal_frame(question))
                },
            }
        } else {
            (None, crate::nlq::llm_hint::detect_temporal_frame(question))
        };

        // 5b. Temporal validity filter: only apply for Current temporal frame.
        // Historical/Comparative/Timeless queries need access to superseded state.
        let superseded_targets = if temporal_frame == crate::nlq::llm_hint::TemporalFrame::Current {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            apply_temporal_validity_filter(&mut fused, graph)
        } else {
            tracing::info!("Skipping temporal filter for {:?} frame", temporal_frame);
            std::collections::HashSet::new()
        };

        // 5b2. State-anchor filter: remove claims whose state_anchor metadata
        // disagrees with the current projected entity state. Only for Current frame.
        if temporal_frame == crate::nlq::llm_hint::TemporalFrame::Current {
            if let Some(ref store) = self.claim_store {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                apply_state_anchor_filter(&mut fused, graph, store, &self.ontology);
            }
        }

        // 5c. LLM-driven dynamic projection from graph edges.
        // Maps LLM hints to ConversationQueryType and uses the unified query path.
        // Both locks are scoped tightly and released after projection completes.
        let dynamic_projection = if let Some(ref hint) = llm_hint {
            let proj = {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                let store = self.structured_memory.read().await;
                let name_registry = crate::conversation::types::NameRegistry::new();
                build_dynamic_projection(graph, hint, question, &store, &name_registry)
            }; // both locks released here
            if let Some(ref _text) = proj {
                explanation.push(format!(
                    "Projection: {:?}/{:?}",
                    hint.structure_hint, hint.intent_hint
                ));
                tracing::info!(
                    "Dynamic projection: {:?}/{:?}",
                    hint.structure_hint,
                    hint.intent_hint
                );
            }
            proj
        } else {
            None
        };

        // Track whether the fast path answered the query (used for DRIFT gating)
        let had_dynamic_projection = dynamic_projection.is_some();

        // 6. Build retrieval context from top results + optional projection
        let retrieval_context = {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            let retrieval = crate::nlq::unified::format_unified_results(
                &fused,
                question,
                graph,
                &memories,
                &superseded_targets,
            );
            match dynamic_projection {
                Some(proj) if !proj.is_empty() => {
                    if retrieval == "No relevant information found." {
                        proj
                    } else {
                        format!("{}\n\n{}", proj, retrieval)
                    }
                },
                _ => retrieval,
            }
        };

        // 6a. Enrich context for preference/recommendation queries with structured analysis
        let retrieval_context = if matches!(
            llm_hint.as_ref().map(|h| &h.structure_hint),
            Some(crate::nlq::llm_hint::StructureHint::PreferenceList)
        ) {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            let entities = extract_entity_names_from_question(graph, question);
            let mut analysis = String::new();
            for entity in &entities {
                let facts =
                    crate::conversation::graph_projection::collect_entity_facts(graph, entity);
                for fact in &facts {
                    if fact.sentiment != 0.5 {
                        analysis.push_str(&format!(
                            "- {} {} (sentiment: {:.1}, current: {})\n",
                            entity, fact.target_name, fact.sentiment, fact.is_current
                        ));
                    }
                }
            }
            if analysis.is_empty() {
                retrieval_context
            } else {
                format!(
                    "PREFERENCE ANALYSIS:\n{}\n\n{}",
                    analysis, retrieval_context
                )
            }
        } else {
            retrieval_context
        };

        // 6b. LLM answer synthesis: produce a focused answer from retrieved context.
        // If the LLM is available, we ask it to answer the question using ONLY the
        // retrieved facts. This replaces the raw bullet-point dump with a direct answer.
        // Falls back to the raw retrieval context if LLM is unavailable or fails.
        let answer = if retrieval_context != "No relevant information found." {
            let synth_client = self
                .synthesis_llm_client
                .as_ref()
                .or(self.unified_llm_client.as_ref());
            if let Some(llm_client) = synth_client {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    synthesize_answer(llm_client.as_ref(), question, &retrieval_context),
                )
                .await
                {
                    Ok(Ok(synthesized)) => {
                        explanation.push("LLM synthesis: ok".to_string());
                        synthesized
                    },
                    Ok(Err(e)) => {
                        tracing::warn!("LLM synthesis failed: {}", e);
                        return Err(GraphError::OperationError(format!(
                            "Answer synthesis failed: {}",
                            e
                        )));
                    },
                    Err(_) => {
                        tracing::warn!("LLM synthesis timed out");
                        return Err(GraphError::OperationError(
                            "Answer synthesis timed out".to_string(),
                        ));
                    },
                }
            } else {
                retrieval_context
            }
        } else {
            retrieval_context
        };

        // 7. Run DRIFT for cross-entity / multi-hop / exploration queries only.
        // Simple current-state lookups are better served by the filtered synthesis
        // above (fewer LLM calls, no stale community summary leakage).
        // Skip DRIFT if the dynamic projection already answered the query.
        let use_drift = self.config.enable_drift_search && !had_dynamic_projection && {
            match llm_hint.as_ref().map(|h| &h.intent_hint) {
                // Multi-hop / cross-entity intents benefit from DRIFT's community context
                Some(crate::nlq::llm_hint::IntentHint::Path)
                | Some(crate::nlq::llm_hint::IntentHint::Subgraph)
                | Some(crate::nlq::llm_hint::IntentHint::Aggregate)
                | Some(crate::nlq::llm_hint::IntentHint::AggregateBalance)
                | Some(crate::nlq::llm_hint::IntentHint::Similarity)
                | Some(crate::nlq::llm_hint::IntentHint::Knowledge) => true,
                // GenericGraph structure hint also suggests cross-entity traversal
                _ => matches!(
                    llm_hint.as_ref().map(|h| &h.structure_hint),
                    Some(crate::nlq::llm_hint::StructureHint::GenericGraph)
                ),
            }
        };
        let (final_answer, drift_explanation) = if use_drift {
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
                        self.run_drift_search(
                            question,
                            &summaries,
                            &fused,
                            llm_client.as_ref(),
                            &temporal_frame,
                        ),
                    )
                    .await
                    {
                        Ok(Ok(drift_result)) => {
                            let explain = format!(
                                "DRIFT: {} communities, {} follow-ups, {} items",
                                drift_result.primer_communities_used.len(),
                                drift_result.followup_queries.len(),
                                drift_result.total_items_retrieved,
                            );
                            (drift_result.answer, Some(explain))
                        },
                        Ok(Err(synth_err)) => {
                            tracing::warn!("DRIFT synthesis failed: {}", synth_err);
                            return Err(GraphError::OperationError(format!(
                                "Answer synthesis failed: {}",
                                synth_err
                            )));
                        },
                        Err(_) => {
                            tracing::warn!("DRIFT search timed out");
                            return Err(GraphError::OperationError(
                                "Answer synthesis timed out".to_string(),
                            ));
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

    /// Run DRIFT search: primer → follow-up → temporal context → fact summary → synthesis.
    async fn run_drift_search(
        &self,
        question: &str,
        community_summaries: &HashMap<u64, crate::community_summary::CommunitySummary>,
        initial_fused: &[(u64, f32)],
        llm_client: &dyn crate::llm_client::LlmClient,
        temporal_frame: &crate::nlq::llm_hint::TemporalFrame,
    ) -> Result<crate::nlq::drift::DriftResult, String> {
        // Use synthesis client for final answer composition if available
        let synthesis_client: &dyn crate::llm_client::LlmClient = self
            .synthesis_llm_client
            .as_ref()
            .map(|c| c.as_ref())
            .unwrap_or(llm_client);
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
        // Phase 3: Temporal Context Assembly — build entity timelines + current state
        let temporal_context = {
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

            // Resolve entity mentions from question and build temporal context
            let entities = extract_entity_names_from_question(graph, question);
            let mut entity_timelines = Vec::new();
            let mut current_state = Vec::new();

            for entity in &entities {
                if let Some(timeline) =
                    crate::conversation::graph_projection::build_entity_timeline_summary(
                        graph, entity,
                    )
                {
                    entity_timelines.push(timeline);
                }
                let projected = crate::conversation::graph_projection::project_entity_state(
                    graph,
                    entity,
                    u64::MAX,
                    None,
                );
                for slot in projected.slots.values() {
                    let value = slot.value.as_deref().unwrap_or(&slot.target_name);
                    current_state
                        .push(format!("{}: {} = {}", entity, slot.association_type, value));
                }
            }

            crate::nlq::drift::DriftTemporalContext {
                entity_timelines,
                current_state,
                temporal_frame: format!("{:?}", temporal_frame),
            }
        };

        let total_items = merged.len();

        // ── DRIFT debug logging ──
        tracing::info!(
            "DRIFT community context ({} items):",
            community_context.len()
        );
        for (i, c) in community_context.iter().enumerate() {
            tracing::info!("  community[{}]: {}", i, &c[..c.len().min(300)]);
        }
        tracing::info!(
            "DRIFT retrieved snippets ({} items):",
            retrieved_snippets.len()
        );
        for (i, s) in retrieved_snippets.iter().enumerate() {
            tracing::info!("  snippet[{}]: {}", i, &s[..s.len().min(300)]);
        }
        tracing::info!(
            "DRIFT temporal context: frame={}",
            temporal_context.temporal_frame
        );
        for s in &temporal_context.current_state {
            tracing::info!("  current_state: {}", s);
        }
        for t in &temporal_context.entity_timelines {
            tracing::info!("  timeline: {}", &t[..t.len().min(500)]);
        }

        // Phase 4: Fact Summary — LLM verifies facts against temporal context
        let fact_summary = if !retrieved_snippets.is_empty() {
            let summary = crate::nlq::drift::drift_fact_summary(
                llm_client,
                question,
                &retrieved_snippets,
                &temporal_context,
                config,
            )
            .await;
            if summary.is_empty() {
                tracing::info!("DRIFT fact_summary: (empty)");
                None
            } else {
                tracing::info!(
                    "DRIFT fact_summary:\n{}",
                    &summary[..summary.len().min(1000)]
                );
                Some(summary)
            }
        } else {
            None
        };

        // Phase 5: Synthesis — with temporal awareness and fact sheet (uses higher-quality model)
        let answer = crate::nlq::drift::drift_synthesis(
            synthesis_client,
            question,
            &community_context,
            &retrieved_snippets,
            &temporal_context,
            fact_summary.as_deref(),
            config,
        )
        .await
        .map_err(|e| {
            tracing::warn!("DRIFT synthesis failed: {}", e);
            e
        })?;

        Ok(crate::nlq::drift::DriftResult {
            answer,
            primer_communities_used: primer_communities,
            followup_queries,
            total_items_retrieved: total_items,
            fact_summary,
        })
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
