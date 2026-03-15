// crates/agent-db-graph/src/integration/queries/nlq.rs
//
// Natural language query entry points and the unified multi-source pipeline
// (BM25 + memory + claims + graph entity resolution + RRF fusion + LLM synthesis).

use super::*;

impl GraphEngine {
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
        self.natural_language_query_with_metadata(
            question,
            pagination,
            session_id,
            include_memories,
            &std::collections::HashMap::new(),
        )
        .await
    }

    /// Execute a natural language query with metadata-based filtering.
    ///
    /// When `metadata_filter` is non-empty, only graph edges whose properties
    /// contain all specified key-value pairs are included in retrieval results.
    pub async fn natural_language_query_with_metadata(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
        include_memories: bool,
        metadata_filter: &std::collections::HashMap<String, serde_json::Value>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        self.natural_language_query_scoped(
            question,
            pagination,
            session_id,
            include_memories,
            "",
            metadata_filter,
        )
        .await
    }

    /// Execute a natural language query scoped to a `group_id` and optional metadata.
    ///
    /// This is the most complete query entry point. `group_id` provides fast
    /// multi-tenant isolation (empty string = unscoped). `metadata_filter`
    /// provides additional property-level filtering.
    pub async fn natural_language_query_scoped(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
        include_memories: bool,
        group_id: &str,
        metadata_filter: &std::collections::HashMap<String, serde_json::Value>,
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
            group_id,
            metadata_filter,
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
        group_id: &str,
        metadata_filter: &std::collections::HashMap<String, serde_json::Value>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        let start = std::time::Instant::now();
        let mut explanation = vec!["Unified NLQ pipeline activated".to_string()];
        let mut ranked_lists: Vec<Vec<(u64, f32)>> = Vec::new();

        // When metadata filtering is active, over-fetch so we have enough
        // results after filtering. The multiplier ensures we don't lose all
        // top results to non-matching metadata.
        let has_scope_filter = !group_id.is_empty() || !metadata_filter.is_empty();
        let fetch_multiplier = if has_scope_filter { 5 } else { 1 };

        // 1. BM25 search (always available)
        let bm25 = self.search_bm25(question, 20 * fetch_multiplier).await;
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
            let vector_hits =
                graph
                    .node_vector_index
                    .search(&query_embedding, 20 * fetch_multiplier, 0.3);
            if !vector_hits.is_empty() {
                explanation.push(format!("Node vector: {} results", vector_hits.len()));
                ranked_lists.push(vector_hits);
            }

            // 1d. Edge/triplet vector search: find edges whose "subject predicate object"
            // text is semantically similar to the query. Resolve hits to source+target nodes.
            let edge_hits =
                graph
                    .edge_vector_index
                    .search(&query_embedding, 10 * fetch_multiplier, 0.4);
            if !edge_hits.is_empty() {
                let mut triplet_node_hits: Vec<(u64, f32)> = Vec::new();
                for &(edge_id, sim) in &edge_hits {
                    if let Some(edge) = graph.edges.get(edge_id) {
                        // When scoped, skip edges that don't match group_id or metadata
                        if has_scope_filter {
                            if !group_id.is_empty() && edge.group_id != group_id {
                                continue;
                            }
                            if !metadata_filter
                                .iter()
                                .all(|(k, v)| edge.properties.get(k) == Some(v))
                            {
                                continue;
                            }
                        }
                        // Demote superseded edges — they'll be filtered for
                        // Current-frame queries but kept for Historical ones
                        let penalty = if edge.valid_until.is_some() { 0.1 } else { 1.0 };
                        triplet_node_hits.push((edge.source, sim * penalty));
                        triplet_node_hits.push((edge.target, sim * 0.8 * penalty));
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

        // 5b3. Epoch filter: catch claims that escaped both the temporal validity
        // filter and the state anchor filter. For single-valued categories, any
        // claim created before the current epoch's valid_from is stale.
        if temporal_frame == crate::nlq::llm_hint::TemporalFrame::Current {
            if let Some(ref store) = self.claim_store {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                apply_epoch_filter(&mut fused, graph, store, &self.ontology);
            }
        }

        // 5c-meta. Group ID + metadata filter: zero out nodes not reachable via
        // edges matching the group_id and metadata key-value pairs.
        let has_group_filter = !group_id.is_empty();
        if has_group_filter || !metadata_filter.is_empty() {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            for (node_id, score) in fused.iter_mut() {
                if *score <= 0.0 {
                    continue;
                }
                // A node passes if ANY incident edge matches group_id AND all metadata
                let edges_in = graph.get_edges_to(*node_id);
                let edges_out = graph.get_edges_from(*node_id);
                let has_matching_edge = edges_in.iter().chain(edges_out.iter()).any(|edge| {
                    let group_ok = !has_group_filter || edge.group_id == group_id;
                    let meta_ok = metadata_filter
                        .iter()
                        .all(|(k, v)| edge.properties.get(k) == Some(v));
                    group_ok && meta_ok
                });
                if !has_matching_edge {
                    *score = 0.0;
                }
            }
            if has_group_filter {
                explanation.push(format!("Group filter: '{}'", group_id));
            }
            if !metadata_filter.is_empty() {
                explanation.push(format!(
                    "Metadata filter: {} keys applied",
                    metadata_filter.len()
                ));
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
                Some(&self.ontology),
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
            // Evict if too many sessions tracked
            if contexts.len() > 1000 {
                let keys: Vec<String> = contexts.keys().take(contexts.len() / 2).cloned().collect();
                for k in keys {
                    contexts.remove(&k);
                }
            }
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
}
