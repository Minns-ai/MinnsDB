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

    pub async fn natural_language_query_federated(
        &self,
        question: &str,
        pagination: &crate::nlq::NlqPagination,
        session_id: Option<&str>,
        include_memories: bool,
        group_id: &str,
        metadata_filter: &std::collections::HashMap<String, serde_json::Value>,
        federated_sources: Option<&Vec<String>>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
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
            federated_sources,
            None, // federated path has not pre-classified — let the unified path do it
        )
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

        // Classify once; route via planner. Structured plans answer
        // directly from a graph-derived projection; UnifiedRetrieval is the
        // fallback that runs the full BM25 + vector + claims + synthesis
        // pipeline. The classified hint is reused if we fall through so we
        // don't pay for a second LLM classification call.
        let precomputed_hint = self.classify_question(&effective_question).await;
        if let Some(ref hint) = precomputed_hint {
            let plan = crate::nlq::planner::plan(hint);
            tracing::info!(
                target: "nlq",
                variant = plan.variant_name(),
                temporal_frame = ?hint.temporal_frame,
                subject = ?hint.subject,
                predicate = ?hint.predicate,
                "nlq.plan"
            );
            match plan {
                crate::nlq::planner::NlqPlan::StateMachineWalk {
                    subject,
                    predicate,
                    frame,
                    intent,
                } => {
                    return self
                        .answer_state_machine_walk(
                            &effective_question,
                            &subject,
                            &predicate,
                            &frame,
                            &intent,
                        )
                        .await;
                },
                crate::nlq::planner::NlqPlan::ActiveEdgeFetch {
                    subject,
                    predicate,
                    intent,
                } => {
                    return self
                        .answer_active_edge_fetch(
                            &effective_question,
                            &subject,
                            &predicate,
                            &intent,
                        )
                        .await;
                },
                crate::nlq::planner::NlqPlan::UnifiedRetrieval { reason: _ } => {
                    // Fall through to the unified retrieval path below,
                    // passing the already-computed hint so the unified
                    // path skips its own classify_with_llm call.
                },
            }
        }

        self.execute_unified_query(
            &effective_question,
            pagination,
            session_id,
            include_memories,
            group_id,
            metadata_filter,
            None,
            precomputed_hint,
        )
        .await
    }

    /// Classify a question using the LLM hint classifier with a 3s timeout.
    /// Returns `None` when no LLM client is configured or the classifier
    /// times out / fails — the caller falls through to UnifiedRetrieval.
    async fn classify_question(
        &self,
        question: &str,
    ) -> Option<crate::nlq::llm_hint::LlmHintResponse> {
        let llm_client = self.unified_llm_client.as_ref()?;
        match tokio::time::timeout(
            std::time::Duration::from_secs(3),
            crate::nlq::llm_hint::classify_with_llm(llm_client.as_ref(), question),
        )
        .await
        {
            Ok(Ok(Some(hint))) => Some(hint),
            Ok(Ok(None)) => None,
            Ok(Err(e)) => {
                tracing::debug!("LLM classification failed (non-fatal): {}", e);
                None
            },
            Err(_) => {
                tracing::debug!("LLM classification timed out");
                None
            },
        }
    }

    /// Answer a `StateMachineWalk` plan: project a state machine from the
    /// graph for `(subject, predicate)`, walk it according to the frame,
    /// and surface the result as an `NlqResponse`. No LLM synthesis call —
    /// the answer is deterministic from the projection.
    async fn answer_state_machine_walk(
        &self,
        question: &str,
        subject: &str,
        predicate: &str,
        frame: &crate::nlq::llm_hint::TemporalFrame,
        intent: &crate::nlq::llm_hint::IntentHint,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        use crate::conversation::graph_projection::project_state_machine;
        use crate::nlq::walkers::walk_state_machine;

        let start = std::time::Instant::now();

        let sm = {
            let inference = self.inference.read().await;
            project_state_machine(inference.graph(), subject, predicate)
        };

        let answer = walk_state_machine(
            &sm,
            frame,
            intent,
            crate::nlq::walkers::DEFAULT_HISTORICAL_LIMIT,
        );

        let phrased = phrase_state_machine_answer(subject, predicate, frame, &answer);

        tracing::info!(
            target: "nlq",
            path = "state_machine_walk",
            subject = subject,
            predicate = predicate,
            frame = ?frame,
            history_len = sm.history.len(),
            answer_value = %answer.value,
            ms = start.elapsed().as_millis() as u64,
            "nlq.structured_answer"
        );

        Ok(crate::nlq::NlqResponse {
            answer: phrased,
            query_used: crate::traversal::GraphQuery::PageRank {
                iterations: 0,
                damping_factor: 0.0,
            },
            result: crate::traversal::QueryResult::Rankings(vec![]),
            intent: crate::nlq::intent::QueryIntent::KnowledgeQuery,
            confidence: 0.95,
            execution_time_ms: start.elapsed().as_millis() as u64,
            explanation: vec![
                "Structured: project_state_machine + walk".to_string(),
                format!(
                    "subject={subject} predicate={predicate} frame={:?} history={}",
                    frame,
                    sm.history.len()
                ),
                format!("question={question}"),
            ],
            total_count: sm.history.len(),
            entities_resolved: vec![],
        })
    }

    /// Answer an `ActiveEdgeFetch` plan: read `(subject, predicate)` current
    /// state directly via `project_entity_state` — no trajectory build.
    async fn answer_active_edge_fetch(
        &self,
        question: &str,
        subject: &str,
        predicate: &str,
        _intent: &crate::nlq::llm_hint::IntentHint,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        use crate::conversation::graph_projection::project_entity_state;

        let start = std::time::Instant::now();

        // Walk only active edges via the existing fast-path projection.
        let predicate_lower = predicate.to_lowercase();
        let value = {
            let inference = self.inference.read().await;
            let projected =
                project_entity_state(inference.graph(), subject, u64::MAX, Some(&self.ontology));
            projected
                .slots
                .values()
                .find(|slot| {
                    let assoc_lower = slot.association_type.to_lowercase();
                    assoc_lower == predicate_lower
                        || assoc_lower.split(':').any(|p| p == predicate_lower)
                })
                .map(|slot| {
                    slot.value
                        .clone()
                        .unwrap_or_else(|| slot.target_name.clone())
                })
                .unwrap_or_default()
        };

        let phrased = if value.is_empty() {
            format!(
                "No current {predicate} found for {subject} (the active-edge \
                 projection returned no matching slot)."
            )
        } else {
            format!("{subject} currently {predicate} {value}.")
        };

        tracing::info!(
            target: "nlq",
            path = "active_edge_fetch",
            subject = subject,
            predicate = predicate,
            answer_value = %value,
            ms = start.elapsed().as_millis() as u64,
            "nlq.structured_answer"
        );

        Ok(crate::nlq::NlqResponse {
            answer: phrased,
            query_used: crate::traversal::GraphQuery::PageRank {
                iterations: 0,
                damping_factor: 0.0,
            },
            result: crate::traversal::QueryResult::Rankings(vec![]),
            intent: crate::nlq::intent::QueryIntent::KnowledgeQuery,
            confidence: 0.95,
            execution_time_ms: start.elapsed().as_millis() as u64,
            explanation: vec![
                "Structured: project_entity_state (active edge only)".to_string(),
                format!("subject={subject} predicate={predicate}"),
                format!("question={question}"),
            ],
            total_count: if value.is_empty() { 0 } else { 1 },
            entities_resolved: vec![],
        })
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
        federated_sources: Option<&Vec<String>>,
        // When `Some`, the classifier was already run by the caller (e.g.
        // `natural_language_query_scoped` after the planner returned
        // UnifiedRetrieval) and the hint can be reused. Avoids paying for
        // a second classification LLM call per query on the fallback path.
        precomputed_hint: Option<crate::nlq::llm_hint::LlmHintResponse>,
    ) -> GraphResult<crate::nlq::NlqResponse> {
        let start = std::time::Instant::now();
        let mut explanation = vec!["Unified NLQ pipeline activated".to_string()];
        let mut ranked_lists: Vec<Vec<(u64, f32)>> = Vec::new();

        // Spawn federated search in parallel with local sources
        let federated_task = match (&self.federated_search, federated_sources) {
            (Some(fed), Some(sources)) if !sources.is_empty() => {
                let fed = Arc::clone(fed);
                let req = crate::federated_search::FederatedSearchRequest {
                    query: question.to_string(),
                    sources: sources.clone(),
                    limit: 10,
                };
                Some(tokio::spawn(async move { fed.search(&req).await }))
            },
            _ => None,
        };

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

        // Edge facts surfaced to the synthesis context — NL sentences built
        // from matched edges' rich text, NOT just their endpoint node IDs.
        // Filled inside the edge-search branch below, consumed when building
        // the synthesis prompt.
        let mut edge_facts_for_synthesis: Vec<(String, f32, Option<u64>)> = Vec::new();

        // 1c + 1d. Hybrid node and edge vector search, run concurrently. Node
        // hits feed BM25 fusion directly; edge hits resolve to source+target
        // nodes under the inference read lock below.
        if !query_embedding.is_empty() {
            let node_query = minns_vectors::Query::builder(query_embedding.clone())
                .top_k(20 * fetch_multiplier)
                .min_score(0.3)
                .build();
            let edge_query = minns_vectors::Query::builder(query_embedding.clone())
                .top_k(10 * fetch_multiplier)
                .min_score(0.4)
                .build();

            let (node_result, edge_result) = tokio::join!(
                self.vectors.nodes.search(&node_query),
                self.vectors.edges.search(&edge_query),
            );

            let vector_hits: Vec<(u64, f32)> = match node_result {
                Ok(hits) => hits.into_iter().map(|h| (h.id as u64, h.score)).collect(),
                Err(e) => {
                    tracing::warn!("NLQ: node vector search failed: {e}");
                    Vec::new()
                },
            };
            if !vector_hits.is_empty() {
                explanation.push(format!("Node vector: {} results", vector_hits.len()));
                ranked_lists.push(vector_hits);
            }

            let edge_hits: Vec<(u64, f32)> = match edge_result {
                Ok(hits) => hits.into_iter().map(|h| (h.id as u64, h.score)).collect(),
                Err(e) => {
                    tracing::warn!("NLQ: edge vector search failed: {e}");
                    Vec::new()
                },
            };

            let inference = self.inference.read().await;
            let graph = inference.graph();
            if !edge_hits.is_empty() {
                let mut triplet_node_hits: Vec<(u64, f32)> = Vec::new();
                // ALSO surface the matched edges as natural-language fact lines
                // to the synthesis context — built from the same rich text the
                // embedding was computed on. Without this, the matched edge's
                // statement is thrown away; only the source / target node IDs
                // survive into the fused ranking, and the answer LLM never
                // sees the sentence that actually answers the question.
                //
                // Failure case before this fix: "Where does the user live?"
                // matched the edge "User recently transferred to the NYC
                // office. [current]" via vector search, but the formatter
                // pulled "user" and "nyc office" as nodes and dropped the
                // sentence. The LLM had no fact stating where the user
                // currently lives, and either said "no info" or matched on
                // the wrong predicate.
                let mut semantic_edge_facts: Vec<(String, f32, Option<u64>)> = Vec::new();
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

                        // Build the same NL sentence the edge was embedded
                        // with. Reuses the helper that compaction uses so the
                        // synthesis sentence and the embedding subject agree.
                        let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &edge.edge_type
                        else {
                            continue;
                        };
                        let source_name = crate::conversation::graph_projection::concept_name_of(
                            graph,
                            edge.source,
                        )
                        .unwrap_or_default();
                        let target_name = crate::conversation::graph_projection::concept_name_of(
                            graph,
                            edge.target,
                        )
                        .unwrap_or_default();
                        if source_name.is_empty() || target_name.is_empty() {
                            continue;
                        }
                        let rich = crate::conversation::compaction::embedding::build_rich_edge_text(
                            &source_name,
                            association_type,
                            &target_name,
                            &edge.properties,
                            edge.valid_until.is_none(),
                        );
                        semantic_edge_facts.push((rich, sim * penalty, edge.valid_from));
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
                // Sort by score DESC so the top match appears first in the
                // synthesis prompt. The formatter caps display to a small N.
                semantic_edge_facts
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if !semantic_edge_facts.is_empty() {
                    explanation.push(format!(
                        "Semantic edge facts: {} candidates",
                        semantic_edge_facts.len()
                    ));
                }
                edge_facts_for_synthesis = semantic_edge_facts;
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
                )
                .await
                {
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

        // 5a. Classify query intent + temporal frame. Reuse the
        // caller-provided hint when available (the planner pre-classifies
        // and routes UnifiedRetrieval back here — re-classifying would
        // burn a second LLM call per fallback query). Only call the
        // classifier when we have no hint to start from.
        let (llm_hint, temporal_frame) = if let Some(hint) = precomputed_hint {
            let frame = hint.temporal_frame.clone();
            (Some(hint), frame)
        } else if let Some(ref llm_client) = self.unified_llm_client {
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

        tracing::info!(
            target: "nlq",
            temporal_frame = ?temporal_frame,
            "nlq.temporal_frame classified"
        );

        // 5b. Temporal validity filter: only apply for Current temporal frame.
        // Historical / First / Last / Comparative / Timeless queries all need
        // access to superseded state (Luna's adoption stays visible even after
        // Max supersedes it for "which pet did I get first").
        let superseded_targets = if temporal_frame == crate::nlq::llm_hint::TemporalFrame::Current {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            apply_temporal_validity_filter(&mut fused, graph)
        } else {
            tracing::info!(
                target: "nlq",
                temporal_frame = ?temporal_frame,
                "nlq.skip_temporal_filter"
            );
            std::collections::HashSet::new()
        };

        // 5b-bis. For First / Last frames, reorder the fused list by edge
        // valid_from so the answer LLM sees the chronologically-leading
        // candidate at the top regardless of how retrieval ranked it.
        // Without this, "which pet did I get first" can correctly include
        // both adoption edges in the pool but still surface the most
        // recently-adopted one first (recency-biased embedding similarity).
        match temporal_frame {
            crate::nlq::llm_hint::TemporalFrame::First
            | crate::nlq::llm_hint::TemporalFrame::Last => {
                let ascending =
                    matches!(temporal_frame, crate::nlq::llm_hint::TemporalFrame::First);
                let inference = self.inference.read().await;
                let graph = inference.graph();
                // Compute each fused node's "anchor time" = earliest valid_from
                // (for First) or latest valid_from (for Last) of any active
                // incident edge. Nodes with no temporal anchor sort to the end.
                let mut anchored: Vec<(u64, f32, Option<u64>)> = fused
                    .into_iter()
                    .map(|(nid, score)| {
                        let mut best: Option<u64> = None;
                        for edge in graph
                            .get_edges_from(nid)
                            .iter()
                            .chain(graph.get_edges_to(nid).iter())
                        {
                            if let Some(vf) = edge.valid_from {
                                best = match (best, ascending) {
                                    (None, _) => Some(vf),
                                    (Some(cur), true) if vf < cur => Some(vf),
                                    (Some(cur), false) if vf > cur => Some(vf),
                                    _ => best,
                                };
                            }
                        }
                        (nid, score, best)
                    })
                    .collect();
                // Sort: anchored items first, ordered ASC/DESC by anchor; then
                // unanchored items in their original score order.
                anchored.sort_by(|a, b| match (a.2, b.2) {
                    (Some(ta), Some(tb)) => {
                        if ascending {
                            ta.cmp(&tb)
                        } else {
                            tb.cmp(&ta)
                        }
                    },
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
                });
                fused = anchored
                    .into_iter()
                    .map(|(nid, score, _)| (nid, score))
                    .collect();
                explanation.push(format!(
                    "Temporal ordering: {} by valid_from",
                    if ascending { "ASC" } else { "DESC" }
                ));
            },
            _ => {},
        }

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
                &edge_facts_for_synthesis,
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

        let retrieval_context = if let Some(task) = federated_task {
            match task.await {
                Ok(Ok(response)) if !response.results.is_empty() => {
                    let fed_text =
                        crate::federated_search::format_federated_context(&response.results);
                    explanation.push(format!("Federated: {} results", response.results.len()));
                    for err in &response.errors {
                        tracing::debug!("Federated source {}: {}", err.source, err.error);
                    }
                    format!("{}\n\nEXTERNAL SOURCES:\n{}", retrieval_context, fed_text)
                },
                Ok(Err(e)) => {
                    tracing::warn!("Federated search error: {}", e);
                    retrieval_context
                },
                Err(e) => {
                    tracing::warn!("Federated search task failed: {}", e);
                    retrieval_context
                },
                _ => retrieval_context,
            }
        } else {
            retrieval_context
        };

        // 6b. LLM answer synthesis: produce a focused answer from retrieved context.
        // If the LLM is available, we ask it to answer the question using ONLY the
        // retrieved facts. This replaces the raw bullet-point dump with a direct answer.
        // Falls back to the raw retrieval context if LLM is unavailable or fails.
        const MAX_CONTEXT_CHARS: usize = 24_000;
        let synthesis_context = if retrieval_context.len() > MAX_CONTEXT_CHARS {
            let mut end = MAX_CONTEXT_CHARS;
            while end < retrieval_context.len() && !retrieval_context.is_char_boundary(end) {
                end -= 1;
            }
            &retrieval_context[..end]
        } else {
            &retrieval_context
        };
        let answer = if retrieval_context != "No relevant information found." {
            let synth_client = self
                .synthesis_llm_client
                .as_ref()
                .or(self.unified_llm_client.as_ref());
            if let Some(llm_client) = synth_client {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    synthesize_answer(llm_client.as_ref(), question, synthesis_context),
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
            // Evict oldest half by last_activity when too many sessions tracked
            if contexts.len() > 1000 {
                let mut by_activity: Vec<(String, u64)> = contexts
                    .iter()
                    .map(|(k, v)| (k.clone(), v.last_activity()))
                    .collect();
                by_activity.sort_by_key(|(_, ts)| *ts);
                let evict_count = by_activity.len() / 2;
                for (k, _) in by_activity.into_iter().take(evict_count) {
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

/// Phrase a walker answer as a single English sentence keyed off the frame.
/// Deterministic — no LLM call. Keeps the structured path fast and gives
/// the same wording for the same projection state.
fn phrase_state_machine_answer(
    subject: &str,
    predicate: &str,
    frame: &crate::nlq::llm_hint::TemporalFrame,
    answer: &crate::nlq::walkers::StateMachineAnswer,
) -> String {
    use crate::nlq::llm_hint::TemporalFrame;

    if answer.all_values.is_empty() && answer.value.is_empty() {
        return format!(
            "No {predicate} record found for {subject} (the state-machine \
             projection has no matching transitions)."
        );
    }

    match frame {
        TemporalFrame::First => format!(
            "{subject}'s first {predicate} target was {value}.",
            value = answer.value,
        ),
        TemporalFrame::Last => format!(
            "{subject}'s most recent {predicate} target was {value}.",
            value = answer.value,
        ),
        TemporalFrame::Historical | TemporalFrame::Comparative => {
            if answer.all_values.is_empty() {
                format!("{subject}'s {predicate} history: {}.", answer.value)
            } else {
                format!(
                    "{subject}'s {predicate} history (oldest → newest): {}.",
                    answer.all_values.join(", "),
                )
            }
        },
        TemporalFrame::Current | TemporalFrame::Timeless => {
            if answer.value.is_empty() {
                format!("{subject} has no current {predicate}.")
            } else {
                format!("{subject} currently {predicate} {}.", answer.value)
            }
        },
    }
}
