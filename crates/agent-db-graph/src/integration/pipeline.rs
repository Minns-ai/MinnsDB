use super::*;
use crate::metadata_normalize::{metadata_value_preview, MetadataRole};
use crate::structures::ConceptType;
use std::collections::HashSet;

/// Extract a string value from event metadata by key.
fn metadata_str(
    metadata: &std::collections::HashMap<String, agent_db_events::core::MetadataValue>,
    key: &str,
) -> Option<String> {
    metadata.get(key).and_then(|v| match v {
        agent_db_events::core::MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    })
}

// ────────── Entity Resolution Helpers ──────────

/// Normalize an entity name for consistent matching.
///
/// Lowercases, trims, strips leading articles, replaces underscores with spaces.
fn normalize_entity_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let trimmed = lower.trim();
    // Strip leading articles
    let stripped = trimmed
        .strip_prefix("the ")
        .or_else(|| trimmed.strip_prefix("a "))
        .or_else(|| trimmed.strip_prefix("an "))
        .unwrap_or(trimmed);
    // Replace underscores with spaces
    stripped.replace('_', " ").trim().to_string()
}

/// Fast fuzzy entity match: checks if two normalized entity names refer to the same entity.
///
/// Returns true if:
/// - Exact match
/// - One is a substring of the other
/// - They share >80% of words
fn fuzzy_entity_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.contains(b) || b.contains(a) {
        return true;
    }
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();
    let intersection = words_a.intersection(&words_b).count();
    let max_len = words_a.len().max(words_b.len());
    if max_len > 0 && intersection as f32 / max_len as f32 > 0.8 {
        return true;
    }
    false
}

/// Resolve an entity to an existing concept node or create a new one.
///
/// Resolution order:
/// 1. Exact match on normalized name (instant)
/// 2. Fuzzy string match across existing concepts (fast scan)
/// 3. Create new node with normalized name
fn resolve_or_create_entity(
    graph: &mut Graph,
    raw_name: &str,
    concept_type: ConceptType,
) -> Option<NodeId> {
    let normalized = normalize_entity_name(raw_name);

    // 1. Exact match on normalized name
    if let Some(&nid) = graph.concept_index.get(&*normalized) {
        return Some(nid);
    }

    // Also try raw name (handles already-normalized names)
    if let Some(&nid) = graph.concept_index.get(raw_name) {
        return Some(nid);
    }

    // 2. Fuzzy match against existing concept names
    let match_found = graph
        .concept_index
        .iter()
        .find(|(existing_name, _)| fuzzy_entity_match(&normalized, existing_name))
        .map(|(_, &nid)| nid);

    if let Some(existing_id) = match_found {
        // Add normalized name as alias
        let interned = graph.interner.intern(&normalized);
        graph.concept_index.insert(interned, existing_id);
        return Some(existing_id);
    }

    // 3. No match — create new node with normalized name
    let node = GraphNode::new(NodeType::Concept {
        concept_name: normalized.clone(),
        concept_type,
        confidence: 0.7,
    });
    match graph.add_node(node) {
        Ok(nid) => {
            let interned = graph.interner.intern(&normalized);
            graph.concept_index.insert(interned, nid);
            // Also index the raw name if different
            if raw_name != normalized {
                let raw_interned = graph.interner.intern(raw_name);
                graph.concept_index.insert(raw_interned, nid);
            }
            tracing::debug!(
                "Entity resolution: created concept node '{}' (raw='{}') id={}",
                normalized,
                raw_name,
                nid
            );
            Some(nid)
        },
        Err(e) => {
            tracing::warn!(
                "Entity resolution: failed to create concept node '{}': {}",
                normalized,
                e
            );
            None
        },
    }
}

/// Derive a sub-key from an object string for multi-valued categories.
/// Takes the first 3 words with 4+ chars, lowercased and joined with underscores.
fn derive_sub_key(object: &str) -> String {
    object
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 4)
        .take(3)
        .collect::<Vec<_>>()
        .join("_")
}

impl GraphEngine {
    /// Process episode for memory formation
    pub(super) async fn process_episode_for_memory(&self, episode: &Episode) -> GraphResult<()> {
        // Load events for summary generation
        let events: Vec<agent_db_events::core::Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        let mut memory_store = self.memory_store.write().await;

        tracing::info!(
            "Memory formation start episode_id={} agent_id={} session_id={} significance={:.3} outcome={:?}",
            episode.id,
            episode.agent_id,
            episode.session_id,
            episode.significance,
            episode.outcome
        );
        if let Some(upsert) = memory_store.store_episode(episode, &events) {
            if upsert.is_new {
                self.stats
                    .total_memories_formed
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
            if let Some(memory) = memory_store.get_memory(upsert.id) {
                drop(memory_store);
                let outputs = crate::contracts::build_episode_record(episode, &[]);
                tracing::info!(
                    "Learning outputs (memory): episode_id={} goal_bucket_id={} behavior_signature={}",
                    outputs.episode_id,
                    outputs.goal_bucket_id,
                    outputs.behavior_signature
                );
                tracing::info!(
                    "Memory formed id={} episode_id={} strength={:.3} relevance={:.3} context_hash={} tier={:?}",
                    upsert.id,
                    episode.id,
                    memory.strength,
                    memory.relevance_score,
                    memory.context.fingerprint,
                    memory.tier
                );
                self.attach_memory_to_graph(episode, &memory).await?;

                // Record in audit trail
                {
                    let mut audit = self.memory_audit_log.write().await;
                    if upsert.is_new {
                        audit.record_add(
                            upsert.id,
                            &memory.summary,
                            &memory.takeaway,
                            crate::memory_audit::MutationActor::Pipeline,
                            Some(format!("episode_id={}", episode.id)),
                        );
                    } else {
                        audit.record_update(
                            upsert.id,
                            "", // old summary not available here (already overwritten)
                            &memory.summary,
                            "",
                            &memory.takeaway,
                            crate::memory_audit::MutationActor::Pipeline,
                            Some(format!("episode_id={} (update)", episode.id)),
                        );
                    }
                }

                // Index into memory BM25 for multi-signal retrieval
                let has_code = events.iter().any(|e| e.is_code);
                {
                    let text = format!(
                        "{} {} {}",
                        memory.summary, memory.takeaway, memory.causal_note
                    );
                    if !text.trim().is_empty() {
                        let mut idx = self.memory_bm25_index.write().await;
                        if has_code {
                            idx.index_document_code(upsert.id, &text);
                        } else {
                            idx.index_document(upsert.id, &text);
                        }
                    }
                }

                // Fire-and-forget: async LLM refinement + embedding
                if let Some(ref refinement) = self.refinement_engine {
                    let memory_id = upsert.id;
                    let store_ref = self.memory_store.clone();
                    let refinement_ref = refinement.clone();
                    let embedding_client = self.embedding_client.clone();
                    let event_narrative = crate::event_content::build_event_narrative(&events);
                    let bm25_ref = self.memory_bm25_index.clone();
                    let audit_ref = self.memory_audit_log.clone();
                    let pre_refine_summary = memory.summary.clone();
                    let pre_refine_takeaway = memory.takeaway.clone();
                    let has_code_for_reindex = has_code;

                    // Gather community context for enrichment
                    let community_ctx = if self.config.enable_context_enrichment {
                        let summaries = self.community_summaries.read().await;
                        if summaries.is_empty() {
                            None
                        } else {
                            let topic = format!("{} {}", memory.summary, memory.takeaway);
                            let ctx = crate::context_enrichment::community_context_for_topic(
                                &topic,
                                &summaries,
                                &self.config.enrichment_config,
                            );
                            if ctx.is_empty() {
                                None
                            } else {
                                Some(ctx)
                            }
                        }
                    } else {
                        None
                    };

                    tokio::spawn(async move {
                        if let Err(e) = refinement_ref
                            .refine_and_embed_memory(
                                memory_id,
                                &store_ref,
                                embedding_client.as_ref(),
                                Some(event_narrative),
                                community_ctx,
                            )
                            .await
                        {
                            tracing::warn!("Memory refinement failed for {}: {}", memory_id, e);
                        } else {
                            // Re-index after refinement updates the summary text
                            let store = store_ref.read().await;
                            if let Some(refined) = store.get_memory(memory_id) {
                                let text = format!(
                                    "{} {} {}",
                                    refined.summary, refined.takeaway, refined.causal_note
                                );
                                if !text.trim().is_empty() {
                                    let mut idx = bm25_ref.write().await;
                                    idx.remove_document(memory_id);
                                    if has_code_for_reindex {
                                        idx.index_document_code(memory_id, &text);
                                    } else {
                                        idx.index_document(memory_id, &text);
                                    }
                                }
                                // Record refinement in audit trail
                                let mut audit = audit_ref.write().await;
                                audit.record_update(
                                    memory_id,
                                    &pre_refine_summary,
                                    &refined.summary,
                                    &pre_refine_takeaway,
                                    &refined.takeaway,
                                    crate::memory_audit::MutationActor::Refinement,
                                    Some("LLM refinement".to_string()),
                                );
                            }
                        }
                    });
                }
            }

            // Check if we should run consolidation
            let new_counter = self
                .episodes_since_consolidation
                .fetch_add(1, AtomicOrdering::Relaxed)
                + 1;
            let should_consolidate = new_counter >= self.config.consolidation_interval;
            if should_consolidate {
                self.episodes_since_consolidation
                    .store(0, AtomicOrdering::Relaxed);
                let store_ref = self.memory_store.clone();
                let engine_ref = self.consolidation_engine.clone();
                let bm25_ref = self.memory_bm25_index.clone();
                let llm_ref = self.unified_llm_client.clone();
                tokio::spawn(async move {
                    let mut store = store_ref.write().await;

                    // Pre-pass: infer goal labels for goalless buckets via LLM
                    let goal_overrides = if let Some(ref llm) = llm_ref {
                        let all_memories = store.list_all_memories();
                        crate::consolidation::infer_goal_labels(llm.as_ref(), &all_memories).await
                    } else {
                        std::collections::HashMap::new()
                    };

                    let mut engine = engine_ref.write().await;
                    let result = engine.run_consolidation(store.as_mut(), &goal_overrides);
                    if result.semantic_created > 0 || result.schema_created > 0 {
                        tracing::info!(
                            "Consolidation pass: {} semantic created, {} schemas created, {} episodes consolidated, {} goals inferred",
                            result.semantic_created,
                            result.schema_created,
                            result.consolidated_episode_ids.len(),
                            goal_overrides.len(),
                        );
                    }
                    // Clean up BM25 entries for deleted memories
                    if result.episodes_deleted > 0 {
                        let mut idx = bm25_ref.write().await;
                        for id in &result.consolidated_episode_ids {
                            idx.remove_document(*id);
                        }
                        for id in &result.consolidated_semantic_ids {
                            idx.remove_document(*id);
                        }
                    }
                });
            }
        } else {
            tracing::info!(
                "Memory formation skipped episode_id={} (not eligible)",
                episode.id
            );
        }

        Ok(())
    }

    /// Process episode for strategy extraction
    pub(super) async fn process_episode_for_strategy(&self, episode: &Episode) -> GraphResult<()> {
        // Extract strategies from Success and Failure episodes
        // Failure episodes produce Constraint strategies (what NOT to do)
        // Partial outcomes are too ambiguous for reliable strategy extraction
        let dominated_outcome = matches!(
            &episode.outcome,
            Some(EpisodeOutcome::Success) | Some(EpisodeOutcome::Failure)
        );
        if !dominated_outcome {
            tracing::info!(
                "Strategy extraction skipped episode_id={} outcome={:?}",
                episode.id,
                episode.outcome
            );
            return Ok(());
        }

        // Get events for this episode
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        if !events.is_empty() {
            let outputs = crate::contracts::build_learning_outputs(episode, &events);
            tracing::info!(
                "Learning outputs (strategy): episode_id={} goal_bucket_id={} transitions={}",
                outputs.episode_record.episode_id,
                outputs.episode_record.goal_bucket_id,
                outputs.abstract_trace.transitions.len()
            );
            let mut strategy_store = self.strategy_store.write().await;
            tracing::info!(
                "Strategy extraction start episode_id={} events={}",
                episode.id,
                events.len()
            );
            if let Some(upsert) = strategy_store.store_episode(episode, &events)? {
                if upsert.is_new {
                    self.stats
                        .total_strategies_extracted
                        .fetch_add(1, AtomicOrdering::Relaxed);
                }
                if let Some(strategy) = strategy_store.get_strategy(upsert.id) {
                    drop(strategy_store);
                    tracing::info!(
                        "Strategy formed id={} episode_id={} quality={:.3} success_count={} failure_count={}",
                        upsert.id,
                        episode.id,
                        strategy.quality_score,
                        strategy.success_count,
                        strategy.failure_count
                    );
                    self.attach_strategy_to_graph(episode, &strategy).await?;

                    // Index into strategy BM25 for multi-signal retrieval
                    let has_code = events.iter().any(|e| e.is_code);
                    {
                        let text = format!(
                            "{} {} {}",
                            strategy.summary, strategy.when_to_use, strategy.action_hint
                        );
                        if !text.trim().is_empty() {
                            let mut idx = self.strategy_bm25_index.write().await;
                            if has_code {
                                idx.index_document_code(upsert.id, &text);
                            } else {
                                idx.index_document(upsert.id, &text);
                            }
                        }
                    }

                    // Fire-and-forget: async LLM refinement + embedding for strategy
                    // Gate on playbook size to avoid refining trivial strategies
                    if let Some(ref refinement) = self.refinement_engine {
                        if strategy.playbook.len() < self.config.min_playbook_steps_for_refinement {
                            tracing::info!(
                            "Skipping LLM refinement for strategy {} (playbook steps={} < min={})",
                            upsert.id,
                            strategy.playbook.len(),
                            self.config.min_playbook_steps_for_refinement
                        );
                        } else {
                            let strategy_id = upsert.id;
                            let strategy_clone = strategy.clone();
                            let store_ref = self.strategy_store.clone();
                            let refinement_ref = refinement.clone();
                            let embedding_client = self.embedding_client.clone();
                            let event_narrative =
                                crate::event_content::build_event_narrative(&events);
                            let bm25_ref = self.strategy_bm25_index.clone();
                            let has_code_for_reindex = has_code;

                            // Gather community context for enrichment
                            let community_ctx = if self.config.enable_context_enrichment {
                                let summaries = self.community_summaries.read().await;
                                if summaries.is_empty() {
                                    None
                                } else {
                                    let topic =
                                        format!("{} {}", strategy.summary, strategy.when_to_use);
                                    let ctx =
                                        crate::context_enrichment::community_context_for_topic(
                                            &topic,
                                            &summaries,
                                            &self.config.enrichment_config,
                                        );
                                    if ctx.is_empty() {
                                        None
                                    } else {
                                        Some(ctx)
                                    }
                                }
                            } else {
                                None
                            };

                            // Gather similar strategies via BM25
                            let similar_strat_ctx = if self.config.enable_context_enrichment {
                                let bm25 = self.strategy_bm25_index.read().await;
                                let hits = bm25.search(&strategy.summary, 4);
                                drop(bm25);
                                let store = self.strategy_store.read().await;
                                let mut strats: Vec<crate::strategies::Strategy> = Vec::new();
                                for (id, _) in hits {
                                    if id != strategy.id {
                                        if let Some(s) = store.get_strategy(id) {
                                            strats.push(s);
                                        }
                                    }
                                }
                                drop(store);
                                if strats.is_empty() {
                                    None
                                } else {
                                    let refs: Vec<&crate::strategies::Strategy> =
                                        strats.iter().collect();
                                    Some(crate::context_enrichment::build_strategy_context(
                                        &refs,
                                        self.config.enrichment_config.max_similar_strategies,
                                    ))
                                }
                            } else {
                                None
                            };

                            tokio::spawn(async move {
                                match refinement_ref
                                    .refine_and_embed_strategy(
                                        &strategy_clone,
                                        embedding_client.as_ref(),
                                        Some(event_narrative),
                                        community_ctx,
                                        similar_strat_ctx,
                                    )
                                    .await
                                {
                                    Ok(refined) => {
                                        // Re-index with refined text
                                        let text = format!(
                                            "{} {} {}",
                                            refined.summary,
                                            refined.when_to_use,
                                            refined.action_hint
                                        );
                                        if !text.trim().is_empty() {
                                            let mut idx = bm25_ref.write().await;
                                            idx.remove_document(strategy_id);
                                            if has_code_for_reindex {
                                                idx.index_document_code(strategy_id, &text);
                                            } else {
                                                idx.index_document(strategy_id, &text);
                                            }
                                        }
                                        let mut store = store_ref.write().await;
                                        if let Err(e) = store.update_strategy(refined) {
                                            tracing::warn!(
                                                "Failed to persist refined strategy {}: {}",
                                                strategy_id,
                                                e
                                            );
                                        }
                                    },
                                    Err(e) => tracing::warn!(
                                        "Strategy refinement failed for {}: {}",
                                        strategy_id,
                                        e
                                    ),
                                }
                            });
                        } // else (playbook size gate)
                    }
                }
            } else {
                tracing::info!(
                    "Strategy extraction produced no strategy for episode_id={}",
                    episode.id
                );
            }
        } else {
            tracing::info!(
                "Strategy extraction skipped episode_id={} (no events)",
                episode.id
            );
        }

        Ok(())
    }

    /// Process episode for reinforcement learning
    pub(super) async fn process_episode_for_reinforcement(
        &self,
        episode: &Episode,
    ) -> GraphResult<()> {
        // Determine success/failure
        let success = matches!(episode.outcome, Some(EpisodeOutcome::Success));

        // Calculate duration from events
        let duration_seconds = {
            let store = self.event_store.read().await;
            if let (Some(start_event), Some(end_event_id)) =
                (store.get(&episode.start_event), episode.end_event)
            {
                if let Some(end_event) = store.get(&end_event_id) {
                    let duration_ns = end_event.timestamp.saturating_sub(start_event.timestamp);
                    (duration_ns as f32) / 1_000_000_000.0
                } else {
                    1.0 // Default
                }
            } else {
                1.0 // Default
            }
        };

        // Calculate metrics
        let metrics = EpisodeMetrics {
            duration_seconds,
            expected_duration_seconds: 5.0, // Default expectation
            quality_score: Some(episode.significance),
            custom_metrics: HashMap::new(),
        };

        // Apply reinforcement
        let mut inference = self.inference.write().await;
        let _result = inference
            .reinforce_patterns(episode, success, Some(metrics))
            .await?;

        self.update_transition_model(episode).await?;

        self.stats
            .total_reinforcements_applied
            .fetch_add(1, AtomicOrdering::Relaxed);

        Ok(())
    }

    /// Process episode for world model training (shadow mode).
    /// Assembles a training tuple and submits it to the critic.
    pub(super) async fn process_episode_for_world_model(
        &self,
        episode: &Episode,
    ) -> GraphResult<()> {
        let Some(ref wm) = self.world_model else {
            return Ok(());
        };

        // Load events for feature extraction
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|id| store.get(id).cloned())
                .collect()
        };

        // Find best matching memory for this episode's context
        let memory = {
            let mut store = self.memory_store.write().await;
            store
                .retrieve_by_context(&episode.context, 1)
                .into_iter()
                .next()
        };

        // Find matching strategy by context hash
        let strategy = {
            let store = self.strategy_store.read().await;
            store
                .get_strategies_for_context(episode.context_signature, 1)
                .into_iter()
                .next()
        };

        // Assemble training tuple
        if let Some(tuple) = world_model::assemble_training_tuple(
            episode,
            &events,
            memory.as_ref(),
            strategy.as_ref(),
        ) {
            let mut wm_guard = wm.write().await;
            wm_guard.submit_training(tuple);
            tracing::debug!(
                "World model training tuple submitted episode_id={} events={} salience={:.3}",
                episode.id,
                events.len(),
                episode.salience_score,
            );
        }

        Ok(())
    }

    /// Write a single extracted fact directly to the graph ledger.
    ///
    /// This is the correct path for compaction-extracted facts:
    ///   1. Resolve entity node (fuzzy match or create)
    ///   2. Walk graph: any active edge for this entity+category?
    ///   3. Same value → skip (idempotent)
    ///   4. Different value → supersede old edge, create new edge
    ///   5. No existing edge → create new edge
    ///
    /// One fact, one graph walk, one edge write. No re-processing.
    /// Write a fact to the graph as an edge. Uses open-ended edge types from LLM extraction.
    ///
    /// Edge type = `{category}:{predicate}` (e.g. `location:lives_in`, `work:works_with`).
    /// Financial facts use special `financial:payment` type with structured amount data.
    ///
    /// No hardcoded supersession — conflict detection is handled separately by
    /// `detect_edge_conflicts` which uses semantic search + LLM comparison
    /// (temporal graph approach).
    pub async fn write_fact_to_graph(
        &self,
        fact: &crate::conversation::compaction::ExtractedFact,
        timestamp: u64,
    ) {
        let category = fact.category.as_deref().unwrap_or("other");
        let is_financial = self.ontology.is_append_only(category);

        // 2. Edge type: canonicalize predicate BEFORE taking the write lock.
        // Try embedding-based canonicalizer first, fall back to sync version.
        let assoc_type = if is_financial {
            let cp = self.ontology.canonical_predicate(category);
            let pred = if cp.is_empty() { "payment" } else { &cp };
            self.ontology.build_edge_type(category, pred)
        } else {
            let canonical = if let Some(ref pc) = self.predicate_canonicalizer {
                pc.canonicalize(category, &fact.predicate).await
            } else {
                self.ontology
                    .canonicalize_predicate(category, &fact.predicate)
                    .into_owned()
            };
            let base = self.ontology.build_edge_type(category, &canonical);
            // Fallback: when Multi-valued category has generic predicate, derive sub-key from object
            if !self.ontology.is_single_valued(category) && canonical == category {
                let sub_key = derive_sub_key(&fact.object);
                if !sub_key.is_empty() {
                    self.ontology.build_edge_type(category, &sub_key)
                } else {
                    base
                }
            } else {
                base
            }
        };

        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // 1. Resolve entity node
        let entity_nid =
            match resolve_or_create_entity(graph, &fact.subject, ConceptType::NamedEntity) {
                Some(nid) => nid,
                None => return,
            };

        let target_ctype = self.ontology.default_target_concept_type(category);

        // 3. Resolve the target value as a concept node
        let target_nid = match resolve_or_create_entity(graph, &fact.object, target_ctype) {
            Some(nid) => nid,
            None => return,
        };

        if is_financial {
            // ── Financial: use LLM-extracted amount + split_with fields ──
            let total_amount = match fact.amount {
                Some(a) if a > 0.0 => a,
                _ => {
                    tracing::debug!(
                        "write_fact_to_graph SKIP financial (no amount) stmt='{}'",
                        fact.statement.get(..80).unwrap_or(&fact.statement)
                    );
                    return;
                },
            };

            let payer_name = normalize_entity_name(&fact.subject);
            let split_with = fact.split_with.as_deref().unwrap_or(&[]);

            let is_all = split_with.iter().any(|s| s.to_lowercase() == "all");
            let beneficiaries: Vec<String> = if is_all {
                let mut participants: Vec<String> = Vec::new();
                for (name, _) in graph.concept_index.iter() {
                    if let Some(&nid) = graph.concept_index.get(name) {
                        let has_financial = graph.get_edges_from(nid).iter().any(|e| {
                            matches!(&e.edge_type, crate::structures::EdgeType::Association { association_type, .. }
                                if crate::ontology::OntologyRegistry::parse_edge_category(association_type)
                                    .map(|c| self.ontology.is_append_only(c)).unwrap_or(false))
                        }) || graph.get_edges_to(nid).iter().any(|e| {
                            matches!(&e.edge_type, crate::structures::EdgeType::Association { association_type, .. }
                                if crate::ontology::OntologyRegistry::parse_edge_category(association_type)
                                    .map(|c| self.ontology.is_append_only(c)).unwrap_or(false))
                        });
                        if has_financial {
                            let norm = normalize_entity_name(name);
                            if norm != payer_name && !participants.contains(&norm) {
                                participants.push(norm);
                            }
                        }
                    }
                }
                participants
            } else {
                split_with
                    .iter()
                    .map(|s| normalize_entity_name(s))
                    .collect()
            };

            if beneficiaries.is_empty() {
                let mut edge = GraphEdge::new(
                    entity_nid,
                    target_nid,
                    crate::structures::EdgeType::Association {
                        association_type: self.ontology.build_edge_type("financial", "payment"),
                        evidence_count: 1,
                        statistical_significance: 0.9,
                    },
                    0.9,
                );
                edge.valid_from = Some(timestamp);
                edge.properties
                    .insert("amount".to_string(), serde_json::json!(total_amount));
                edge.properties.insert(
                    "statement".to_string(),
                    serde_json::Value::String(fact.statement.clone()),
                );
                let new_eid = graph.add_edge(edge);
                tracing::info!(
                    "write_fact_to_graph financial eid={:?} '{}' -> '{}' amount={:.2}",
                    new_eid,
                    fact.subject,
                    fact.object,
                    total_amount
                );
            } else {
                let num_people = beneficiaries.len() as f64 + 1.0;
                let per_person = total_amount / num_people;
                for beneficiary_name in &beneficiaries {
                    let ben_nid = match resolve_or_create_entity(
                        graph,
                        beneficiary_name,
                        ConceptType::NamedEntity,
                    ) {
                        Some(nid) => nid,
                        None => continue,
                    };
                    let mut edge = GraphEdge::new(
                        entity_nid,
                        ben_nid,
                        crate::structures::EdgeType::Association {
                            association_type: self.ontology.build_edge_type("financial", "payment"),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        0.9,
                    );
                    edge.valid_from = Some(timestamp);
                    edge.properties
                        .insert("amount".to_string(), serde_json::json!(per_person));
                    edge.properties.insert(
                        "statement".to_string(),
                        serde_json::Value::String(fact.statement.clone()),
                    );
                    let new_eid = graph.add_edge(edge);
                    tracing::info!("write_fact_to_graph financial eid={:?} '{}' -> '{}' amount={:.2} (total={:.2}/{})",
                        new_eid, fact.subject, beneficiary_name, per_person, total_amount, num_people as u64);
                }
            }
        } else {
            // ── General fact: create edge, dedup exact matches ──

            // Check for exact duplicate (same entity → same target, same edge type, still active)
            let already_exists = graph.get_edges_from(entity_nid).iter().any(|e| {
                if let crate::structures::EdgeType::Association {
                    association_type, ..
                } = &e.edge_type
                {
                    association_type == &assoc_type
                        && e.target == target_nid
                        && e.valid_until.is_none()
                } else {
                    false
                }
            });
            if already_exists {
                tracing::debug!(
                    "write_fact_to_graph SKIP duplicate '{}' '{}'->'{}'",
                    assoc_type,
                    fact.subject,
                    fact.object
                );
                return;
            }

            // ── Supersede active edges, scoped by cardinality ──
            // Single-valued categories (location, work, education): ALWAYS supersede
            // existing active edges — don't rely on LLM setting is_update.
            // Multi-valued: only supersede when is_update is explicitly true.
            let single_valued = self.ontology.is_single_valued(category);
            if single_valued || fact.is_update.unwrap_or(false) {
                let category_prefix = format!("{}:", category);
                let edges_to_supersede: Vec<crate::structures::EdgeId> = graph
                    .get_edges_from(entity_nid)
                    .iter()
                    .filter(|e| {
                        if let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &e.edge_type
                        {
                            if e.valid_until.is_some() || e.target == target_nid {
                                return false;
                            }
                            if single_valued {
                                // Single-valued: supersede all active edges in the category
                                association_type.starts_with(&category_prefix)
                            } else {
                                // Multi-valued: only supersede edges with exact same assoc_type
                                association_type == &assoc_type
                            }
                        } else {
                            false
                        }
                    })
                    .map(|e| e.id)
                    .collect();

                // Collect target names before mutation (read-only pass)
                let supersede_info: Vec<(crate::structures::EdgeId, String)> = edges_to_supersede
                    .iter()
                    .filter_map(|eid| {
                        graph.edges.get(*eid).map(|e| {
                            let name = crate::conversation::graph_projection::concept_name_of(
                                graph, e.target,
                            )
                            .unwrap_or_default();
                            (*eid, name)
                        })
                    })
                    .collect();

                for (eid, old_target) in &supersede_info {
                    if let Some(e) = graph.edges.get_mut(*eid) {
                        e.valid_until = Some(timestamp);
                        graph.dirty_edges.insert(*eid);
                        tracing::info!(
                            "write_fact_to_graph is_update=true superseded '{}' -> '{}' (replaced by '{}')",
                            assoc_type, old_target, fact.object
                        );
                    }
                }

                // ── depends_on cascade: supersede edges whose depends_on references
                // the superseded target. E.g., "visits Meiji Shrine" depends_on
                // "User lives in Tokyo" — when Tokyo is superseded, Meiji Shrine goes too.
                if !edges_to_supersede.is_empty() {
                    // Collect superseded target names (read-only pass)
                    let superseded_targets: Vec<String> = edges_to_supersede
                        .iter()
                        .filter_map(|eid| graph.edges.get(*eid))
                        .filter_map(|e| {
                            crate::conversation::graph_projection::concept_name_of(graph, e.target)
                        })
                        .collect();

                    if !superseded_targets.is_empty() {
                        // Collect edges to cascade-supersede and their info (read-only pass)
                        let dep_edges_info: Vec<(crate::structures::EdgeId, String, String)> =
                            graph
                                .get_edges_from(entity_nid)
                                .iter()
                                .filter(|e| {
                                    if e.valid_until.is_some() {
                                        return false;
                                    }
                                    if let Some(serde_json::Value::String(dep)) =
                                        e.properties.get("depends_on")
                                    {
                                        let dep_lower = dep.to_lowercase();
                                        superseded_targets
                                            .iter()
                                            .any(|t| dep_lower.contains(&t.to_lowercase()))
                                    } else {
                                        false
                                    }
                                })
                                .map(|e| {
                                    let target_name =
                                        crate::conversation::graph_projection::concept_name_of(
                                            graph, e.target,
                                        )
                                        .unwrap_or_default();
                                    let dep_text = e
                                        .properties
                                        .get("depends_on")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?")
                                        .to_string();
                                    (e.id, target_name, dep_text)
                                })
                                .collect();

                        // Mutation pass
                        for (eid, dep_target, dep_text) in &dep_edges_info {
                            if let Some(e) = graph.edges.get_mut(*eid) {
                                e.valid_until = Some(timestamp);
                                graph.dirty_edges.insert(*eid);
                                tracing::info!(
                                    "write_fact_to_graph depends_on cascade: superseded edge -> '{}' (depends_on: '{}')",
                                    dep_target, dep_text
                                );
                            }
                        }
                    }
                }
            }

            // Create new edge
            let mut edge = GraphEdge::new(
                entity_nid,
                target_nid,
                crate::structures::EdgeType::Association {
                    association_type: assoc_type.clone(),
                    evidence_count: 1,
                    statistical_significance: 0.9,
                },
                0.9,
            );
            edge.valid_from = Some(timestamp);
            edge.properties.insert(
                "statement".to_string(),
                serde_json::Value::String(fact.statement.clone()),
            );
            edge.properties.insert(
                "value".to_string(),
                serde_json::Value::String(fact.object.clone()),
            );
            if let Some(cat) = &fact.category {
                edge.properties.insert(
                    "category".to_string(),
                    serde_json::Value::String(cat.clone()),
                );
            }
            if let Some(ref dep) = fact.depends_on {
                edge.properties.insert(
                    "depends_on".to_string(),
                    serde_json::Value::String(dep.clone()),
                );
            }
            if let Some(ref ts) = fact.temporal_signal {
                edge.properties.insert(
                    "temporal_signal".to_string(),
                    serde_json::Value::String(ts.clone()),
                );
            }
            // Wire sentiment from extraction to edge properties (preference facts)
            if let Some(sent) = fact.sentiment {
                edge.properties
                    .insert("sentiment".to_string(), serde_json::json!(sent));
            }
            let new_eid = graph.add_edge(edge);
            tracing::info!(
                "write_fact_to_graph edge eid={:?} '{}' '{}'->'{}'",
                new_eid,
                assoc_type,
                fact.subject,
                fact.object
            );

            // Auto-register unknown categories in the domain registry
            if self.ontology.resolve(category).is_none() && category != "other" {
                let cardinality = fact
                    .cardinality_hint
                    .as_deref()
                    .map(|h| match h {
                        "single" => crate::domain_schema::Cardinality::Single,
                        "append" => crate::domain_schema::Cardinality::Append,
                        _ => crate::domain_schema::Cardinality::Multi,
                    })
                    .unwrap_or(crate::domain_schema::Cardinality::Multi);
                if self
                    .ontology
                    .register_category(category, cardinality, "", false)
                {
                    tracing::info!(
                        "Auto-registered domain category '{}' with cardinality {:?}",
                        category,
                        cardinality
                    );
                    // Persist as concept node (backward compat)
                    if let Some(slot) = self
                        .ontology
                        .learned_slots()
                        .iter()
                        .find(|s| s.category == category)
                    {
                        crate::domain_schema::persist_domain_slot(graph, slot);
                    }
                }
            } else {
                self.ontology.record_usage(category);
            }

            // For symmetric properties, also create the reverse edge
            if self.ontology.is_symmetric(category) {
                let rev_exists = graph.get_edges_from(target_nid).iter().any(|e| {
                    if let crate::structures::EdgeType::Association {
                        association_type, ..
                    } = &e.edge_type
                    {
                        association_type == &assoc_type
                            && e.target == entity_nid
                            && e.valid_until.is_none()
                    } else {
                        false
                    }
                });
                if !rev_exists {
                    let mut rev_edge = GraphEdge::new(
                        target_nid,
                        entity_nid,
                        crate::structures::EdgeType::Association {
                            association_type: assoc_type.clone(),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        0.9,
                    );
                    rev_edge.valid_from = Some(timestamp);
                    rev_edge.properties.insert(
                        "statement".to_string(),
                        serde_json::Value::String(fact.statement.clone()),
                    );
                    graph.add_edge(rev_edge);
                }
            }
        }
    }

    /// Detect and resolve conflicts between a new fact and existing edges.
    ///
    /// Uses semantic search on the entity's existing edges to find potentially
    /// conflicting facts, then asks the LLM if the new fact contradicts/supersedes
    /// any of them (temporal graph approach). Conflicting edges get `valid_until` set.
    pub async fn detect_edge_conflicts(
        &self,
        fact: &crate::conversation::compaction::ExtractedFact,
        timestamp: u64,
    ) {
        // Skip append-only/skip-conflict-detection properties
        let category = fact.category.as_deref().unwrap_or("other");
        if self.ontology.skip_conflict_detection(category) {
            return;
        }

        let entity_name = normalize_entity_name(&fact.subject);

        // 1. Collect existing active edges from this entity with statement text
        let candidates: Vec<(crate::structures::EdgeId, String)> = {
            let inference = self.inference.read().await;
            let graph = inference.graph();

            let entity_nid = match graph.concept_index.get(&*entity_name) {
                Some(&nid) => nid,
                None => return,
            };

            graph
                .get_edges_from(entity_nid)
                .iter()
                .filter(|e| e.valid_until.is_none()) // only active edges
                .filter(|e| {
                    matches!(
                        &e.edge_type,
                        crate::structures::EdgeType::Association { .. }
                    )
                })
                .filter_map(|e| {
                    let stmt = e
                        .properties
                        .get("statement")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())?;
                    Some((e.id, stmt))
                })
                .collect()
        };

        if candidates.is_empty() {
            return;
        }

        // 2. Find semantically similar existing facts using simple heuristic first:
        //    same category prefix = potential conflict candidate
        let new_statement = &fact.statement;
        let conflict_candidates: Vec<(crate::structures::EdgeId, String)> = {
            let inference = self.inference.read().await;
            let graph = inference.graph();

            candidates
                .iter()
                .filter(|(eid, _stmt)| {
                    if let Some(e) = graph.edges.get(*eid) {
                        if let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &e.edge_type
                        {
                            // Same category prefix = potential conflict
                            let prefix = association_type.split(':').next().unwrap_or("");
                            prefix == category
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                })
                .cloned()
                .collect()
        };

        if conflict_candidates.is_empty() {
            return;
        }

        // 3. Use LLM to check if new fact contradicts/supersedes any existing fact
        let llm = match &self.unified_llm_client {
            Some(llm) => llm.clone(),
            None => {
                // No LLM available — fall back to simple negation check
                let mut inference = self.inference.write().await;
                let graph = inference.graph_mut();
                for (eid, old_stmt) in &conflict_candidates {
                    if crate::maintenance::is_contradiction(new_statement, old_stmt) {
                        if let Some(e) = graph.edges.get_mut(*eid) {
                            e.valid_until = Some(timestamp);
                            graph.dirty_edges.insert(*eid);
                        }
                        tracing::info!("write_fact_to_graph CONFLICT (negation) invalidated eid={} old='{}' new='{}'",
                            eid, old_stmt.get(..60).unwrap_or(old_stmt), new_statement.get(..60).unwrap_or(new_statement));
                    }
                }
                return;
            },
        };

        // Build LLM prompt with all candidates
        let existing_facts: String = conflict_candidates
            .iter()
            .enumerate()
            .map(|(i, (_, stmt))| format!("{}. {}", i + 1, stmt))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Given a NEW fact and EXISTING facts about the same entity, identify which existing facts are contradicted or superseded by the new fact.\n\n\
            EXISTING FACTS:\n{}\n\n\
            NEW FACT: {}\n\n\
            Return a JSON array of the numbers of facts that are CONTRADICTED or SUPERSEDED by the new fact.\n\
            If the new fact does NOT contradict any existing fact (e.g. both can be true simultaneously), return an empty array [].\n\
            Examples:\n\
            - \"User lives in Lisbon\" superseded by \"User moved to NYC\" → [1]\n\
            - \"User works with Alice\" NOT superseded by \"User works with Bob\" → []\n\
            - \"User likes coffee\" superseded by \"User no longer drinks coffee\" → [1]\n\
            Output ONLY the JSON array, nothing else.",
            existing_facts, new_statement
        );

        let request = crate::llm_client::LlmRequest {
            system_prompt:
                "You detect factual contradictions. Output only a JSON array of numbers."
                    .to_string(),
            user_prompt: prompt,
            temperature: 0.0,
            max_tokens: 50,
            json_mode: true,
        };

        match tokio::time::timeout(std::time::Duration::from_secs(10), llm.complete(request)).await
        {
            Ok(Ok(response)) => {
                // Parse the JSON array of conflicting fact indices
                if let Ok(indices) = serde_json::from_str::<Vec<usize>>(&response.content) {
                    if !indices.is_empty() {
                        let mut inference = self.inference.write().await;
                        let graph = inference.graph_mut();
                        for idx in indices {
                            if idx > 0 && idx <= conflict_candidates.len() {
                                let (eid, old_stmt) = &conflict_candidates[idx - 1]; // 1-indexed
                                if let Some(e) = graph.edges.get_mut(*eid) {
                                    e.valid_until = Some(timestamp);
                                    graph.dirty_edges.insert(*eid);
                                }
                                tracing::info!(
                                    "write_fact_to_graph CONFLICT (LLM) invalidated eid={} old='{}' new='{}'",
                                    eid,
                                    old_stmt.get(..60).unwrap_or(old_stmt),
                                    new_statement.get(..60).unwrap_or(new_statement)
                                );
                            }
                        }
                    }
                }
            },
            Ok(Err(e)) => {
                tracing::debug!("Conflict detection LLM call failed: {}", e);
            },
            Err(_) => {
                tracing::debug!("Conflict detection LLM call timed out");
            },
        }
    }

    /// Process episode for automatic state tracking and ledger extraction.
    ///
    /// Scans episode events for state-change and transaction signals, then
    /// updates the structured memory store.
    pub(super) async fn process_episode_for_state_tracking(
        &self,
        episode: &Episode,
    ) -> GraphResult<()> {
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|id| store.get(id).cloned())
                .collect()
        };

        tracing::info!(
            "STATE_TRACKING episode_id={} event_count={} event_ids={:?}",
            episode.id,
            events.len(),
            episode.events.iter().take(5).collect::<Vec<_>>()
        );

        if events.is_empty() {
            tracing::info!(
                "STATE_TRACKING episode_id={} SKIPPED (no events found in store)",
                episode.id
            );
            return Ok(());
        }

        // Pre-resolve all entity names to NodeIds while holding inference lock,
        // then drop it before acquiring structured_memory write lock to avoid deadlock.
        struct StateCandidate {
            entity: String,
            attribute: String,
            new_state: String,
            node_id: NodeId,
            timestamp: u64,
            /// Semantic category for supersession grouping (e.g. "location", "routine").
            /// Facts with the same entity + category supersede each other.
            category: Option<String>,
        }
        struct LedgerCandidate {
            from: String,
            to: String,
            from_id: NodeId,
            to_id: NodeId,
            amount: f64,
            description: String,
            direction: crate::structured_memory::LedgerDirection,
            timestamp: u64,
        }
        struct RelationshipCandidate {
            subject: String,
            object: String,
            relation_type: String,
        }
        struct PreferenceCandidate {
            entity: String,
            item: String,
            category: String,
            sentiment: f64,
        }

        // Track Event→Concept mentions for creating About edges later.
        // Maps event_graph_node_id → Vec<concept_node_id>.
        let mut event_concept_mentions: HashSet<(NodeId, NodeId)> = HashSet::new();

        let (
            mut state_candidates,
            ledger_candidates,
            relationship_candidates,
            preference_candidates,
        ) = {
            let mut inference = self.inference.write().await;

            // Helper: ensure a concept node exists in the graph, creating it if missing.
            // Uses fuzzy entity resolution to match "coffee shop" = "Coffee Shop" etc.
            let ensure_node =
                |graph: &mut Graph, name: &str, ctype: ConceptType| -> Option<NodeId> {
                    resolve_or_create_entity(graph, name, ctype)
                };

            let graph = inference.graph_mut();

            let mut state_candidates = Vec::new();
            let mut ledger_candidates = Vec::new();
            let mut relationship_candidates = Vec::new();
            let mut preference_candidates = Vec::new();

            for event in &events {
                // Look up this event's graph node ID for creating About edges
                let event_graph_nid = graph.event_index.get(&event.id).copied();

                // Scan event content for known concept names → track for About edges.
                // This connects Conversation events to entity Concept nodes.
                if let Some(egnid) = event_graph_nid {
                    let content = match &event.event_type {
                        EventType::Conversation { content, .. } => Some(content.as_str()),
                        EventType::Action { action_name, .. } => Some(action_name.as_str()),
                        _ => None,
                    };
                    if let Some(text) = content {
                        let lower = text.to_lowercase();
                        for (concept_name, &concept_nid) in graph.concept_index.iter() {
                            if lower.contains(&concept_name.to_lowercase()) {
                                event_concept_mentions.insert((egnid, concept_nid));
                            }
                        }
                    }
                }

                // ---- Normalize metadata keys via alias matching ----
                let event_type_name = match &event.event_type {
                    EventType::Conversation { .. } => "Conversation",
                    EventType::Observation {
                        observation_type, ..
                    } => observation_type.as_str(),
                    EventType::Action { action_name, .. } => action_name.as_str(),
                    EventType::Context { context_type, .. } => context_type.as_str(),
                    _ => "other",
                };
                let meta_keys: Vec<&str> = event.metadata.keys().map(|k| k.as_str()).collect();
                tracing::info!(
                    "STATE_TRACKING event_id={} type={} graph_nid={:?} metadata_keys={:?}",
                    event.id,
                    event_type_name,
                    event_graph_nid,
                    meta_keys
                );
                let mut normalized = self.metadata_normalizer.normalize(&event.metadata);

                // LLM fallback: if alias resolved < 2 roles and event has >= 2 metadata keys
                if normalized.roles.len() < 2 && event.metadata.len() >= 2 {
                    if let Some(ref llm) = self.metadata_llm_normalizer {
                        let pairs: Vec<(String, String)> = event
                            .metadata
                            .iter()
                            .map(|(k, v)| (k.clone(), metadata_value_preview(v)))
                            .collect();
                        match tokio::time::timeout(
                            Duration::from_millis(self.config.metadata_normalization_timeout_ms),
                            llm.normalize_keys(&pairs),
                        )
                        .await
                        {
                            Ok(Ok(Some(mappings))) => {
                                normalized = self.metadata_normalizer.apply_llm_mappings(
                                    &mappings,
                                    &event.metadata,
                                    &normalized,
                                );
                            },
                            Ok(Ok(None)) => {},
                            Ok(Err(e)) => {
                                tracing::warn!("Metadata LLM normalizer error: {}", e)
                            },
                            Err(_) => {
                                tracing::warn!("Metadata LLM normalizer timed out")
                            },
                        }
                    }
                }

                // ---- State change detection ----
                let entity_name = normalized.get_str(MetadataRole::Entity, &event.metadata);
                let new_state = normalized.get_str(MetadataRole::NewState, &event.metadata);
                let _old_state = normalized.get_str(MetadataRole::OldState, &event.metadata);

                // Recognized event patterns for state changes
                let is_state_event = matches!(&event.event_type, EventType::Context { context_type, .. } if context_type == "state_update")
                    || event.metadata.contains_key("entity_state")
                    || event.metadata.contains_key("new_state")
                    || (normalized.has_role(MetadataRole::Entity)
                        && normalized.has_role(MetadataRole::NewState))
                    || match &event.event_type {
                        EventType::Action { action_name, .. } => {
                            action_name.contains("update_status")
                                || action_name.contains("set_state")
                                || action_name.contains("transition")
                        },
                        EventType::Observation {
                            observation_type, ..
                        } => observation_type == "state_change",
                        _ => false,
                    };

                tracing::info!(
                    "STATE_TRACKING event_id={} entity={:?} new_state={:?} is_state_event={}",
                    event.id,
                    entity_name.as_deref().unwrap_or("NONE"),
                    new_state.as_deref().unwrap_or("NONE"),
                    is_state_event
                );

                // Section E guards: ALL must pass
                if let (Some(ref entity), Some(ref new_st)) = (&entity_name, &new_state) {
                    if !entity.is_empty()
                        && !new_st.is_empty()
                        && is_state_event
                        && event.timestamp > 0
                    {
                        if let Some(node_id) = ensure_node(graph, entity, ConceptType::NamedEntity)
                        {
                            if let Some(egnid) = event_graph_nid {
                                event_concept_mentions.insert((egnid, node_id));
                            }
                            let _trigger = match &event.event_type {
                                EventType::Action { action_name, .. } => action_name.clone(),
                                _ => "auto".to_string(),
                            };
                            // Derive attribute from metadata if available, otherwise
                            // infer from context_type or event metadata keys.
                            let attribute = metadata_str(&event.metadata, "attribute")
                                .or_else(|| match &event.event_type {
                                    EventType::Context { context_type, .. } => {
                                        Some(context_type.clone())
                                    },
                                    _ => None,
                                })
                                .unwrap_or_else(|| "other".to_string());
                            let category = metadata_str(&event.metadata, "category");
                            tracing::info!(
                                "STATE_TRACKING CANDIDATE entity='{}' attr='{}' new_state='{}' category={:?} node_id={}",
                                entity, attribute, new_st, category, node_id
                            );
                            state_candidates.push(StateCandidate {
                                entity: entity.clone(),
                                attribute,
                                new_state: new_st.clone(),
                                node_id,
                                timestamp: event.timestamp,
                                category,
                            });
                        }
                    } else {
                        tracing::info!(
                            "STATE_TRACKING REJECTED entity_empty={} state_empty={} is_state_event={} ts={}",
                            entity.is_empty(),
                            new_st.is_empty(),
                            is_state_event,
                            event.timestamp
                        );
                    }
                }

                // ---- Ledger/transaction detection ----
                let amount = normalized.get_f64(MetadataRole::Amount, &event.metadata);
                let from_entity = normalized.get_str(MetadataRole::From, &event.metadata);
                let to_entity = normalized.get_str(MetadataRole::To, &event.metadata);

                let is_transaction_event = event.metadata.contains_key("amount")
                    || event.metadata.contains_key("transaction")
                    || (normalized.has_role(MetadataRole::Amount)
                        && (normalized.has_role(MetadataRole::From)
                            || normalized.has_role(MetadataRole::To)));

                if let (Some(amt), Some(ref from), Some(ref to)) =
                    (amount, &from_entity, &to_entity)
                {
                    if !from.is_empty()
                        && !to.is_empty()
                        && is_transaction_event
                        && event.timestamp > 0
                        && amt.is_finite()
                    {
                        let from_id = ensure_node(graph, from, ConceptType::NamedEntity);
                        let to_id = ensure_node(graph, to, ConceptType::NamedEntity);

                        if let (Some(fid), Some(tid)) = (from_id, to_id) {
                            if let Some(egnid) = event_graph_nid {
                                event_concept_mentions.insert((egnid, fid));
                                event_concept_mentions.insert((egnid, tid));
                            }
                            let description = normalized
                                .get_str(MetadataRole::Description, &event.metadata)
                                .unwrap_or_else(|| "auto-extracted".to_string());

                            // Detect direction from metadata, default to Credit
                            let direction = normalized
                                .get_str(MetadataRole::Direction, &event.metadata)
                                .map(|d| {
                                    if d.eq_ignore_ascii_case("debit") {
                                        crate::structured_memory::LedgerDirection::Debit
                                    } else {
                                        crate::structured_memory::LedgerDirection::Credit
                                    }
                                })
                                .unwrap_or(crate::structured_memory::LedgerDirection::Credit);

                            ledger_candidates.push(LedgerCandidate {
                                from: from.clone(),
                                to: to.clone(),
                                from_id: fid,
                                to_id: tid,
                                amount: amt,
                                description,
                                direction,
                                timestamp: event.timestamp,
                            });
                        } else {
                            tracing::debug!(
                                "Ledger tracking: entities '{}' or '{}' not found in graph",
                                from,
                                to
                            );
                        }
                    }
                }

                // ---- Relationship detection ----
                if event.metadata.contains_key("relationship") {
                    let subject = event
                        .metadata
                        .get("subject")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let object = event
                        .metadata
                        .get("object")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let relation_type = event
                        .metadata
                        .get("relation_type")
                        .and_then(crate::metadata_normalize::metadata_as_str);

                    if let (Some(subj), Some(obj), Some(rel)) = (subject, object, relation_type) {
                        if !subj.is_empty() && !obj.is_empty() && !rel.is_empty() {
                            // Track mentions for About edges (concept IDs resolved in Phase B)
                            if let Some(egnid) = event_graph_nid {
                                if let Some(&snid) = graph.concept_index.get(subj.as_str()) {
                                    event_concept_mentions.insert((egnid, snid));
                                }
                                if let Some(&onid) = graph.concept_index.get(obj.as_str()) {
                                    event_concept_mentions.insert((egnid, onid));
                                }
                            }
                            relationship_candidates.push(RelationshipCandidate {
                                subject: subj,
                                object: obj,
                                relation_type: rel,
                            });
                        }
                    }
                }

                // ---- Preference detection ----
                if event.metadata.contains_key("preference") {
                    let entity = event
                        .metadata
                        .get("entity")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let item = event
                        .metadata
                        .get("item")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let category = event
                        .metadata
                        .get("category")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let sentiment = event
                        .metadata
                        .get("sentiment")
                        .and_then(crate::metadata_normalize::metadata_as_f64)
                        .unwrap_or(0.5);

                    if let (Some(ent), Some(itm), Some(cat)) = (entity, item, category) {
                        if !ent.is_empty() && !itm.is_empty() {
                            if let Some(egnid) = event_graph_nid {
                                if let Some(&enid) = graph.concept_index.get(ent.as_str()) {
                                    event_concept_mentions.insert((egnid, enid));
                                }
                                if let Some(&inid) = graph.concept_index.get(itm.as_str()) {
                                    event_concept_mentions.insert((egnid, inid));
                                }
                            }
                            preference_candidates.push(PreferenceCandidate {
                                entity: ent,
                                item: itm,
                                category: cat,
                                sentiment,
                            });
                        }
                    }
                }
            }

            (
                state_candidates,
                ledger_candidates,
                relationship_candidates,
                preference_candidates,
            )
        };
        // inference lock is now dropped

        // Apply state changes, ledger entries, relationships, and preferences
        let has_work = !state_candidates.is_empty()
            || !ledger_candidates.is_empty()
            || !relationship_candidates.is_empty()
            || !preference_candidates.is_empty();
        if has_work {
            // ---- Phase A: Write to StructuredMemoryStore (iterate by reference) ----
            {
                let mut sm = self.structured_memory.write().await;

                // State transitions are now tracked exclusively via graph edges (Phase B).
                // StructuredMemoryStore state machines are no longer populated here.
                // The graph IS the state machine — transitions are edges with valid_from timestamps.

                for lc in &ledger_candidates {
                    let key = crate::structured_memory::ledger_key(lc.from_id, lc.to_id);

                    if sm.get(&key).is_none() {
                        sm.upsert(
                            &key,
                            crate::structured_memory::MemoryTemplate::Ledger {
                                entity_pair: (lc.from.clone(), lc.to.clone()),
                                entries: vec![],
                                balance: 0.0,
                                provenance:
                                    crate::structured_memory::MemoryProvenance::EpisodePipeline,
                            },
                        );
                    }

                    let entry = crate::structured_memory::LedgerEntry {
                        timestamp: lc.timestamp,
                        amount: lc.amount,
                        description: lc.description.clone(),
                        direction: lc.direction.clone(),
                    };

                    if let Err(e) = sm.ledger_append(&key, entry) {
                        tracing::debug!("Ledger append failed for {}: {}", key, e);
                    } else {
                        tracing::debug!(
                            "Auto ledger entry: {} -> {} amount={} (key={})",
                            lc.from,
                            lc.to,
                            lc.amount,
                            key
                        );
                    }
                }

                // Relationships are stored as graph edges only — no StructuredMemoryStore tree.
                // Tree projections are built at query time by walking graph edges (graph_projection.rs).
                for rc in &relationship_candidates {
                    tracing::debug!(
                        "Auto relationship: {} <-> {} ({})",
                        rc.subject,
                        rc.object,
                        rc.relation_type
                    );
                }

                for pc in &preference_candidates {
                    let key = format!("prefs:{}:{}", pc.entity, pc.category);
                    if sm.get(&key).is_none() {
                        sm.upsert(
                            &key,
                            crate::structured_memory::MemoryTemplate::PreferenceList {
                                entity: pc.entity.clone(),
                                ranked_items: vec![],
                                provenance:
                                    crate::structured_memory::MemoryProvenance::EpisodePipeline,
                            },
                        );
                    }
                    let rank =
                        if let Some(crate::structured_memory::MemoryTemplate::PreferenceList {
                            ranked_items,
                            ..
                        }) = sm.get(&key)
                        {
                            ranked_items.len()
                        } else {
                            0
                        };
                    if let Err(e) = sm.preference_update(&key, &pc.item, rank, Some(pc.sentiment)) {
                        tracing::debug!("Preference update failed for {}: {}", key, e);
                    } else {
                        tracing::debug!(
                            "Auto preference: {} -> {} (category={}, sentiment={})",
                            pc.entity,
                            pc.item,
                            pc.category,
                            pc.sentiment
                        );
                    }
                }
            }
            // structured_memory lock dropped

            tracing::info!(
                "STATE_TRACKING PHASE_A_DONE episode_id={} state_candidates={} ledger_candidates={} relationship_candidates={} preference_candidates={} about_edge_mentions={}",
                episode.id,
                state_candidates.len(),
                ledger_candidates.len(),
                relationship_candidates.len(),
                preference_candidates.len(),
                event_concept_mentions.len()
            );

            // ---- Phase B: Create graph edges for each candidate ----
            {
                let mut inference = self.inference.write().await;

                tracing::info!(
                    "STATE_TRACKING PHASE_B_START episode_id={} state={} ledger={} relationship={} preference={} about={}",
                    episode.id,
                    state_candidates.len(),
                    ledger_candidates.len(),
                    relationship_candidates.len(),
                    preference_candidates.len(),
                    event_concept_mentions.len()
                );

                // Helper: ensure a concept node exists, returning its NodeId.
                // Uses fuzzy entity resolution to avoid duplicate nodes.
                let ensure_concept =
                    |graph: &mut Graph, name: &str, ctype: ConceptType| -> Option<NodeId> {
                        resolve_or_create_entity(graph, name, ctype)
                    };

                // ---- State changes → graph edges ----
                // Use category for supersession: facts with the same entity + category
                // supersede each other (latest wins). Falls back to attribute if no category.
                //
                // Deduplicate: keep only the LATEST candidate per (entity, supersession_key).
                // This prevents edge bloat when the episode is reprocessed with accumulated events.
                {
                    let mut best: std::collections::HashMap<(String, String), usize> =
                        std::collections::HashMap::new();
                    for (i, sc) in state_candidates.iter().enumerate() {
                        let key = (
                            sc.entity.to_lowercase(),
                            sc.category.as_deref().unwrap_or(&sc.attribute).to_string(),
                        );
                        match best.entry(key) {
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(i);
                            },
                            std::collections::hash_map::Entry::Occupied(mut e) => {
                                if sc.timestamp > state_candidates[*e.get()].timestamp {
                                    e.insert(i);
                                }
                            },
                        }
                    }
                    let keep: std::collections::HashSet<usize> = best.into_values().collect();
                    let deduped: Vec<_> = state_candidates
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| keep.contains(i))
                        .map(|(_, sc)| sc)
                        .collect();
                    state_candidates = deduped;
                }
                tracing::info!(
                    "PHASE_B state candidates after dedup: {}",
                    state_candidates.len()
                );

                let mut state_edges_created = 0usize;
                let mut state_edges_superseded = 0usize;
                for (idx, sc) in state_candidates.iter().enumerate() {
                    let graph = inference.graph_mut();
                    let entity_nid = sc.node_id;

                    let attribute = &sc.attribute;
                    // Edge type uses category when available, attribute as fallback
                    let supersession_key = sc.category.as_deref().unwrap_or(attribute);
                    let cp = self.ontology.canonical_predicate(supersession_key);
                    let predicate = if cp.is_empty() { supersession_key } else { &cp };
                    let assoc_type = self.ontology.build_edge_type(supersession_key, predicate);

                    tracing::info!(
                        "PHASE_B state[{}/{}] entity='{}' nid={} attr='{}' category={:?} supersession_key='{}' new_state='{}' ts={}",
                        idx + 1,
                        state_candidates.len(),
                        sc.entity,
                        entity_nid,
                        attribute,
                        sc.category,
                        supersession_key,
                        sc.new_state,
                        sc.timestamp
                    );

                    // Ensure concept node for the new state value
                    let state_ctype = self.ontology.default_target_concept_type(supersession_key);
                    let state_nid = match ensure_concept(graph, &sc.new_state, state_ctype) {
                        Some(nid) => nid,
                        None => continue,
                    };

                    // Check for idempotency: if there's already an active edge
                    // with the same type pointing to the same target, skip it.
                    // This prevents edge bloat when the same episode is reprocessed.
                    let is_cascade_trigger = self.ontology.triggers_cascade(supersession_key);
                    let mut already_exists = false;
                    let mut edges_to_supersede: Vec<EdgeId> = Vec::new();
                    for edge in graph.get_edges_from(entity_nid).iter() {
                        if let EdgeType::Association {
                            association_type, ..
                        } = &edge.edge_type
                        {
                            if edge.valid_until.is_none() {
                                if association_type == &assoc_type {
                                    if edge.target == state_nid {
                                        // Exact same active edge already exists — idempotent skip
                                        already_exists = true;
                                    } else {
                                        // Different target — must supersede
                                        edges_to_supersede.push(edge.id);
                                    }
                                } else if is_cascade_trigger {
                                    // Ontology-driven cascade: supersede edges whose
                                    // category is cascade-dependent.
                                    let edge_category: String = if let Some(rest) =
                                        association_type.strip_prefix("state:")
                                    {
                                        self.ontology
                                            .decode_legacy_state_assoc(association_type)
                                            .map(|(cat, _)| cat)
                                            .unwrap_or_else(|| rest.to_string())
                                    } else {
                                        association_type.split(':').next().unwrap_or("").to_string()
                                    };
                                    if self.ontology.is_cascade_dependent(&edge_category) {
                                        edges_to_supersede.push(edge.id);
                                    }
                                }
                            }
                        }
                    }

                    if already_exists && edges_to_supersede.is_empty() {
                        tracing::info!(
                            "PHASE_B SKIP (idempotent) '{}' entity={} -> state='{}' already active",
                            assoc_type,
                            sc.entity,
                            sc.new_state
                        );
                        continue;
                    }

                    if !edges_to_supersede.is_empty() {
                        tracing::info!(
                            "PHASE_B superseding {} active '{}' edges from entity nid={}",
                            edges_to_supersede.len(),
                            assoc_type,
                            entity_nid
                        );
                    }
                    for eid in &edges_to_supersede {
                        if let Some(e) = graph.edges.get_mut(*eid) {
                            e.valid_until = Some(sc.timestamp);
                            graph.dirty_edges.insert(*eid);
                            state_edges_superseded += 1;
                        }
                    }

                    // Create new state edge
                    let mut edge = GraphEdge::new(
                        entity_nid,
                        state_nid,
                        EdgeType::Association {
                            association_type: assoc_type.clone(),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        0.9,
                    );
                    edge.valid_from = Some(sc.timestamp);
                    edge.properties.insert(
                        "attribute".to_string(),
                        serde_json::Value::String(attribute.to_string()),
                    );
                    edge.properties.insert(
                        "value".to_string(),
                        serde_json::Value::String(sc.new_state.clone()),
                    );
                    if let Some(cat) = &sc.category {
                        edge.properties.insert(
                            "category".to_string(),
                            serde_json::Value::String(cat.clone()),
                        );
                    }
                    let new_eid = graph.add_edge(edge);
                    state_edges_created += 1;

                    tracing::info!(
                        "PHASE_B created edge eid={:?} '{}' {} -> {} (entity='{}' -> state='{}')",
                        new_eid,
                        assoc_type,
                        entity_nid,
                        state_nid,
                        sc.entity,
                        sc.new_state
                    );
                }

                tracing::info!(
                    "PHASE_B state edges done: created={} superseded={}",
                    state_edges_created,
                    state_edges_superseded
                );

                // ---- Transactions → graph edges (append-only, each payment is a separate edge) ----
                for lc in &ledger_candidates {
                    let graph = inference.graph_mut();
                    let mut edge = GraphEdge::new(
                        lc.from_id,
                        lc.to_id,
                        EdgeType::Association {
                            association_type: self.ontology.build_edge_type("financial", "payment"),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        (lc.amount as f32).clamp(0.0, 1.0),
                    );
                    edge.valid_from = Some(lc.timestamp);
                    edge.properties
                        .insert("amount".to_string(), serde_json::json!(lc.amount));
                    edge.properties.insert(
                        "description".to_string(),
                        serde_json::Value::String(lc.description.clone()),
                    );
                    graph.add_edge(edge);

                    tracing::info!(
                        "PHASE_B ledger edge: {} -> {} amount={} (ts={})",
                        lc.from,
                        lc.to,
                        lc.amount,
                        lc.timestamp
                    );
                }

                // ---- Relationships → bidirectional graph edges ----
                for rc in &relationship_candidates {
                    let graph = inference.graph_mut();
                    let subj_nid =
                        match ensure_concept(graph, &rc.subject, ConceptType::NamedEntity) {
                            Some(nid) => nid,
                            None => continue,
                        };
                    let obj_nid = match ensure_concept(graph, &rc.object, ConceptType::NamedEntity)
                    {
                        Some(nid) => nid,
                        None => continue,
                    };

                    let assoc_type = self
                        .ontology
                        .build_edge_type("relationship", &rc.relation_type);
                    let props = serde_json::json!({
                        "relation_type": rc.relation_type,
                    });
                    // Use add_or_strengthen for idempotency (bidirectional)
                    Self::add_or_strengthen_association(
                        graph,
                        subj_nid,
                        obj_nid,
                        &assoc_type,
                        0.8,
                        props.clone(),
                    );
                    Self::add_or_strengthen_association(
                        graph,
                        obj_nid,
                        subj_nid,
                        &assoc_type,
                        0.8,
                        props,
                    );

                    tracing::info!(
                        "PHASE_B relationship edge: {} <-> {} (type={})",
                        rc.subject,
                        rc.object,
                        rc.relation_type
                    );
                }

                // ---- Preferences → graph edges ----
                for pc in &preference_candidates {
                    let graph = inference.graph_mut();
                    let entity_nid =
                        match ensure_concept(graph, &pc.entity, ConceptType::NamedEntity) {
                            Some(nid) => nid,
                            None => continue,
                        };
                    let item_nid = match ensure_concept(graph, &pc.item, ConceptType::NamedEntity) {
                        Some(nid) => nid,
                        None => continue,
                    };

                    let assoc_type = self.ontology.build_edge_type("preference", &pc.category);
                    let props = serde_json::json!({
                        "category": pc.category,
                        "sentiment": pc.sentiment,
                    });
                    Self::add_or_strengthen_association(
                        graph,
                        entity_nid,
                        item_nid,
                        &assoc_type,
                        pc.sentiment as f32,
                        props,
                    );

                    tracing::info!(
                        "PHASE_B preference edge: {} -> {} (category={} sentiment={})",
                        pc.entity,
                        pc.item,
                        pc.category,
                        pc.sentiment
                    );
                }

                // ---- Event → Concept About edges ----
                // Connect event nodes to the concept nodes they mention.
                // This enables entity resolution to find relevant events via neighbor traversal.
                {
                    let graph = inference.graph_mut();
                    for (event_nid, concept_nid) in &event_concept_mentions {
                        let edge = GraphEdge::new(
                            *event_nid,
                            *concept_nid,
                            EdgeType::About {
                                relevance_score: 0.9,
                                mention_count: 1,
                                entity_role: crate::claims::types::EntityRole::default(),
                                predicate: Some("mentions".to_string()),
                            },
                            0.8,
                        );
                        graph.add_edge(edge);
                    }
                    if !event_concept_mentions.is_empty() {
                        tracing::info!("PHASE_B about edges: {}", event_concept_mentions.len());
                    }
                }

                // Final summary: count graph nodes and edges
                {
                    let graph = inference.graph_mut();
                    let total_nodes = graph.nodes.len();
                    let total_edges = graph.edges.len();
                    let concept_count = graph.concept_index.len();
                    let active_state_edges = graph
                        .edges
                        .iter()
                        .filter(|(_, e)| {
                            if let EdgeType::Association {
                                association_type, ..
                            } = &e.edge_type
                            {
                                association_type.starts_with("state:") && e.valid_until.is_none()
                            } else {
                                false
                            }
                        })
                        .count();
                    let superseded_state_edges = graph
                        .edges
                        .iter()
                        .filter(|(_, e)| {
                            if let EdgeType::Association {
                                association_type, ..
                            } = &e.edge_type
                            {
                                association_type.starts_with("state:") && e.valid_until.is_some()
                            } else {
                                false
                            }
                        })
                        .count();
                    tracing::info!(
                        "STATE_TRACKING PHASE_B_DONE episode_id={} total_nodes={} concept_nodes={} total_edges={} active_state_edges={} superseded_state_edges={}",
                        episode.id,
                        total_nodes,
                        concept_count,
                        total_edges,
                        active_state_edges,
                        superseded_state_edges
                    );
                }
            }
            // inference lock dropped

            // Embed newly created concept nodes asynchronously
            let nodes_to_embed: Vec<(NodeId, String)> = {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                graph
                    .concept_index
                    .iter()
                    .filter_map(|(name, &nid)| {
                        graph.get_node(nid).and_then(|node| {
                            if node.embedding.is_empty() {
                                Some((nid, name.to_string()))
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            };
            if !nodes_to_embed.is_empty() {
                if let Some(ref ec) = self.embedding_client {
                    let ec = ec.clone();
                    let inference_ref = self.inference.clone();
                    tokio::spawn(async move {
                        for (nid, text) in nodes_to_embed {
                            match ec
                                .embed(crate::claims::EmbeddingRequest {
                                    text,
                                    context: None,
                                })
                                .await
                            {
                                Ok(resp) if !resp.embedding.is_empty() => {
                                    let mut inf = inference_ref.write().await;
                                    let graph = inf.graph_mut();
                                    if let Some(node) = graph.get_node_mut(nid) {
                                        node.embedding = resp.embedding.clone();
                                    }
                                    graph.node_vector_index.insert(nid, resp.embedding);
                                },
                                Ok(_) => {},
                                Err(e) => {
                                    tracing::debug!("Node embedding failed for nid={}: {}", nid, e);
                                },
                            }
                        }
                    });
                }
            }
        }

        Ok(())
    }

    pub(super) async fn update_transition_model(&self, episode: &Episode) -> GraphResult<()> {
        let should_update = matches!(
            episode.outcome,
            Some(EpisodeOutcome::Success) | Some(EpisodeOutcome::Failure)
        );
        if !should_update {
            return Ok(());
        }

        let success = matches!(episode.outcome, Some(EpisodeOutcome::Success));
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        if events.is_empty() {
            return Ok(());
        }

        let outputs = crate::contracts::build_learning_outputs(episode, &events);
        let mut model = self.transition_model.write().await;
        model.update_from_trace(
            outputs.episode_record.goal_bucket_id,
            &outputs.abstract_trace,
            outputs.episode_record.episode_id,
            success,
        );
        tracing::info!(
            "Transition model updated episode_id={} goal_bucket_id={} transitions={} success={}",
            outputs.episode_record.episode_id,
            outputs.episode_record.goal_bucket_id,
            outputs.abstract_trace.transitions.len(),
            success
        );

        Ok(())
    }
}
