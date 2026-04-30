// crates/agent-db-graph/src/integration/pipeline/episode_memory.rs
//
// Memory formation from episodes: store episode as memory, attach to graph,
// index in BM25, fire-and-forget LLM refinement, and periodic consolidation.

use super::*;

impl GraphEngine {
    /// Process episode for memory formation
    pub(crate) async fn process_episode_for_memory(&self, episode: &Episode) -> GraphResult<()> {
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

                    let bg_permit = self.background_semaphore.clone();
                    tokio::spawn(async move {
                        let _permit = bg_permit.acquire().await;
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
            self.episodes_since_consolidation
                .fetch_add(1, AtomicOrdering::Relaxed);
            let should_consolidate = loop {
                let current = self
                    .episodes_since_consolidation
                    .load(AtomicOrdering::Relaxed);
                if current < self.config.consolidation_interval {
                    break false;
                }
                match self.episodes_since_consolidation.compare_exchange_weak(
                    current,
                    0,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => break true,
                    Err(_) => continue,
                }
            };
            if should_consolidate {
                let store_ref = self.memory_store.clone();
                let engine_ref = self.consolidation_engine.clone();
                let bm25_ref = self.memory_bm25_index.clone();
                let llm_ref = self.unified_llm_client.clone();
                let bg_permit = self.background_semaphore.clone();
                tokio::spawn(async move {
                    let _permit = bg_permit.acquire().await;
                    // Pre-pass: infer goal labels for goalless buckets via LLM
                    // Extract data under read lock, drop it, then make LLM call without holding lock
                    let goal_overrides = if let Some(ref llm) = llm_ref {
                        let all_memories = {
                            let store = store_ref.read().await;
                            store.list_all_memories()
                        };
                        crate::consolidation::infer_goal_labels(llm.as_ref(), &all_memories).await
                    } else {
                        std::collections::HashMap::new()
                    };

                    let mut store = store_ref.write().await;
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
}
