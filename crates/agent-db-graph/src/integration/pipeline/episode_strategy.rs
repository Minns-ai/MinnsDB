// crates/agent-db-graph/src/integration/pipeline/episode_strategy.rs
//
// Strategy extraction from episodes: extract strategies from Success/Failure episodes,
// attach to graph, index in BM25, fire-and-forget LLM refinement.

use super::*;

impl GraphEngine {
    /// Process episode for strategy extraction
    pub(crate) async fn process_episode_for_strategy(&self, episode: &Episode) -> GraphResult<()> {
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
}
