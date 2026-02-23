use super::*;

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
                self.stats.write().await.total_memories_formed += 1;
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

                // Fire-and-forget: async LLM refinement + embedding
                if let Some(ref refinement) = self.refinement_engine {
                    let memory_id = upsert.id;
                    let store_ref = self.memory_store.clone();
                    let refinement_ref = refinement.clone();
                    let embedding_client = self.embedding_client.clone();
                    let event_narrative = crate::event_content::build_event_narrative(&events);
                    tokio::spawn(async move {
                        if let Err(e) = refinement_ref
                            .refine_and_embed_memory(
                                memory_id,
                                &store_ref,
                                embedding_client.as_ref(),
                                Some(event_narrative),
                            )
                            .await
                        {
                            tracing::warn!("Memory refinement failed for {}: {}", memory_id, e);
                        }
                    });
                }
            }

            // Check if we should run consolidation
            let should_consolidate = {
                let mut counter = self.episodes_since_consolidation.write().await;
                *counter += 1;
                *counter >= self.config.consolidation_interval
            };
            if should_consolidate {
                *self.episodes_since_consolidation.write().await = 0;
                let store_ref = self.memory_store.clone();
                let engine_ref = self.consolidation_engine.clone();
                tokio::spawn(async move {
                    let mut store = store_ref.write().await;
                    let mut engine = engine_ref.write().await;
                    let result = engine.run_consolidation(store.as_mut());
                    if result.semantic_created > 0 || result.schema_created > 0 {
                        tracing::info!(
                            "Consolidation pass: {} semantic created, {} schemas created, {} episodes consolidated",
                            result.semantic_created,
                            result.schema_created,
                            result.consolidated_episode_ids.len()
                        );
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
                    self.stats.write().await.total_strategies_extracted += 1;
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

                    // Fire-and-forget: async LLM refinement + embedding for strategy
                    if let Some(ref refinement) = self.refinement_engine {
                        let strategy_id = upsert.id;
                        let strategy_clone = strategy.clone();
                        let store_ref = self.strategy_store.clone();
                        let refinement_ref = refinement.clone();
                        let embedding_client = self.embedding_client.clone();
                        let event_narrative =
                            crate::event_content::build_event_narrative(&events);
                        tokio::spawn(async move {
                            match refinement_ref
                                .refine_and_embed_strategy(
                                    &strategy_clone,
                                    embedding_client.as_ref(),
                                    Some(event_narrative),
                                )
                                .await
                            {
                                Ok(refined) => {
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

        self.stats.write().await.total_reinforcements_applied += 1;

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
