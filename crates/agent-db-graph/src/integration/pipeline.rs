use super::*;
use crate::metadata_normalize::{metadata_value_preview, MetadataRole};

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
                    let has_code_for_reindex = has_code;
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
                            }
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
                let bm25_ref = self.memory_bm25_index.clone();
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
                    if let Some(ref refinement) = self.refinement_engine {
                        let strategy_id = upsert.id;
                        let strategy_clone = strategy.clone();
                        let store_ref = self.strategy_store.clone();
                        let refinement_ref = refinement.clone();
                        let embedding_client = self.embedding_client.clone();
                        let event_narrative = crate::event_content::build_event_narrative(&events);
                        let bm25_ref = self.strategy_bm25_index.clone();
                        let has_code_for_reindex = has_code;
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
                                    // Re-index with refined text
                                    let text = format!(
                                        "{} {} {}",
                                        refined.summary, refined.when_to_use, refined.action_hint
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

        if events.is_empty() {
            return Ok(());
        }

        // Pre-resolve all entity names to NodeIds while holding inference lock,
        // then drop it before acquiring structured_memory write lock to avoid deadlock.
        struct StateCandidate {
            entity: String,
            new_state: String,
            old_state: Option<String>,
            node_id: NodeId,
            trigger: String,
            timestamp: u64,
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

        let (state_candidates, ledger_candidates) = {
            let inference = self.inference.read().await;
            let graph = inference.graph();

            let mut state_candidates = Vec::new();
            let mut ledger_candidates = Vec::new();

            for event in &events {
                // ---- Normalize metadata keys via alias matching ----
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
                let old_state = normalized.get_str(MetadataRole::OldState, &event.metadata);

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

                // Section E guards: ALL must pass
                if let (Some(ref entity), Some(ref new_st)) = (&entity_name, &new_state) {
                    if !entity.is_empty()
                        && !new_st.is_empty()
                        && is_state_event
                        && event.timestamp > 0
                    {
                        if let Some(&node_id) = graph.concept_index.get(entity.as_str()) {
                            let trigger = match &event.event_type {
                                EventType::Action { action_name, .. } => action_name.clone(),
                                _ => "auto".to_string(),
                            };
                            state_candidates.push(StateCandidate {
                                entity: entity.clone(),
                                new_state: new_st.clone(),
                                old_state: old_state.clone(),
                                node_id,
                                trigger,
                                timestamp: event.timestamp,
                            });
                        } else {
                            tracing::debug!(
                                "State tracking: entity '{}' not found in graph indices",
                                entity
                            );
                        }
                    } else {
                        tracing::debug!(
                            "State tracking guard rejected: entity_empty={} state_empty={} is_state_event={} ts={}",
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
                        let from_id = graph.concept_index.get(from.as_str()).copied();
                        let to_id = graph.concept_index.get(to.as_str()).copied();

                        if let (Some(fid), Some(tid)) = (from_id, to_id) {
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
            }

            (state_candidates, ledger_candidates)
        };
        // inference lock is now dropped

        // Apply state changes and ledger entries under structured_memory write lock
        if !state_candidates.is_empty() || !ledger_candidates.is_empty() {
            let mut sm = self.structured_memory.write().await;

            for sc in state_candidates {
                let key = crate::structured_memory::state_key(sc.node_id);

                // Create state machine if not exists
                if sm.get(&key).is_none() {
                    let initial_state = sc.old_state.as_deref().unwrap_or("unknown");
                    sm.upsert(
                        &key,
                        crate::structured_memory::MemoryTemplate::StateMachine {
                            entity: sc.entity.clone(),
                            current_state: initial_state.to_string(),
                            history: vec![],
                            provenance: crate::structured_memory::MemoryProvenance::EpisodePipeline,
                        },
                    );
                }

                if let Err(e) = sm.state_transition(&key, &sc.new_state, &sc.trigger, sc.timestamp)
                {
                    tracing::debug!("State transition failed for {}: {}", key, e);
                } else {
                    tracing::debug!(
                        "Auto state transition: {} -> {} (entity={})",
                        sc.old_state.as_deref().unwrap_or("?"),
                        sc.new_state,
                        sc.entity
                    );
                }
            }

            for lc in ledger_candidates {
                let key = crate::structured_memory::ledger_key(lc.from_id, lc.to_id);

                // Create ledger if not exists
                if sm.get(&key).is_none() {
                    sm.upsert(
                        &key,
                        crate::structured_memory::MemoryTemplate::Ledger {
                            entity_pair: (lc.from.clone(), lc.to.clone()),
                            entries: vec![],
                            balance: 0.0,
                            provenance: crate::structured_memory::MemoryProvenance::EpisodePipeline,
                        },
                    );
                }

                let entry = crate::structured_memory::LedgerEntry {
                    timestamp: lc.timestamp,
                    amount: lc.amount,
                    description: lc.description,
                    direction: lc.direction,
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
