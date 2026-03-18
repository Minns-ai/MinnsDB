use super::*;

impl GraphEngine {
    /// Process a single event and update the graph
    ///
    /// **Automatic Self-Evolution Pipeline:**
    /// 1. Event ordering and graph construction
    /// 2. Episode detection from event stream
    /// 3. Memory formation from significant episodes
    /// 4. Strategy extraction from successful episodes
    /// 5. Reinforcement learning from outcomes
    ///
    /// **Semantic Memory Control:**
    /// - If `enable_semantic` is Some(true), semantic memory will be processed for this event
    /// - If `enable_semantic` is Some(false), semantic memory will be skipped for this event
    /// - If `enable_semantic` is None, falls back to `config.enable_semantic_memory`
    pub async fn process_event_with_options(
        &self,
        event: Event,
        enable_semantic: Option<bool>,
    ) -> GraphResult<GraphOperationResult> {
        let start_time = std::time::Instant::now();
        tracing::info!(
            "GraphEngine process_event start id={} agent_id={} session_id={} type={}",
            event.id,
            event.agent_id,
            event.session_id,
            Self::event_type_name(&event.event_type)
        );
        let mut result = GraphOperationResult {
            nodes_created: Vec::new(),
            relationships_discovered: 0,
            patterns_detected: Vec::new(),
            processing_time_ms: 0,
            errors: Vec::new(),
        };

        // Store event for episode processing (ring-buffer capped)
        {
            let mut store = self.event_store.write().await;
            let mut order = self.event_store_order.write().await;
            store.insert(event.id, event.clone());
            order.push_back(event.id);
            // Evict oldest events when over cap
            let cap = self.config.max_event_store_size;
            while store.len() > cap {
                if let Some(old_id) = order.pop_front() {
                    store.remove(&old_id);
                } else {
                    break;
                }
            }
        }

        // Extract NER features if semantic memory is enabled
        // Check per-request override first, then fall back to config
        let should_extract_semantic = enable_semantic.unwrap_or(self.config.enable_semantic_memory);

        if should_extract_semantic {
            // Claims pipeline calls NER internally and uses the result for
            // entity overlap scoring, LLM prompt grounding, and entity attachment.
            self.extract_claims_async(&event).await;
        }

        // Step 1: Order the event (handles out-of-order arrival)
        tracing::info!("Ordering event {}", event.id);
        let ordering_result = self.event_ordering.process_event(event.clone()).await?;
        if !ordering_result.issues.is_empty() {
            for issue in &ordering_result.issues {
                tracing::warn!("Ordering issue event_id={} issue={:?}", event.id, issue);
            }
        }

        // Step 2: Process all ready events through graph construction
        for ready_event in ordering_result.ready_events {
            // Add to processing buffer
            {
                let mut buffer = self.event_buffer.write().await;
                buffer.push(ready_event.clone());
            }

            // Process through inference engine
            tracing::info!("Inference processing event {}", ready_event.id);
            tracing::info!(
                "Acquiring inference write lock for event {}",
                ready_event.id
            );
            let nodes_result = {
                let mut inference = self.inference.write().await;
                inference.process_event(ready_event.clone())
            };
            let _event_node_id = match nodes_result {
                Ok(nodes) => {
                    let first_node = nodes.first().copied();
                    result.nodes_created.extend(nodes.clone());

                    // Auto-index newly created nodes
                    tracing::info!(
                        "Auto-index start event_id={} nodes={}",
                        ready_event.id,
                        nodes.len()
                    );
                    self.auto_index_nodes(&nodes).await?;
                    tracing::info!("Auto-index done event_id={}", ready_event.id);
                    first_node
                },
                Err(e) => {
                    result.errors.push(e);
                    None
                },
            };

            // Code Intelligence: AST parsing for CodeFile events
            // Only proceed if base event processing succeeded (event_node_id is Some)
            #[cfg(feature = "code-intelligence")]
            if _event_node_id.is_some() {
                if let EventType::CodeFile {
                    ref content,
                    ref file_path,
                    ref language,
                    ..
                } = ready_event.event_type
                {
                    if let Some(ref parser) = self.ast_parser {
                        let detected = agent_db_ast::AstParser::detect_language(file_path);
                        let lang = language.as_deref().or(detected.as_deref());
                        if let Some(lang) = lang {
                            match parser.parse(content, lang, file_path) {
                                Ok(parse_result) => {
                                    let entity_count = parse_result.entities.len();
                                    let rel_count = parse_result.relationships.len();
                                    let mut inference = self.inference.write().await;
                                    let graph = inference.graph_mut();
                                    match crate::code_graph::ingest_parse_result(
                                        graph,
                                        &parse_result,
                                        _event_node_id,
                                    ) {
                                        Ok(code_nodes) => {
                                            let num_code_nodes = code_nodes.len();
                                            tracing::info!(
                                            "AST ingestion for {}: {} entities, {} relationships, {} concept nodes created",
                                            file_path, entity_count, rel_count, num_code_nodes
                                        );

                                            // Build embedding text for each code node
                                            let nodes_to_embed: Vec<(crate::structures::NodeId, String)> = code_nodes
                                                .iter()
                                                .filter_map(|&nid| {
                                                    let node = graph.get_node(nid)?;
                                                    // Only embed nodes without embeddings
                                                    if !node.embedding.is_empty() {
                                                        return None;
                                                    }
                                                    Some((nid, crate::code_graph::build_code_embedding_text(node)))
                                                })
                                                .collect();

                                            result.nodes_created.extend(code_nodes);

                                            // Drop write lock before async embedding
                                            drop(inference);

                                            if !nodes_to_embed.is_empty() {
                                                self.embed_nodes_async(nodes_to_embed);
                                            }
                                        },
                                        Err(e) => {
                                            tracing::warn!(
                                                "AST graph ingestion failed for {}: {}",
                                                file_path,
                                                e
                                            );
                                        },
                                    }
                                },
                                Err(e) => {
                                    tracing::warn!("AST parsing failed for {}: {}", file_path, e);
                                },
                            }
                        }
                    }
                }
            }

            // Process through scoped inference engine (session + agent_type isolation)
            let scoped_event = crate::scoped_inference::ScopedEvent {
                event: ready_event.clone(),
                agent_type: ready_event.agent_type.clone(),
                priority: 0.0,
                scope_metadata: crate::scoped_inference::ScopeMetadata {
                    workspace_id: None,
                    user_id: None,
                    environment: None,
                    tags: Vec::new(),
                },
            };
            tracing::info!("Scoped inference processing event {}", ready_event.id);
            if let Err(e) = self
                .scoped_inference
                .process_scoped_event(scoped_event)
                .await
            {
                result.errors.push(e);
            }

            // World Model: bottom-up prediction error (mode-aware)
            if self.config.effective_world_model_mode() != WorldModelMode::Disabled {
                if let Some(ref wm) = self.world_model {
                    let wm_guard = wm.read().await;
                    let event_features = world_model::extract_event_features_raw(&ready_event);
                    let memory_features = agent_db_world_model::MemoryFeatures {
                        tier: 0,
                        strength: 0.5,
                        access_count: 1,
                        context_fingerprint: ready_event.context.fingerprint,
                        goal_bucket_id: 0,
                    };
                    let strategy_features = agent_db_world_model::StrategyFeatures {
                        quality_score: 0.5,
                        expected_success: 0.5,
                        expected_value: 0.5,
                        confidence: 0.0,
                        goal_bucket_id: 0,
                        behavior_signature_hash: 0,
                    };
                    let policy_features = agent_db_world_model::PolicyFeatures {
                        goal_count: ready_event
                            .context
                            .active_goals
                            .len()
                            .min(u32::MAX as usize) as u32,
                        top_goal_priority: ready_event
                            .context
                            .active_goals
                            .iter()
                            .map(|g| g.priority)
                            .fold(0.5f32, f32::max),
                        resource_cpu_percent: 0.0,
                        resource_memory_bytes: 0,
                        context_fingerprint: ready_event.context.fingerprint,
                    };
                    let error = wm_guard.prediction_error(
                        &event_features,
                        &memory_features,
                        &strategy_features,
                        &policy_features,
                    );
                    tracing::debug!(
                        "World model prediction_error event_id={} total_z={:.2} layer={:?}",
                        ready_event.id,
                        error.total_z,
                        error.mismatch_layer,
                    );

                    // Repair logging (Full mode only)
                    if self.config.effective_world_model_mode() == WorldModelMode::Full
                        && agent_db_planning::repair::should_repair(
                            &error,
                            &self.config.planning_config,
                        )
                    {
                        let scope = agent_db_planning::repair::determine_repair_scope(&error, 0);
                        tracing::info!(
                            "World model repair triggered event_id={} total_z={:.2} scope={:?}",
                            ready_event.id,
                            error.total_z,
                            scope,
                        );
                    }
                }
            }

            // Bottom-up execution validation (Full mode only)
            // Fire-and-forget: don't block the main pipeline
            if self.config.planning_config.generation_mode
                == agent_db_planning::GenerationMode::Full
                && !self.active_executions.is_empty()
            {
                let session_id = ready_event.session_id;
                // Find active executions matching this event's session
                let matching_exec_ids: Vec<u64> = self
                    .active_executions
                    .iter()
                    .filter_map(|entry| {
                        let state = entry.value().try_read();
                        if let Ok(s) = state {
                            if s.session_id == session_id {
                                Some(*entry.key())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                if !matching_exec_ids.is_empty() {
                    let engine_executions = self.active_executions.clone();
                    let engine_world_model = self.world_model.clone();
                    let _engine_action_gen = self.action_generator.clone();
                    let _engine_strategy_gen = self.strategy_generator.clone();
                    let planning_config = self.config.planning_config.clone();
                    let event_clone = ready_event.clone();

                    tokio::spawn(async move {
                        for exec_id in matching_exec_ids {
                            if let Some(exec_arc) = engine_executions.get(&exec_id) {
                                let exec_arc = exec_arc.clone();
                                // Just log the prediction error — actual repair would
                                // need the full GraphEngine which we can't move into spawn.
                                if let Some(ref wm) = engine_world_model {
                                    if let Ok(wm_guard) = wm.try_read() {
                                        if let Ok(state) = exec_arc.try_read() {
                                            let event_features =
                                                world_model::extract_event_features_raw(
                                                    &event_clone,
                                                );
                                            let memory_features =
                                                agent_db_world_model::MemoryFeatures {
                                                    tier: 0,
                                                    strength: 0.5,
                                                    access_count: 1,
                                                    context_fingerprint: state.context_fingerprint,
                                                    goal_bucket_id: state.strategy.goal_bucket_id,
                                                };
                                            let strategy_features =
                                                agent_db_world_model::StrategyFeatures {
                                                    quality_score: state.strategy.confidence,
                                                    expected_success: state.strategy.confidence,
                                                    expected_value: 0.5,
                                                    confidence: state.strategy.confidence,
                                                    goal_bucket_id: state.strategy.goal_bucket_id,
                                                    behavior_signature_hash: 0,
                                                };
                                            let policy_features =
                                                agent_db_world_model::PolicyFeatures {
                                                    goal_count: 1,
                                                    top_goal_priority: 0.8,
                                                    resource_cpu_percent: 0.0,
                                                    resource_memory_bytes: 0,
                                                    context_fingerprint: state.context_fingerprint,
                                                };

                                            let error = wm_guard.prediction_error(
                                                &event_features,
                                                &memory_features,
                                                &strategy_features,
                                                &policy_features,
                                            );

                                            if agent_db_planning::repair::should_repair(
                                                &error,
                                                &planning_config,
                                            ) {
                                                let scope =
                                                    agent_db_planning::repair::determine_repair_scope(
                                                        &error,
                                                        state.consecutive_action_repairs,
                                                    );
                                                tracing::info!(
                                                    exec_id,
                                                    total_z = error.total_z,
                                                    ?scope,
                                                    "Bottom-up execution validation: repair needed"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            }

            if let Err(e) = self.handle_learning_event(&ready_event).await {
                result.errors.push(e);
            }

            // Step 3: Self-Evolution Pipeline - Episode Detection
            if self.config.auto_episode_detection {
                // Check for completed episodes (must drop write lock before acquiring read lock)
                let episode_update = {
                    self.episode_detector
                        .write()
                        .await
                        .process_event(&ready_event)
                };

                if let Some(episode_update) = episode_update {
                    let (episode_id, is_correction) = match episode_update {
                        crate::episodes::EpisodeUpdate::Completed(id) => (id, false),
                        crate::episodes::EpisodeUpdate::Corrected(id) => (id, true),
                    };
                    result.patterns_detected.push(format!(
                        "{}_{}",
                        if is_correction {
                            "episode_corrected"
                        } else {
                            "episode_completed"
                        },
                        episode_id
                    ));

                    let episodes: Vec<Episode> = self
                        .episode_detector
                        .read()
                        .await
                        .get_completed_episodes()
                        .to_vec();
                    if let Some(episode) = episodes.iter().find(|e| e.id == episode_id) {
                        if !is_correction {
                            self.stats
                                .total_episodes_detected
                                .fetch_add(1, AtomicOrdering::Relaxed);
                        }

                        if self.config.auto_memory_formation && should_extract_semantic {
                            self.process_episode_for_memory(episode).await?;
                        }

                        // Structured memory: auto-extract state changes and ledger entries
                        self.process_episode_for_state_tracking(episode).await?;

                        if self.config.auto_strategy_extraction {
                            self.process_episode_for_strategy(episode).await?;
                        }

                        if self.config.auto_reinforcement_learning {
                            if is_correction {
                                self.update_transition_model(episode).await?;
                            } else {
                                self.process_episode_for_reinforcement(episode).await?;
                            }
                        }

                        // World model training (mode-aware)
                        if self.config.effective_world_model_mode() != WorldModelMode::Disabled
                            && self.world_model.is_some()
                        {
                            if let Err(e) = self.process_episode_for_world_model(episode).await {
                                tracing::warn!("World model training tuple assembly failed: {}", e);
                            }
                        }
                    }
                }
            }
        }

        // Update ordering statistics
        if ordering_result.reordering_occurred {
            result
                .patterns_detected
                .push("event_reordering_occurred".to_string());
        }

        // Update statistics
        {
            let new_count = self
                .stats
                .total_events_processed
                .fetch_add(1, AtomicOrdering::Relaxed)
                + 1;
            self.stats
                .total_nodes_created
                .fetch_add(result.nodes_created.len() as u64, AtomicOrdering::Relaxed);

            let processing_time = start_time.elapsed().as_millis() as f64;
            {
                let mut derived = self.stats.derived.lock();
                derived.last_operation_time = std::time::Instant::now();
                derived.average_processing_time_ms = (derived.average_processing_time_ms
                    * (new_count as f64 - 1.0)
                    + processing_time)
                    / new_count as f64;
            }

            // Run Louvain community detection periodically
            if self.config.enable_louvain && new_count % self.config.louvain_interval == 0 {
                if let Err(e) = self.run_community_detection().await {
                    result.errors.push(e);
                } else {
                    result
                        .patterns_detected
                        .push("louvain_communities_updated".to_string());
                }
            }
        }

        // Check if we need to process batch
        let should_process_batch = {
            let buffer = self.event_buffer.read().await;
            buffer.len() >= self.config.batch_size
        };

        if should_process_batch {
            self.process_batch().await?;
        }

        // Persist graph to redb when the backend is available
        if self.redb_backend.is_some() {
            let total = self
                .stats
                .total_events_processed
                .load(AtomicOrdering::Relaxed);
            let last_persistence = self.last_persistence.load(AtomicOrdering::Relaxed);

            if total - last_persistence >= self.config.persistence_interval {
                if let Err(e) = self.persist_graph_state().await {
                    tracing::warn!("Graph persistence failed (will retry next interval): {}", e);
                }
            }
        }

        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        tracing::info!(
            "GraphEngine process_event done id={} nodes_created={} patterns_detected={} errors={} duration_ms={}",
            event.id,
            result.nodes_created.len(),
            result.patterns_detected.len(),
            result.errors.len(),
            result.processing_time_ms
        );
        Ok(result)
    }

    fn event_type_name(event_type: &EventType) -> &'static str {
        match event_type {
            EventType::Action { .. } => "Action",
            EventType::Observation { .. } => "Observation",
            EventType::Cognitive { .. } => "Cognitive",
            EventType::Communication { .. } => "Communication",
            EventType::Learning { .. } => "Learning",
            EventType::Context { .. } => "Context",
            EventType::Conversation { .. } => "Conversation",
            EventType::CodeReview { .. } => "CodeReview",
            EventType::CodeFile { .. } => "CodeFile",
        }
    }

    async fn handle_learning_event(&self, event: &Event) -> GraphResult<()> {
        let EventType::Learning {
            event: learning_event,
        } = &event.event_type
        else {
            return Ok(());
        };

        let now = std::time::Instant::now();
        match learning_event {
            LearningEvent::MemoryRetrieved {
                query_id,
                memory_ids,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
                            memory_ids: Vec::new(),
                            memory_used: Vec::new(),
                            strategy_ids: Vec::new(),
                            strategy_used: Vec::new(),
                            claim_ids: Vec::new(),
                            claims_used: Vec::new(),
                            last_updated: now,
                        });
                trace.memory_ids = memory_ids.clone();
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry memory_retrieved query_id={} count={}",
                    query_id,
                    memory_ids.len()
                );
            },
            LearningEvent::MemoryUsed {
                query_id,
                memory_id,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
                            memory_ids: Vec::new(),
                            memory_used: Vec::new(),
                            strategy_ids: Vec::new(),
                            strategy_used: Vec::new(),
                            claim_ids: Vec::new(),
                            claims_used: Vec::new(),
                            last_updated: now,
                        });
                if !trace.memory_used.contains(memory_id) {
                    trace.memory_used.push(*memory_id);
                }
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry memory_used query_id={} memory_id={}",
                    query_id,
                    memory_id
                );
            },
            LearningEvent::StrategyServed {
                query_id,
                strategy_ids,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
                            memory_ids: Vec::new(),
                            memory_used: Vec::new(),
                            strategy_ids: Vec::new(),
                            strategy_used: Vec::new(),
                            claim_ids: Vec::new(),
                            claims_used: Vec::new(),
                            last_updated: now,
                        });
                trace.strategy_ids = strategy_ids.clone();
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry strategy_served query_id={} count={}",
                    query_id,
                    strategy_ids.len()
                );
            },
            LearningEvent::StrategyUsed {
                query_id,
                strategy_id,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
                            memory_ids: Vec::new(),
                            memory_used: Vec::new(),
                            strategy_ids: Vec::new(),
                            strategy_used: Vec::new(),
                            claim_ids: Vec::new(),
                            claims_used: Vec::new(),
                            last_updated: now,
                        });
                if !trace.strategy_used.contains(strategy_id) {
                    trace.strategy_used.push(*strategy_id);
                }
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry strategy_used query_id={} strategy_id={}",
                    query_id,
                    strategy_id
                );
            },
            LearningEvent::ClaimRetrieved {
                query_id,
                claim_ids,
            } => {
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
                            memory_ids: Vec::new(),
                            memory_used: Vec::new(),
                            strategy_ids: Vec::new(),
                            strategy_used: Vec::new(),
                            claim_ids: Vec::new(),
                            claims_used: Vec::new(),
                            last_updated: now,
                        });
                trace.claim_ids = claim_ids.clone();
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry claim_retrieved query_id={} count={}",
                    query_id,
                    claim_ids.len()
                );
            },
            LearningEvent::ClaimUsed { query_id, claim_id } => {
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
                            memory_ids: Vec::new(),
                            memory_used: Vec::new(),
                            strategy_ids: Vec::new(),
                            strategy_used: Vec::new(),
                            claim_ids: Vec::new(),
                            claims_used: Vec::new(),
                            last_updated: now,
                        });
                if !trace.claims_used.contains(claim_id) {
                    trace.claims_used.push(*claim_id);
                }
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry claim_used query_id={} claim_id={}",
                    query_id,
                    claim_id
                );
            },
            LearningEvent::Outcome { query_id, success } => {
                tracing::info!(
                    "Learning telemetry outcome query_id={} success={}",
                    query_id,
                    success
                );
                self.apply_learning_outcome(query_id, *success).await?;
            },
        }

        Ok(())
    }

    async fn apply_learning_outcome(&self, query_id: &str, success: bool) -> GraphResult<()> {
        // DashMap provides lock-free concurrent access
        let trace = self.decision_traces.remove(query_id).map(|(_, v)| v);

        let Some(trace) = trace else {
            return Ok(());
        };

        // Auto-promote: served strategies/claims/memories that weren't explicitly
        // marked as "used" still contributed to the response. Include them
        // in outcome feedback so the feedback loop closes automatically.
        let memory_used: Vec<u64> = {
            let mut ids = trace.memory_used.clone();
            for mid in &trace.memory_ids {
                if !ids.contains(mid) {
                    ids.push(*mid);
                }
            }
            ids
        };

        let strategy_used: Vec<u64> = {
            let mut ids = trace.strategy_used.clone();
            for sid in &trace.strategy_ids {
                if !ids.contains(sid) {
                    ids.push(*sid);
                }
            }
            ids
        };

        let claims_used: Vec<u64> = {
            let mut ids = trace.claims_used.clone();
            for cid in &trace.claim_ids {
                if !ids.contains(cid) {
                    ids.push(*cid);
                }
            }
            ids
        };

        if !memory_used.is_empty() {
            let mut store = self.memory_store.write().await;
            for memory_id in &memory_used {
                let applied = store.apply_outcome(*memory_id, success);
                tracing::info!(
                    "Learning outcome applied to memory_id={} success={} applied={}",
                    memory_id,
                    success,
                    applied
                );
            }
        }

        if !strategy_used.is_empty() {
            let mut store = self.strategy_store.write().await;
            for strategy_id in &strategy_used {
                let updated = store.update_strategy_outcome(*strategy_id, success).is_ok();
                tracing::info!(
                    "Learning outcome applied to strategy_id={} success={} updated={}",
                    strategy_id,
                    success,
                    updated
                );
            }
        }

        // World model: feedback training tuple from outcome (Phase 6)
        if self.config.effective_world_model_mode() != WorldModelMode::Disabled {
            if let Some(ref wm) = self.world_model {
                let policy = agent_db_world_model::PolicyFeatures {
                    goal_count: 1,
                    top_goal_priority: 0.8,
                    resource_cpu_percent: 0.0,
                    resource_memory_bytes: 0,
                    context_fingerprint: 0,
                };
                let memory = agent_db_world_model::MemoryFeatures {
                    tier: 0,
                    strength: 0.5,
                    access_count: memory_used.len().min(u32::MAX as usize) as u32,
                    context_fingerprint: 0,
                    goal_bucket_id: 0,
                };
                let strategy = agent_db_world_model::StrategyFeatures {
                    quality_score: if success { 0.8 } else { 0.2 },
                    expected_success: if success { 0.8 } else { 0.2 },
                    expected_value: 0.5,
                    confidence: 0.5,
                    goal_bucket_id: 0,
                    behavior_signature_hash: 0,
                };
                let event = agent_db_world_model::EventFeatures {
                    event_type_hash: 0,
                    action_name_hash: 0,
                    context_fingerprint: 0,
                    outcome_success: if success { 1.0 } else { 0.0 },
                    significance: 0.7,
                    temporal_delta_ns: 0.0,
                    duration_ns: 0.0,
                };

                let tuple = agent_db_world_model::TrainingTuple {
                    event_features: event,
                    memory_features: memory,
                    strategy_features: strategy,
                    policy_features: policy,
                    is_positive: success,
                    weight: if success { 0.5 } else { 1.0 },
                };
                wm.write().await.submit_training(tuple);
                tracing::debug!(
                    "World model outcome training tuple query_id={} success={}",
                    query_id,
                    success,
                );
            }
        }

        if !claims_used.is_empty() {
            if let Some(ref claim_store) = self.claim_store {
                for claim_id in &claims_used {
                    match claim_store.record_outcome(*claim_id, success) {
                        Ok(Some(updated)) => {
                            tracing::info!(
                                "Learning outcome applied to claim_id={} success={} pos={} neg={}",
                                claim_id,
                                success,
                                updated.positive_outcomes(),
                                updated.negative_outcomes()
                            );
                            // Avoidance generation: if failure AND >= 2 negative outcomes
                            if !success && updated.negative_outcomes() >= 2 {
                                let avoidance_text = format!(
                                    "Avoid relying on: {} — this has led to repeated failures",
                                    updated.claim_text
                                );
                                let mut avoidance = crate::claims::types::DerivedClaim::new(
                                    claim_store.next_id().unwrap_or(0),
                                    avoidance_text,
                                    updated.supporting_evidence.clone(),
                                    0.5,
                                    updated.embedding.clone(),
                                    updated.source_event_id,
                                    updated.episode_id,
                                    updated.thread_id.clone(),
                                    updated.user_id.clone(),
                                    updated.workspace_id.clone(),
                                );
                                avoidance.claim_type = crate::claims::types::ClaimType::Avoidance;
                                avoidance.entities = updated.entities.clone();
                                // Copy subject_entity from the source claim
                                avoidance.subject_entity = updated.subject_entity.clone();
                                avoidance.category = updated.category.clone();
                                if let Err(e) = claim_store.store(&avoidance) {
                                    tracing::warn!(
                                        "Failed to store avoidance claim for claim_id={}: {}",
                                        claim_id,
                                        e
                                    );
                                } else {
                                    tracing::info!(
                                        "Generated avoidance claim id={} from repeatedly-failing claim_id={}",
                                        avoidance.id,
                                        claim_id
                                    );
                                    // Stamp state anchors on the avoidance claim
                                    if let Some(ref subj) = avoidance.subject_entity {
                                        let inference_guard = self.inference.read().await;
                                        let projected =
                                            crate::conversation::graph_projection::project_entity_state(
                                                inference_guard.graph(),
                                                subj,
                                                u64::MAX,
                                                Some(self.ontology.as_ref()),
                                            );
                                        drop(inference_guard);
                                        let mut anchor_meta: std::collections::HashMap<
                                            String,
                                            String,
                                        > = std::collections::HashMap::new();
                                        for slot in projected.slots.values() {
                                            let cat = slot
                                                .association_type
                                                .split(':')
                                                .next()
                                                .unwrap_or("");
                                            if self.ontology.is_single_valued(cat) {
                                                let key = format!(
                                                    "state_anchor:{}",
                                                    slot.association_type
                                                );
                                                let val = slot
                                                    .value
                                                    .as_deref()
                                                    .unwrap_or(&slot.target_name);
                                                anchor_meta.insert(key, val.to_string());
                                            }
                                        }
                                        if !anchor_meta.is_empty() {
                                            let _ = claim_store
                                                .update_metadata(avoidance.id, &anchor_meta);
                                            tracing::debug!(
                                                "Stamped {} state anchors on avoidance claim {}",
                                                anchor_meta.len(),
                                                avoidance.id,
                                            );
                                        }
                                    }
                                }
                            }
                        },
                        Ok(None) => {
                            tracing::warn!("Claim not found for outcome: claim_id={}", claim_id);
                        },
                        Err(e) => {
                            tracing::error!(
                                "Failed to record outcome for claim_id={}: {}",
                                claim_id,
                                e
                            );
                        },
                    }
                }
            }
        }

        Ok(())
    }

    /// Process a single event (convenience wrapper that uses config default for semantic memory)
    pub async fn process_event(&self, event: Event) -> GraphResult<GraphOperationResult> {
        self.process_event_with_options(event, None).await
    }

    /// Process multiple events in batch
    pub async fn process_events(&self, events: Vec<Event>) -> GraphResult<GraphOperationResult> {
        let start_time = std::time::Instant::now();
        let mut combined_result = GraphOperationResult {
            nodes_created: Vec::new(),
            relationships_discovered: 0,
            patterns_detected: Vec::new(),
            processing_time_ms: 0,
            errors: Vec::new(),
        };

        let event_count = events.len() as u64;

        // Process through inference engine (no buffer — results are returned immediately)
        match self.inference.write().await.process_events(events) {
            Ok(inference_results) => {
                combined_result.nodes_created = inference_results.nodes_created;
                combined_result.patterns_detected = (0..inference_results.patterns_detected)
                    .map(|i| format!("pattern_{}", i))
                    .collect();
                combined_result.relationships_discovered = inference_results.events_processed;
            },
            Err(e) => {
                combined_result.errors.push(e);
            },
        }

        // Update statistics
        {
            self.stats
                .total_events_processed
                .fetch_add(event_count, AtomicOrdering::Relaxed);
            self.stats.total_nodes_created.fetch_add(
                combined_result.nodes_created.len() as u64,
                AtomicOrdering::Relaxed,
            );
            self.stats.total_patterns_detected.fetch_add(
                combined_result.patterns_detected.len() as u64,
                AtomicOrdering::Relaxed,
            );
            self.stats.derived.lock().last_operation_time = std::time::Instant::now();
        }

        combined_result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(combined_result)
    }

    /// Process buffered events in batch
    pub(super) async fn process_batch(&self) -> GraphResult<()> {
        let events = {
            let mut buffer = self.event_buffer.write().await;
            let events = buffer.clone();
            buffer.clear();
            events
        };

        if !events.is_empty() {
            tracing::info!("Acquiring inference write lock for batch processing");
            let mut inference = self.inference.write().await;
            let _results = inference.process_events(events)?;
        }

        Ok(())
    }

    /// Flush all buffered events (useful for shutdown or testing)
    pub async fn flush_all_buffers(&self) -> GraphResult<()> {
        // Flush the event ordering buffers
        let buffered_events = self.event_ordering.flush_all_buffers().await?;

        // Process any remaining buffered events
        for event in buffered_events {
            let _ = self.process_event(event).await; // Ignore errors during shutdown
        }

        Ok(())
    }
}
