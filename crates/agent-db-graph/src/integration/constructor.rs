use super::*;

impl GraphEngine {
    /// Create a new graph engine with default configuration
    pub async fn new() -> GraphResult<Self> {
        Self::with_config(GraphEngineConfig::default()).await
    }

    /// Create a graph engine with custom configuration
    pub async fn with_config(config: GraphEngineConfig) -> GraphResult<Self> {
        // Validate and clamp config values
        let mut config = config;
        config.validate();

        // Apply Free Tier overrides if requested via environment
        let config = if std::env::var("SERVICE_PROFILE").unwrap_or_default() == "free" {
            tracing::info!("Applying Free Tier resource limits (0.25 CPU, 768MB RAM cap, No NER)");
            let mut free_config = config;
            free_config.enable_louvain = false;
            free_config.redb_cache_size_mb = 64;
            free_config.memory_cache_size = 1000;
            free_config.strategy_cache_size = 500;
            free_config.ner_workers = 0; // Disable NER workers for free tier
            free_config.claim_workers = 1;
            free_config.embedding_workers = 1;
            free_config
        } else {
            config
        };

        // Wire memory TTL from top-level config into memory formation config
        let mut memory_config = config.memory_config.clone();
        if config.default_memory_ttl_secs.is_some() && memory_config.default_ttl_secs.is_none() {
            memory_config.default_ttl_secs = config.default_memory_ttl_secs;
        }

        let mut inf_config = config.inference_config.clone();
        inf_config.max_graph_size = config.max_graph_size;
        let inference = Arc::new(RwLock::new(GraphInference::with_config(inf_config)));

        let traversal = Arc::new(GraphTraversal::new());

        let event_ordering = Arc::new(EventOrderingEngine::new(config.ordering_config.clone()));

        let scoped_inference = Arc::new(
            crate::scoped_inference::ScopedInferenceEngine::new(
                config.scoped_inference_config.clone(),
            )
            .await?,
        );

        // Initialize self-evolution components
        // Note: EpisodeDetector requires an Arc<Graph> but doesn't actually use it for detection
        // It uses event-based heuristics instead. We provide an empty graph for API compatibility.
        let graph_for_episodes = Arc::new(Graph::new());
        let episode_detector = Arc::new(RwLock::new(EpisodeDetector::new(
            graph_for_episodes,
            config.episode_config.clone(),
        )));

        // Initialize stores based on storage backend configuration
        let (memory_store, strategy_store, redb_backend): (
            MemoryStoreType,
            StrategyStoreType,
            Option<Arc<RedbBackend>>,
        ) = match config.storage_backend {
            StorageBackend::InMemory => {
                tracing::info!("Initializing with InMemory storage backend");
                let mem = Arc::new(RwLock::new(Box::new(InMemoryMemoryStore::new(
                    memory_config.clone(),
                )) as Box<dyn MemoryStore>));
                let strat = Arc::new(RwLock::new(Box::new(InMemoryStrategyStore::new(
                    config.strategy_config.clone(),
                )) as Box<dyn StrategyStore>));
                (mem, strat, None)
            },
            StorageBackend::Persistent => {
                tracing::info!(
                    "Initializing with Persistent storage backend (redb) at {:?}",
                    config.redb_path
                );

                // Initialize redb backend
                let redb_config = RedbConfig {
                    data_path: config.redb_path.clone(),
                    cache_size_bytes: config.redb_cache_size_mb * 1024 * 1024,
                    repair_on_open: false,
                };
                let backend = Arc::new(RedbBackend::open(redb_config).map_err(|e| {
                    GraphError::OperationError(format!("Failed to open redb: {:?}", e))
                })?);

                // Create memory store with LRU cache
                let mut mem_store = RedbMemoryStore::new(
                    backend.clone(),
                    memory_config.clone(),
                    config.memory_cache_size.max(1000), // Minimum 1000 entries
                );
                mem_store.initialize().map_err(|e| {
                    GraphError::OperationError(format!(
                        "Failed to initialize memory store: {:?}",
                        e
                    ))
                })?;

                // Create strategy store with LRU cache
                let mut strat_store = RedbStrategyStore::new(
                    backend.clone(),
                    config.strategy_config.clone(),
                    config.strategy_cache_size.max(500), // Minimum 500 entries
                );
                strat_store.initialize().map_err(|e| {
                    GraphError::OperationError(format!(
                        "Failed to initialize strategy store: {:?}",
                        e
                    ))
                })?;

                let mem = Arc::new(RwLock::new(Box::new(mem_store) as Box<dyn MemoryStore>));
                let strat = Arc::new(RwLock::new(Box::new(strat_store) as Box<dyn StrategyStore>));

                tracing::info!(
                    "Persistent storage initialized: memory_cache={}, strategy_cache={}",
                    config.memory_cache_size,
                    config.strategy_cache_size
                );

                (mem, strat, Some(backend))
            },
        };

        let transition_model = Arc::new(RwLock::new(TransitionModel::new(
            TransitionModelConfig::default(),
        )));

        // Initialize advanced graph features
        let index_manager = Arc::new(RwLock::new(IndexManager::new()));

        // Auto-create common indexes for fast lookups
        {
            let mut idx_mgr = index_manager.write().await;

            // Context hash index (exact match, high frequency)
            idx_mgr.create_index(
                "context_hash_idx".to_string(),
                "context_hash".to_string(),
                IndexType::Hash,
            )?;

            // Agent type index (exact match)
            idx_mgr.create_index(
                "agent_type_idx".to_string(),
                "agent_type".to_string(),
                IndexType::Hash,
            )?;

            // Event type index (exact match)
            idx_mgr.create_index(
                "event_type_idx".to_string(),
                "event_type".to_string(),
                IndexType::Hash,
            )?;

            // Significance index (range queries)
            idx_mgr.create_index(
                "significance_idx".to_string(),
                "significance".to_string(),
                IndexType::BTree,
            )?;
        }

        let louvain = Arc::new(LouvainAlgorithm::new());
        let centrality = Arc::new(CentralityMeasures::new());
        let random_walker = Arc::new(RandomWalker::with_config(
            crate::algorithms::RandomWalkConfig {
                restart_probability: 0.15,
                ..Default::default()
            },
        ));
        let temporal_reachability = Arc::new(TemporalReachability::new());
        let label_propagation = Arc::new(LabelPropagationAlgorithm::new());

        // Initialize semantic memory components if enabled
        let (ner_queue, ner_store) = if config.enable_semantic_memory {
            tracing::info!("Initializing semantic memory with NER extraction");

            // Create NER extractor (external service)
            let extractor = Arc::new(
                agent_db_ner::NerServiceExtractor::new(agent_db_ner::NerServiceConfig {
                    base_url: config.ner_service_url.clone(),
                    request_timeout_ms: config.ner_request_timeout_ms,
                    model: config.ner_model.clone(),
                    max_retries: 3,
                    retry_delay_ms: 100,
                })
                .map_err(|e| {
                    GraphError::OperationError(format!("Failed to initialize NER client: {}", e))
                })?,
            );

            // Create NER extraction queue
            let queue = Arc::new(agent_db_ner::NerExtractionQueue::new(
                extractor,
                config.ner_workers,
            ));

            // Create NER storage
            let store = if let Some(path) = &config.ner_storage_path {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                let store = agent_db_ner::NerFeatureStore::new(path).map_err(|e| {
                    GraphError::OperationError(format!("Failed to initialize NER storage: {}", e))
                })?;
                Some(Arc::new(store))
            } else {
                None
            };

            tracing::info!(
                "Semantic memory initialized with {} NER workers",
                config.ner_workers
            );
            (Some(queue), store)
        } else {
            tracing::warn!("NER disabled — claims will have no entity annotations");
            (None, None)
        };

        // Initialize claim extraction components if semantic memory is enabled
        let (claim_queue, claim_store, llm_client, embedding_client) = if config
            .enable_semantic_memory
        {
            tracing::info!("Initializing claim extraction pipeline");

            // Create claim store
            let store = if let Some(path) = &config.claim_storage_path {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                let store = crate::claims::ClaimStore::new(path).map_err(|e| {
                    GraphError::OperationError(format!("Failed to initialize claim storage: {}", e))
                })?;
                Some(Arc::new(store))
            } else {
                None
            };

            // Create LLM client
            let client: Arc<dyn crate::claims::LlmClient> = if let Some(key) =
                &config.openai_api_key
            {
                Arc::new(crate::claims::OpenAiClient::new(
                    key.clone(),
                    config.llm_model.clone(),
                ))
            } else {
                tracing::warn!(
                        "No LLM_API_KEY set — claims extraction will use MockClient (no real claims produced)"
                    );
                Arc::new(crate::claims::MockClient::new())
            };

            // Create embedding client (requires API key — no mock fallback)
            let embedding_client: Option<Arc<dyn crate::claims::EmbeddingClient>> = if let Some(
                key,
            ) =
                &config.openai_api_key
            {
                Some(Arc::new(crate::claims::OpenAiEmbeddingClient::new(
                    key.clone(),
                    "text-embedding-3-small".to_string(),
                )))
            } else {
                tracing::warn!(
                        "No openai_api_key — embedding generation disabled (semantic search will be keyword-only)"
                    );
                None
            };

            // Create claim extraction queue (requires embedding client)
            let queue = if let (Some(ref store), Some(ref emb)) = (&store, &embedding_client) {
                let extraction_config = crate::claims::ClaimExtractionConfig {
                    max_claims_per_input: config.claim_max_per_input,
                    min_confidence: config.claim_min_confidence,
                    min_evidence_length: 10,
                    enable_dedup: true,
                    maintenance_config: config.maintenance_config.clone(),
                    custom_instructions: config.custom_extraction_instructions.clone(),
                    extraction_includes: config.extraction_includes.clone(),
                    extraction_excludes: config.extraction_excludes.clone(),
                    few_shot_examples: config.extraction_few_shot_examples.clone(),
                };

                let queue = Arc::new(crate::claims::ClaimExtractionQueue::new(
                    client.clone(),
                    emb.clone(),
                    store.clone(),
                    config.claim_workers,
                    extraction_config,
                ));
                Some(queue)
            } else {
                None
            };

            tracing::info!(
                "Claim extraction initialized with {} workers",
                config.claim_workers
            );
            (queue, store, Some(client), embedding_client)
        } else {
            (None, None, None, None)
        };

        // Initialize embedding generation components if semantic memory is enabled
        let (embedding_queue, embedding_client) = if config.enable_semantic_memory
            && config.enable_embedding_generation
        {
            tracing::info!("Initializing embedding generation pipeline");

            // Use the client created above if available, otherwise create a new one
            let client: Option<Arc<dyn crate::claims::EmbeddingClient>> =
                if let Some(ref client) = embedding_client {
                    Some(client.clone())
                } else if let Some(key) = &config.openai_api_key {
                    Some(Arc::new(crate::claims::OpenAiEmbeddingClient::new(
                        key.clone(),
                        "text-embedding-3-small".to_string(),
                    )))
                } else {
                    tracing::warn!("No openai_api_key — embedding generation pipeline disabled");
                    None
                };

            // Create embedding queue if both client and store are available
            let queue = if let (Some(ref emb), Some(ref store)) = (&client, &claim_store) {
                let queue = Arc::new(crate::claims::EmbeddingQueue::new(
                    emb.clone(),
                    store.clone(),
                    config.embedding_workers,
                ));
                Some(queue)
            } else {
                None
            };

            tracing::info!(
                "Embedding generation initialized with {} workers",
                config.embedding_workers
            );
            (queue, client)
        } else {
            (None, embedding_client)
        };

        // 10x/100x: Build consolidation + refinement before config is moved
        let consolidation_engine_arc =
            Arc::new(RwLock::new(crate::consolidation::ConsolidationEngine::new(
                config.consolidation_config.clone(),
                100_000,
            )));
        let refinement_engine_arc = {
            let mut refine_config = config.refinement_config.clone();
            if config.openai_api_key.is_some() {
                refine_config.enable_llm_refinement = true;
            }
            if config.openai_api_key.is_some() || config.enable_semantic_memory {
                refine_config.enable_summary_embedding = true;
            }
            let engine = crate::refinement::RefinementEngine::new(
                refine_config,
                config.openai_api_key.clone(),
            );
            Some(Arc::new(engine))
        };

        // Initialize world model (mode-aware)
        let world_model = if config.effective_world_model_mode() != WorldModelMode::Disabled {
            tracing::info!(
                "Initializing world model (mode={:?})",
                config.effective_world_model_mode()
            );
            Some(Arc::new(RwLock::new(EbmWorldModel::new(
                config.world_model_config.clone(),
            ))))
        } else {
            None
        };

        // Initialize planning orchestrator + generators (LLM or mock)
        let (planning_orchestrator, strategy_generator_arc, action_generator_arc) =
            if config.planning_config.enable_strategy_generation
                || config.planning_config.enable_action_generation
            {
                tracing::info!("Initializing planning orchestrator");
                let orch = PlanningOrchestrator::new(config.planning_config.clone());

                if let Some(ref api_key) = config.planning_llm_api_key {
                    tracing::info!(
                        "Using LLM generators (provider={})",
                        config.planning_llm_provider
                    );
                    let llm_client: Arc<dyn agent_db_planning::llm_client::PlanningLlmClient> =
                        match config.planning_llm_provider.as_str() {
                            "anthropic" => {
                                Arc::new(super::planning_llm_adapter::AnthropicPlanningClient::new(
                                    api_key.clone(),
                                    config.planning_config.llm_model.clone(),
                                ))
                            },
                            _ => Arc::new(super::planning_llm_adapter::OpenAiPlanningClient::new(
                                api_key.clone(),
                                config.planning_config.llm_model.clone(),
                            )),
                        };
                    let sg: Arc<dyn agent_db_planning::StrategyGenerator> =
                        Arc::new(agent_db_planning::llm_generator::LlmStrategyGenerator::new(
                            llm_client.clone(),
                            config.planning_config.clone(),
                        ));
                    let ag: Arc<dyn agent_db_planning::ActionGenerator> =
                        Arc::new(agent_db_planning::llm_generator::LlmActionGenerator::new(
                            llm_client,
                            config.planning_config.clone(),
                        ));
                    (Some(orch), Some(sg), Some(ag))
                } else {
                    tracing::info!("Using mock generators (no planning LLM API key)");
                    let sg: Arc<dyn agent_db_planning::StrategyGenerator> = Arc::new(
                        agent_db_planning::mock_generator::MockStrategyGenerator::new(
                            config.planning_config.strategy_candidates_k,
                        ),
                    );
                    let ag: Arc<dyn agent_db_planning::ActionGenerator> =
                        Arc::new(agent_db_planning::mock_generator::MockActionGenerator::new(
                            config.planning_config.action_candidates_n,
                        ));
                    (Some(orch), Some(sg), Some(ag))
                }
            } else {
                (None, None, None)
            };

        // Initialize unified LLM client (used for NLQ hint classification, conversation, etc.)
        let unified_llm_client: Option<Arc<dyn crate::llm_client::LlmClient>> = {
            let hint_key = config
                .nlq_hint_api_key
                .clone()
                .or_else(|| config.openai_api_key.clone());
            if let Some(key) = hint_key {
                let model = config.nlq_hint_model.clone();
                let client: Arc<dyn crate::llm_client::LlmClient> = match config
                    .nlq_hint_provider
                    .as_str()
                {
                    "anthropic" => Arc::new(crate::llm_client::AnthropicLlmClient::new(key, model)),
                    _ => Arc::new(crate::llm_client::OpenAiLlmClient::new(key, model)),
                };
                if config.enable_nlq_hint {
                    tracing::info!(
                        "NLQ hint classifier enabled (provider={}, model={})",
                        config.nlq_hint_provider,
                        config.nlq_hint_model,
                    );
                }
                Some(client)
            } else {
                if config.enable_nlq_hint {
                    tracing::warn!(
                        "NLQ hint enabled but no API key found (nlq_hint_api_key / openai_api_key) — hint disabled"
                    );
                }
                None
            }
        };

        // Initialize metadata normalizer (alias matching is always on)
        let metadata_normalizer =
            crate::metadata_normalize::MetadataNormalizer::new(&config.metadata_alias_config);

        // Initialize LLM metadata normalizer if enabled
        let metadata_llm_normalizer: Option<
            Arc<dyn crate::metadata_normalize::MetadataLlmNormalizer>,
        > = if config.enable_metadata_normalization {
            let hint_key = config
                .nlq_hint_api_key
                .clone()
                .or_else(|| config.openai_api_key.clone());
            let model = config
                .metadata_normalization_model
                .clone()
                .unwrap_or_else(|| config.nlq_hint_model.clone());
            if let Some(key) = hint_key {
                let client: Arc<dyn crate::metadata_normalize::MetadataLlmNormalizer> =
                    match config.nlq_hint_provider.as_str() {
                        "anthropic" => Arc::new(
                            crate::metadata_normalize::AnthropicMetadataNormalizer::new(key, model),
                        ),
                        _ => Arc::new(crate::metadata_normalize::OpenAiMetadataNormalizer::new(
                            key, model,
                        )),
                    };
                tracing::info!(
                    "Metadata LLM normalizer enabled (provider={})",
                    config.nlq_hint_provider,
                );
                Some(client)
            } else {
                tracing::warn!(
                    "Metadata normalization enabled but no API key found — LLM fallback disabled"
                );
                None
            }
        } else {
            None
        };

        // Initialize multi-signal retrieval BM25 indexes and populate from existing stores
        let memory_bm25_index = {
            let mut idx = crate::indexing::Bm25Index::new();
            let store = memory_store.read().await;
            for mem in store.list_all_memories() {
                let text = format!("{} {} {}", mem.summary, mem.takeaway, mem.causal_note);
                if !text.trim().is_empty() {
                    idx.index_document(mem.id, &text);
                }
            }
            let count = idx.stats().total_docs;
            if count > 0 {
                tracing::info!("Memory BM25 index populated with {} documents", count);
            }
            Arc::new(RwLock::new(idx))
        };
        let strategy_bm25_index = {
            let mut idx = crate::indexing::Bm25Index::new();
            let store = strategy_store.read().await;
            for strat in store.list_all_strategies() {
                let text = format!(
                    "{} {} {}",
                    strat.summary, strat.when_to_use, strat.action_hint
                );
                if !text.trim().is_empty() {
                    idx.index_document(strat.id, &text);
                }
            }
            let count = idx.stats().total_docs;
            if count > 0 {
                tracing::info!("Strategy BM25 index populated with {} documents", count);
            }
            Arc::new(RwLock::new(idx))
        };

        // Initialize RedbGraphStore alongside the existing backend
        let graph_store = if let Some(ref backend) = redb_backend {
            let store = crate::redb_graph_store::RedbGraphStore::new(
                backend.clone(),
                8, // max loaded partitions
            );
            tracing::info!("RedbGraphStore initialized for unified graph persistence");
            Some(Arc::new(RwLock::new(store)))
        } else {
            None
        };

        let engine = Self {
            inference,
            traversal,
            event_ordering,
            scoped_inference,
            episode_detector,
            memory_store,
            strategy_store,
            transition_model,
            event_store: Arc::new(RwLock::new(HashMap::new())),
            event_store_order: Arc::new(RwLock::new(VecDeque::new())),
            decision_traces: Arc::new(dashmap::DashMap::new()),
            redb_backend,
            graph_store,
            config,
            stats: Arc::new(GraphEngineStats::default()),
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            last_persistence: Arc::new(AtomicU64::new(0)),
            index_manager,
            louvain,
            centrality,
            random_walker,
            temporal_reachability,
            label_propagation,
            memory_bm25_index,
            strategy_bm25_index,
            ner_queue,
            ner_store,
            claim_queue,
            claim_store,
            llm_client,
            embedding_queue,
            embedding_client,
            // 10x/100x: Consolidation + Refinement — built BEFORE config is moved
            consolidation_engine: consolidation_engine_arc,
            refinement_engine: refinement_engine_arc,
            episodes_since_consolidation: Arc::new(AtomicU64::new(0)),
            world_model,
            planning_orchestrator,
            strategy_generator: strategy_generator_arc,
            action_generator: action_generator_arc,
            structured_memory: Arc::new(RwLock::new(
                crate::structured_memory::StructuredMemoryStore::new(),
            )),
            nlq_pipeline: crate::nlq::NlqPipeline::new(),
            nlq_contexts: tokio::sync::Mutex::new(HashMap::new()),
            conversation_states: tokio::sync::Mutex::new(HashMap::new()),
            unified_llm_client,
            metadata_normalizer,
            metadata_llm_normalizer,
            memory_audit_log: Arc::new(RwLock::new(crate::memory_audit::MemoryAuditLog::new())),
            conversation_summaries: Arc::new(RwLock::new(HashMap::new())),
            community_summaries: Arc::new(RwLock::new(HashMap::new())),
            goal_store: Arc::new(RwLock::new(crate::goal_store::GoalStore::new())),
            active_executions: Arc::new(dashmap::DashMap::new()),
            next_execution_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        };

        // Restore graph state from redb if available
        match engine.restore_graph_state().await {
            Ok((nodes, edges)) if nodes > 0 || edges > 0 => {
                tracing::info!("Restored graph from disk: {} nodes, {} edges", nodes, edges);
            },
            Ok(_) => {},
            Err(e) => {
                tracing::warn!("Failed to restore graph state (starting fresh): {}", e);
            },
        }

        // BUG 3 fix: Restore transition model from redb (version-aware)
        if let Some(ref backend) = engine.redb_backend {
            match backend.get_raw(table_names::TRANSITION_STATS, b"__model__") {
                Ok(Some(raw)) => {
                    let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                    match TransitionModel::from_bytes(bytes, TransitionModelConfig::default()) {
                        Ok(restored) => {
                            let ep_count = restored.episode_count();
                            *engine.transition_model.write().await = restored;
                            tracing::info!(
                                "Restored transition model from disk ({} episodes)",
                                ep_count
                            );
                        },
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize transition model (starting fresh): {}",
                                e
                            );
                        },
                    }
                },
                Ok(None) => {},
                Err(e) => {
                    tracing::warn!("Failed to read transition model from redb: {:?}", e);
                },
            }
        }

        // BUG 4 fix: Restore episode detector from redb (version-aware)
        if let Some(ref backend) = engine.redb_backend {
            match backend.get_raw(table_names::EPISODE_CATALOG, b"__detector__") {
                Ok(Some(raw)) => {
                    let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                    let mut ed = engine.episode_detector.write().await;
                    match ed.restore_from_bytes(bytes) {
                        Ok(()) => {
                            tracing::info!(
                                "Restored episode detector from disk (active={}, completed={})",
                                ed.get_active_episode(0).is_some() as u32, // just log something
                                ed.get_completed_episodes().len()
                            );
                        },
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize episode detector (starting fresh): {}",
                                e
                            );
                        },
                    }
                },
                Ok(None) => {},
                Err(e) => {
                    tracing::warn!("Failed to read episode detector from redb: {:?}", e);
                },
            }
        }

        // BUG 12 fix: Restore consolidation counter from redb (version-aware)
        if let Some(ref backend) = engine.redb_backend {
            match backend.get_raw(table_names::ID_ALLOCATOR, b"consolidation_counter") {
                Ok(Some(raw)) => {
                    let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                    if bytes.len() == 8 {
                        let counter = u64::from_be_bytes(
                            bytes.try_into().unwrap_or([0u8; 8])
                        );
                        engine
                            .episodes_since_consolidation
                            .store(counter, AtomicOrdering::Relaxed);
                        tracing::info!("Restored consolidation counter from disk: {}", counter);
                    }
                },
                Ok(None) => {},
                Err(e) => {
                    tracing::warn!("Failed to read consolidation counter from redb: {:?}", e);
                },
            }
        }

        // Restore world model from redb (version-aware)
        if engine.config.effective_world_model_mode() != WorldModelMode::Disabled {
            if let (Some(ref wm), Some(ref backend)) = (&engine.world_model, &engine.redb_backend) {
                match backend.get_raw(table_names::WORLD_MODEL, b"__weights__") {
                    Ok(Some(raw)) => {
                        let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                        match EbmWorldModel::from_bytes(bytes) {
                            Ok(restored) => {
                                let stats = restored.energy_stats();
                                *wm.write().await = restored;
                                tracing::info!(
                                    "Restored world model from disk (trained={}, scored={})",
                                    stats.total_trained,
                                    stats.total_scored
                                );
                            },
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to deserialize world model (starting fresh): {}",
                                    e
                                );
                            },
                        }
                    },
                    Ok(None) => {},
                    Err(e) => {
                        tracing::warn!("Failed to read world model from redb: {:?}", e);
                    },
                }
            }
        }

        Ok(engine)
    }
}
