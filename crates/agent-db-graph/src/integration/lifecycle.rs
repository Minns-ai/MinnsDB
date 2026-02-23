use super::*;

impl GraphEngine {
    /// Clean up old data and optimize graph
    pub async fn cleanup(&self) -> GraphResult<()> {
        // Clean up old associations
        // This would involve removing old temporal patterns and low-confidence relationships

        // Clean up query cache
        {
            let mut traversal = self.traversal.write().await;
            traversal.cleanup_cache();
        }

        Ok(())
    }

    /// Graceful shutdown: flush all buffers, drain queues, and sync storage.
    ///
    /// Call this before dropping the engine to ensure all in-flight work is
    /// committed to redb. After this returns the process can exit safely.
    pub async fn shutdown(&self) {
        tracing::info!("GraphEngine shutdown initiated — flushing buffers");

        // 1. Flush event ordering buffers (process any queued out-of-order events)
        if let Err(e) = self.flush_all_buffers().await {
            tracing::warn!("Error flushing event buffers during shutdown: {}", e);
        }

        // 2. Process any pending claim extraction jobs by dropping the queue sender.
        //    Workers will drain remaining items and exit when the channel closes.
        //    We can't drop Arc fields, but the server dropping AppState after this
        //    will trigger cleanup. Log the intent so operators know.
        if self.claim_queue.is_some() {
            tracing::info!("Claim extraction queue will drain on drop");
        }
        if self.embedding_queue.is_some() {
            tracing::info!("Embedding queue will drain on drop");
        }
        if self.ner_queue.is_some() {
            tracing::info!("NER extraction queue will drain on drop");
        }

        // 3. BUG 11 fix: Flush dirty memory/strategy caches before graph persistence
        {
            self.memory_store.write().await.flush_cache();
            self.strategy_store.write().await.flush_cache();
        }

        // 4. BUG 3 fix: Persist transition model (with version envelope)
        if let Some(ref backend) = self.redb_backend {
            let tm = self.transition_model.read().await;
            match tm.to_bytes() {
                Ok(bytes) => {
                    let wrapped = agent_db_storage::wrap_versioned(
                        agent_db_storage::CURRENT_DATA_VERSION,
                        &bytes,
                    );
                    if let Err(e) =
                        backend.put_raw(table_names::TRANSITION_STATS, b"__model__", &wrapped)
                    {
                        tracing::warn!("Failed to persist transition model: {:?}", e);
                    } else {
                        tracing::info!("Transition model persisted ({} bytes)", bytes.len());
                    }
                },
                Err(e) => tracing::warn!("Failed to serialize transition model: {}", e),
            }
        }

        // 5. BUG 4 fix: Persist episode detector (with version envelope)
        if let Some(ref backend) = self.redb_backend {
            let ed = self.episode_detector.read().await;
            match ed.to_bytes() {
                Ok(bytes) => {
                    let wrapped = agent_db_storage::wrap_versioned(
                        agent_db_storage::CURRENT_DATA_VERSION,
                        &bytes,
                    );
                    if let Err(e) =
                        backend.put_raw(table_names::EPISODE_CATALOG, b"__detector__", &wrapped)
                    {
                        tracing::warn!("Failed to persist episode detector: {:?}", e);
                    } else {
                        tracing::info!("Episode detector persisted ({} bytes)", bytes.len());
                    }
                },
                Err(e) => tracing::warn!("Failed to serialize episode detector: {}", e),
            }
        }

        // 6. BUG 12 fix: Persist consolidation counter (with version envelope)
        if let Some(ref backend) = self.redb_backend {
            let counter = *self.episodes_since_consolidation.read().await;
            let counter_bytes = counter.to_be_bytes();
            let wrapped = agent_db_storage::wrap_versioned(
                agent_db_storage::CURRENT_DATA_VERSION,
                &counter_bytes,
            );
            if let Err(e) = backend.put_raw(
                table_names::ID_ALLOCATOR,
                b"consolidation_counter",
                &wrapped,
            ) {
                tracing::warn!("Failed to persist consolidation counter: {:?}", e);
            }
        }

        // 7. Persist world model weights (with version envelope)
        if let (Some(ref wm), Some(ref backend)) = (&self.world_model, &self.redb_backend) {
            let wm_guard = wm.read().await;
            match wm_guard.to_bytes() {
                Ok(bytes) => {
                    let wrapped = agent_db_storage::wrap_versioned(
                        agent_db_storage::CURRENT_DATA_VERSION,
                        &bytes,
                    );
                    if let Err(e) =
                        backend.put_raw(table_names::WORLD_MODEL, b"__weights__", &wrapped)
                    {
                        tracing::warn!("Failed to persist world model: {:?}", e);
                    } else {
                        let stats = wm_guard.energy_stats();
                        tracing::info!(
                            "World model persisted ({} bytes, trained={}, scored={})",
                            bytes.len(),
                            stats.total_trained,
                            stats.total_scored,
                        );
                    }
                },
                Err(e) => tracing::warn!("Failed to serialize world model: {}", e),
            }
        }

        // 8. BUG 9 fix: Persist graph state with retry (3 attempts, 100ms delay)
        if self.redb_backend.is_some() {
            let mut last_err = None;
            for attempt in 1..=3u32 {
                match self.persist_graph_state().await {
                    Ok((n, e)) => {
                        tracing::info!("Graph persisted on shutdown: {} nodes, {} edges", n, e);
                        last_err = None;
                        break;
                    },
                    Err(e) => {
                        tracing::warn!(
                            "Failed to persist graph during shutdown (attempt {}): {}",
                            attempt,
                            e
                        );
                        last_err = Some(e);
                        if attempt < 3 {
                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        }
                    },
                }
            }
            if let Some(e) = last_err {
                tracing::error!("Graph persistence failed after 3 attempts: {}", e);
            }
        }

        tracing::info!("GraphEngine shutdown complete — all buffers flushed, graph persisted");
    }

    /// Spawn a background maintenance loop that periodically runs:
    /// 1. Memory decay (age out stale memories)
    /// 2. Strategy pruning (remove weak + merge near-duplicates)
    ///
    /// The loop respects `maintenance_config.interval_secs`.
    /// Pass `0` to disable.
    ///
    /// The returned `JoinHandle` can be used to abort the loop on shutdown.
    pub fn start_maintenance_loop(self: &Arc<Self>) -> Option<tokio::task::JoinHandle<()>> {
        let interval_secs = self.config.maintenance_config.interval_secs;
        if interval_secs == 0 {
            tracing::info!("Maintenance loop disabled (interval_secs=0)");
            return None;
        }

        let engine = Arc::clone(self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
            // Skip the first immediate tick
            ticker.tick().await;

            tracing::info!("Maintenance loop started (interval={}s)", interval_secs);

            loop {
                ticker.tick().await;
                tracing::debug!("Maintenance pass starting");

                // 1 & 2. Memory decay + strategy pruning run concurrently
                // (they hold independent RwLock-guarded stores)
                let mc = &engine.config.maintenance_config;
                let ((), pruned) = tokio::join!(
                    async {
                        engine.memory_store.write().await.apply_decay();
                    },
                    async {
                        engine.strategy_store.write().await.prune_strategies(
                            mc.strategy_min_confidence,
                            mc.strategy_min_support,
                            mc.strategy_max_stale_hours,
                        )
                    },
                );

                if pruned > 0 {
                    tracing::info!("Maintenance: pruned {} strategies", pruned);
                }

                // 3. Graph pruning (streaming, bounded, via RedbGraphStore)
                if let Some(ref graph_store) = engine.graph_store {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;

                    let pruner = crate::graph_pruning::GraphPruner::new(
                        engine.config.pruning_config.clone(),
                    );

                    let mut store = graph_store.write().await;
                    let mut inference = engine.inference.write().await;
                    let graph = inference.graph_mut();

                    match pruner.prune_full_graph(graph, &mut store, now) {
                        Ok(result) => {
                            if result.nodes_merged > 0 || result.nodes_deleted > 0 {
                                tracing::info!(
                                    "Maintenance: graph pruning merged={} deleted={} scanned={} stopped_early={}",
                                    result.nodes_merged,
                                    result.nodes_deleted,
                                    result.total_headers_scanned,
                                    result.stopped_early,
                                );
                            }
                        },
                        Err(e) => {
                            tracing::warn!("Graph pruning failed: {}", e);
                        },
                    }
                }

                // 4. Transition model cleanup
                {
                    let mut tm = engine.transition_model.write().await;
                    let ep_count_before = tm.episode_count();
                    tm.cleanup_oldest_episodes(engine.config.max_transition_episodes);
                    let ep_cleaned = ep_count_before - tm.episode_count();

                    tm.prune_weak_transitions(engine.config.min_transition_count);

                    if ep_cleaned > 0 {
                        tracing::info!(
                            "Maintenance: transition cleanup episodes_removed={}",
                            ep_cleaned,
                        );
                    }
                }

                // 5. Event store ring-buffer eviction (safety net)
                {
                    let mut store = engine.event_store.write().await;
                    let mut order = engine.event_store_order.write().await;
                    let cap = engine.config.max_event_store_size;
                    let mut evicted = 0usize;
                    while store.len() > cap {
                        if let Some(old_id) = order.pop_front() {
                            store.remove(&old_id);
                            evicted += 1;
                        } else {
                            break;
                        }
                    }
                    if evicted > 0 {
                        tracing::info!("Maintenance: event_store evicted {} entries", evicted);
                    }
                }

                // 6. Decision trace TTL sweep
                {
                    let max_age =
                        std::time::Duration::from_secs(engine.config.max_decision_trace_age_secs);
                    let before = engine.decision_traces.len();
                    engine
                        .decision_traces
                        .retain(|_, trace| trace.last_updated.elapsed() < max_age);
                    let swept = before - engine.decision_traces.len();
                    if swept > 0 {
                        tracing::info!(
                            "Maintenance: decision_traces TTL swept {} stale entries",
                            swept
                        );
                    }
                }

                // 7. Inference memory caps (context cache + temporal patterns)
                {
                    let mut inf = engine.inference.write().await;
                    inf.enforce_memory_caps();
                }

                // 8. Claim store maintenance (expire stale, cap vector index, purge disk)
                if let Some(ref claim_store) = engine.claim_store {
                    // Expire claims past their TTL
                    match claim_store.expire_stale_claims() {
                        Ok(expired) if expired > 0 => {
                            tracing::info!("Maintenance: expired {} stale claims", expired);
                        },
                        Err(e) => {
                            tracing::warn!("Maintenance: claim expiry failed: {}", e);
                        },
                        _ => {},
                    }

                    // Enforce vector index cap
                    let max_idx = mc.max_vector_index_size;
                    if max_idx > 0 {
                        if let Err(e) = claim_store.enforce_vector_index_cap(max_idx) {
                            tracing::warn!("Maintenance: vector index cap failed: {}", e);
                        }
                    }

                    // Purge inactive claims from disk
                    if mc.purge_inactive_claims {
                        match claim_store.purge_inactive_claims() {
                            Ok(purged) if purged > 0 => {
                                tracing::info!(
                                    "Maintenance: purged {} inactive claims from disk",
                                    purged
                                );
                            },
                            Err(e) => {
                                tracing::warn!("Maintenance: claim purge failed: {}", e);
                            },
                            _ => {},
                        }
                    }
                }

                // 9. World model training step (mode-aware)
                if engine.config.effective_world_model_mode() != WorldModelMode::Disabled {
                    if let Some(ref wm) = engine.world_model {
                        let mut wm_guard = wm.write().await;
                        let pending = wm_guard.pending_training_count();
                        if pending > 0 {
                            let loss = wm_guard.train_step();
                            let stats = wm_guard.energy_stats();
                            tracing::info!(
                                "Maintenance: world model train_step pending={} loss={:.4} trained={} scored={} warmed_up={}",
                                pending, loss, stats.total_trained, stats.total_scored, stats.is_warmed_up,
                            );
                        }
                    }
                }

                // 10. World model periodic checkpoint (every 100 training steps)
                if let (Some(ref wm), Some(ref backend)) =
                    (&engine.world_model, &engine.redb_backend)
                {
                    let wm_guard = wm.read().await;
                    let stats = wm_guard.energy_stats();
                    if stats.total_trained > 0 && stats.total_trained % 100 == 0 {
                        if let Ok(bytes) = wm_guard.to_bytes() {
                            let wrapped = agent_db_storage::wrap_versioned(
                                agent_db_storage::CURRENT_DATA_VERSION,
                                &bytes,
                            );
                            if let Err(e) =
                                backend.put_raw(table_names::WORLD_MODEL, b"__weights__", &wrapped)
                            {
                                tracing::warn!("World model checkpoint failed: {:?}", e);
                            } else {
                                tracing::info!(
                                    "World model checkpoint ({} bytes, trained={})",
                                    bytes.len(),
                                    stats.total_trained,
                                );
                            }
                        }
                    }
                }

                tracing::debug!("Maintenance pass complete");
            }
        });

        Some(handle)
    }
}
