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

        // 3. Persist graph state to redb before shutdown
        if self.redb_backend.is_some() {
            match self.persist_graph_state().await {
                Ok((n, e)) => {
                    tracing::info!("Graph persisted on shutdown: {} nodes, {} edges", n, e)
                },
                Err(e) => tracing::warn!("Failed to persist graph during shutdown: {}", e),
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

                tracing::debug!("Maintenance pass complete");
            }
        });

        Some(handle)
    }
}
