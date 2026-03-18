use super::*;

use crate::export::{export_to_writer, import_from_reader, ImportMode, ImportStats};

impl GraphEngine {
    /// Flush all in-memory caches and persist graph/transition/episode/counter
    /// state to redb. Must be called before `export_backend()` to ensure the
    /// export captures everything.
    pub async fn export_prepare(&self) -> GraphResult<()> {
        // Flush caches to ensure all in-memory state is on disk
        self.memory_store.write().await.flush_cache();
        self.strategy_store.write().await.flush_cache();

        // Persist graph state
        if self.redb_backend.is_some() {
            self.persist_graph_state().await?;
        }

        // Persist transition model
        if let Some(ref backend) = self.redb_backend {
            let tm = self.transition_model.read().await;
            if let Ok(bytes) = tm.to_bytes() {
                let wrapped = agent_db_storage::wrap_versioned(
                    agent_db_storage::CURRENT_DATA_VERSION,
                    &bytes,
                );
                let _ = backend.put_raw(table_names::TRANSITION_STATS, b"__model__", &wrapped);
            }

            let ed = self.episode_detector.read().await;
            if let Ok(bytes) = ed.to_bytes() {
                let wrapped = agent_db_storage::wrap_versioned(
                    agent_db_storage::CURRENT_DATA_VERSION,
                    &bytes,
                );
                let _ = backend.put_raw(table_names::EPISODE_CATALOG, b"__detector__", &wrapped);
            }

            let counter = self
                .episodes_since_consolidation
                .load(AtomicOrdering::Relaxed);
            let counter_bytes = counter.to_be_bytes();
            let wrapped = agent_db_storage::wrap_versioned(
                agent_db_storage::CURRENT_DATA_VERSION,
                &counter_bytes,
            );
            let _ = backend.put_raw(
                table_names::ID_ALLOCATOR,
                b"consolidation_counter",
                &wrapped,
            );
        }

        Ok(())
    }

    /// Get the ReDB backend, if persistent storage is configured.
    pub fn redb_backend(&self) -> Option<&RedbBackend> {
        self.redb_backend.as_ref().map(|arc| arc.as_ref())
    }

    /// Get a reference to the redb backend for sync export operations.
    /// Call `export_prepare()` first.
    pub fn export_backend(&self) -> GraphResult<&RedbBackend> {
        self.redb_backend
            .as_ref()
            .map(|arc| arc.as_ref())
            .ok_or_else(|| {
                GraphError::OperationError("Export requires persistent storage backend".to_string())
            })
    }

    /// Convenience wrapper: prepare + export in one call.
    /// Suitable for tests and non-HTTP contexts.
    pub async fn export<W: std::io::Write>(&self, writer: W) -> GraphResult<u64> {
        self.export_prepare().await?;
        let backend = self.export_backend()?;
        export_to_writer(backend, writer)
            .map_err(|e| GraphError::OperationError(format!("Export failed: {}", e)))
    }

    /// Sync import: reads from a binary v2 stream and writes records to redb.
    /// Does NOT reinitialize in-memory stores — call `import_finalize()` after.
    ///
    /// Replace mode requires exclusive access — no concurrent graph operations
    /// during import.
    pub fn import_sync<R: std::io::Read>(
        &self,
        reader: R,
        mode: ImportMode,
    ) -> GraphResult<ImportStats> {
        let backend = self.redb_backend.as_ref().ok_or_else(|| {
            GraphError::OperationError("Import requires persistent storage backend".to_string())
        })?;

        import_from_reader(backend, reader, mode)
            .map_err(|e| GraphError::OperationError(format!("Import failed: {}", e)))
    }

    /// Async finalization after a successful import: reinitializes in-memory
    /// stores and restores graph/transition/episode/counter state.
    ///
    /// Do NOT call this on import failure — the caller should retry or restore.
    pub async fn import_finalize(&self) -> GraphResult<()> {
        // Re-initialize memory/strategy stores so they pick up imported records
        // and reset their ID allocators to avoid collisions
        self.memory_store.write().await.reinitialize();
        self.strategy_store.write().await.reinitialize();

        // Restore graph state from the newly imported data
        self.restore_graph_state().await?;

        // Restore transition model
        if let Some(ref backend) = self.redb_backend {
            if let Ok(Some(raw)) = backend.get_raw(table_names::TRANSITION_STATS, b"__model__") {
                let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                if let Ok(restored) =
                    TransitionModel::from_bytes(bytes, TransitionModelConfig::default())
                {
                    *self.transition_model.write().await = restored;
                }
            }

            if let Ok(Some(raw)) = backend.get_raw(table_names::EPISODE_CATALOG, b"__detector__") {
                let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                let mut ed = self.episode_detector.write().await;
                let _ = ed.restore_from_bytes(bytes);
            }

            if let Ok(Some(raw)) =
                backend.get_raw(table_names::ID_ALLOCATOR, b"consolidation_counter")
            {
                let (_version, bytes) = agent_db_storage::unwrap_versioned(&raw);
                if bytes.len() == 8 {
                    let counter = u64::from_be_bytes(bytes.try_into().unwrap());
                    self.episodes_since_consolidation
                        .store(counter, AtomicOrdering::Relaxed);
                }
            }
        }

        Ok(())
    }

    /// Convenience wrapper: sync import + async finalize in one call.
    /// Suitable for tests and non-HTTP contexts.
    ///
    /// Replace mode requires exclusive access — no concurrent graph operations
    /// during import.
    pub async fn import<R: std::io::Read>(
        &self,
        reader: R,
        mode: ImportMode,
    ) -> GraphResult<ImportStats> {
        let stats = self.import_sync(reader, mode)?;
        self.import_finalize().await?;
        Ok(stats)
    }
}
