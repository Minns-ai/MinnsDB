//! Store boundaries for memory and strategy layers.
//!
//! These abstractions keep the graph engine independent from persistence.

use agent_db_core::types::{AgentId, ContextHash, SessionId};
use agent_db_events::core::{Event, EventContext};
use agent_db_storage::{
    deserialize_versioned, serialize_versioned, BatchOperation, ForEachError, RedbBackend,
    StorageResult,
};
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::episodes::Episode;
use crate::memory::{
    Memory, MemoryFormation, MemoryFormationConfig, MemoryId, MemoryStats, MemoryType, MemoryUpsert,
};
use crate::strategies::{
    Strategy, StrategyExtractionConfig, StrategyExtractor, StrategyId, StrategySimilarityQuery,
    StrategyStats, StrategyUpsert,
};

pub trait MemoryStore: Send + Sync {
    fn store_episode(&mut self, episode: &Episode, events: &[Event]) -> Option<MemoryUpsert>;
    fn get_memory(&self, memory_id: MemoryId) -> Option<Memory>;
    fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory>;
    fn retrieve_by_context(&mut self, context: &EventContext, limit: usize) -> Vec<Memory>;
    fn retrieve_by_context_similar(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory>;
    fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool;
    fn get_stats(&self) -> MemoryStats;
    fn apply_decay(&mut self);

    // ========== Consolidation API ==========
    /// List all memories (for consolidation passes)
    fn list_all_memories(&self) -> Vec<Memory>;
    /// Store a pre-built consolidated memory (Semantic or Schema tier)
    fn store_consolidated_memory(&mut self, memory: Memory);
    /// Mark a memory as consolidated, linking it to a higher-tier memory and applying decay.
    /// When `into_tier` is `Schema`, sets `schema_id = Some(into_id)`.
    /// When `into_tier` is `Semantic`, leaves `schema_id` unchanged (None).
    fn mark_consolidated(
        &mut self,
        memory_id: MemoryId,
        into_id: MemoryId,
        into_tier: crate::memory::MemoryTier,
        decay: f32,
    );
    /// Schema-first retrieval: returns Schema > Semantic > Episodic, preferring higher tiers
    fn retrieve_hierarchical(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
    ) -> Vec<Memory> {
        // Default: fall back to flat retrieval, implementations can override
        self.retrieve_by_context_similar(context, limit, min_similarity, agent_id, None)
    }

    // ========== Batch Operations ==========
    /// Store multiple consolidated memories in a single atomic transaction.
    /// Default: falls back to individual calls.
    fn store_consolidated_memories_batch(&mut self, memories: Vec<Memory>) {
        for m in memories {
            self.store_consolidated_memory(m);
        }
    }

    /// Flush all dirty cache entries to persistent storage.
    /// No-op for in-memory stores; redb stores batch-persist the cache.
    fn flush_cache(&mut self) {}

    /// Re-scan persistent storage to reset the ID allocator and clear caches.
    /// Called after import to pick up externally-written records.
    /// No-op for in-memory stores.
    fn reinitialize(&mut self) {}

    /// Mark multiple memories as consolidated in a single atomic transaction.
    /// Each tuple: (memory_id, into_id, into_tier, decay).
    /// Default: falls back to individual calls.
    fn mark_consolidated_batch(
        &mut self,
        batch: Vec<(MemoryId, MemoryId, crate::memory::MemoryTier, f32)>,
    ) {
        for (mid, into_id, tier, decay) in batch {
            self.mark_consolidated(mid, into_id, tier, decay);
        }
    }

    /// Delete multiple memories in a single atomic transaction.
    /// Returns the number of memories actually deleted.
    fn delete_memories_batch(&mut self, ids: Vec<MemoryId>) -> usize {
        let mut deleted = 0;
        for id in ids {
            if let Some(memory) = self.get_memory(id) {
                // Default: just remove from store (in-memory stores can override)
                let _ = memory;
                deleted += 1;
            }
        }
        deleted
    }
}

pub struct InMemoryMemoryStore {
    inner: MemoryFormation,
}

impl InMemoryMemoryStore {
    pub fn new(config: MemoryFormationConfig) -> Self {
        Self {
            inner: MemoryFormation::new(config),
        }
    }
}

impl MemoryStore for InMemoryMemoryStore {
    fn store_episode(&mut self, episode: &Episode, events: &[Event]) -> Option<MemoryUpsert> {
        self.inner.form_memory(episode, events)
    }

    fn get_memory(&self, memory_id: MemoryId) -> Option<Memory> {
        self.inner.get_memory(memory_id).cloned()
    }

    fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory> {
        self.inner
            .retrieve_by_agent(agent_id, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    fn retrieve_by_context(&mut self, context: &EventContext, limit: usize) -> Vec<Memory> {
        self.inner.retrieve_by_context(context, limit)
    }

    fn retrieve_by_context_similar(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory> {
        self.inner
            .retrieve_by_context_similar(context, limit, min_similarity, agent_id, session_id)
    }

    fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool {
        self.inner.apply_outcome(memory_id, success)
    }

    fn get_stats(&self) -> MemoryStats {
        self.inner.get_stats()
    }

    fn apply_decay(&mut self) {
        self.inner.apply_decay();
    }

    fn list_all_memories(&self) -> Vec<Memory> {
        self.inner.list_all()
    }

    fn store_consolidated_memory(&mut self, memory: Memory) {
        self.inner.store_direct(memory);
    }

    fn mark_consolidated(
        &mut self,
        memory_id: MemoryId,
        into_id: MemoryId,
        into_tier: crate::memory::MemoryTier,
        decay: f32,
    ) {
        self.inner
            .mark_consolidated(memory_id, into_id, into_tier, decay);
    }

    fn retrieve_hierarchical(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
    ) -> Vec<Memory> {
        self.inner
            .retrieve_hierarchical(context, limit, min_similarity, agent_id)
    }

    fn delete_memories_batch(&mut self, ids: Vec<MemoryId>) -> usize {
        let mut deleted = 0;
        for id in ids {
            if self.inner.remove_memory(id) {
                deleted += 1;
            }
        }
        deleted
    }
}

/// Redb-backed memory store with LRU cache for scalability
///
/// **Architecture**: Hot/Cold separation
/// - Hot: Recently accessed memories in LRU cache (bounded size)
/// - Cold: All memories persisted in redb (unbounded)
///
/// Uses the following redb tables:
/// - memory_records: memory_id -> Memory  (main storage)
/// - mem_by_context_hash: (context_fingerprint, memory_id) -> ()  (exact context index)
/// - mem_by_bucket: (agent_id, memory_id) -> ()  (agent index)
/// - mem_by_goal_bucket: (goal_bucket_id, memory_id) -> ()  (goal bucket index for consolidation)
/// - mem_feature_postings: (feature, memory_id) -> relevance_score  (similarity search)
pub struct RedbMemoryStore {
    backend: Arc<RedbBackend>,
    config: MemoryFormationConfig,
    /// LRU cache for hot memories (bounded size, O(1) eviction)
    memory_cache: lru::LruCache<MemoryId, Memory>,
    /// IDs of cache entries whose mutations have not yet been flushed to redb
    dirty_ids: HashSet<MemoryId>,
    /// Next memory ID allocator
    next_memory_id: MemoryId,
    // ========== Cached stats counters (Phase 5C) ==========
    total_memories: usize,
    episodic_count: usize,
    semantic_count: usize,
    schema_count: usize,
    sum_strength: f32,
    sum_access_count: u64,
    /// Refcount map: agent_id -> number of memories for that agent
    agent_refcounts: rustc_hash::FxHashMap<AgentId, u32>,
    /// Refcount map: context_fingerprint -> number of memories with that fingerprint
    context_refcounts: rustc_hash::FxHashMap<u64, u32>,
}

impl RedbMemoryStore {
    /// Create a new redb memory store with LRU cache
    ///
    /// # Arguments
    /// * `backend` - Redb backend for persistence
    /// * `config` - Memory formation configuration
    /// * `max_cache_size` - Maximum number of memories to keep in LRU cache
    pub fn new(
        backend: Arc<RedbBackend>,
        config: MemoryFormationConfig,
        max_cache_size: usize,
    ) -> Self {
        let cap = NonZeroUsize::new(max_cache_size).unwrap_or(NonZeroUsize::new(1024).unwrap());
        Self {
            backend,
            config,
            memory_cache: lru::LruCache::new(cap),
            dirty_ids: HashSet::new(),
            next_memory_id: 1,
            total_memories: 0,
            episodic_count: 0,
            semantic_count: 0,
            schema_count: 0,
            sum_strength: 0.0,
            sum_access_count: 0,
            agent_refcounts: rustc_hash::FxHashMap::default(),
            context_refcounts: rustc_hash::FxHashMap::default(),
        }
    }

    /// Initialize next_memory_id by scanning existing memories and building stats counters.
    pub fn initialize(&mut self) -> StorageResult<()> {
        // Try to load persisted ID allocator first
        if let Ok(Some(id_bytes)) = self.backend.get_raw("id_allocator", b"next_memory_id") {
            if id_bytes.len() >= 8 {
                self.next_memory_id = u64::from_be_bytes(id_bytes[..8].try_into().unwrap());
            }
        }

        // Reset stats counters
        self.total_memories = 0;
        self.episodic_count = 0;
        self.semantic_count = 0;
        self.schema_count = 0;
        self.sum_strength = 0.0;
        self.sum_access_count = 0;
        self.agent_refcounts.clear();
        self.context_refcounts.clear();

        // Streaming scan to build stats and find max ID
        let mut skipped_count = 0usize;
        let scan_result: Result<(), ForEachError<std::convert::Infallible>> = self
            .backend
            .for_each_prefix_raw("memory_records", vec![], |_key, value| {
                match deserialize_versioned::<Memory>(value) {
                    Ok(memory) => {
                        if memory.id >= self.next_memory_id {
                            self.next_memory_id = memory.id + 1;
                        }
                        // Update stats counters
                        self.total_memories += 1;
                        match memory.tier {
                            crate::memory::MemoryTier::Episodic => self.episodic_count += 1,
                            crate::memory::MemoryTier::Semantic => self.semantic_count += 1,
                            crate::memory::MemoryTier::Schema => self.schema_count += 1,
                        }
                        self.sum_strength += memory.strength;
                        self.sum_access_count += memory.access_count as u64;
                        *self.agent_refcounts.entry(memory.agent_id).or_insert(0) += 1;
                        *self
                            .context_refcounts
                            .entry(memory.context.fingerprint)
                            .or_insert(0) += 1;
                    },
                    Err(_) => {
                        skipped_count += 1;
                    },
                }
                Ok(())
            });

        if let Err(ForEachError::Storage(e)) = scan_result {
            tracing::error!("Failed to scan memories during init: {:?}", e);
        }
        if skipped_count > 0 {
            tracing::warn!(
                "initialize: skipped {} corrupt memory records (likely v1 compact format with serde_json::Value fields)",
                skipped_count
            );
        }

        tracing::info!(
            "Initialized RedbMemoryStore: next_memory_id={}, total={}, episodic={}, semantic={}, schema={}",
            self.next_memory_id,
            self.total_memories,
            self.episodic_count,
            self.semantic_count,
            self.schema_count
        );
        Ok(())
    }

    /// Load memory from cache or redb. Cache hit auto-promotes to MRU.
    fn load_memory(&mut self, memory_id: MemoryId) -> Option<Memory> {
        // Check cache first — get() promotes to MRU automatically
        if let Some(memory) = self.memory_cache.get(&memory_id) {
            return Some(memory.clone());
        }

        // Cache miss - load from redb (versioned-aware)
        match self
            .backend
            .get_versioned::<_, Memory>("memory_records", memory_id.to_be_bytes())
        {
            Ok(Some(memory)) => {
                self.cache_memory(memory.clone());
                tracing::debug!("Loaded memory {} from redb into cache", memory_id);
                Some(memory)
            },
            Ok(None) => None,
            Err(e) => {
                tracing::error!("Failed to load memory {} from redb: {:?}", memory_id, e);
                None
            },
        }
    }

    /// Add memory to cache, flushing any dirty evicted entry.
    fn cache_memory(&mut self, memory: Memory) {
        let id = memory.id;
        // If cache is full and we're inserting a new key, check if LRU entry is dirty
        if !self.memory_cache.contains(&id)
            && self.memory_cache.len() == self.memory_cache.cap().get()
        {
            if let Some((&lru_id, _)) = self.memory_cache.peek_lru() {
                if self.dirty_ids.contains(&lru_id) {
                    // Pop and flush the dirty LRU entry before it's evicted
                    if let Some((evicted_id, evicted_mem)) = self.memory_cache.pop_lru() {
                        self.dirty_ids.remove(&evicted_id);
                        if let Ok(ops) = self.persist_memory_ops(&evicted_mem) {
                            if let Err(e) = self.backend.write_batch(ops) {
                                tracing::error!(
                                    "Failed to persist evicted dirty memory {}: {:?}",
                                    evicted_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
        self.memory_cache.put(id, memory);
    }

    /// Increment stats counters when a new memory is stored.
    fn stats_increment(&mut self, memory: &Memory) {
        self.total_memories += 1;
        match memory.tier {
            crate::memory::MemoryTier::Episodic => self.episodic_count += 1,
            crate::memory::MemoryTier::Semantic => self.semantic_count += 1,
            crate::memory::MemoryTier::Schema => self.schema_count += 1,
        }
        self.sum_strength += memory.strength;
        self.sum_access_count += memory.access_count as u64;
        *self.agent_refcounts.entry(memory.agent_id).or_insert(0) += 1;
        *self
            .context_refcounts
            .entry(memory.context.fingerprint)
            .or_insert(0) += 1;
    }

    /// Decrement stats counters when a memory is deleted.
    fn stats_decrement(&mut self, memory: &Memory) {
        self.total_memories = self.total_memories.saturating_sub(1);
        match memory.tier {
            crate::memory::MemoryTier::Episodic => {
                self.episodic_count = self.episodic_count.saturating_sub(1)
            },
            crate::memory::MemoryTier::Semantic => {
                self.semantic_count = self.semantic_count.saturating_sub(1)
            },
            crate::memory::MemoryTier::Schema => {
                self.schema_count = self.schema_count.saturating_sub(1)
            },
        }
        self.sum_strength = (self.sum_strength - memory.strength).max(0.0);
        self.sum_access_count = self
            .sum_access_count
            .saturating_sub(memory.access_count as u64);
        if let Some(count) = self.agent_refcounts.get_mut(&memory.agent_id) {
            *count = count.saturating_sub(1);
        }
        if let Some(count) = self.context_refcounts.get_mut(&memory.context.fingerprint) {
            *count = count.saturating_sub(1);
        }
    }

    /// Persist a memory to redb
    fn persist_memory(&self, memory: &Memory) -> StorageResult<()> {
        let ops = self.persist_memory_ops(memory)?;
        self.backend.write_batch(ops)
    }

    /// Build the batch operations needed to persist a memory (record + indexes)
    /// without committing them. Callers can accumulate multiple sets of ops and
    /// flush once via `backend.write_batch()`.
    fn persist_memory_ops(&self, memory: &Memory) -> StorageResult<Vec<BatchOperation>> {
        let mut ops = Vec::with_capacity(4);

        // Main record (versioned envelope)
        let value = serialize_versioned(memory)?;
        ops.push(BatchOperation::Put {
            table_name: "memory_records".to_string(),
            key: memory.id.to_be_bytes().to_vec(),
            value,
        });

        // Secondary indexes
        ops.extend(build_memory_index_ops(memory)?);

        Ok(ops)
    }

    /// Delete a memory from redb
    #[allow(dead_code)]
    fn delete_memory(&self, memory: &Memory) -> StorageResult<()> {
        let ops = self.delete_memory_ops(memory);
        self.backend.write_batch(ops)
    }

    /// Build batch operations to delete a memory (record + all indexes).
    fn delete_memory_ops(&self, memory: &Memory) -> Vec<BatchOperation> {
        let mut ops = Vec::with_capacity(4);

        // Delete main record
        ops.push(BatchOperation::Delete {
            table_name: "memory_records".to_string(),
            key: memory.id.to_be_bytes().to_vec(),
        });

        // Delete context index
        let mut context_key = Vec::with_capacity(16);
        context_key.extend_from_slice(&memory.context.fingerprint.to_be_bytes());
        context_key.extend_from_slice(&memory.id.to_be_bytes());
        ops.push(BatchOperation::Delete {
            table_name: "mem_by_context_hash".to_string(),
            key: context_key,
        });

        // Delete agent index
        let mut agent_key = Vec::with_capacity(16);
        agent_key.extend_from_slice(&memory.agent_id.to_be_bytes());
        agent_key.extend_from_slice(&memory.id.to_be_bytes());
        ops.push(BatchOperation::Delete {
            table_name: "mem_by_bucket".to_string(),
            key: agent_key,
        });

        // Delete goal bucket index
        let goal_bucket = memory.context.goal_bucket_id;
        if goal_bucket != 0 {
            let mut gb_key = Vec::with_capacity(16);
            gb_key.extend_from_slice(&goal_bucket.to_be_bytes());
            gb_key.extend_from_slice(&memory.id.to_be_bytes());
            ops.push(BatchOperation::Delete {
                table_name: "mem_by_goal_bucket".to_string(),
                key: gb_key,
            });
        }

        ops
    }
    /// Load memories belonging to a specific goal bucket.
    /// Scans the mem_by_goal_bucket index and loads each memory via cache.
    fn get_memories_for_goal_bucket(&mut self, goal_bucket_id: u64, limit: usize) -> Vec<Memory> {
        let prefix = goal_bucket_id.to_be_bytes();
        let results: Vec<(Vec<u8>, ())> =
            match self.backend.scan_prefix("mem_by_goal_bucket", prefix) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Failed to scan goal bucket index: {:?}", e);
                    return Vec::new();
                },
            };

        let mut memories = Vec::new();
        for (key, _) in results {
            if key.len() >= 16 {
                let memory_id = u64::from_be_bytes(key[8..16].try_into().unwrap());
                if let Some(memory) = self.load_memory(memory_id) {
                    memories.push(memory);
                    if memories.len() >= limit {
                        break;
                    }
                }
            }
        }
        memories
    }
}

impl MemoryStore for RedbMemoryStore {
    fn store_episode(&mut self, episode: &Episode, events: &[Event]) -> Option<MemoryUpsert> {
        use crate::episodes::EpisodeOutcome;
        use crate::memory::synthesize_memory_summary;
        use agent_db_core::types::current_timestamp;

        // Check significance threshold
        if episode.significance < self.config.min_significance {
            return None;
        }

        // Check if episode has ended
        episode.end_timestamp?;

        // Allocate new memory ID and persist the updated counter atomically
        let memory_id = self.next_memory_id;
        self.next_memory_id += 1;

        // Determine memory type based on outcome
        let memory_type = match episode.outcome {
            Some(EpisodeOutcome::Failure) => {
                let failure_pattern =
                    format!("Failed episode {} - avoid similar context", episode.id);
                MemoryType::Negative {
                    failure_severity: episode.significance,
                    failure_pattern,
                }
            },
            _ => MemoryType::Episodic {
                significance: episode.significance,
            },
        };

        // Create memory
        let current_time = current_timestamp();
        let prediction_weighted_strength =
            self.config.initial_strength * (1.0 + episode.prediction_error);

        let mut context = episode.context.clone();
        if context.fingerprint == 0 {
            context.fingerprint = context.compute_fingerprint();
        }
        if context.goal_bucket_id == 0 {
            context.goal_bucket_id = context.compute_goal_bucket_id();
        }

        // Generate natural language summary + causal analysis
        let summary = synthesize_memory_summary(episode, events);
        let causal_note = crate::memory::synthesize_causal_note(episode, events);
        let takeaway = crate::memory::synthesize_takeaway(episode, events);

        let memory = Memory {
            id: memory_id,
            agent_id: episode.agent_id,
            session_id: episode.session_id,
            episode_id: episode.id,
            summary,
            takeaway,
            causal_note,
            summary_embedding: Vec::new(),
            tier: crate::memory::MemoryTier::Episodic,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: crate::memory::ConsolidationStatus::Active,
            context,
            key_events: episode.events.clone(),
            strength: prediction_weighted_strength.min(self.config.max_strength),
            relevance_score: episode.significance * (1.0 + episode.prediction_error * 0.5),
            formed_at: current_time,
            last_accessed: current_time,
            access_count: 0,
            outcome: episode
                .outcome
                .clone()
                .unwrap_or(EpisodeOutcome::Interrupted),
            memory_type,
            metadata: std::collections::HashMap::new(),
        };

        // Persist to redb (record + indexes + updated ID allocator in one batch)
        match self.persist_memory_ops(&memory) {
            Ok(mut ops) => {
                // Persist the updated ID allocator atomically
                ops.push(BatchOperation::Put {
                    table_name: "id_allocator".to_string(),
                    key: b"next_memory_id".to_vec(),
                    value: self.next_memory_id.to_be_bytes().to_vec(),
                });
                if let Err(e) = self.backend.write_batch(ops) {
                    tracing::error!("Failed to persist memory {}: {:?}", memory_id, e);
                    return None;
                }
            },
            Err(e) => {
                tracing::error!(
                    "Failed to build persist ops for memory {}: {:?}",
                    memory_id,
                    e
                );
                return None;
            },
        }

        // Update stats and cache
        self.stats_increment(&memory);
        self.cache_memory(memory);

        Some(MemoryUpsert {
            id: memory_id,
            is_new: true,
        })
    }

    fn get_memory(&self, memory_id: MemoryId) -> Option<Memory> {
        // Check cache (peek doesn't promote but avoids needing &mut self)
        if let Some(memory) = self.memory_cache.peek(&memory_id) {
            return Some(memory.clone());
        }

        // Load from redb (versioned-aware)
        match self
            .backend
            .get_versioned::<_, Memory>("memory_records", memory_id.to_be_bytes())
        {
            Ok(memory) => memory,
            Err(e) => {
                tracing::error!("Failed to load memory {} from storage: {:?}", memory_id, e);
                None
            },
        }
    }

    fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory> {
        // Scan agent index
        let agent_prefix = agent_id.to_be_bytes();
        let results: Vec<(Vec<u8>, ())> =
            match self.backend.scan_prefix("mem_by_bucket", agent_prefix) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Failed to scan agent memories: {:?}", e);
                    return Vec::new();
                },
            };

        // Extract memory IDs from keys
        let mut memories = Vec::new();
        for (key, _) in results {
            if key.len() >= 16 {
                let memory_id = u64::from_be_bytes(key[8..16].try_into().unwrap());
                if let Some(memory) = self.get_memory(memory_id) {
                    memories.push(memory);
                    if memories.len() >= limit {
                        break;
                    }
                }
            }
        }

        memories
    }

    fn retrieve_by_context(&mut self, context: &EventContext, limit: usize) -> Vec<Memory> {
        // Scan context index
        let context_hash = if context.fingerprint == 0 {
            context.compute_fingerprint()
        } else {
            context.fingerprint
        };

        let context_prefix = context_hash.to_be_bytes();
        let results: Vec<(Vec<u8>, ())> = match self
            .backend
            .scan_prefix("mem_by_context_hash", context_prefix)
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to scan context memories: {:?}", e);
                return Vec::new();
            },
        };

        // Extract memory IDs and load memories
        let mut memories = Vec::new();
        for (key, _) in results {
            if key.len() >= 16 {
                let memory_id = u64::from_be_bytes(key[8..16].try_into().unwrap());
                if let Some(memory) = self.load_memory(memory_id) {
                    memories.push(memory);
                    if memories.len() >= limit {
                        break;
                    }
                }
            }
        }

        memories
    }

    fn retrieve_by_context_similar(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        _session_id: Option<SessionId>,
    ) -> Vec<Memory> {
        // For now, just retrieve by exact context and filter
        // TODO: Implement proper similarity search using embeddings
        let mut candidates = self.retrieve_by_context(context, limit * 2);

        // Filter by agent if specified
        if let Some(aid) = agent_id {
            candidates.retain(|m| m.agent_id == aid);
        }

        // Filter by similarity threshold using embeddings when available
        candidates.retain(|m| {
            let similarity = if m.context.fingerprint == context.fingerprint {
                1.0
            } else {
                match (
                    context.embeddings.as_deref(),
                    m.context.embeddings.as_deref(),
                ) {
                    (Some(q), Some(e)) if !q.is_empty() && !e.is_empty() => {
                        agent_db_core::utils::cosine_similarity(q, e)
                    },
                    _ => 0.0,
                }
            };
            similarity >= min_similarity
        });

        candidates.truncate(limit);
        candidates
    }

    fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool {
        use crate::memory::{
            memory_outcome_score, META_NEGATIVE_OUTCOMES, META_POSITIVE_OUTCOMES, META_Q_VALUE,
            Q_ALPHA,
        };

        // Load memory
        let mut memory = match self.load_memory(memory_id) {
            Some(m) => m,
            None => return false,
        };

        memory.access_count += 1;
        memory.last_accessed = agent_db_core::types::current_timestamp();

        // Update lifetime counters in metadata (lossless)
        let counter_key = if success {
            META_POSITIVE_OUTCOMES
        } else {
            META_NEGATIVE_OUTCOMES
        };
        let current_count: u32 = memory
            .metadata
            .get(counter_key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        memory
            .metadata
            .insert(counter_key.to_string(), (current_count + 1).to_string());

        // Update EMA Q-value: Q = Q + alpha(r - Q)
        let r = if success { 1.0_f32 } else { 0.0 };
        let q_old: f32 = memory
            .metadata
            .get(META_Q_VALUE)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let q_new = q_old + Q_ALPHA * (r - q_old);
        memory
            .metadata
            .insert(META_Q_VALUE.to_string(), format!("{:.6}", q_new));

        // Compute piecewise score
        let piecewise_score = memory_outcome_score(&memory);

        // Blend: preserve formation quality while incorporating outcome feedback
        memory.strength =
            (memory.strength * 0.5 + piecewise_score * 0.5).min(self.config.max_strength);

        // Adjust relevance score
        if success {
            memory.relevance_score = (memory.relevance_score + 0.02).min(1.0);
        } else {
            memory.relevance_score = (memory.relevance_score - 0.02).max(0.0);
        }

        // Persist updated memory
        if let Err(e) = self.persist_memory(&memory) {
            tracing::error!(
                "Failed to persist memory {} after outcome: {:?}",
                memory_id,
                e
            );
            return false;
        }

        // Update cache
        self.cache_memory(memory);

        true
    }

    fn get_stats(&self) -> MemoryStats {
        // Use cached counters (O(1) instead of full scan)
        let total = self.total_memories;
        if total == 0 {
            return MemoryStats {
                total_memories: 0,
                avg_strength: 0.0,
                avg_access_count: 0,
                agents_with_memories: 0,
                unique_contexts: 0,
                episodic_count: 0,
                semantic_count: 0,
                schema_count: 0,
            };
        }

        MemoryStats {
            total_memories: total,
            avg_strength: self.sum_strength / total as f32,
            avg_access_count: (self.sum_access_count / total as u64) as u32,
            agents_with_memories: self.agent_refcounts.values().filter(|&&c| c > 0).count(),
            unique_contexts: self.context_refcounts.values().filter(|&&c| c > 0).count(),
            episodic_count: self.episodic_count,
            semantic_count: self.semantic_count,
            schema_count: self.schema_count,
        }
    }

    fn apply_decay(&mut self) {
        use agent_db_core::types::current_timestamp;

        let current_time = current_timestamp();
        let hour_in_ns = 3_600_000_000_000u64;
        let decay_rate = self.config.decay_rate_per_hour;
        let forget_threshold = self.config.forget_threshold;

        // Stage 1: Streaming read pass over ALL persisted records (cold + hot)
        // Collect batch operations without holding a read txn during writes.
        let mut pending_ops: Vec<BatchOperation> = Vec::new();
        let mut op_batches: Vec<Vec<BatchOperation>> = Vec::new();
        let mut forgotten_ids: Vec<MemoryId> = Vec::new();
        let mut forgotten_metas: Vec<(MemoryId, crate::memory::MemoryTier, f32, u64, u64)> =
            Vec::new(); // (id, tier, strength, agent_id, fingerprint)
        let mut modified_cache_updates: Vec<(MemoryId, f32)> = Vec::new(); // (id, new_strength)
        let mut skipped_count = 0usize;

        let scan_result: Result<(), ForEachError<std::convert::Infallible>> = self
            .backend
            .for_each_prefix_raw("memory_records", vec![], |_key, value| {
                let memory: Memory = match deserialize_versioned(value) {
                    Ok(m) => m,
                    Err(_) => {
                        skipped_count += 1;
                        return Ok(());
                    },
                };

                let time_elapsed_ns = current_time.saturating_sub(memory.last_accessed);
                let hours_elapsed = (time_elapsed_ns / hour_in_ns) as f32;
                let decay_amount = decay_rate * hours_elapsed;
                let new_strength = (memory.strength - decay_amount).max(0.0);

                if new_strength < forget_threshold {
                    // Memory should be forgotten — build delete ops
                    pending_ops.extend(self.delete_memory_ops(&memory));
                    forgotten_ids.push(memory.id);
                    forgotten_metas.push((
                        memory.id,
                        memory.tier,
                        memory.strength,
                        memory.agent_id,
                        memory.context.fingerprint,
                    ));
                } else if (memory.strength - new_strength).abs() > f32::EPSILON {
                    // Memory decayed but survived — build update ops
                    let mut updated = memory.clone();
                    updated.strength = new_strength;
                    match self.persist_memory_ops(&updated) {
                        Ok(mem_ops) => pending_ops.extend(mem_ops),
                        Err(e) => {
                            tracing::error!(
                                "apply_decay: failed to build ops for memory {}: {:?}",
                                memory.id,
                                e
                            );
                        },
                    }
                    modified_cache_updates.push((memory.id, new_strength));
                }

                // Batch in chunks of 5000 to avoid unbounded Vec growth
                if pending_ops.len() >= 5_000 {
                    op_batches.push(std::mem::take(&mut pending_ops));
                }

                Ok(())
            });

        if let Err(ForEachError::Storage(e)) = scan_result {
            tracing::error!("apply_decay: streaming scan failed: {:?}", e);
            return;
        }
        if skipped_count > 0 {
            tracing::warn!(
                "apply_decay: skipped {} corrupt memory records",
                skipped_count
            );
        }
        // Final partial batch
        if !pending_ops.is_empty() {
            op_batches.push(pending_ops);
        }

        // Stage 2: Write pass — commit all batches
        for batch in &op_batches {
            if !batch.is_empty() {
                if let Err(e) = self.backend.write_batch(batch.clone()) {
                    tracing::error!("apply_decay: failed to write batch: {:?}", e);
                }
            }
        }

        // Update cache for modified entries (if present)
        for (id, new_strength) in &modified_cache_updates {
            if let Some(cached) = self.memory_cache.peek_mut(id) {
                let old_strength = cached.strength;
                cached.strength = *new_strength;
                self.sum_strength += new_strength - old_strength;
                self.dirty_ids.remove(id); // Just persisted
            }
        }

        // Remove forgotten entries from cache + dirty set + update stats
        for (id, tier, strength, agent_id, fingerprint) in &forgotten_metas {
            self.memory_cache.pop(id);
            self.dirty_ids.remove(id);
            // Decrement stats
            self.total_memories = self.total_memories.saturating_sub(1);
            match tier {
                crate::memory::MemoryTier::Episodic => {
                    self.episodic_count = self.episodic_count.saturating_sub(1)
                },
                crate::memory::MemoryTier::Semantic => {
                    self.semantic_count = self.semantic_count.saturating_sub(1)
                },
                crate::memory::MemoryTier::Schema => {
                    self.schema_count = self.schema_count.saturating_sub(1)
                },
            }
            self.sum_strength = (self.sum_strength - strength).max(0.0);
            if let Some(count) = self.agent_refcounts.get_mut(agent_id) {
                *count = count.saturating_sub(1);
            }
            if let Some(count) = self.context_refcounts.get_mut(fingerprint) {
                *count = count.saturating_sub(1);
            }
        }

        if !forgotten_ids.is_empty() {
            tracing::info!(
                "apply_decay: forgot {} memories below threshold, updated {} surviving",
                forgotten_ids.len(),
                modified_cache_updates.len()
            );
        }
    }

    fn list_all_memories(&self) -> Vec<Memory> {
        // Streaming scan: collect IDs of cached entries for override
        let cached_ids: HashSet<MemoryId> = self.memory_cache.iter().map(|(&id, _)| id).collect();

        let mut all: Vec<Memory> = Vec::new();
        let mut skipped_count = 0usize;

        // Stream from redb, skip entries that are in cache (cache is fresher)
        let scan_result: Result<(), ForEachError<std::convert::Infallible>> = self
            .backend
            .for_each_prefix_raw("memory_records", vec![], |_key, value| {
                match deserialize_versioned::<Memory>(value) {
                    Ok(memory) => {
                        if !cached_ids.contains(&memory.id) {
                            all.push(memory);
                        }
                    },
                    Err(_) => {
                        skipped_count += 1;
                    },
                }
                Ok(())
            });

        if let Err(ForEachError::Storage(e)) = scan_result {
            tracing::error!("Failed to stream memories from redb: {:?}", e);
        }
        if skipped_count > 0 {
            tracing::warn!(
                "list_all_memories: skipped {} corrupt memory records",
                skipped_count
            );
        }

        // Add cached versions (fresher)
        for (_, memory) in self.memory_cache.iter() {
            all.push(memory.clone());
        }

        all
    }

    fn store_consolidated_memory(&mut self, memory: Memory) {
        let id = memory.id;
        if id >= self.next_memory_id {
            self.next_memory_id = id + 1;
        }
        // Persist
        if let Err(e) = self.persist_memory(&memory) {
            tracing::error!("Failed to persist consolidated memory {}: {:?}", id, e);
        }
        // Update stats + cache
        self.stats_increment(&memory);
        self.cache_memory(memory);
    }

    fn mark_consolidated(
        &mut self,
        memory_id: MemoryId,
        into_id: MemoryId,
        into_tier: crate::memory::MemoryTier,
        decay: f32,
    ) {
        // Update in cache or load from redb
        if let Some(memory) = self.memory_cache.get_mut(&memory_id) {
            let old_strength = memory.strength;
            memory.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
            if into_tier == crate::memory::MemoryTier::Schema {
                memory.schema_id = Some(into_id);
            }
            memory.strength *= decay;
            self.sum_strength += memory.strength - old_strength;
            let mem_clone = memory.clone();
            let _ = self.persist_memory(&mem_clone);
        } else if let Some(mut memory) = self.load_memory(memory_id) {
            let old_strength = memory.strength;
            memory.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
            if into_tier == crate::memory::MemoryTier::Schema {
                memory.schema_id = Some(into_id);
            }
            memory.strength *= decay;
            self.sum_strength += memory.strength - old_strength;
            let _ = self.persist_memory(&memory);
            self.cache_memory(memory);
        }
    }

    fn retrieve_hierarchical(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
    ) -> Vec<Memory> {
        const MAX_CANDIDATES: usize = 500;

        let query_fp = if context.fingerprint != 0 {
            context.fingerprint
        } else {
            context.compute_fingerprint()
        };
        let query_bucket = if context.goal_bucket_id != 0 {
            context.goal_bucket_id
        } else {
            context.compute_goal_bucket_id()
        };
        let query_emb = context.embeddings.as_deref().unwrap_or(&[]);

        // Use goal bucket index for a targeted scan when goals are available.
        // Fall back to full scan only when no goal bucket exists.
        let candidates: Vec<Memory> = if query_bucket != 0 {
            self.get_memories_for_goal_bucket(query_bucket, MAX_CANDIDATES)
        } else {
            self.list_all_memories()
        };

        let mut scored: Vec<(f32, Memory)> = candidates
            .into_iter()
            .filter(|m| m.consolidation_status != crate::memory::ConsolidationStatus::Archived)
            .filter(|m| agent_id.is_none_or(|aid| m.agent_id == aid))
            .filter_map(|m| {
                // Memories in the same goal bucket get a baseline similarity of 0.5
                // since they share the same goal set even if fingerprints differ.
                let bucket_sim = if query_bucket != 0 && m.context.goal_bucket_id == query_bucket {
                    0.5f32
                } else {
                    0.0
                };
                let fp_sim = if m.context.fingerprint == query_fp {
                    1.0f32
                } else {
                    0.0
                };
                let emb_sim = if !query_emb.is_empty() {
                    let m_emb = m.context.embeddings.as_deref().unwrap_or(&[]);
                    if !m_emb.is_empty() {
                        agent_db_core::utils::cosine_similarity(query_emb, m_emb)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let sim = fp_sim.max(emb_sim).max(bucket_sim);
                if sim >= min_similarity {
                    let tier_boost = match m.tier {
                        crate::memory::MemoryTier::Schema => 0.3,
                        crate::memory::MemoryTier::Semantic => 0.15,
                        crate::memory::MemoryTier::Episodic => 0.0,
                    };
                    Some((sim + tier_boost, m))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(_, m)| m).collect()
    }

    fn flush_cache(&mut self) {
        if self.dirty_ids.is_empty() {
            return;
        }

        let mut all_ops: Vec<BatchOperation> = Vec::new();
        let mut flushed = 0usize;

        for &id in &self.dirty_ids {
            if let Some(memory) = self.memory_cache.peek(&id) {
                match self.persist_memory_ops(memory) {
                    Ok(ops) => {
                        all_ops.extend(ops);
                        flushed += 1;
                    },
                    Err(e) => {
                        tracing::error!(
                            "flush_cache: failed to build ops for memory {}: {:?}",
                            id,
                            e
                        );
                    },
                }
            }
        }

        if !all_ops.is_empty() {
            if let Err(e) = self.backend.write_batch(all_ops) {
                tracing::error!("flush_cache: failed to batch-persist memories: {:?}", e);
            } else {
                tracing::info!("flush_cache: persisted {} dirty memories", flushed);
                self.dirty_ids.clear();
            }
        }
    }

    fn reinitialize(&mut self) {
        // Clear the LRU cache and dirty set so reads go back to redb
        self.memory_cache.clear();
        self.dirty_ids.clear();
        // Re-initialize (scans redb, rebuilds stats, loads persisted ID)
        self.next_memory_id = 1;
        if let Err(e) = self.initialize() {
            tracing::error!("RedbMemoryStore reinitialize failed: {:?}", e);
        }
        tracing::info!(
            "RedbMemoryStore reinitialize: next_memory_id={}",
            self.next_memory_id
        );
    }

    fn store_consolidated_memories_batch(&mut self, memories: Vec<Memory>) {
        let mut all_ops: Vec<BatchOperation> = Vec::new();

        for memory in &memories {
            let id = memory.id;
            if id >= self.next_memory_id {
                self.next_memory_id = id + 1;
            }
            match self.persist_memory_ops(memory) {
                Ok(ops) => all_ops.extend(ops),
                Err(e) => {
                    tracing::error!(
                        "Failed to build ops for consolidated memory {}: {:?}",
                        id,
                        e
                    );
                },
            }
        }

        // Single atomic write for all memories
        if !all_ops.is_empty() {
            if let Err(e) = self.backend.write_batch(all_ops) {
                tracing::error!("Failed to batch-persist consolidated memories: {:?}", e);
            }
        }

        // Cache and update stats after successful persist
        for memory in memories {
            self.stats_increment(&memory);
            self.cache_memory(memory);
        }
    }

    fn mark_consolidated_batch(
        &mut self,
        batch: Vec<(MemoryId, MemoryId, crate::memory::MemoryTier, f32)>,
    ) {
        let mut all_ops: Vec<BatchOperation> = Vec::new();
        let mut updated_memories: Vec<Memory> = Vec::new();

        for (memory_id, into_id, into_tier, decay) in batch {
            let memory = if let Some(mem) = self.memory_cache.get_mut(&memory_id) {
                let old_strength = mem.strength;
                mem.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
                if into_tier == crate::memory::MemoryTier::Schema {
                    mem.schema_id = Some(into_id);
                }
                mem.strength *= decay;
                self.sum_strength += mem.strength - old_strength;
                Some(mem.clone())
            } else {
                self.load_memory(memory_id).map(|mut m| {
                    let old_strength = m.strength;
                    m.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
                    if into_tier == crate::memory::MemoryTier::Schema {
                        m.schema_id = Some(into_id);
                    }
                    m.strength *= decay;
                    self.sum_strength += m.strength - old_strength;
                    m
                })
            };

            if let Some(mem) = memory {
                match self.persist_memory_ops(&mem) {
                    Ok(ops) => all_ops.extend(ops),
                    Err(e) => {
                        tracing::error!(
                            "Failed to build ops for mark_consolidated {}: {:?}",
                            memory_id,
                            e
                        );
                    },
                }
                updated_memories.push(mem);
            }
        }

        // Single atomic write
        if !all_ops.is_empty() {
            if let Err(e) = self.backend.write_batch(all_ops) {
                tracing::error!("Failed to batch-persist mark_consolidated: {:?}", e);
            }
        }

        // Cache updated memories
        for mem in updated_memories {
            self.cache_memory(mem);
        }
    }

    fn delete_memories_batch(&mut self, ids: Vec<MemoryId>) -> usize {
        let mut all_ops: Vec<BatchOperation> = Vec::new();
        let mut deleted_metas: Vec<Memory> = Vec::new();

        for id in &ids {
            // Try cache first, then redb
            let memory = if let Some(mem) = self.memory_cache.peek(id) {
                Some(mem.clone())
            } else {
                match self
                    .backend
                    .get_versioned::<_, Memory>("memory_records", id.to_be_bytes())
                {
                    Ok(mem) => mem,
                    Err(e) => {
                        tracing::warn!("delete_memories_batch: failed to load {}: {:?}", id, e);
                        None
                    },
                }
            };

            if let Some(memory) = memory {
                all_ops.extend(self.delete_memory_ops(&memory));
                deleted_metas.push(memory);
            }
        }

        let deleted_count = deleted_metas.len();

        // Single atomic write
        if !all_ops.is_empty() {
            if let Err(e) = self.backend.write_batch(all_ops) {
                tracing::error!("delete_memories_batch: failed to write: {:?}", e);
                return 0;
            }
        }

        // Remove from cache + dirty set + update stats
        for memory in &deleted_metas {
            self.memory_cache.pop(&memory.id);
            self.dirty_ids.remove(&memory.id);
            self.stats_decrement(memory);
        }

        if deleted_count > 0 {
            tracing::info!("delete_memories_batch: deleted {} memories", deleted_count);
        }
        deleted_count
    }
}

pub trait StrategyStore: Send + Sync {
    fn store_episode(
        &mut self,
        episode: &Episode,
        events: &[agent_db_events::core::Event],
    ) -> crate::GraphResult<Option<StrategyUpsert>>;
    fn get_strategy(&self, strategy_id: StrategyId) -> Option<Strategy>;
    fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy>;
    fn get_strategies_for_context(&self, context_hash: ContextHash, limit: usize) -> Vec<Strategy>;
    fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)>;
    fn update_strategy_outcome(
        &mut self,
        strategy_id: StrategyId,
        success: bool,
    ) -> crate::GraphResult<()>;
    fn get_stats(&self) -> StrategyStats;

    /// Update a strategy in the store (e.g. after LLM refinement).
    fn update_strategy(&mut self, strategy: Strategy) -> crate::GraphResult<()>;

    // ========== Pruning API ==========
    /// Remove weak / stale strategies and merge near-duplicates.
    /// Returns the total number of strategies removed.
    fn prune_strategies(
        &mut self,
        min_confidence: f32,
        min_support: u32,
        max_stale_hours: f32,
    ) -> usize;

    /// List all strategies (for pruning / analytics).
    fn list_all_strategies(&self) -> Vec<Strategy>;

    /// Flush all dirty cache entries to persistent storage.
    /// No-op for in-memory stores; redb stores batch-persist the cache.
    fn flush_cache(&mut self) {}

    /// Re-scan persistent storage to reset the ID allocator and clear caches.
    /// Called after import to pick up externally-written records.
    /// No-op for in-memory stores.
    fn reinitialize(&mut self) {}
}

pub struct InMemoryStrategyStore {
    inner: StrategyExtractor,
}

impl InMemoryStrategyStore {
    pub fn new(config: StrategyExtractionConfig) -> Self {
        Self {
            inner: StrategyExtractor::new(config),
        }
    }
}

impl StrategyStore for InMemoryStrategyStore {
    fn store_episode(
        &mut self,
        episode: &Episode,
        events: &[agent_db_events::core::Event],
    ) -> crate::GraphResult<Option<StrategyUpsert>> {
        self.inner.extract_from_episode(episode, events)
    }

    fn get_strategy(&self, strategy_id: StrategyId) -> Option<Strategy> {
        self.inner.get_strategy(strategy_id).cloned()
    }

    fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy> {
        self.inner
            .get_agent_strategies(agent_id, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    fn get_strategies_for_context(&self, context_hash: ContextHash, limit: usize) -> Vec<Strategy> {
        self.inner
            .get_strategies_for_context(context_hash, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)> {
        self.inner.find_similar_strategies(query)
    }

    fn update_strategy_outcome(
        &mut self,
        strategy_id: StrategyId,
        success: bool,
    ) -> crate::GraphResult<()> {
        self.inner.update_strategy_outcome(strategy_id, success)
    }

    fn get_stats(&self) -> StrategyStats {
        self.inner.get_stats()
    }

    fn update_strategy(&mut self, strategy: Strategy) -> crate::GraphResult<()> {
        self.inner.insert_loaded_strategy(strategy)
    }

    fn prune_strategies(
        &mut self,
        min_confidence: f32,
        min_support: u32,
        max_stale_hours: f32,
    ) -> usize {
        let pruned = self
            .inner
            .prune_weak_strategies(min_confidence, min_support, max_stale_hours);
        let merged = self.inner.merge_similar_strategies();
        pruned.len() + merged
    }

    fn list_all_strategies(&self) -> Vec<Strategy> {
        self.inner
            .list_all_strategies()
            .into_iter()
            .cloned()
            .collect()
    }
}

/// Redb-backed strategy store with LRU cache for scalability
///
/// **Architecture**: Hot/Cold separation
/// - Hot: Recently accessed strategies in LRU cache (bounded size)
/// - Cold: All strategies persisted in redb (unbounded)
///
/// Uses the following redb tables:
/// - strategy_records: strategy_id → Strategy  (main storage)
/// - strategy_by_bucket: (goal_bucket_id, strategy_id) → empty  (index)
/// - strategy_by_signature: (behavior_signature, strategy_id) → empty  (index)
/// - strategy_feature_postings: (feature, strategy_id) → quality_score  (similarity search)
pub struct RedbStrategyStore {
    backend: Arc<RedbBackend>,
    #[allow(dead_code)]
    config: StrategyExtractionConfig,
    /// LRU cache for hot strategies (bounded size, O(1) eviction)
    strategy_cache: lru::LruCache<StrategyId, Strategy>,
    /// IDs of cache entries whose mutations have not yet been flushed to redb
    dirty_strategy_ids: HashSet<StrategyId>,
    /// Next strategy ID allocator
    next_strategy_id: StrategyId,
    /// Strategy extractor for creating new strategies (stateless helper)
    strategy_extractor: StrategyExtractor,
}

impl RedbStrategyStore {
    /// Create a new redb strategy store with LRU cache
    ///
    /// # Arguments
    /// * `backend` - Redb backend for persistence
    /// * `config` - Strategy extraction configuration
    /// * `max_cache_size` - Maximum number of strategies to keep in LRU cache
    pub fn new(
        backend: Arc<RedbBackend>,
        config: StrategyExtractionConfig,
        max_cache_size: usize,
    ) -> Self {
        let cap = NonZeroUsize::new(max_cache_size).unwrap_or(NonZeroUsize::new(1024).unwrap());
        Self {
            backend,
            config: config.clone(),
            strategy_cache: lru::LruCache::new(cap),
            dirty_strategy_ids: HashSet::new(),
            next_strategy_id: 1,
            strategy_extractor: StrategyExtractor::new(config),
        }
    }

    /// Initialize next_strategy_id by scanning existing strategies
    pub fn initialize(&mut self) -> StorageResult<()> {
        // Try to load persisted ID allocator first
        if let Ok(Some(id_bytes)) = self.backend.get_raw("id_allocator", b"next_strategy_id") {
            if id_bytes.len() >= 8 {
                self.next_strategy_id = u64::from_be_bytes(id_bytes[..8].try_into().unwrap());
            }
        }

        // Streaming scan to find max ID (in case persisted ID is stale)
        let mut skipped_count = 0usize;
        let scan_result: Result<(), ForEachError<std::convert::Infallible>> = self
            .backend
            .for_each_prefix_raw("strategy_records", vec![], |_key, value| {
                match deserialize_versioned::<Strategy>(value) {
                    Ok(strategy) => {
                        if strategy.id >= self.next_strategy_id {
                            self.next_strategy_id = strategy.id + 1;
                        }
                    },
                    Err(_) => {
                        skipped_count += 1;
                    },
                }
                Ok(())
            });

        if let Err(ForEachError::Storage(e)) = scan_result {
            tracing::error!("Failed to scan strategies during init: {:?}", e);
        }
        if skipped_count > 0 {
            tracing::warn!(
                "initialize: skipped {} corrupt strategy records",
                skipped_count
            );
        }

        tracing::info!(
            "Initialized RedbStrategyStore with next_strategy_id={}",
            self.next_strategy_id
        );
        Ok(())
    }

    /// Load strategy from cache or redb. Cache hit auto-promotes to MRU.
    fn load_strategy(&mut self, strategy_id: StrategyId) -> Option<Strategy> {
        // Check cache first — get() promotes to MRU automatically
        if let Some(strategy) = self.strategy_cache.get(&strategy_id) {
            return Some(strategy.clone());
        }

        // Cache miss - load from redb (versioned-aware)
        match self
            .backend
            .get_versioned::<_, Strategy>("strategy_records", strategy_id.to_be_bytes())
        {
            Ok(Some(strategy)) => {
                self.cache_strategy(strategy.clone());
                tracing::debug!("Loaded strategy {} from redb into cache", strategy_id);
                Some(strategy)
            },
            Ok(None) => None,
            Err(e) => {
                tracing::error!("Failed to load strategy {} from redb: {:?}", strategy_id, e);
                None
            },
        }
    }

    /// Add strategy to cache, flushing any dirty evicted entry.
    fn cache_strategy(&mut self, strategy: Strategy) {
        let id = strategy.id;
        // If cache is full and we're inserting a new key, check if LRU entry is dirty
        if !self.strategy_cache.contains(&id)
            && self.strategy_cache.len() == self.strategy_cache.cap().get()
        {
            if let Some((&lru_id, _)) = self.strategy_cache.peek_lru() {
                if self.dirty_strategy_ids.contains(&lru_id) {
                    if let Some((evicted_id, evicted_strat)) = self.strategy_cache.pop_lru() {
                        self.dirty_strategy_ids.remove(&evicted_id);
                        if let Ok(ops) = self.persist_strategy_ops(&evicted_strat) {
                            if let Err(e) = self.backend.write_batch(ops) {
                                tracing::error!(
                                    "Failed to persist evicted dirty strategy {}: {:?}",
                                    evicted_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
        self.strategy_cache.put(id, strategy);
    }

    /// Persist a strategy to redb
    fn persist_strategy(&self, strategy: &Strategy) -> StorageResult<()> {
        let ops = self.persist_strategy_ops(strategy)?;
        self.backend.write_batch(ops)
    }

    /// Build batch operations for persisting a strategy (record + indexes)
    /// without committing. Allows callers to accumulate ops and flush once.
    fn persist_strategy_ops(&self, strategy: &Strategy) -> StorageResult<Vec<BatchOperation>> {
        let mut ops = Vec::with_capacity(4);

        // Main record (versioned envelope)
        let value = serialize_versioned(strategy)?;
        ops.push(BatchOperation::Put {
            table_name: "strategy_records".to_string(),
            key: strategy.id.to_be_bytes().to_vec(),
            value,
        });

        // Secondary indexes
        ops.extend(build_strategy_index_ops(strategy)?);

        Ok(ops)
    }

    /// Delete a strategy from redb
    fn delete_strategy(&self, strategy: &Strategy) -> StorageResult<()> {
        // Delete main record
        self.backend
            .delete("strategy_records", strategy.id.to_be_bytes())?;

        // Delete bucket index
        let mut bucket_key = Vec::with_capacity(16);
        bucket_key.extend_from_slice(&strategy.goal_bucket_id.to_be_bytes());
        bucket_key.extend_from_slice(&strategy.id.to_be_bytes());
        self.backend.delete("strategy_by_bucket", bucket_key)?;

        // Delete signature index
        let signature_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            strategy.behavior_signature.hash(&mut hasher);
            hasher.finish()
        };
        let mut signature_key = Vec::with_capacity(16);
        signature_key.extend_from_slice(&signature_hash.to_be_bytes());
        signature_key.extend_from_slice(&strategy.id.to_be_bytes());
        self.backend
            .delete("strategy_by_signature", signature_key)?;

        // Delete agent index
        let mut agent_key = Vec::with_capacity(16);
        agent_key.extend_from_slice(&strategy.agent_id.to_be_bytes());
        agent_key.extend_from_slice(&strategy.id.to_be_bytes());
        self.backend
            .delete("strategy_feature_postings", agent_key)?;

        Ok(())
    }
}

impl StrategyStore for RedbStrategyStore {
    fn store_episode(
        &mut self,
        episode: &Episode,
        events: &[agent_db_events::core::Event],
    ) -> crate::GraphResult<Option<StrategyUpsert>> {
        // Extract strategy using extractor (stateless)
        let upsert = self
            .strategy_extractor
            .extract_from_episode(episode, events)?;

        // Persist to redb if a strategy was extracted
        if let Some(ref upsert_result) = upsert {
            if let Some(strategy) = self.strategy_extractor.get_strategy(upsert_result.id) {
                if let Err(e) = self.persist_strategy(strategy) {
                    tracing::error!("Failed to persist strategy {}: {:?}", upsert_result.id, e);
                    return Ok(None);
                }

                // Add to LRU cache
                self.cache_strategy(strategy.clone());
            }
        }

        Ok(upsert)
    }

    fn get_strategy(&self, strategy_id: StrategyId) -> Option<Strategy> {
        // Check cache (peek doesn't promote but avoids needing &mut self)
        if let Some(strategy) = self.strategy_cache.peek(&strategy_id) {
            return Some(strategy.clone());
        }

        // Fall back to persistent storage (versioned-aware)
        match self
            .backend
            .get_versioned::<_, Strategy>("strategy_records", strategy_id.to_be_bytes())
        {
            Ok(Some(strategy)) => Some(strategy),
            Ok(None) => None,
            Err(e) => {
                tracing::error!(
                    "Failed to load strategy {} from storage: {:?}",
                    strategy_id,
                    e
                );
                None
            },
        }
    }

    fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy> {
        // Scan agent index (using strategy_feature_postings table)
        let agent_prefix = agent_id.to_be_bytes();
        let results: Vec<(Vec<u8>, f32)> = match self
            .backend
            .scan_prefix("strategy_feature_postings", agent_prefix)
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to scan agent strategies: {:?}", e);
                return Vec::new();
            },
        };

        // Extract strategy IDs from keys
        let mut strategies = Vec::new();
        for (key, _quality) in results {
            if key.len() >= 16 {
                let strategy_id = u64::from_be_bytes(key[8..16].try_into().unwrap());
                if let Some(strategy) = self.get_strategy(strategy_id) {
                    strategies.push(strategy);
                    if strategies.len() >= limit {
                        break;
                    }
                }
            }
        }

        strategies
    }

    fn get_strategies_for_context(&self, context_hash: ContextHash, limit: usize) -> Vec<Strategy> {
        // Scan goal bucket index (using context_hash as proxy)
        let bucket_prefix = context_hash.to_be_bytes();
        let results: Vec<(Vec<u8>, ())> = match self
            .backend
            .scan_prefix("strategy_by_bucket", bucket_prefix)
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to scan context strategies: {:?}", e);
                return Vec::new();
            },
        };

        // Extract strategy IDs and load strategies
        let mut strategies = Vec::new();
        for (key, _) in results {
            if key.len() >= 16 {
                let strategy_id = u64::from_be_bytes(key[8..16].try_into().unwrap());
                if let Some(strategy) = self.get_strategy(strategy_id) {
                    strategies.push(strategy);
                    if strategies.len() >= limit {
                        break;
                    }
                }
            }
        }

        strategies
    }

    fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)> {
        // Use extractor's similarity logic (delegates to in-memory helper)
        self.strategy_extractor.find_similar_strategies(query)
    }

    fn update_strategy_outcome(
        &mut self,
        strategy_id: StrategyId,
        success: bool,
    ) -> crate::GraphResult<()> {
        // Load strategy
        let mut strategy = match self.load_strategy(strategy_id) {
            Some(s) => s,
            None => {
                return Err(crate::GraphError::InvalidOperation(format!(
                    "Strategy {} not found",
                    strategy_id
                )))
            },
        };

        // Update outcome
        if success {
            strategy.success_count += 1;
        } else {
            strategy.failure_count += 1;
        }

        // Recalculate quality score
        let total = strategy.success_count + strategy.failure_count;
        if total > 0 {
            strategy.quality_score = strategy.success_count as f32 / total as f32;
        }

        // Persist updated strategy
        if let Err(e) = self.persist_strategy(&strategy) {
            tracing::error!(
                "Failed to persist strategy {} after outcome: {:?}",
                strategy_id,
                e
            );
            return Err(crate::GraphError::OperationError(format!(
                "Failed to persist strategy: {:?}",
                e
            )));
        }

        // Update cache
        self.cache_strategy(strategy);

        Ok(())
    }

    fn get_stats(&self) -> StrategyStats {
        let strategies: Vec<(Vec<u8>, Strategy)> =
            match self.backend.scan_prefix_raw("strategy_records", vec![]) {
                Ok(raw) => raw
                    .into_iter()
                    .filter_map(|(k, v)| deserialize_versioned::<Strategy>(&v).ok().map(|s| (k, s)))
                    .collect(),
                Err(_) => {
                    return StrategyStats {
                        total_strategies: 0,
                        high_quality_strategies: 0,
                        agents_with_strategies: 0,
                        average_quality: 0.0,
                    };
                },
            };

        let total = strategies.len();
        let high_quality = strategies
            .iter()
            .filter(|(_, s)| s.quality_score > 0.8)
            .count();

        let avg_quality = if total > 0 {
            strategies.iter().map(|(_, s)| s.quality_score).sum::<f32>() / total as f32
        } else {
            0.0
        };

        let agents: std::collections::HashSet<u64> =
            strategies.iter().map(|(_, s)| s.agent_id).collect();

        StrategyStats {
            total_strategies: total,
            high_quality_strategies: high_quality,
            agents_with_strategies: agents.len(),
            average_quality: avg_quality,
        }
    }

    fn update_strategy(&mut self, strategy: Strategy) -> crate::GraphResult<()> {
        let id = strategy.id;
        // Persist to redb
        if let Err(e) = self.persist_strategy(&strategy) {
            tracing::error!("Failed to persist updated strategy {}: {:?}", id, e);
            return Err(crate::GraphError::OperationError(format!(
                "Failed to persist strategy: {:?}",
                e
            )));
        }
        // Update cache
        self.cache_strategy(strategy);
        Ok(())
    }

    fn prune_strategies(
        &mut self,
        min_confidence: f32,
        min_support: u32,
        max_stale_hours: f32,
    ) -> usize {
        // Prune from in-memory extractor
        let pruned_ids = self.strategy_extractor.prune_weak_strategies(
            min_confidence,
            min_support,
            max_stale_hours,
        );
        let merged = self.strategy_extractor.merge_similar_strategies();

        // Delete pruned strategies from redb + cache
        for id in &pruned_ids {
            self.strategy_cache.pop(id);
            self.dirty_strategy_ids.remove(id);
            // Build a temporary strategy to clean up indexes
            if let Ok(Some(strategy)) = self
                .backend
                .get_versioned::<_, Strategy>("strategy_records", id.to_be_bytes())
            {
                let _ = self.delete_strategy(&strategy);
            } else {
                let _ = self.backend.delete("strategy_records", id.to_be_bytes());
            }
        }

        pruned_ids.len() + merged
    }

    fn flush_cache(&mut self) {
        if self.dirty_strategy_ids.is_empty() {
            return;
        }

        let mut all_ops: Vec<BatchOperation> = Vec::new();
        let mut flushed = 0usize;

        for &id in &self.dirty_strategy_ids {
            if let Some(strategy) = self.strategy_cache.peek(&id) {
                match self.persist_strategy_ops(strategy) {
                    Ok(ops) => {
                        all_ops.extend(ops);
                        flushed += 1;
                    },
                    Err(e) => {
                        tracing::error!(
                            "flush_cache: failed to build ops for strategy {}: {:?}",
                            id,
                            e
                        );
                    },
                }
            }
        }

        if !all_ops.is_empty() {
            if let Err(e) = self.backend.write_batch(all_ops) {
                tracing::error!("flush_cache: failed to batch-persist strategies: {:?}", e);
            } else {
                tracing::info!("flush_cache: persisted {} dirty strategies", flushed);
                self.dirty_strategy_ids.clear();
            }
        }
    }

    fn reinitialize(&mut self) {
        self.strategy_cache.clear();
        self.dirty_strategy_ids.clear();
        self.next_strategy_id = 1;
        if let Err(e) = self.initialize() {
            tracing::error!("RedbStrategyStore reinitialize failed: {:?}", e);
        }
        tracing::info!(
            "RedbStrategyStore reinitialize: next_strategy_id={}",
            self.next_strategy_id
        );
    }

    fn list_all_strategies(&self) -> Vec<Strategy> {
        let cached_ids: HashSet<StrategyId> =
            self.strategy_cache.iter().map(|(&id, _)| id).collect();

        let mut all: Vec<Strategy> = Vec::new();
        let mut skipped_count = 0usize;

        // Stream from redb, skip cached entries
        let scan_result: Result<(), ForEachError<std::convert::Infallible>> = self
            .backend
            .for_each_prefix_raw("strategy_records", vec![], |_key, value| {
                match deserialize_versioned::<Strategy>(value) {
                    Ok(strategy) => {
                        if !cached_ids.contains(&strategy.id) {
                            all.push(strategy);
                        }
                    },
                    Err(_) => {
                        skipped_count += 1;
                    },
                }
                Ok(())
            });

        if let Err(ForEachError::Storage(e)) = scan_result {
            tracing::error!("Failed to stream strategies from redb: {:?}", e);
        }
        if skipped_count > 0 {
            tracing::warn!(
                "list_all_strategies: skipped {} corrupt strategy records",
                skipped_count
            );
        }

        // Add cached versions (fresher)
        for (_, strategy) in self.strategy_cache.iter() {
            all.push(strategy.clone());
        }

        all
    }
}

// ========== Public index-building helpers (used by persist + import) ==========

/// Build the secondary index batch operations for a memory (excluding the main record).
/// Used by both `RedbMemoryStore::persist_memory_ops()` and the export/import system.
pub fn build_memory_index_ops(memory: &Memory) -> StorageResult<Vec<BatchOperation>> {
    let mut ops = Vec::with_capacity(3);

    // Index by context fingerprint
    let mut context_key = Vec::with_capacity(16);
    context_key.extend_from_slice(&memory.context.fingerprint.to_be_bytes());
    context_key.extend_from_slice(&memory.id.to_be_bytes());
    ops.push(BatchOperation::Put {
        table_name: "mem_by_context_hash".to_string(),
        key: context_key,
        value: rmp_serde::to_vec(&())
            .map_err(|e| agent_db_storage::StorageError::Serialization(e.to_string()))?,
    });

    // Index by agent
    let mut agent_key = Vec::with_capacity(16);
    agent_key.extend_from_slice(&memory.agent_id.to_be_bytes());
    agent_key.extend_from_slice(&memory.id.to_be_bytes());
    ops.push(BatchOperation::Put {
        table_name: "mem_by_bucket".to_string(),
        key: agent_key,
        value: rmp_serde::to_vec(&())
            .map_err(|e| agent_db_storage::StorageError::Serialization(e.to_string()))?,
    });

    // Index by goal bucket
    let goal_bucket = memory.context.goal_bucket_id;
    if goal_bucket != 0 {
        let mut gb_key = Vec::with_capacity(16);
        gb_key.extend_from_slice(&goal_bucket.to_be_bytes());
        gb_key.extend_from_slice(&memory.id.to_be_bytes());
        ops.push(BatchOperation::Put {
            table_name: "mem_by_goal_bucket".to_string(),
            key: gb_key,
            value: rmp_serde::to_vec(&())
                .map_err(|e| agent_db_storage::StorageError::Serialization(e.to_string()))?,
        });
    }

    Ok(ops)
}

/// Build the secondary index batch operations for a strategy (excluding the main record).
/// Used by both `RedbStrategyStore::persist_strategy_ops()` and the export/import system.
pub fn build_strategy_index_ops(strategy: &Strategy) -> StorageResult<Vec<BatchOperation>> {
    let mut ops = Vec::with_capacity(3);

    // Index by goal bucket
    let mut bucket_key = Vec::with_capacity(16);
    bucket_key.extend_from_slice(&strategy.goal_bucket_id.to_be_bytes());
    bucket_key.extend_from_slice(&strategy.id.to_be_bytes());
    ops.push(BatchOperation::Put {
        table_name: "strategy_by_bucket".to_string(),
        key: bucket_key,
        value: rmp_serde::to_vec(&())
            .map_err(|e| agent_db_storage::StorageError::Serialization(e.to_string()))?,
    });

    // Index by behavior signature
    let signature_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        strategy.behavior_signature.hash(&mut hasher);
        hasher.finish()
    };
    let mut signature_key = Vec::with_capacity(16);
    signature_key.extend_from_slice(&signature_hash.to_be_bytes());
    signature_key.extend_from_slice(&strategy.id.to_be_bytes());
    ops.push(BatchOperation::Put {
        table_name: "strategy_by_signature".to_string(),
        key: signature_key,
        value: rmp_serde::to_vec(&())
            .map_err(|e| agent_db_storage::StorageError::Serialization(e.to_string()))?,
    });

    // Index by agent
    let mut agent_key = Vec::with_capacity(16);
    agent_key.extend_from_slice(&strategy.agent_id.to_be_bytes());
    agent_key.extend_from_slice(&strategy.id.to_be_bytes());
    let quality_bytes = rmp_serde::to_vec(&strategy.quality_score)
        .map_err(|e| agent_db_storage::StorageError::Serialization(e.to_string()))?;
    ops.push(BatchOperation::Put {
        table_name: "strategy_feature_postings".to_string(),
        key: agent_key,
        value: quality_bytes,
    });

    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeOutcome;
    use crate::memory::{
        memory_negative_outcomes, memory_outcome_score, memory_positive_outcomes,
        ConsolidationStatus, MemoryTier, MemoryType,
    };
    use agent_db_events::core::EventContext;

    /// Helper: create a minimal Memory for testing
    fn make_test_memory(id: MemoryId) -> Memory {
        Memory {
            id,
            agent_id: 1,
            session_id: 1,
            episode_id: 1,
            summary: "test memory".to_string(),
            takeaway: String::new(),
            causal_note: String::new(),
            summary_embedding: Vec::new(),
            tier: MemoryTier::Episodic,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: ConsolidationStatus::Active,
            context: EventContext::default(),
            key_events: Vec::new(),
            strength: 0.5,
            relevance_score: 0.5,
            formed_at: 1000,
            last_accessed: 1000,
            access_count: 0,
            outcome: EpisodeOutcome::Success,
            memory_type: MemoryType::Episodic { significance: 0.5 },
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Helper: create a RedbMemoryStore with a temp directory
    fn make_test_store(dir: &tempfile::TempDir) -> RedbMemoryStore {
        let backend = Arc::new(
            RedbBackend::open(agent_db_storage::RedbConfig {
                data_path: dir.path().join("test.redb"),
                cache_size_bytes: 4 * 1024 * 1024,
                repair_on_open: false,
            })
            .unwrap(),
        );
        let config = MemoryFormationConfig::default();
        let mut store = RedbMemoryStore::new(backend, config, 128);
        store.initialize().unwrap();
        store
    }

    #[test]
    fn test_redb_memory_piecewise_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = make_test_store(&dir);

        // Store a memory via store_consolidated_memory
        let mem = make_test_memory(1);
        let initial_strength = mem.strength;
        store.store_consolidated_memory(mem);

        // Apply 3 positive outcomes
        for _ in 0..3 {
            assert!(store.apply_outcome(1, true));
        }

        // Apply 1 negative outcome
        assert!(store.apply_outcome(1, false));

        // Load the memory and verify piecewise scoring was applied
        let mem = store.load_memory(1).expect("memory should exist");
        assert_eq!(memory_positive_outcomes(&mem), 3);
        assert_eq!(memory_negative_outcomes(&mem), 1);

        // Strength should differ from initial (piecewise blended)
        assert!(
            (mem.strength - initial_strength).abs() > 0.001,
            "strength should have changed via piecewise scoring"
        );

        // Piecewise score should be well-defined
        let score = memory_outcome_score(&mem);
        assert!(score > 0.0 && score <= 1.0);

        // Relevance should have shifted
        // +3 success * 0.02 = +0.06, -1 failure * 0.02 = -0.02 → net +0.04
        let expected_relevance = 0.5 + 0.04;
        assert!(
            (mem.relevance_score - expected_relevance).abs() < 0.001,
            "relevance_score {} should be ~{}",
            mem.relevance_score,
            expected_relevance
        );
    }

    #[test]
    fn test_redb_memory_embedding_similarity() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = make_test_store(&dir);

        let shared_fp = 42u64;

        let mut mem1 = make_test_memory(1);
        mem1.context.fingerprint = shared_fp;
        mem1.context.embeddings = Some(vec![1.0, 0.0, 0.0]);
        store.store_consolidated_memory(mem1);

        let mut mem2 = make_test_memory(2);
        mem2.context.fingerprint = shared_fp;
        mem2.context.embeddings = Some(vec![0.9, 0.1, 0.0]);
        store.store_consolidated_memory(mem2);

        let mut mem3 = make_test_memory(3);
        mem3.context.fingerprint = 999; // different fingerprint
        mem3.context.embeddings = Some(vec![0.0, 0.0, 1.0]);
        store.store_consolidated_memory(mem3);

        // Query with shared_fp: exact-match returns mem1 and mem2 (similarity=1.0)
        let mut query_ctx = EventContext::default();
        query_ctx.fingerprint = shared_fp;
        query_ctx.embeddings = Some(vec![1.0, 0.0, 0.0]);

        let results = store.retrieve_by_context_similar(&query_ctx, 10, 0.5, None, None);
        let ids: Vec<MemoryId> = results.iter().map(|m| m.id).collect();
        assert!(ids.contains(&1), "mem1 should match (exact fingerprint)");
        assert!(ids.contains(&2), "mem2 should match (exact fingerprint)");
        assert!(
            !ids.contains(&3),
            "mem3 should NOT match (different fingerprint)"
        );

        // Query with fp=999: only mem3 is a candidate, exact match → passes
        query_ctx.fingerprint = 999;
        let results2 = store.retrieve_by_context_similar(&query_ctx, 10, 0.5, None, None);
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].id, 3);

        // Verify cosine utility works correctly (unit-level sanity check)
        let sim = agent_db_core::utils::cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);
        assert!(
            sim.abs() < 0.01,
            "orthogonal vectors should have ~0.0 similarity"
        );
        let sim2 = agent_db_core::utils::cosine_similarity(&[1.0, 0.0, 0.0], &[0.9, 0.1, 0.0]);
        assert!(
            sim2 > 0.9,
            "similar vectors should have high cosine similarity"
        );
    }
}
