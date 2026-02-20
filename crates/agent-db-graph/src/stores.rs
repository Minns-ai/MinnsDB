//! Store boundaries for memory and strategy layers.
//!
//! These abstractions keep the graph engine independent from persistence.

use agent_db_core::types::{AgentId, ContextHash, SessionId};
use agent_db_events::core::{Event, EventContext};
use agent_db_storage::{BatchOperation, RedbBackend, StorageResult};
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
}

/// LRU cache entry for memories
#[derive(Debug, Clone)]
struct MemoryCacheEntry {
    memory: Memory,
    last_accessed: std::time::Instant,
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
    /// LRU cache for hot memories (bounded size)
    memory_cache: std::collections::HashMap<MemoryId, MemoryCacheEntry>,
    /// Maximum number of memories to keep in cache
    max_cache_size: usize,
    /// Next memory ID allocator
    next_memory_id: MemoryId,
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
        Self {
            backend,
            config,
            memory_cache: std::collections::HashMap::new(),
            max_cache_size,
            next_memory_id: 1,
        }
    }

    /// Initialize next_memory_id by scanning existing memories
    pub fn initialize(&mut self) -> StorageResult<()> {
        // Scan all memory IDs to find the highest
        let memories: Vec<(Vec<u8>, Memory)> =
            self.backend.scan_prefix("memory_records", vec![])?;

        for (_, memory) in memories {
            if memory.id >= self.next_memory_id {
                self.next_memory_id = memory.id + 1;
            }
        }

        tracing::info!(
            "Initialized RedbMemoryStore with next_memory_id={}",
            self.next_memory_id
        );
        Ok(())
    }

    /// Evict least recently used memory from cache if cache is full
    fn evict_if_needed(&mut self) {
        if self.memory_cache.len() >= self.max_cache_size {
            // Find LRU entry
            if let Some((&lru_id, _)) = self
                .memory_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
            {
                self.memory_cache.remove(&lru_id);
                tracing::debug!("Evicted memory {} from cache (LRU)", lru_id);
            }
        }
    }

    /// Load memory from cache or redb
    fn load_memory(&mut self, memory_id: MemoryId) -> Option<Memory> {
        // Check cache first
        if let Some(entry) = self.memory_cache.get_mut(&memory_id) {
            entry.last_accessed = std::time::Instant::now();
            return Some(entry.memory.clone());
        }

        // Cache miss - load from redb
        match self
            .backend
            .get::<_, Memory>("memory_records", memory_id.to_be_bytes())
        {
            Ok(Some(memory)) => {
                // Add to cache
                self.evict_if_needed();
                self.memory_cache.insert(
                    memory_id,
                    MemoryCacheEntry {
                        memory: memory.clone(),
                        last_accessed: std::time::Instant::now(),
                    },
                );
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

    /// Add memory to cache
    fn cache_memory(&mut self, memory: Memory) {
        self.evict_if_needed();
        self.memory_cache.insert(
            memory.id,
            MemoryCacheEntry {
                memory,
                last_accessed: std::time::Instant::now(),
            },
        );
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

        // Main record
        let value =
            bincode::serialize(memory).map_err(agent_db_storage::StorageError::Serialization)?;
        ops.push(BatchOperation::Put {
            table_name: "memory_records".to_string(),
            key: memory.id.to_be_bytes().to_vec(),
            value,
        });

        // Index by context fingerprint
        let mut context_key = Vec::with_capacity(16);
        context_key.extend_from_slice(&memory.context.fingerprint.to_be_bytes());
        context_key.extend_from_slice(&memory.id.to_be_bytes());
        ops.push(BatchOperation::Put {
            table_name: "mem_by_context_hash".to_string(),
            key: context_key,
            value: bincode::serialize(&())
                .map_err(agent_db_storage::StorageError::Serialization)?,
        });

        // Index by agent
        let mut agent_key = Vec::with_capacity(16);
        agent_key.extend_from_slice(&memory.agent_id.to_be_bytes());
        agent_key.extend_from_slice(&memory.id.to_be_bytes());
        ops.push(BatchOperation::Put {
            table_name: "mem_by_bucket".to_string(),
            key: agent_key,
            value: bincode::serialize(&())
                .map_err(agent_db_storage::StorageError::Serialization)?,
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
                value: serde_json::to_vec(&())
                    .map_err(|e| agent_db_storage::StorageError::DatabaseError(e.to_string()))?,
            });
        }

        Ok(ops)
    }

    /// Delete a memory from redb
    #[allow(dead_code)]
    fn delete_memory(&self, memory: &Memory) -> StorageResult<()> {
        // Delete main record
        self.backend
            .delete("memory_records", memory.id.to_be_bytes())?;

        // Delete context index
        let mut context_key = Vec::with_capacity(16);
        context_key.extend_from_slice(&memory.context.fingerprint.to_be_bytes());
        context_key.extend_from_slice(&memory.id.to_be_bytes());
        self.backend.delete("mem_by_context_hash", context_key)?;

        // Delete agent index
        let mut agent_key = Vec::with_capacity(16);
        agent_key.extend_from_slice(&memory.agent_id.to_be_bytes());
        agent_key.extend_from_slice(&memory.id.to_be_bytes());
        self.backend.delete("mem_by_bucket", agent_key)?;

        // Delete goal bucket index
        let goal_bucket = memory.context.goal_bucket_id;
        if goal_bucket != 0 {
            let mut gb_key = Vec::with_capacity(16);
            gb_key.extend_from_slice(&goal_bucket.to_be_bytes());
            gb_key.extend_from_slice(&memory.id.to_be_bytes());
            self.backend.delete("mem_by_goal_bucket", gb_key)?;
        }

        Ok(())
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

        // Allocate new memory ID
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

        // Persist to redb
        if let Err(e) = self.persist_memory(&memory) {
            tracing::error!("Failed to persist memory {}: {:?}", memory_id, e);
            return None;
        }

        // Add to cache
        self.cache_memory(memory);

        Some(MemoryUpsert {
            id: memory_id,
            is_new: true,
        })
    }

    fn get_memory(&self, memory_id: MemoryId) -> Option<Memory> {
        // This needs to be mutable to update cache access time
        // For now, just check cache without updating access time
        if let Some(entry) = self.memory_cache.get(&memory_id) {
            return Some(entry.memory.clone());
        }

        // Load from redb
        match self
            .backend
            .get::<_, Memory>("memory_records", memory_id.to_be_bytes())
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

        // Filter by similarity threshold (using context fingerprint as proxy)
        candidates.retain(|m| {
            let similarity = if m.context.fingerprint == context.fingerprint {
                1.0
            } else {
                0.5 // TODO: Implement proper similarity metric
            };
            similarity >= min_similarity
        });

        candidates.truncate(limit);
        candidates
    }

    fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool {
        // Load memory
        let mut memory = match self.load_memory(memory_id) {
            Some(m) => m,
            None => return false,
        };

        // Update strength based on outcome
        if success {
            memory.strength =
                (memory.strength + self.config.access_strength_boost).min(self.config.max_strength);
        } else {
            memory.strength = (memory.strength - self.config.access_strength_boost * 0.5).max(0.0);
        }

        memory.access_count += 1;
        memory.last_accessed = agent_db_core::types::current_timestamp();

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
        // Full scan of all memories from redb for accurate stats
        let all_memories: Vec<(Vec<u8>, Memory)> = self
            .backend
            .scan_prefix("memory_records", vec![])
            .unwrap_or_default();

        let total = all_memories.len();
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

        let avg_strength = all_memories.iter().map(|(_, m)| m.strength).sum::<f32>() / total as f32;
        let avg_access_count = all_memories
            .iter()
            .map(|(_, m)| m.access_count)
            .sum::<u32>()
            / total as u32;

        let mut agents: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut contexts: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut episodic_count = 0usize;
        let mut semantic_count = 0usize;
        let mut schema_count = 0usize;

        for (_, m) in &all_memories {
            agents.insert(m.agent_id);
            contexts.insert(m.context.fingerprint);
            match m.tier {
                crate::memory::MemoryTier::Episodic => episodic_count += 1,
                crate::memory::MemoryTier::Semantic => semantic_count += 1,
                crate::memory::MemoryTier::Schema => schema_count += 1,
            }
        }

        MemoryStats {
            total_memories: total,
            avg_strength,
            avg_access_count,
            agents_with_memories: agents.len(),
            unique_contexts: contexts.len(),
            episodic_count,
            semantic_count,
            schema_count,
        }
    }

    fn apply_decay(&mut self) {
        // Decay is expensive for persistent store
        // Only decay cached memories, lazy decay others on access
        use agent_db_core::types::current_timestamp;

        let current_time = current_timestamp();
        let hour_in_ns = 3_600_000_000_000u64;

        let mut to_remove = Vec::new();

        for (id, entry) in self.memory_cache.iter_mut() {
            let time_elapsed_ns = current_time.saturating_sub(entry.memory.last_accessed);
            let hours_elapsed = (time_elapsed_ns / hour_in_ns) as f32;

            let decay_amount = self.config.decay_rate_per_hour * hours_elapsed;
            entry.memory.strength = (entry.memory.strength - decay_amount).max(0.0);

            if entry.memory.strength < self.config.forget_threshold {
                to_remove.push(*id);
            }
        }

        // Remove forgotten memories from cache
        for id in to_remove {
            self.memory_cache.remove(&id);
        }
    }

    fn list_all_memories(&self) -> Vec<Memory> {
        // Scan redb for all memories, use cache version where available
        let mut all = std::collections::HashMap::new();

        // First, load everything from redb
        let scan_result: Result<Vec<(Vec<u8>, Memory)>, _> =
            self.backend.scan_prefix("memory_records", vec![]);
        match scan_result {
            Ok(records) => {
                for (_, memory) in records {
                    all.insert(memory.id, memory);
                }
            },
            Err(e) => {
                tracing::error!("Failed to scan memories from redb: {:?}", e);
            },
        }

        // Override with cached versions (fresher)
        for (id, entry) in &self.memory_cache {
            all.insert(*id, entry.memory.clone());
        }

        all.into_values().collect()
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
        // Cache
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
        if let Some(entry) = self.memory_cache.get_mut(&memory_id) {
            entry.memory.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
            if into_tier == crate::memory::MemoryTier::Schema {
                entry.memory.schema_id = Some(into_id);
            }
            entry.memory.strength *= decay;
            let mem_clone = entry.memory.clone();
            let _ = self.persist_memory(&mem_clone);
        } else if let Some(mut memory) = self.load_memory(memory_id) {
            memory.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
            if into_tier == crate::memory::MemoryTier::Schema {
                memory.schema_id = Some(into_id);
            }
            memory.strength *= decay;
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

        // Cache them all after successful persist
        for memory in memories {
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
            let memory = if let Some(entry) = self.memory_cache.get_mut(&memory_id) {
                entry.memory.consolidation_status =
                    crate::memory::ConsolidationStatus::Consolidated;
                if into_tier == crate::memory::MemoryTier::Schema {
                    entry.memory.schema_id = Some(into_id);
                }
                entry.memory.strength *= decay;
                Some(entry.memory.clone())
            } else {
                self.load_memory(memory_id).map(|mut m| {
                    m.consolidation_status = crate::memory::ConsolidationStatus::Consolidated;
                    if into_tier == crate::memory::MemoryTier::Schema {
                        m.schema_id = Some(into_id);
                    }
                    m.strength *= decay;
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

/// LRU cache entry for strategies
#[derive(Debug, Clone)]
struct StrategyCacheEntry {
    strategy: Strategy,
    last_accessed: std::time::Instant,
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
    /// LRU cache for hot strategies (bounded size)
    strategy_cache: std::collections::HashMap<StrategyId, StrategyCacheEntry>,
    /// Maximum number of strategies to keep in cache
    max_cache_size: usize,
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
        Self {
            backend,
            config: config.clone(),
            strategy_cache: std::collections::HashMap::new(),
            max_cache_size,
            next_strategy_id: 1,
            strategy_extractor: StrategyExtractor::new(config),
        }
    }

    /// Initialize next_strategy_id by scanning existing strategies
    pub fn initialize(&mut self) -> StorageResult<()> {
        // Scan all strategy IDs to find the highest
        let strategies: Vec<(Vec<u8>, Strategy)> =
            self.backend.scan_prefix("strategy_records", vec![])?;

        for (_, strategy) in strategies {
            if strategy.id >= self.next_strategy_id {
                self.next_strategy_id = strategy.id + 1;
            }
        }

        tracing::info!(
            "Initialized RedbStrategyStore with next_strategy_id={}",
            self.next_strategy_id
        );
        Ok(())
    }

    /// Evict least recently used strategy from cache if cache is full
    fn evict_if_needed(&mut self) {
        if self.strategy_cache.len() >= self.max_cache_size {
            // Find LRU entry
            if let Some((&lru_id, _)) = self
                .strategy_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
            {
                self.strategy_cache.remove(&lru_id);
                tracing::debug!("Evicted strategy {} from cache (LRU)", lru_id);
            }
        }
    }

    /// Load strategy from cache or redb
    fn load_strategy(&mut self, strategy_id: StrategyId) -> Option<Strategy> {
        // Check cache first
        if let Some(entry) = self.strategy_cache.get_mut(&strategy_id) {
            entry.last_accessed = std::time::Instant::now();
            return Some(entry.strategy.clone());
        }

        // Cache miss - load from redb
        match self
            .backend
            .get::<_, Strategy>("strategy_records", strategy_id.to_be_bytes())
        {
            Ok(Some(strategy)) => {
                // Add to cache
                self.evict_if_needed();
                self.strategy_cache.insert(
                    strategy_id,
                    StrategyCacheEntry {
                        strategy: strategy.clone(),
                        last_accessed: std::time::Instant::now(),
                    },
                );
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

    /// Add strategy to cache
    fn cache_strategy(&mut self, strategy: Strategy) {
        self.evict_if_needed();
        self.strategy_cache.insert(
            strategy.id,
            StrategyCacheEntry {
                strategy,
                last_accessed: std::time::Instant::now(),
            },
        );
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

        // Main record
        let value =
            bincode::serialize(strategy).map_err(agent_db_storage::StorageError::Serialization)?;
        ops.push(BatchOperation::Put {
            table_name: "strategy_records".to_string(),
            key: strategy.id.to_be_bytes().to_vec(),
            value,
        });

        // Index by goal bucket
        let mut bucket_key = Vec::with_capacity(16);
        bucket_key.extend_from_slice(&strategy.goal_bucket_id.to_be_bytes());
        bucket_key.extend_from_slice(&strategy.id.to_be_bytes());
        ops.push(BatchOperation::Put {
            table_name: "strategy_by_bucket".to_string(),
            key: bucket_key,
            value: bincode::serialize(&())
                .map_err(agent_db_storage::StorageError::Serialization)?,
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
            value: bincode::serialize(&())
                .map_err(agent_db_storage::StorageError::Serialization)?,
        });

        // Index by agent
        let mut agent_key = Vec::with_capacity(16);
        agent_key.extend_from_slice(&strategy.agent_id.to_be_bytes());
        agent_key.extend_from_slice(&strategy.id.to_be_bytes());
        let quality_bytes = bincode::serialize(&strategy.quality_score)
            .map_err(agent_db_storage::StorageError::Serialization)?;
        ops.push(BatchOperation::Put {
            table_name: "strategy_feature_postings".to_string(),
            key: agent_key,
            value: quality_bytes,
        });

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
        // Check cache first (read-only, can't update access time)
        if let Some(entry) = self.strategy_cache.get(&strategy_id) {
            return Some(entry.strategy.clone());
        }

        // Fall back to persistent storage
        match self
            .backend
            .get::<_, Strategy>("strategy_records", strategy_id.to_be_bytes())
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
            match self.backend.scan_prefix("strategy_records", vec![]) {
                Ok(s) => s,
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
            self.strategy_cache.remove(id);
            // Build a temporary strategy to clean up indexes
            if let Ok(Some(strategy)) = self
                .backend
                .get::<_, Strategy>("strategy_records", id.to_be_bytes())
            {
                let _ = self.delete_strategy(&strategy);
            } else {
                let _ = self.backend.delete("strategy_records", id.to_be_bytes());
            }
        }

        pruned_ids.len() + merged
    }

    fn list_all_strategies(&self) -> Vec<Strategy> {
        let mut all = std::collections::HashMap::new();

        // Load from redb
        let scan: Result<Vec<(Vec<u8>, Strategy)>, _> =
            self.backend.scan_prefix("strategy_records", vec![]);
        if let Ok(records) = scan {
            for (_, strategy) in records {
                all.insert(strategy.id, strategy);
            }
        }

        // Override with cached versions
        for (id, entry) in &self.strategy_cache {
            all.insert(*id, entry.strategy.clone());
        }

        all.into_values().collect()
    }
}
