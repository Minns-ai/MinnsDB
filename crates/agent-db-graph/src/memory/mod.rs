// crates/agent-db-graph/src/memory/mod.rs
//
// Memory Formation and Retrieval System
//
// Implements episodic memory formation from event sequences with context-based retrieval,
// strength tracking, and time-based decay.

use crate::episodes::{Episode, EpisodeId, EpisodeOutcome};
use agent_db_core::types::{
    current_timestamp, AgentId, ContextHash, EventId, SessionId, Timestamp,
};
use agent_db_events::core::{Event, EventContext};
use std::collections::HashMap;

mod similarity;
pub(crate) use similarity::calculate_context_similarity;

mod synthesis;
pub use synthesis::{synthesize_causal_note, synthesize_memory_summary, synthesize_takeaway};

/// Unique identifier for a memory
pub type MemoryId = u64;

#[derive(Debug, Clone)]
pub struct MemoryUpsert {
    pub id: MemoryId,
    pub is_new: bool,
}

/// Memory represents a consolidated experience that can be retrieved and applied
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Memory {
    /// Unique memory identifier
    pub id: MemoryId,

    /// Agent that formed this memory
    pub agent_id: AgentId,

    /// Session that formed this memory
    pub session_id: SessionId,

    /// Episode this memory was formed from (None for consolidated memories)
    pub episode_id: EpisodeId,

    // ========== LLM-Retrievable Fields ==========
    /// Natural language summary of what happened
    #[serde(default)]
    pub summary: String,

    /// Key takeaway / lesson learned from this experience
    #[serde(default)]
    pub takeaway: String,

    /// Why this succeeded or failed — causal explanation
    #[serde(default)]
    pub causal_note: String,

    /// Embedding of the summary text for semantic retrieval
    #[serde(default)]
    pub summary_embedding: Vec<f32>,

    // ========== Hierarchy Fields ==========
    /// Memory tier in the consolidation hierarchy
    #[serde(default)]
    pub tier: MemoryTier,

    /// IDs of episodic memories this was consolidated from (for Semantic/Schema tiers)
    #[serde(default)]
    pub consolidated_from: Vec<MemoryId>,

    /// Schema ID this memory was consolidated into (for Episodic tier)
    #[serde(default)]
    pub schema_id: Option<MemoryId>,

    /// Consolidation lifecycle status
    #[serde(default)]
    pub consolidation_status: ConsolidationStatus,

    // ========== Original Fields ==========
    /// Context snapshot when memory was formed
    pub context: EventContext,

    /// Key events that define this memory
    pub key_events: Vec<EventId>,

    /// Memory strength (0.0 to 1.0)
    pub strength: f32,

    /// Relevance score for retrieval (0.0 to 1.0)
    pub relevance_score: f32,

    /// Timestamp when memory was formed
    pub formed_at: Timestamp,

    /// Last access timestamp
    pub last_accessed: Timestamp,

    /// Number of times this memory has been accessed
    pub access_count: u32,

    /// Episode outcome
    pub outcome: EpisodeOutcome,

    /// Memory type
    pub memory_type: MemoryType,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

// ========== Memory Hierarchy Types ==========

/// Tier in the memory consolidation hierarchy (Episodic → Semantic → Schema)
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MemoryTier {
    /// Raw experience from a single episode
    #[default]
    Episodic,
    /// Generalized knowledge consolidated from multiple episodic memories
    Semantic,
    /// Reusable mental model consolidated from multiple semantic memories
    Schema,
}

/// Lifecycle status in the consolidation pipeline
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConsolidationStatus {
    /// Active — not yet consolidated
    #[default]
    Active,
    /// Has been consolidated into a higher-tier memory; decay can be accelerated
    Consolidated,
    /// Archived — kept for audit but excluded from retrieval
    Archived,
}

/// Type of memory
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MemoryType {
    /// Episodic memory - specific experience
    Episodic { significance: f32 },

    /// Working memory - active context (not yet implemented in MVP)
    Working,

    /// Semantic memory - abstracted knowledge (not yet implemented in MVP)
    Semantic,

    /// Negative memory - failed experience to avoid repeating
    Negative {
        /// Failure significance (0.0 to 1.0) - how badly it failed
        failure_severity: f32,
        /// What went wrong - brief description of failure pattern
        failure_pattern: String,
    },
}

/// Configuration for memory formation
#[derive(Debug, Clone)]
pub struct MemoryFormationConfig {
    pub min_significance: f32,
    pub initial_strength: f32,
    pub decay_rate_per_hour: f32,
    pub access_strength_boost: f32,
    pub max_strength: f32,
    pub forget_threshold: f32,
}

impl Default for MemoryFormationConfig {
    fn default() -> Self {
        Self {
            min_significance: 0.3,
            initial_strength: 0.7,
            decay_rate_per_hour: 0.05,
            access_strength_boost: 0.1,
            max_strength: 1.0,
            forget_threshold: 0.1,
        }
    }
}

/// Memory formation engine
pub struct MemoryFormation {
    /// All formed memories
    memories: HashMap<MemoryId, Memory>,
    /// Memory index by agent
    agent_memories: HashMap<AgentId, Vec<MemoryId>>,
    /// Memory index by context hash
    context_index: HashMap<ContextHash, Vec<MemoryId>>,
    /// Memory index by episode
    episode_index: HashMap<EpisodeId, MemoryId>,
    /// Configuration
    config: MemoryFormationConfig,
    /// Next memory ID
    next_memory_id: MemoryId,
}

impl MemoryFormation {
    /// Create a new memory formation engine
    pub fn new(config: MemoryFormationConfig) -> Self {
        Self {
            memories: HashMap::new(),
            agent_memories: HashMap::new(),
            context_index: HashMap::new(),
            episode_index: HashMap::new(),
            config,
            next_memory_id: 1,
        }
    }

    /// Form a memory from an episode
    ///
    /// Returns the memory ID if a memory was formed, None if episode doesn't meet criteria
    pub fn form_memory(&mut self, episode: &Episode, events: &[Event]) -> Option<MemoryUpsert> {
        if let Some(existing_id) = self.episode_index.get(&episode.id).copied() {
            let updated = self.update_memory_from_episode(existing_id, episode);
            if updated {
                return Some(MemoryUpsert {
                    id: existing_id,
                    is_new: false,
                });
            }
            return Some(MemoryUpsert {
                id: existing_id,
                is_new: false,
            });
        }

        // Check if episode meets significance threshold
        if episode.significance < self.config.min_significance {
            tracing::info!(
                "Memory formation rejected episode_id={} significance={:.3} min={:.3}",
                episode.id,
                episode.significance,
                self.config.min_significance
            );
            return None;
        }

        // Check if episode has ended
        if episode.end_timestamp.is_none() {
            tracing::info!(
                "Memory formation rejected episode_id={} (no end_timestamp)",
                episode.id
            );
            return None;
        }

        let memory_id = self.next_memory_id;
        self.next_memory_id += 1;

        let current_time = current_timestamp();

        // Phase 1 Feature C: Determine memory type based on outcome
        // Failures create Negative memories for avoidance learning
        let memory_type = match episode.outcome {
            Some(EpisodeOutcome::Failure) => {
                // Create negative memory to avoid repeating this failure
                let failure_pattern = format!(
                    "Failed episode {} with significance {:.2} - avoid similar context",
                    episode.id, episode.significance
                );
                MemoryType::Negative {
                    failure_severity: episode.significance,
                    failure_pattern,
                }
            },
            _ => {
                // Success, Partial, or Interrupted create episodic memories
                MemoryType::Episodic {
                    significance: episode.significance,
                }
            },
        };

        // Phase 1 Feature A: Apply prediction error weighting to memory strength
        // Surprising outcomes (high prediction error) get stronger initial strength
        let prediction_weighted_strength =
            self.config.initial_strength * (1.0 + episode.prediction_error);

        // Use the latest episode context snapshot for retrieval
        let mut context = episode.context.clone();
        if context.fingerprint == 0 {
            context.fingerprint = context.compute_fingerprint();
        }
        if context.goal_bucket_id == 0 {
            context.goal_bucket_id = context.compute_goal_bucket_id();
        }

        // Generate natural language summary + causal analysis from events
        let summary = synthesize_memory_summary(episode, events);
        let causal_note = synthesize_causal_note(episode, events);
        let takeaway = synthesize_takeaway(episode, events);

        // Create memory from episode
        let memory = Memory {
            id: memory_id,
            agent_id: episode.agent_id,
            session_id: episode.session_id,
            episode_id: episode.id,
            summary,
            takeaway,
            causal_note,
            summary_embedding: Vec::new(), // Populated async by refinement pipeline
            tier: MemoryTier::Episodic,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: ConsolidationStatus::Active,
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
            metadata: HashMap::new(),
        };

        // Store memory
        self.memories.insert(memory_id, memory.clone());

        // Index by agent
        self.agent_memories
            .entry(episode.agent_id)
            .or_default()
            .push(memory_id);

        // Index by context
        self.context_index
            .entry(memory.context.fingerprint)
            .or_default()
            .push(memory_id);

        // Index by episode
        self.episode_index.insert(episode.id, memory_id);

        tracing::info!(
            "Memory stored id={} episode_id={} agent_id={} session_id={} strength={:.3} relevance={:.3} context_hash={}",
            memory_id,
            episode.id,
            episode.agent_id,
            episode.session_id,
            memory.strength,
            memory.relevance_score,
            memory.context.fingerprint
        );

        Some(MemoryUpsert {
            id: memory_id,
            is_new: true,
        })
    }

    fn update_memory_from_episode(&mut self, memory_id: MemoryId, episode: &Episode) -> bool {
        let Some(memory) = self.memories.get_mut(&memory_id) else {
            return false;
        };

        let old_fingerprint = memory.context.fingerprint;
        let mut context = episode.context.clone();
        if context.fingerprint == 0 {
            context.fingerprint = context.compute_fingerprint();
        }
        if context.goal_bucket_id == 0 {
            context.goal_bucket_id = context.compute_goal_bucket_id();
        }

        memory.context = context;
        memory.key_events = episode.events.clone();
        memory.outcome = episode
            .outcome
            .clone()
            .unwrap_or(EpisodeOutcome::Interrupted);
        memory.memory_type = match episode.outcome {
            Some(EpisodeOutcome::Failure) => MemoryType::Negative {
                failure_severity: episode.significance,
                failure_pattern: format!(
                    "Failed episode {} with significance {:.2} - avoid similar context",
                    episode.id, episode.significance
                ),
            },
            _ => MemoryType::Episodic {
                significance: episode.significance,
            },
        };

        let prediction_weighted_strength =
            self.config.initial_strength * (1.0 + episode.prediction_error);
        memory.strength = memory
            .strength
            .max(prediction_weighted_strength)
            .min(self.config.max_strength);
        memory.relevance_score = episode.significance * (1.0 + episode.prediction_error * 0.5);

        if old_fingerprint != memory.context.fingerprint {
            if let Some(list) = self.context_index.get_mut(&old_fingerprint) {
                list.retain(|id| *id != memory_id);
            }
            self.context_index
                .entry(memory.context.fingerprint)
                .or_default()
                .push(memory_id);
        }

        tracing::info!(
            "Memory updated from episode correction id={} episode_id={} outcome={:?} strength={:.3} relevance={:.3}",
            memory_id,
            episode.id,
            memory.outcome,
            memory.strength,
            memory.relevance_score
        );

        true
    }

    /// Retrieve memories by context similarity
    ///
    /// Returns top-k most relevant memories sorted by relevance
    pub fn retrieve_by_context(&mut self, context: &EventContext, limit: usize) -> Vec<Memory> {
        let context_hash = context.fingerprint;
        tracing::info!(
            "Memory retrieve_by_context hash={} limit={}",
            context_hash,
            limit
        );

        // Get candidate memory IDs from context index
        let candidate_ids: Vec<MemoryId> = self
            .context_index
            .get(&context_hash)
            .cloned()
            .unwrap_or_default();

        // Update access tracking and collect memories
        let current_time = current_timestamp();
        let mut candidates: Vec<Memory> = candidate_ids
            .iter()
            .filter_map(|id| {
                if let Some(memory) = self.memories.get_mut(id) {
                    memory.access_count += 1;
                    memory.last_accessed = current_time;

                    // Boost strength on access
                    memory.strength = (memory.strength + self.config.access_strength_boost)
                        .min(self.config.max_strength);

                    Some(memory.clone())
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance score (higher is better)
        candidates.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top-k
        {
            let result = candidates.into_iter().take(limit).collect::<Vec<_>>();
            tracing::info!("Memory retrieve_by_context results={}", result.len());
            result
        }
    }

    /// Retrieve memories by context similarity with optional filtering
    pub fn retrieve_by_context_similar(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory> {
        tracing::info!(
            "Memory retrieve_by_context_similar limit={} min_similarity={:.3} agent_id={:?} session_id={:?}",
            limit,
            min_similarity,
            agent_id,
            session_id
        );
        let current_time = current_timestamp();
        let mut candidates: Vec<(f32, Memory)> = self
            .memories
            .values_mut()
            .filter_map(|memory| {
                if let Some(agent_id) = agent_id {
                    if memory.agent_id != agent_id {
                        return None;
                    }
                }
                if let Some(session_id) = session_id {
                    if memory.session_id != session_id {
                        return None;
                    }
                }

                let similarity = calculate_context_similarity(context, &memory.context);
                if similarity < min_similarity {
                    return None;
                }

                // Update access tracking
                memory.access_count += 1;
                memory.last_accessed = current_time;

                // Boost strength on access
                memory.strength = (memory.strength + self.config.access_strength_boost)
                    .min(self.config.max_strength);

                let score = similarity * memory.relevance_score;
                Some((score, memory.clone()))
            })
            .collect();

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        {
            let result = candidates
                .into_iter()
                .take(limit)
                .map(|(_, memory)| memory)
                .collect::<Vec<_>>();
            tracing::info!(
                "Memory retrieve_by_context_similar results={}",
                result.len()
            );
            result
        }
    }

    /// Apply explicit outcome feedback to a memory (used -> outcome)
    pub fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool {
        let current_time = current_timestamp();
        let Some(memory) = self.memories.get_mut(&memory_id) else {
            return false;
        };

        memory.access_count += 1;
        memory.last_accessed = current_time;

        if success {
            memory.strength =
                (memory.strength + self.config.access_strength_boost).min(self.config.max_strength);
            memory.relevance_score = (memory.relevance_score + 0.02).min(1.0);
        } else {
            memory.strength = (memory.strength - self.config.decay_rate_per_hour)
                .max(self.config.forget_threshold);
            memory.relevance_score = (memory.relevance_score - 0.02).max(0.0);
        }

        true
    }

    /// Retrieve memories for a specific agent
    pub fn retrieve_by_agent(&self, agent_id: AgentId, limit: usize) -> Vec<&Memory> {
        self.agent_memories
            .get(&agent_id)
            .map(|ids| {
                let mut memories: Vec<&Memory> =
                    ids.iter().filter_map(|id| self.memories.get(id)).collect();

                // Sort by relevance
                memories.sort_by(|a, b| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                memories.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    /// Apply memory decay based on time elapsed
    ///
    /// Should be called periodically to age memories
    pub fn apply_decay(&mut self) {
        let current_time = current_timestamp();
        let hour_in_ns = 3_600_000_000_000u64;

        let mut to_forget = Vec::new();

        for (id, memory) in self.memories.iter_mut() {
            // Calculate time since last access in hours
            let time_elapsed_ns = current_time.saturating_sub(memory.last_accessed);
            let hours_elapsed = (time_elapsed_ns / hour_in_ns) as f32;

            // Apply decay
            let decay_amount = self.config.decay_rate_per_hour * hours_elapsed;
            memory.strength = (memory.strength - decay_amount).max(0.0);

            // Mark for forgetting if below threshold
            if memory.strength < self.config.forget_threshold {
                to_forget.push(*id);
            }
        }

        // Remove forgotten memories
        for id in to_forget {
            if let Some(memory) = self.memories.remove(&id) {
                // Remove from agent index
                if let Some(agent_mems) = self.agent_memories.get_mut(&memory.agent_id) {
                    agent_mems.retain(|&mid| mid != id);
                }

                // Remove from context index
                if let Some(context_mems) = self.context_index.get_mut(&memory.context.fingerprint)
                {
                    context_mems.retain(|&mid| mid != id);
                }
            }
        }
    }

    /// Get a specific memory by ID
    pub fn get_memory(&self, memory_id: MemoryId) -> Option<&Memory> {
        self.memories.get(&memory_id)
    }

    /// Get total memory count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let total_memories = self.memories.len();
        let avg_strength = if total_memories > 0 {
            self.memories.values().map(|m| m.strength).sum::<f32>() / total_memories as f32
        } else {
            0.0
        };

        let avg_access_count = if total_memories > 0 {
            self.memories.values().map(|m| m.access_count).sum::<u32>() / total_memories as u32
        } else {
            0
        };

        let mut episodic_count = 0usize;
        let mut semantic_count = 0usize;
        let mut schema_count = 0usize;
        for m in self.memories.values() {
            match m.tier {
                MemoryTier::Episodic => episodic_count += 1,
                MemoryTier::Semantic => semantic_count += 1,
                MemoryTier::Schema => schema_count += 1,
            }
        }

        MemoryStats {
            total_memories,
            avg_strength,
            avg_access_count,
            agents_with_memories: self.agent_memories.len(),
            unique_contexts: self.context_index.len(),
            episodic_count,
            semantic_count,
            schema_count,
        }
    }

    /// Insert a memory loaded from persistent storage
    pub fn insert_loaded_memory(&mut self, memory: Memory) -> Result<(), String> {
        let memory_id = memory.id;
        let agent_id = memory.agent_id;
        let episode_id = memory.episode_id;
        let context_hash = memory.context.fingerprint;

        // Update next_memory_id if needed
        if memory_id >= self.next_memory_id {
            self.next_memory_id = memory_id + 1;
        }

        // Store memory
        self.memories.insert(memory_id, memory);

        // Index by agent
        self.agent_memories
            .entry(agent_id)
            .or_default()
            .push(memory_id);

        // Index by context
        self.context_index
            .entry(context_hash)
            .or_default()
            .push(memory_id);

        // Index by episode
        self.episode_index.insert(episode_id, memory_id);

        Ok(())
    }

    // ========== Consolidation API ==========

    /// List all memories (for consolidation engine)
    pub fn list_all(&self) -> Vec<Memory> {
        self.memories.values().cloned().collect()
    }

    /// Remove a memory by ID. Returns true if it existed.
    pub fn remove_memory(&mut self, memory_id: MemoryId) -> bool {
        if let Some(memory) = self.memories.remove(&memory_id) {
            // Clean indexes
            if let Some(ids) = self.agent_memories.get_mut(&memory.agent_id) {
                ids.retain(|id| *id != memory_id);
            }
            if let Some(ids) = self.context_index.get_mut(&memory.context.fingerprint) {
                ids.retain(|id| *id != memory_id);
            }
            self.episode_index.retain(|_, id| *id != memory_id);
            true
        } else {
            false
        }
    }

    /// Store a pre-built consolidated memory directly (bypasses episode formation)
    pub fn store_direct(&mut self, memory: Memory) {
        let id = memory.id;
        let agent_id = memory.agent_id;
        let episode_id = memory.episode_id;
        let fp = memory.context.fingerprint;

        if id >= self.next_memory_id {
            self.next_memory_id = id + 1;
        }

        self.memories.insert(id, memory);
        self.agent_memories.entry(agent_id).or_default().push(id);
        self.context_index.entry(fp).or_default().push(id);
        if episode_id > 0 {
            self.episode_index.insert(episode_id, id);
        }
    }

    /// Mark a memory as consolidated into a higher-tier memory, applying strength decay
    pub fn mark_consolidated(
        &mut self,
        memory_id: MemoryId,
        into_id: MemoryId,
        into_tier: MemoryTier,
        decay: f32,
    ) {
        if let Some(m) = self.memories.get_mut(&memory_id) {
            m.consolidation_status = ConsolidationStatus::Consolidated;
            if into_tier == MemoryTier::Schema {
                m.schema_id = Some(into_id);
            }
            m.strength *= decay; // Accelerated decay for consolidated memories
        }
    }

    /// Schema-first hierarchical retrieval.
    /// Returns memories preferring Schema > Semantic > Episodic within the limit.
    pub fn retrieve_hierarchical(
        &self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
    ) -> Vec<Memory> {
        use agent_db_core::utils::cosine_similarity;

        let query_fp = if context.fingerprint != 0 {
            context.fingerprint
        } else {
            context.compute_fingerprint()
        };

        let query_embedding = context.embeddings.as_deref().unwrap_or(&[]);

        // Score all active memories
        let mut scored: Vec<(f32, &Memory)> = self
            .memories
            .values()
            .filter(|m| m.consolidation_status != ConsolidationStatus::Archived)
            .filter(|m| agent_id.is_none_or(|aid| m.agent_id == aid))
            .filter_map(|m| {
                // Compute similarity
                let fp_sim = if m.context.fingerprint == query_fp {
                    1.0f32
                } else {
                    0.0
                };
                let emb_sim = if !query_embedding.is_empty() {
                    let m_emb = m.context.embeddings.as_deref().unwrap_or(&[]);
                    if !m_emb.is_empty() {
                        cosine_similarity(query_embedding, m_emb)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let sim = fp_sim.max(emb_sim);

                if sim >= min_similarity {
                    // Tier boost: Schema > Semantic > Episodic
                    let tier_boost = match m.tier {
                        MemoryTier::Schema => 0.3,
                        MemoryTier::Semantic => 0.15,
                        MemoryTier::Episodic => 0.0,
                    };
                    Some((sim + tier_boost, m))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, m)| m.clone())
            .collect()
    }
}

/// Statistics about memory formation and retrieval
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub avg_strength: f32,
    pub avg_access_count: u32,
    pub agents_with_memories: usize,
    pub unique_contexts: usize,
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub schema_count: usize,
}
