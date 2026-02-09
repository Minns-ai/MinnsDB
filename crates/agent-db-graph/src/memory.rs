// crates/agent-db-graph/src/memory.rs
//
// Memory Formation and Retrieval System
//
// Implements episodic memory formation from event sequences with context-based retrieval,
// strength tracking, and time-based decay.

use crate::episodes::{Episode, EpisodeId, EpisodeOutcome};
use agent_db_core::types::{
    current_timestamp, AgentId, ContextHash, EventId, SessionId, Timestamp,
};
use agent_db_core::utils::cosine_similarity;
use agent_db_events::core::{ActionOutcome, Event, EventContext, EventType};
use std::collections::{HashMap, HashSet};

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
    Episodic {
        /// Episode significance (0.0 to 1.0)
        significance: f32,
    },

    /// Working memory - active context (not yet implemented in MVP)
    Working,

    /// Semantic memory - abstracted knowledge (not yet implemented in MVP)
    Semantic,

    // ========== Phase 1 Upgrade (Feature C) ==========
    /// Negative memory - failed experience to avoid repeating
    /// Used to prevent the agent from repeating mistakes
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
    /// Minimum episode significance to form memory
    pub min_significance: f32,

    /// Initial memory strength
    pub initial_strength: f32,

    /// Decay rate per hour (0.0 to 1.0)
    pub decay_rate_per_hour: f32,

    /// Strength boost per access
    pub access_strength_boost: f32,

    /// Maximum memory strength
    pub max_strength: f32,

    /// Minimum strength before memory is forgotten
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

                let similarity = Self::calculate_context_similarity(context, &memory.context);
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

    fn calculate_context_similarity(a: &EventContext, b: &EventContext) -> f32 {
        if let (Some(embed_a), Some(embed_b)) = (&a.embeddings, &b.embeddings) {
            return cosine_similarity(embed_a, embed_b);
        }

        Self::fallback_context_similarity(a, b)
    }

    fn fallback_context_similarity(a: &EventContext, b: &EventContext) -> f32 {
        let env_similarity = Self::key_overlap_ratio(
            a.environment.variables.keys(),
            b.environment.variables.keys(),
        );
        let goals_similarity = Self::id_overlap_ratio(
            a.active_goals.iter().map(|goal| goal.id),
            b.active_goals.iter().map(|goal| goal.id),
        );
        let resources_similarity = Self::resource_similarity(a, b);

        let mut total = 0.0;
        let mut weight = 0.0;

        if env_similarity >= 0.0 {
            total += env_similarity * 0.4;
            weight += 0.4;
        }

        if goals_similarity >= 0.0 {
            total += goals_similarity * 0.3;
            weight += 0.3;
        }

        total += resources_similarity * 0.3;
        weight += 0.3;

        if weight == 0.0 {
            0.0
        } else {
            (total / weight).clamp(0.0, 1.0)
        }
    }

    fn key_overlap_ratio<'a, I, J>(a: I, b: J) -> f32
    where
        I: Iterator<Item = &'a String>,
        J: Iterator<Item = &'a String>,
    {
        let set_a: HashSet<&String> = a.collect();
        let set_b: HashSet<&String> = b.collect();
        if set_a.is_empty() && set_b.is_empty() {
            return -1.0;
        }

        let intersection = set_a.intersection(&set_b).count() as f32;
        let union = set_a.union(&set_b).count() as f32;

        if union == 0.0 {
            -1.0
        } else {
            intersection / union
        }
    }

    fn id_overlap_ratio<I, J>(a: I, b: J) -> f32
    where
        I: Iterator<Item = u64>,
        J: Iterator<Item = u64>,
    {
        let set_a: HashSet<u64> = a.collect();
        let set_b: HashSet<u64> = b.collect();
        if set_a.is_empty() && set_b.is_empty() {
            return -1.0;
        }

        let intersection = set_a.intersection(&set_b).count() as f32;
        let union = set_a.union(&set_b).count() as f32;

        if union == 0.0 {
            -1.0
        } else {
            intersection / union
        }
    }

    fn resource_similarity(a: &EventContext, b: &EventContext) -> f32 {
        let cpu_a = a.resources.computational.cpu_percent;
        let cpu_b = b.resources.computational.cpu_percent;
        let cpu_max = cpu_a.max(cpu_b).max(1.0);
        let cpu_sim = 1.0 - ((cpu_a - cpu_b).abs() / cpu_max).min(1.0);

        let mem_a = a.resources.computational.memory_bytes as f32;
        let mem_b = b.resources.computational.memory_bytes as f32;
        let mem_max = mem_a.max(mem_b);
        let mem_sim = if mem_max == 0.0 {
            1.0
        } else {
            1.0 - ((mem_a - mem_b).abs() / mem_max).min(1.0)
        };

        ((cpu_sim + mem_sim) / 2.0).clamp(0.0, 1.0)
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

        MemoryStats {
            total_memories,
            avg_strength,
            avg_access_count,
            agents_with_memories: self.agent_memories.len(),
            unique_contexts: self.context_index.len(),
        }
    }

    /// Insert a memory loaded from persistent storage
    ///
    /// This is used to restore memories from disk without going through
    /// the episode formation process. Used during initialization.
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
    pub fn mark_consolidated(&mut self, memory_id: MemoryId, into_id: MemoryId, decay: f32) {
        if let Some(m) = self.memories.get_mut(&memory_id) {
            m.consolidation_status = ConsolidationStatus::Consolidated;
            m.schema_id = Some(into_id);
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

/// Truncate a string to `max_len` chars, appending "…" if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}…", &s[..max_len])
    } else {
        s.to_string()
    }
}

/// Truncate a `serde_json::Value` to a readable string of at most `max_len` chars.
fn truncate_value(v: &serde_json::Value, max_len: usize) -> String {
    // For strings, extract the inner string to avoid extra quotes
    let raw = match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    truncate_str(&raw, max_len)
}

/// Synthesize a natural language summary from an episode and its events.
///
/// The summary is designed to be directly usable by an LLM for retrieval-augmented
/// generation — it describes *what happened*, *what was done*, and *how it ended*
/// in plain English.
pub fn synthesize_memory_summary(episode: &Episode, events: &[Event]) -> String {
    let mut parts: Vec<String> = Vec::new();

    // 1. Goal context
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    if !goals.is_empty() {
        if goals.len() == 1 {
            parts.push(format!("Goal: {}", goals[0]));
        } else {
            parts.push(format!("Goals: {}", goals.join("; ")));
        }
    }

    // 2. Environment context (user, intent, key vars)
    let env_vars = &episode.context.environment.variables;
    let mut env_parts = Vec::new();
    if let Some(user) = env_vars.get("user_id").or_else(|| env_vars.get("user")) {
        env_parts.push(format!("user={}", user));
    }
    if let Some(intent) = env_vars
        .get("intent_type")
        .or_else(|| env_vars.get("intent"))
    {
        env_parts.push(format!("intent={}", intent));
    }
    // Include up to 3 other notable variables
    for (k, v) in env_vars.iter() {
        if env_parts.len() >= 5 {
            break;
        }
        if k != "user_id" && k != "user" && k != "intent_type" && k != "intent" {
            env_parts.push(format!("{}={}", k, v));
        }
    }
    if !env_parts.is_empty() {
        parts.push(format!("Context: {}", env_parts.join(", ")));
    }

    // 3. Walk events and extract narrative
    let mut actions: Vec<String> = Vec::new();
    let mut observations: Vec<String> = Vec::new();
    let mut context_texts: Vec<String> = Vec::new();
    let mut communications: Vec<String> = Vec::new();

    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                outcome,
                ..
            } => {
                let outcome_str = match outcome {
                    ActionOutcome::Success { result } => {
                        format!("succeeded: {}", truncate_value(result, 120))
                    },
                    ActionOutcome::Failure { error, .. } => {
                        format!("failed: {}", truncate_str(error, 120))
                    },
                    ActionOutcome::Partial { result, issues } => {
                        format!(
                            "partial: {} (issues: {:?})",
                            truncate_value(result, 80),
                            issues
                        )
                    },
                };
                actions.push(format!("'{}' {}", action_name, outcome_str));
            },
            EventType::Observation {
                observation_type,
                data,
                source,
                confidence,
                ..
            } => {
                observations.push(format!(
                    "[{}] from '{}' (conf {:.0}%): {}",
                    observation_type,
                    source,
                    confidence * 100.0,
                    truncate_value(data, 150)
                ));
            },
            EventType::Context {
                text, context_type, ..
            } => {
                context_texts.push(format!("[{}] {}", context_type, truncate_str(text, 200)));
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                communications.push(format!(
                    "{} agent {} -> agent {}: {}",
                    message_type,
                    sender,
                    recipient,
                    truncate_value(content, 150)
                ));
            },
            _ => {}, // Cognitive & Learning are internal machinery, skip for narrative
        }
    }

    if !context_texts.is_empty() {
        // Limit to first 3 to keep summary tight
        let shown: Vec<&str> = context_texts.iter().take(3).map(|s| s.as_str()).collect();
        parts.push(format!("Input: {}", shown.join(" | ")));
    }
    if !actions.is_empty() {
        let shown: Vec<&str> = actions.iter().take(5).map(|s| s.as_str()).collect();
        parts.push(format!("Actions: {}", shown.join("; ")));
    }
    if !observations.is_empty() {
        let shown: Vec<&str> = observations.iter().take(3).map(|s| s.as_str()).collect();
        parts.push(format!("Observations: {}", shown.join("; ")));
    }
    if !communications.is_empty() {
        let shown: Vec<&str> = communications.iter().take(2).map(|s| s.as_str()).collect();
        parts.push(format!("Comms: {}", shown.join("; ")));
    }

    // 4. Outcome
    let outcome_str = match &episode.outcome {
        Some(EpisodeOutcome::Success) => "Outcome: Success",
        Some(EpisodeOutcome::Failure) => "Outcome: Failure",
        Some(EpisodeOutcome::Partial) => "Outcome: Partial success",
        Some(EpisodeOutcome::Interrupted) | None => "Outcome: Interrupted/unknown",
    };
    parts.push(outcome_str.to_string());

    // 5. Stats
    parts.push(format!(
        "({} events, significance {:.0}%)",
        events.len(),
        episode.significance * 100.0
    ));

    if parts.is_empty() {
        format!(
            "Episode {} for agent {} — no detailed events available.",
            episode.id, episode.agent_id
        )
    } else {
        parts.join(". ")
    }
}

/// Synthesize a causal explanation for why the episode succeeded or failed.
pub fn synthesize_causal_note(episode: &Episode, events: &[Event]) -> String {
    let outcome = episode
        .outcome
        .clone()
        .unwrap_or(EpisodeOutcome::Interrupted);

    let mut causes: Vec<String> = Vec::new();

    // Analyze action outcomes for causal signal
    let mut successes = 0u32;
    let mut failures = 0u32;
    let mut last_failure_error = String::new();
    let mut last_success_action = String::new();

    for event in events {
        if let EventType::Action {
            action_name,
            outcome: action_out,
            ..
        } = &event.event_type
        {
            match action_out {
                ActionOutcome::Success { .. } => {
                    successes += 1;
                    last_success_action = action_name.clone();
                },
                ActionOutcome::Failure { error, .. } => {
                    failures += 1;
                    last_failure_error = error.clone();
                },
                ActionOutcome::Partial { issues, .. } => {
                    causes.push(format!(
                        "Action '{}' partially succeeded with issues: {:?}",
                        action_name, issues
                    ));
                },
            }
        }
    }

    match outcome {
        EpisodeOutcome::Success => {
            if failures == 0 {
                causes.push(format!(
                    "All {} actions succeeded cleanly (last: '{}')",
                    successes, last_success_action
                ));
            } else {
                causes.push(format!(
                    "Recovered from {} failure(s) — final action '{}' succeeded",
                    failures, last_success_action
                ));
            }
            // Goal context
            for goal in &episode.context.active_goals {
                if goal.progress >= 0.8 && !goal.description.is_empty() {
                    causes.push(format!(
                        "Goal '{}' reached {:.0}% progress",
                        goal.description,
                        goal.progress * 100.0
                    ));
                }
            }
        },
        EpisodeOutcome::Failure => {
            if !last_failure_error.is_empty() {
                causes.push(format!(
                    "Failed because: {}",
                    truncate_str(&last_failure_error, 200)
                ));
            } else {
                causes.push("Episode ended in failure without a clear action error".to_string());
            }
            if successes > 0 {
                causes.push(format!(
                    "{} action(s) succeeded before the failure occurred",
                    successes
                ));
            }
        },
        EpisodeOutcome::Partial => {
            causes.push(format!(
                "Partial: {} action(s) succeeded, {} failed",
                successes, failures
            ));
        },
        EpisodeOutcome::Interrupted => {
            causes.push("Episode was interrupted before completion".to_string());
        },
    }

    if episode.prediction_error > 0.3 {
        causes.push(format!(
            "This was a surprising outcome (prediction error {:.0}%)",
            episode.prediction_error * 100.0
        ));
    }

    if causes.is_empty() {
        "No causal signal extracted from events.".to_string()
    } else {
        causes.join(". ")
    }
}

/// Synthesize the single most important lesson from this episode.
pub fn synthesize_takeaway(episode: &Episode, events: &[Event]) -> String {
    let outcome = episode
        .outcome
        .clone()
        .unwrap_or(EpisodeOutcome::Interrupted);

    // Find the pivotal action (last success for successful episodes, last failure for failed)
    let mut pivotal_action: Option<(&str, &ActionOutcome)> = None;
    for event in events.iter().rev() {
        if let EventType::Action {
            action_name,
            outcome: action_out,
            ..
        } = &event.event_type
        {
            match (&outcome, action_out) {
                (EpisodeOutcome::Success, ActionOutcome::Success { .. }) => {
                    pivotal_action = Some((action_name, action_out));
                    break;
                },
                (EpisodeOutcome::Failure, ActionOutcome::Failure { .. }) => {
                    pivotal_action = Some((action_name, action_out));
                    break;
                },
                _ => {},
            }
        }
    }

    // Build takeaway
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    let goal_str = if goals.is_empty() {
        "this task".to_string()
    } else {
        goals[0].to_string()
    };

    match (&outcome, pivotal_action) {
        (EpisodeOutcome::Success, Some((action, ActionOutcome::Success { result }))) => {
            format!(
                "For '{}': action '{}' was the key step that led to success (result: {}).",
                goal_str,
                action,
                truncate_value(result, 100)
            )
        },
        (EpisodeOutcome::Failure, Some((action, ActionOutcome::Failure { error, .. }))) => {
            format!(
                "For '{}': action '{}' caused failure — {}. Avoid this in similar contexts.",
                goal_str,
                action,
                truncate_str(error, 100)
            )
        },
        (EpisodeOutcome::Success, _) => {
            format!(
                "Successfully completed '{}' with {} actions and significance {:.0}%.",
                goal_str,
                events.len(),
                episode.significance * 100.0
            )
        },
        (EpisodeOutcome::Failure, _) => {
            format!(
                "Failed '{}' — review approach for this context to avoid repeating.",
                goal_str
            )
        },
        (EpisodeOutcome::Partial, _) => {
            format!(
                "Partially completed '{}' — some actions succeeded, others need improvement.",
                goal_str
            )
        },
        (EpisodeOutcome::Interrupted, _) => {
            format!(
                "'{}' was interrupted — retry when context is stable.",
                goal_str
            )
        },
    }
}

/// Statistics about memory formation and retrieval
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of memories
    pub total_memories: usize,

    /// Average memory strength
    pub avg_strength: f32,

    /// Average access count per memory
    pub avg_access_count: u32,

    /// Number of agents with memories
    pub agents_with_memories: usize,

    /// Number of unique contexts in memory
    pub unique_contexts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::{Episode, EpisodeOutcome};
    use agent_db_events::{
        ComputationalResources, EnvironmentState, ResourceState, TemporalContext,
    };
    use std::collections::HashMap;

    fn test_context() -> EventContext {
        EventContext::new(
            EnvironmentState {
                variables: HashMap::new(),
                spatial: None,
                temporal: TemporalContext {
                    time_of_day: None,
                    deadlines: vec![],
                    patterns: vec![],
                },
            },
            vec![],
            ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 0.0,
                    memory_bytes: 0,
                    storage_bytes: 0,
                    network_bandwidth: 0,
                },
                external: HashMap::new(),
            },
        )
    }

    fn create_test_episode(id: EpisodeId, agent_id: AgentId, significance: f32) -> Episode {
        let context = test_context();
        Episode {
            id,
            episode_version: 1,
            agent_id,
            start_event: 1,
            end_event: Some(2),
            events: vec![1, 2],
            session_id: 1,
            context_signature: context.fingerprint,
            context,
            outcome: Some(EpisodeOutcome::Success),
            start_timestamp: current_timestamp(),
            end_timestamp: Some(current_timestamp()),
            significance,
            // Phase 1 fields
            prediction_error: 0.0,
            self_judged_quality: None,
            salience_score: 0.5,
        }
    }

    #[test]
    fn test_memory_formation_from_episode() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        let episode = create_test_episode(1, 1, 0.8);
        let memory_id = memory_formation.form_memory(&episode, &[]);

        assert!(memory_id.is_some());
        assert_eq!(memory_formation.memory_count(), 1);
    }

    #[test]
    fn test_memory_not_formed_below_significance() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        let episode = create_test_episode(1, 1, 0.1); // Low significance
        let memory_id = memory_formation.form_memory(&episode, &[]);

        assert!(memory_id.is_none());
        assert_eq!(memory_formation.memory_count(), 0);
    }

    #[test]
    fn test_memory_retrieval_by_agent() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        // Form memories for different agents
        let episode1 = create_test_episode(1, 1, 0.8);
        let episode2 = create_test_episode(2, 1, 0.7);
        let episode3 = create_test_episode(3, 2, 0.9);

        memory_formation.form_memory(&episode1, &[]);
        memory_formation.form_memory(&episode2, &[]);
        memory_formation.form_memory(&episode3, &[]);

        let agent1_memories = memory_formation.retrieve_by_agent(1, 10);
        assert_eq!(agent1_memories.len(), 2);

        let agent2_memories = memory_formation.retrieve_by_agent(2, 10);
        assert_eq!(agent2_memories.len(), 1);
    }

    #[test]
    fn test_memory_strength_boost_on_access() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        let episode = create_test_episode(1, 1, 0.8);
        let memory_id = memory_formation.form_memory(&episode, &[]).unwrap().id;

        let initial_strength = memory_formation.get_memory(memory_id).unwrap().strength;

        // Access memory multiple times
        let context = memory_formation
            .get_memory(memory_id)
            .unwrap()
            .context
            .clone();

        memory_formation.retrieve_by_context(&context, 10);
        memory_formation.retrieve_by_context(&context, 10);

        let final_strength = memory_formation.get_memory(memory_id).unwrap().strength;
        assert!(final_strength > initial_strength);
    }
}
