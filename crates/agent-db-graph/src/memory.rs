// crates/agent-db-graph/src/memory.rs
//
// Memory Formation and Retrieval System
//
// Implements episodic memory formation from event sequences with context-based retrieval,
// strength tracking, and time-based decay.

use crate::episodes::{Episode, EpisodeId, EpisodeOutcome};
use agent_db_core::types::{AgentId, EventId, Timestamp, ContextHash, current_timestamp};
use agent_db_events::core::EventContext;
use agent_db_events::{EnvironmentState, TemporalContext, ResourceState, ComputationalResources};
use std::collections::HashMap;

/// Unique identifier for a memory
pub type MemoryId = u64;

/// Memory represents a consolidated experience that can be retrieved and applied
#[derive(Debug, Clone)]
pub struct Memory {
    /// Unique memory identifier
    pub id: MemoryId,

    /// Agent that formed this memory
    pub agent_id: AgentId,

    /// Episode this memory was formed from
    pub episode_id: EpisodeId,

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

/// Type of memory
#[derive(Debug, Clone, PartialEq)]
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
            config,
            next_memory_id: 1,
        }
    }

    /// Form a memory from an episode
    ///
    /// Returns the memory ID if a memory was formed, None if episode doesn't meet criteria
    pub fn form_memory(&mut self, episode: &Episode) -> Option<MemoryId> {
        // Check if episode meets significance threshold
        if episode.significance < self.config.min_significance {
            return None;
        }

        // Check if episode has ended
        if episode.end_timestamp.is_none() {
            return None;
        }

        let memory_id = self.next_memory_id;
        self.next_memory_id += 1;

        let current_time = current_timestamp();

        // Create memory from episode
        let memory = Memory {
            id: memory_id,
            agent_id: episode.agent_id,
            episode_id: episode.id,
            context: EventContext {
                environment: EnvironmentState {
                    variables: HashMap::new(),
                    spatial: None,
                    temporal: TemporalContext {
                        time_of_day: None,
                        deadlines: vec![],
                        patterns: vec![],
                    },
                },
                active_goals: vec![],
                resources: ResourceState {
                    computational: ComputationalResources {
                        cpu_percent: 0.0,
                        memory_bytes: 0,
                        storage_bytes: 0,
                        network_bandwidth: 0,
                    },
                    external: HashMap::new(),
                },
                fingerprint: episode.context_signature,
                embeddings: None,
            },
            key_events: episode.events.clone(),
            strength: self.config.initial_strength,
            relevance_score: episode.significance,
            formed_at: current_time,
            last_accessed: current_time,
            access_count: 0,
            outcome: episode.outcome.clone().unwrap_or(EpisodeOutcome::Interrupted),
            memory_type: MemoryType::Episodic {
                significance: episode.significance,
            },
            metadata: HashMap::new(),
        };

        // Store memory
        self.memories.insert(memory_id, memory.clone());

        // Index by agent
        self.agent_memories
            .entry(episode.agent_id)
            .or_insert_with(Vec::new)
            .push(memory_id);

        // Index by context
        self.context_index
            .entry(episode.context_signature)
            .or_insert_with(Vec::new)
            .push(memory_id);

        Some(memory_id)
    }

    /// Retrieve memories by context similarity
    ///
    /// Returns top-k most relevant memories sorted by relevance
    pub fn retrieve_by_context(
        &mut self,
        context: &EventContext,
        limit: usize,
    ) -> Vec<Memory> {
        let context_hash = context.fingerprint;

        // Get candidate memory IDs from context index
        let candidate_ids: Vec<MemoryId> = self
            .context_index
            .get(&context_hash)
            .map(|ids| ids.clone())
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
        candidates.into_iter().take(limit).collect()
    }

    /// Retrieve memories for a specific agent
    pub fn retrieve_by_agent(&self, agent_id: AgentId, limit: usize) -> Vec<&Memory> {
        self.agent_memories
            .get(&agent_id)
            .map(|ids| {
                let mut memories: Vec<&Memory> = ids
                    .iter()
                    .filter_map(|id| self.memories.get(id))
                    .collect();

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
                if let Some(context_mems) = self.context_index.get_mut(&memory.context.fingerprint) {
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

    fn create_test_episode(id: EpisodeId, agent_id: AgentId, significance: f32) -> Episode {
        Episode {
            id,
            agent_id,
            start_event: 1,
            end_event: Some(2),
            events: vec![1, 2],
            context_signature: 12345,
            outcome: Some(EpisodeOutcome::Success),
            start_timestamp: current_timestamp(),
            end_timestamp: Some(current_timestamp()),
            significance,
        }
    }

    #[test]
    fn test_memory_formation_from_episode() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        let episode = create_test_episode(1, 1, 0.8);
        let memory_id = memory_formation.form_memory(&episode);

        assert!(memory_id.is_some());
        assert_eq!(memory_formation.memory_count(), 1);
    }

    #[test]
    fn test_memory_not_formed_below_significance() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        let episode = create_test_episode(1, 1, 0.1); // Low significance
        let memory_id = memory_formation.form_memory(&episode);

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

        memory_formation.form_memory(&episode1);
        memory_formation.form_memory(&episode2);
        memory_formation.form_memory(&episode3);

        let agent1_memories = memory_formation.retrieve_by_agent(1, 10);
        assert_eq!(agent1_memories.len(), 2);

        let agent2_memories = memory_formation.retrieve_by_agent(2, 10);
        assert_eq!(agent2_memories.len(), 1);
    }

    #[test]
    fn test_memory_strength_boost_on_access() {
        let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());

        let episode = create_test_episode(1, 1, 0.8);
        let memory_id = memory_formation.form_memory(&episode).unwrap();

        let initial_strength = memory_formation.get_memory(memory_id).unwrap().strength;

        // Access memory multiple times
        let context = EventContext {
            environment: EnvironmentState {
                variables: HashMap::new(),
                spatial: None,
                temporal: TemporalContext {
                    time_of_day: None,
                    deadlines: vec![],
                    patterns: vec![],
                },
            },
            active_goals: vec![],
            resources: ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 50.0,
                    memory_bytes: 1024 * 1024 * 1024,
                    storage_bytes: 10 * 1024 * 1024 * 1024,
                    network_bandwidth: 1000 * 1000,
                },
                external: HashMap::new(),
            },
            fingerprint: 12345,
            embeddings: None,
        };

        memory_formation.retrieve_by_context(&context, 10);
        memory_formation.retrieve_by_context(&context, 10);

        let final_strength = memory_formation.get_memory(memory_id).unwrap().strength;
        assert!(final_strength > initial_strength);
    }
}
