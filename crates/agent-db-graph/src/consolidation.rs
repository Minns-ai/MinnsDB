// crates/agent-db-graph/src/consolidation.rs
//
// Memory Consolidation Engine
//
// Implements the 3-layer memory hierarchy:
//   Episodic (raw experience) → Semantic (generalized knowledge) → Schema (reusable mental model)
//
// Consolidation runs periodically and:
//   1. Groups episodic memories by goal bucket
//   2. When N+ episodic memories share a bucket → synthesize a Semantic memory
//   3. When M+ semantic memories overlap → synthesize a Schema memory
//   4. Marks consolidated episodes for accelerated decay

use crate::episodes::EpisodeOutcome;
use crate::memory::{ConsolidationStatus, Memory, MemoryId, MemoryTier, MemoryType};
use crate::stores::MemoryStore;
use agent_db_core::types::current_timestamp;
use std::collections::HashMap;

/// Configuration for the consolidation engine
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum episodic memories in the same goal bucket before consolidation into Semantic
    pub episodic_threshold: usize,
    /// Minimum semantic memories with overlapping patterns before consolidation into Schema
    pub semantic_threshold: usize,
    /// Strength decay multiplier applied to consolidated episodic memories (0.0–1.0)
    pub post_consolidation_decay: f32,
    /// Whether to archive (hide from retrieval) episodic memories after consolidation
    pub archive_after_consolidation: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
        }
    }
}

/// Result of a consolidation pass
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of new Semantic memories created
    pub semantic_created: usize,
    /// Number of new Schema memories created
    pub schema_created: usize,
    /// IDs of episodic memories that were consolidated (and decayed)
    pub consolidated_episode_ids: Vec<MemoryId>,
    /// IDs of semantic memories that were consolidated into schemas
    pub consolidated_semantic_ids: Vec<MemoryId>,
}

/// The Consolidation Engine that runs the hierarchy pipeline.
pub struct ConsolidationEngine {
    pub config: ConsolidationConfig,
    next_consolidated_id: MemoryId,
}

impl ConsolidationEngine {
    pub fn new(config: ConsolidationConfig, starting_id: MemoryId) -> Self {
        Self {
            config,
            next_consolidated_id: starting_id,
        }
    }

    /// Run a full consolidation pass over the memory store.
    ///
    /// Returns details of what was created / consolidated.
    pub fn run_consolidation(&mut self, store: &mut dyn MemoryStore) -> ConsolidationResult {
        let mut result = ConsolidationResult::default();

        // --- Phase 1: Episodic → Semantic ---
        let all_memories = store.list_all_memories();
        let episodic = Self::filter_tier(&all_memories, &MemoryTier::Episodic);
        let grouped = Self::group_by_goal_bucket(&episodic);

        for (goal_bucket_id, memories) in &grouped {
            // Only active, non-consolidated episodic memories
            let eligible: Vec<&Memory> = memories
                .iter()
                .filter(|m| m.consolidation_status == ConsolidationStatus::Active)
                .copied()
                .collect();

            if eligible.len() < self.config.episodic_threshold {
                continue;
            }

            // Synthesize semantic memory
            let semantic = self.synthesize_semantic(&eligible, *goal_bucket_id);
            let semantic_id = semantic.id;
            let episode_ids: Vec<MemoryId> = eligible.iter().map(|m| m.id).collect();

            store.store_consolidated_memory(semantic);

            // Mark consolidated episodic memories
            for &mid in &episode_ids {
                store.mark_consolidated(mid, semantic_id, self.config.post_consolidation_decay);
            }
            result.consolidated_episode_ids.extend(episode_ids);
            result.semantic_created += 1;
        }

        // --- Phase 2: Semantic → Schema ---
        let all_memories = store.list_all_memories();
        let semantics = Self::filter_tier(&all_memories, &MemoryTier::Semantic);
        let sem_grouped = Self::group_by_goal_bucket(&semantics);

        for (goal_bucket_id, memories) in &sem_grouped {
            let eligible: Vec<&Memory> = memories
                .iter()
                .filter(|m| m.consolidation_status == ConsolidationStatus::Active)
                .copied()
                .collect();

            if eligible.len() < self.config.semantic_threshold {
                continue;
            }

            let schema = self.synthesize_schema(&eligible, *goal_bucket_id);
            let schema_id = schema.id;
            let sem_ids: Vec<MemoryId> = eligible.iter().map(|m| m.id).collect();

            store.store_consolidated_memory(schema);

            for &mid in &sem_ids {
                store.mark_consolidated(mid, schema_id, self.config.post_consolidation_decay);
            }
            result.consolidated_semantic_ids.extend(sem_ids);
            result.schema_created += 1;
        }

        result
    }

    // ---- Synthesis helpers ----

    fn synthesize_semantic(&mut self, memories: &[&Memory], goal_bucket_id: u64) -> Memory {
        let id = self.next_id();
        let agent_id = memories[0].agent_id;

        // Aggregate statistics
        let total_strength: f32 = memories.iter().map(|m| m.strength).sum();
        let avg_strength = total_strength / memories.len() as f32;
        let avg_relevance: f32 =
            memories.iter().map(|m| m.relevance_score).sum::<f32>() / memories.len() as f32;

        // Count outcomes
        let mut success = 0u32;
        let mut failure = 0u32;
        for m in memories {
            match m.outcome {
                EpisodeOutcome::Success => success += 1,
                EpisodeOutcome::Failure => failure += 1,
                _ => {},
            }
        }

        // Build summary by merging individual summaries
        let mut summary_parts = Vec::new();
        summary_parts.push(format!(
            "Generalized from {} episodes in goal bucket {}.",
            memories.len(),
            goal_bucket_id
        ));
        summary_parts.push(format!(
            "Success rate: {:.0}% ({} succeeded, {} failed).",
            if success + failure > 0 {
                success as f32 / (success + failure) as f32 * 100.0
            } else {
                0.0
            },
            success,
            failure
        ));

        // Extract common themes from individual takeaways
        let takeaways: Vec<&str> = memories
            .iter()
            .map(|m| m.takeaway.as_str())
            .filter(|t| !t.is_empty())
            .collect();
        if !takeaways.is_empty() {
            summary_parts.push(format!(
                "Key patterns: {}",
                takeaways
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ")
            ));
        }

        // Extract common causal patterns
        let causal_notes: Vec<&str> = memories
            .iter()
            .map(|m| m.causal_note.as_str())
            .filter(|n| !n.is_empty())
            .collect();
        let consolidated_causal = if causal_notes.is_empty() {
            String::new()
        } else {
            format!(
                "Causal patterns across episodes: {}",
                causal_notes
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        };

        let consolidated_takeaway = if success > failure {
            format!(
                "This approach works reliably for goal bucket {} ({:.0}% success across {} episodes).",
                goal_bucket_id,
                success as f32 / (success + failure) as f32 * 100.0,
                memories.len()
            )
        } else {
            format!(
                "This approach needs improvement for goal bucket {} — only {:.0}% success across {} episodes.",
                goal_bucket_id,
                if success + failure > 0 { success as f32 / (success + failure) as f32 * 100.0 } else { 0.0 },
                memories.len()
            )
        };

        let source_ids: Vec<MemoryId> = memories.iter().map(|m| m.id).collect();
        let now = current_timestamp();

        Memory {
            id,
            agent_id,
            session_id: 0, // Consolidated memories span sessions
            episode_id: 0, // Not tied to a single episode
            summary: summary_parts.join(" "),
            takeaway: consolidated_takeaway,
            causal_note: consolidated_causal,
            summary_embedding: Vec::new(),
            tier: MemoryTier::Semantic,
            consolidated_from: source_ids,
            schema_id: None,
            consolidation_status: ConsolidationStatus::Active,
            context: memories[0].context.clone(), // Use first memory's context as template
            key_events: Vec::new(),
            strength: avg_strength.min(1.0),
            relevance_score: avg_relevance.min(1.0),
            formed_at: now,
            last_accessed: now,
            access_count: 0,
            outcome: if success >= failure {
                EpisodeOutcome::Success
            } else {
                EpisodeOutcome::Failure
            },
            memory_type: MemoryType::Semantic,
            metadata: {
                let mut m = HashMap::new();
                m.insert("goal_bucket_id".to_string(), goal_bucket_id.to_string());
                m.insert("source_count".to_string(), memories.len().to_string());
                m.insert("tier".to_string(), "semantic".to_string());
                m
            },
        }
    }

    fn synthesize_schema(&mut self, semantics: &[&Memory], goal_bucket_id: u64) -> Memory {
        let id = self.next_id();
        let agent_id = semantics[0].agent_id;

        let total_source_count: usize = semantics.iter().map(|m| m.consolidated_from.len()).sum();
        let avg_strength: f32 =
            semantics.iter().map(|m| m.strength).sum::<f32>() / semantics.len() as f32;
        let avg_relevance: f32 =
            semantics.iter().map(|m| m.relevance_score).sum::<f32>() / semantics.len() as f32;

        let mut success = 0u32;
        let mut failure = 0u32;
        for m in semantics {
            match m.outcome {
                EpisodeOutcome::Success => success += 1,
                EpisodeOutcome::Failure => failure += 1,
                _ => {},
            }
        }

        let summary = format!(
            "Schema: Reusable mental model for goal bucket {}. Distilled from {} semantic memories covering {} total episodes. \
             Overall success rate: {:.0}%. This schema captures the core pattern that works across varied contexts in this domain.",
            goal_bucket_id,
            semantics.len(),
            total_source_count,
            if success + failure > 0 {
                success as f32 / (success + failure) as f32 * 100.0
            } else {
                0.0
            }
        );

        // Merge takeaways from semantic level
        let merged_takeaways: Vec<&str> = semantics
            .iter()
            .map(|m| m.takeaway.as_str())
            .filter(|t| !t.is_empty())
            .collect();

        let takeaway = if merged_takeaways.is_empty() {
            format!(
                "This is a proven mental model for goal bucket {} with {} supporting episodes.",
                goal_bucket_id, total_source_count
            )
        } else {
            format!(
                "Core insight: {} (validated across {} semantic memories, {} episodes)",
                merged_takeaways.join("; "),
                semantics.len(),
                total_source_count
            )
        };

        let source_ids: Vec<MemoryId> = semantics.iter().map(|m| m.id).collect();
        let now = current_timestamp();

        Memory {
            id,
            agent_id,
            session_id: 0,
            episode_id: 0,
            summary,
            takeaway,
            causal_note: format!(
                "Schema consolidated from {} semantic memories. Represents the dominant causal pattern for goal bucket {}.",
                semantics.len(),
                goal_bucket_id
            ),
            summary_embedding: Vec::new(),
            tier: MemoryTier::Schema,
            consolidated_from: source_ids,
            schema_id: None,
            consolidation_status: ConsolidationStatus::Active,
            context: semantics[0].context.clone(),
            key_events: Vec::new(),
            strength: (avg_strength * 1.2).min(1.0), // Schemas are stronger
            relevance_score: (avg_relevance * 1.1).min(1.0),
            formed_at: now,
            last_accessed: now,
            access_count: 0,
            outcome: if success >= failure {
                EpisodeOutcome::Success
            } else {
                EpisodeOutcome::Failure
            },
            memory_type: MemoryType::Semantic, // Still MemoryType::Semantic (the tier field captures the true level)
            metadata: {
                let mut m = HashMap::new();
                m.insert("goal_bucket_id".to_string(), goal_bucket_id.to_string());
                m.insert("source_count".to_string(), total_source_count.to_string());
                m.insert("semantic_count".to_string(), semantics.len().to_string());
                m.insert("tier".to_string(), "schema".to_string());
                m
            },
        }
    }

    // ---- Utility ----

    fn next_id(&mut self) -> MemoryId {
        let id = self.next_consolidated_id;
        self.next_consolidated_id += 1;
        id
    }

    fn filter_tier<'a>(memories: &'a [Memory], tier: &MemoryTier) -> Vec<&'a Memory> {
        memories.iter().filter(|m| m.tier == *tier).collect()
    }

    fn group_by_goal_bucket<'a>(memories: &[&'a Memory]) -> HashMap<u64, Vec<&'a Memory>> {
        let mut groups: HashMap<u64, Vec<&'a Memory>> = HashMap::new();
        for m in memories {
            let bucket = m.context.fingerprint; // Use context fingerprint as goal bucket proxy
            groups.entry(bucket).or_default().push(m);
        }
        groups
    }
}

/// Strategy evolution engine — detects when a new strategy supersedes an older one.
pub struct StrategyEvolution;

impl StrategyEvolution {
    /// Check if a new strategy supersedes any existing strategies in the same goal bucket.
    /// Returns IDs of strategies that should be marked as superseded.
    pub fn detect_supersession(
        new_strategy: &crate::strategies::Strategy,
        existing: &[crate::strategies::Strategy],
    ) -> Vec<crate::strategies::StrategyId> {
        let mut superseded = Vec::new();

        for old in existing {
            // Same goal bucket, same behavior signature, new one is better
            if old.goal_bucket_id == new_strategy.goal_bucket_id
                && old.behavior_signature == new_strategy.behavior_signature
                && old.id != new_strategy.id
                && new_strategy.quality_score > old.quality_score
                && new_strategy.support_count > old.support_count
            {
                superseded.push(old.id);
            }
        }

        superseded
    }

    /// Calculate the lineage depth (how many generations of evolution).
    pub fn calculate_lineage_depth(
        strategy: &crate::strategies::Strategy,
        all_strategies: &[crate::strategies::Strategy],
    ) -> u32 {
        let mut depth = 0u32;
        let mut current = strategy;
        while let Some(parent_id) = current.parent_strategy {
            if let Some(parent) = all_strategies.iter().find(|s| s.id == parent_id) {
                depth += 1;
                current = parent;
                if depth > 100 {
                    break; // Safety guard
                }
            } else {
                break;
            }
        }
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::EpisodeOutcome;
    use crate::memory::{
        ConsolidationStatus, Memory, MemoryFormationConfig, MemoryTier, MemoryType,
    };
    use agent_db_events::core::EventContext;

    fn make_episodic_memory(id: MemoryId, fingerprint: u64, outcome: EpisodeOutcome) -> Memory {
        let ctx = EventContext {
            fingerprint,
            ..Default::default()
        };

        Memory {
            id,
            agent_id: 1,
            session_id: 100,
            episode_id: id,
            summary: format!("Episode {} happened", id),
            takeaway: format!("Lesson from episode {}", id),
            causal_note: format!("Because of X in episode {}", id),
            summary_embedding: Vec::new(),
            tier: MemoryTier::Episodic,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: ConsolidationStatus::Active,
            context: ctx,
            key_events: Vec::new(),
            strength: 0.8,
            relevance_score: 0.7,
            formed_at: 1000,
            last_accessed: 1000,
            access_count: 0,
            outcome,
            memory_type: MemoryType::Episodic { significance: 0.8 },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_consolidation_episodic_to_semantic() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // Insert 4 episodic memories with the same fingerprint (goal bucket)
        for i in 1..=4 {
            let mem = make_episodic_memory(i, 999, EpisodeOutcome::Success);
            store.store_consolidated_memory(mem);
        }

        assert_eq!(store.list_all_memories().len(), 4);

        let mut engine = ConsolidationEngine::new(config, 100);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(
            result.semantic_created, 1,
            "Should create 1 semantic memory"
        );
        assert_eq!(
            result.consolidated_episode_ids.len(),
            4,
            "Should mark all 4 episodic memories"
        );

        // Check the semantic memory exists
        let all = store.list_all_memories();
        let semantic_mems: Vec<&Memory> = all
            .iter()
            .filter(|m| m.tier == MemoryTier::Semantic)
            .collect();
        assert_eq!(semantic_mems.len(), 1);
        assert!(semantic_mems[0]
            .summary
            .contains("Generalized from 4 episodes"));
        assert_eq!(semantic_mems[0].consolidated_from.len(), 4);

        // Check the episodic memories were marked
        for i in 1..=4u64 {
            let m = store.get_memory(i).unwrap();
            assert_eq!(m.consolidation_status, ConsolidationStatus::Consolidated);
        }
    }

    #[test]
    fn test_consolidation_semantic_to_schema() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 2,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // Create enough episodic memories to produce 3 semantic memories
        // We need 3 groups of 2 episodic memories each, all in the same goal bucket
        // But with different episodes so we get different semantic memories
        //
        // Actually, one group of 6 episodic memories will consolidate into 1 semantic.
        // For 3 semantics, we need 3 runs or manual insertion.
        // Let's just insert semantic memories directly.
        for i in 50..53u64 {
            let mut mem = make_episodic_memory(i, 999, EpisodeOutcome::Success);
            mem.tier = MemoryTier::Semantic;
            mem.summary = format!("Semantic memory {} about goal bucket 999", i);
            mem.takeaway = format!("Pattern {} works reliably", i);
            mem.consolidated_from = vec![i * 10, i * 10 + 1];
            store.store_consolidated_memory(mem);
        }

        let mut engine = ConsolidationEngine::new(config, 200);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(result.schema_created, 1, "Should create 1 schema");
        assert_eq!(result.consolidated_semantic_ids.len(), 3);

        let all = store.list_all_memories();
        let schemas: Vec<&Memory> = all
            .iter()
            .filter(|m| m.tier == MemoryTier::Schema)
            .collect();
        assert_eq!(schemas.len(), 1);
        assert!(schemas[0].summary.contains("Schema: Reusable mental model"));
    }
}
