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

/// How Phase 2 groups semantic memories into schemas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchemaGroupingMode {
    /// Group by exact context fingerprint (original behaviour).
    ExactFingerprint,
    /// Greedy centroid-based embedding clustering.
    #[default]
    EmbeddingCentroid,
    /// Complete-link (mutual) embedding clustering — every member must meet threshold against every other member.
    EmbeddingMutual,
}

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
    /// Maximum number of semantic memories created per goal bucket per consolidation pass (Phase 1)
    pub max_semantics_per_bucket_per_pass: usize,
    /// How Phase 2 groups semantic memories into schemas
    pub schema_grouping_mode: SchemaGroupingMode,
    /// Cosine similarity threshold for embedding-based schema grouping
    pub schema_similarity_threshold: f32,
    /// Maximum semantic candidates scanned during embedding schema grouping
    pub max_schema_candidates: usize,
    /// Maximum members in a single schema group
    pub max_schema_group_size: usize,
    /// Maximum schema groups created per consolidation pass
    pub max_schema_groups_per_pass: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            max_semantics_per_bucket_per_pass: 3,
            schema_grouping_mode: SchemaGroupingMode::default(),
            schema_similarity_threshold: 0.80,
            max_schema_candidates: 200,
            max_schema_group_size: 50,
            max_schema_groups_per_pass: 5,
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

        // --- Phase 1: Episodic → Semantic (multi-semantic chunking) ---
        let all_memories = store.list_all_memories();
        let episodic = Self::filter_tier(&all_memories, &MemoryTier::Episodic);
        let grouped = Self::group_by_goal_bucket(&episodic);

        // Collect all Phase 1 operations, then flush in batch
        let mut phase1_memories: Vec<Memory> = Vec::new();
        let mut phase1_marks: Vec<(MemoryId, MemoryId, MemoryTier, f32)> = Vec::new();

        for (goal_bucket_id, memories) in &grouped {
            // Only active, non-consolidated episodic memories
            let mut eligible: Vec<&Memory> = memories
                .iter()
                .filter(|m| m.consolidation_status == ConsolidationStatus::Active)
                .copied()
                .collect();

            if eligible.len() < self.config.episodic_threshold {
                continue;
            }

            // Sort by (formed_at asc, id asc) for deterministic chunking
            eligible.sort_by(|a, b| a.formed_at.cmp(&b.formed_at).then(a.id.cmp(&b.id)));

            // Chunk into groups of episodic_threshold; only full chunks are processed
            let threshold = self.config.episodic_threshold;
            let full_chunks = eligible.len() / threshold;
            let chunks_to_process = full_chunks.min(self.config.max_semantics_per_bucket_per_pass);

            for chunk_idx in 0..chunks_to_process {
                let start = chunk_idx * threshold;
                let chunk = &eligible[start..start + threshold];

                let semantic = self.synthesize_semantic(chunk, *goal_bucket_id);
                let semantic_id = semantic.id;
                let episode_ids: Vec<MemoryId> = chunk.iter().map(|m| m.id).collect();

                phase1_memories.push(semantic);

                for &mid in &episode_ids {
                    phase1_marks.push((
                        mid,
                        semantic_id,
                        MemoryTier::Semantic,
                        self.config.post_consolidation_decay,
                    ));
                }
                result.consolidated_episode_ids.extend(episode_ids);
                result.semantic_created += 1;
            }
        }

        // Single atomic batch write for all Phase 1 operations
        if !phase1_memories.is_empty() {
            store.store_consolidated_memories_batch(phase1_memories);
        }
        if !phase1_marks.is_empty() {
            store.mark_consolidated_batch(phase1_marks);
        }

        // --- Phase 2: Semantic → Schema ---
        let all_memories = store.list_all_memories();
        let semantics = Self::filter_tier(&all_memories, &MemoryTier::Semantic);
        let sem_grouped = Self::group_by_goal_bucket(&semantics);

        // Collect all Phase 2 operations, then flush in batch
        let mut phase2_memories: Vec<Memory> = Vec::new();
        let mut phase2_marks: Vec<(MemoryId, MemoryId, MemoryTier, f32)> = Vec::new();

        for (goal_bucket_id, memories) in &sem_grouped {
            let eligible: Vec<&Memory> = memories
                .iter()
                .filter(|m| m.consolidation_status == ConsolidationStatus::Active)
                .copied()
                .collect();

            if eligible.len() < self.config.semantic_threshold {
                continue;
            }

            let groups = match self.config.schema_grouping_mode {
                SchemaGroupingMode::ExactFingerprint => self.group_by_fingerprint(&eligible),
                SchemaGroupingMode::EmbeddingCentroid => {
                    self.group_by_embedding_centroid(&eligible)
                },
                SchemaGroupingMode::EmbeddingMutual => self.group_by_embedding_mutual(&eligible),
            };

            let groups_cap = groups
                .into_iter()
                .take(self.config.max_schema_groups_per_pass);

            for group in groups_cap {
                if group.len() < self.config.semantic_threshold {
                    continue;
                }

                let group_refs: Vec<&Memory> = group.to_vec();
                let mut schema = self.synthesize_schema(&group_refs, *goal_bucket_id);

                // Annotate schema metadata with grouping info
                schema.metadata.insert(
                    "grouping_mode".to_string(),
                    format!("{:?}", self.config.schema_grouping_mode),
                );
                schema.metadata.insert(
                    "similarity_threshold".to_string(),
                    self.config.schema_similarity_threshold.to_string(),
                );
                if let Some(seed) = group.first() {
                    schema
                        .metadata
                        .insert("seed_memory_id".to_string(), seed.id.to_string());
                }
                schema
                    .metadata
                    .insert("member_count".to_string(), group.len().to_string());
                let member_ids: Vec<String> = group.iter().map(|m| m.id.to_string()).collect();
                schema
                    .metadata
                    .insert("member_ids".to_string(), member_ids.join(","));

                let schema_id = schema.id;
                let sem_ids: Vec<MemoryId> = group.iter().map(|m| m.id).collect();

                phase2_memories.push(schema);

                for &mid in &sem_ids {
                    phase2_marks.push((
                        mid,
                        schema_id,
                        MemoryTier::Schema,
                        self.config.post_consolidation_decay,
                    ));
                }
                result.consolidated_semantic_ids.extend(sem_ids);
                result.schema_created += 1;
            }
        }

        // Single atomic batch write for all Phase 2 operations
        if !phase2_memories.is_empty() {
            store.store_consolidated_memories_batch(phase2_memories);
        }
        if !phase2_marks.is_empty() {
            store.mark_consolidated_batch(phase2_marks);
        }

        result
    }

    // ---- Phase 2 grouping strategies ----

    /// ExactFingerprint: group semantics by their context fingerprint (original behaviour).
    fn group_by_fingerprint<'a>(&self, eligible: &[&'a Memory]) -> Vec<Vec<&'a Memory>> {
        let mut fp_groups: HashMap<u64, Vec<&'a Memory>> = HashMap::new();
        for m in eligible {
            fp_groups.entry(m.context.fingerprint).or_default().push(m);
        }
        fp_groups.into_values().collect()
    }

    /// EmbeddingCentroid: greedy centroid-based clustering.
    /// Seed ordering: strength desc, formed_at desc, id desc.
    fn group_by_embedding_centroid<'a>(&self, eligible: &[&'a Memory]) -> Vec<Vec<&'a Memory>> {
        let mut sorted: Vec<&'a Memory> = eligible.to_vec();
        // Seed ordering: strength desc, formed_at desc, id desc
        sorted.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.formed_at.cmp(&a.formed_at))
                .then(b.id.cmp(&a.id))
        });

        let mut assigned: Vec<bool> = vec![false; sorted.len()];
        let mut groups: Vec<Vec<&'a Memory>> = Vec::new();

        for seed_idx in 0..sorted.len() {
            if assigned[seed_idx] {
                continue;
            }
            if groups.len() >= self.config.max_schema_groups_per_pass {
                break;
            }

            let seed = sorted[seed_idx];
            let seed_emb = Self::get_embedding(seed);
            if seed_emb.is_empty() {
                continue;
            }

            assigned[seed_idx] = true;
            let mut group: Vec<&'a Memory> = vec![seed];
            let mut centroid = seed_emb.clone();
            let mut scanned = 0usize;

            for cand_idx in 0..sorted.len() {
                if assigned[cand_idx] || cand_idx == seed_idx {
                    continue;
                }
                if scanned >= self.config.max_schema_candidates {
                    break;
                }
                if group.len() >= self.config.max_schema_group_size {
                    break;
                }
                scanned += 1;

                let cand = sorted[cand_idx];
                let cand_emb = Self::get_embedding(cand);
                if cand_emb.is_empty() {
                    continue;
                }

                let sim_centroid = cosine_similarity(&centroid, &cand_emb);
                let sim_seed = cosine_similarity(&seed_emb, &cand_emb);

                if sim_centroid >= self.config.schema_similarity_threshold
                    && sim_seed >= self.config.schema_similarity_threshold
                {
                    // Update centroid as running mean
                    let n = group.len() as f32;
                    centroid = centroid
                        .iter()
                        .zip(cand_emb.iter())
                        .map(|(c, e)| (c * n + e) / (n + 1.0))
                        .collect();
                    assigned[cand_idx] = true;
                    group.push(cand);
                }
            }

            groups.push(group);
        }

        groups
    }

    /// EmbeddingMutual: complete-link clustering.
    /// Candidate must meet threshold against every existing member.
    fn group_by_embedding_mutual<'a>(&self, eligible: &[&'a Memory]) -> Vec<Vec<&'a Memory>> {
        let mut sorted: Vec<&'a Memory> = eligible.to_vec();
        sorted.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.formed_at.cmp(&a.formed_at))
                .then(b.id.cmp(&a.id))
        });

        let mut assigned: Vec<bool> = vec![false; sorted.len()];
        let mut groups: Vec<Vec<&'a Memory>> = Vec::new();

        // Pre-compute embeddings
        let embeddings: Vec<Vec<f32>> = sorted.iter().map(|m| Self::get_embedding(m)).collect();

        for seed_idx in 0..sorted.len() {
            if assigned[seed_idx] {
                continue;
            }
            if groups.len() >= self.config.max_schema_groups_per_pass {
                break;
            }
            if embeddings[seed_idx].is_empty() {
                continue;
            }

            assigned[seed_idx] = true;
            let mut group_indices: Vec<usize> = vec![seed_idx];
            let mut scanned = 0usize;

            for cand_idx in 0..sorted.len() {
                if assigned[cand_idx] || cand_idx == seed_idx {
                    continue;
                }
                if scanned >= self.config.max_schema_candidates {
                    break;
                }
                if group_indices.len() >= self.config.max_schema_group_size {
                    break;
                }
                scanned += 1;

                if embeddings[cand_idx].is_empty() {
                    continue;
                }

                // Must meet threshold against ALL current members
                let meets_all = group_indices.iter().all(|&member_idx| {
                    cosine_similarity(&embeddings[member_idx], &embeddings[cand_idx])
                        >= self.config.schema_similarity_threshold
                });

                if meets_all {
                    assigned[cand_idx] = true;
                    group_indices.push(cand_idx);
                }
            }

            let group: Vec<&'a Memory> = group_indices.iter().map(|&i| sorted[i]).collect();
            groups.push(group);
        }

        groups
    }

    /// Get the best available embedding for a memory.
    /// Prefers summary_embedding; falls back to context.embeddings.
    fn get_embedding(memory: &Memory) -> Vec<f32> {
        if !memory.summary_embedding.is_empty() {
            return memory.summary_embedding.clone();
        }
        if let Some(ref emb) = memory.context.embeddings {
            if !emb.is_empty() {
                return emb.clone();
            }
        }
        Vec::new()
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
            groups.entry(m.context.goal_bucket_id).or_default().push(m);
        }
        groups
    }
}

/// Cosine similarity between two vectors. Returns 0.0 if either is empty or zero-magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
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

    fn make_episodic_memory(id: MemoryId, goal_bucket: u64, outcome: EpisodeOutcome) -> Memory {
        let ctx = EventContext {
            fingerprint: goal_bucket, // Legacy: use same value as fallback
            goal_bucket_id: goal_bucket,
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
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // Insert 4 episodic memories with the same fingerprint (goal bucket)
        // With chunking: threshold=3, so only 1 full chunk of 3 is processed (4th is leftover)
        for i in 1..=4 {
            let mut mem = make_episodic_memory(i, 999, EpisodeOutcome::Success);
            mem.formed_at = 1000 + i; // Distinct formed_at for deterministic ordering
            store.store_consolidated_memory(mem);
        }

        assert_eq!(store.list_all_memories().len(), 4);

        let mut engine = ConsolidationEngine::new(config, 100);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(
            result.semantic_created, 1,
            "Should create 1 semantic memory (1 full chunk of 3)"
        );
        assert_eq!(
            result.consolidated_episode_ids.len(),
            3,
            "Should mark only 3 episodic memories (1 full chunk), leaving 1 leftover"
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
            .contains("Generalized from 3 episodes"));
        assert_eq!(semantic_mems[0].consolidated_from.len(), 3);

        // Check the consolidated episodic memories have schema_id = None
        let consolidated_ids = &result.consolidated_episode_ids;
        for &mid in consolidated_ids {
            let m = store.get_memory(mid).unwrap();
            assert_eq!(m.consolidation_status, ConsolidationStatus::Consolidated);
            assert_eq!(
                m.schema_id, None,
                "Episodic memories consolidated into Semantic must NOT have schema_id set"
            );
        }

        // The leftover (4th) episodic memory should still be Active
        let leftover: Vec<&Memory> = all
            .iter()
            .filter(|m| {
                m.tier == MemoryTier::Episodic
                    && m.consolidation_status == ConsolidationStatus::Active
            })
            .collect();
        assert_eq!(
            leftover.len(),
            1,
            "1 leftover episodic should remain active"
        );
    }

    #[test]
    fn test_consolidation_semantic_to_schema() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 2,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            // Use ExactFingerprint so all 3 semantics (same fingerprint) group together
            schema_grouping_mode: SchemaGroupingMode::ExactFingerprint,
            ..Default::default()
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

        let created_schema_id = schemas[0].id;

        // Schema memory itself should have schema_id = None
        assert_eq!(
            schemas[0].schema_id, None,
            "Schema memory itself should have schema_id = None"
        );

        // Semantic memories consolidated into the schema must have schema_id = Some(schema_id)
        for i in 50..53u64 {
            let m = store.get_memory(i).unwrap();
            assert_eq!(m.consolidation_status, ConsolidationStatus::Consolidated);
            assert_eq!(
                m.schema_id,
                Some(created_schema_id),
                "Semantic memories consolidated into Schema must have schema_id set to the schema's ID"
            );
        }
    }

    #[test]
    fn test_full_pipeline_schema_id_correctness() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            // With chunking: 9 episodics / threshold 3 = 3 full chunks, capped at max_semantics_per_bucket_per_pass=3
            schema_grouping_mode: SchemaGroupingMode::ExactFingerprint,
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // Create 9 episodic memories, same goal bucket
        for i in 1..=9u64 {
            let mut mem = make_episodic_memory(i, 42, EpisodeOutcome::Success);
            mem.formed_at = 1000 + i;
            store.store_consolidated_memory(mem);
        }

        let mut engine = ConsolidationEngine::new(config.clone(), 100);

        // Single pass: Phase 1 creates 3 semantics, Phase 2 immediately creates a schema from them
        let result = engine.run_consolidation(&mut store);
        assert_eq!(
            result.semantic_created, 3,
            "Should create 3 semantics from 9 episodics (3 chunks of 3)"
        );
        assert_eq!(
            result.schema_created, 1,
            "Should create 1 schema from 3 semantics in same pass"
        );

        let all = store.list_all_memories();
        let schemas: Vec<&Memory> = all
            .iter()
            .filter(|m| m.tier == MemoryTier::Schema)
            .collect();
        assert_eq!(schemas.len(), 1);
        let schema_id = schemas[0].id;

        // All 3 semantic memories should have schema_id set (consolidated in same pass)
        for sem in all.iter().filter(|m| m.tier == MemoryTier::Semantic) {
            assert_eq!(sem.consolidation_status, ConsolidationStatus::Consolidated);
            assert_eq!(
                sem.schema_id,
                Some(schema_id),
                "Semantic memories must point to the schema after Phase 2"
            );
        }

        // Episodic memories: consolidated, but schema_id must remain None
        for ep in all.iter().filter(|m| m.tier == MemoryTier::Episodic) {
            assert_eq!(ep.consolidation_status, ConsolidationStatus::Consolidated);
            assert_eq!(
                ep.schema_id, None,
                "Episodic memories must never get schema_id set"
            );
        }
    }

    #[test]
    fn test_consolidation_groups_by_goal_bucket_not_fingerprint() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // Create 4 episodic memories with SAME goal_bucket_id but DIFFERENT fingerprints.
        // This simulates same goals pursued across different environments.
        let shared_bucket = 777u64;
        for i in 1..=4u64 {
            let ctx = EventContext {
                fingerprint: 1000 + i,         // Each has a unique fingerprint
                goal_bucket_id: shared_bucket, // But same goal bucket
                ..Default::default()
            };
            let mem = Memory {
                id: i,
                agent_id: 1,
                session_id: 100,
                episode_id: i,
                summary: format!("Episode {} happened", i),
                takeaway: format!("Lesson from episode {}", i),
                causal_note: format!("Because of X in episode {}", i),
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
                outcome: EpisodeOutcome::Success,
                memory_type: MemoryType::Episodic { significance: 0.8 },
                metadata: HashMap::new(),
            };
            store.store_consolidated_memory(mem);
        }

        let mut engine = ConsolidationEngine::new(config, 100);
        let result = engine.run_consolidation(&mut store);

        // With chunking: 4 episodics / threshold 3 = 1 full chunk of 3
        assert_eq!(
            result.semantic_created, 1,
            "Memories with same goal_bucket_id but different fingerprints must consolidate"
        );
        assert_eq!(
            result.consolidated_episode_ids.len(),
            3,
            "Only 1 full chunk of 3 processed"
        );
    }

    #[test]
    fn test_different_goal_buckets_do_not_consolidate() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // 2 memories in bucket A, 2 in bucket B (threshold is 3, so neither consolidates)
        for i in 1..=2u64 {
            let mem = make_episodic_memory(i, 100, EpisodeOutcome::Success);
            store.store_consolidated_memory(mem);
        }
        for i in 3..=4u64 {
            let mem = make_episodic_memory(i, 200, EpisodeOutcome::Success);
            store.store_consolidated_memory(mem);
        }

        let mut engine = ConsolidationEngine::new(config, 100);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(
            result.semantic_created, 0,
            "Different goal buckets must not merge"
        );
    }

    #[test]
    fn test_multi_semantic_and_schema_in_one_run() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            max_semantics_per_bucket_per_pass: 3,
            schema_grouping_mode: SchemaGroupingMode::ExactFingerprint,
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // 9 episodics with same goal bucket -> 3 semantics (3 chunks of 3) -> 1 schema
        for i in 1..=9u64 {
            let mut mem = make_episodic_memory(i, 42, EpisodeOutcome::Success);
            mem.formed_at = 1000 + i;
            store.store_consolidated_memory(mem);
        }

        let mut engine = ConsolidationEngine::new(config, 100);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(
            result.semantic_created, 3,
            "9 episodics / 3 threshold = 3 semantics"
        );
        assert_eq!(
            result.consolidated_episode_ids.len(),
            9,
            "All 9 episodics consumed"
        );
        assert_eq!(
            result.schema_created, 1,
            "3 semantics >= threshold 3 = 1 schema"
        );
        assert_eq!(
            result.consolidated_semantic_ids.len(),
            3,
            "All 3 semantics consumed"
        );

        let all = store.list_all_memories();
        let schemas: Vec<&Memory> = all
            .iter()
            .filter(|m| m.tier == MemoryTier::Schema)
            .collect();
        assert_eq!(schemas.len(), 1);
    }

    fn make_semantic_with_embedding(
        id: MemoryId,
        goal_bucket: u64,
        fingerprint: u64,
        embedding: Vec<f32>,
    ) -> Memory {
        let ctx = EventContext {
            fingerprint,
            goal_bucket_id: goal_bucket,
            ..Default::default()
        };

        Memory {
            id,
            agent_id: 1,
            session_id: 0,
            episode_id: 0,
            summary: format!("Semantic memory {}", id),
            takeaway: format!("Pattern {}", id),
            causal_note: String::new(),
            summary_embedding: embedding,
            tier: MemoryTier::Semantic,
            consolidated_from: vec![id * 10, id * 10 + 1],
            schema_id: None,
            consolidation_status: ConsolidationStatus::Active,
            context: ctx,
            key_events: Vec::new(),
            strength: 0.8,
            relevance_score: 0.7,
            formed_at: 1000 + id,
            last_accessed: 1000 + id,
            access_count: 0,
            outcome: EpisodeOutcome::Success,
            memory_type: MemoryType::Semantic,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_embedding_schema_grouping_centroid() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            schema_grouping_mode: SchemaGroupingMode::EmbeddingCentroid,
            schema_similarity_threshold: 0.80,
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // 3 semantics with similar embeddings but DIFFERENT fingerprints
        // These should cluster via embedding similarity, NOT fingerprint
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.98, 0.1, 0.0]; // cosine ~0.995 with emb1
        let emb3 = vec![0.95, 0.15, 0.0]; // cosine ~0.988 with emb1

        store.store_consolidated_memory(make_semantic_with_embedding(50, 42, 1000, emb1));
        store.store_consolidated_memory(make_semantic_with_embedding(51, 42, 2000, emb2));
        store.store_consolidated_memory(make_semantic_with_embedding(52, 42, 3000, emb3));

        let mut engine = ConsolidationEngine::new(config, 200);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(
            result.schema_created, 1,
            "Similar embeddings should form 1 schema via centroid clustering"
        );
        assert_eq!(result.consolidated_semantic_ids.len(), 3);

        let all = store.list_all_memories();
        let schemas: Vec<&Memory> = all
            .iter()
            .filter(|m| m.tier == MemoryTier::Schema)
            .collect();
        assert_eq!(schemas.len(), 1);
        assert_eq!(
            schemas[0].metadata.get("grouping_mode").unwrap(),
            "EmbeddingCentroid"
        );
        assert_eq!(schemas[0].metadata.get("member_count").unwrap(), "3");
    }

    #[test]
    fn test_embedding_schema_grouping_mutual() {
        use crate::stores::InMemoryMemoryStore;

        let config = ConsolidationConfig {
            episodic_threshold: 3,
            semantic_threshold: 3,
            post_consolidation_decay: 0.3,
            archive_after_consolidation: false,
            schema_grouping_mode: SchemaGroupingMode::EmbeddingMutual,
            schema_similarity_threshold: 0.80,
            ..Default::default()
        };

        let mut store = InMemoryMemoryStore::new(MemoryFormationConfig::default());

        // 3 semantics with similar embeddings but different fingerprints
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.98, 0.1, 0.0];
        let emb3 = vec![0.95, 0.15, 0.0];

        store.store_consolidated_memory(make_semantic_with_embedding(50, 42, 1000, emb1));
        store.store_consolidated_memory(make_semantic_with_embedding(51, 42, 2000, emb2));
        store.store_consolidated_memory(make_semantic_with_embedding(52, 42, 3000, emb3));

        let mut engine = ConsolidationEngine::new(config, 200);
        let result = engine.run_consolidation(&mut store);

        assert_eq!(
            result.schema_created, 1,
            "Similar embeddings should form 1 schema via mutual clustering"
        );
        assert_eq!(result.consolidated_semantic_ids.len(), 3);

        let all = store.list_all_memories();
        let schemas: Vec<&Memory> = all
            .iter()
            .filter(|m| m.tier == MemoryTier::Schema)
            .collect();
        assert_eq!(schemas.len(), 1);
        assert_eq!(
            schemas[0].metadata.get("grouping_mode").unwrap(),
            "EmbeddingMutual"
        );
    }

    #[test]
    fn test_cosine_similarity_helper() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 1e-6);

        // Empty
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }
}
