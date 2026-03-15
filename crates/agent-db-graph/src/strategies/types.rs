// crates/agent-db-graph/src/strategies/types.rs
//
// Type definitions, constants, and configuration for strategy extraction.

use crate::episodes::EpisodeOutcome;
use agent_db_core::types::{AgentId, ContextHash, Timestamp};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a strategy
pub type StrategyId = u64;

// Piecewise outcome scoring constants (same keys/values as claims/memories for consistency)
pub(crate) const META_POSITIVE_OUTCOMES: &str = "_positive_outcomes";
pub(crate) const META_NEGATIVE_OUTCOMES: &str = "_negative_outcomes";
pub(crate) const META_Q_VALUE: &str = "_q_value";
pub(crate) const Q_KICK_IN: u32 = 5;
pub(crate) const Q_ALPHA: f32 = 0.3;

/// Compute a deterministic goal bucket id from a set of goal ids.
///
/// Matches the algorithm used by `EventContext::compute_goal_bucket_id()`:
/// sort goal ids ascending, serialize each as u64 LE, BLAKE3 hash, take
/// first 8 bytes as u64 LE. Priority is deliberately excluded so that
/// minor priority changes do not split buckets.
///
/// Returns 0 when the input slice is empty.
pub fn compute_goal_bucket_id_from_ids(goal_ids: &[u64]) -> u64 {
    if goal_ids.is_empty() {
        return 0;
    }
    let mut sorted = goal_ids.to_vec();
    sorted.sort();
    let mut canonical_bytes = Vec::with_capacity(sorted.len() * 8);
    for id in &sorted {
        canonical_bytes.extend_from_slice(&id.to_le_bytes());
    }
    let hash = blake3::hash(&canonical_bytes);
    let bytes: [u8; 8] = hash.as_bytes()[0..8].try_into().unwrap();
    u64::from_le_bytes(bytes)
}

#[derive(Debug, Clone)]
pub struct StrategyUpsert {
    pub id: StrategyId,
    pub is_new: bool,
}

/// Type of strategy (positive or constraint)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StrategyType {
    /// Success-correlated strategy (do this)
    Positive,
    /// Failure-correlated constraint (avoid this)
    Constraint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MotifClass {
    Transition,
    Anchor,
    Macro,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MotifStats {
    pub success_count: u32,
    pub failure_count: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct EpisodeMotifRecord {
    pub outcome: EpisodeOutcome,
    pub motifs: HashSet<String>,
}

/// A generalizable strategy extracted from successful experiences
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Strategy {
    /// Unique strategy identifier
    pub id: StrategyId,

    /// Human-readable strategy name
    pub name: String,

    // ========== LLM-Retrievable Fields (10x) ==========
    /// Natural language summary of this strategy
    #[serde(default)]
    pub summary: String,

    /// Natural language description of when to use this strategy
    #[serde(default)]
    pub when_to_use: String,

    /// Conditions where this strategy should NOT be used
    #[serde(default)]
    pub when_not_to_use: String,

    /// Known failure modes and how to recover
    #[serde(default)]
    pub failure_modes: Vec<String>,

    /// Executable playbook — structured steps with branching and recovery (100x)
    #[serde(default)]
    pub playbook: Vec<PlaybookStep>,

    /// What would have happened if a different approach was taken
    #[serde(default)]
    pub counterfactual: String,

    /// Strategy IDs this one supersedes (makes obsolete)
    #[serde(default)]
    pub supersedes: Vec<StrategyId>,

    /// Domains where this strategy has been shown to work
    #[serde(default)]
    pub applicable_domains: Vec<String>,

    /// How many generations of evolution this strategy has gone through
    #[serde(default)]
    pub lineage_depth: u32,

    /// Embedding of the summary text for semantic retrieval
    #[serde(default)]
    pub summary_embedding: Vec<f32>,

    // ========== Original Fields ==========
    /// Agent that created/used this strategy
    pub agent_id: AgentId,

    /// Reasoning steps that define this strategy
    pub reasoning_steps: Vec<ReasoningStep>,

    /// Context patterns where this strategy applies
    pub context_patterns: Vec<ContextPattern>,

    /// Success indicators
    pub success_indicators: Vec<String>,

    /// Failure patterns to avoid
    pub failure_patterns: Vec<String>,

    /// Strategy confidence/quality (0.0 to 1.0)
    pub quality_score: f32,

    /// Number of successful applications
    pub success_count: u32,

    /// Number of failed applications
    pub failure_count: u32,

    /// Total support count (episodes supporting this strategy)
    pub support_count: u32,

    /// Strategy type
    pub strategy_type: StrategyType,

    /// Precondition (when this applies)
    pub precondition: String,

    /// Action/policy hint (what to do or avoid)
    pub action_hint: String,

    /// Expected success with Bayesian smoothing
    pub expected_success: f32,

    /// Expected cost proxy (steps/time/tokens)
    pub expected_cost: f32,

    /// Calibrated utility (uplift or negative for constraints)
    pub expected_value: f32,

    /// Confidence derived from support and variance
    pub confidence: f32,

    /// Near contexts where it failed (contradictions)
    pub contradictions: Vec<String>,

    /// Goal bucket identifier for retrieval
    pub goal_bucket_id: u64,

    /// Behavior signature (motif)
    pub behavior_signature: String,

    /// Source episodes this was extracted from
    /// Note: Not persisted to storage (episodes stored separately in EpisodeCatalog)
    #[serde(skip)]
    pub source_episodes: Vec<crate::episodes::Episode>,

    /// When this strategy was created
    pub created_at: Timestamp,

    /// Last time this strategy was used
    pub last_used: Timestamp,

    /// Metadata
    pub metadata: HashMap<String, String>,

    // ========== Phase 1 Upgrades (Features J & K) ==========
    /// Self-judged quality score from the agent (0.0 to 1.0)
    /// None if agent hasn't provided self-assessment
    pub self_judged_quality: Option<f32>,

    /// Outcomes from source episodes (for tracking success/failure patterns)
    pub source_outcomes: Vec<EpisodeOutcome>,

    /// Strategy version number (increments when refined)
    pub version: u32,

    /// Parent strategy ID if this was refined from another strategy
    pub parent_strategy: Option<StrategyId>,
}

/// A single step in an executable playbook
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PlaybookStep {
    /// Step number (1-indexed)
    pub step: u32,

    /// What to do
    pub action: String,

    /// Prerequisite condition (when to execute this step)
    #[serde(default)]
    pub condition: String,

    /// Condition under which this step should be skipped
    #[serde(default)]
    pub skip_if: String,

    /// Branching conditions: maps a condition string to an alternative action
    #[serde(default)]
    pub branches: Vec<PlaybookBranch>,

    /// Recovery instruction if this step fails
    #[serde(default)]
    pub recovery: String,

    /// Unique identifier for this step (for non-linear playbook navigation)
    #[serde(default)]
    pub step_id: String,

    /// ID of the step to execute after this one (overrides sequential ordering)
    #[serde(default)]
    pub next_step_id: Option<String>,
}

/// A branch in a playbook step
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlaybookBranch {
    /// The condition that triggers this branch
    pub condition: String,
    /// What to do when this condition is met
    pub action: String,
    /// ID of the step to jump to when this branch is taken
    #[serde(default)]
    pub next_step_id: Option<String>,
}

/// A single reasoning step in a strategy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReasoningStep {
    /// Step description
    pub description: String,

    /// When this step applies
    pub applicability: String,

    /// Expected outcome
    pub expected_outcome: Option<String>,

    /// Order in sequence
    pub sequence_order: usize,
}

/// Pattern describing when a strategy is applicable
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextPattern {
    /// Environment characteristics
    pub environment_type: Option<String>,

    /// Task type
    pub task_type: Option<String>,

    /// Resource constraints
    pub resource_constraints: Vec<String>,

    /// Goal characteristics
    pub goal_characteristics: Vec<String>,

    /// Pattern match confidence
    pub match_confidence: f32,
}

/// Configuration for strategy extraction
#[derive(Debug, Clone)]
pub struct StrategyExtractionConfig {
    /// Minimum episode significance to extract strategies
    pub min_significance: f32,

    /// Minimum success rate to create strategy
    pub min_success_rate: f32,

    /// Minimum occurrences before extraction
    pub min_occurrences: u32,

    /// Maximum strategies to store per agent
    pub max_strategies_per_agent: usize,

    /// Eligibility threshold (0.0 to 1.0)
    pub eligibility_threshold: f32,

    /// Minimum events in an episode to consider for strategy extraction
    pub min_episode_events: usize,

    /// Minimum quality score to accept a new strategy
    pub min_quality_score: f32,

    /// Maximum strategies allowed per goal bucket before quality-gating
    pub max_strategies_per_bucket: usize,

    /// Eligibility weights
    pub w_novelty: f32,
    pub w_outcome_utility: f32,
    pub w_difficulty: f32,
    pub w_reuse_potential: f32,
    pub w_redundancy: f32,

    /// Distiller settings
    pub distill_every: u32,
    pub motif_window_k: usize,
    pub min_support_success: u32,
    pub min_support_failure: u32,
    pub min_lift: f32,
    pub min_uplift: f32,
    pub min_holdout_coverage: u32,
    pub holdout_size: usize,
    pub cache_max: usize,
    pub alpha: f32,
    pub beta: f32,
    pub drift_max_drop: f32,
    pub conflict_margin: f32,
}

impl Default for StrategyExtractionConfig {
    fn default() -> Self {
        Self {
            min_significance: 0.6,
            min_success_rate: 0.7,
            min_occurrences: 3,
            max_strategies_per_agent: 100,
            eligibility_threshold: 0.6,
            min_episode_events: 3,
            min_quality_score: 0.65,
            max_strategies_per_bucket: 15,
            w_novelty: 0.25,
            w_outcome_utility: 0.25,
            w_difficulty: 0.2,
            w_reuse_potential: 0.2,
            w_redundancy: 0.25,
            distill_every: 5,
            motif_window_k: 2,
            min_support_success: 5,
            min_support_failure: 5,
            min_lift: 1.0,
            min_uplift: 0.10,
            min_holdout_coverage: 3,
            holdout_size: 10,
            cache_max: 50,
            alpha: 1.0,
            beta: 1.0,
            drift_max_drop: 0.25,
            conflict_margin: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategySimilarityQuery {
    pub goal_ids: Vec<u64>,
    pub tool_names: Vec<String>,
    pub result_types: Vec<String>,
    pub context_hash: Option<ContextHash>,
    pub agent_id: Option<AgentId>,
    pub min_score: f32,
    pub limit: usize,
}

/// Statistics about strategy extraction
#[derive(Debug, Clone)]
pub struct StrategyStats {
    /// Total number of strategies
    pub total_strategies: usize,

    /// Strategies with quality > 0.8
    pub high_quality_strategies: usize,

    /// Number of agents with strategies
    pub agents_with_strategies: usize,

    /// Average quality across all strategies
    pub average_quality: f32,
}

/// Compute word-level Jaccard similarity between two strings.
///
/// Splits on whitespace, compares as sets. Returns 0.0 if both are empty.
pub(crate) fn word_jaccard(a: &str, b: &str) -> f32 {
    let a_words: HashSet<&str> = a.split_whitespace().collect();
    let b_words: HashSet<&str> = b.split_whitespace().collect();
    let intersection = a_words.intersection(&b_words).count() as f32;
    let union = a_words.union(&b_words).count() as f32;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

pub(crate) fn outcome_to_counts(outcome: Option<&EpisodeOutcome>) -> (u32, u32) {
    match outcome {
        Some(EpisodeOutcome::Success) | Some(EpisodeOutcome::Partial) => (1, 0),
        Some(EpisodeOutcome::Failure) => (0, 1),
        Some(EpisodeOutcome::Interrupted) | None => (0, 0),
    }
}
