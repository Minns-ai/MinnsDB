// crates/agent-db-graph/src/strategies.rs
//
// Strategy Extraction Module
//
// Extracts generalizable strategies from successful episodes and reasoning traces,
// enabling agents to reuse proven approaches in similar contexts.

use crate::episodes::{Episode, EpisodeId, EpisodeOutcome};
use crate::GraphResult;
use agent_db_core::types::{current_timestamp, AgentId, ContextHash, Timestamp};
use agent_db_events::core::{ActionOutcome, CognitiveType, Event, EventType, MetadataValue};
use rustc_hash::FxHashMap;
use serde_json::json;
use std::collections::{HashMap, HashSet};

/// Unique identifier for a strategy
pub type StrategyId = u64;

// Piecewise outcome scoring constants (same keys/values as claims/memories for consistency)
const META_POSITIVE_OUTCOMES: &str = "_positive_outcomes";
const META_NEGATIVE_OUTCOMES: &str = "_negative_outcomes";
const META_Q_VALUE: &str = "_q_value";
const Q_KICK_IN: u32 = 5;
const Q_ALPHA: f32 = 0.3;

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
enum MotifClass {
    Transition,
    Anchor,
    Macro,
}

#[derive(Debug, Clone, Default)]
struct MotifStats {
    success_count: u32,
    failure_count: u32,
}

#[derive(Debug, Clone)]
struct EpisodeMotifRecord {
    outcome: EpisodeOutcome,
    motifs: HashSet<String>,
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
    pub source_episodes: Vec<Episode>,

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
            eligibility_threshold: 0.5,
            w_novelty: 0.25,
            w_outcome_utility: 0.25,
            w_difficulty: 0.2,
            w_reuse_potential: 0.2,
            w_redundancy: 0.1,
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

/// Strategy extraction engine
pub struct StrategyExtractor {
    /// All extracted strategies
    strategies: FxHashMap<StrategyId, Strategy>,

    /// Strategy index by agent
    agent_strategies: FxHashMap<AgentId, Vec<StrategyId>>,

    /// Strategy index by context hash
    context_index: FxHashMap<ContextHash, Vec<StrategyId>>,

    /// Strategy index by goal bucket
    goal_bucket_index: FxHashMap<u64, Vec<StrategyId>>,

    /// Strategy index by behavior signature
    behavior_index: HashMap<String, Vec<StrategyId>>,

    /// Context counts for novelty estimation
    context_counts: FxHashMap<(AgentId, u64, ContextHash), u32>,

    /// Goal bucket occurrence counts (per agent)
    goal_bucket_counts: FxHashMap<(AgentId, u64), u32>,

    /// Motif stats by goal bucket (per agent)
    motif_stats_by_bucket: FxHashMap<(AgentId, u64), HashMap<String, MotifStats>>,

    /// Episode cache for validation (per agent + goal bucket)
    episode_cache_by_bucket: FxHashMap<(AgentId, u64), Vec<EpisodeMotifRecord>>,

    /// Strategy signature index to prevent duplicates
    strategy_signature_index: HashMap<String, StrategyId>,

    /// Episode to strategy index (idempotency)
    episode_index: FxHashMap<EpisodeId, StrategyId>,

    /// Episode outcome tracking for corrections
    episode_outcomes: FxHashMap<EpisodeId, EpisodeOutcome>,

    /// Configuration
    config: StrategyExtractionConfig,

    /// Next strategy ID
    next_strategy_id: StrategyId,
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

impl StrategyExtractor {
    /// Create a new strategy extractor
    pub fn new(config: StrategyExtractionConfig) -> Self {
        Self {
            strategies: FxHashMap::default(),
            agent_strategies: FxHashMap::default(),
            context_index: FxHashMap::default(),
            goal_bucket_index: FxHashMap::default(),
            behavior_index: HashMap::new(),
            context_counts: FxHashMap::default(),
            goal_bucket_counts: FxHashMap::default(),
            motif_stats_by_bucket: FxHashMap::default(),
            episode_cache_by_bucket: FxHashMap::default(),
            strategy_signature_index: HashMap::new(),
            episode_index: FxHashMap::default(),
            episode_outcomes: FxHashMap::default(),
            config,
            next_strategy_id: 1,
        }
    }

    /// Extract strategies from a successful episode
    ///
    /// Analyzes the episode's events to identify reusable patterns and strategies
    pub fn extract_from_episode(
        &mut self,
        episode: &Episode,
        events: &[Event],
    ) -> GraphResult<Option<StrategyUpsert>> {
        if let Some(existing_id) = self.episode_index.get(&episode.id).copied() {
            let new_outcome = episode
                .outcome
                .clone()
                .unwrap_or(EpisodeOutcome::Interrupted);
            self.apply_episode_outcome_correction(episode.id, existing_id, &new_outcome);
            return Ok(Some(StrategyUpsert {
                id: existing_id,
                is_new: false,
            }));
        }

        let outcome = episode
            .outcome
            .clone()
            .unwrap_or(EpisodeOutcome::Interrupted);
        let strategy_type = match outcome {
            EpisodeOutcome::Success => StrategyType::Positive,
            EpisodeOutcome::Failure => StrategyType::Constraint,
            EpisodeOutcome::Partial => StrategyType::Positive,
            EpisodeOutcome::Interrupted => StrategyType::Constraint,
        };

        let goal_bucket_id = self.derive_goal_bucket_id(episode);
        let behavior_signature = self.compute_behavior_signature(events);
        let eligibility_score =
            self.calculate_eligibility_score(episode, &behavior_signature, goal_bucket_id, events);
        if eligibility_score < self.config.eligibility_threshold {
            tracing::info!(
                "Strategy extraction rejected episode_id={} eligibility={:.3} min={:.3}",
                episode.id,
                eligibility_score,
                self.config.eligibility_threshold
            );
            return Ok(None);
        }

        // Extract reasoning traces from cognitive events
        let reasoning_steps = self.extract_reasoning_steps(events)?;

        if reasoning_steps.is_empty() {
            tracing::info!(
                "Strategy extraction note episode_id={} (no reasoning steps)",
                episode.id
            );
        }

        // Extract context patterns
        let context_patterns = self.extract_context_patterns(events)?;

        // Identify success indicators
        let success_indicators = self.identify_success_indicators(events)?;

        // Phase 1 Feature K: Extract failure patterns from failed events
        let failure_patterns = self.extract_failure_patterns(events, &episode.outcome)?;

        // Create strategy
        let strategy_id = self.next_strategy_id;
        self.next_strategy_id += 1;

        // Phase 1: Initialize quality with prediction-error weighting
        let quality_with_prediction = episode.significance * (1.0 + episode.prediction_error * 0.3);

        let (precondition, action_hint, expected_cost) =
            self.extract_behavior_skeleton(events, &strategy_type, goal_bucket_id);
        let (expected_success, expected_value, confidence) =
            self.derive_calibrated_metrics(&strategy_type, &outcome, events.len() as u32);
        // Generate natural language summary
        let summary = synthesize_strategy_summary(
            &strategy_type,
            &outcome,
            episode,
            events,
            &success_indicators,
            &failure_patterns,
        );

        // Generate 10x/100x fields
        let when_to_use = synthesize_when_to_use(&strategy_type, episode, events);
        let when_not_to_use = synthesize_when_not_to_use(&strategy_type, episode, events);
        let failure_mode_hints = synthesize_failure_modes(events);
        let playbook = build_playbook(events, &strategy_type);
        let counterfactual = synthesize_counterfactual(&outcome, events);

        // Detect applicable domains from goals + context
        let applicable_domains: Vec<String> = episode
            .context
            .active_goals
            .iter()
            .filter_map(|g| {
                if g.description.is_empty() {
                    None
                } else {
                    // Extract domain keyword from goal description
                    Some(
                        g.description
                            .split_whitespace()
                            .take(4)
                            .collect::<Vec<_>>()
                            .join(" "),
                    )
                }
            })
            .collect();

        let mut strategy = Strategy {
            id: strategy_id,
            name: match strategy_type {
                StrategyType::Positive => {
                    format!("strategy_{}_ep_{}", episode.agent_id, episode.id)
                },
                StrategyType::Constraint => {
                    format!("constraint_{}_ep_{}", episode.agent_id, episode.id)
                },
            },
            summary,
            when_to_use,
            when_not_to_use,
            failure_modes: failure_mode_hints,
            playbook,
            counterfactual,
            supersedes: Vec::new(),
            applicable_domains,
            lineage_depth: 0,
            summary_embedding: Vec::new(),
            agent_id: episode.agent_id,
            reasoning_steps,
            context_patterns: context_patterns.clone(),
            success_indicators,
            failure_patterns,
            quality_score: quality_with_prediction.min(1.0),
            success_count: if matches!(outcome, EpisodeOutcome::Success) {
                1
            } else {
                0
            },
            failure_count: if matches!(outcome, EpisodeOutcome::Failure) {
                1
            } else {
                0
            },
            support_count: 1,
            strategy_type,
            precondition,
            action_hint,
            expected_success,
            expected_cost,
            expected_value,
            confidence,
            contradictions: Vec::new(),
            goal_bucket_id,
            behavior_signature: behavior_signature.clone(),
            source_episodes: vec![episode.clone()],
            created_at: current_timestamp(),
            last_used: current_timestamp(),
            metadata: HashMap::new(),
            // Phase 1 fields
            self_judged_quality: episode.self_judged_quality,
            source_outcomes: vec![outcome.clone()],
            version: 1,
            parent_strategy: None,
        };

        let (goal_ids, tool_names, result_types) = self.extract_graph_signature(episode, events);
        strategy
            .metadata
            .insert("goal_ids".to_string(), json!(goal_ids).to_string());
        strategy
            .metadata
            .insert("tool_names".to_string(), json!(tool_names).to_string());
        strategy
            .metadata
            .insert("result_types".to_string(), json!(result_types).to_string());
        strategy
            .metadata
            .insert("goal_bucket_id".to_string(), goal_bucket_id.to_string());
        strategy
            .metadata
            .insert("behavior_signature".to_string(), behavior_signature.clone());
        strategy.metadata.insert(
            "strategy_type".to_string(),
            format!("{:?}", strategy.strategy_type),
        );
        let strategy_signature = self.compute_strategy_signature(&strategy);
        strategy
            .metadata
            .insert("strategy_signature".to_string(), strategy_signature.clone());

        let stored_id = self.store_strategy(strategy, &context_patterns, goal_bucket_id);
        self.episode_index.insert(episode.id, stored_id);
        self.episode_outcomes.insert(episode.id, outcome.clone());

        *self
            .context_counts
            .entry((episode.agent_id, goal_bucket_id, episode.context_signature))
            .or_insert(0) += 1;
        let bucket_count = {
            let bucket = self
                .goal_bucket_counts
                .entry((episode.agent_id, goal_bucket_id))
                .or_insert(0);
            *bucket += 1;
            *bucket
        };

        self.update_motif_stats(episode.agent_id, goal_bucket_id, outcome.clone(), events);
        if self.should_distill(bucket_count) {
            self.run_contrastive_distiller(episode.agent_id, goal_bucket_id);
        }

        tracing::info!(
            "Strategy stored id={} episode_id={} agent_id={} quality={:.3}",
            stored_id,
            episode.id,
            episode.agent_id,
            quality_with_prediction.min(1.0)
        );

        Ok(Some(StrategyUpsert {
            id: stored_id,
            is_new: true,
        }))
    }

    /// Extract reasoning steps from cognitive events with generalization
    fn extract_reasoning_steps(&self, events: &[Event]) -> GraphResult<Vec<ReasoningStep>> {
        let mut raw_steps = Vec::new();
        let mut order = 0;

        // First pass: collect all reasoning traces
        for event in events {
            if let EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                reasoning_trace,
                ..
            } = &event.event_type
            {
                // Only extract from reasoning events
                for trace_step in reasoning_trace {
                    raw_steps.push((trace_step.clone(), order));
                    order += 1;
                }
            }
        }

        // Second pass: generalize the steps
        let mut generalized_steps = Vec::new();
        for (raw_step, seq_order) in raw_steps {
            let generalized = self.generalize_reasoning_step(&raw_step);

            // Determine applicability based on abstraction level
            let applicability = if self.is_highly_abstract(&generalized) {
                "general".to_string()
            } else if self.is_parameterized(&generalized) {
                "contextual".to_string()
            } else {
                "specific".to_string()
            };

            generalized_steps.push(ReasoningStep {
                description: generalized,
                applicability,
                expected_outcome: self.infer_expected_outcome(&raw_step),
                sequence_order: seq_order,
            });
        }

        Ok(generalized_steps)
    }

    /// Generalize a reasoning step by abstracting specific values
    fn generalize_reasoning_step(&self, step: &str) -> String {
        let mut generalized = step.to_string();

        // Pattern 1: Abstract file paths (simple pattern matching)
        // Look for common path patterns: /path, C:\path, ./path, ../path
        let path_patterns = ["/", "\\", "./", "../", "C:", "D:", "E:"];
        for pattern in &path_patterns {
            if generalized.contains(pattern) {
                // Simple heuristic: if contains path separator and looks like a path
                let words: Vec<&str> = generalized.split_whitespace().collect();
                let mut new_words = Vec::new();
                for word in words {
                    if word.contains('/')
                        || word.contains('\\')
                        || (word.starts_with('.') && word.len() > 1)
                    {
                        new_words.push("<file_path>");
                    } else {
                        new_words.push(word);
                    }
                }
                generalized = new_words.join(" ");
                break;
            }
        }

        // Pattern 2: Abstract error messages
        let error_keywords = [
            "error:",
            "exception:",
            "failure:",
            "error ",
            "exception ",
            "failed",
        ];
        for keyword in &error_keywords {
            if generalized.to_lowercase().contains(keyword) {
                // Replace error message with placeholder
                if let Some(pos) = generalized.to_lowercase().find(keyword) {
                    let before = &generalized[..pos];
                    let after_pos = pos + keyword.len();
                    // Find end of error message (next period, newline, or end)
                    let end_pos = generalized[after_pos..]
                        .find(['.', '\n', '\r'])
                        .map(|i| after_pos + i)
                        .unwrap_or(generalized.len());
                    generalized = format!(
                        "{}<error_type>: <error_message>{}",
                        before,
                        &generalized[end_pos..]
                    );
                }
                break;
            }
        }

        // Pattern 3: Abstract URLs
        if generalized.contains("http://") || generalized.contains("https://") {
            let words: Vec<&str> = generalized.split_whitespace().collect();
            let new_words: Vec<String> = words
                .iter()
                .map(|w| {
                    if w.starts_with("http://") || w.starts_with("https://") {
                        "<url>".to_string()
                    } else {
                        w.to_string()
                    }
                })
                .collect();
            generalized = new_words.join(" ");
        }

        // Pattern 4: Abstract large numbers (but keep small ones for structure)
        let words: Vec<&str> = generalized.split_whitespace().collect();
        let new_words: Vec<String> = words
            .iter()
            .map(|w| {
                if let Ok(num) = w.parse::<u64>() {
                    if num > 100 {
                        "<number>".to_string()
                    } else {
                        w.to_string()
                    }
                } else {
                    w.to_string()
                }
            })
            .collect();
        generalized = new_words.join(" ");

        // Pattern 5: Identify common reasoning structures
        generalized = self.identify_reasoning_structures(&generalized);

        generalized
    }

    /// Identify and label common reasoning structures
    fn identify_reasoning_structures(&self, step: &str) -> String {
        let step_lower = step.to_lowercase();

        // If-then structure
        if step_lower.contains("if")
            && (step_lower.contains("then") || step_lower.contains("check"))
        {
            return format!("[IF-THEN] {}", step);
        }

        // Try-catch/error handling
        if step_lower.contains("try")
            || step_lower.contains("catch")
            || step_lower.contains("handle error")
        {
            return format!("[ERROR-HANDLING] {}", step);
        }

        // Decomposition
        if step_lower.contains("break down")
            || step_lower.contains("decompose")
            || step_lower.contains("split")
        {
            return format!("[DECOMPOSE] {}", step);
        }

        // Verification/validation
        if step_lower.contains("verify")
            || step_lower.contains("validate")
            || step_lower.contains("check")
        {
            return format!("[VERIFY] {}", step);
        }

        // Search/lookup
        if step_lower.contains("search")
            || step_lower.contains("find")
            || step_lower.contains("lookup")
        {
            return format!("[SEARCH] {}", step);
        }

        step.to_string()
    }

    /// Check if a step is highly abstract (reusable across domains)
    fn is_highly_abstract(&self, step: &str) -> bool {
        let abstract_patterns = [
            "[IF-THEN]",
            "[ERROR-HANDLING]",
            "[DECOMPOSE]",
            "[VERIFY]",
            "[SEARCH]",
        ];

        abstract_patterns
            .iter()
            .any(|pattern| step.contains(pattern))
    }

    /// Check if a step is parameterized (has placeholders)
    fn is_parameterized(&self, step: &str) -> bool {
        step.contains("<") && step.contains(">")
    }

    /// Infer expected outcome from reasoning step
    fn infer_expected_outcome(&self, step: &str) -> Option<String> {
        let step_lower = step.to_lowercase();

        if step_lower.contains("solve") || step_lower.contains("fix") {
            Some("problem_resolved".to_string())
        } else if step_lower.contains("verify") || step_lower.contains("check") {
            Some("validation_result".to_string())
        } else if step_lower.contains("find") || step_lower.contains("search") {
            Some("item_found".to_string())
        } else if step_lower.contains("decompose") || step_lower.contains("break down") {
            Some("subproblems_identified".to_string())
        } else {
            None
        }
    }

    /// Extract context patterns from events
    fn extract_context_patterns(&self, events: &[Event]) -> GraphResult<Vec<ContextPattern>> {
        let mut patterns = Vec::new();

        // Analyze the first event's context as representative
        if let Some(first_event) = events.first() {
            let context = &first_event.context;

            let pattern = ContextPattern {
                environment_type: Some("general".to_string()),
                task_type: Some(self.infer_task_type(events)),
                resource_constraints: self.extract_resource_constraints(context),
                goal_characteristics: context
                    .active_goals
                    .iter()
                    .map(|g| g.description.clone())
                    .collect(),
                match_confidence: 0.8,
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Infer task type from event sequence
    fn infer_task_type(&self, events: &[Event]) -> String {
        // Simple heuristic based on event types
        let has_cognitive = events
            .iter()
            .any(|e| matches!(e.event_type, EventType::Cognitive { .. }));

        let has_action = events
            .iter()
            .any(|e| matches!(e.event_type, EventType::Action { .. }));

        if has_cognitive && has_action {
            "problem_solving".to_string()
        } else if has_action {
            "execution".to_string()
        } else {
            "analysis".to_string()
        }
    }

    /// Extract resource constraints from context
    fn extract_resource_constraints(
        &self,
        context: &agent_db_events::core::EventContext,
    ) -> Vec<String> {
        let mut constraints = Vec::new();

        if context.resources.computational.cpu_percent > 80.0 {
            constraints.push("high_cpu_usage".to_string());
        }

        if context.resources.computational.memory_bytes > 8_000_000_000 {
            constraints.push("high_memory_usage".to_string());
        }

        constraints
    }

    /// Identify success indicators from events
    fn identify_success_indicators(&self, events: &[Event]) -> GraphResult<Vec<String>> {
        let mut indicators = Vec::new();

        for event in events {
            if let EventType::Action {
                outcome: agent_db_events::core::ActionOutcome::Success { .. },
                ..
            } = &event.event_type
            {
                indicators.push("action_succeeded".to_string());
            }
        }

        if !indicators.is_empty() {
            indicators.push("episode_completed".to_string());
        }

        Ok(indicators)
    }

    /// Phase 1 Feature K: Extract failure patterns from failed events
    /// Identifies what went wrong so the agent can avoid repeating mistakes
    fn extract_failure_patterns(
        &self,
        events: &[Event],
        outcome: &Option<EpisodeOutcome>,
    ) -> GraphResult<Vec<String>> {
        let mut patterns = Vec::new();

        // Only extract failure patterns if episode failed
        if !matches!(outcome, Some(EpisodeOutcome::Failure)) {
            return Ok(patterns);
        }

        // Extract patterns from failed actions
        for event in events {
            if let EventType::Action {
                outcome: agent_db_events::core::ActionOutcome::Failure { error, .. },
                action_name,
                ..
            } = &event.event_type
            {
                // Record what action failed
                patterns.push(format!("avoid_action:{}", action_name));

                // Record the error message
                patterns.push(format!("failure_reason:{}", error));
            }
        }

        // Add general failure context if no specific patterns found
        if patterns.is_empty() {
            patterns.push("episode_failed_unknown_reason".to_string());
        }

        // Add high-level failure indicator
        patterns.push("avoid_similar_context".to_string());

        Ok(patterns)
    }

    fn extract_graph_signature(
        &self,
        episode: &Episode,
        events: &[Event],
    ) -> (Vec<u64>, Vec<String>, Vec<String>) {
        let goal_ids = episode
            .context
            .active_goals
            .iter()
            .map(|goal| goal.id)
            .collect::<Vec<_>>();

        let tool_names = self.extract_tool_names(events);
        let result_types = self.extract_result_types(events);

        (goal_ids, tool_names, result_types)
    }

    fn extract_tool_names(&self, events: &[Event]) -> Vec<String> {
        let mut tools: HashSet<String> = HashSet::new();

        for event in events {
            for key in ["tool", "tool_name", "tools", "tool_used"] {
                if let Some(value) = event.metadata.get(key) {
                    self.collect_tools_from_metadata(value, &mut tools);
                }
            }

            if let EventType::Action { parameters, .. } = &event.event_type {
                self.collect_tools_from_json(parameters, &mut tools);
            }
        }

        let mut list: Vec<String> = tools.into_iter().collect();
        list.sort();
        list
    }

    fn collect_tools_from_metadata(&self, value: &MetadataValue, tools: &mut HashSet<String>) {
        match value {
            MetadataValue::String(name) => {
                if !name.trim().is_empty() {
                    tools.insert(name.trim().to_string());
                }
            },
            MetadataValue::Json(json) => {
                self.collect_tools_from_json(json, tools);
            },
            _ => {},
        }
    }

    fn collect_tools_from_json(&self, value: &serde_json::Value, tools: &mut HashSet<String>) {
        match value {
            serde_json::Value::String(name) => {
                if !name.trim().is_empty() {
                    tools.insert(name.trim().to_string());
                }
            },
            serde_json::Value::Array(items) => {
                for item in items {
                    self.collect_tools_from_json(item, tools);
                }
            },
            serde_json::Value::Object(map) => {
                for key in ["tool", "tool_name", "tools", "tool_used"] {
                    if let Some(value) = map.get(key) {
                        self.collect_tools_from_json(value, tools);
                    }
                }
            },
            _ => {},
        }
    }

    fn extract_result_types(&self, events: &[Event]) -> Vec<String> {
        let mut types: HashSet<String> = HashSet::new();

        for event in events {
            match &event.event_type {
                EventType::Action { outcome, .. } => match outcome {
                    agent_db_events::core::ActionOutcome::Success { .. } => {
                        types.insert("action_success".to_string());
                    },
                    agent_db_events::core::ActionOutcome::Failure { .. } => {
                        types.insert("action_failure".to_string());
                    },
                    agent_db_events::core::ActionOutcome::Partial { .. } => {
                        types.insert("action_partial".to_string());
                    },
                },
                EventType::Observation { .. } => {
                    types.insert("observation".to_string());
                },
                EventType::Cognitive { .. } => {
                    types.insert("cognitive_output".to_string());
                },
                EventType::Communication { .. } => {
                    types.insert("communication".to_string());
                },
                EventType::Learning { .. } => {
                    types.insert("learning_telemetry".to_string());
                },
                EventType::Context { .. } => {
                    types.insert("context".to_string());
                },
            }
        }

        let mut list: Vec<String> = types.into_iter().collect();
        list.sort();
        list
    }

    /// Convert context pattern to hash for indexing
    fn pattern_to_hash(&self, pattern: &ContextPattern) -> Option<ContextHash> {
        // Simplified hashing - in production would use proper hash function
        pattern.task_type.as_ref().map(|t| t.len() as u64)
    }

    /// Retrieve strategies applicable to a context
    pub fn get_strategies_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
    ) -> Vec<&Strategy> {
        self.context_index
            .get(&context_hash)
            .map(|ids| {
                let mut strategies: Vec<&Strategy> = ids
                    .iter()
                    .filter_map(|id| self.strategies.get(id))
                    .collect();

                // Sort by quality score
                strategies.sort_by(|a, b| {
                    b.quality_score
                        .partial_cmp(&a.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                strategies.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    fn apply_episode_outcome_correction(
        &mut self,
        episode_id: EpisodeId,
        strategy_id: StrategyId,
        new_outcome: &EpisodeOutcome,
    ) {
        let prev_outcome = self.episode_outcomes.get(&episode_id).cloned();
        if prev_outcome.as_ref() == Some(new_outcome) {
            return;
        }

        let Some(strategy) = self.strategies.get_mut(&strategy_id) else {
            return;
        };

        let (prev_success, prev_failure) = outcome_to_counts(prev_outcome.as_ref());
        let (new_success, new_failure) = outcome_to_counts(Some(new_outcome));

        if prev_success > 0 {
            strategy.success_count = strategy.success_count.saturating_sub(prev_success);
        }
        if prev_failure > 0 {
            strategy.failure_count = strategy.failure_count.saturating_sub(prev_failure);
        }
        strategy.success_count = strategy.success_count.saturating_add(new_success);
        strategy.failure_count = strategy.failure_count.saturating_add(new_failure);

        let total = strategy.success_count + strategy.failure_count;
        if total > 0 {
            strategy.quality_score = strategy.success_count as f32 / total as f32;
        }
        strategy.support_count = total;
        strategy.expected_success = (strategy.success_count as f32 + 1.0) / (total as f32 + 2.0);
        strategy.confidence = 1.0 - (-((total as f32) / 3.0)).exp();

        self.episode_outcomes
            .insert(episode_id, new_outcome.clone());

        tracing::info!(
            "Strategy updated from episode correction strategy_id={} episode_id={} success_count={} failure_count={}",
            strategy_id,
            episode_id,
            strategy.success_count,
            strategy.failure_count
        );
    }

    /// Find strategies similar to a query signature
    pub fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)> {
        tracing::info!(
            "Strategy similarity query goals={} tools={} results={} context_hash={:?} agent_id={:?} min_score={:.3} limit={}",
            query.goal_ids.len(),
            query.tool_names.len(),
            query.result_types.len(),
            query.context_hash,
            query.agent_id,
            query.min_score,
            query.limit
        );
        let goal_bucket_id = compute_goal_bucket_id_from_ids(&query.goal_ids);

        let candidate_ids: Vec<StrategyId> = if let Some(context_hash) = query.context_hash {
            self.context_index
                .get(&context_hash)
                .cloned()
                .unwrap_or_default()
        } else if goal_bucket_id != 0 {
            self.goal_bucket_index
                .get(&goal_bucket_id)
                .cloned()
                .unwrap_or_default()
        } else {
            self.strategies.keys().copied().collect()
        };

        let query_goals: HashSet<u64> = query.goal_ids.iter().copied().collect();
        let query_tools: HashSet<String> = query.tool_names.iter().cloned().collect();
        let query_results: HashSet<String> = query.result_types.iter().cloned().collect();

        let goal_weight = if query_goals.is_empty() { 0.0 } else { 0.5 };
        let tool_weight = if query_tools.is_empty() { 0.0 } else { 0.3 };
        let result_weight = if query_results.is_empty() { 0.0 } else { 0.2 };
        let weight_sum = goal_weight + tool_weight + result_weight;

        let mut scored: Vec<(Strategy, f32)> = candidate_ids
            .into_iter()
            .filter_map(|id| self.strategies.get(&id))
            .filter(|strategy| {
                if let Some(agent_id) = query.agent_id {
                    strategy.agent_id == agent_id
                } else {
                    true
                }
            })
            .map(|strategy| {
                let (goal_ids, tool_names, result_types) = self.parse_graph_signature(strategy);
                let score = if weight_sum == 0.0 {
                    0.0
                } else {
                    let goals_score = Self::jaccard_u64(&query_goals, &goal_ids);
                    let tools_score = Self::jaccard_string(&query_tools, &tool_names);
                    let results_score = Self::jaccard_string(&query_results, &result_types);
                    (goals_score * goal_weight
                        + tools_score * tool_weight
                        + results_score * result_weight)
                        / weight_sum
                };
                (strategy.clone(), score)
            })
            .filter(|(_, score)| *score >= query.min_score)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        {
            let result = scored.into_iter().take(query.limit).collect::<Vec<_>>();
            tracing::info!("Strategy similarity results={}", result.len());
            result
        }
    }

    /// Get a strategy by ID
    pub fn get_strategy(&self, strategy_id: StrategyId) -> Option<&Strategy> {
        self.strategies.get(&strategy_id)
    }

    fn parse_graph_signature(
        &self,
        strategy: &Strategy,
    ) -> (HashSet<u64>, HashSet<String>, HashSet<String>) {
        let goal_ids = strategy
            .metadata
            .get("goal_ids")
            .and_then(|value| serde_json::from_str::<Vec<u64>>(value).ok())
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        let tool_names = strategy
            .metadata
            .get("tool_names")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        let result_types = strategy
            .metadata
            .get("result_types")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        (goal_ids, tool_names, result_types)
    }

    fn derive_goal_bucket_id(&self, episode: &Episode) -> u64 {
        // Prefer the pre-computed goal_bucket_id from EventContext when available.
        if episode.context.goal_bucket_id != 0 {
            return episode.context.goal_bucket_id;
        }
        let goal_ids: Vec<u64> = episode
            .context
            .active_goals
            .iter()
            .map(|goal| goal.id)
            .collect();
        compute_goal_bucket_id_from_ids(&goal_ids)
    }

    fn compute_behavior_signature(&self, events: &[Event]) -> String {
        let skeleton = self.build_behavior_skeleton(events);
        let joined = skeleton.join(">");
        format!("{:x}", self.hash_str(&joined))
    }

    fn build_behavior_skeleton(&self, events: &[Event]) -> Vec<String> {
        let mut skeleton = Vec::new();
        for event in events {
            match &event.event_type {
                EventType::Observation { .. } => skeleton.push("Observe".to_string()),
                EventType::Cognitive { process_type, .. } => {
                    skeleton.push(format!("Think:{:?}", process_type));
                },
                EventType::Action { action_name, .. } => {
                    let tool = self
                        .extract_tool_from_metadata(event)
                        .map(|t| format!(":{}", t))
                        .unwrap_or_default();
                    skeleton.push(format!("Act:{}{}", action_name, tool));
                },
                EventType::Communication { .. } => skeleton.push("Communicate".to_string()),
                EventType::Learning { .. } => skeleton.push("Learn".to_string()),
                EventType::Context { context_type, .. } => {
                    skeleton.push(format!("Context:{}", context_type));
                },
            }
        }
        skeleton
    }

    fn extract_behavior_skeleton(
        &self,
        events: &[Event],
        strategy_type: &StrategyType,
        goal_bucket_id: u64,
    ) -> (String, String, f32) {
        let skeleton = self.build_behavior_skeleton(events);
        let action_hint = match strategy_type {
            StrategyType::Positive => format!("repeat sequence: {}", skeleton.join(" > ")),
            StrategyType::Constraint => format!("avoid sequence: {}", skeleton.join(" > ")),
        };
        let precondition = format!("goal_bucket={} contexts={}", goal_bucket_id, skeleton.len());
        let expected_cost = (events.len() as f32).min(50.0);
        (precondition, action_hint, expected_cost)
    }

    fn calculate_eligibility_score(
        &self,
        episode: &Episode,
        behavior_signature: &str,
        goal_bucket_id: u64,
        events: &[Event],
    ) -> f32 {
        let context_count = *self
            .context_counts
            .get(&(episode.agent_id, goal_bucket_id, episode.context_signature))
            .unwrap_or(&0);
        let novelty = 1.0 / (1.0 + context_count as f32);

        let outcome_utility = match episode
            .outcome
            .clone()
            .unwrap_or(EpisodeOutcome::Interrupted)
        {
            EpisodeOutcome::Success => 1.0,
            EpisodeOutcome::Partial => 0.7,
            EpisodeOutcome::Failure => 0.8,
            EpisodeOutcome::Interrupted => 0.4,
        };

        let difficulty = ((events.len() as f32) / 10.0).min(1.0);

        let bucket_count = *self
            .goal_bucket_counts
            .get(&(episode.agent_id, goal_bucket_id))
            .unwrap_or(&0);
        let reuse_potential = if bucket_count == 0 {
            0.0
        } else {
            (bucket_count as f32 / 10.0).min(1.0)
        };

        let redundancy = self.estimate_redundancy(goal_bucket_id, behavior_signature);

        let score = self.config.w_novelty * novelty
            + self.config.w_outcome_utility * outcome_utility
            + self.config.w_difficulty * difficulty
            + self.config.w_reuse_potential * reuse_potential
            - self.config.w_redundancy * redundancy;

        score.clamp(0.0, 1.0)
    }

    fn estimate_redundancy(&self, goal_bucket_id: u64, behavior_signature: &str) -> f32 {
        if let Some(ids) = self.goal_bucket_index.get(&goal_bucket_id) {
            for id in ids {
                if let Some(strategy) = self.strategies.get(id) {
                    if strategy.behavior_signature == behavior_signature {
                        return 1.0;
                    }
                }
            }
        }
        0.0
    }

    fn derive_calibrated_metrics(
        &self,
        strategy_type: &StrategyType,
        outcome: &EpisodeOutcome,
        support_count: u32,
    ) -> (f32, f32, f32) {
        let expected_success = match outcome {
            EpisodeOutcome::Success => 0.8,
            EpisodeOutcome::Partial => 0.6,
            EpisodeOutcome::Failure => 0.4,
            EpisodeOutcome::Interrupted => 0.4,
        };
        let expected_value = match strategy_type {
            StrategyType::Positive => expected_success,
            StrategyType::Constraint => -expected_success,
        };
        let confidence = 1.0 - (-((support_count as f32) / 3.0)).exp();
        (expected_success, expected_value, confidence)
    }

    fn compute_strategy_signature(&self, strategy: &Strategy) -> String {
        let raw = format!(
            "{}|{}|{:?}",
            strategy.precondition, strategy.action_hint, strategy.strategy_type
        );
        format!("{:x}", self.hash_str(&raw))
    }

    fn store_strategy(
        &mut self,
        mut strategy: Strategy,
        context_patterns: &[ContextPattern],
        goal_bucket_id: u64,
    ) -> StrategyId {
        let signature = strategy
            .metadata
            .get("strategy_signature")
            .cloned()
            .unwrap_or_else(|| self.compute_strategy_signature(&strategy));

        if let Some(existing_id) = self.strategy_signature_index.get(&signature).copied() {
            if let Some(existing) = self.strategies.get_mut(&existing_id) {
                let new_support = existing.support_count + strategy.support_count;
                existing.support_count = new_support;
                existing.success_count += strategy.success_count;
                existing.failure_count += strategy.failure_count;
                existing.last_used = current_timestamp();
                existing
                    .source_episodes
                    .append(&mut strategy.source_episodes);
                existing
                    .source_outcomes
                    .append(&mut strategy.source_outcomes);

                let expected_success = (existing.success_count as f32 + self.config.alpha)
                    / (existing.support_count as f32 + self.config.alpha + self.config.beta);
                existing.expected_success = expected_success;
                existing.expected_value = match existing.strategy_type {
                    StrategyType::Positive => expected_success,
                    StrategyType::Constraint => -expected_success,
                };
                existing.confidence = 1.0 - (-((existing.support_count as f32) / 3.0)).exp();

                if existing.reasoning_steps.is_empty() && !strategy.reasoning_steps.is_empty() {
                    existing.reasoning_steps = strategy.reasoning_steps;
                }

                return existing_id;
            }
        }

        let strategy_id = strategy.id;
        self.strategy_signature_index.insert(signature, strategy_id);
        let agent_id = strategy.agent_id;
        self.strategies.insert(strategy_id, strategy);

        self.agent_strategies
            .entry(agent_id)
            .or_default()
            .push(strategy_id);

        for pattern in context_patterns {
            if let Some(context_hash) = self.pattern_to_hash(pattern) {
                self.context_index
                    .entry(context_hash)
                    .or_default()
                    .push(strategy_id);
            }
        }

        self.goal_bucket_index
            .entry(goal_bucket_id)
            .or_default()
            .push(strategy_id);

        if let Some(sig) = self
            .strategies
            .get(&strategy_id)
            .and_then(|s| s.metadata.get("behavior_signature"))
            .cloned()
        {
            self.behavior_index
                .entry(sig)
                .or_default()
                .push(strategy_id);
        }

        strategy_id
    }

    fn should_distill(&self, bucket_count: u32) -> bool {
        bucket_count.is_multiple_of(self.config.distill_every)
    }

    fn update_motif_stats(
        &mut self,
        agent_id: AgentId,
        goal_bucket_id: u64,
        outcome: EpisodeOutcome,
        events: &[Event],
    ) {
        let motifs = self.extract_motifs(events);
        let bucket_stats = self
            .motif_stats_by_bucket
            .entry((agent_id, goal_bucket_id))
            .or_default();

        for motif in motifs.iter() {
            let stats = bucket_stats.entry(motif.clone()).or_default();
            match outcome {
                EpisodeOutcome::Success | EpisodeOutcome::Partial => stats.success_count += 1,
                EpisodeOutcome::Failure | EpisodeOutcome::Interrupted => stats.failure_count += 1,
            }
        }

        let cache = self
            .episode_cache_by_bucket
            .entry((agent_id, goal_bucket_id))
            .or_default();
        cache.push(EpisodeMotifRecord { outcome, motifs });
        if cache.len() > self.config.cache_max {
            cache.remove(0);
        }
    }

    fn run_contrastive_distiller(&mut self, agent_id: AgentId, goal_bucket_id: u64) {
        let bucket_stats = match self.motif_stats_by_bucket.get(&(agent_id, goal_bucket_id)) {
            Some(stats) => stats.clone(),
            None => return,
        };
        let cache = match self
            .episode_cache_by_bucket
            .get(&(agent_id, goal_bucket_id))
        {
            Some(records) => records.clone(),
            None => return,
        };

        let (success_total, failure_total) = self.count_outcomes(&cache);
        if success_total == 0 && failure_total == 0 {
            return;
        }

        let baseline_success = (success_total as f32 + self.config.alpha)
            / (success_total as f32 + failure_total as f32 + self.config.alpha + self.config.beta);
        let baseline_failure = 1.0 - baseline_success;

        for (motif, stats) in bucket_stats {
            let s = stats.success_count as f32;
            let f = stats.failure_count as f32;
            let success_total_f = success_total as f32;
            let failure_total_f = failure_total as f32;

            let p_s =
                (s + self.config.alpha) / (success_total_f + self.config.alpha + self.config.beta);
            let p_f =
                (f + self.config.alpha) / (failure_total_f + self.config.alpha + self.config.beta);

            let lift = Self::log_odds(p_s) - Self::log_odds(p_f);
            let uplift = p_s - baseline_success;
            let failure_uplift = p_f - baseline_failure;

            let strategy_type = if lift >= self.config.min_lift && uplift >= self.config.min_uplift
            {
                StrategyType::Positive
            } else if lift <= -self.config.min_lift && failure_uplift >= self.config.min_uplift {
                StrategyType::Constraint
            } else {
                continue;
            };

            let support = match strategy_type {
                StrategyType::Positive => stats.success_count,
                StrategyType::Constraint => stats.failure_count,
            };

            if strategy_type == StrategyType::Positive && support < self.config.min_support_success
            {
                continue;
            }
            if strategy_type == StrategyType::Constraint
                && support < self.config.min_support_failure
            {
                continue;
            }

            if !self.validate_candidate(&cache, &motif, strategy_type, baseline_success) {
                continue;
            }

            let precondition = format!("goal_bucket={} motif={}", goal_bucket_id, motif);
            let action_hint = match strategy_type {
                StrategyType::Positive => format!("prefer motif: {}", motif),
                StrategyType::Constraint => format!("avoid motif: {}", motif),
            };
            let expected_success = p_s;
            let expected_value = match strategy_type {
                StrategyType::Positive => uplift,
                StrategyType::Constraint => -failure_uplift,
            };
            let confidence = 1.0 - (-((support as f32) / 3.0)).exp();

            let strategy_id = self.next_strategy_id;
            self.next_strategy_id += 1;

            // Build summary for distiller-generated strategy
            let distiller_summary = match strategy_type {
                StrategyType::Positive => format!(
                    "DO this when applicable. Prefer motif pattern '{}' in goal_bucket {}. Success rate {:.0}%, uplift {:.0}%, supported by {} episodes. Confidence {:.0}%.",
                    motif, goal_bucket_id, p_s * 100.0, uplift * 100.0, support, confidence * 100.0
                ),
                StrategyType::Constraint => format!(
                    "AVOID this pattern. Motif '{}' in goal_bucket {} correlates with failure. Failure uplift {:.0}%, supported by {} episodes. Confidence {:.0}%.",
                    motif, goal_bucket_id, failure_uplift * 100.0, support, confidence * 100.0
                ),
            };

            // Generate 10x fields for distiller-generated strategies
            let distiller_when_to_use = match strategy_type {
                StrategyType::Positive => format!(
                    "Use when goal bucket is {} and the motif '{}' is applicable. Best when success rate > {:.0}%.",
                    goal_bucket_id, motif, p_s * 100.0
                ),
                StrategyType::Constraint => format!(
                    "Avoid when goal bucket is {} and the motif '{}' appears. Failure correlation is strong ({:.0}%).",
                    goal_bucket_id, motif, failure_uplift * 100.0
                ),
            };
            let distiller_when_not_to_use = match strategy_type {
                StrategyType::Positive => format!(
                    "Do not use when context significantly differs from goal_bucket {} or when contradictions were observed.",
                    goal_bucket_id
                ),
                StrategyType::Constraint => format!(
                    "Safe to ignore when success rate in similar contexts is already > {:.0}% without this motif.",
                    p_s * 100.0
                ),
            };

            let mut strategy = Strategy {
                id: strategy_id,
                name: match strategy_type {
                    StrategyType::Positive => format!("strategy_{}_motif", goal_bucket_id),
                    StrategyType::Constraint => format!("constraint_{}_motif", goal_bucket_id),
                },
                summary: distiller_summary,
                when_to_use: distiller_when_to_use,
                when_not_to_use: distiller_when_not_to_use,
                failure_modes: Vec::new(),
                playbook: Vec::new(), // Distiller strategies are motif-level, no step playbook
                counterfactual: String::new(),
                supersedes: Vec::new(),
                applicable_domains: Vec::new(),
                lineage_depth: 0,
                summary_embedding: Vec::new(),
                agent_id,
                reasoning_steps: Vec::new(),
                context_patterns: Vec::new(),
                success_indicators: Vec::new(),
                failure_patterns: vec![motif.clone()],
                quality_score: (expected_success).min(1.0),
                success_count: if strategy_type == StrategyType::Positive {
                    support
                } else {
                    0
                },
                failure_count: if strategy_type == StrategyType::Constraint {
                    support
                } else {
                    0
                },
                support_count: support,
                strategy_type,
                precondition,
                action_hint,
                expected_success,
                expected_cost: 1.0,
                expected_value,
                confidence,
                contradictions: Vec::new(),
                goal_bucket_id,
                behavior_signature: motif.clone(),
                source_episodes: Vec::new(),
                created_at: current_timestamp(),
                last_used: current_timestamp(),
                metadata: HashMap::new(),
                self_judged_quality: None,
                source_outcomes: Vec::new(),
                version: 1,
                parent_strategy: None,
            };

            strategy.metadata.insert(
                "strategy_signature".to_string(),
                self.compute_strategy_signature(&strategy),
            );
            strategy
                .metadata
                .insert("behavior_signature".to_string(), motif.clone());
            strategy
                .metadata
                .insert("goal_bucket_id".to_string(), goal_bucket_id.to_string());
            strategy.metadata.insert(
                "strategy_type".to_string(),
                format!("{:?}", strategy.strategy_type),
            );

            let _ = self.store_strategy(strategy, &[], goal_bucket_id);
        }
    }

    fn validate_candidate(
        &self,
        cache: &[EpisodeMotifRecord],
        motif: &str,
        strategy_type: StrategyType,
        baseline_success: f32,
    ) -> bool {
        if cache.len() < self.config.holdout_size {
            return false;
        }

        let holdout = &cache[cache.len().saturating_sub(self.config.holdout_size)..];
        let mut matches = Vec::new();
        for record in holdout {
            if record.motifs.contains(motif) {
                matches.push(record.outcome.clone());
            }
        }

        if matches.len() < self.config.min_holdout_coverage as usize {
            return false;
        }

        let success_matches = matches
            .iter()
            .filter(|o| matches!(o, EpisodeOutcome::Success))
            .count();
        let failure_matches = matches
            .iter()
            .filter(|o| matches!(o, EpisodeOutcome::Failure))
            .count();

        let precision = success_matches as f32 / matches.len().max(1) as f32;
        let failure_rate = failure_matches as f32 / matches.len().max(1) as f32;

        let baseline_failure = 1.0 - baseline_success;
        let passes_precision = match strategy_type {
            StrategyType::Positive => precision >= baseline_success + self.config.min_uplift,
            StrategyType::Constraint => failure_rate >= baseline_failure + self.config.min_uplift,
        };

        if !passes_precision {
            return false;
        }

        let mid = matches.len() / 2;
        if mid > 0 {
            let (first, second) = matches.split_at(mid);
            let first_success = first
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Success))
                .count();
            let second_success = second
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Success))
                .count();
            let first_failure = first
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Failure))
                .count();
            let second_failure = second
                .iter()
                .filter(|o| matches!(o, EpisodeOutcome::Failure))
                .count();

            match strategy_type {
                StrategyType::Positive => {
                    let first_rate = first_success as f32 / first.len().max(1) as f32;
                    let second_rate = second_success as f32 / second.len().max(1) as f32;
                    if first_rate - second_rate > self.config.drift_max_drop {
                        return false;
                    }
                },
                StrategyType::Constraint => {
                    let first_rate = first_failure as f32 / first.len().max(1) as f32;
                    let second_rate = second_failure as f32 / second.len().max(1) as f32;
                    if first_rate - second_rate > self.config.drift_max_drop {
                        return false;
                    }
                },
            }
        }

        true
    }

    fn extract_motifs(&self, events: &[Event]) -> HashSet<String> {
        let mut motifs = HashSet::new();
        let tokens = self.build_behavior_skeleton(events);

        for i in 0..tokens.len().saturating_sub(1) {
            let left = tokens[i].clone();
            let right = tokens[i + 1].clone();
            motifs.insert(Self::motif_key(
                MotifClass::Transition,
                format!("{}->{}", left, right),
            ));
        }

        let anchors = self.find_anchor_indices(events);
        for anchor in anchors {
            let start = anchor.saturating_sub(self.config.motif_window_k);
            let end = (anchor + self.config.motif_window_k).min(tokens.len().saturating_sub(1));
            let window = tokens[start..=end].join(">");
            motifs.insert(Self::motif_key(MotifClass::Anchor, window));
        }

        let action_tokens = self.build_action_tokens(events);
        for n in 3..=6 {
            if action_tokens.len() < n {
                continue;
            }
            for i in 0..=action_tokens.len() - n {
                let seq = action_tokens[i..i + n].join(">");
                motifs.insert(Self::motif_key(MotifClass::Macro, seq));
            }
        }

        motifs
    }

    fn find_anchor_indices(&self, events: &[Event]) -> Vec<usize> {
        let mut anchors = Vec::new();
        for (idx, event) in events.iter().enumerate() {
            match &event.event_type {
                EventType::Action {
                    action_name,
                    outcome,
                    ..
                } => {
                    if action_name == "user_feedback" {
                        anchors.push(idx);
                    }
                    match outcome {
                        agent_db_events::core::ActionOutcome::Success { .. } => anchors.push(idx),
                        agent_db_events::core::ActionOutcome::Failure { .. } => anchors.push(idx),
                        _ => {},
                    }
                },
                EventType::Observation { data, .. } => {
                    let text = data.to_string().to_lowercase();
                    if text.contains("error")
                        || text.contains("failed")
                        || text.contains("exception")
                    {
                        anchors.push(idx);
                    }
                },
                _ => {},
            }
        }
        anchors
    }

    fn build_action_tokens(&self, events: &[Event]) -> Vec<String> {
        let mut tokens = Vec::new();
        for event in events {
            if let EventType::Action { action_name, .. } = &event.event_type {
                let tool = self
                    .extract_tool_from_metadata(event)
                    .map(|t| format!(":{}", t))
                    .unwrap_or_default();
                tokens.push(format!("{}{}", action_name, tool));
            }
        }
        tokens
    }

    fn motif_key(class: MotifClass, token: String) -> String {
        format!("{:?}::{}", class, token)
    }

    fn count_outcomes(&self, records: &[EpisodeMotifRecord]) -> (u32, u32) {
        let mut success = 0;
        let mut failure = 0;
        for record in records {
            match record.outcome {
                EpisodeOutcome::Success | EpisodeOutcome::Partial => success += 1,
                EpisodeOutcome::Failure | EpisodeOutcome::Interrupted => failure += 1,
            }
        }
        (success, failure)
    }

    fn log_odds(p: f32) -> f32 {
        let clamped = p.clamp(0.001, 0.999);
        (clamped / (1.0 - clamped)).ln()
    }

    fn extract_tool_from_metadata(&self, event: &Event) -> Option<String> {
        for key in ["tool_name", "tool", "tool_used"] {
            if let Some(value) = event.metadata.get(key) {
                if let Some(tool) = self.metadata_to_string(value) {
                    return Some(tool);
                }
            }
        }
        None
    }

    fn metadata_to_string(&self, value: &MetadataValue) -> Option<String> {
        match value {
            MetadataValue::String(s) => Some(s.clone()),
            MetadataValue::Integer(i) => Some(i.to_string()),
            MetadataValue::Float(f) => Some(format!("{}", f)),
            _ => None,
        }
    }

    fn hash_str(&self, value: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn jaccard_u64(a: &HashSet<u64>, b: &HashSet<u64>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    fn jaccard_string(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Get all strategies for an agent
    pub fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<&Strategy> {
        self.agent_strategies
            .get(&agent_id)
            .map(|ids| {
                let mut strategies: Vec<&Strategy> = ids
                    .iter()
                    .filter_map(|id| self.strategies.get(id))
                    .collect();

                // Sort by quality score
                strategies.sort_by(|a, b| {
                    b.quality_score
                        .partial_cmp(&a.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                strategies.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    /// Update strategy based on new usage outcome.
    ///
    /// Keeps existing counters and raw ratio (`quality_score`) for backward compat.
    /// Adds EMA Q-value in metadata and piecewise-blended `confidence` that
    /// reflects both sample size and outcome quality.
    pub fn update_strategy_outcome(
        &mut self,
        strategy_id: StrategyId,
        success: bool,
    ) -> GraphResult<()> {
        if let Some(strategy) = self.strategies.get_mut(&strategy_id) {
            // Lossless counters (keep existing)
            if success {
                strategy.success_count += 1;
            } else {
                strategy.failure_count += 1;
            }

            // Raw win ratio (keep for backward compat)
            let total = strategy.success_count + strategy.failure_count;
            if total > 0 {
                strategy.quality_score = strategy.success_count as f32 / total as f32;
            }
            strategy.support_count = total;

            // Bayesian expected success (keep for small N)
            strategy.expected_success =
                (strategy.success_count as f32 + 1.0) / (total as f32 + 2.0);

            // Update EMA Q-value in metadata: Q = Q + α(r − Q)
            let r = if success { 1.0_f32 } else { 0.0 };
            let q_old: f32 = strategy
                .metadata
                .get(META_Q_VALUE)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5);
            let q_new = q_old + Q_ALPHA * (r - q_old);
            strategy
                .metadata
                .insert(META_Q_VALUE.to_string(), format!("{:.6}", q_new));

            // Store lifetime counters in metadata for audit
            let pos_key = if success {
                META_POSITIVE_OUTCOMES
            } else {
                META_NEGATIVE_OUTCOMES
            };
            let current_count: u32 = strategy
                .metadata
                .get(pos_key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            strategy
                .metadata
                .insert(pos_key.to_string(), (current_count + 1).to_string());

            // Piecewise score: Bayesian for small N, EMA for large N
            let piecewise_score = if total < Q_KICK_IN {
                strategy.expected_success // Bayesian
            } else {
                q_new // EMA Q-value
            };

            // Confidence = sample-size factor × piecewise outcome quality
            let sample_confidence = 1.0 - (-((total as f32) / 3.0)).exp();
            strategy.confidence = sample_confidence * piecewise_score;

            strategy.last_used = current_timestamp();
        }

        Ok(())
    }

    /// Get strategy statistics
    pub fn get_stats(&self) -> StrategyStats {
        StrategyStats {
            total_strategies: self.strategies.len(),
            high_quality_strategies: self
                .strategies
                .values()
                .filter(|s| s.quality_score > 0.8)
                .count(),
            agents_with_strategies: self.agent_strategies.len(),
            average_quality: if !self.strategies.is_empty() {
                self.strategies
                    .values()
                    .map(|s| s.quality_score)
                    .sum::<f32>()
                    / self.strategies.len() as f32
            } else {
                0.0
            },
        }
    }

    /// List all strategies (used by maintenance / pruning).
    pub fn list_all_strategies(&self) -> Vec<&Strategy> {
        self.strategies.values().collect()
    }

    /// Prune weak / stale strategies that no longer contribute value.
    ///
    /// Returns the IDs of strategies that were removed.
    pub fn prune_weak_strategies(
        &mut self,
        min_confidence: f32,
        min_support: u32,
        max_stale_hours: f32,
    ) -> Vec<StrategyId> {
        let now = current_timestamp();
        let hour_ns = 3_600_000_000_000u64;

        let to_remove: Vec<StrategyId> = self
            .strategies
            .values()
            .filter(|s| {
                let hours_since_use = (now.saturating_sub(s.last_used) / hour_ns) as f32;

                // Remove if BOTH low confidence and low support
                let weak = s.confidence < min_confidence && s.support_count < min_support;
                // Remove if stale AND weak
                let stale_and_weak =
                    hours_since_use > max_stale_hours && s.support_count < min_support;

                weak || stale_and_weak
            })
            .map(|s| s.id)
            .collect();

        for id in &to_remove {
            if let Some(strategy) = self.strategies.remove(id) {
                // Clean indexes
                if let Some(ids) = self.agent_strategies.get_mut(&strategy.agent_id) {
                    ids.retain(|sid| sid != id);
                }
                if let Some(ids) = self.goal_bucket_index.get_mut(&strategy.goal_bucket_id) {
                    ids.retain(|sid| sid != id);
                }
                if let Some(ids) = self.behavior_index.get_mut(&strategy.behavior_signature) {
                    ids.retain(|sid| sid != id);
                }
                // Remove from signature index
                self.strategy_signature_index.retain(|_, v| v != id);
                // Remove from episode index
                self.episode_index.retain(|_, v| v != id);
            }
        }

        if !to_remove.is_empty() {
            // Clean accumulator maps: collect surviving (agent_id, goal_bucket_id) pairs
            let surviving_keys: std::collections::HashSet<(AgentId, u64)> = self
                .strategies
                .values()
                .map(|s| (s.agent_id, s.goal_bucket_id))
                .collect();

            // Remove entries from context_counts whose (agent_id, goal_bucket_id) is gone
            self.context_counts.retain(|&(agent_id, bucket_id, _), _| {
                surviving_keys.contains(&(agent_id, bucket_id))
            });

            // Remove entries from goal_bucket_counts whose key is gone
            self.goal_bucket_counts
                .retain(|key, _| surviving_keys.contains(key));

            // Remove entries from motif_stats_by_bucket whose key is gone
            self.motif_stats_by_bucket
                .retain(|key, _| surviving_keys.contains(key));

            // Remove entries from episode_cache_by_bucket whose key is gone
            self.episode_cache_by_bucket
                .retain(|key, _| surviving_keys.contains(key));

            // Cap episode_outcomes at 10,000 entries (trim oldest by episode_id)
            if self.episode_outcomes.len() > 10_000 {
                let mut ids: Vec<EpisodeId> = self.episode_outcomes.keys().copied().collect();
                ids.sort_unstable();
                let cutoff = ids[ids.len() - 10_000];
                self.episode_outcomes.retain(|&id, _| id >= cutoff);
            }

            // Clean context_index: remove pruned strategy IDs from vectors, drop empty vectors
            self.context_index.retain(|_, ids| {
                ids.retain(|id| self.strategies.contains_key(id));
                !ids.is_empty()
            });

            tracing::info!(
                "Strategy pruning removed {} weak/stale strategies",
                to_remove.len()
            );
        }

        to_remove
    }

    /// Merge near-duplicate strategies within the same goal bucket.
    ///
    /// Two strategies are "near-duplicates" if they share the same agent, goal bucket,
    /// strategy type, AND have overlapping behavior signatures in the behavior_index.
    /// The weaker strategy is merged into the stronger one (support counts transferred).
    ///
    /// Returns the number of strategies merged (removed).
    pub fn merge_similar_strategies(&mut self) -> usize {
        let mut merged = 0usize;
        let mut to_merge: Vec<(StrategyId, StrategyId)> = Vec::new(); // (victim, survivor)

        // Group strategies by (agent_id, goal_bucket_id, strategy_type)
        let mut groups: FxHashMap<(AgentId, u64, StrategyType), Vec<StrategyId>> =
            FxHashMap::default();
        for s in self.strategies.values() {
            groups
                .entry((s.agent_id, s.goal_bucket_id, s.strategy_type))
                .or_default()
                .push(s.id);
        }

        for ids in groups.values() {
            if ids.len() < 2 {
                continue;
            }
            // Compare all pairs within the group
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let id_a = ids[i];
                    let id_b = ids[j];
                    let (a, b) = match (self.strategies.get(&id_a), self.strategies.get(&id_b)) {
                        (Some(a), Some(b)) => (a, b),
                        _ => continue,
                    };

                    // Word-level Jaccard on action_hint
                    let a_words: HashSet<&str> = a.action_hint.split_whitespace().collect();
                    let b_words: HashSet<&str> = b.action_hint.split_whitespace().collect();
                    let intersection = a_words.intersection(&b_words).count() as f32;
                    let union = a_words.union(&b_words).count() as f32;
                    let jaccard = if union > 0.0 {
                        intersection / union
                    } else {
                        0.0
                    };

                    if jaccard >= 0.70 {
                        // Merge weaker into stronger
                        let (victim, survivor) = if a.quality_score >= b.quality_score {
                            (id_b, id_a)
                        } else {
                            (id_a, id_b)
                        };
                        to_merge.push((victim, survivor));
                    }
                }
            }
        }

        // Deduplicate merge pairs (a victim can only be merged once)
        let mut already_merged = HashSet::new();
        for (victim, survivor) in to_merge {
            if already_merged.contains(&victim) || already_merged.contains(&survivor) {
                continue;
            }
            already_merged.insert(victim);

            // Transfer counts
            if let Some(victim_strategy) = self.strategies.remove(&victim) {
                if let Some(survivor_strategy) = self.strategies.get_mut(&survivor) {
                    survivor_strategy.support_count += victim_strategy.support_count;
                    survivor_strategy.success_count += victim_strategy.success_count;
                    survivor_strategy.failure_count += victim_strategy.failure_count;

                    // Record supersession
                    survivor_strategy.supersedes.push(victim);

                    // Recalculate confidence
                    let total = survivor_strategy.support_count as f32;
                    survivor_strategy.confidence = 1.0 - (-(total / 3.0)).exp();
                    survivor_strategy.expected_success =
                        (survivor_strategy.success_count as f32 + 1.0) / (total + 2.0);
                }

                // Clean indexes for victim
                if let Some(ids) = self.agent_strategies.get_mut(&victim_strategy.agent_id) {
                    ids.retain(|sid| *sid != victim);
                }
                if let Some(ids) = self
                    .goal_bucket_index
                    .get_mut(&victim_strategy.goal_bucket_id)
                {
                    ids.retain(|sid| *sid != victim);
                }
                if let Some(ids) = self
                    .behavior_index
                    .get_mut(&victim_strategy.behavior_signature)
                {
                    ids.retain(|sid| *sid != victim);
                }
                self.strategy_signature_index.retain(|_, v| *v != victim);
                self.episode_index.retain(|_, v| *v != victim);

                merged += 1;
            }
        }

        if merged > 0 {
            tracing::info!(
                "Strategy merge: combined {} near-duplicate strategies",
                merged
            );
        }
        merged
    }

    /// Insert a strategy loaded from persistent storage
    ///
    /// This is used to restore strategies from disk without going through
    /// the extraction process. Used during initialization.
    pub fn insert_loaded_strategy(&mut self, strategy: Strategy) -> Result<(), crate::GraphError> {
        let strategy_id = strategy.id;
        let agent_id = strategy.agent_id;
        let goal_bucket_id = strategy.goal_bucket_id;
        let behavior_signature = strategy.behavior_signature.clone();

        // Update next_strategy_id if needed
        if strategy_id >= self.next_strategy_id {
            self.next_strategy_id = strategy_id + 1;
        }

        // Store strategy
        self.strategies.insert(strategy_id, strategy);

        // Index by agent
        self.agent_strategies
            .entry(agent_id)
            .or_default()
            .push(strategy_id);

        // Index by goal bucket
        self.goal_bucket_index
            .entry(goal_bucket_id)
            .or_default()
            .push(strategy_id);

        // Index by behavior signature
        self.behavior_index
            .entry(behavior_signature)
            .or_default()
            .push(strategy_id);

        Ok(())
    }
}

fn outcome_to_counts(outcome: Option<&EpisodeOutcome>) -> (u32, u32) {
    match outcome {
        Some(EpisodeOutcome::Success) | Some(EpisodeOutcome::Partial) => (1, 0),
        Some(EpisodeOutcome::Failure) => (0, 1),
        Some(EpisodeOutcome::Interrupted) | None => (0, 0),
    }
}

use crate::event_content::{
    extract_action_description, extract_cognitive_summary, extract_communication_summary,
    extract_context_summary, extract_observation_summary,
};

/// Truncate a string to `max_len` chars, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

/// Synthesize a natural language strategy summary from an episode and its events.
///
/// The summary describes *when* to use this strategy, *what steps* to follow,
/// *what success looks like*, and *what to avoid* — all in plain English an LLM can use.
pub fn synthesize_strategy_summary(
    strategy_type: &StrategyType,
    outcome: &EpisodeOutcome,
    episode: &Episode,
    events: &[Event],
    success_indicators: &[String],
    failure_patterns: &[String],
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // 1. Strategy type framing — derive from goals + actions, not generic templates
    let goal_desc: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    match strategy_type {
        StrategyType::Positive => {
            if !goal_desc.is_empty() {
                parts.push(format!("Strategy for achieving: {}", goal_desc.join("; ")));
            } else {
                parts.push("Proven approach from a successful episode.".to_string());
            }
        },
        StrategyType::Constraint => {
            if !goal_desc.is_empty() {
                parts.push(format!(
                    "Constraint: avoid this pattern when pursuing: {}",
                    goal_desc.join("; ")
                ));
            } else {
                parts.push("Constraint: avoid this pattern (led to failure).".to_string());
            }
        },
    }

    // 2. When — goals and context
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();
    if !goals.is_empty() {
        parts.push(format!("When: {}", goals.join("; ")));
    }

    let env_vars = &episode.context.environment.variables;
    let mut context_hints = Vec::new();
    if let Some(intent) = env_vars
        .get("intent_type")
        .or_else(|| env_vars.get("intent"))
    {
        context_hints.push(format!("intent={}", intent));
    }
    for (k, v) in env_vars.iter().take(3) {
        if k != "intent_type" && k != "intent" && k != "user_id" && k != "user" {
            context_hints.push(format!("{}={}", k, v));
        }
    }
    if !context_hints.is_empty() {
        parts.push(format!("Context: {}", context_hints.join(", ")));
    }

    // 3. Steps — walk events and produce a human-readable sequence
    let mut steps: Vec<String> = Vec::new();
    let mut step_num = 1u32;
    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                parameters,
                outcome: action_outcome,
                ..
            } => {
                let desc = extract_action_description(action_name, parameters, action_outcome);
                steps.push(format!("{}. {}", step_num, desc));
                step_num += 1;
            },
            EventType::Context {
                context_type, text, ..
            } => {
                steps.push(format!(
                    "{}. Receive {}",
                    step_num,
                    extract_context_summary(text, context_type)
                ));
                step_num += 1;
            },
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                steps.push(format!(
                    "{}. Observe {}",
                    step_num,
                    extract_observation_summary(observation_type, data, *confidence, source)
                ));
                step_num += 1;
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                steps.push(format!(
                    "{}. {}",
                    step_num,
                    extract_cognitive_summary(process_type, input, output, reasoning_trace)
                ));
                step_num += 1;
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                steps.push(format!(
                    "{}. {}",
                    step_num,
                    extract_communication_summary(message_type, *sender, *recipient, content,)
                ));
                step_num += 1;
            },
            _ => {},
        }
    }
    if !steps.is_empty() {
        // Limit to first 8 steps to keep summary digestible
        let shown: Vec<&str> = steps.iter().take(8).map(|s| s.as_str()).collect();
        let suffix = if steps.len() > 8 {
            format!(" ... ({} more steps)", steps.len() - 8)
        } else {
            String::new()
        };
        parts.push(format!("Steps: {}{}", shown.join(" → "), suffix));
    }

    // 4. Success indicators
    if !success_indicators.is_empty() {
        let shown: Vec<&str> = success_indicators
            .iter()
            .take(3)
            .map(|s| s.as_str())
            .collect();
        parts.push(format!("Success looks like: {}", shown.join("; ")));
    }

    // 5. Failure patterns
    if !failure_patterns.is_empty() {
        let shown: Vec<&str> = failure_patterns
            .iter()
            .take(3)
            .map(|s| s.as_str())
            .collect();
        parts.push(format!("Avoid: {}", shown.join("; ")));
    }

    // 6. Outcome + stats
    let outcome_label = match outcome {
        EpisodeOutcome::Success => "succeeded",
        EpisodeOutcome::Failure => "failed",
        EpisodeOutcome::Partial => "partially succeeded",
        EpisodeOutcome::Interrupted => "was interrupted",
    };
    parts.push(format!(
        "This episode {} ({} events, significance {:.0}%).",
        outcome_label,
        events.len(),
        episode.significance * 100.0
    ));

    parts.join(" ")
}

// ============================================================================
// 10x / 100x Synthesis Functions
// ============================================================================

/// Generate "when to use" in natural language.
fn synthesize_when_to_use(
    strategy_type: &StrategyType,
    episode: &Episode,
    events: &[Event],
) -> String {
    let goals: Vec<&str> = episode
        .context
        .active_goals
        .iter()
        .map(|g| g.description.as_str())
        .filter(|d| !d.is_empty())
        .collect();

    let env_hints: Vec<String> = episode
        .context
        .environment
        .variables
        .iter()
        .filter(|(k, _)| k != &"user_id" && k != &"user")
        .take(3)
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();

    // Find top cognitive reasoning if available
    let cognitive_hint: Option<String> = events.iter().find_map(|e| {
        if let EventType::Cognitive {
            process_type,
            input,
            output,
            reasoning_trace,
        } = &e.event_type
        {
            Some(extract_cognitive_summary(
                process_type,
                input,
                output,
                reasoning_trace,
            ))
        } else {
            None
        }
    });

    match strategy_type {
        StrategyType::Positive => {
            let mut parts = Vec::new();
            if !goals.is_empty() {
                parts.push(format!("Use when the goal is: {}", goals.join("; ")));
            }
            if !env_hints.is_empty() {
                parts.push(format!("Context matches: {}", env_hints.join(", ")));
            }
            if let Some(reasoning) = cognitive_hint {
                parts.push(format!("Agent reasoning: {}", reasoning));
            }
            if parts.is_empty() {
                parts.push("Use when facing a similar task context.".to_string());
            }
            parts.join(". ")
        },
        StrategyType::Constraint => {
            let mut parts = Vec::new();
            if !goals.is_empty() {
                parts.push(format!("Watch out when the goal is: {}", goals.join("; ")));
            }
            if let Some(reasoning) = cognitive_hint {
                parts.push(format!(
                    "Agent reasoning that led to failure: {}",
                    reasoning
                ));
            }
            parts.push(
                "Applies when the agent is about to repeat a pattern that previously failed"
                    .to_string(),
            );
            parts.join(". ")
        },
    }
}

/// Generate "when NOT to use" in natural language.
fn synthesize_when_not_to_use(
    strategy_type: &StrategyType,
    episode: &Episode,
    events: &[Event],
) -> String {
    match strategy_type {
        StrategyType::Positive => {
            let mut reasons = Vec::new();
            // Identify specific failure points and their context
            for event in events {
                if let EventType::Action {
                    action_name,
                    parameters,
                    outcome: ActionOutcome::Failure { error, .. },
                    ..
                } = &event.event_type
                {
                    let desc = extract_action_description(
                        action_name,
                        parameters,
                        &ActionOutcome::Failure {
                            error: error.clone(),
                            error_code: 0,
                        },
                    );
                    reasons.push(format!(
                        "Fragile when: {}",
                        truncate_str(&desc, 120)
                    ));
                }
            }
            if episode.significance < 0.3 {
                reasons.push("Low-significance episode — may not generalize to other contexts".to_string());
            }
            let goals: Vec<&str> = episode
                .context
                .active_goals
                .iter()
                .map(|g| g.description.as_str())
                .filter(|d| !d.is_empty())
                .collect();
            if !goals.is_empty() {
                reasons.push(format!(
                    "Do not use when goals differ significantly from: {}",
                    goals.join("; ")
                ));
            } else {
                reasons.push(
                    "Do not use when the context significantly differs from the original episode"
                        .to_string(),
                );
            }
            reasons.join(". ")
        },
        StrategyType::Constraint => {
            "Safe to ignore when a newer strategy explicitly supersedes this constraint, or when the failure pattern is no longer applicable to the current context."
                .to_string()
        },
    }
}

/// Extract known failure modes from event data.
fn synthesize_failure_modes(events: &[Event]) -> Vec<String> {
    let mut modes = Vec::new();
    for event in events {
        if let EventType::Action {
            action_name,
            parameters,
            outcome: outcome @ ActionOutcome::Failure { .. },
            ..
        } = &event.event_type
        {
            modes.push(extract_action_description(action_name, parameters, outcome));
        }
        if let EventType::Action {
            action_name,
            parameters,
            outcome: outcome @ ActionOutcome::Partial { .. },
            ..
        } = &event.event_type
        {
            modes.push(extract_action_description(action_name, parameters, outcome));
        }
    }
    modes.truncate(5); // Keep top 5
    modes
}

/// Build an executable playbook from the event sequence (100x).
fn build_playbook(events: &[Event], strategy_type: &StrategyType) -> Vec<PlaybookStep> {
    let mut steps = Vec::new();
    let mut step_num = 1u32;

    for event in events {
        match &event.event_type {
            EventType::Action {
                action_name,
                parameters,
                outcome: action_outcome,
                ..
            } => {
                let mut branches = Vec::new();
                let mut recovery = String::new();
                let action_desc =
                    extract_action_description(action_name, parameters, action_outcome);

                match action_outcome {
                    ActionOutcome::Failure { error, .. } => {
                        recovery = format!(
                            "On failure ({}): retry or use alternative approach",
                            truncate_str(error, 80)
                        );
                    },
                    ActionOutcome::Partial { issues, .. } => {
                        recovery = format!(
                            "On partial success: address {:?}",
                            issues.iter().take(2).cloned().collect::<Vec<_>>()
                        );
                    },
                    ActionOutcome::Success { .. } => {
                        // For constraint strategies, add a branch to avoid this action
                        if *strategy_type == StrategyType::Constraint {
                            branches.push(PlaybookBranch {
                                condition: "If this pattern appears".to_string(),
                                action: format!("Skip '{}' and use alternative", action_name),
                                next_step_id: None,
                            });
                        }
                    },
                }

                steps.push(PlaybookStep {
                    step: step_num,
                    action: action_desc,
                    condition: String::new(),
                    skip_if: String::new(),
                    branches,
                    recovery,
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Context {
                context_type, text, ..
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: format!("Receive {}", extract_context_summary(text, context_type)),
                    condition: String::new(),
                    skip_if: "No input available".to_string(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: format!(
                        "Observe {}",
                        extract_observation_summary(observation_type, data, *confidence, source)
                    ),
                    condition: String::new(),
                    skip_if: String::new(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: extract_cognitive_summary(process_type, input, output, reasoning_trace),
                    condition: String::new(),
                    skip_if: String::new(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                steps.push(PlaybookStep {
                    step: step_num,
                    action: extract_communication_summary(
                        message_type,
                        *sender,
                        *recipient,
                        content,
                    ),
                    condition: String::new(),
                    skip_if: String::new(),
                    branches: Vec::new(),
                    recovery: String::new(),
                    step_id: String::new(),
                    next_step_id: None,
                });
                step_num += 1;
            },
            _ => {},
        }
    }

    // Cap at 12 steps
    steps.truncate(12);
    steps
}

/// Generate a counterfactual: what would have happened differently.
fn synthesize_counterfactual(outcome: &EpisodeOutcome, events: &[Event]) -> String {
    // Find failure points with descriptive context
    let failures: Vec<String> = events
        .iter()
        .filter_map(|e| {
            if let EventType::Action {
                action_name,
                parameters,
                outcome: outcome @ ActionOutcome::Failure { .. },
                ..
            } = &e.event_type
            {
                Some(extract_action_description(action_name, parameters, outcome))
            } else {
                None
            }
        })
        .collect();

    // Find cognitive reasoning that explains what was tried
    let reasoning: Option<String> = events.iter().find_map(|e| {
        if let EventType::Cognitive {
            process_type,
            input,
            output,
            reasoning_trace,
        } = &e.event_type
        {
            Some(extract_cognitive_summary(
                process_type,
                input,
                output,
                reasoning_trace,
            ))
        } else {
            None
        }
    });

    let mut result = match outcome {
        EpisodeOutcome::Success => {
            if failures.is_empty() {
                "All actions succeeded; no obvious alternative path needed.".to_string()
            } else {
                format!(
                    "Despite failures ({}), the episode recovered. Skipping those steps could have been faster.",
                    failures.iter().take(2).cloned().collect::<Vec<_>>().join("; ")
                )
            }
        },
        EpisodeOutcome::Failure => {
            if failures.is_empty() {
                "Episode failed without clear action errors; the approach may need rethinking."
                    .to_string()
            } else {
                format!(
                    "If these had been handled differently: {}. Retry, alternative, or skip could lead to success.",
                    failures.iter().take(2).cloned().collect::<Vec<_>>().join("; ")
                )
            }
        },
        EpisodeOutcome::Partial => {
            "A more conservative approach (smaller steps, more validation) could have improved completeness.".to_string()
        },
        EpisodeOutcome::Interrupted => {
            "Ensuring preconditions and resources before starting would reduce interruption risk.".to_string()
        },
    };

    if let Some(reasoning) = reasoning {
        result.push_str(&format!(" Agent's reasoning: {}", reasoning));
    }

    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_bucket_id_differs_for_different_goal_sets_with_same_min() {
        // Both sets share the same minimum goal id (1), but differ in the full set.
        let bucket_a = compute_goal_bucket_id_from_ids(&[1, 2, 3]);
        let bucket_b = compute_goal_bucket_id_from_ids(&[1, 4, 5]);

        assert_ne!(
            bucket_a, bucket_b,
            "Different goal sets with the same min id must produce different bucket ids"
        );
    }

    #[test]
    fn test_goal_bucket_id_identical_for_same_goals() {
        let bucket_a = compute_goal_bucket_id_from_ids(&[3, 1, 2]);
        let bucket_b = compute_goal_bucket_id_from_ids(&[1, 2, 3]);

        assert_eq!(
            bucket_a, bucket_b,
            "Identical goal sets (regardless of order) must produce the same bucket id"
        );
    }

    #[test]
    fn test_goal_bucket_id_empty_returns_zero() {
        assert_eq!(compute_goal_bucket_id_from_ids(&[]), 0);
    }

    #[test]
    fn test_goal_bucket_id_matches_event_context() {
        use agent_db_events::core::{EventContext, Goal};

        let goals = vec![
            Goal {
                id: 10,
                description: String::new(),
                priority: 0.5,
                deadline: None,
                progress: 0.0,
                subgoals: Vec::new(),
            },
            Goal {
                id: 3,
                description: String::new(),
                priority: 0.9,
                deadline: None,
                progress: 0.0,
                subgoals: Vec::new(),
            },
        ];

        let ctx = EventContext::new(Default::default(), goals.clone(), Default::default());

        let strategy_bucket = compute_goal_bucket_id_from_ids(&[10, 3]);

        assert_eq!(
            ctx.goal_bucket_id, strategy_bucket,
            "Strategy bucket id must match EventContext.goal_bucket_id for the same goals"
        );
    }

    fn make_test_episode(outcome: EpisodeOutcome) -> Episode {
        use agent_db_core::types::Timestamp;
        use agent_db_events::core::EventContext;
        Episode {
            id: 1,
            episode_version: 1,
            agent_id: 1,
            start_event: 1u128,
            end_event: Some(2u128),
            events: vec![1u128, 2u128],
            session_id: 1,
            context_signature: 0,
            context: EventContext::default(),
            outcome: Some(outcome.clone()),
            start_timestamp: Timestamp::from(1000u64),
            end_timestamp: Some(Timestamp::from(2000u64)),
            significance: 0.8,
            prediction_error: 0.0,
            self_judged_quality: None,
            salience_score: 0.5,
            last_event_timestamp: Some(Timestamp::from(2000u64)),
            consecutive_outcome_count: 0,
        }
    }

    fn make_test_event(event_type: EventType) -> Event {
        use agent_db_core::types::Timestamp;
        Event {
            id: 1u128,
            timestamp: Timestamp::from(1000u64),
            agent_id: 1,
            agent_type: "test".to_string(),
            session_id: 1,
            event_type,
            causality_chain: vec![],
            context: agent_db_events::core::EventContext::default(),
            metadata: Default::default(),
            context_size_bytes: 0,
            segment_pointer: None,
        }
    }

    #[test]
    fn test_playbook_not_execute_raw() {
        let events = vec![make_test_event(EventType::Action {
            action_name: "cognitive_plan".to_string(),
            parameters: serde_json::json!({"query": "plan deployment"}),
            outcome: ActionOutcome::Success {
                result: serde_json::json!({"text": "plan created"}),
            },
            duration_ns: 1000,
        })];

        let playbook = build_playbook(&events, &StrategyType::Positive);
        assert!(!playbook.is_empty());
        // Should NOT contain "Execute 'cognitive_plan'"
        assert!(
            !playbook[0].action.starts_with("Execute"),
            "Playbook action should not be raw 'Execute', got: {}",
            playbook[0].action
        );
        // Should contain humanized description
        assert!(
            playbook[0].action.contains("Cognitive Plan"),
            "Playbook action should contain humanized name, got: {}",
            playbook[0].action
        );
    }

    #[test]
    fn test_summary_not_do_this() {
        let episode = make_test_episode(EpisodeOutcome::Success);
        let events = vec![make_test_event(EventType::Action {
            action_name: "search".to_string(),
            parameters: serde_json::json!({"query": "find users"}),
            outcome: ActionOutcome::Success {
                result: serde_json::json!({"text": "found 10 users"}),
            },
            duration_ns: 1000,
        })];

        let summary = synthesize_strategy_summary(
            &StrategyType::Positive,
            &EpisodeOutcome::Success,
            &episode,
            &events,
            &[],
            &[],
        );
        // Should NOT start with generic "DO this when applicable"
        assert!(
            !summary.contains("DO this when applicable"),
            "Summary should not contain generic template, got: {}",
            summary
        );
    }

    #[test]
    fn test_cognitive_in_playbook() {
        let events = vec![make_test_event(EventType::Cognitive {
            process_type: agent_db_events::core::CognitiveType::Reasoning,
            input: serde_json::json!("analyze options"),
            output: serde_json::json!("chose option A"),
            reasoning_trace: vec!["step1".to_string()],
        })];

        let playbook = build_playbook(&events, &StrategyType::Positive);
        assert!(!playbook.is_empty());
        // Should NOT contain "Think (Reasoning)" — should have actual content
        assert!(
            !playbook[0].action.starts_with("Think ("),
            "Playbook cognitive step should not be generic 'Think', got: {}",
            playbook[0].action
        );
        assert!(
            playbook[0].action.contains("Reasoning"),
            "Should contain reasoning label, got: {}",
            playbook[0].action
        );
        assert!(
            playbook[0].action.contains("chose option A"),
            "Should contain cognitive output, got: {}",
            playbook[0].action
        );
    }

    #[test]
    fn test_strategy_ema_q_value() {
        let mut extractor = StrategyExtractor::new(Default::default());
        // Create a minimal strategy
        let strategy = Strategy {
            id: 1,
            name: "test_strategy".to_string(),
            summary: String::new(),
            when_to_use: String::new(),
            when_not_to_use: String::new(),
            failure_modes: Vec::new(),
            playbook: Vec::new(),
            counterfactual: String::new(),
            supersedes: Vec::new(),
            applicable_domains: Vec::new(),
            lineage_depth: 0,
            summary_embedding: Vec::new(),
            agent_id: 1,
            reasoning_steps: Vec::new(),
            context_patterns: Vec::new(),
            success_indicators: Vec::new(),
            failure_patterns: Vec::new(),
            quality_score: 0.0,
            success_count: 0,
            failure_count: 0,
            support_count: 0,
            strategy_type: StrategyType::Positive,
            precondition: String::new(),
            action_hint: String::new(),
            expected_success: 0.5,
            expected_cost: 0.0,
            expected_value: 0.0,
            confidence: 0.0,
            contradictions: Vec::new(),
            goal_bucket_id: 1,
            behavior_signature: String::new(),
            source_episodes: Vec::new(),
            created_at: current_timestamp(),
            last_used: current_timestamp(),
            metadata: HashMap::new(),
            self_judged_quality: None,
            source_outcomes: Vec::new(),
            version: 1,
            parent_strategy: None,
        };
        extractor.strategies.insert(1, strategy);

        // Record 3 successes (Bayesian phase)
        extractor.update_strategy_outcome(1, true).unwrap();
        extractor.update_strategy_outcome(1, true).unwrap();
        extractor.update_strategy_outcome(1, true).unwrap();

        let s = extractor.strategies.get(&1).unwrap();
        assert_eq!(s.success_count, 3);
        assert_eq!(s.failure_count, 0);
        // Q-value should be in metadata
        assert!(s.metadata.contains_key(META_Q_VALUE));
        let q: f32 = s.metadata.get(META_Q_VALUE).unwrap().parse().unwrap();
        assert!(
            q > 0.5,
            "Q should be above 0.5 after 3 successes, got {}",
            q
        );
        // Confidence should reflect both sample size and outcome quality
        assert!(s.confidence > 0.0, "Confidence should be positive");

        // Record 3 more (crosses Q_KICK_IN=5)
        extractor.update_strategy_outcome(1, true).unwrap();
        extractor.update_strategy_outcome(1, true).unwrap();
        extractor.update_strategy_outcome(1, false).unwrap();

        let s = extractor.strategies.get(&1).unwrap();
        assert_eq!(s.success_count, 5);
        assert_eq!(s.failure_count, 1);
        // Now in Q phase — confidence should reflect outcome quality
        let q: f32 = s.metadata.get(META_Q_VALUE).unwrap().parse().unwrap();
        assert!(q > 0.5, "Q should still be >0.5, got {}", q);
        // Confidence = sample_confidence * piecewise_score, should be substantial
        assert!(
            s.confidence > 0.3,
            "Confidence should be substantial, got {}",
            s.confidence
        );
    }
}
