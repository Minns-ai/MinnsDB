// crates/agent-db-graph/src/strategies.rs
//
// Strategy Extraction Module
//
// Extracts generalizable strategies from successful episodes and reasoning traces,
// enabling agents to reuse proven approaches in similar contexts.

use crate::episodes::{Episode, EpisodeId, EpisodeOutcome};
use crate::GraphResult;
use agent_db_core::types::{current_timestamp, AgentId, ContextHash, Timestamp};
use agent_db_events::core::{CognitiveType, Event, EventType, MetadataValue};
use serde_json::json;
use std::collections::{HashMap, HashSet};

/// Unique identifier for a strategy
pub type StrategyId = u64;

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
    strategies: HashMap<StrategyId, Strategy>,

    /// Strategy index by agent
    agent_strategies: HashMap<AgentId, Vec<StrategyId>>,

    /// Strategy index by context hash
    context_index: HashMap<ContextHash, Vec<StrategyId>>,

    /// Strategy index by goal bucket
    goal_bucket_index: HashMap<u64, Vec<StrategyId>>,

    /// Strategy index by behavior signature
    behavior_index: HashMap<String, Vec<StrategyId>>,

    /// Context counts for novelty estimation
    context_counts: HashMap<(AgentId, u64, ContextHash), u32>,

    /// Goal bucket occurrence counts (per agent)
    goal_bucket_counts: HashMap<(AgentId, u64), u32>,

    /// Motif stats by goal bucket (per agent)
    motif_stats_by_bucket: HashMap<(AgentId, u64), HashMap<String, MotifStats>>,

    /// Episode cache for validation (per agent + goal bucket)
    episode_cache_by_bucket: HashMap<(AgentId, u64), Vec<EpisodeMotifRecord>>,

    /// Strategy signature index to prevent duplicates
    strategy_signature_index: HashMap<String, StrategyId>,

    /// Episode to strategy index (idempotency)
    episode_index: HashMap<EpisodeId, StrategyId>,

    /// Episode outcome tracking for corrections
    episode_outcomes: HashMap<EpisodeId, EpisodeOutcome>,

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
            strategies: HashMap::new(),
            agent_strategies: HashMap::new(),
            context_index: HashMap::new(),
            goal_bucket_index: HashMap::new(),
            behavior_index: HashMap::new(),
            context_counts: HashMap::new(),
            goal_bucket_counts: HashMap::new(),
            motif_stats_by_bucket: HashMap::new(),
            episode_cache_by_bucket: HashMap::new(),
            strategy_signature_index: HashMap::new(),
            episode_index: HashMap::new(),
            episode_outcomes: HashMap::new(),
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
        let goal_bucket_id = if !query.goal_ids.is_empty() {
            *query.goal_ids.iter().min().unwrap_or(&0)
        } else {
            0
        };

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
        let mut goals: Vec<u64> = episode
            .context
            .active_goals
            .iter()
            .map(|goal| goal.id)
            .collect();
        goals.sort();
        goals.first().copied().unwrap_or(0)
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

            let mut strategy = Strategy {
                id: strategy_id,
                name: match strategy_type {
                    StrategyType::Positive => format!("strategy_{}_motif", goal_bucket_id),
                    StrategyType::Constraint => format!("constraint_{}_motif", goal_bucket_id),
                },
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

    /// Update strategy based on new usage outcome
    pub fn update_strategy_outcome(
        &mut self,
        strategy_id: StrategyId,
        success: bool,
    ) -> GraphResult<()> {
        if let Some(strategy) = self.strategies.get_mut(&strategy_id) {
            if success {
                strategy.success_count += 1;
            } else {
                strategy.failure_count += 1;
            }

            // Update quality score
            let total = strategy.success_count + strategy.failure_count;
            if total > 0 {
                strategy.quality_score = strategy.success_count as f32 / total as f32;
            }
            strategy.support_count = total;
            strategy.expected_success =
                (strategy.success_count as f32 + 1.0) / (total as f32 + 2.0);
            strategy.confidence = 1.0 - (-((total as f32) / 3.0)).exp();

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
