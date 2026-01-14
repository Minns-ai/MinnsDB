// crates/agent-db-graph/src/strategies.rs
//
// Strategy Extraction Module
//
// Extracts generalizable strategies from successful episodes and reasoning traces,
// enabling agents to reuse proven approaches in similar contexts.

use crate::{GraphResult, GraphError};
use crate::episodes::{Episode, EpisodeOutcome};
use agent_db_core::types::{AgentId, EventId, Timestamp, ContextHash, current_timestamp};
use agent_db_events::core::{Event, EventType, CognitiveType};
use std::collections::HashMap;

/// Unique identifier for a strategy
pub type StrategyId = u64;

/// A generalizable strategy extracted from successful experiences
#[derive(Debug, Clone)]
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

    /// Source episodes this was extracted from
    pub source_episodes: Vec<Episode>,

    /// When this strategy was created
    pub created_at: Timestamp,

    /// Last time this strategy was used
    pub last_used: Timestamp,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// A single reasoning step in a strategy
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
}

impl Default for StrategyExtractionConfig {
    fn default() -> Self {
        Self {
            min_significance: 0.6,
            min_success_rate: 0.7,
            min_occurrences: 3,
            max_strategies_per_agent: 100,
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

    /// Configuration
    config: StrategyExtractionConfig,

    /// Next strategy ID
    next_strategy_id: StrategyId,
}

impl StrategyExtractor {
    /// Create a new strategy extractor
    pub fn new(config: StrategyExtractionConfig) -> Self {
        Self {
            strategies: HashMap::new(),
            agent_strategies: HashMap::new(),
            context_index: HashMap::new(),
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
    ) -> GraphResult<Option<StrategyId>> {
        // Only extract from successful episodes
        if episode.outcome != Some(EpisodeOutcome::Success) {
            return Ok(None);
        }

        // Check significance threshold
        if episode.significance < self.config.min_significance {
            return Ok(None);
        }

        // Extract reasoning traces from cognitive events
        let reasoning_steps = self.extract_reasoning_steps(events)?;

        if reasoning_steps.is_empty() {
            return Ok(None);
        }

        // Extract context patterns
        let context_patterns = self.extract_context_patterns(events)?;

        // Identify success indicators
        let success_indicators = self.identify_success_indicators(events)?;

        // Create strategy
        let strategy_id = self.next_strategy_id;
        self.next_strategy_id += 1;

        let strategy = Strategy {
            id: strategy_id,
            name: format!("strategy_{}_ep_{}", episode.agent_id, episode.id),
            agent_id: episode.agent_id,
            reasoning_steps,
            context_patterns: context_patterns.clone(),
            success_indicators,
            failure_patterns: vec![],
            quality_score: episode.significance,
            success_count: 1,
            failure_count: 0,
            source_episodes: vec![episode.clone()],
            created_at: current_timestamp(),
            last_used: current_timestamp(),
            metadata: HashMap::new(),
        };

        // Store strategy
        self.strategies.insert(strategy_id, strategy);

        // Index by agent
        self.agent_strategies
            .entry(episode.agent_id)
            .or_insert_with(Vec::new)
            .push(strategy_id);

        // Index by context
        for pattern in context_patterns {
            if let Some(context_hash) = self.pattern_to_hash(&pattern) {
                self.context_index
                    .entry(context_hash)
                    .or_insert_with(Vec::new)
                    .push(strategy_id);
            }
        }

        Ok(Some(strategy_id))
    }

    /// Extract reasoning steps from cognitive events
    fn extract_reasoning_steps(&self, events: &[Event]) -> GraphResult<Vec<ReasoningStep>> {
        let mut steps = Vec::new();
        let mut order = 0;

        for event in events {
            if let EventType::Cognitive {
                process_type,
                reasoning_trace,
                ..
            } = &event.event_type
            {
                // Only extract from reasoning events
                if let CognitiveType::Reasoning = process_type {
                    for trace_step in reasoning_trace {
                        steps.push(ReasoningStep {
                            description: trace_step.clone(),
                            applicability: "general".to_string(),
                            expected_outcome: None,
                            sequence_order: order,
                        });
                        order += 1;
                    }
                }
            }
        }

        Ok(steps)
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
        let has_cognitive = events.iter().any(|e| {
            matches!(e.event_type, EventType::Cognitive { .. })
        });

        let has_action = events.iter().any(|e| {
            matches!(e.event_type, EventType::Action { .. })
        });

        if has_cognitive && has_action {
            "problem_solving".to_string()
        } else if has_action {
            "execution".to_string()
        } else {
            "analysis".to_string()
        }
    }

    /// Extract resource constraints from context
    fn extract_resource_constraints(&self, context: &agent_db_events::core::EventContext) -> Vec<String> {
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
            if let EventType::Action { outcome, .. } = &event.event_type {
                if let agent_db_events::core::ActionOutcome::Success { .. } = outcome {
                    indicators.push("action_succeeded".to_string());
                }
            }
        }

        if !indicators.is_empty() {
            indicators.push("episode_completed".to_string());
        }

        Ok(indicators)
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
                self.strategies.values().map(|s| s.quality_score).sum::<f32>()
                    / self.strategies.len() as f32
            } else {
                0.0
            },
        }
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
