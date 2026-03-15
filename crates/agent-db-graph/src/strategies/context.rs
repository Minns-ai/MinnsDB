// crates/agent-db-graph/src/strategies/context.rs
//
// Context pattern extraction, task type inference, and related helpers.

use crate::episodes::EpisodeOutcome;
use crate::GraphResult;
use agent_db_events::core::{Event, EventType};

use super::extractor::StrategyExtractor;
use super::types::ContextPattern;

impl StrategyExtractor {
    /// Extract context patterns from events
    pub(crate) fn extract_context_patterns(&self, events: &[Event]) -> GraphResult<Vec<ContextPattern>> {
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
    pub(crate) fn infer_task_type(&self, events: &[Event]) -> String {
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
    pub(crate) fn extract_resource_constraints(
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
    pub(crate) fn identify_success_indicators(&self, events: &[Event]) -> GraphResult<Vec<String>> {
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
    pub(crate) fn extract_failure_patterns(
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
}
