//! Deterministic mock generators for testing.
//!
//! These produce fixed, predictable candidates without any LLM calls,
//! enabling unit tests for the full pipeline from day one.

use crate::types::{
    ActionGenerationRequest, Criteria, ExpectedEvent, GeneratedActionPlan, GeneratedStep,
    GeneratedStrategyPlan, PlanningError, RiskFlag, RiskSeverity, StepKind,
    StrategyGenerationRequest,
};
use crate::{ActionGenerator, StrategyGenerator};
use agent_db_world_model::{CriticReport, PredictionErrorReport};

/// Mock strategy generator that returns deterministic candidates.
pub struct MockStrategyGenerator {
    /// Number of candidates to return (ignores request.k_candidates).
    pub fixed_k: usize,
    /// If true, returns an error instead of candidates.
    pub should_fail: bool,
}

impl MockStrategyGenerator {
    pub fn new(fixed_k: usize) -> Self {
        Self {
            fixed_k,
            should_fail: false,
        }
    }

    pub fn failing() -> Self {
        Self {
            fixed_k: 0,
            should_fail: true,
        }
    }
}

#[async_trait::async_trait]
impl StrategyGenerator for MockStrategyGenerator {
    async fn generate(
        &self,
        request: StrategyGenerationRequest,
    ) -> Result<Vec<GeneratedStrategyPlan>, PlanningError> {
        if self.should_fail {
            return Err(PlanningError::GenerationFailed(
                "mock generator configured to fail".to_string(),
            ));
        }

        let mut candidates = Vec::with_capacity(self.fixed_k);
        for i in 0..self.fixed_k {
            candidates.push(make_mock_strategy(
                &request.goal_description,
                request.goal_bucket_id,
                i as u32,
            ));
        }
        Ok(candidates)
    }

    async fn revise(
        &self,
        candidate: &GeneratedStrategyPlan,
        _diagnostics: &CriticReport,
        _request: &StrategyGenerationRequest,
    ) -> Result<Vec<GeneratedStrategyPlan>, PlanningError> {
        // Mock revision: just return the candidate with slightly higher confidence
        let mut revised = candidate.clone();
        revised.confidence = (revised.confidence + 0.1).min(1.0);
        revised.rationale = Some("Revised by mock generator".to_string());
        Ok(vec![revised])
    }
}

/// Mock action generator that returns deterministic candidates.
pub struct MockActionGenerator {
    pub fixed_n: usize,
    pub should_fail: bool,
}

impl MockActionGenerator {
    pub fn new(fixed_n: usize) -> Self {
        Self {
            fixed_n,
            should_fail: false,
        }
    }

    pub fn failing() -> Self {
        Self {
            fixed_n: 0,
            should_fail: true,
        }
    }
}

#[async_trait::async_trait]
impl ActionGenerator for MockActionGenerator {
    async fn generate(
        &self,
        request: ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError> {
        if self.should_fail {
            return Err(PlanningError::GenerationFailed(
                "mock action generator configured to fail".to_string(),
            ));
        }

        let step = request
            .strategy
            .steps
            .get(request.current_step_index)
            .ok_or_else(|| {
                PlanningError::GenerationFailed(format!(
                    "step index {} out of bounds",
                    request.current_step_index
                ))
            })?;

        let mut candidates = Vec::with_capacity(self.fixed_n);
        for i in 0..self.fixed_n {
            candidates.push(make_mock_action(&step.action_type, i as u32));
        }
        Ok(candidates)
    }

    async fn repair(
        &self,
        current_step: &GeneratedStep,
        _error_report: &PredictionErrorReport,
        _request: &ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError> {
        // Mock repair: return a single recovery action
        Ok(vec![make_mock_action(
            &format!("repair_{}", current_step.action_type),
            0,
        )])
    }
}

// ────────────────────────────────────────────────────────────────
// Factory helpers
// ────────────────────────────────────────────────────────────────

fn make_mock_strategy(
    goal_description: &str,
    goal_bucket_id: u64,
    variant: u32,
) -> GeneratedStrategyPlan {
    GeneratedStrategyPlan {
        goal_bucket_id,
        goal_description: goal_description.to_string(),
        steps: vec![
            GeneratedStep {
                step_number: 1,
                step_kind: StepKind::Validation,
                action_type: "check_preconditions".to_string(),
                parameters: serde_json::json!({"variant": variant}),
                description: Some("Validate preconditions".to_string()),
                precondition: None,
                success_criteria: Some(Criteria {
                    description: "All preconditions met".to_string(),
                    check_type: "event_type_match".to_string(),
                    parameters: serde_json::json!({"expected": "success"}),
                }),
                failure_criteria: None,
                skip_if: None,
                max_retries: 0,
                timeout_ms: Some(5000),
                branches: vec![],
                recovery: None,
            },
            GeneratedStep {
                step_number: 2,
                step_kind: StepKind::Action,
                action_type: format!("execute_main_{}", variant),
                parameters: serde_json::json!({"goal": goal_description, "variant": variant}),
                description: Some(format!("Execute main action (variant {})", variant)),
                precondition: None,
                success_criteria: Some(Criteria {
                    description: "Action completed successfully".to_string(),
                    check_type: "event_type_match".to_string(),
                    parameters: serde_json::json!({"expected": "success"}),
                }),
                failure_criteria: Some(Criteria {
                    description: "Action failed".to_string(),
                    check_type: "event_type_match".to_string(),
                    parameters: serde_json::json!({"expected": "failure"}),
                }),
                skip_if: None,
                max_retries: 1,
                timeout_ms: Some(30000),
                branches: vec![],
                recovery: Some("fallback_action".to_string()),
            },
            GeneratedStep {
                step_number: 3,
                step_kind: StepKind::Observation,
                action_type: "verify_result".to_string(),
                parameters: serde_json::json!({}),
                description: Some("Verify the result".to_string()),
                precondition: None,
                success_criteria: None,
                failure_criteria: None,
                skip_if: None,
                max_retries: 0,
                timeout_ms: Some(10000),
                branches: vec![],
                recovery: None,
            },
        ],
        preconditions: vec![Criteria {
            description: "System is ready".to_string(),
            check_type: "value_range".to_string(),
            parameters: serde_json::json!({"field": "status", "value": "ready"}),
        }],
        stop_conditions: vec![Criteria {
            description: "Goal achieved".to_string(),
            check_type: "event_type_match".to_string(),
            parameters: serde_json::json!({"expected": "goal_achieved"}),
        }],
        fallback_steps: vec![GeneratedStep {
            step_number: 100,
            step_kind: StepKind::Recovery,
            action_type: "fallback_action".to_string(),
            parameters: serde_json::json!({}),
            description: Some("Fallback recovery".to_string()),
            precondition: None,
            success_criteria: None,
            failure_criteria: None,
            skip_if: None,
            max_retries: 0,
            timeout_ms: None,
            branches: vec![],
            recovery: None,
        }],
        risk_flags: vec![RiskFlag {
            description: format!("Mock risk for variant {}", variant),
            severity: RiskSeverity::Low,
            mitigation: Some("No real risk (mock)".to_string()),
        }],
        assumptions: vec!["Mock assumption: system is available".to_string()],
        confidence: 0.7 + (variant as f32 * 0.05),
        rationale: Some(format!(
            "Mock strategy variant {} for goal: {}",
            variant, goal_description
        )),
    }
}

fn make_mock_action(action_type: &str, variant: u32) -> GeneratedActionPlan {
    GeneratedActionPlan {
        action_type: action_type.to_string(),
        parameters: serde_json::json!({"variant": variant}),
        preconditions: vec![],
        expected_event: ExpectedEvent {
            event_type: "Action".to_string(),
            expected_outcome: "success".to_string(),
            expected_significance: 0.7,
        },
        success_criteria: Some(Criteria {
            description: "Action succeeded".to_string(),
            check_type: "event_type_match".to_string(),
            parameters: serde_json::json!({"expected": "success"}),
        }),
        failure_criteria: None,
        timeout_ms: Some(10000),
        max_retries: 1,
        fallback_action: None,
        risk_flags: vec![],
        confidence: 0.8 + (variant as f32 * 0.05),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EventContext, MemoryContext, StrategyContext};

    fn make_strategy_request() -> StrategyGenerationRequest {
        StrategyGenerationRequest {
            goal_description: "Complete the task".to_string(),
            goal_bucket_id: 100,
            context_fingerprint: 12345,
            relevant_memories: vec![MemoryContext {
                summary: "Previous success".to_string(),
                tier: "Episodic".to_string(),
                strength: 0.8,
                takeaway: None,
                causal_note: None,
            }],
            similar_strategies: vec![StrategyContext {
                name: "Old strategy".to_string(),
                summary: "Did stuff".to_string(),
                quality_score: 0.75,
                when_to_use: "When stuff needs doing".to_string(),
            }],
            recent_events: vec![EventContext {
                event_type: "Action".to_string(),
                action_name: Some("init".to_string()),
                outcome: Some("success".to_string()),
                timestamp: 1000,
            }],
            transition_hints: vec![],
            constraints: vec!["Must complete within 30s".to_string()],
            k_candidates: 3,
        }
    }

    #[tokio::test]
    async fn test_mock_strategy_generator_produces_k_candidates() {
        let gen = MockStrategyGenerator::new(3);
        let request = make_strategy_request();
        let candidates = gen.generate(request).await.unwrap();
        assert_eq!(candidates.len(), 3);

        // Each should be valid
        for (i, c) in candidates.iter().enumerate() {
            assert!(!c.steps.is_empty());
            assert!(!c.goal_description.is_empty());
            assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
            assert!(c.rationale.is_some());
            // Variant should be embedded in the plan
            assert!(c
                .rationale
                .as_ref()
                .unwrap()
                .contains(&format!("variant {}", i)),);
        }
    }

    #[tokio::test]
    async fn test_mock_strategy_generator_fails_when_configured() {
        let gen = MockStrategyGenerator::failing();
        let request = make_strategy_request();
        let result = gen.generate(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_action_generator_produces_n_candidates() {
        let gen = MockActionGenerator::new(2);
        let strategy = MockStrategyGenerator::new(1)
            .generate(make_strategy_request())
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let request = ActionGenerationRequest {
            strategy,
            current_step_index: 1,
            recent_events: vec![],
            context_fingerprint: 12345,
            n_candidates: 2,
        };

        let actions = gen.generate(request).await.unwrap();
        assert_eq!(actions.len(), 2);
        for a in &actions {
            assert!(!a.action_type.is_empty());
            assert!(a.confidence >= 0.0 && a.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_mock_revision_increases_confidence() {
        let gen = MockStrategyGenerator::new(1);
        let request = make_strategy_request();
        let candidates = gen.generate(request.clone()).await.unwrap();
        let original = &candidates[0];

        let report = CriticReport {
            total_energy: 1.0,
            policy_strategy_energy: 0.5,
            strategy_memory_energy: 0.3,
            memory_event_energy: 0.2,
            novelty_z: 1.5,
            is_novel: false,
            mismatch_layer: agent_db_world_model::MismatchLayer::None,
            confidence: 0.8,
            support_count: 50,
        };

        let revised = gen.revise(original, &report, &request).await.unwrap();
        assert_eq!(revised.len(), 1);
        assert!(revised[0].confidence > original.confidence);
    }
}
