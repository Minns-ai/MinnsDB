//! LLM-backed strategy and action generators.
//!
//! These generators use a `PlanningLlmClient` to produce candidates via
//! structured LLM completion, then parse the JSON response into typed plans.

use std::sync::Arc;

use agent_db_world_model::{CriticReport, PredictionErrorReport};

use crate::llm_client::{CompletionRequest, PlanningLlmClient};
use crate::prompts;
use crate::types::{
    ActionGenerationRequest, GeneratedActionPlan, GeneratedStep, GeneratedStrategyPlan,
    PlanningConfig, PlanningError, StrategyGenerationRequest,
};
use crate::{ActionGenerator, StrategyGenerator};

/// LLM-backed strategy generator.
pub struct LlmStrategyGenerator {
    client: Arc<dyn PlanningLlmClient>,
    config: PlanningConfig,
}

impl LlmStrategyGenerator {
    pub fn new(client: Arc<dyn PlanningLlmClient>, config: PlanningConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait::async_trait]
impl StrategyGenerator for LlmStrategyGenerator {
    async fn generate(
        &self,
        request: StrategyGenerationRequest,
    ) -> Result<Vec<GeneratedStrategyPlan>, PlanningError> {
        let system = prompts::strategy_system_prompt(request.k_candidates);
        let user = prompts::strategy_user_prompt(&request);

        let completion = self
            .client
            .complete(CompletionRequest {
                system_prompt: system,
                user_prompt: user,
                temperature: self.config.llm_temperature,
                max_tokens: self.config.llm_max_tokens,
                model: Some(self.config.llm_model.clone()),
            })
            .await?;

        parse_strategies(&completion.content)
    }

    async fn revise(
        &self,
        candidate: &GeneratedStrategyPlan,
        diagnostics: &CriticReport,
        request: &StrategyGenerationRequest,
    ) -> Result<Vec<GeneratedStrategyPlan>, PlanningError> {
        let system = prompts::strategy_revision_prompt();
        let candidate_json = serde_json::to_string_pretty(candidate).map_err(|e| {
            PlanningError::GenerationFailed(format!("failed to serialize candidate: {}", e))
        })?;
        let user = format!(
            "Original strategy:\n{}\n\nCritic diagnostics:\n- Total energy: {:.3}\n- Policy→Strategy energy: {:.3}\n- Strategy→Memory energy: {:.3}\n- Memory→Event energy: {:.3}\n- Novelty z-score: {:.2}\n- Mismatch layer: {:?}\n\nOriginal goal: {}\nGoal bucket: {}",
            candidate_json,
            diagnostics.total_energy,
            diagnostics.policy_strategy_energy,
            diagnostics.strategy_memory_energy,
            diagnostics.memory_event_energy,
            diagnostics.novelty_z,
            diagnostics.mismatch_layer,
            request.goal_description,
            request.goal_bucket_id,
        );

        let completion = self
            .client
            .complete(CompletionRequest {
                system_prompt: system,
                user_prompt: user,
                temperature: self.config.llm_temperature,
                max_tokens: self.config.llm_max_tokens,
                model: Some(self.config.llm_model.clone()),
            })
            .await?;

        parse_strategies(&completion.content)
    }
}

/// LLM-backed action generator.
pub struct LlmActionGenerator {
    client: Arc<dyn PlanningLlmClient>,
    config: PlanningConfig,
}

impl LlmActionGenerator {
    pub fn new(client: Arc<dyn PlanningLlmClient>, config: PlanningConfig) -> Self {
        Self { client, config }
    }
}

#[async_trait::async_trait]
impl ActionGenerator for LlmActionGenerator {
    async fn generate(
        &self,
        request: ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError> {
        let system = prompts::action_system_prompt(request.n_candidates);
        let user = prompts::action_user_prompt(&request);

        let completion = self
            .client
            .complete(CompletionRequest {
                system_prompt: system,
                user_prompt: user,
                temperature: self.config.llm_temperature,
                max_tokens: self.config.llm_max_tokens,
                model: Some(self.config.llm_model.clone()),
            })
            .await?;

        parse_actions(&completion.content)
    }

    async fn repair(
        &self,
        current_step: &GeneratedStep,
        error_report: &PredictionErrorReport,
        _request: &ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError> {
        let system = prompts::action_repair_prompt();
        let step_json = serde_json::to_string_pretty(current_step).map_err(|e| {
            PlanningError::GenerationFailed(format!("failed to serialize current step: {}", e))
        })?;
        let user = format!(
            "Current step:\n{}\n\nPrediction error report:\n- Event energy: {:.3}\n- Memory energy: {:.3}\n- Strategy energy: {:.3}\n- Total z-score: {:.2}\n- Mismatch layer: {:?}",
            step_json,
            error_report.event_energy,
            error_report.memory_energy,
            error_report.strategy_energy,
            error_report.total_z,
            error_report.mismatch_layer,
        );

        let completion = self
            .client
            .complete(CompletionRequest {
                system_prompt: system,
                user_prompt: user,
                temperature: self.config.llm_temperature,
                max_tokens: self.config.llm_max_tokens,
                model: Some(self.config.llm_model.clone()),
            })
            .await?;

        parse_actions(&completion.content)
    }
}

// ────────────────────────────────────────────────────────────────
// JSON parsing helpers
// ────────────────────────────────────────────────────────────────

/// Parse LLM response content into strategy plans.
fn parse_strategies(content: &str) -> Result<Vec<GeneratedStrategyPlan>, PlanningError> {
    let trimmed = strip_markdown_fences(content);
    serde_json::from_str::<Vec<GeneratedStrategyPlan>>(trimmed).map_err(|e| {
        PlanningError::GenerationFailed(format!("failed to parse strategy JSON: {}", e))
    })
}

/// Parse LLM response content into action plans.
fn parse_actions(content: &str) -> Result<Vec<GeneratedActionPlan>, PlanningError> {
    let trimmed = strip_markdown_fences(content);
    serde_json::from_str::<Vec<GeneratedActionPlan>>(trimmed)
        .map_err(|e| PlanningError::GenerationFailed(format!("failed to parse action JSON: {}", e)))
}

/// Strip markdown code fences if present (e.g., ```json ... ```).
fn strip_markdown_fences(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with("```") {
        // Find end of first line (skip ```json or ```)
        let after_first_fence = s.find('\n').map(|i| i + 1).unwrap_or(0);
        let inner = &s[after_first_fence..];
        // Strip trailing ```
        if let Some(end) = inner.rfind("```") {
            return inner[..end].trim();
        }
        return inner.trim();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::MockPlanningLlmClient;
    use crate::types::{EventContext, MemoryContext, StrategyContext};

    fn make_valid_strategy_json(k: usize) -> String {
        let mut strategies = Vec::new();
        for i in 0..k {
            strategies.push(serde_json::json!({
                "goal_bucket_id": 42,
                "goal_description": "Deploy the service",
                "steps": [{
                    "step_number": 1,
                    "step_kind": "Action",
                    "action_type": format!("deploy_v{}", i),
                    "parameters": {},
                    "description": null,
                    "precondition": null,
                    "success_criteria": null,
                    "failure_criteria": null,
                    "skip_if": null,
                    "max_retries": 1,
                    "timeout_ms": 30000,
                    "branches": [],
                    "recovery": null
                }],
                "preconditions": [],
                "stop_conditions": [],
                "fallback_steps": [],
                "risk_flags": [],
                "assumptions": ["System is ready"],
                "confidence": 0.8,
                "rationale": format!("Strategy variant {}", i)
            }));
        }
        serde_json::to_string(&strategies).unwrap()
    }

    fn make_valid_action_json(n: usize) -> String {
        let mut actions = Vec::new();
        for i in 0..n {
            actions.push(serde_json::json!({
                "action_type": format!("action_v{}", i),
                "parameters": {},
                "preconditions": [],
                "expected_event": {
                    "event_type": "Action",
                    "expected_outcome": "success",
                    "expected_significance": 0.7
                },
                "success_criteria": null,
                "failure_criteria": null,
                "timeout_ms": 10000,
                "max_retries": 1,
                "fallback_action": null,
                "risk_flags": [],
                "confidence": 0.85
            }));
        }
        serde_json::to_string(&actions).unwrap()
    }

    fn make_strategy_request() -> StrategyGenerationRequest {
        StrategyGenerationRequest {
            goal_description: "Deploy the service".to_string(),
            goal_bucket_id: 42,
            context_fingerprint: 12345,
            relevant_memories: vec![MemoryContext {
                summary: "Previous success".to_string(),
                tier: "Episodic".to_string(),
                strength: 0.8,
                takeaway: None,
                causal_note: None,
            }],
            similar_strategies: vec![StrategyContext {
                name: "old".to_string(),
                summary: "old strategy".to_string(),
                quality_score: 0.7,
                when_to_use: "always".to_string(),
            }],
            recent_events: vec![EventContext {
                event_type: "Action".to_string(),
                action_name: Some("build".to_string()),
                outcome: Some("success".to_string()),
                timestamp: 1000,
            }],
            transition_hints: vec![],
            constraints: vec![],
            k_candidates: 3,
        }
    }

    #[tokio::test]
    async fn test_llm_strategy_generator_valid_json() {
        let json = make_valid_strategy_json(3);
        let client = Arc::new(MockPlanningLlmClient::new(json));
        let gen = LlmStrategyGenerator::new(client, PlanningConfig::default());

        let result = gen.generate(make_strategy_request()).await;
        assert!(result.is_ok());
        let strategies = result.unwrap();
        assert_eq!(strategies.len(), 3);
        assert_eq!(strategies[0].goal_description, "Deploy the service");
    }

    #[tokio::test]
    async fn test_llm_strategy_generator_malformed_json() {
        let client = Arc::new(MockPlanningLlmClient::new("not valid json".to_string()));
        let gen = LlmStrategyGenerator::new(client, PlanningConfig::default());

        let result = gen.generate(make_strategy_request()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("failed to parse strategy JSON"));
    }

    #[tokio::test]
    async fn test_llm_strategy_generator_client_fails() {
        let client = Arc::new(MockPlanningLlmClient::failing());
        let gen = LlmStrategyGenerator::new(client, PlanningConfig::default());

        let result = gen.generate(make_strategy_request()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mock client"));
    }

    #[tokio::test]
    async fn test_llm_action_generator_valid_json() {
        let json = make_valid_action_json(2);
        let client = Arc::new(MockPlanningLlmClient::new(json));
        let gen = LlmActionGenerator::new(client, PlanningConfig::default());

        let strategy = GeneratedStrategyPlan {
            goal_bucket_id: 42,
            goal_description: "test".to_string(),
            steps: vec![],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.8,
            rationale: None,
        };
        let request = ActionGenerationRequest {
            strategy,
            current_step_index: 0,
            recent_events: vec![],
            context_fingerprint: 12345,
            n_candidates: 2,
        };

        let result = gen.generate(request).await;
        assert!(result.is_ok());
        let actions = result.unwrap();
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].action_type, "action_v0");
    }

    #[tokio::test]
    async fn test_llm_action_generator_client_fails() {
        let client = Arc::new(MockPlanningLlmClient::failing());
        let gen = LlmActionGenerator::new(client, PlanningConfig::default());

        let strategy = GeneratedStrategyPlan {
            goal_bucket_id: 42,
            goal_description: "test".to_string(),
            steps: vec![],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.8,
            rationale: None,
        };
        let request = ActionGenerationRequest {
            strategy,
            current_step_index: 0,
            recent_events: vec![],
            context_fingerprint: 12345,
            n_candidates: 2,
        };

        let result = gen.generate(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_llm_strategy_revision() {
        let json = make_valid_strategy_json(1);
        let client = Arc::new(MockPlanningLlmClient::new(json));
        let gen = LlmStrategyGenerator::new(client, PlanningConfig::default());

        let original = GeneratedStrategyPlan {
            goal_bucket_id: 42,
            goal_description: "Deploy the service".to_string(),
            steps: vec![],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.5,
            rationale: None,
        };
        let diagnostics = CriticReport {
            total_energy: 2.0,
            policy_strategy_energy: 0.8,
            strategy_memory_energy: 0.6,
            memory_event_energy: 0.6,
            novelty_z: 1.5,
            is_novel: false,
            mismatch_layer: agent_db_world_model::MismatchLayer::Strategy,
            confidence: 0.8,
            support_count: 50,
        };

        let result = gen
            .revise(&original, &diagnostics, &make_strategy_request())
            .await;
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_llm_action_repair() {
        let json = make_valid_action_json(1);
        let client = Arc::new(MockPlanningLlmClient::new(json));
        let gen = LlmActionGenerator::new(client, PlanningConfig::default());

        let step = GeneratedStep {
            step_number: 1,
            step_kind: crate::types::StepKind::Action,
            action_type: "deploy".to_string(),
            parameters: serde_json::json!({}),
            description: None,
            precondition: None,
            success_criteria: None,
            failure_criteria: None,
            skip_if: None,
            max_retries: 0,
            timeout_ms: None,
            branches: vec![],
            recovery: None,
        };
        let error = PredictionErrorReport {
            event_energy: 1.0,
            memory_energy: 0.5,
            strategy_energy: 0.3,
            event_z: 2.5,
            memory_z: 0.5,
            strategy_z: 0.5,
            total_z: 2.5,
            mismatch_layer: agent_db_world_model::MismatchLayer::Event,
        };
        let request = ActionGenerationRequest {
            strategy: GeneratedStrategyPlan {
                goal_bucket_id: 42,
                goal_description: "test".to_string(),
                steps: vec![],
                preconditions: vec![],
                stop_conditions: vec![],
                fallback_steps: vec![],
                risk_flags: vec![],
                assumptions: vec![],
                confidence: 0.8,
                rationale: None,
            },
            current_step_index: 0,
            recent_events: vec![],
            context_fingerprint: 12345,
            n_candidates: 1,
        };

        let result = gen.repair(&step, &error, &request).await;
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn test_strip_markdown_fences() {
        assert_eq!(strip_markdown_fences("hello"), "hello");
        assert_eq!(strip_markdown_fences("```json\n[1,2]\n```"), "[1,2]");
        assert_eq!(strip_markdown_fences("```\n[1]\n```"), "[1]");
        assert_eq!(strip_markdown_fences("  [1,2]  "), "[1,2]");
    }
}
