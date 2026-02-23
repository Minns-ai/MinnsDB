//! Prompt templates for LLM-based strategy and action generation.
//!
//! Each function produces a prompt string suitable for passing to a
//! `PlanningLlmClient`. The prompts embed JSON schemas so the LLM
//! produces structured output that can be parsed directly.

use crate::types::{ActionGenerationRequest, StrategyGenerationRequest};

/// System prompt for strategy generation.
///
/// Instructs the LLM to produce `k` strategy candidates as a JSON array
/// conforming to the `GeneratedStrategyPlan` schema.
pub fn strategy_system_prompt(k: usize) -> String {
    format!(
        r#"You are a strategy planning engine. Generate exactly {k} candidate strategies as a JSON array.

Each strategy must conform to this schema:
{{
  "goal_bucket_id": <u64>,
  "goal_description": "<string>",
  "steps": [
    {{
      "step_number": <u32>,
      "step_kind": "Action" | "Observation" | "Decision" | "Validation" | "Recovery",
      "action_type": "<identifier>",
      "parameters": {{}},
      "description": "<optional string>",
      "precondition": null | {{ "description": "<string>", "check_type": "<string>", "parameters": {{}} }},
      "success_criteria": null | {{ "description": "<string>", "check_type": "<string>", "parameters": {{}} }},
      "failure_criteria": null | {{ "description": "<string>", "check_type": "<string>", "parameters": {{}} }},
      "skip_if": null,
      "max_retries": <u32>,
      "timeout_ms": null | <u64>,
      "branches": [],
      "recovery": null | "<string>"
    }}
  ],
  "preconditions": [],
  "stop_conditions": [],
  "fallback_steps": [],
  "risk_flags": [{{ "description": "<string>", "severity": "Low" | "Medium" | "High" | "Critical", "mitigation": null | "<string>" }}],
  "assumptions": ["<string>"],
  "confidence": <f32 0.0-1.0>,
  "rationale": "<string>"
}}

Return ONLY the JSON array. No markdown, no explanation."#
    )
}

/// User prompt for strategy generation.
///
/// Serializes the request context (goal, memories, strategies, events, constraints)
/// so the LLM can use it for generation.
pub fn strategy_user_prompt(request: &StrategyGenerationRequest) -> String {
    let memories_json = serde_json::to_string_pretty(&request.relevant_memories)
        .unwrap_or_else(|_| "[]".to_string());
    let strategies_json = serde_json::to_string_pretty(&request.similar_strategies)
        .unwrap_or_else(|_| "[]".to_string());
    let events_json =
        serde_json::to_string_pretty(&request.recent_events).unwrap_or_else(|_| "[]".to_string());
    let transitions_section = if request.transition_hints.is_empty() {
        String::new()
    } else {
        let json = serde_json::to_string_pretty(&request.transition_hints)
            .unwrap_or_else(|_| "[]".to_string());
        format!("\nEmpirical Transition Success Rates:\n{}\n", json)
    };
    let constraints = if request.constraints.is_empty() {
        "None".to_string()
    } else {
        format!("- {}", request.constraints.join("\n- "))
    };

    format!(
        r#"Generate {k} strategy candidates for this goal.

Goal: {goal}
Goal Bucket ID: {bucket_id}
Context Fingerprint: {ctx_fp}

Relevant Memories:
{memories}

Similar Past Strategies:
{strategies}

Recent Events:
{events}
{transitions}Constraints:
{constraints}"#,
        k = request.k_candidates,
        goal = request.goal_description,
        bucket_id = request.goal_bucket_id,
        ctx_fp = request.context_fingerprint,
        memories = memories_json,
        strategies = strategies_json,
        events = events_json,
        transitions = transitions_section,
        constraints = constraints,
    )
}

/// System prompt for strategy revision after critic diagnostics.
pub fn strategy_revision_prompt() -> String {
    r#"You are a strategy revision engine. You will receive:
1. An original strategy plan
2. Critic diagnostics (energy scores and mismatch analysis)

Revise the strategy to address the diagnostics. Return a JSON array containing
one revised strategy that:
- Addresses the identified mismatches
- Maintains or improves confidence
- Keeps the same goal_bucket_id and goal_description

Return ONLY the JSON array. No markdown, no explanation."#
        .to_string()
}

/// System prompt for action generation.
///
/// Instructs the LLM to produce `n` candidate actions as a JSON array
/// conforming to the `GeneratedActionPlan` schema.
pub fn action_system_prompt(n: usize) -> String {
    format!(
        r#"You are an action planning engine. Generate exactly {n} candidate actions as a JSON array.

Each action must conform to this schema:
{{
  "action_type": "<identifier>",
  "parameters": {{}},
  "preconditions": [],
  "expected_event": {{
    "event_type": "<string>",
    "expected_outcome": "success" | "failure" | "partial",
    "expected_significance": <f32 0.0-1.0>
  }},
  "success_criteria": null | {{ "description": "<string>", "check_type": "<string>", "parameters": {{}} }},
  "failure_criteria": null | {{ "description": "<string>", "check_type": "<string>", "parameters": {{}} }},
  "timeout_ms": null | <u64>,
  "max_retries": <u32>,
  "fallback_action": null | "<string>",
  "risk_flags": [],
  "confidence": <f32 0.0-1.0>
}}

Return ONLY the JSON array. No markdown, no explanation."#
    )
}

/// User prompt for action generation.
pub fn action_user_prompt(request: &ActionGenerationRequest) -> String {
    let strategy_json =
        serde_json::to_string_pretty(&request.strategy).unwrap_or_else(|_| "{}".to_string());
    let events_json =
        serde_json::to_string_pretty(&request.recent_events).unwrap_or_else(|_| "[]".to_string());

    format!(
        r#"Generate {n} action candidates for step {step} of this strategy.

Strategy:
{strategy}

Current Step Index: {step}
Context Fingerprint: {ctx_fp}

Recent Events:
{events}"#,
        n = request.n_candidates,
        step = request.current_step_index,
        strategy = strategy_json,
        ctx_fp = request.context_fingerprint,
        events = events_json,
    )
}

/// System prompt for action repair after prediction error.
pub fn action_repair_prompt() -> String {
    r#"You are an action repair engine. You will receive:
1. The current step that triggered a prediction error
2. A prediction error report with energy scores and mismatch analysis

Generate a repaired action that addresses the prediction error.
Return a JSON array containing one repaired action plan.

Return ONLY the JSON array. No markdown, no explanation."#
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EventContext, MemoryContext, StrategyContext};

    fn make_strategy_request() -> StrategyGenerationRequest {
        StrategyGenerationRequest {
            goal_description: "Deploy the service".to_string(),
            goal_bucket_id: 42,
            context_fingerprint: 12345,
            relevant_memories: vec![MemoryContext {
                summary: "For Deploy the service: What worked: Built image and ran health check. Success rate: 2/3 episodes.".to_string(),
                tier: "Semantic".to_string(),
                strength: 0.9,
                takeaway: Some("Effective approach for Deploy the service: Build and test in staging first.".to_string()),
                causal_note: Some("Succeeded because: Health check passed before traffic shift.".to_string()),
            }],
            similar_strategies: vec![StrategyContext {
                name: "blue-green-deploy".to_string(),
                summary: "Blue-green deployment".to_string(),
                quality_score: 0.85,
                when_to_use: "Production deployments".to_string(),
            }],
            recent_events: vec![EventContext {
                event_type: "Action".to_string(),
                action_name: Some("build".to_string()),
                outcome: Some("success".to_string()),
                timestamp: 1000,
            }],
            transition_hints: vec![],
            constraints: vec!["Must complete within 5 minutes".to_string()],
            k_candidates: 3,
        }
    }

    #[test]
    fn test_strategy_system_prompt_contains_k() {
        let prompt = strategy_system_prompt(5);
        assert!(prompt.contains("exactly 5"));
        assert!(prompt.contains("goal_bucket_id"));
        assert!(prompt.contains("JSON array"));
    }

    #[test]
    fn test_strategy_user_prompt_contains_context() {
        let request = make_strategy_request();
        let prompt = strategy_user_prompt(&request);
        assert!(prompt.contains("Deploy the service"));
        assert!(prompt.contains("42"));
        assert!(prompt.contains("What worked: Built image"));
        assert!(prompt.contains("Effective approach for Deploy the service"));
        assert!(prompt.contains("blue-green-deploy"));
        assert!(prompt.contains("5 minutes"));
    }

    #[test]
    fn test_action_system_prompt_contains_n() {
        let prompt = action_system_prompt(3);
        assert!(prompt.contains("exactly 3"));
        assert!(prompt.contains("action_type"));
    }

    #[test]
    fn test_strategy_revision_prompt() {
        let prompt = strategy_revision_prompt();
        assert!(prompt.contains("revision"));
        assert!(prompt.contains("diagnostics"));
    }

    #[test]
    fn test_action_repair_prompt() {
        let prompt = action_repair_prompt();
        assert!(prompt.contains("repair"));
        assert!(prompt.contains("prediction error"));
    }

    #[test]
    fn test_strategy_user_prompt_empty_constraints() {
        let mut request = make_strategy_request();
        request.constraints = vec![];
        let prompt = strategy_user_prompt(&request);
        assert!(prompt.contains("Constraints:\nNone"));
        assert!(
            !prompt.contains("- None"),
            "Should not produce '- None' for empty constraints"
        );
    }

    #[test]
    fn test_strategy_user_prompt_with_constraints() {
        let mut request = make_strategy_request();
        request.constraints = vec!["fast".to_string(), "safe".to_string()];
        let prompt = strategy_user_prompt(&request);
        assert!(prompt.contains("- fast\n- safe"));
    }
}
