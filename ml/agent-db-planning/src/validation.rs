//! Plan validation — structural checks applied before scoring.
//!
//! Catches malformed plans early (empty steps, missing action types,
//! invalid step numbers, etc.) before they waste critic compute.

use crate::types::{
    GeneratedActionPlan, GeneratedStrategyPlan, ValidationError, ValidationSeverity,
};
use crate::PlanValidator;

/// Default plan validator implementing structural checks.
pub struct DefaultPlanValidator;

impl DefaultPlanValidator {
    pub fn new() -> Self {
        Self
    }
}

impl PlanValidator for DefaultPlanValidator {
    fn validate_strategy(&self, plan: &GeneratedStrategyPlan) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Must have at least one step
        if plan.steps.is_empty() {
            errors.push(ValidationError {
                field: "steps".to_string(),
                message: "strategy must have at least one step".to_string(),
                severity: ValidationSeverity::Error,
            });
        }

        // Goal description must not be empty
        if plan.goal_description.trim().is_empty() {
            errors.push(ValidationError {
                field: "goal_description".to_string(),
                message: "goal description must not be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }

        // Confidence must be in [0, 1]
        if plan.confidence < 0.0 || plan.confidence > 1.0 {
            errors.push(ValidationError {
                field: "confidence".to_string(),
                message: format!("confidence must be in [0, 1], got {}", plan.confidence),
                severity: ValidationSeverity::Error,
            });
        }

        // Validate each step
        let mut seen_step_numbers = std::collections::HashSet::new();
        for (idx, step) in plan.steps.iter().enumerate() {
            // action_type must not be empty
            if step.action_type.trim().is_empty() {
                errors.push(ValidationError {
                    field: format!("steps[{}].action_type", idx),
                    message: "action_type must not be empty".to_string(),
                    severity: ValidationSeverity::Error,
                });
            }

            // step_number should be unique
            if !seen_step_numbers.insert(step.step_number) {
                errors.push(ValidationError {
                    field: format!("steps[{}].step_number", idx),
                    message: format!("duplicate step_number: {}", step.step_number),
                    severity: ValidationSeverity::Error,
                });
            }

            // Branch goto_step should reference valid step numbers
            for (branch_idx, branch) in step.branches.iter().enumerate() {
                if !plan.steps.iter().any(|s| s.step_number == branch.goto_step) {
                    errors.push(ValidationError {
                        field: format!("steps[{}].branches[{}].goto_step", idx, branch_idx),
                        message: format!(
                            "goto_step {} references non-existent step",
                            branch.goto_step
                        ),
                        severity: ValidationSeverity::Warning,
                    });
                }
            }
        }

        // Fallback steps should also have valid action_types
        for (idx, step) in plan.fallback_steps.iter().enumerate() {
            if step.action_type.trim().is_empty() {
                errors.push(ValidationError {
                    field: format!("fallback_steps[{}].action_type", idx),
                    message: "fallback action_type must not be empty".to_string(),
                    severity: ValidationSeverity::Warning,
                });
            }
        }

        errors
    }

    fn validate_action(&self, plan: &GeneratedActionPlan) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // action_type must not be empty
        if plan.action_type.trim().is_empty() {
            errors.push(ValidationError {
                field: "action_type".to_string(),
                message: "action_type must not be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }

        // Confidence must be in [0, 1]
        if plan.confidence < 0.0 || plan.confidence > 1.0 {
            errors.push(ValidationError {
                field: "confidence".to_string(),
                message: format!("confidence must be in [0, 1], got {}", plan.confidence),
                severity: ValidationSeverity::Error,
            });
        }

        // expected_event.event_type must not be empty
        if plan.expected_event.event_type.trim().is_empty() {
            errors.push(ValidationError {
                field: "expected_event.event_type".to_string(),
                message: "expected event type must not be empty".to_string(),
                severity: ValidationSeverity::Error,
            });
        }

        // expected_significance should be in [0, 1]
        if plan.expected_event.expected_significance < 0.0
            || plan.expected_event.expected_significance > 1.0
        {
            errors.push(ValidationError {
                field: "expected_event.expected_significance".to_string(),
                message: format!(
                    "expected significance must be in [0, 1], got {}",
                    plan.expected_event.expected_significance
                ),
                severity: ValidationSeverity::Warning,
            });
        }

        errors
    }
}

/// Check if any errors in the list have `Error` severity.
pub fn has_errors(errors: &[ValidationError]) -> bool {
    errors
        .iter()
        .any(|e| matches!(e.severity, ValidationSeverity::Error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Criteria, ExpectedEvent, GeneratedStep, StepBranch, StepKind};

    fn make_valid_strategy() -> GeneratedStrategyPlan {
        GeneratedStrategyPlan {
            goal_bucket_id: 100,
            goal_description: "Test goal".to_string(),
            steps: vec![GeneratedStep {
                step_number: 1,
                step_kind: StepKind::Action,
                action_type: "test_action".to_string(),
                parameters: serde_json::json!({}),
                description: Some("Do the thing".to_string()),
                precondition: None,
                success_criteria: None,
                failure_criteria: None,
                skip_if: None,
                max_retries: 0,
                timeout_ms: None,
                branches: vec![],
                recovery: None,
            }],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.8,
            rationale: None,
        }
    }

    fn make_valid_action() -> GeneratedActionPlan {
        GeneratedActionPlan {
            action_type: "test_action".to_string(),
            parameters: serde_json::json!({"key": "value"}),
            preconditions: vec![],
            expected_event: ExpectedEvent {
                event_type: "Action".to_string(),
                expected_outcome: "success".to_string(),
                expected_significance: 0.7,
            },
            success_criteria: None,
            failure_criteria: None,
            timeout_ms: Some(5000),
            max_retries: 1,
            fallback_action: None,
            risk_flags: vec![],
            confidence: 0.9,
        }
    }

    #[test]
    fn test_valid_strategy_passes() {
        let validator = DefaultPlanValidator::new();
        let errors = validator.validate_strategy(&make_valid_strategy());
        assert!(
            errors.is_empty(),
            "valid strategy should have no errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_empty_steps_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_strategy();
        plan.steps.clear();
        let errors = validator.validate_strategy(&plan);
        assert!(has_errors(&errors));
    }

    #[test]
    fn test_empty_goal_description_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_strategy();
        plan.goal_description = "  ".to_string();
        let errors = validator.validate_strategy(&plan);
        assert!(has_errors(&errors));
    }

    #[test]
    fn test_invalid_confidence_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_strategy();
        plan.confidence = 1.5;
        let errors = validator.validate_strategy(&plan);
        assert!(has_errors(&errors));
    }

    #[test]
    fn test_empty_action_type_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_strategy();
        plan.steps[0].action_type = "".to_string();
        let errors = validator.validate_strategy(&plan);
        assert!(has_errors(&errors));
    }

    #[test]
    fn test_duplicate_step_numbers_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_strategy();
        plan.steps.push(GeneratedStep {
            step_number: 1, // duplicate
            step_kind: StepKind::Action,
            action_type: "another_action".to_string(),
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
        });
        let errors = validator.validate_strategy(&plan);
        assert!(has_errors(&errors));
    }

    #[test]
    fn test_invalid_branch_goto_warns() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_strategy();
        plan.steps[0].branches.push(StepBranch {
            condition: Criteria {
                description: "test".to_string(),
                check_type: "event_type_match".to_string(),
                parameters: serde_json::json!({}),
            },
            goto_step: 999, // non-existent
        });
        let errors = validator.validate_strategy(&plan);
        assert!(!errors.is_empty());
        // Should be a warning, not an error
        assert!(errors
            .iter()
            .any(|e| matches!(e.severity, ValidationSeverity::Warning)));
    }

    #[test]
    fn test_valid_action_passes() {
        let validator = DefaultPlanValidator::new();
        let errors = validator.validate_action(&make_valid_action());
        assert!(errors.is_empty());
    }

    #[test]
    fn test_action_empty_type_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_action();
        plan.action_type = "".to_string();
        let errors = validator.validate_action(&plan);
        assert!(has_errors(&errors));
    }

    #[test]
    fn test_action_empty_event_type_rejected() {
        let validator = DefaultPlanValidator::new();
        let mut plan = make_valid_action();
        plan.expected_event.event_type = "".to_string();
        let errors = validator.validate_action(&plan);
        assert!(has_errors(&errors));
    }
}
