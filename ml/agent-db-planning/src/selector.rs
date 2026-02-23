//! Candidate selection logic using novelty z-score and confidence.
//!
//! The selector decides what to do with scored strategy candidates:
//! - **Accept**: z < accept_z (familiar, compatible)
//! - **Revise**: accept_z ≤ z < revise_z (somewhat novel, try to improve)
//! - **Experimental**: revise_z ≤ z < reject_z (novel but not impossible)
//! - **Reject**: z ≥ reject_z (too incompatible)
//!
//! A confidence gate ensures the model doesn't block viable strategies when
//! it hasn't seen enough data.

use agent_db_world_model::CriticReport;

use crate::types::{
    GeneratedStrategyPlan, PlanningConfig, ScoredCandidate, SelectionDecision,
    SelectionDecisionKind,
};

/// Select the best candidate from a set of scored candidates.
///
/// Returns a [`SelectionDecision`] indicating what to do with the best candidate.
pub fn select_best(
    candidates: &[ScoredCandidate],
    config: &PlanningConfig,
) -> Option<SelectionDecision> {
    if candidates.is_empty() {
        return None;
    }

    // Sort by total_energy ascending (lowest = most compatible)
    let best = candidates
        .iter()
        .min_by(|a, b| {
            a.report
                .total_energy
                .partial_cmp(&b.report.total_energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    Some(make_decision(best.plan.clone(), &best.report, config))
}

/// Make a selection decision for a single scored candidate.
pub fn make_decision(
    plan: GeneratedStrategyPlan,
    report: &CriticReport,
    config: &PlanningConfig,
) -> SelectionDecision {
    // Confidence gate: if confidence is too low, accept (model can't reliably reject)
    if report.confidence < config.min_confidence {
        return SelectionDecision::Accept(plan);
    }

    // Decision based on novelty z-score
    if report.novelty_z < config.accept_z {
        SelectionDecision::Accept(plan)
    } else if report.novelty_z < config.revise_z {
        SelectionDecision::Revise {
            candidate: plan,
            diagnostics: report.clone(),
        }
    } else if report.novelty_z < config.reject_z {
        SelectionDecision::Experimental(plan)
    } else {
        SelectionDecision::Reject {
            reason: format!(
                "z={:.1}, exceeds reject threshold {:.1}",
                report.novelty_z, config.reject_z
            ),
            diagnostics: report.clone(),
        }
    }
}

/// Classify a decision into its kind (for embedding in `ScoredCandidate`).
pub fn classify_decision(report: &CriticReport, config: &PlanningConfig) -> SelectionDecisionKind {
    if report.confidence < config.min_confidence {
        return SelectionDecisionKind::Accept;
    }

    if report.novelty_z < config.accept_z {
        SelectionDecisionKind::Accept
    } else if report.novelty_z < config.revise_z {
        SelectionDecisionKind::Revise
    } else if report.novelty_z < config.reject_z {
        SelectionDecisionKind::Experimental
    } else {
        SelectionDecisionKind::Reject
    }
}

/// Check if a decision represents acceptance (Accept or Experimental).
pub fn is_accepted(decision: &SelectionDecision) -> bool {
    matches!(
        decision,
        SelectionDecision::Accept(_) | SelectionDecision::Experimental(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GeneratedStep, StepKind};
    use agent_db_world_model::MismatchLayer;

    fn make_plan() -> GeneratedStrategyPlan {
        GeneratedStrategyPlan {
            goal_bucket_id: 100,
            goal_description: "test".to_string(),
            steps: vec![GeneratedStep {
                step_number: 1,
                step_kind: StepKind::Action,
                action_type: "test".to_string(),
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

    fn make_report(novelty_z: f32, confidence: f32) -> CriticReport {
        CriticReport {
            total_energy: -1.0,
            policy_strategy_energy: -0.5,
            strategy_memory_energy: -0.3,
            memory_event_energy: -0.2,
            novelty_z,
            is_novel: novelty_z > 2.0,
            mismatch_layer: MismatchLayer::None,
            confidence,
            support_count: 50,
        }
    }

    #[test]
    fn test_accept_low_z() {
        let config = PlanningConfig::default();
        let decision = make_decision(make_plan(), &make_report(0.5, 0.8), &config);
        assert!(matches!(decision, SelectionDecision::Accept(_)));
    }

    #[test]
    fn test_revise_medium_z() {
        let config = PlanningConfig::default();
        let decision = make_decision(make_plan(), &make_report(1.5, 0.8), &config);
        assert!(matches!(decision, SelectionDecision::Revise { .. }));
    }

    #[test]
    fn test_experimental_high_z() {
        let config = PlanningConfig::default();
        let decision = make_decision(make_plan(), &make_report(2.5, 0.8), &config);
        assert!(matches!(decision, SelectionDecision::Experimental(_)));
    }

    #[test]
    fn test_reject_very_high_z() {
        let config = PlanningConfig::default();
        let decision = make_decision(make_plan(), &make_report(3.5, 0.8), &config);
        assert!(matches!(decision, SelectionDecision::Reject { .. }));
    }

    #[test]
    fn test_confidence_gate_accepts_when_low() {
        let config = PlanningConfig::default();
        // Even with very high z, low confidence → accept
        let decision = make_decision(make_plan(), &make_report(5.0, 0.1), &config);
        assert!(matches!(decision, SelectionDecision::Accept(_)));
    }

    #[test]
    fn test_select_best_picks_lowest_energy() {
        let config = PlanningConfig::default();
        let candidates = vec![
            ScoredCandidate {
                plan: make_plan(),
                report: make_report(0.5, 0.8),
                decision: SelectionDecisionKind::Accept,
            },
            ScoredCandidate {
                plan: {
                    let mut p = make_plan();
                    p.goal_description = "better plan".to_string();
                    p
                },
                report: CriticReport {
                    total_energy: -2.0, // lower = better
                    ..make_report(0.3, 0.8)
                },
                decision: SelectionDecisionKind::Accept,
            },
        ];

        let decision = select_best(&candidates, &config).unwrap();
        match decision {
            SelectionDecision::Accept(plan) => {
                assert_eq!(plan.goal_description, "better plan");
            },
            _ => panic!("expected Accept"),
        }
    }

    #[test]
    fn test_select_best_empty_returns_none() {
        let config = PlanningConfig::default();
        assert!(select_best(&[], &config).is_none());
    }

    #[test]
    fn test_classify_decision() {
        let config = PlanningConfig::default();
        assert_eq!(
            classify_decision(&make_report(0.5, 0.8), &config),
            SelectionDecisionKind::Accept
        );
        assert_eq!(
            classify_decision(&make_report(1.5, 0.8), &config),
            SelectionDecisionKind::Revise
        );
        assert_eq!(
            classify_decision(&make_report(2.5, 0.8), &config),
            SelectionDecisionKind::Experimental
        );
        assert_eq!(
            classify_decision(&make_report(3.5, 0.8), &config),
            SelectionDecisionKind::Reject
        );
    }

    #[test]
    fn test_is_accepted() {
        assert!(is_accepted(&SelectionDecision::Accept(make_plan())));
        assert!(is_accepted(&SelectionDecision::Experimental(make_plan())));
        assert!(!is_accepted(&SelectionDecision::Reject {
            reason: "test".to_string(),
            diagnostics: make_report(3.5, 0.8),
        }));
    }
}
