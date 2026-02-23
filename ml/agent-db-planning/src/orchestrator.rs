//! End-to-end pipeline orchestrator.
//!
//! Coordinates the full flow: generate → validate → score → select → (revise loop).
//! This is the main entry point for the planning engine.

use agent_db_world_model::{
    CriticReport, EventFeatures, MemoryFeatures, PolicyFeatures, StrategyFeatures, WorldModelCritic,
};

use crate::selector;
use crate::types::{
    ActionGenerationRequest, GeneratedActionPlan, GeneratedStrategyPlan, PlanningConfig,
    PlanningError, ScoredAction, ScoredCandidate, SelectionDecision, StrategyGenerationRequest,
};
use crate::validation::{self, DefaultPlanValidator};
use crate::{ActionGenerator, PlanValidator, StrategyGenerator};

/// The planning orchestrator coordinates generation, validation, scoring, and selection.
pub struct PlanningOrchestrator {
    config: PlanningConfig,
    validator: DefaultPlanValidator,
}

impl PlanningOrchestrator {
    pub fn new(config: PlanningConfig) -> Self {
        Self {
            config,
            validator: DefaultPlanValidator::new(),
        }
    }

    /// Generate and score strategy candidates for a goal.
    ///
    /// Pipeline: generate → validate → score → select → (optional revision loop).
    ///
    /// Returns scored candidates sorted by energy (best first), plus the
    /// overall selection decision.
    pub async fn generate_strategies(
        &self,
        request: StrategyGenerationRequest,
        strategy_generator: &dyn StrategyGenerator,
        critic: Option<&dyn WorldModelCritic>,
        policy_features: &PolicyFeatures,
    ) -> Result<(Vec<ScoredCandidate>, SelectionDecision), PlanningError> {
        // Step 1: Generate candidates
        let candidates = strategy_generator.generate(request.clone()).await?;
        if candidates.is_empty() {
            return Err(PlanningError::AllCandidatesRejected(
                "generator produced no candidates".to_string(),
            ));
        }

        // Step 2: Validate and score
        let scored = self.validate_and_score_strategies(candidates, critic, policy_features);
        if scored.is_empty() {
            return Err(PlanningError::AllCandidatesRejected(
                "all candidates failed validation".to_string(),
            ));
        }

        // Step 3: Select best
        let decision = selector::select_best(&scored, &self.config).ok_or_else(|| {
            PlanningError::AllCandidatesRejected("no valid candidates after scoring".to_string())
        })?;

        // Step 4: Revision loop (if needed)
        match &decision {
            SelectionDecision::Revise {
                candidate,
                diagnostics,
            } => {
                let revised = self
                    .revision_loop(
                        candidate.clone(),
                        diagnostics.clone(),
                        &request,
                        strategy_generator,
                        critic,
                        policy_features,
                    )
                    .await?;
                Ok(revised)
            },
            _ => Ok((scored, decision)),
        }
    }

    /// Revision loop: revise → re-validate → re-score → re-select.
    /// Limited to `max_revision_rounds` iterations.
    async fn revision_loop(
        &self,
        candidate: GeneratedStrategyPlan,
        diagnostics: CriticReport,
        request: &StrategyGenerationRequest,
        strategy_generator: &dyn StrategyGenerator,
        critic: Option<&dyn WorldModelCritic>,
        policy_features: &PolicyFeatures,
    ) -> Result<(Vec<ScoredCandidate>, SelectionDecision), PlanningError> {
        let mut current_candidate = candidate;
        let mut current_diagnostics = diagnostics;

        for round in 0..self.config.max_revision_rounds {
            tracing::debug!(round, "revision loop iteration");

            // Generate revised candidates
            let revised = strategy_generator
                .revise(&current_candidate, &current_diagnostics, request)
                .await?;

            if revised.is_empty() {
                tracing::warn!("revision produced no candidates, accepting original");
                return Ok((vec![], SelectionDecision::Experimental(current_candidate)));
            }

            // Validate and score revised candidates
            let scored = self.validate_and_score_strategies(revised, critic, policy_features);
            if scored.is_empty() {
                continue; // try again
            }

            let decision = selector::select_best(&scored, &self.config)
                .unwrap_or_else(|| SelectionDecision::Experimental(current_candidate.clone()));

            match &decision {
                SelectionDecision::Accept(_) | SelectionDecision::Experimental(_) => {
                    return Ok((scored, decision));
                },
                SelectionDecision::Revise {
                    candidate: new_candidate,
                    diagnostics: new_diagnostics,
                } => {
                    current_candidate = new_candidate.clone();
                    current_diagnostics = new_diagnostics.clone();
                    // Continue loop
                },
                SelectionDecision::Reject { reason, .. } => {
                    tracing::warn!(round, reason = %reason, "revision rejected, trying again");
                    // Continue loop with original
                },
            }
        }

        // Exhausted revision rounds → accept as experimental
        tracing::warn!(
            rounds = self.config.max_revision_rounds,
            "exhausted revision rounds, accepting as experimental"
        );
        Ok((vec![], SelectionDecision::Experimental(current_candidate)))
    }

    /// Validate and score a list of strategy candidates.
    fn validate_and_score_strategies(
        &self,
        candidates: Vec<GeneratedStrategyPlan>,
        critic: Option<&dyn WorldModelCritic>,
        policy_features: &PolicyFeatures,
    ) -> Vec<ScoredCandidate> {
        let mut scored = Vec::new();

        for candidate in candidates {
            // Validate
            let errors = self.validator.validate_strategy(&candidate);
            if validation::has_errors(&errors) {
                tracing::debug!(
                    errors = errors.len(),
                    goal = %candidate.goal_description,
                    "candidate failed validation, discarding"
                );
                continue;
            }

            // Score
            let report = if let Some(critic) = critic {
                let strategy_features = extract_strategy_features(&candidate);
                critic.score_strategy(policy_features, &strategy_features)
            } else {
                // No critic — return a neutral report
                default_report()
            };

            let decision_kind = selector::classify_decision(&report, &self.config);

            scored.push(ScoredCandidate {
                plan: candidate,
                report,
                decision: decision_kind,
            });
        }

        // Sort by total_energy ascending (best first)
        scored.sort_by(|a, b| {
            a.report
                .total_energy
                .partial_cmp(&b.report.total_energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored
    }

    /// Generate and score action candidates for a strategy step.
    ///
    /// The `feasibility_fn` parameter computes feasibility (0.0–1.0) for an action.
    /// Pass `None` to skip feasibility scoring (all actions get feasibility 1.0).
    /// In production, GraphEngine provides this via `TransitionModel::posterior_success()`.
    pub async fn generate_actions(
        &self,
        request: ActionGenerationRequest,
        action_generator: &dyn ActionGenerator,
        critic: Option<&dyn WorldModelCritic>,
        policy_features: &PolicyFeatures,
        memory_features: &MemoryFeatures,
        feasibility_fn: Option<&(dyn Fn(&GeneratedActionPlan) -> f32 + Send + Sync)>,
    ) -> Result<Vec<ScoredAction>, PlanningError> {
        let actions = action_generator.generate(request.clone()).await?;
        if actions.is_empty() {
            return Err(PlanningError::AllCandidatesRejected(
                "action generator produced no candidates".to_string(),
            ));
        }

        let strategy_features = extract_strategy_features(&request.strategy);

        let mut scored: Vec<ScoredAction> = actions
            .into_iter()
            .filter_map(|action| {
                // Validate
                let errors = self.validator.validate_action(&action);
                if validation::has_errors(&errors) {
                    return None;
                }

                // Score via critic
                let energy = if let Some(critic) = critic {
                    let event_features = extract_action_event_features(&action);
                    let report = critic.score(
                        policy_features,
                        &strategy_features,
                        memory_features,
                        &event_features,
                    );
                    report.total_energy
                } else {
                    0.0
                };

                // Score via feasibility function (TransitionModel integration point)
                let feasibility = feasibility_fn.map(|f| f(&action)).unwrap_or(1.0);

                Some(ScoredAction {
                    plan: action,
                    energy,
                    feasibility,
                })
            })
            .collect();

        // Sort by combined score (energy weighted 0.6 + infeasibility 0.4)
        scored.sort_by(|a, b| {
            let score_a = a.energy * 0.6 + (1.0 - a.feasibility) * 0.4;
            let score_b = b.energy * 0.6 + (1.0 - b.feasibility) * 0.4;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored)
    }
}

// ────────────────────────────────────────────────────────────────
// Feature extraction helpers
// ────────────────────────────────────────────────────────────────

/// Extract StrategyFeatures from a GeneratedStrategyPlan.
fn extract_strategy_features(plan: &GeneratedStrategyPlan) -> StrategyFeatures {
    StrategyFeatures {
        quality_score: plan.confidence,
        expected_success: plan.confidence,
        expected_value: 1.0,
        confidence: plan.confidence,
        goal_bucket_id: plan.goal_bucket_id,
        behavior_signature_hash: compute_plan_hash(plan),
    }
}

/// Extract EventFeatures from a GeneratedActionPlan's expected event.
fn extract_action_event_features(action: &GeneratedActionPlan) -> EventFeatures {
    EventFeatures {
        event_type_hash: hash_str(&action.expected_event.event_type),
        action_name_hash: hash_str(&action.action_type),
        context_fingerprint: 0, // will be filled by caller in production
        outcome_success: if action.expected_event.expected_outcome == "success" {
            1.0
        } else {
            0.0
        },
        significance: action.expected_event.expected_significance,
        temporal_delta_ns: 0.0,
        duration_ns: action.timeout_ms.unwrap_or(10000) as f64 * 1e6,
    }
}

/// Compute a hash for a plan (used as behavior_signature_hash).
fn compute_plan_hash(plan: &GeneratedStrategyPlan) -> u64 {
    let mut hash: u64 = plan.goal_bucket_id;
    for step in &plan.steps {
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(hash_str(&step.action_type));
    }
    hash
}

/// Simple string hash (FNV-1a).
fn hash_str(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Default critic report when no critic is available.
fn default_report() -> CriticReport {
    CriticReport {
        total_energy: 0.0,
        policy_strategy_energy: 0.0,
        strategy_memory_energy: 0.0,
        memory_event_energy: 0.0,
        novelty_z: 0.0,
        is_novel: false,
        mismatch_layer: agent_db_world_model::MismatchLayer::None,
        confidence: 0.0,
        support_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_generator::{MockActionGenerator, MockStrategyGenerator};
    use crate::types::{EventContext, MemoryContext, StrategyContext, StrategyGenerationRequest};
    use agent_db_world_model::EbmWorldModel;

    fn make_request() -> StrategyGenerationRequest {
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
            constraints: vec![],
            k_candidates: 3,
        }
    }

    fn make_policy() -> PolicyFeatures {
        PolicyFeatures {
            goal_count: 2,
            top_goal_priority: 0.8,
            resource_cpu_percent: 30.0,
            resource_memory_bytes: 1_000_000_000,
            context_fingerprint: 12345,
        }
    }

    #[tokio::test]
    async fn test_generate_strategies_no_critic() {
        let orchestrator = PlanningOrchestrator::new(PlanningConfig::default());
        let generator = MockStrategyGenerator::new(3);
        let request = make_request();
        let policy = make_policy();

        let (scored, decision) = orchestrator
            .generate_strategies(request, &generator, None, &policy)
            .await
            .unwrap();

        assert_eq!(scored.len(), 3);
        // Without critic, all should be accepted (confidence=0 → below min_confidence gate)
        assert!(selector::is_accepted(&decision));
    }

    #[tokio::test]
    async fn test_generate_strategies_with_critic() {
        let orchestrator = PlanningOrchestrator::new(PlanningConfig::default());
        let generator = MockStrategyGenerator::new(3);
        let critic = EbmWorldModel::new(Default::default());
        let request = make_request();
        let policy = make_policy();

        let (scored, decision) = orchestrator
            .generate_strategies(request, &generator, Some(&critic), &policy)
            .await
            .unwrap();

        assert!(!scored.is_empty());
        // With untrained critic, confidence is 0 → accept gate
        assert!(selector::is_accepted(&decision));
    }

    #[tokio::test]
    async fn test_generate_strategies_failing_generator() {
        let orchestrator = PlanningOrchestrator::new(PlanningConfig::default());
        let generator = MockStrategyGenerator::failing();
        let request = make_request();
        let policy = make_policy();

        let result = orchestrator
            .generate_strategies(request, &generator, None, &policy)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_actions_no_critic() {
        let orchestrator = PlanningOrchestrator::new(PlanningConfig::default());
        let strategy_gen = MockStrategyGenerator::new(1);
        let action_gen = MockActionGenerator::new(2);

        let strategy = strategy_gen
            .generate(make_request())
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

        let policy = make_policy();
        let memory = MemoryFeatures {
            tier: 1,
            strength: 0.8,
            access_count: 10,
            context_fingerprint: 12345,
            goal_bucket_id: 100,
        };

        let scored = orchestrator
            .generate_actions(request, &action_gen, None, &policy, &memory, None)
            .await
            .unwrap();

        assert_eq!(scored.len(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_deterministic() {
        let orchestrator = PlanningOrchestrator::new(PlanningConfig::default());
        let generator = MockStrategyGenerator::new(3);
        let request = make_request();
        let policy = make_policy();

        let (scored1, _) = orchestrator
            .generate_strategies(request.clone(), &generator, None, &policy)
            .await
            .unwrap();
        let (scored2, _) = orchestrator
            .generate_strategies(request, &generator, None, &policy)
            .await
            .unwrap();

        // Same input → same output
        assert_eq!(scored1.len(), scored2.len());
        for (a, b) in scored1.iter().zip(scored2.iter()) {
            assert_eq!(a.plan.goal_description, b.plan.goal_description);
        }
    }

    #[test]
    fn test_hash_str_deterministic() {
        assert_eq!(hash_str("test"), hash_str("test"));
        assert_ne!(hash_str("test"), hash_str("other"));
    }

    #[test]
    fn test_extract_strategy_features() {
        let plan = GeneratedStrategyPlan {
            goal_bucket_id: 42,
            goal_description: "test".to_string(),
            steps: vec![],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.9,
            rationale: None,
        };
        let features = extract_strategy_features(&plan);
        assert_eq!(features.goal_bucket_id, 42);
        assert_eq!(features.quality_score, 0.9);
    }
}
