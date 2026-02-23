//! Execution state machine and repair loop.
//!
//! Manages the lifecycle of plan execution:
//! - Start execution with a strategy and context
//! - Validate incoming events against predictions
//! - Trigger repair when prediction errors exceed thresholds
//! - Advance through strategy steps

use super::*;
use agent_db_planning::repair::{self, RepairScope};

/// State of an active execution.
#[derive(Debug)]
pub struct ExecutionState {
    /// The strategy being executed.
    pub strategy: agent_db_planning::GeneratedStrategyPlan,
    /// Current step index within the strategy.
    pub current_step_index: usize,
    /// Current active action (if any).
    pub current_action: Option<agent_db_planning::GeneratedActionPlan>,
    /// Consecutive action repairs without advancing.
    pub consecutive_action_repairs: u32,
    /// Total repairs across all steps.
    pub total_repairs: u32,
    /// Goal description for context.
    pub goal_description: String,
    /// Context fingerprint for scoring.
    pub context_fingerprint: u64,
    /// Session ID for event correlation.
    pub session_id: u64,
}

/// Result of validating an event against the execution's predictions.
#[derive(Debug)]
pub struct ExecutionValidationResult {
    /// The prediction error report from the world model.
    pub prediction_error: agent_db_world_model::PredictionErrorReport,
    /// Whether repair was triggered.
    pub repair_triggered: bool,
    /// Repair result (if repair was triggered).
    pub repair_result: Option<RepairResult>,
}

/// Result of a repair operation.
#[derive(Debug)]
pub struct RepairResult {
    /// The scope of repair that was performed.
    pub scope: RepairScope,
    /// Repaired action plans (if action repair).
    pub repaired_actions: Vec<agent_db_planning::GeneratedActionPlan>,
    /// Revised strategy plans (if strategy revision).
    pub repaired_strategies: Vec<agent_db_planning::GeneratedStrategyPlan>,
}

impl GraphEngine {
    /// Start executing a strategy.
    ///
    /// Requires `GenerationMode::Full`. Stores the execution state and returns
    /// an execution ID for subsequent validation and advancement calls.
    pub async fn start_execution(
        &self,
        strategy: agent_db_planning::GeneratedStrategyPlan,
        context_fingerprint: u64,
        session_id: u64,
    ) -> GraphResult<u64> {
        if self.config.planning_config.generation_mode != agent_db_planning::GenerationMode::Full {
            return Err(GraphError::OperationError(
                "Execution requires GenerationMode::Full".to_string(),
            ));
        }

        let exec_id = self
            .next_execution_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let state = ExecutionState {
            goal_description: strategy.goal_description.clone(),
            strategy,
            current_step_index: 0,
            current_action: None,
            consecutive_action_repairs: 0,
            total_repairs: 0,
            context_fingerprint,
            session_id,
        };

        self.active_executions
            .insert(exec_id, Arc::new(RwLock::new(state)));

        tracing::info!(
            exec_id,
            session_id,
            "Started execution"
        );

        Ok(exec_id)
    }

    /// Validate an event against the active execution's predictions.
    ///
    /// Computes prediction error via the world model and triggers repair
    /// if the error exceeds the configured threshold.
    pub async fn validate_execution_event(
        &self,
        exec_id: u64,
        event: &Event,
    ) -> GraphResult<ExecutionValidationResult> {
        let exec_arc = self
            .active_executions
            .get(&exec_id)
            .ok_or_else(|| {
                GraphError::OperationError(format!("Execution {} not found", exec_id))
            })?
            .clone();

        let exec_state = exec_arc.read().await;

        // Need world model for prediction error
        let wm = self.world_model.as_ref().ok_or_else(|| {
            GraphError::OperationError("World model not initialized".to_string())
        })?;

        let wm_guard = wm.read().await;

        let event_features = world_model::extract_event_features_raw(event);
        let memory_features = agent_db_world_model::MemoryFeatures {
            tier: 0,
            strength: 0.5,
            access_count: 1,
            context_fingerprint: exec_state.context_fingerprint,
            goal_bucket_id: exec_state.strategy.goal_bucket_id,
        };
        let strategy_features = agent_db_world_model::StrategyFeatures {
            quality_score: exec_state.strategy.confidence,
            expected_success: exec_state.strategy.confidence,
            expected_value: 0.5,
            confidence: exec_state.strategy.confidence,
            goal_bucket_id: exec_state.strategy.goal_bucket_id,
            behavior_signature_hash: 0,
        };
        let policy_features = agent_db_world_model::PolicyFeatures {
            goal_count: 1,
            top_goal_priority: 0.8,
            resource_cpu_percent: 0.0,
            resource_memory_bytes: 0,
            context_fingerprint: exec_state.context_fingerprint,
        };

        let error = wm_guard.prediction_error(
            &event_features,
            &memory_features,
            &strategy_features,
            &policy_features,
        );

        drop(wm_guard);

        // Check if repair is needed
        let should_repair =
            repair::should_repair(&error, &self.config.planning_config);

        if !should_repair {
            return Ok(ExecutionValidationResult {
                prediction_error: error,
                repair_triggered: false,
                repair_result: None,
            });
        }

        // Drop read lock before taking write lock for repair
        let consecutive = exec_state.consecutive_action_repairs;
        drop(exec_state);

        let repair_result = self
            .handle_prediction_error(exec_id, &error, consecutive)
            .await?;

        Ok(ExecutionValidationResult {
            prediction_error: error,
            repair_triggered: true,
            repair_result: Some(repair_result),
        })
    }

    /// Handle a prediction error by determining repair scope and executing repair.
    async fn handle_prediction_error(
        &self,
        exec_id: u64,
        error: &agent_db_world_model::PredictionErrorReport,
        consecutive_action_repairs: u32,
    ) -> GraphResult<RepairResult> {
        let scope = repair::determine_repair_scope(error, consecutive_action_repairs);

        tracing::info!(
            exec_id,
            total_z = error.total_z,
            ?scope,
            "Handling prediction error"
        );

        match scope {
            RepairScope::ActionRepair => {
                let repaired = self.repair_current_action(exec_id, error).await?;
                // Update execution state
                if let Some(exec_arc) = self.active_executions.get(&exec_id) {
                    let mut state = exec_arc.write().await;
                    state.consecutive_action_repairs += 1;
                    state.total_repairs += 1;
                }
                Ok(RepairResult {
                    scope,
                    repaired_actions: repaired,
                    repaired_strategies: vec![],
                })
            }
            RepairScope::StrategyRevision => {
                let revised = self.revise_execution_strategy(exec_id, error).await?;
                // Reset action repair counter on strategy revision
                if let Some(exec_arc) = self.active_executions.get(&exec_id) {
                    let mut state = exec_arc.write().await;
                    state.consecutive_action_repairs = 0;
                    state.total_repairs += 1;
                }
                Ok(RepairResult {
                    scope,
                    repaired_actions: vec![],
                    repaired_strategies: revised,
                })
            }
            RepairScope::PolicyObservation => {
                tracing::warn!(
                    exec_id,
                    total_z = error.total_z,
                    "Policy-level mismatch detected — no automated repair, logging only"
                );
                Ok(RepairResult {
                    scope,
                    repaired_actions: vec![],
                    repaired_strategies: vec![],
                })
            }
        }
    }

    /// Repair the current action using the action generator.
    async fn repair_current_action(
        &self,
        exec_id: u64,
        error: &agent_db_world_model::PredictionErrorReport,
    ) -> GraphResult<Vec<agent_db_planning::GeneratedActionPlan>> {
        let exec_arc = self
            .active_executions
            .get(&exec_id)
            .ok_or_else(|| {
                GraphError::OperationError(format!("Execution {} not found", exec_id))
            })?
            .clone();

        let exec_state = exec_arc.read().await;

        let ag = self.action_generator.as_ref().ok_or_else(|| {
            GraphError::OperationError("Action generator not initialized".to_string())
        })?;

        // Get the current step
        let step = exec_state
            .strategy
            .steps
            .get(exec_state.current_step_index)
            .ok_or_else(|| {
                GraphError::OperationError(format!(
                    "Step index {} out of bounds",
                    exec_state.current_step_index
                ))
            })?;

        let request = agent_db_planning::ActionGenerationRequest {
            strategy: exec_state.strategy.clone(),
            current_step_index: exec_state.current_step_index,
            recent_events: vec![],
            context_fingerprint: exec_state.context_fingerprint,
            n_candidates: self.config.planning_config.action_candidates_n,
        };

        let repaired = ag
            .repair(step, error, &request)
            .await
            .map_err(|e| {
                GraphError::OperationError(format!("Action repair failed: {}", e))
            })?;

        tracing::info!(
            exec_id,
            repaired_count = repaired.len(),
            "Action repair produced candidates"
        );

        Ok(repaired)
    }

    /// Revise the execution strategy using the strategy generator.
    async fn revise_execution_strategy(
        &self,
        exec_id: u64,
        error: &agent_db_world_model::PredictionErrorReport,
    ) -> GraphResult<Vec<agent_db_planning::GeneratedStrategyPlan>> {
        let exec_arc = self
            .active_executions
            .get(&exec_id)
            .ok_or_else(|| {
                GraphError::OperationError(format!("Execution {} not found", exec_id))
            })?
            .clone();

        let exec_state = exec_arc.read().await;

        let sg = self.strategy_generator.as_ref().ok_or_else(|| {
            GraphError::OperationError("Strategy generator not initialized".to_string())
        })?;

        let diagnostics = repair::error_to_diagnostics(error);
        let request = agent_db_planning::StrategyGenerationRequest {
            goal_description: exec_state.goal_description.clone(),
            goal_bucket_id: exec_state.strategy.goal_bucket_id,
            context_fingerprint: exec_state.context_fingerprint,
            relevant_memories: vec![],
            similar_strategies: vec![],
            recent_events: vec![],
            transition_hints: vec![],
            constraints: vec![],
            k_candidates: self.config.planning_config.strategy_candidates_k,
        };

        let revised = sg
            .revise(&exec_state.strategy, &diagnostics, &request)
            .await
            .map_err(|e| {
                GraphError::OperationError(format!("Strategy revision failed: {}", e))
            })?;

        tracing::info!(
            exec_id,
            revised_count = revised.len(),
            "Strategy revision produced candidates"
        );

        Ok(revised)
    }

    /// Advance the execution to the next step.
    ///
    /// Returns `true` if there are more steps, `false` if the strategy is complete.
    pub async fn advance_execution_step(&self, exec_id: u64) -> GraphResult<bool> {
        let exec_arc = self
            .active_executions
            .get(&exec_id)
            .ok_or_else(|| {
                GraphError::OperationError(format!("Execution {} not found", exec_id))
            })?
            .clone();

        let mut state = exec_arc.write().await;
        state.current_step_index += 1;
        state.consecutive_action_repairs = 0;
        state.current_action = None;

        let has_more = state.current_step_index < state.strategy.steps.len();

        tracing::info!(
            exec_id,
            step = state.current_step_index,
            has_more,
            "Advanced execution step"
        );

        Ok(has_more)
    }

    /// Set the current active action for an execution.
    pub async fn set_current_action(
        &self,
        exec_id: u64,
        action: agent_db_planning::GeneratedActionPlan,
    ) -> GraphResult<()> {
        let exec_arc = self
            .active_executions
            .get(&exec_id)
            .ok_or_else(|| {
                GraphError::OperationError(format!("Execution {} not found", exec_id))
            })?
            .clone();

        let mut state = exec_arc.write().await;
        state.current_action = Some(action);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_start_execution_requires_full_mode() {
        let engine = GraphEngine::new().await.unwrap();
        let strategy = agent_db_planning::GeneratedStrategyPlan {
            goal_bucket_id: 1,
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
        let result = engine.start_execution(strategy, 42, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Full"));
    }

    #[tokio::test]
    async fn test_start_execution_success() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.generation_mode = agent_db_planning::GenerationMode::Full;
        config.planning_config.enable_strategy_generation = true;
        config.planning_config.enable_action_generation = true;
        config.planning_config.repair_enabled = true;
        config.world_model_mode = agent_db_planning::WorldModelMode::Full;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let strategy = agent_db_planning::GeneratedStrategyPlan {
            goal_bucket_id: 1,
            goal_description: "deploy service".to_string(),
            steps: vec![agent_db_planning::GeneratedStep {
                step_number: 1,
                step_kind: agent_db_planning::StepKind::Action,
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
            }],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.8,
            rationale: None,
        };

        let exec_id = engine.start_execution(strategy, 42, 1).await.unwrap();
        assert!(exec_id > 0);
        assert!(engine.active_executions.contains_key(&exec_id));
    }

    #[tokio::test]
    async fn test_advance_execution_step() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.generation_mode = agent_db_planning::GenerationMode::Full;
        config.planning_config.enable_strategy_generation = true;
        config.world_model_mode = agent_db_planning::WorldModelMode::Full;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let strategy = agent_db_planning::GeneratedStrategyPlan {
            goal_bucket_id: 1,
            goal_description: "test".to_string(),
            steps: vec![
                agent_db_planning::GeneratedStep {
                    step_number: 1,
                    step_kind: agent_db_planning::StepKind::Action,
                    action_type: "step1".to_string(),
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
                },
                agent_db_planning::GeneratedStep {
                    step_number: 2,
                    step_kind: agent_db_planning::StepKind::Action,
                    action_type: "step2".to_string(),
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
                },
            ],
            preconditions: vec![],
            stop_conditions: vec![],
            fallback_steps: vec![],
            risk_flags: vec![],
            assumptions: vec![],
            confidence: 0.8,
            rationale: None,
        };

        let exec_id = engine.start_execution(strategy, 42, 1).await.unwrap();

        // Advance from step 0 → 1 (still has more)
        let has_more = engine.advance_execution_step(exec_id).await.unwrap();
        assert!(has_more);

        // Advance from step 1 → 2 (no more steps)
        let has_more = engine.advance_execution_step(exec_id).await.unwrap();
        assert!(!has_more);
    }

    #[tokio::test]
    async fn test_validate_execution_not_found() {
        let mut config = GraphEngineConfig::default();
        config.world_model_mode = agent_db_planning::WorldModelMode::Shadow;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let event = agent_db_events::Event::new(
            1,                          // agent_id
            "test_agent".to_string(),   // agent_type
            1,                          // session_id
            agent_db_events::EventType::Action {
                action_name: "test".to_string(),
                parameters: serde_json::json!({}),
                outcome: agent_db_events::core::ActionOutcome::Success {
                    result: serde_json::json!({"ok": true}),
                },
                duration_ns: 1_000_000,
            },
            Default::default(),
        );

        let result = engine.validate_execution_event(999, &event).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }
}
