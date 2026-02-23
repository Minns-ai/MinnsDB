//! Planning integration: strategy/action generation via the planning orchestrator.
//!
//! Wires `agent-db-planning` into `GraphEngine` with generators (LLM or mock).
//! Generates candidates, scores them via the world model critic, and supports
//! the full plan_for_goal pipeline with context population from stores.

use super::*;

fn event_type_name_static(event_type: &EventType) -> &'static str {
    match event_type {
        EventType::Action { .. } => "Action",
        EventType::Observation { .. } => "Observation",
        EventType::Cognitive { .. } => "Cognitive",
        EventType::Communication { .. } => "Communication",
        EventType::Learning { .. } => "Learning",
        EventType::Context { .. } => "Context",
    }
}

/// Result of the `plan_for_goal()` pipeline.
#[derive(Debug)]
pub struct PlanningResult {
    /// Scored strategy candidates (best first).
    pub strategy_candidates: Vec<agent_db_planning::ScoredCandidate>,
    /// Scored action candidates for the best strategy's first step (if enabled).
    pub action_candidates: Vec<agent_db_planning::ScoredAction>,
    /// The generation mode that was active.
    pub mode: agent_db_planning::GenerationMode,
    /// The goal that was planned for.
    pub goal_description: String,
    /// The goal bucket ID.
    pub goal_bucket_id: u64,
}

impl GraphEngine {
    // ─────────────────── plan_for_goal pipeline ───────────────────

    /// End-to-end planning pipeline for a goal.
    ///
    /// 1. Populates context from actual stores (memories, strategies, events)
    /// 2. Generates and scores strategy candidates
    /// 3. For the best strategy, generates action candidates (if enabled)
    /// 4. Returns the full `PlanningResult`
    pub async fn plan_for_goal(
        &self,
        goal: &str,
        goal_bucket_id: u64,
        context_fp: u64,
        session_id: u64,
    ) -> GraphResult<PlanningResult> {
        if self.config.planning_config.generation_mode
            == agent_db_planning::GenerationMode::Disabled
        {
            return Err(GraphError::OperationError(
                "Planning is disabled (GenerationMode::Disabled)".to_string(),
            ));
        }

        // Populate context from stores
        let memories = self
            .get_planning_memories(goal_bucket_id, context_fp, 5)
            .await;
        let strategies = self.get_planning_strategies(goal_bucket_id, 3).await;
        let events = self.get_planning_events(session_id, 10).await;
        let transition_hints = self.get_transition_hints(goal_bucket_id, 10).await;
        let avoidance_constraints = self.get_avoidance_constraints(5).await;

        tracing::info!(
            "plan_for_goal: goal={}, memories={}, strategies={}, events={}, transitions={}, constraints={}",
            goal,
            memories.len(),
            strategies.len(),
            events.len(),
            transition_hints.len(),
            avoidance_constraints.len(),
        );

        // Generate strategy candidates with populated context
        let strategy_candidates = self
            .generate_strategy_candidates_with_context(
                goal,
                goal_bucket_id,
                context_fp,
                memories,
                strategies,
                events.clone(),
                transition_hints,
                avoidance_constraints,
            )
            .await?;

        // Generate action candidates for best strategy's first step
        let action_candidates = if self.config.planning_config.enable_action_generation
            && !strategy_candidates.is_empty()
        {
            let best_strategy = &strategy_candidates[0].plan;
            match self
                .generate_action_candidates(best_strategy, 0, context_fp)
                .await
            {
                Ok(actions) => actions,
                Err(e) => {
                    tracing::warn!("Action generation failed (continuing without): {}", e);
                    vec![]
                },
            }
        } else {
            vec![]
        };

        Ok(PlanningResult {
            strategy_candidates,
            action_candidates,
            mode: self.config.planning_config.generation_mode,
            goal_description: goal.to_string(),
            goal_bucket_id,
        })
    }

    /// Generate strategy candidates with pre-populated context.
    #[allow(clippy::too_many_arguments)]
    async fn generate_strategy_candidates_with_context(
        &self,
        goal_description: &str,
        goal_bucket_id: u64,
        context_fingerprint: u64,
        memories: Vec<agent_db_planning::MemoryContext>,
        strategies: Vec<agent_db_planning::StrategyContext>,
        events: Vec<agent_db_planning::EventContext>,
        transition_hints: Vec<agent_db_planning::TransitionHint>,
        constraints: Vec<String>,
    ) -> GraphResult<Vec<agent_db_planning::ScoredCandidate>> {
        let Some(ref orch) = self.planning_orchestrator else {
            return Err(GraphError::OperationError(
                "Planning orchestrator not initialized".to_string(),
            ));
        };
        let Some(ref sg) = self.strategy_generator else {
            return Err(GraphError::OperationError(
                "Strategy generator not initialized".to_string(),
            ));
        };

        let request = agent_db_planning::StrategyGenerationRequest {
            goal_description: goal_description.to_string(),
            goal_bucket_id,
            context_fingerprint,
            relevant_memories: memories,
            similar_strategies: strategies,
            recent_events: events,
            transition_hints,
            constraints,
            k_candidates: self.config.planning_config.strategy_candidates_k,
        };

        let policy = agent_db_world_model::PolicyFeatures {
            goal_count: 1,
            top_goal_priority: 0.8,
            resource_cpu_percent: 0.0,
            resource_memory_bytes: 0,
            context_fingerprint,
        };

        let result = if let Some(ref wm) = self.world_model {
            let wm_guard = wm.read().await;
            if wm_guard.energy_stats().is_warmed_up {
                orch.generate_strategies(request, sg.as_ref(), Some(&*wm_guard), &policy)
                    .await
            } else {
                orch.generate_strategies(request, sg.as_ref(), None, &policy)
                    .await
            }
        } else {
            orch.generate_strategies(request, sg.as_ref(), None, &policy)
                .await
        };

        match result {
            Ok((scored, decision)) => {
                tracing::info!(
                    "plan_for_goal: {} strategy candidates, decision={:?}",
                    scored.len(),
                    std::mem::discriminant(&decision),
                );
                Ok(scored)
            },
            Err(e) => Err(GraphError::OperationError(format!(
                "Strategy generation failed: {}",
                e
            ))),
        }
    }

    // ─────────────────── Avoidance constraints ───────────────────

    /// Retrieve active avoidance claims and format them as constraint strings.
    async fn get_avoidance_constraints(&self, limit: usize) -> Vec<String> {
        let Some(ref cs) = self.claim_store else {
            return vec![];
        };
        match cs.find_avoidance_claims(limit) {
            Ok(claims) => claims
                .into_iter()
                .map(|c| format!("AVOID: {}", c.claim_text))
                .collect(),
            Err(e) => {
                tracing::warn!("Failed to retrieve avoidance claims: {}", e);
                vec![]
            },
        }
    }

    // ─────────────────── Context population helpers ───────────────────

    /// Retrieve relevant memories for planning context.
    ///
    /// Uses hierarchical retrieval (goal-bucket index → embedding sim → tier boost)
    /// so Schema/Semantic memories for the current goal surface first. Falls back
    /// to agent-scoped retrieval when no goal-relevant memories exist yet.
    async fn get_planning_memories(
        &self,
        goal_bucket_id: u64,
        context_fp: u64,
        limit: usize,
    ) -> Vec<agent_db_planning::MemoryContext> {
        let mut store = self.memory_store.write().await;

        // Build a minimal query context with goal_bucket_id and fingerprint
        // so retrieve_hierarchical can use the goal-bucket index and fingerprint match.
        let query_context = agent_db_events::core::EventContext {
            fingerprint: context_fp,
            goal_bucket_id,
            ..Default::default()
        };

        // Hierarchical: goal-bucket index scan → embedding sim → tier boosting
        // min_similarity 0.1 lets bucket matches (0.5 baseline) through while filtering noise
        let mut memories = store.retrieve_hierarchical(&query_context, limit, 0.1, None);

        // Fall back to agent-scoped retrieval if no goal-relevant memories exist yet
        if memories.is_empty() {
            memories = store.get_agent_memories(0, limit);
            if memories.is_empty() {
                memories = store.get_agent_memories(1, limit);
            }
        }

        memories
            .into_iter()
            .map(|m| agent_db_planning::MemoryContext {
                summary: m.summary.clone(),
                tier: format!("{:?}", m.tier),
                strength: m.strength,
                takeaway: if m.takeaway.is_empty() {
                    None
                } else {
                    Some(m.takeaway.clone())
                },
                causal_note: if m.causal_note.is_empty() {
                    None
                } else {
                    Some(m.causal_note.clone())
                },
            })
            .collect()
    }

    /// Retrieve similar strategies for planning context.
    async fn get_planning_strategies(
        &self,
        goal_bucket_id: u64,
        limit: usize,
    ) -> Vec<agent_db_planning::StrategyContext> {
        let store = self.strategy_store.read().await;
        let query = StrategySimilarityQuery {
            goal_ids: vec![goal_bucket_id],
            tool_names: vec![],
            result_types: vec![],
            context_hash: None,
            agent_id: None,
            min_score: 0.0,
            limit,
        };
        store
            .find_similar_strategies(query)
            .into_iter()
            .map(|(s, _score)| agent_db_planning::StrategyContext {
                name: format!("strategy_{}", s.id),
                summary: s
                    .playbook
                    .first()
                    .map(|step| step.action.clone())
                    .unwrap_or_else(|| "unnamed".to_string()),
                quality_score: s.quality_score,
                when_to_use: s.when_to_use.clone(),
            })
            .collect()
    }

    /// Retrieve recent events for planning context.
    async fn get_planning_events(
        &self,
        session_id: u64,
        limit: usize,
    ) -> Vec<agent_db_planning::EventContext> {
        let store = self.event_store.read().await;
        let order = self.event_store_order.read().await;

        order
            .iter()
            .rev()
            .filter_map(|id| store.get(id))
            .filter(|e| e.session_id == session_id)
            .take(limit)
            .map(|e| {
                let (action_name, outcome_str) = match &e.event_type {
                    EventType::Action {
                        action_name,
                        outcome,
                        ..
                    } => {
                        let outcome = match outcome {
                            agent_db_events::core::ActionOutcome::Success { .. } => {
                                "success".to_string()
                            },
                            agent_db_events::core::ActionOutcome::Failure { error, .. } => {
                                format!("failure: {}", error)
                            },
                            agent_db_events::core::ActionOutcome::Partial { .. } => {
                                "partial".to_string()
                            },
                        };
                        (Some(action_name.clone()), Some(outcome))
                    },
                    _ => (None, None),
                };
                agent_db_planning::EventContext {
                    event_type: event_type_name_static(&e.event_type).to_string(),
                    action_name,
                    outcome: outcome_str,
                    timestamp: e.timestamp,
                }
            })
            .collect()
    }

    /// Retrieve empirical transition success rates from the Markov model for this goal bucket.
    ///
    /// Returns top-K transitions sorted by observation count, filtered to those
    /// with at least 2 observations (to avoid noisy single-sample data).
    async fn get_transition_hints(
        &self,
        goal_bucket_id: u64,
        limit: usize,
    ) -> Vec<agent_db_planning::TransitionHint> {
        let tm = self.transition_model.read().await;
        let config = tm.config();
        tm.top_transitions(goal_bucket_id, limit)
            .into_iter()
            .filter(|(_, _, _, stats)| stats.count >= 2)
            .map(
                |(state, action, next_state, stats)| agent_db_planning::TransitionHint {
                    from_state: state,
                    action,
                    to_state: next_state,
                    success_rate: stats.posterior_success(config),
                    observation_count: stats.count,
                },
            )
            .collect()
    }

    // ─────────────────── Original standalone methods ───────────────────

    /// Generate and score strategy candidates for a goal.
    /// Uses generators and optional world model critic.
    /// Returns scored candidates sorted by energy (best first).
    pub async fn generate_strategy_candidates(
        &self,
        goal_description: &str,
        goal_bucket_id: u64,
        context_fingerprint: u64,
    ) -> GraphResult<Vec<agent_db_planning::ScoredCandidate>> {
        let Some(ref orch) = self.planning_orchestrator else {
            return Err(GraphError::OperationError(
                "Planning orchestrator not initialized".to_string(),
            ));
        };
        let Some(ref sg) = self.strategy_generator else {
            return Err(GraphError::OperationError(
                "Strategy generator not initialized".to_string(),
            ));
        };

        // Build request context from stores
        let request = agent_db_planning::StrategyGenerationRequest {
            goal_description: goal_description.to_string(),
            goal_bucket_id,
            context_fingerprint,
            relevant_memories: vec![],
            similar_strategies: vec![],
            recent_events: vec![],
            transition_hints: vec![],
            constraints: vec![],
            k_candidates: self.config.planning_config.strategy_candidates_k,
        };

        // Build policy features
        let policy = agent_db_world_model::PolicyFeatures {
            goal_count: 1,
            top_goal_priority: 0.8,
            resource_cpu_percent: 0.0,
            resource_memory_bytes: 0,
            context_fingerprint,
        };

        // Use world model as critic if available and warmed up
        let result: Result<
            (
                Vec<agent_db_planning::ScoredCandidate>,
                agent_db_planning::SelectionDecision,
            ),
            agent_db_planning::PlanningError,
        > = if let Some(ref wm) = self.world_model {
            let wm_guard = wm.read().await;
            if wm_guard.energy_stats().is_warmed_up {
                orch.generate_strategies(request, sg.as_ref(), Some(&*wm_guard), &policy)
                    .await
            } else {
                orch.generate_strategies(request, sg.as_ref(), None, &policy)
                    .await
            }
        } else {
            orch.generate_strategies(request, sg.as_ref(), None, &policy)
                .await
        };

        match result {
            Ok((scored, decision)) => {
                tracing::info!(
                    "Planning: generated {} strategy candidates, decision={:?}",
                    scored.len(),
                    std::mem::discriminant(&decision),
                );
                Ok(scored)
            },
            Err(e) => Err(GraphError::OperationError(format!(
                "Strategy generation failed: {}",
                e
            ))),
        }
    }

    /// Generate and score action candidates for a strategy step.
    pub async fn generate_action_candidates(
        &self,
        strategy: &agent_db_planning::GeneratedStrategyPlan,
        step_index: usize,
        context_fingerprint: u64,
    ) -> GraphResult<Vec<agent_db_planning::ScoredAction>> {
        let Some(ref orch) = self.planning_orchestrator else {
            return Err(GraphError::OperationError(
                "Planning orchestrator not initialized".to_string(),
            ));
        };
        let Some(ref ag) = self.action_generator else {
            return Err(GraphError::OperationError(
                "Action generator not initialized".to_string(),
            ));
        };

        let request = agent_db_planning::ActionGenerationRequest {
            strategy: strategy.clone(),
            current_step_index: step_index,
            recent_events: vec![],
            context_fingerprint,
            n_candidates: self.config.planning_config.action_candidates_n,
        };

        let policy = agent_db_world_model::PolicyFeatures {
            goal_count: 1,
            top_goal_priority: 0.8,
            resource_cpu_percent: 0.0,
            resource_memory_bytes: 0,
            context_fingerprint,
        };

        let memory = agent_db_world_model::MemoryFeatures {
            tier: 0,
            strength: 0.5,
            access_count: 1,
            context_fingerprint,
            goal_bucket_id: strategy.goal_bucket_id,
        };

        let result: Result<Vec<agent_db_planning::ScoredAction>, agent_db_planning::PlanningError> =
            if let Some(ref wm) = self.world_model {
                let wm_guard = wm.read().await;
                if wm_guard.energy_stats().is_warmed_up {
                    orch.generate_actions(
                        request,
                        ag.as_ref(),
                        Some(&*wm_guard),
                        &policy,
                        &memory,
                        None,
                    )
                    .await
                } else {
                    orch.generate_actions(request, ag.as_ref(), None, &policy, &memory, None)
                        .await
                }
            } else {
                orch.generate_actions(request, ag.as_ref(), None, &policy, &memory, None)
                    .await
            };

        match result {
            Ok(scored) => {
                tracing::info!(
                    "Planning: generated {} action candidates for step {}",
                    scored.len(),
                    step_index,
                );
                Ok(scored)
            },
            Err(e) => Err(GraphError::OperationError(format!(
                "Action generation failed: {}",
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_strategies_without_orchestrator() {
        let engine = GraphEngine::new().await.unwrap();
        let result = engine
            .generate_strategy_candidates("test goal", 1, 42)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not initialized"));
    }

    #[tokio::test]
    async fn test_generate_strategies_with_mock() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.enable_strategy_generation = true;
        config.planning_config.strategy_candidates_k = 3;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let result = engine
            .generate_strategy_candidates("deploy service", 1, 42)
            .await;
        assert!(result.is_ok());
        let candidates = result.unwrap();
        assert_eq!(candidates.len(), 3);
    }

    #[tokio::test]
    async fn test_generate_actions_without_orchestrator() {
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
        let result = engine.generate_action_candidates(&strategy, 0, 42).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_actions_with_mock() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.enable_action_generation = true;
        config.planning_config.action_candidates_n = 2;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let strategy = agent_db_planning::GeneratedStrategyPlan {
            goal_bucket_id: 1,
            goal_description: "deploy service".to_string(),
            steps: vec![agent_db_planning::GeneratedStep {
                step_number: 0,
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
        let result = engine.generate_action_candidates(&strategy, 0, 42).await;
        assert!(result.is_ok());
        let actions = result.unwrap();
        assert_eq!(actions.len(), 2);
    }

    #[tokio::test]
    async fn test_generate_strategies_with_world_model() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.enable_strategy_generation = true;
        config.planning_config.strategy_candidates_k = 2;
        config.world_model_mode = WorldModelMode::Shadow;
        let engine = GraphEngine::with_config(config).await.unwrap();

        // World model not warmed up, so critic won't be used
        let result = engine
            .generate_strategy_candidates("test goal", 1, 42)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_plan_for_goal_disabled() {
        let engine = GraphEngine::new().await.unwrap();
        let result = engine.plan_for_goal("test goal", 1, 42, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Disabled"));
    }

    #[tokio::test]
    async fn test_plan_for_goal_generate_and_score() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.enable_strategy_generation = true;
        config.planning_config.enable_action_generation = true;
        config.planning_config.strategy_candidates_k = 2;
        config.planning_config.action_candidates_n = 2;
        config.planning_config.generation_mode =
            agent_db_planning::GenerationMode::GenerateAndScore;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let result = engine.plan_for_goal("deploy service", 1, 42, 1).await;
        assert!(result.is_ok());
        let planning_result = result.unwrap();
        assert_eq!(planning_result.strategy_candidates.len(), 2);
        assert_eq!(planning_result.action_candidates.len(), 2);
        assert_eq!(planning_result.goal_description, "deploy service");
        assert_eq!(planning_result.goal_bucket_id, 1);
        assert_eq!(
            planning_result.mode,
            agent_db_planning::GenerationMode::GenerateAndScore
        );
    }

    #[tokio::test]
    async fn test_plan_for_goal_strategies_only() {
        let mut config = GraphEngineConfig::default();
        config.planning_config.enable_strategy_generation = true;
        config.planning_config.enable_action_generation = false;
        config.planning_config.strategy_candidates_k = 3;
        config.planning_config.generation_mode = agent_db_planning::GenerationMode::GenerateOnly;
        let engine = GraphEngine::with_config(config).await.unwrap();

        let result = engine.plan_for_goal("test goal", 1, 42, 1).await;
        assert!(result.is_ok());
        let planning_result = result.unwrap();
        assert_eq!(planning_result.strategy_candidates.len(), 3);
        assert!(planning_result.action_candidates.is_empty());
    }
}
