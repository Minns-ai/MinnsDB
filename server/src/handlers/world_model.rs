// World model + planning handlers

use crate::errors::ApiError;
use crate::state::AppState;
use crate::write_lanes::WriteJob;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Deserialize;
use std::sync::Arc;

// GET /api/world-model/stats - Get world model energy statistics (extended with planning info)
pub async fn get_world_model_stats(State(state): State<AppState>) -> impl IntoResponse {
    let _permit = match state.read_gate.acquire().await {
        Ok(p) => p,
        Err(e) => return ApiError::ServiceUnavailable(e).into_response(),
    };

    match state.engine.get_world_model_stats().await {
        Some(stats) => Json(serde_json::json!({
            "enabled": true,
            "mode": format!("{:?}", state.engine.config.effective_world_model_mode()),
            "running_mean": stats.running_mean,
            "running_variance": stats.running_variance,
            "total_scored": stats.total_scored,
            "total_trained": stats.total_trained,
            "avg_loss": stats.avg_loss,
            "is_warmed_up": stats.is_warmed_up,
            "planning": {
                "strategy_generation_enabled": state.engine.config.planning_config.enable_strategy_generation,
                "action_generation_enabled": state.engine.config.planning_config.enable_action_generation,
            }
        }))
        .into_response(),
        None => Json(serde_json::json!({
            "enabled": false,
            "mode": format!("{:?}", state.engine.config.effective_world_model_mode()),
            "planning": {
                "strategy_generation_enabled": state.engine.config.planning_config.enable_strategy_generation,
                "action_generation_enabled": state.engine.config.planning_config.enable_action_generation,
            }
        }))
        .into_response(),
    }
}

// ─────────────────── Planning endpoints ───────────────────

#[derive(Deserialize)]
pub struct GenerateStrategiesRequest {
    pub goal_description: String,
    #[serde(default)]
    pub goal_bucket_id: u64,
    #[serde(default)]
    pub context_fingerprint: u64,
}

// POST /api/planning/strategies
pub async fn generate_strategies(
    State(state): State<AppState>,
    Json(body): Json<GenerateStrategiesRequest>,
) -> impl IntoResponse {
    match state
        .write_lanes
        .submit_and_await(body.context_fingerprint, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    match engine
                        .generate_strategy_candidates(
                            &body.goal_description,
                            body.goal_bucket_id,
                            body.context_fingerprint,
                        )
                        .await
                    {
                        Ok(candidates) => {
                            let results: Vec<serde_json::Value> = candidates
                                .iter()
                                .map(|c| {
                                    serde_json::json!({
                                        "goal_description": c.plan.goal_description,
                                        "steps": c.plan.steps.len(),
                                        "confidence": c.plan.confidence,
                                        "total_energy": c.report.total_energy,
                                        "decision": format!("{:?}", c.decision),
                                    })
                                })
                                .collect();
                            Ok(serde_json::json!({
                                "ok": true,
                                "candidates": results,
                            }))
                        },
                        Err(e) => Err(e.to_string()),
                    }
                })
            }),
            result_tx: tx,
        })
        .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => ApiError::from(e).into_response(),
    }
}

#[derive(Deserialize)]
pub struct GenerateActionsRequest {
    pub goal_description: String,
    #[serde(default)]
    pub goal_bucket_id: u64,
    #[serde(default)]
    pub step_index: usize,
    #[serde(default)]
    pub context_fingerprint: u64,
}

// POST /api/planning/actions
pub async fn generate_actions(
    State(state): State<AppState>,
    Json(body): Json<GenerateActionsRequest>,
) -> impl IntoResponse {
    let context_fingerprint = body.context_fingerprint;
    match state
        .write_lanes
        .submit_and_await(context_fingerprint, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let strategy = build_minimal_strategy(
                        body.goal_bucket_id,
                        body.goal_description,
                        body.step_index,
                    );
                    match engine
                        .generate_action_candidates(
                            &strategy,
                            body.step_index,
                            body.context_fingerprint,
                        )
                        .await
                    {
                        Ok(actions) => {
                            let results: Vec<serde_json::Value> = actions
                                .iter()
                                .map(|a| {
                                    serde_json::json!({
                                        "action_type": a.plan.action_type,
                                        "confidence": a.plan.confidence,
                                        "energy": a.energy,
                                        "feasibility": a.feasibility,
                                    })
                                })
                                .collect();
                            Ok(serde_json::json!({
                                "ok": true,
                                "actions": results,
                            }))
                        },
                        Err(e) => Err(e.to_string()),
                    }
                })
            }),
            result_tx: tx,
        })
        .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => ApiError::from(e).into_response(),
    }
}

// ─────────────────── Plan-for-Goal (Phase 5) ───────────────────

#[derive(Deserialize)]
pub struct PlanForGoalRequest {
    pub goal_description: String,
    #[serde(default)]
    pub goal_bucket_id: u64,
    #[serde(default)]
    pub context_fingerprint: u64,
    #[serde(default)]
    pub session_id: u64,
}

// POST /api/planning/plan
pub async fn plan_for_goal(
    State(state): State<AppState>,
    Json(body): Json<PlanForGoalRequest>,
) -> impl IntoResponse {
    let session_id = body.session_id;
    match state
        .write_lanes
        .submit_and_await(session_id, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    match engine
                        .plan_for_goal(
                            &body.goal_description,
                            body.goal_bucket_id,
                            body.context_fingerprint,
                            body.session_id,
                        )
                        .await
                    {
                        Ok(result) => {
                            let strategies: Vec<serde_json::Value> = result
                                .strategy_candidates
                                .iter()
                                .map(|c| {
                                    serde_json::json!({
                                        "goal_description": c.plan.goal_description,
                                        "steps": c.plan.steps.len(),
                                        "confidence": c.plan.confidence,
                                        "total_energy": c.report.total_energy,
                                        "decision": format!("{:?}", c.decision),
                                    })
                                })
                                .collect();
                            let actions: Vec<serde_json::Value> = result
                                .action_candidates
                                .iter()
                                .map(|a| {
                                    serde_json::json!({
                                        "action_type": a.plan.action_type,
                                        "confidence": a.plan.confidence,
                                        "energy": a.energy,
                                        "feasibility": a.feasibility,
                                    })
                                })
                                .collect();
                            Ok(serde_json::json!({
                                "ok": true,
                                "mode": format!("{:?}", result.mode),
                                "goal_description": result.goal_description,
                                "goal_bucket_id": result.goal_bucket_id,
                                "strategy_candidates": strategies,
                                "action_candidates": actions,
                            }))
                        },
                        Err(e) => Err(e.to_string()),
                    }
                })
            }),
            result_tx: tx,
        })
        .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => ApiError::from(e).into_response(),
    }
}

// ─────────────────── Execution (Phase 6) ───────────────────

#[derive(Deserialize)]
pub struct StartExecutionRequest {
    pub goal_description: String,
    #[serde(default)]
    pub goal_bucket_id: u64,
    #[serde(default)]
    pub context_fingerprint: u64,
    #[serde(default)]
    pub session_id: u64,
}

// POST /api/planning/execute
pub async fn start_execution(
    State(state): State<AppState>,
    Json(body): Json<StartExecutionRequest>,
) -> impl IntoResponse {
    let session_id = body.session_id;
    match state
        .write_lanes
        .submit_and_await(session_id, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let strategy =
                        build_minimal_strategy(body.goal_bucket_id, body.goal_description, 0);
                    match engine
                        .start_execution(strategy, body.context_fingerprint, body.session_id)
                        .await
                    {
                        Ok(exec_id) => Ok(serde_json::json!({
                            "ok": true,
                            "execution_id": exec_id,
                        })),
                        Err(e) => Err(e.to_string()),
                    }
                })
            }),
            result_tx: tx,
        })
        .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => ApiError::from(e).into_response(),
    }
}

#[derive(Deserialize)]
pub struct ValidateExecutionRequest {
    pub execution_id: u64,
    pub event: serde_json::Value,
}

// POST /api/planning/validate
pub async fn validate_execution_event(
    State(state): State<AppState>,
    Json(body): Json<ValidateExecutionRequest>,
) -> impl IntoResponse {
    // Parse the event from JSON before submitting to write lane
    let event: agent_db_events::Event = match serde_json::from_value(body.event) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "ok": false,
                    "error": format!("Invalid event: {}", e),
                })),
            )
                .into_response();
        },
    };

    let execution_id = body.execution_id;
    match state
        .write_lanes
        .submit_and_await(execution_id, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    match engine
                        .validate_execution_event(execution_id, &event)
                        .await
                    {
                        Ok(result) => Ok(serde_json::json!({
                            "ok": true,
                            "prediction_error": {
                                "total_z": result.prediction_error.total_z,
                                "event_z": result.prediction_error.event_z,
                                "memory_z": result.prediction_error.memory_z,
                                "strategy_z": result.prediction_error.strategy_z,
                                "mismatch_layer": format!("{:?}", result.prediction_error.mismatch_layer),
                            },
                            "repair_triggered": result.repair_triggered,
                            "repair_result": result.repair_result.as_ref().map(|r| {
                                serde_json::json!({
                                    "scope": format!("{:?}", r.scope),
                                    "repaired_actions": r.repaired_actions.len(),
                                    "repaired_strategies": r.repaired_strategies.len(),
                                })
                            }),
                        })),
                        Err(e) => Err(e.to_string()),
                    }
                })
            }),
            result_tx: tx,
        })
        .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => ApiError::from(e).into_response(),
    }
}

fn build_minimal_strategy(
    goal_bucket_id: u64,
    goal_description: String,
    step_index: usize,
) -> agent_db_planning::GeneratedStrategyPlan {
    agent_db_planning::GeneratedStrategyPlan {
        goal_bucket_id,
        goal_description,
        steps: vec![agent_db_planning::GeneratedStep {
            step_number: step_index as u32,
            step_kind: agent_db_planning::StepKind::Action,
            action_type: "generated".to_string(),
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
        confidence: 0.5,
        rationale: None,
    }
}
