//! Subscription REST endpoints.
//!
//! POST   /api/subscriptions          — create subscription (returns initial results)
//! GET    /api/subscriptions           — list active subscriptions
//! DELETE /api/subscriptions/:id       — unsubscribe
//! GET    /api/subscriptions/:id/poll  — poll for pending updates

use crate::errors::ApiError;
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Wire protocol types (Step 2: Message Protocol)
// ---------------------------------------------------------------------------

/// Request to create a subscription.
#[derive(Deserialize)]
pub struct SubscribeRequest {
    /// MinnsQL query string.
    pub query: String,
    /// Optional group_id scope.
    #[serde(default)]
    pub group_id: Option<String>,
}

/// Response when a subscription is created.
#[derive(Serialize)]
pub struct SubscribeResponse {
    pub subscription_id: u64,
    pub initial: SubscriptionResultSet,
    pub strategy: String,
}

/// Tabular result set (shared between initial results and poll updates).
#[derive(Serialize, Clone)]
pub struct SubscriptionResultSet {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// A single subscription's update from polling.
#[derive(Serialize)]
pub struct SubscriptionUpdateResponse {
    pub subscription_id: u64,
    pub inserts: Vec<Vec<serde_json::Value>>,
    pub deletes: Vec<Vec<serde_json::Value>>,
    pub count: Option<i64>,
    pub was_full_rerun: bool,
}

/// Response from the poll endpoint.
#[derive(Serialize)]
pub struct PollResponse {
    pub updates: Vec<SubscriptionUpdateResponse>,
}

/// Summary of an active subscription for the list endpoint.
#[derive(Serialize)]
pub struct SubscriptionInfo {
    pub subscription_id: u64,
    pub query: String,
    pub strategy: String,
    pub cached_row_count: usize,
}

/// Response listing all active subscriptions.
#[derive(Serialize)]
pub struct ListSubscriptionsResponse {
    pub subscriptions: Vec<SubscriptionInfo>,
}

// ---------------------------------------------------------------------------
// Handlers (Step 3: REST Polling Endpoints)
// ---------------------------------------------------------------------------

/// POST /api/subscriptions — create a new subscription.
pub async fn create_subscription(
    State(state): State<AppState>,
    Json(request): Json<SubscribeRequest>,
) -> Result<Json<SubscribeResponse>, ApiError> {
    if request.query.len() > 4096 {
        return Err(ApiError::BadRequest(
            "Query too long (max 4096 bytes)".into(),
        ));
    }

    // Parse and plan outside of any lock.
    let ast = agent_db_graph::query_lang::parser::Parser::parse(&request.query)
        .map_err(|e| match &e {
            agent_db_graph::query_lang::QueryError::ParseError { message, position } => {
                ApiError::BadRequest(format!("Parse error at position {}: {}", position, message))
            }
            _ => ApiError::BadRequest(format!("{}", e)),
        })?;
    let plan = agent_db_graph::query_lang::planner::plan(ast)
        .map_err(|e| ApiError::BadRequest(format!("{}", e)))?;

    // Subscribe under read lock + manager lock (short hold).
    let (sub_id, initial_output, strategy) = {
        let _permit = state
            .read_gate
            .acquire()
            .await
            .map_err(ApiError::ServiceUnavailable)?;

        let inference = state.engine.inference().read().await;
        let graph = inference.graph();
        let ontology = state.engine.ontology();

        let mut sub_mgr = state.subscription_manager.lock().await;

        let (sub_id, initial_output) = sub_mgr
            .subscribe(plan, graph, &ontology)
            .map_err(|e| match e {
                agent_db_graph::query_lang::QueryError::ExecutionError(ref msg)
                    if msg.contains("Subscription limit") =>
                {
                    ApiError::ServiceUnavailable(format!("{}", e))
                }
                _ => ApiError::Internal(format!("Subscription failed: {}", e)),
            })?;

        let strategy = if let Some(sub_state) = sub_mgr.get_subscription(sub_id) {
            match &sub_state.incremental_plan.strategy {
                agent_db_graph::subscription::incremental::MaintenanceStrategy::Incremental => {
                    "incremental".to_string()
                }
                agent_db_graph::subscription::incremental::MaintenanceStrategy::FullRerun {
                    reason,
                } => format!("full_rerun: {}", reason),
            }
        } else {
            "unknown".to_string()
        };

        (sub_id, initial_output, strategy)
    };
    // All locks dropped.

    // Store query text separately (no lock contention with manager).
    state
        .subscription_queries
        .lock()
        .await
        .insert(sub_id, request.query);

    let initial = SubscriptionResultSet {
        columns: initial_output.columns,
        rows: initial_output
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(|v| v.to_json()).collect())
            .collect(),
    };

    Ok(Json(SubscribeResponse {
        subscription_id: sub_id,
        initial,
        strategy,
    }))
}

/// DELETE /api/subscriptions/:id — remove a subscription.
pub async fn delete_subscription(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let removed = {
        let mut sub_mgr = state.subscription_manager.lock().await;
        sub_mgr.unsubscribe(id)
    };
    // Lock dropped before acquiring subscription_queries.
    if removed {
        state.subscription_queries.lock().await.remove(&id);
        Ok(Json(serde_json::json!({ "unsubscribed": id })))
    } else {
        Err(ApiError::NotFound(format!(
            "Subscription {} not found",
            id
        )))
    }
}

/// GET /api/subscriptions/:id/poll — poll for pending updates.
///
/// The background task processes deltas continuously. This endpoint
/// just drains the buffered updates for the requested subscription.
pub async fn poll_subscription(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<PollResponse>, ApiError> {
    let pending = {
        let mut sub_mgr = state.subscription_manager.lock().await;
        sub_mgr.take_pending(id)
    };

    let updates: Vec<SubscriptionUpdateResponse> = pending
        .into_iter()
        .map(format_update)
        .collect();

    Ok(Json(PollResponse { updates }))
}

/// Convert an internal SubscriptionUpdate to the wire format.
fn format_update(u: agent_db_graph::subscription::manager::SubscriptionUpdate) -> SubscriptionUpdateResponse {
    SubscriptionUpdateResponse {
        subscription_id: u.subscription_id,
        inserts: u
            .inserts
            .into_iter()
            .map(|(_, values)| values.into_iter().map(|v| v.to_json()).collect())
            .collect(),
        deletes: u
            .deletes
            .into_iter()
            .map(|row_id| {
                row_id
                    .slots()
                    .iter()
                    .map(|(slot, eid)| match eid {
                        agent_db_graph::subscription::incremental::BoundEntityId::Node(nid) => {
                            serde_json::json!({"slot": slot, "node_id": nid})
                        }
                        agent_db_graph::subscription::incremental::BoundEntityId::Edge(eid) => {
                            serde_json::json!({"slot": slot, "edge_id": eid})
                        }
                    })
                    .collect()
            })
            .collect(),
        count: u.count,
        was_full_rerun: u.was_full_rerun,
    }
}

/// GET /api/subscriptions — list all active subscriptions.
pub async fn list_subscriptions(
    State(state): State<AppState>,
) -> Result<Json<ListSubscriptionsResponse>, ApiError> {
    // Collect data under lock, drop locks, then build response.
    let sub_list = {
        let sub_mgr = state.subscription_manager.lock().await;
        sub_mgr.list_subscriptions()
    };
    let queries = {
        let q = state.subscription_queries.lock().await;
        q.clone()
    };

    let subscriptions = sub_list
        .into_iter()
        .map(|(id, row_count, strategy)| SubscriptionInfo {
            subscription_id: id,
            query: queries.get(&id).cloned().unwrap_or_default(),
            strategy,
            cached_row_count: row_count,
        })
        .collect();

    Ok(Json(ListSubscriptionsResponse { subscriptions }))
}
