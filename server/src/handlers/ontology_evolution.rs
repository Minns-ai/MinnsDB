// Ontology Evolution API handlers

use crate::errors::ApiError;
use crate::state::AppState;
use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

/// Request body for ontology TTL upload.
#[derive(Deserialize)]
pub struct UploadOntologyRequest {
    /// Raw Turtle (TTL) content to parse and register.
    pub ttl: String,
}

/// POST /api/ontology/upload — Upload new ontology TTL, register properties, run cascade inference.
pub async fn upload_ontology(
    State(state): State<AppState>,
    Json(body): Json<UploadOntologyRequest>,
) -> Result<Json<Value>, ApiError> {
    match state.engine.upload_ontology_ttl(&body.ttl).await {
        Ok((registered, cascade_updates)) => Ok(Json(json!({
            "status": "ok",
            "properties_registered": registered,
            "cascade_properties_updated": cascade_updates,
        }))),
        Err(e) => Err(ApiError::BadRequest(e)),
    }
}

/// POST /api/ontology/cascade-inference — Run LLM cascade inference on current ontology.
pub async fn run_cascade_inference(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    match state.engine.run_ontology_cascade_inference().await {
        Ok(updated) => Ok(Json(json!({
            "status": "ok",
            "properties_updated": updated,
        }))),
        Err(e) => Err(ApiError::Internal(format!("Cascade inference failed: {}", e))),
    }
}

/// GET /api/ontology/properties — List all current ontology properties.
pub async fn list_ontology_properties(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let ids = state.engine.list_ontology_property_ids();
    let properties = state.engine.list_ontology_properties_json();
    Ok(Json(json!({ "properties": properties, "count": ids.len() })))
}

/// GET /api/ontology/observations — List tracked predicate observations.
pub async fn list_ontology_observations(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let (observations, stats) = state.engine.list_ontology_observations().await;
    Ok(Json(json!({
        "observations": observations,
        "stats": stats,
    })))
}

/// POST /api/ontology/discover — Trigger behavior inference + hierarchy discovery + cascade inference.
pub async fn discover_ontology(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    // Phase 1: Behavior inference
    let inference_ids = state
        .engine
        .run_ontology_evolution_pass()
        .await
        .map_err(|e| ApiError::Internal(format!("Evolution pass failed: {}", e)))?;

    // Phase 2: LLM hierarchy discovery (if LLM available)
    let hierarchy_ids: Vec<u64> = state
        .engine
        .run_ontology_hierarchy_discovery()
        .await
        .unwrap_or_default();

    // Phase 3: LLM cascade/temporal dependency inference
    // Analyzes all properties and infers cascadeDependents/cascadeDependent relationships
    let cascade_updates: usize = state
        .engine
        .run_ontology_cascade_inference()
        .await
        .unwrap_or(0);

    let all_ids: Vec<u64> = inference_ids.into_iter().chain(hierarchy_ids).collect();

    Ok(Json(json!({
        "proposals_created": all_ids.len(),
        "proposal_ids": all_ids,
        "cascade_properties_updated": cascade_updates,
    })))
}

/// GET /api/ontology/proposals — List all proposals.
pub async fn list_ontology_proposals(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let proposals = state.engine.ontology_evolution_proposals().await;
    let count = proposals.len();
    Ok(Json(json!({ "proposals": proposals, "count": count })))
}

/// GET /api/ontology/proposals/:id — Get proposal details.
pub async fn get_ontology_proposal(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, ApiError> {
    let proposals = state.engine.ontology_evolution_proposals().await;
    match proposals.iter().find(|p| p.id == id) {
        Some(p) => Ok(Json(serde_json::to_value(p).unwrap_or(json!(null)))),
        None => Err(ApiError::NotFound(format!("Proposal {} not found", id))),
    }
}

/// POST /api/ontology/proposals/:id/approve — Approve and apply a proposal.
pub async fn approve_ontology_proposal(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, ApiError> {
    match state.engine.approve_and_apply_ontology_proposal(id).await {
        Ok(registered) => Ok(Json(json!({
            "status": "applied",
            "properties_registered": registered,
        }))),
        Err(e) => Err(ApiError::BadRequest(e)),
    }
}

/// POST /api/ontology/proposals/:id/reject — Reject a proposal.
pub async fn reject_ontology_proposal(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, ApiError> {
    if state.engine.reject_ontology_proposal(id).await {
        Ok(Json(json!({ "status": "rejected" })))
    } else {
        Err(ApiError::BadRequest(format!(
            "Could not reject proposal {} (not pending)",
            id
        )))
    }
}

/// GET /api/ontology/stats — Ontology evolution statistics.
pub async fn ontology_evolution_stats(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let stats = state.engine.ontology_evolution_stats().await;
    Ok(Json(serde_json::to_value(stats).unwrap_or(json!(null))))
}
