// Semantic memory and claims handlers

use crate::errors::ApiError;
use crate::models::{
    ClaimEntityResponse, ClaimListQuery, ClaimResponse, ClaimSearchRequest,
    EmbeddingProcessResponse, EvidenceSpanResponse, PaginationQuery,
};
use crate::state::AppState;
use crate::write_lanes::WriteJob;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use std::sync::Arc;
use tracing::info;

/// Convert a DerivedClaim into a ClaimResponse.
fn claim_to_response(
    claim: agent_db_graph::claims::DerivedClaim,
    similarity: Option<f32>,
) -> ClaimResponse {
    let tw = claim.temporal_weight();
    ClaimResponse {
        claim_id: claim.id,
        claim_text: claim.claim_text,
        confidence: claim.confidence,
        source_event_id: claim.source_event_id,
        similarity,
        evidence_spans: claim
            .supporting_evidence
            .into_iter()
            .map(|span| EvidenceSpanResponse {
                start_offset: span.start_offset,
                end_offset: span.end_offset,
                text_snippet: span.text_snippet,
            })
            .collect(),
        support_count: claim.support_count,
        status: format!("{:?}", claim.status),
        created_at: claim.created_at,
        last_accessed: claim.last_accessed,
        claim_type: claim.claim_type.to_string(),
        subject_entity: claim.subject_entity,
        expires_at: claim.expires_at,
        temporal_weight: tw,
        superseded_by: claim.superseded_by,
        entities: claim
            .entities
            .into_iter()
            .map(|e| ClaimEntityResponse {
                text: e.text,
                label: e.label,
            })
            .collect(),
    }
}

// POST /api/claims/search - Search for similar claims
pub async fn search_claims(
    State(state): State<AppState>,
    Json(payload): Json<ClaimSearchRequest>,
) -> Result<Json<Vec<ClaimResponse>>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!(
        "Searching for claims: query={} top_k={} min_similarity={}",
        payload.query_text, payload.top_k, payload.min_similarity
    );

    let results = state
        .engine
        .search_similar_claims(&payload.query_text, payload.top_k, payload.min_similarity)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to search claims: {}", e)))?;

    let responses: Vec<ClaimResponse> = results
        .into_iter()
        .map(|(claim, sim)| claim_to_response(claim, Some(sim)))
        .collect();

    info!("Found {} similar claims", responses.len());

    Ok(Json(responses))
}

// POST /api/embeddings/process - Process pending embeddings
pub async fn process_embeddings(
    State(state): State<AppState>,
    Query(params): Query<PaginationQuery>,
) -> Result<Json<EmbeddingProcessResponse>, ApiError> {
    info!(
        "Processing pending embeddings (batch_size={})",
        params.limit
    );

    let limit = params.limit;
    let result = state
        .write_lanes
        .submit_and_await(0, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let count = engine
                        .process_pending_embeddings(limit)
                        .await
                        .map_err(|e| format!("Failed to process embeddings: {}", e))?;
                    Ok(serde_json::json!({ "claims_processed": count }))
                })
            }),
            result_tx: tx,
        })
        .await?;

    let claims_processed = result["claims_processed"].as_u64().unwrap_or(0) as usize;

    info!(
        "Processed {} claims for embedding generation",
        claims_processed
    );

    Ok(Json(EmbeddingProcessResponse {
        claims_processed,
        success: true,
    }))
}

// GET /api/claims/:id - Get a specific claim
pub async fn get_claim(
    State(state): State<AppState>,
    Path(claim_id): Path<u64>,
) -> Result<Json<ClaimResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!("Fetching claim {}", claim_id);

    let claim_store = state
        .engine
        .claim_store()
        .ok_or_else(|| ApiError::Internal("Claim store not initialized".to_string()))?;

    let claim = claim_store
        .get(claim_id)
        .map_err(|e| ApiError::Internal(format!("Failed to retrieve claim: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Claim {} not found", claim_id)))?;

    Ok(Json(claim_to_response(claim, None)))
}

// GET /api/claims - List all claims
pub async fn list_claims(
    State(state): State<AppState>,
    Query(params): Query<ClaimListQuery>,
) -> Result<Json<Vec<ClaimResponse>>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!("Listing claims (limit={})", params.limit);

    let claim_store = state
        .engine
        .claim_store()
        .ok_or_else(|| ApiError::Internal("Claim store not initialized".to_string()))?;

    let claims = claim_store
        .get_all_active(params.limit)
        .map_err(|e| ApiError::Internal(format!("Failed to retrieve claims: {}", e)))?;

    let responses: Vec<ClaimResponse> = claims
        .into_iter()
        .filter(|claim| {
            if let Some(event_id) = params.event_id {
                claim.source_event_id == event_id
            } else {
                true
            }
        })
        .map(|claim| claim_to_response(claim, None))
        .collect();

    info!("Found {} claims", responses.len());

    Ok(Json(responses))
}
