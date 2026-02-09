// Graph visualization handlers

use crate::errors::ApiError;
use crate::models::{
    GraphContextQuery, GraphEdgeResponse, GraphNodeResponse, GraphQuery, GraphResponse,
    StatsResponse,
};
use crate::state::AppState;
use axum::{
    extract::{Query, State},
    Json,
};
use tracing::info;

// GET /api/graph - Get graph structure
pub async fn get_graph(
    State(state): State<AppState>,
    Query(query): Query<GraphQuery>,
) -> Result<Json<GraphResponse>, ApiError> {
    info!("Getting graph structure");

    let graph_data = state
        .engine
        .get_graph_structure(query.limit, query.session_id, query.agent_type)
        .await;

    let nodes: Vec<GraphNodeResponse> = graph_data
        .nodes
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            label: n.label.unwrap_or_else(|| format!("Node {}", n.id)),
            node_type: n.node_type,
            created_at: n.created_at,
            properties: n.properties,
        })
        .collect();

    let edges: Vec<GraphEdgeResponse> = graph_data
        .edges
        .into_iter()
        .map(|e| GraphEdgeResponse {
            id: e.id,
            from: e.from,
            to: e.to,
            edge_type: e.edge_type,
            weight: e.weight,
            confidence: e.confidence,
        })
        .collect();

    Ok(Json(GraphResponse { nodes, edges }))
}

// GET /api/graph/context - Get context-centered graph structure
pub async fn get_graph_for_context(
    State(state): State<AppState>,
    Query(query): Query<GraphContextQuery>,
) -> Result<Json<GraphResponse>, ApiError> {
    info!(
        "Getting graph structure for context: {}",
        query.context_hash
    );

    let graph_data = state
        .engine
        .get_graph_structure_for_context(
            query.context_hash,
            query.limit,
            query.session_id,
            query.agent_type,
        )
        .await;

    let nodes: Vec<GraphNodeResponse> = graph_data
        .nodes
        .into_iter()
        .map(|n| GraphNodeResponse {
            id: n.id,
            label: n.label.unwrap_or_else(|| format!("Node {}", n.id)),
            node_type: n.node_type,
            created_at: n.created_at,
            properties: n.properties,
        })
        .collect();

    let edges: Vec<GraphEdgeResponse> = graph_data
        .edges
        .into_iter()
        .map(|e| GraphEdgeResponse {
            id: e.id,
            from: e.from,
            to: e.to,
            edge_type: e.edge_type,
            weight: e.weight,
            confidence: e.confidence,
        })
        .collect();

    Ok(Json(GraphResponse { nodes, edges }))
}

// GET /api/stats - Get system statistics
pub async fn get_stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, ApiError> {
    info!("Getting system statistics");

    let stats = state.engine.get_engine_stats().await;
    let store_metrics = state.engine.get_store_metrics().await;

    Ok(Json(StatsResponse {
        total_events_processed: stats.total_events_processed,
        total_nodes_created: stats.total_nodes_created,
        total_episodes_detected: stats.total_episodes_detected,
        total_memories_formed: stats.total_memories_formed,
        total_strategies_extracted: stats.total_strategies_extracted,
        total_reinforcements_applied: stats.total_reinforcements_applied,
        average_processing_time_ms: stats.average_processing_time_ms,
        stores: store_metrics,
    }))
}
