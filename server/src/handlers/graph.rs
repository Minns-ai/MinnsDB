// Graph visualization and mutation handlers

use crate::errors::ApiError;
use crate::models::{
    GraphContextQuery, GraphEdgeResponse, GraphNodeResponse, GraphPersistResponse, GraphQuery,
    GraphResponse, StatsResponse,
};
use crate::state::AppState;
use crate::write_lanes::WriteJob;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Serialize;
use tracing::info;

// GET /api/graph - Get graph structure
pub async fn get_graph(
    State(state): State<AppState>,
    Query(query): Query<GraphQuery>,
) -> Result<Json<GraphResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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

// POST /api/graph/persist - Force-flush graph state to disk
pub async fn persist_graph(
    State(state): State<AppState>,
) -> Result<Json<GraphPersistResponse>, ApiError> {
    info!("Force-persisting graph state to disk");

    let result = state
        .write_lanes
        .submit_and_await(0, |tx| WriteJob::PersistGraph { result_tx: tx })
        .await?;

    let nodes_persisted = result["nodes_persisted"].as_u64().unwrap_or(0) as usize;
    let edges_persisted = result["edges_persisted"].as_u64().unwrap_or(0) as usize;

    Ok(Json(GraphPersistResponse {
        success: true,
        nodes_persisted,
        edges_persisted,
    }))
}

// GET /api/stats - Get system statistics
pub async fn get_stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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

// -- Delete responses --

#[derive(Serialize)]
pub struct DeleteNodeResponse {
    pub deleted: bool,
    pub node_id: u64,
}

#[derive(Serialize)]
pub struct DeleteEdgeResponse {
    pub deleted: bool,
    pub edge_id: u64,
}

// DELETE /api/graph/nodes/:id - Hard delete a node and all its edges
pub async fn delete_node(
    State(state): State<AppState>,
    Path(node_id): Path<u64>,
) -> Result<Json<DeleteNodeResponse>, ApiError> {
    info!("Deleting node {}", node_id);

    if state.engine.delete_node(node_id).await {
        Ok(Json(DeleteNodeResponse {
            deleted: true,
            node_id,
        }))
    } else {
        Err(ApiError::NotFound(format!("node {} not found", node_id)))
    }
}

// DELETE /api/graph/edges/:id - Hard delete a single edge
pub async fn delete_edge(
    State(state): State<AppState>,
    Path(edge_id): Path<u64>,
) -> Result<Json<DeleteEdgeResponse>, ApiError> {
    info!("Deleting edge {}", edge_id);

    if state.engine.delete_edge(edge_id).await {
        Ok(Json(DeleteEdgeResponse {
            deleted: true,
            edge_id,
        }))
    } else {
        Err(ApiError::NotFound(format!("edge {} not found", edge_id)))
    }
}
