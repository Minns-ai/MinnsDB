// Advanced analytics and graph algorithm handlers

use crate::errors::ApiError;
use crate::models::{
    AnalyticsResponse, CentralityScoresResponse, CommunitiesResponse, CommunityResponse,
    IndexStatsResponse, LearningMetricsResponse,
};
use crate::state::AppState;
use axum::{extract::State, Json};

// GET /api/analytics - Get comprehensive graph analytics
pub async fn get_analytics(
    State(state): State<AppState>,
) -> Result<Json<AnalyticsResponse>, ApiError> {
    let metrics = state
        .engine
        .get_analytics()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get analytics: {}", e)))?;

    Ok(Json(AnalyticsResponse {
        node_count: metrics.node_count,
        edge_count: metrics.edge_count,
        connected_components: metrics.connected_components,
        largest_component_size: metrics.largest_component_size,
        average_path_length: metrics.average_path_length,
        diameter: metrics.diameter,
        clustering_coefficient: metrics.clustering_coefficient,
        average_clustering: metrics.average_clustering,
        modularity: metrics.modularity,
        community_count: metrics.community_count,
        learning_metrics: LearningMetricsResponse {
            total_events: metrics.learning_metrics.total_events,
            unique_contexts: metrics.learning_metrics.unique_contexts,
            learned_patterns: metrics.learning_metrics.learned_patterns,
            strong_memories: metrics.learning_metrics.strong_memories,
            overall_success_rate: metrics.learning_metrics.overall_success_rate,
            average_edge_weight: metrics.learning_metrics.average_edge_weight,
        },
    }))
}

// GET /api/indexes - Get property index statistics
pub async fn get_indexes(
    State(state): State<AppState>,
) -> Result<Json<Vec<IndexStatsResponse>>, ApiError> {
    let index_stats = state.engine.get_index_stats().await;

    let response: Vec<IndexStatsResponse> = index_stats
        .into_iter()
        .map(|stats| IndexStatsResponse {
            insert_count: stats.insert_count,
            query_count: stats.query_count,
            range_query_count: stats.range_query_count,
            hit_count: stats.hit_count,
            miss_count: stats.miss_count,
            last_accessed: stats.last_accessed,
        })
        .collect();

    Ok(Json(response))
}

// GET /api/communities - Detect and return graph communities
pub async fn get_communities(
    State(state): State<AppState>,
) -> Result<Json<CommunitiesResponse>, ApiError> {
    let result = state
        .engine
        .detect_communities()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to detect communities: {}", e)))?;

    let communities: Vec<CommunityResponse> = result
        .communities
        .into_iter()
        .map(|(id, nodes)| CommunityResponse {
            community_id: id,
            size: nodes.len(),
            node_ids: nodes,
        })
        .collect();

    Ok(Json(CommunitiesResponse {
        communities,
        modularity: result.modularity,
        iterations: result.iterations,
        community_count: result.community_count,
    }))
}

// GET /api/centrality - Get centrality scores for all nodes
pub async fn get_centrality(
    State(state): State<AppState>,
) -> Result<Json<Vec<CentralityScoresResponse>>, ApiError> {
    let all_centralities = state
        .engine
        .get_all_centrality_scores()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to calculate centrality: {}", e)))?;

    // Get all node IDs from one of the centrality maps
    let node_ids: Vec<u64> = all_centralities.degree.keys().copied().collect();

    let mut scores: Vec<CentralityScoresResponse> = node_ids
        .into_iter()
        .map(|node_id| {
            let combined = all_centralities.combined_score(node_id);

            CentralityScoresResponse {
                node_id,
                degree: all_centralities
                    .degree
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                betweenness: all_centralities
                    .betweenness
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                closeness: all_centralities
                    .closeness
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                eigenvector: all_centralities
                    .eigenvector
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                pagerank: all_centralities
                    .pagerank
                    .get(&node_id)
                    .copied()
                    .unwrap_or(0.0),
                combined,
            }
        })
        .collect();

    // Sort by combined score descending
    scores.sort_by(|a, b| {
        b.combined
            .partial_cmp(&a.combined)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(Json(scores))
}
