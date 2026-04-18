// Advanced analytics and graph algorithm handlers

use crate::errors::ApiError;
use crate::models::{
    AnalyticsResponse, CausalPathQuery, CausalPathResponse, CentralityScoresResponse,
    CommunitiesQuery, CommunitiesResponse, CommunityResponse, IndexStatsResponse,
    LearningMetricsResponse, PprNodeScore, PprQuery, PprResponse, ReachabilityNodeResponse,
    ReachabilityQuery, ReachabilityResponse,
};
use crate::state::AppState;
use axum::extract::{Query, State};
use axum::Json;

// GET /api/analytics - Get comprehensive graph analytics
pub async fn get_analytics(
    State(state): State<AppState>,
) -> Result<Json<AnalyticsResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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
    Query(params): Query<CommunitiesQuery>,
) -> Result<Json<CommunitiesResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    let result = state
        .engine
        .detect_communities_with_algorithm(params.algorithm.as_deref())
        .await
        .map_err(|e| match &e {
            agent_db_graph::GraphError::InvalidQuery(_) => ApiError::BadRequest(e.to_string()),
            _ => ApiError::Internal(format!("Failed to detect communities: {}", e)),
        })?;

    let resolved_algorithm = params
        .algorithm
        .as_deref()
        .unwrap_or(&state.engine.config.community_algorithm)
        .to_string();

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
        algorithm: resolved_algorithm,
    }))
}

// GET /api/centrality - Get centrality scores for all nodes
pub async fn get_centrality(
    State(state): State<AppState>,
) -> Result<Json<Vec<CentralityScoresResponse>>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

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

// GET /api/ppr - Compute PersonalizedPageRank from a source node
pub async fn get_ppr(
    State(state): State<AppState>,
    Query(params): Query<PprQuery>,
) -> Result<Json<PprResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    let start = std::time::Instant::now();

    let ppr_scores = state
        .engine
        .personalized_pagerank(params.source_node_id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to compute PPR: {}", e)))?;

    let min_score = params.min_score.unwrap_or(0.001);
    let limit = params.limit.unwrap_or(100).min(10_000);

    let mut scores: Vec<PprNodeScore> = ppr_scores
        .into_iter()
        .filter(|(_, score)| *score >= min_score)
        .map(|(node_id, score)| PprNodeScore { node_id, score })
        .collect();

    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores.truncate(limit);

    tracing::info!(
        source = params.source_node_id,
        results = scores.len(),
        duration_ms = start.elapsed().as_millis() as u64,
        "PPR computed"
    );

    Ok(Json(PprResponse {
        source_node_id: params.source_node_id,
        algorithm: "personalized_pagerank".to_string(),
        scores,
    }))
}

// GET /api/reachability - Compute temporal reachability from a source node
pub async fn get_reachability(
    State(state): State<AppState>,
    Query(params): Query<ReachabilityQuery>,
) -> Result<Json<ReachabilityResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    let start = std::time::Instant::now();
    let max_hops = params.max_hops.unwrap_or(0);
    let max_results = params.max_results.unwrap_or(500).min(10_000);

    let result = state
        .engine
        .temporal_reachability_from(params.source, max_hops)
        .await
        .map_err(|e| match &e {
            agent_db_graph::GraphError::InvalidQuery(_) => ApiError::BadRequest(e.to_string()),
            _ => ApiError::Internal(format!("Failed to compute reachability: {}", e)),
        })?;

    let mut reachable: Vec<ReachabilityNodeResponse> = result
        .reachable
        .iter()
        .map(|(&node_id, record)| ReachabilityNodeResponse {
            node_id,
            origin: record.origin,
            arrival_time: record.arrival_time,
            hops: record.hops,
            predecessor: record.predecessor,
        })
        .collect();

    // Sort by arrival time for consistent output
    reachable.sort_by_key(|r| r.arrival_time);
    reachable.truncate(max_results);

    tracing::info!(
        source = params.source,
        max_hops = max_hops,
        reachable_count = result.reachable.len(),
        duration_ms = start.elapsed().as_millis() as u64,
        "Temporal reachability computed"
    );

    Ok(Json(ReachabilityResponse {
        source_node_id: params.source,
        reachable_count: result.reachable.len(),
        max_depth: result.max_depth,
        edges_traversed: result.edges_traversed,
        reachable,
    }))
}

// GET /api/causal-path - Find causal path between two nodes
pub async fn get_causal_path(
    State(state): State<AppState>,
    Query(params): Query<CausalPathQuery>,
) -> Result<Json<CausalPathResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    // Handle source == target trivially
    if params.source == params.target {
        return Ok(Json(CausalPathResponse {
            source: params.source,
            target: params.target,
            found: true,
            path: Some(vec![params.source]),
        }));
    }

    let path = state
        .engine
        .causal_path(params.source, params.target)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to compute causal path: {}", e)))?;

    let found = path.is_some();
    Ok(Json(CausalPathResponse {
        source: params.source,
        target: params.target,
        found,
        path,
    }))
}
