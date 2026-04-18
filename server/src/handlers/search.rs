// Full-text search handlers with BM25 and hybrid search

use crate::errors::ApiError;
use crate::models::{SearchRequest, SearchResponse, SearchResultItem};
use crate::state::AppState;
use agent_db_graph::indexing::SearchMode;
use axum::{extract::State, Json};
use tracing::info;

/// POST /api/search - Search nodes using BM25, semantic, or hybrid search
pub async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!(
        "Search request: query='{}' mode={:?} limit={}",
        request.query, request.mode, request.limit
    );

    let results = match request.mode {
        SearchMode::Keyword => {
            // BM25 keyword search only
            let bm25_results = state
                .engine
                .search_bm25(&request.query, request.limit)
                .await;

            let mut items = Vec::new();
            for (node_id, score) in bm25_results {
                let node = state.engine.get_node(node_id).await;
                let (node_type, properties) = if let Some(n) = node {
                    let props =
                        serde_json::to_value(&n.properties).unwrap_or(serde_json::Value::Null);
                    (n.type_name().to_string(), props)
                } else {
                    ("unknown".to_string(), serde_json::Value::Null)
                };

                items.push(SearchResultItem {
                    node_id,
                    score,
                    node_type,
                    properties,
                });
            }
            items
        },
        SearchMode::Semantic => {
            // Semantic embedding search on claims
            let semantic_results = state
                .engine
                .search_claims_semantic(&request.query, request.limit, 0.5)
                .await
                .unwrap_or_else(|e| {
                    info!("Semantic search failed, falling back to BM25: {}", e);
                    vec![]
                });

            // If no semantic results, fall back to BM25
            let results = if semantic_results.is_empty() {
                info!("No semantic results, falling back to BM25");
                state
                    .engine
                    .search_bm25(&request.query, request.limit)
                    .await
            } else {
                semantic_results
            };

            let mut items = Vec::new();
            for (node_id, score) in results {
                let node = state.engine.get_node(node_id).await;
                let (node_type, properties) = if let Some(n) = node {
                    let props =
                        serde_json::to_value(&n.properties).unwrap_or(serde_json::Value::Null);
                    (n.type_name().to_string(), props)
                } else {
                    ("unknown".to_string(), serde_json::Value::Null)
                };

                items.push(SearchResultItem {
                    node_id,
                    score,
                    node_type,
                    properties,
                });
            }
            items
        },
        SearchMode::Hybrid => {
            let (bm25_results, semantic_result) = tokio::join!(
                state.engine.search_bm25(&request.query, request.limit * 2),
                state.engine.search_claims_semantic(&request.query, request.limit * 2, 0.5),
            );
            let semantic_results = semantic_result.unwrap_or_else(|e| {
                info!("Semantic search failed or unavailable: {}", e);
                vec![]
            });

            info!(
                "Hybrid search: {} BM25 results, {} semantic results",
                bm25_results.len(),
                semantic_results.len()
            );

            // Use fusion strategy if provided, otherwise use default (RRF)
            let fusion_strategy = request
                .fusion_strategy
                .unwrap_or_else(agent_db_graph::indexing::FusionStrategy::default);

            // Fuse results
            let fused = fusion_strategy.fuse(bm25_results, semantic_results);

            let mut items = Vec::new();
            for (node_id, score) in fused.into_iter().take(request.limit) {
                let node = state.engine.get_node(node_id).await;
                let (node_type, properties) = if let Some(n) = node {
                    let props =
                        serde_json::to_value(&n.properties).unwrap_or(serde_json::Value::Null);
                    (n.type_name().to_string(), props)
                } else {
                    ("unknown".to_string(), serde_json::Value::Null)
                };

                items.push(SearchResultItem {
                    node_id,
                    score,
                    node_type,
                    properties,
                });
            }
            items
        },
    };

    let total = results.len();
    let mode = format!("{:?}", request.mode);

    info!("Search completed: {} results found", total);

    Ok(Json(SearchResponse {
        results,
        mode,
        total,
    }))
}
