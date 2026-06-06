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

    // Group scoping: when a non-empty group_id is supplied, restrict results
    // to that group and over-fetch from the (shared) indexes so the
    // post-filter can still return a full page. Empty/omitted spans all
    // groups, matching NLQ semantics.
    let group = request.group_id.as_deref().map(str::trim).unwrap_or("");
    let fetch = if group.is_empty() {
        request.limit
    } else {
        request.limit.saturating_mul(8).min(500).max(request.limit)
    };

    let raw: Vec<(u64, f32)> = match request.mode {
        SearchMode::Keyword => state.engine.search_bm25(&request.query, fetch).await,
        SearchMode::Semantic => {
            let semantic_results = state
                .engine
                .search_claims_semantic(&request.query, fetch, 0.5)
                .await
                .unwrap_or_else(|e| {
                    info!("Semantic search failed, falling back to BM25: {}", e);
                    vec![]
                });
            if semantic_results.is_empty() {
                info!("No semantic results, falling back to BM25");
                state.engine.search_bm25(&request.query, fetch).await
            } else {
                semantic_results
            }
        },
        SearchMode::Hybrid => {
            let (bm25_results, semantic_result) = tokio::join!(
                state.engine.search_bm25(&request.query, fetch * 2),
                state
                    .engine
                    .search_claims_semantic(&request.query, fetch * 2, 0.5),
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

            fusion_strategy.fuse(bm25_results, semantic_results)
        },
    };

    // Build the response, enforcing the group scope. A non-empty group keeps
    // only nodes in that group and drops hits that map to no node, so a
    // scoped search can never surface another group's data even though the
    // BM25 / semantic indexes are shared across groups.
    let mut results = Vec::new();
    for (node_id, score) in raw {
        let node = state.engine.get_node(node_id).await;
        if !group.is_empty() && node.as_ref().map(|n| n.group_id.as_str()) != Some(group) {
            continue;
        }
        let (node_type, properties) = if let Some(n) = node {
            let props = serde_json::to_value(&n.properties).unwrap_or(serde_json::Value::Null);
            (n.type_name().to_string(), props)
        } else {
            ("unknown".to_string(), serde_json::Value::Null)
        };
        results.push(SearchResultItem {
            node_id,
            score,
            node_type,
            properties,
        });
        if results.len() >= request.limit {
            break;
        }
    }

    let total = results.len();
    let mode = format!("{:?}", request.mode);

    info!("Search completed: {} results found", total);

    Ok(Json(SearchResponse {
        results,
        mode,
        total,
    }))
}
