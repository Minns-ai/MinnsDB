// Prometheus metrics endpoint.
//
// Serves the text-format scrape body. The recorder is installed once at
// process start in `main.rs`; this handler just renders the current
// snapshot. Health-level p50/p95/p99 latencies remain on `/api/health`
// (human-readable); this endpoint is for time-series scraping.

use axum::{extract::State, http::header, response::IntoResponse};

use crate::state::AppState;

pub async fn metrics_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    let body = state.metrics_handle.render();
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}
