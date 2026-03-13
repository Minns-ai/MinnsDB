// Health check and informational endpoints

use crate::errors::ApiError;
use crate::models::{
    HealthResponse, LaneHealth, ReadGateHealth, SequenceTrackerHealth, WriteLanesHealth,
};
use crate::state::AppState;
use axum::{extract::State, Json};
use std::sync::atomic::Ordering::Relaxed;

// GET /api/health - Health check endpoint
pub async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let health = state.engine.get_health_metrics().await;

    // ── Write lane metrics ──────────────────────────────────────────────
    let wm = &state.write_lanes.metrics;
    let num_lanes = state.write_lanes.num_lanes();
    let mut lanes = Vec::with_capacity(num_lanes);
    for i in 0..num_lanes {
        lanes.push(LaneHealth {
            lane_id: i,
            depth: wm.lane_depth(i),
            in_flight: wm.per_lane_in_flight[i].load(Relaxed),
            completed: wm.per_lane_completed[i].load(Relaxed),
            rejected: wm.per_lane_rejected[i].load(Relaxed),
        });
    }
    let mut write_latency = wm.write_latency.lock();
    let write_lanes = WriteLanesHealth {
        num_lanes,
        lanes,
        total_submitted: wm.total_submitted.load(Relaxed),
        total_completed: wm.total_completed.load(Relaxed),
        total_rejected: wm.total_rejected.load(Relaxed),
        write_p50_ms: write_latency.percentile(0.50) as f64 / 1000.0,
        write_p95_ms: write_latency.percentile(0.95) as f64 / 1000.0,
        write_p99_ms: write_latency.percentile(0.99) as f64 / 1000.0,
    };
    drop(write_latency);

    // ── Read gate metrics ───────────────────────────────────────────────
    let rm = &state.read_gate.metrics;
    let mut read_latency = rm.read_latency.lock();
    let read_gate = ReadGateHealth {
        permits_total: state.read_gate.permits_total,
        in_flight: rm.in_flight.load(Relaxed),
        completed: rm.completed.load(Relaxed),
        rejected: rm.rejected.load(Relaxed),
        read_p50_ms: read_latency.percentile(0.50) as f64 / 1000.0,
        read_p95_ms: read_latency.percentile(0.95) as f64 / 1000.0,
        read_p99_ms: read_latency.percentile(0.99) as f64 / 1000.0,
    };
    drop(read_latency);

    // ── Sequence tracker metrics ────────────────────────────────────────
    let sequence_tracker = SequenceTrackerHealth {
        tracked_domains: state.seq_tracker.domain_count(),
    };

    Ok(Json(HealthResponse {
        status: if health.is_healthy {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        },
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.started_at.elapsed().as_secs(),
        is_healthy: health.is_healthy,
        node_count: health.node_count,
        edge_count: health.edge_count,
        processing_rate: health.processing_rate,
        write_lanes,
        read_gate,
        sequence_tracker,
    }))
}

// GET / - Root endpoint
pub async fn root() -> &'static str {
    "MinnsDB REST API Server v0.2.0\n\n\
     Core Endpoints:\n\
     POST /api/events - Process event\n\
     GET /api/memories/agent/:id - Get agent memories\n\
     POST /api/memories/context - Get memories by context\n\
     GET /api/strategies/agent/:id - Get agent strategies\n\
     POST /api/strategies/similar - Find similar strategies\n\
     GET /api/suggestions - Get action suggestions\n\
     GET /api/episodes - Get episodes\n\
     GET /api/stats - Get statistics\n\
     GET /api/graph - Get graph visualization data\n\
     GET /api/graph/context - Get context graph visualization data\n\
     GET /api/health - Health check\n\n\
     Advanced Graph Features:\n\
     GET /api/analytics - Graph analytics with learning metrics\n\
     GET /api/indexes - Property index statistics\n\
     GET /api/communities - Community detection (Louvain)\n\
     GET /api/centrality - Node centrality scores\n\n\
     Natural Language & Conversation:\n\
     POST /api/nlq - Natural language graph query\n\
     POST /api/conversations/ingest - Ingest conversation sessions (batch)\n\
     POST /api/messages - Accept a single conversation message\n\n\
     GET /docs - API documentation"
}

// GET /docs - API documentation
pub async fn docs() -> &'static str {
    "MinnsDB API Documentation\n\n\
     See API_REFERENCE.md for complete documentation."
}
