// Event processing handlers

use crate::errors::ApiError;
use crate::models::{PaginationQuery, ProcessEventRequest, ProcessEventResponse};
use crate::state::AppState;
use agent_db_events::{core::EventType, Event};
use axum::{
    extract::{Query, State},
    Json,
};
use tracing::info;

fn event_type_name(event_type: &EventType) -> &'static str {
    match event_type {
        EventType::Action { .. } => "Action",
        EventType::Observation { .. } => "Observation",
        EventType::Cognitive { .. } => "Cognitive",
        EventType::Communication { .. } => "Communication",
        EventType::Learning { .. } => "Learning",
        EventType::Context { .. } => "Context",
    }
}

// POST /api/events - Process a new event
pub async fn process_event(
    State(state): State<AppState>,
    Json(payload): Json<ProcessEventRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    let start = std::time::Instant::now();
    let event_id = payload.event.id;
    info!(
        "Processing event: id={} agent_id={} session_id={} type={}",
        event_id,
        payload.event.agent_id,
        payload.event.session_id,
        event_type_name(&payload.event.event_type)
    );

    // Auto-compute fingerprint if not provided (fingerprint == 0)
    let mut event = payload.event;
    if event.context.fingerprint == 0 {
        info!("Event {} missing fingerprint, computing.", event.id);
        event.context.fingerprint = event.context.compute_fingerprint();
    }

    let result = state
        .engine
        .process_event_with_options(event, Some(payload.enable_semantic))
        .await
        .map_err(|e| {
            info!("Error processing event: {:?}", e);
            ApiError::Internal(e.to_string())
        })?;

    info!(
        "Processed event {}: nodes_created={} patterns_detected={} processing_time_ms={} total_handler_ms={}",
        event_id,
        result.nodes_created.len(),
        result.patterns_detected.len(),
        result.processing_time_ms,
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        nodes_created: result.nodes_created.len(),
        patterns_detected: result.patterns_detected.len(),
        processing_time_ms: result.processing_time_ms,
    }))
}

// GET /api/events - Get recent events
pub async fn get_events(
    State(state): State<AppState>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<Event>>, ApiError> {
    info!("Getting recent events");
    let events = state.engine.get_recent_events(pagination.limit).await;
    Ok(Json(events))
}

// GET /api/episodes - Get completed episodes
pub async fn get_episodes(
    State(state): State<AppState>,
    Query(pagination): Query<PaginationQuery>,
) -> Result<Json<Vec<crate::models::EpisodeResponse>>, ApiError> {
    info!("Getting episodes");

    let episodes = state.engine.get_completed_episodes().await;

    let response: Vec<crate::models::EpisodeResponse> = episodes
        .into_iter()
        .take(pagination.limit)
        .map(|e| crate::models::EpisodeResponse {
            id: e.id,
            agent_id: e.agent_id,
            event_count: e.events.len(),
            significance: e.significance,
            outcome: e.outcome.map(|o| format!("{:?}", o)),
        })
        .collect();

    Ok(Json(response))
}
