// Event processing handlers

use crate::errors::ApiError;
use crate::models::{
    PaginationQuery, ProcessEventRequest, ProcessEventResponse, SimpleEventRequest,
    StateChangeEventRequest, TransactionEventRequest,
};
use crate::state::AppState;
use agent_db_events::{
    core::{ActionOutcome, EventType, MetadataValue},
    Event,
};
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
        EventType::Conversation { .. } => "Conversation",
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
        event_id,
        nodes_created: result.nodes_created.len(),
        patterns_detected: result.patterns_detected.len(),
        processing_time_ms: result.processing_time_ms,
    }))
}

// POST /api/events/simple - Simplified event submission for easy integration
pub async fn process_simple_event(
    State(state): State<AppState>,
    Json(payload): Json<SimpleEventRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    info!(
        "Processing simple event: agent_id={} action={} session_id={}",
        payload.agent_id, payload.action, payload.session_id
    );

    // Convert simple request to full Event with defaults
    let event = Event {
        id: Default::default(),        // Will be auto-generated
        timestamp: Default::default(), // Will be auto-generated
        agent_id: payload.agent_id,
        agent_type: payload.agent_type,
        session_id: payload.session_id,
        event_type: EventType::Action {
            action_name: payload.action.clone(),
            parameters: payload.data,
            outcome: if payload.success.unwrap_or(true) {
                ActionOutcome::Success {
                    result: serde_json::json!({"status": "completed"}),
                }
            } else {
                ActionOutcome::Failure {
                    error: "Action failed".to_string(),
                    error_code: 1,
                }
            },
            duration_ns: 0, // Unknown for simple events
        },
        causality_chain: Vec::new(),
        context: Default::default(), // Use minimal defaults
        metadata: Default::default(),
        context_size_bytes: 0,
        segment_pointer: None,
        is_code: false,
    };

    // Process the event through the standard pipeline
    let start = std::time::Instant::now();
    let event_id = event.id;

    let result = state
        .engine
        .process_event_with_options(event, Some(payload.enable_semantic))
        .await
        .map_err(|e| {
            info!("Error processing simple event: {:?}", e);
            ApiError::Internal(e.to_string())
        })?;

    info!(
        "Processed simple event {}: nodes_created={} patterns_detected={} processing_time_ms={} total_handler_ms={}",
        event_id,
        result.nodes_created.len(),
        result.patterns_detected.len(),
        result.processing_time_ms,
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        event_id,
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

// Canonical metadata keys that extra_metadata cannot override
const CANONICAL_STATE_KEYS: &[&str] = &["entity", "new_state", "old_state"];
const CANONICAL_TXN_KEYS: &[&str] = &[
    "from",
    "to",
    "amount",
    "direction",
    "description",
    "transaction",
];

// POST /api/events/state-change — typed state-change event
pub async fn process_state_change_event(
    State(state): State<AppState>,
    Json(payload): Json<StateChangeEventRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    info!(
        "Processing state-change event: agent_id={} entity={} new_state={}",
        payload.agent_id, payload.entity, payload.new_state
    );

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("entity".to_string(), MetadataValue::String(payload.entity));
    metadata.insert(
        "new_state".to_string(),
        MetadataValue::String(payload.new_state),
    );
    if let Some(old_state) = payload.old_state {
        metadata.insert("old_state".to_string(), MetadataValue::String(old_state));
    }
    if let Some(trigger) = payload.trigger {
        metadata.insert("trigger".to_string(), MetadataValue::String(trigger));
    }

    // Merge extra_metadata, silently ignoring canonical key collisions
    for (k, v) in payload.extra_metadata {
        if CANONICAL_STATE_KEYS.contains(&k.as_str()) {
            tracing::warn!(
                "extra_metadata key '{}' collides with canonical field — ignored",
                k
            );
            continue;
        }
        metadata.insert(k, MetadataValue::Json(v));
    }

    let event = Event {
        id: Default::default(),
        timestamp: Default::default(),
        agent_id: payload.agent_id,
        agent_type: payload.agent_type,
        session_id: payload.session_id,
        event_type: EventType::Context {
            text: String::new(),
            context_type: "state_update".to_string(),
            language: None,
        },
        causality_chain: Vec::new(),
        context: Default::default(),
        metadata,
        context_size_bytes: 0,
        segment_pointer: None,
        is_code: false,
    };

    let start = std::time::Instant::now();
    let event_id = event.id;

    let result = state
        .engine
        .process_event_with_options(event, Some(payload.enable_semantic))
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    info!(
        "Processed state-change event {}: nodes_created={} total_handler_ms={}",
        event_id,
        result.nodes_created.len(),
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        event_id,
        nodes_created: result.nodes_created.len(),
        patterns_detected: result.patterns_detected.len(),
        processing_time_ms: result.processing_time_ms,
    }))
}

// POST /api/events/transaction — typed transaction event
pub async fn process_transaction_event(
    State(state): State<AppState>,
    Json(payload): Json<TransactionEventRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    info!(
        "Processing transaction event: agent_id={} from={} to={} amount={}",
        payload.agent_id, payload.from, payload.to, payload.amount
    );

    // Validate amount is finite
    if !payload.amount.is_finite() {
        return Err(ApiError::BadRequest(
            "amount must be a finite number (not NaN or Infinity)".to_string(),
        ));
    }

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("from".to_string(), MetadataValue::String(payload.from));
    metadata.insert("to".to_string(), MetadataValue::String(payload.to));
    metadata.insert("amount".to_string(), MetadataValue::Float(payload.amount));
    metadata.insert("transaction".to_string(), MetadataValue::Boolean(true));

    if let Some(direction) = payload.direction {
        metadata.insert("direction".to_string(), MetadataValue::String(direction));
    }
    if let Some(description) = payload.description {
        metadata.insert(
            "description".to_string(),
            MetadataValue::String(description),
        );
    }

    // Merge extra_metadata, silently ignoring canonical key collisions
    for (k, v) in payload.extra_metadata {
        if CANONICAL_TXN_KEYS.contains(&k.as_str()) {
            tracing::warn!(
                "extra_metadata key '{}' collides with canonical field — ignored",
                k
            );
            continue;
        }
        metadata.insert(k, MetadataValue::Json(v));
    }

    let event = Event {
        id: Default::default(),
        timestamp: Default::default(),
        agent_id: payload.agent_id,
        agent_type: payload.agent_type,
        session_id: payload.session_id,
        event_type: EventType::Action {
            action_name: "transaction".to_string(),
            parameters: serde_json::json!({}),
            outcome: ActionOutcome::Success {
                result: serde_json::json!({"status": "completed"}),
            },
            duration_ns: 0,
        },
        causality_chain: Vec::new(),
        context: Default::default(),
        metadata,
        context_size_bytes: 0,
        segment_pointer: None,
        is_code: false,
    };

    let start = std::time::Instant::now();
    let event_id = event.id;

    let result = state
        .engine
        .process_event_with_options(event, Some(payload.enable_semantic))
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    info!(
        "Processed transaction event {}: nodes_created={} total_handler_ms={}",
        event_id,
        result.nodes_created.len(),
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        event_id,
        nodes_created: result.nodes_created.len(),
        patterns_detected: result.patterns_detected.len(),
        processing_time_ms: result.processing_time_ms,
    }))
}
