// Code intelligence handlers

use crate::errors::ApiError;
use crate::models::{
    CodeEntityResult, CodeFileRequest, CodeReviewRequest, CodeSearchRequest, CodeSearchResponse,
    ProcessEventResponse,
};
use crate::state::AppState;
use crate::write_lanes::WriteJob;
use agent_db_events::core::{CodeReviewAction, EventType};
use agent_db_events::Event;
use axum::{extract::State, Json};
use tracing::info;

// POST /api/events/code-review — Submit a code review event
pub async fn process_code_review(
    State(state): State<AppState>,
    Json(payload): Json<CodeReviewRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    info!(
        "Processing code review event: review_id={} repository={} action={}",
        payload.review_id, payload.repository, payload.action
    );

    let action = match payload.action.as_str() {
        "approve" => CodeReviewAction::Approve,
        "request_changes" => CodeReviewAction::RequestChanges,
        _ => CodeReviewAction::Comment,
    };

    let event = Event {
        id: Default::default(),
        timestamp: Default::default(),
        agent_id: payload.agent_id,
        agent_type: payload.agent_type,
        session_id: payload.session_id,
        event_type: EventType::CodeReview {
            review_id: payload.review_id,
            action,
            body: payload.body,
            file_path: payload.file_path,
            line_range: payload.line_range,
            repository: payload.repository,
            title: payload.title,
        },
        causality_chain: Vec::new(),
        context: Default::default(),
        metadata: Default::default(),
        context_size_bytes: 0,
        segment_pointer: None,
        is_code: true,
    };

    let start = std::time::Instant::now();
    let event_id = event.id;
    let session_id = event.session_id;

    let result = state
        .write_lanes
        .submit_and_await(session_id, |tx| WriteJob::ProcessEvent {
            event: Box::new(event),
            enable_semantic: Some(payload.enable_semantic),
            result_tx: tx,
        })
        .await?;

    let nodes_created = result["nodes_created"].as_u64().unwrap_or(0) as usize;
    let patterns_detected = result["patterns_detected"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or_else(|| result["patterns_detected"].as_u64().unwrap_or(0) as usize);
    let processing_time_ms = result["processing_time_ms"].as_u64().unwrap_or(0);

    info!(
        "Processed code review {}: nodes_created={} total_handler_ms={}",
        event_id,
        nodes_created,
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        event_id,
        nodes_created,
        patterns_detected,
        processing_time_ms,
    }))
}

// POST /api/events/code-file — Submit a code file snapshot
pub async fn process_code_file(
    State(state): State<AppState>,
    Json(payload): Json<CodeFileRequest>,
) -> Result<Json<ProcessEventResponse>, ApiError> {
    info!(
        "Processing code file event: file_path={} language={:?}",
        payload.file_path, payload.language
    );

    let event = Event {
        id: Default::default(),
        timestamp: Default::default(),
        agent_id: payload.agent_id,
        agent_type: payload.agent_type,
        session_id: payload.session_id,
        event_type: EventType::CodeFile {
            file_path: payload.file_path,
            content: payload.content,
            language: payload.language,
            repository: payload.repository,
            git_ref: payload.git_ref,
        },
        causality_chain: Vec::new(),
        context: Default::default(),
        metadata: Default::default(),
        context_size_bytes: 0,
        segment_pointer: None,
        is_code: true,
    };

    let start = std::time::Instant::now();
    let event_id = event.id;
    let session_id = event.session_id;

    let result = state
        .write_lanes
        .submit_and_await(session_id, |tx| WriteJob::ProcessEvent {
            event: Box::new(event),
            enable_semantic: Some(payload.enable_semantic),
            result_tx: tx,
        })
        .await?;

    let nodes_created = result["nodes_created"].as_u64().unwrap_or(0) as usize;
    let patterns_detected = result["patterns_detected"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or_else(|| result["patterns_detected"].as_u64().unwrap_or(0) as usize);
    let processing_time_ms = result["processing_time_ms"].as_u64().unwrap_or(0);

    info!(
        "Processed code file {}: nodes_created={} total_handler_ms={}",
        event_id,
        nodes_created,
        start.elapsed().as_millis()
    );

    Ok(Json(ProcessEventResponse {
        success: true,
        event_id,
        nodes_created,
        patterns_detected,
        processing_time_ms,
    }))
}

// POST /api/code/search — Structural code search
pub async fn search_code(
    State(state): State<AppState>,
    Json(payload): Json<CodeSearchRequest>,
) -> Result<Json<CodeSearchResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!(
        "Code search: name_pattern={:?} kind={:?} language={:?} file_pattern={:?}",
        payload.name_pattern, payload.kind, payload.language, payload.file_pattern
    );

    let matches = state
        .engine
        .search_code_entities(
            payload.name_pattern.as_deref(),
            payload.kind.as_deref(),
            payload.language.as_deref(),
            payload.file_pattern.as_deref(),
            payload.limit,
        )
        .await;

    let total = matches.len();
    let entities = matches
        .into_iter()
        .map(|m| CodeEntityResult {
            name: m.name,
            qualified_name: m.qualified_name,
            kind: m.kind,
            file_path: m.file_path,
            language: m.language,
            line_range: m.line_range,
            signature: m.signature,
            doc_comment: m.doc_comment,
            visibility: m.visibility,
        })
        .collect();
    Ok(Json(CodeSearchResponse {
        entities,
        total_matches: total,
    }))
}
