//! Job status + subscribe handlers.
//!
//! Two endpoints back the async ingest path:
//!
//! - `GET  /api/jobs/{id}` — point-in-time JSON status. Cheap. Polled by
//!   demo/integration clients that prefer HTTP over WebSocket.
//!
//! - `WS   /api/jobs/{id}/subscribe` — live event stream. The job's
//!   broadcast channel is forwarded to the socket so production agents
//!   see state transitions in real time without polling overhead.
//!
//! Both endpoints read from the same `JobStore` and surface the same
//! `JobState` enum, so the two transports always agree on the truth.

use crate::errors::ApiError;
use crate::jobs::JobState;
use crate::state::AppState;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use tracing::{debug, warn};

/// GET /api/jobs/{id} — point-in-time job status.
///
/// Returns the full `JobState` as JSON so the polling client can see the
/// final response body (when `Done`) without a separate fetch.
pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job_state = state
        .jobs
        .get(&id)
        .map_err(|_| ApiError::NotFound(format!("job {} not found", id)))?;
    Ok(Json(json!({ "job_id": id, "state": job_state })).into_response())
}

/// GET /api/jobs/{id}/subscribe — WebSocket upgrade for live job events.
///
/// On connect, sends the current state immediately so the client doesn't
/// race the broadcast channel. Then forwards every transition. Closes
/// the socket when the job reaches a terminal state — clients that need
/// to keep monitoring after Done/Failed can poll the GET endpoint
/// instead.
pub async fn subscribe_job(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    ws.on_upgrade(move |socket| handle_subscribe_socket(socket, state, id))
}

async fn handle_subscribe_socket(mut socket: WebSocket, state: AppState, id: String) {
    // Send the current state as the very first frame so subscribers can
    // proceed even when they connect after a transition has already
    // fired (broadcast channels don't replay history past their
    // capacity).
    let initial = match state.jobs.get(&id) {
        Ok(s) => s,
        Err(_) => {
            let _ = socket
                .send(Message::Text(
                    json!({"error": "not_found", "job_id": id}).to_string(),
                ))
                .await;
            return;
        },
    };
    if !send_state(&mut socket, &id, &initial).await {
        return;
    }
    if initial.is_terminal() {
        return;
    }

    let mut rx = match state.jobs.subscribe(&id) {
        Ok(rx) => rx,
        Err(_) => {
            // Raced with eviction between get + subscribe — surface
            // a close frame and bail.
            let _ = socket
                .send(Message::Text(
                    json!({"error": "vanished", "job_id": id}).to_string(),
                ))
                .await;
            return;
        },
    };

    loop {
        tokio::select! {
            // Job state change → forward.
            recv = rx.recv() => {
                match recv {
                    Ok(new_state) => {
                        let terminal = new_state.is_terminal();
                        if !send_state(&mut socket, &id, &new_state).await {
                            break;
                        }
                        if terminal {
                            // Client has the final result — close cleanly.
                            let _ = socket.send(Message::Close(None)).await;
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        // Slow consumer dropped frames. Resync via the
                        // store rather than tearing down the socket.
                        warn!(job_id = %id, skipped, "job subscriber lagged; resyncing");
                        if let Ok(state) = state.jobs.get(&id) {
                            let terminal = state.is_terminal();
                            if !send_state(&mut socket, &id, &state).await {
                                break;
                            }
                            if terminal {
                                let _ = socket.send(Message::Close(None)).await;
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            // Client pinged / sent control frame → handle.
            recv = socket.recv() => {
                match recv {
                    Some(Ok(Message::Ping(p))) => {
                        let _ = socket.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => {
                        debug!(job_id = %id, err = %e, "subscribe socket recv error");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Helper: serialise a JobState to JSON and push it onto the socket.
/// Returns `false` when the socket has gone away so the caller breaks
/// the loop without panicking.
async fn send_state(socket: &mut WebSocket, id: &str, s: &JobState) -> bool {
    let frame = json!({ "job_id": id, "state": s }).to_string();
    socket.send(Message::Text(frame)).await.is_ok()
}
