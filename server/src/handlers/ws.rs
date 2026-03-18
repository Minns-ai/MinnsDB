//! WebSocket handler for real-time subscription updates.
//!
//! GET /api/subscriptions/ws → WebSocket upgrade
//!
//! Wire protocol (JSON text frames):
//!
//! Client → Server:
//!   {"type": "subscribe", "query": "MATCH ...", "request_id": "abc"}
//!   {"type": "unsubscribe", "subscription_id": 7}
//!   {"type": "ping"}
//!
//! Server → Client:
//!   {"type": "subscribed", "subscription_id": 7, "request_id": "abc", "initial": {...}}
//!   {"type": "update", "subscription_id": 7, "inserts": [...], "deletes": [...]}
//!   {"type": "unsubscribed", "subscription_id": 7}
//!   {"type": "error", "message": "...", "request_id": "abc"}
//!   {"type": "pong"}

use crate::state::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Subscribe {
        query: String,
        #[serde(default)]
        request_id: Option<String>,
    },
    Unsubscribe {
        subscription_id: u64,
    },
    Ping,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage {
    Subscribed {
        subscription_id: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        initial: WsResultSet,
    },
    Unsubscribed {
        subscription_id: u64,
    },
    Update {
        subscription_id: u64,
        inserts: Vec<Vec<serde_json::Value>>,
        deletes: Vec<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        count: Option<i64>,
    },
    Error {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
    },
    Pong,
}

#[derive(Serialize)]
struct WsResultSet {
    columns: Vec<String>,
    rows: Vec<Vec<serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// GET /api/subscriptions/ws — upgrade to WebSocket.
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    info!("WebSocket connection established");

    // Track this connection's subscriptions for cleanup on disconnect.
    let mut my_subs: HashSet<u64> = HashSet::new();

    // Heartbeat: send ping every 30s.
    let mut heartbeat = interval(Duration::from_secs(30));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Poll interval: check for pending updates every 100ms.
    let mut poll_interval = interval(Duration::from_millis(100));
    poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    'outer: loop {
        tokio::select! {
            // Incoming message from client.
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        let reply = handle_client_message(&text, &state, &mut my_subs).await;
                        if let Some(msg) = reply {
                            let json = serde_json::to_string(&msg).unwrap_or_default();
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                break 'outer;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break 'outer,
                    Some(Ok(Message::Ping(data))) => {
                        let _ = socket.send(Message::Pong(data)).await;
                    }
                    Some(Ok(_)) => {} // Ignore binary, pong, etc.
                    Some(Err(e)) => {
                        warn!("WebSocket error: {}", e);
                        break 'outer;
                    }
                }
            }
            // Push pending updates to client.
            _ = poll_interval.tick() => {
                if my_subs.is_empty() {
                    continue;
                }
                let updates = {
                    let mut mgr = state.subscription_manager.lock().await;
                    let mut all = Vec::new();
                    for &sub_id in &my_subs {
                        all.extend(mgr.take_pending(sub_id));
                    }
                    all
                };
                for u in updates {
                    let msg = ServerMessage::Update {
                        subscription_id: u.subscription_id,
                        inserts: u
                            .inserts
                            .into_iter()
                            .map(|(_, vals)| vals.into_iter().map(|v| v.to_json()).collect())
                            .collect(),
                        deletes: u
                            .deletes
                            .into_iter()
                            .map(|row_id| {
                                let slots: Vec<serde_json::Value> = row_id
                                    .slots()
                                    .iter()
                                    .map(|(slot, eid)| match eid {
                                        agent_db_graph::subscription::incremental::BoundEntityId::Node(nid) => {
                                            serde_json::json!({"slot": slot, "node_id": nid})
                                        }
                                        agent_db_graph::subscription::incremental::BoundEntityId::Edge(eid) => {
                                            serde_json::json!({"slot": slot, "edge_id": eid})
                                        }
                                    })
                                    .collect();
                                serde_json::json!(slots)
                            })
                            .collect(),
                        count: u.count,
                    };
                    let json = serde_json::to_string(&msg).unwrap_or_default();
                    if socket.send(Message::Text(json.into())).await.is_err() {
                        break 'outer;
                    }
                }
            }
            // Heartbeat ping.
            _ = heartbeat.tick() => {
                if socket.send(Message::Ping(vec![].into())).await.is_err() {
                    break 'outer;
                }
            }
        }
    }

    // Cleanup: unsubscribe all this connection's subscriptions.
    if !my_subs.is_empty() {
        let mut mgr = state.subscription_manager.lock().await;
        for sub_id in &my_subs {
            mgr.unsubscribe(*sub_id);
        }
        info!(
            "WebSocket disconnected, cleaned up {} subscriptions",
            my_subs.len()
        );
    } else {
        debug!("WebSocket disconnected (no active subscriptions)");
    }
}

async fn handle_client_message(
    text: &str,
    state: &AppState,
    my_subs: &mut HashSet<u64>,
) -> Option<ServerMessage> {
    let msg: ClientMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            return Some(ServerMessage::Error {
                message: format!("Invalid message: {}", e),
                request_id: None,
            });
        },
    };

    match msg {
        ClientMessage::Subscribe { query, request_id } => {
            if query.len() > 4096 {
                return Some(ServerMessage::Error {
                    message: "Query too long (max 4096 bytes)".into(),
                    request_id,
                });
            }

            // Parse and plan.
            let ast = match agent_db_graph::query_lang::parser::Parser::parse(&query) {
                Ok(a) => a,
                Err(e) => {
                    return Some(ServerMessage::Error {
                        message: format!("{}", e),
                        request_id,
                    });
                },
            };
            let plan = match agent_db_graph::query_lang::planner::plan(ast) {
                Ok(p) => p,
                Err(e) => {
                    return Some(ServerMessage::Error {
                        message: format!("{}", e),
                        request_id,
                    });
                },
            };

            // Subscribe under locks.
            let result = {
                let inference = state.engine.inference().read().await;
                let graph = inference.graph();
                let ontology = state.engine.ontology();
                let mut mgr = state.subscription_manager.lock().await;
                mgr.subscribe(plan, graph, &ontology)
            };

            match result {
                Ok((sub_id, output)) => {
                    my_subs.insert(sub_id);
                    // Store query text.
                    state
                        .subscription_queries
                        .lock()
                        .await
                        .insert(sub_id, query);

                    Some(ServerMessage::Subscribed {
                        subscription_id: sub_id,
                        request_id,
                        initial: WsResultSet {
                            columns: output.columns,
                            rows: output
                                .rows
                                .into_iter()
                                .map(|row| row.into_iter().map(|v| v.to_json()).collect())
                                .collect(),
                        },
                    })
                },
                Err(e) => Some(ServerMessage::Error {
                    message: format!("{}", e),
                    request_id,
                }),
            }
        },
        ClientMessage::Unsubscribe { subscription_id } => {
            let removed = {
                let mut mgr = state.subscription_manager.lock().await;
                mgr.unsubscribe(subscription_id)
            };
            if removed {
                my_subs.remove(&subscription_id);
                state
                    .subscription_queries
                    .lock()
                    .await
                    .remove(&subscription_id);
                Some(ServerMessage::Unsubscribed { subscription_id })
            } else {
                Some(ServerMessage::Error {
                    message: format!("Subscription {} not found", subscription_id),
                    request_id: None,
                })
            }
        },
        ClientMessage::Ping => Some(ServerMessage::Pong),
    }
}
