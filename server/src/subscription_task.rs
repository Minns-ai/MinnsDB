//! Background subscription processing task.
//!
//! Drains the broadcast channel on a configurable interval, processes all
//! subscriptions, and buffers results in the manager's `pending_updates`.
//! This decouples delta processing from poll/WebSocket request latency.

use agent_db_graph::subscription::manager::SubscriptionManager;
use agent_db_graph::GraphEngine;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Default processing interval in milliseconds.
const DEFAULT_INTERVAL_MS: u64 = 50;

/// Spawn the background subscription processing task.
/// Returns the `CancellationToken` and `JoinHandle` so the caller can
/// trigger a graceful stop and await completion.
pub fn spawn(
    engine: Arc<GraphEngine>,
    manager: Arc<Mutex<SubscriptionManager>>,
) -> (CancellationToken, tokio::task::JoinHandle<()>) {
    let cancel = CancellationToken::new();
    let interval_ms = std::env::var("SUBSCRIPTION_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_INTERVAL_MS);

    let token = cancel.clone();
    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
        // Don't burst-catch-up if processing takes longer than the interval.
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                _ = interval.tick() => {}
            }

            // Check if there are any subscriptions before acquiring locks.
            {
                let mgr = manager.lock().await;
                if mgr.subscription_count() == 0 {
                    continue;
                }
            }

            // Acquire read lock on graph, then manager lock, process deltas.
            let inference = engine.inference().read().await;
            let graph = inference.graph();
            let ontology = engine.ontology();

            let mut mgr = manager.lock().await;
            mgr.drain_and_process(graph, ontology);
            // Locks dropped here.
        }

        tracing::info!("Subscription task stopped");
    });

    (token, handle)
}
