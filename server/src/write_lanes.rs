//! Write Lanes — striped, bounded admission for write operations.
//!
//! Events within the same session route to the same lane (hash(session_id) % N),
//! preserving FIFO ordering per session. Backpressure: if a lane's bounded
//! channel is full, the caller receives a 503 (Service Unavailable).

use agent_db_graph::GraphEngine;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

// ─── WriteJob ────────────────────────────────────────────────────────────────

/// Boxed async closure that takes an `Arc<GraphEngine>` and returns a result.
pub type GenericWriteOp = Box<
    dyn FnOnce(
            Arc<GraphEngine>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<serde_json::Value, String>> + Send>,
        > + Send,
>;

/// A unit of work submitted to a write lane.
pub enum WriteJob {
    /// Process a single event through the graph engine pipeline.
    ProcessEvent {
        event: Box<agent_db_events::Event>,
        enable_semantic: Option<bool>,
        result_tx: oneshot::Sender<Result<serde_json::Value, String>>,
    },
    /// Persist graph state to storage.
    PersistGraph {
        result_tx: oneshot::Sender<Result<serde_json::Value, String>>,
    },
    /// Generic write operation carried as a boxed async closure.
    /// Used for structured memory ops, planning writes, import, etc.
    GenericWrite {
        operation: GenericWriteOp,
        result_tx: oneshot::Sender<Result<serde_json::Value, String>>,
    },
    /// Hard-delete a node and its incident edges. Zero-alloc dispatch.
    DeleteNode {
        node_id: u64,
        result_tx: oneshot::Sender<Result<serde_json::Value, String>>,
    },
    /// Hard-delete a single edge. Zero-alloc dispatch.
    DeleteEdge {
        edge_id: u64,
        result_tx: oneshot::Sender<Result<serde_json::Value, String>>,
    },
}

// ─── WriteError ─────────────────────────────────────────────────────────────

/// Errors returned by `submit()` and `submit_and_await()`.
pub enum WriteError {
    /// Lane is full or shut down — maps to 503.
    LaneUnavailable(String),
    /// The worker dropped the oneshot before sending a result.
    WorkerDropped,
    /// The operation itself returned an Err(String).
    OperationFailed(String),
}

// ─── LatencyTracker ──────────────────────────────────────────────────────────

/// Ring buffer of latency samples (microseconds) for percentile computation.
pub struct LatencyTracker {
    samples: Vec<u64>,
    head: usize,
    count: usize,
    capacity: usize,
    /// Cached sorted copy — invalidated on `record()`, built lazily in `percentile()`.
    sorted_cache: Option<Vec<u64>>,
}

impl LatencyTracker {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1); // Prevent divide-by-zero in record()
        Self {
            samples: vec![0; capacity],
            head: 0,
            count: 0,
            capacity,
            sorted_cache: None,
        }
    }

    pub fn record(&mut self, micros: u64) {
        self.samples[self.head] = micros;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        self.sorted_cache = None; // invalidate
    }

    /// Compute percentile (0.0–1.0). Returns 0 if no samples.
    /// Repeated calls (e.g. p50, p95, p99) reuse a cached sorted buffer.
    pub fn percentile(&mut self, p: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let sorted = self.sorted_cache.get_or_insert_with(|| {
            let mut v: Vec<u64> = if self.count < self.capacity {
                self.samples[..self.count].to_vec()
            } else {
                self.samples.clone()
            };
            v.sort_unstable();
            v
        });
        let idx = ((p * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

// ─── WriteLaneMetrics ────────────────────────────────────────────────────────

/// Aggregate metrics for all write lanes.
pub struct WriteLaneMetrics {
    pub per_lane_in_flight: Vec<AtomicU64>,
    pub per_lane_completed: Vec<AtomicU64>,
    pub per_lane_rejected: Vec<AtomicU64>,
    pub total_submitted: AtomicU64,
    pub total_completed: AtomicU64,
    pub total_rejected: AtomicU64,
    pub write_latency: parking_lot::Mutex<LatencyTracker>,
}

impl WriteLaneMetrics {
    fn new(num_lanes: usize) -> Self {
        let mut in_flight = Vec::with_capacity(num_lanes);
        let mut completed = Vec::with_capacity(num_lanes);
        let mut rejected = Vec::with_capacity(num_lanes);
        for _ in 0..num_lanes {
            in_flight.push(AtomicU64::new(0));
            completed.push(AtomicU64::new(0));
            rejected.push(AtomicU64::new(0));
        }
        Self {
            per_lane_in_flight: in_flight,
            per_lane_completed: completed,
            per_lane_rejected: rejected,
            total_submitted: AtomicU64::new(0),
            total_completed: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            write_latency: parking_lot::Mutex::new(LatencyTracker::new(1000)),
        }
    }

    /// Current in-flight job count for this lane.
    pub fn lane_depth(&self, lane: usize) -> u64 {
        self.per_lane_in_flight[lane].load(Ordering::Acquire)
    }
}

// ─── WriteLanes ──────────────────────────────────────────────────────────────

/// Striped write lane pool. Routes write operations by session_id hash.
pub struct WriteLanes {
    /// Lane senders. Wrapped in a Mutex so `drain()` can drop them, which
    /// closes the channels and lets workers see channel closure promptly.
    lane_txs: parking_lot::Mutex<Vec<mpsc::Sender<WriteJob>>>,
    num_lanes: usize,
    pub metrics: Arc<WriteLaneMetrics>,
    cancel: CancellationToken,
    worker_handles: parking_lot::Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl WriteLanes {
    /// Create a new write lane pool and spawn worker tasks.
    ///
    /// `num_lanes`: number of lanes (default: num_cpus/2, min 2, max 8).
    /// `capacity`: bounded channel capacity per lane (default: 128).
    pub fn new(engine: Arc<GraphEngine>, num_lanes: usize, capacity: usize) -> Self {
        let num_lanes = num_lanes.clamp(2, 8);
        let metrics = Arc::new(WriteLaneMetrics::new(num_lanes));
        let cancel = CancellationToken::new();
        let mut lane_txs = Vec::with_capacity(num_lanes);
        let mut handles = Vec::with_capacity(num_lanes);

        for lane_id in 0..num_lanes {
            let (tx, rx) = mpsc::channel(capacity);
            lane_txs.push(tx);

            let engine_clone = engine.clone();
            let metrics_clone = metrics.clone();
            let cancel_clone = cancel.clone();
            let handle = tokio::spawn(lane_worker(
                engine_clone,
                rx,
                lane_id,
                metrics_clone,
                cancel_clone,
            ));
            handles.push(handle);
        }

        Self {
            lane_txs: parking_lot::Mutex::new(lane_txs),
            num_lanes,
            metrics,
            cancel,
            worker_handles: parking_lot::Mutex::new(handles),
        }
    }

    /// Gracefully drain all write lanes: signal cancellation, drop senders
    /// so workers see channel closure, then await all worker tasks.
    pub async fn drain(&self) {
        self.cancel.cancel();
        // Drop all senders so workers unblock from recv() immediately.
        self.lane_txs.lock().clear();
        let handles: Vec<_> = self.worker_handles.lock().drain(..).collect();
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Submit a write job, routed by `routing_key` (typically session_id).
    ///
    /// Returns `Err` if the lane's channel is full after a 5-second timeout
    /// or if the lanes have been cancelled.
    pub async fn submit(&self, routing_key: u64, job: WriteJob) -> Result<(), String> {
        let lane_index = (routing_key as usize) % self.num_lanes;
        self.metrics.total_submitted.fetch_add(1, Ordering::Relaxed);

        // Atomic check — no lock
        if self.cancel.is_cancelled() {
            self.metrics.total_rejected.fetch_add(1, Ordering::Relaxed);
            return Err("Write lanes have been shut down".to_string());
        }

        // Clone the sender under the lock (cheap Arc bump), then release
        // the lock before the potentially blocking send.
        let sender = {
            let txs = self.lane_txs.lock();
            match txs.get(lane_index) {
                Some(tx) => tx.clone(),
                None => {
                    self.metrics.total_rejected.fetch_add(1, Ordering::Relaxed);
                    self.metrics.per_lane_rejected[lane_index].fetch_add(1, Ordering::Relaxed);
                    return Err(format!(
                        "Write lane {} is dead (senders dropped)",
                        lane_index
                    ));
                },
            }
        };

        match tokio::time::timeout(std::time::Duration::from_secs(5), sender.send(job)).await {
            Ok(Ok(())) => {
                self.metrics.per_lane_in_flight[lane_index].fetch_add(1, Ordering::Release);
                Ok(())
            },
            Ok(Err(_)) => {
                self.metrics.total_rejected.fetch_add(1, Ordering::Relaxed);
                self.metrics.per_lane_rejected[lane_index].fetch_add(1, Ordering::Relaxed);
                Err(format!("Write lane {} is dead (worker exited)", lane_index))
            },
            Err(_) => {
                self.metrics.total_rejected.fetch_add(1, Ordering::Relaxed);
                self.metrics.per_lane_rejected[lane_index].fetch_add(1, Ordering::Relaxed);
                Err(format!(
                    "Write lane {} is full (backpressure timeout after 5s)",
                    lane_index
                ))
            },
        }
    }

    /// Submit a write job and await its oneshot result.
    ///
    /// This is a convenience wrapper combining `submit()` + oneshot recv,
    /// eliminating boilerplate in handler functions.
    pub async fn submit_and_await(
        &self,
        routing_key: u64,
        job_fn: impl FnOnce(oneshot::Sender<Result<serde_json::Value, String>>) -> WriteJob,
    ) -> Result<serde_json::Value, WriteError> {
        let (tx, rx) = oneshot::channel();
        let job = job_fn(tx);
        self.submit(routing_key, job)
            .await
            .map_err(WriteError::LaneUnavailable)?;
        rx.await
            .map_err(|_| WriteError::WorkerDropped)?
            .map_err(WriteError::OperationFailed)
    }

    /// Number of lanes.
    pub fn num_lanes(&self) -> usize {
        self.num_lanes
    }
}

// ─── Lane Worker ─────────────────────────────────────────────────────────────

async fn lane_worker(
    engine: Arc<GraphEngine>,
    mut rx: mpsc::Receiver<WriteJob>,
    lane_id: usize,
    metrics: Arc<WriteLaneMetrics>,
    cancel: CancellationToken,
) {
    tracing::info!("Write lane {} worker started", lane_id);

    loop {
        let job = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                // Drain remaining buffered jobs before exiting
                while let Ok(job) = rx.try_recv() {
                    execute_job(&engine, job, lane_id, &metrics).await;
                }
                break;
            }
            job = rx.recv() => {
                match job {
                    Some(j) => j,
                    None => break, // channel closed
                }
            }
        };

        execute_job(&engine, job, lane_id, &metrics).await;
    }

    tracing::info!("Write lane {} worker exiting", lane_id);
}

async fn execute_job(
    engine: &Arc<GraphEngine>,
    job: WriteJob,
    lane_id: usize,
    metrics: &WriteLaneMetrics,
) {
    let start = std::time::Instant::now();

    match job {
        WriteJob::ProcessEvent {
            event,
            enable_semantic,
            result_tx,
        } => {
            let r = engine
                .process_event_with_options(*event, enable_semantic)
                .await;
            let response = match r {
                Ok(op) => Ok(serde_json::json!({
                    "success": true,
                    "nodes_created": op.nodes_created.len(),
                    "patterns_detected": op.patterns_detected,
                    "processing_time_ms": op.processing_time_ms,
                })),
                Err(e) => Err(e.to_string()),
            };
            let _ = result_tx.send(response);
        },
        WriteJob::PersistGraph { result_tx } => {
            let r = engine.full_persist().await;
            let response = match r {
                Ok((nodes, edges)) => Ok(serde_json::json!({
                    "success": true,
                    "nodes_persisted": nodes,
                    "edges_persisted": edges,
                })),
                Err(e) => Err(e.to_string()),
            };
            let _ = result_tx.send(response);
        },
        WriteJob::GenericWrite {
            operation,
            result_tx,
        } => {
            let result = operation(engine.clone()).await;
            let _ = result_tx.send(result);
        },
        WriteJob::DeleteNode { node_id, result_tx } => {
            let deleted = engine.delete_node(node_id).await;
            let _ = result_tx.send(Ok(serde_json::json!({
                "deleted": deleted,
                "node_id": node_id,
            })));
        },
        WriteJob::DeleteEdge { edge_id, result_tx } => {
            let deleted = engine.delete_edge(edge_id).await;
            let _ = result_tx.send(Ok(serde_json::json!({
                "deleted": deleted,
                "edge_id": edge_id,
            })));
        },
    }

    let elapsed_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
    metrics.per_lane_in_flight[lane_id].fetch_sub(1, Ordering::Release);
    metrics.per_lane_completed[lane_id].fetch_add(1, Ordering::Relaxed);
    metrics.total_completed.fetch_add(1, Ordering::Relaxed);
    metrics.write_latency.lock().record(elapsed_us);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_tracker_empty() {
        let mut t = LatencyTracker::new(100);
        assert_eq!(t.percentile(0.5), 0);
        assert_eq!(t.percentile(0.99), 0);
    }

    #[test]
    fn latency_tracker_basic() {
        let mut t = LatencyTracker::new(100);
        for i in 1..=100 {
            t.record(i);
        }
        // p50 should be ~50, p99 should be ~99
        let p50 = t.percentile(0.5);
        let p99 = t.percentile(0.99);
        assert!((45..=55).contains(&p50), "p50 was {}", p50);
        assert!((95..=100).contains(&p99), "p99 was {}", p99);
    }

    #[test]
    fn latency_tracker_cache_invalidation() {
        let mut t = LatencyTracker::new(100);
        t.record(10);
        t.record(20);
        let _ = t.percentile(0.5);
        assert!(t.sorted_cache.is_some());
        t.record(30);
        assert!(t.sorted_cache.is_none());
        // Next percentile call rebuilds cache
        let _ = t.percentile(0.5);
        assert!(t.sorted_cache.is_some());
    }

    #[test]
    fn latency_tracker_zero_capacity() {
        // capacity clamped to 1
        let mut t = LatencyTracker::new(0);
        assert_eq!(t.capacity, 1);
        t.record(42);
        assert_eq!(t.percentile(0.5), 42);
    }

    #[test]
    fn latency_tracker_overflow() {
        // Ring buffer wraps correctly
        let mut t = LatencyTracker::new(3);
        t.record(10);
        t.record(20);
        t.record(30);
        // Buffer full; next record overwrites oldest
        t.record(40);
        assert_eq!(t.count, 3);
        // samples should be [40, 20, 30] (head wrapped around)
        let p50 = t.percentile(0.5);
        assert_eq!(p50, 30); // sorted: [20, 30, 40], median = 30
    }
}
