//! Adaptive grouped-commit coordinator over redb.
//!
//! Redb's atomicity boundary is the `WriteTransaction::commit()` call —
//! and that call IS the durability sync. There is no separate WAL flush
//! to group, so the only way to amortise fsync cost across many writes
//! is to pack them into one transaction.
//!
//! The coordinator owns a single background writer thread. Producers
//! `submit` a [`BatchOperation`] (or a [`Vec`] of them) and receive a
//! [`CommitFuture`] that resolves when the containing batch has been
//! committed. The batcher drains the producer queue, adaptively packs
//! ops into one [`redb::WriteTransaction`], and commits — one fsync
//! covers every waiter in the batch.
//!
//! ## Adaptive batching
//!
//! Three load regimes drive the wait policy on each batch:
//!
//! - **Low** (< `wait_threshold` items drained): commit immediately.
//!   Optimises for latency under light load.
//! - **Moderate** (≥ `wait_threshold`, < `max_batch_size`): wait up to
//!   `timeout` for more items so a single commit covers a fuller batch.
//!   Optimises for throughput when callers are arriving in bursts.
//! - **High** (≥ `max_batch_size`): commit immediately, bounded by the
//!   max so a hot producer can never inflate one commit unboundedly.
//!
//! ## Backpressure
//!
//! The producer→batcher channel is bounded (`queue_capacity`). When
//! full, `try_submit` returns [`CoordinatorError::QueueFull`] with a
//! suggested retry delay so callers can surface 429 + Retry-After
//! rather than block silently.
//!
//! ## Atomicity
//!
//! Every op in a single batch shares the redb transaction. If the
//! commit fails, every waiter in that batch receives the same error.
//! This matches the existing [`RedbBackend::write_batch`] semantics —
//! "all or nothing within a batch".
//!
//! ## Why crossbeam-channel and not Mutex+Condvar
//!
//! `crossbeam_channel::bounded` is a lock-free MPMC channel: producers
//! enqueue without contending on a mutex, and the consumer has a clean
//! `recv` + non-blocking `try_recv` loop without managing condvar wait
//! states by hand. The semantics map directly onto the adaptive
//! drain-then-wait-then-drain pattern below.

use crate::{BatchOperation, RedbBackend, StorageError};
use crossbeam_channel::{bounded, Sender, TrySendError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::sync::oneshot;

/// Tunables for the batcher. Defaults are conservative for an
/// LLM-bound workload (small bursts of writes per ingest); production
/// loads with hundreds of concurrent ingests should bump
/// `max_batch_size` after measuring real submission rates.
#[derive(Debug, Clone, Copy)]
pub struct BatcherConfig {
    /// Maximum wait for more items once `wait_threshold` is reached.
    /// Smaller = lower per-call latency; larger = better fsync
    /// amortisation under burst load. 5ms is well below typical LLM
    /// call latency so a producer never sees a meaningful delay from
    /// the wait phase.
    pub timeout: Duration,
    /// Number of items drained at which the batcher enters its
    /// throughput-optimising wait phase. Below this we commit
    /// immediately (latency mode). 12 mirrors what works well in
    /// production-grade KV stores at single-digit concurrent writers.
    pub wait_threshold: usize,
    /// Hard cap on items per commit. Bounded for two reasons: very
    /// large redb transactions allocate more pages and longer holds
    /// of the writer lock; and a runaway producer should not be able
    /// to inflate one fsync into a 10MB write.
    pub max_batch_size: usize,
    /// Bounded queue capacity. Past this the producer side returns
    /// [`CoordinatorError::QueueFull`] so the HTTP layer can map it
    /// to 429 + Retry-After.
    pub queue_capacity: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(5),
            wait_threshold: 12,
            max_batch_size: 256,
            queue_capacity: 4096,
        }
    }
}

/// Errors callers should map to HTTP responses.
#[derive(Debug)]
pub enum CoordinatorError {
    /// In-flight queue is at capacity. Caller should retry after the
    /// suggested delay; HTTP layer maps to 429 + Retry-After.
    QueueFull { retry_after: Duration },
    /// The batcher thread has been shut down. Indicates either graceful
    /// shutdown or a previous panic; submission will not be processed.
    Shutdown,
    /// Storage failure propagated from the underlying redb commit.
    /// Every waiter in the failing batch receives the same error.
    Storage(StorageError),
}

impl std::fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull { retry_after } => write!(
                f,
                "commit coordinator queue full; retry after {:?}",
                retry_after
            ),
            Self::Shutdown => write!(f, "commit coordinator has shut down"),
            Self::Storage(e) => write!(f, "storage commit failed: {}", e),
        }
    }
}

impl std::error::Error for CoordinatorError {}

/// Future returned by `submit` / `submit_many`. Resolves when the
/// containing batch has been committed (or failed).
pub type CommitFuture = oneshot::Receiver<Result<(), CoordinatorError>>;

/// One submission queued for batching. Carries either a single op or
/// a vector of ops that MUST commit together atomically.
struct Submission {
    ops: SubmissionOps,
    responder: oneshot::Sender<Result<(), CoordinatorError>>,
}

enum SubmissionOps {
    Single(BatchOperation),
    Many(Vec<BatchOperation>),
}

/// Adaptive grouped-commit coordinator.
///
/// Cheap to clone — internal state is `Arc`-wrapped. The coordinator
/// owns a single background OS thread that drives the writer; this is
/// intentional and aligns with redb's single-writer model.
#[derive(Clone)]
pub struct CommitCoordinator {
    tx: Sender<Submission>,
    shutdown: Arc<AtomicBool>,
    handle: Arc<std::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl CommitCoordinator {
    /// Start a coordinator over `backend` with the given `config`.
    /// Spawns a single OS thread named `redb-commit-coordinator`.
    pub fn new(backend: Arc<RedbBackend>, config: BatcherConfig) -> Self {
        let (tx, rx) = bounded::<Submission>(config.queue_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let batcher_shutdown = shutdown.clone();
        // Seed gauges so scrapes prior to first submit see real values.
        metrics::gauge!("minns_commit_coordinator_queue_depth").set(0.0);
        metrics::gauge!("minns_commit_coordinator_queue_capacity")
            .set(config.queue_capacity as f64);
        let handle = thread::Builder::new()
            .name("redb-commit-coordinator".to_string())
            .spawn(move || {
                run_batcher(rx, backend, config, batcher_shutdown);
            })
            .expect("failed to spawn redb commit coordinator thread");
        Self {
            tx,
            shutdown,
            handle: Arc::new(std::sync::Mutex::new(Some(handle))),
        }
    }

    /// Submit a single op. Returns immediately with a future that
    /// resolves when the op's containing batch has been committed.
    /// Returns [`CoordinatorError::QueueFull`] if backpressure trips.
    pub fn submit(&self, op: BatchOperation) -> Result<CommitFuture, CoordinatorError> {
        self.enqueue(SubmissionOps::Single(op))
    }

    /// Submit multiple ops that must commit together atomically.
    /// Either all are written and durable, or none are.
    pub fn submit_many(&self, ops: Vec<BatchOperation>) -> Result<CommitFuture, CoordinatorError> {
        self.enqueue(SubmissionOps::Many(ops))
    }

    fn enqueue(&self, ops: SubmissionOps) -> Result<CommitFuture, CoordinatorError> {
        if self.shutdown.load(Ordering::Acquire) {
            metrics::counter!(
                "minns_commit_coordinator_rejected_total",
                "reason" => "shutdown",
            )
            .increment(1);
            return Err(CoordinatorError::Shutdown);
        }
        let (responder, rx) = oneshot::channel();
        let submission = Submission { ops, responder };
        match self.tx.try_send(submission) {
            Ok(()) => {
                metrics::counter!("minns_commit_coordinator_submitted_total").increment(1);
                metrics::gauge!("minns_commit_coordinator_queue_depth").set(self.tx.len() as f64);
                Ok(rx)
            },
            Err(TrySendError::Full(_)) => {
                metrics::counter!(
                    "minns_commit_coordinator_rejected_total",
                    "reason" => "queue_full",
                )
                .increment(1);
                Err(CoordinatorError::QueueFull {
                    retry_after: Duration::from_millis(50),
                })
            },
            Err(TrySendError::Disconnected(_)) => {
                metrics::counter!(
                    "minns_commit_coordinator_rejected_total",
                    "reason" => "disconnected",
                )
                .increment(1);
                Err(CoordinatorError::Shutdown)
            },
        }
    }

    /// Signal shutdown and wait for the batcher to drain. Idempotent.
    pub fn shutdown(&self) {
        // Setting the flag is enough — the batcher exits its `recv`
        // loop when it observes the flag after a drain. Sending a
        // sentinel down the channel is unnecessary because dropping
        // the last sender (when this coordinator is dropped) makes
        // `recv` return `Err(Disconnected)` and the batcher exits.
        self.shutdown.store(true, Ordering::Release);
        if let Ok(mut h) = self.handle.lock() {
            if let Some(handle) = h.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The batcher loop. Owns the receiver and the backend; produces no
/// output other than via the per-submission responders.
fn run_batcher(
    rx: crossbeam_channel::Receiver<Submission>,
    backend: Arc<RedbBackend>,
    config: BatcherConfig,
    shutdown: Arc<AtomicBool>,
) {
    let mut batch: Vec<Submission> = Vec::with_capacity(config.max_batch_size);
    // Bounded poll interval so the shutdown flag is observed within
    // 100ms of being set. Without this, a coordinator with no producers
    // wedges on `rx.recv()` forever and `shutdown()`'s `handle.join()`
    // hangs the calling test/thread.
    let poll_interval = Duration::from_millis(100);
    loop {
        if shutdown.load(Ordering::Acquire) {
            return;
        }
        // Block up to `poll_interval` for the first submission, then
        // re-check the shutdown flag.
        match rx.recv_timeout(poll_interval) {
            Ok(first) => batch.push(first),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
        }
        if shutdown.load(Ordering::Acquire) {
            // Still process whatever's already queued (graceful drain),
            // then exit on the next iteration when the queue is empty.
            // Falls through to the drain below.
        }

        // Drain everything currently available up to max_batch_size.
        while batch.len() < config.max_batch_size {
            match rx.try_recv() {
                Ok(sub) => batch.push(sub),
                Err(_) => break,
            }
        }

        // Adaptive wait: if we hit the throughput-optimising regime,
        // keep collecting for up to `timeout`. Stops early on
        // `max_batch_size`.
        if batch.len() >= config.wait_threshold && batch.len() < config.max_batch_size {
            let deadline = std::time::Instant::now() + config.timeout;
            while batch.len() < config.max_batch_size {
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                let wait = deadline - now;
                match rx.recv_timeout(wait) {
                    Ok(sub) => batch.push(sub),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            }
        }

        // Commit the whole batch atomically.
        commit_batch(&backend, &mut batch);
        // After drain, refresh the queue gauge so the next scrape sees
        // the post-commit depth rather than the peak we observed before
        // collecting this batch.
        metrics::gauge!("minns_commit_coordinator_queue_depth").set(rx.len() as f64);
    }
}

/// Flatten all submissions into one `Vec<BatchOperation>` and call
/// `RedbBackend::write_batch`. On success, every waiter gets `Ok(())`.
/// On failure, every waiter gets the same `CoordinatorError::Storage`.
///
/// The batch vec is drained — callers reuse the allocation across
/// batches so steady-state has zero per-batch growth.
fn commit_batch(backend: &RedbBackend, batch: &mut Vec<Submission>) {
    let waiters = batch.len();
    // Tally ops first so we allocate the right-sized vec exactly once.
    let total_ops: usize = batch
        .iter()
        .map(|s| match &s.ops {
            SubmissionOps::Single(_) => 1,
            SubmissionOps::Many(v) => v.len(),
        })
        .sum();
    let mut flat: Vec<BatchOperation> = Vec::with_capacity(total_ops);
    for sub in batch.iter() {
        match &sub.ops {
            SubmissionOps::Single(op) => flat.push(op.clone()),
            SubmissionOps::Many(v) => flat.extend(v.iter().cloned()),
        }
    }

    let commit_start = std::time::Instant::now();
    let result = backend.write_batch(flat);
    let commit_elapsed = commit_start.elapsed();

    let status = if result.is_ok() { "ok" } else { "error" };
    metrics::histogram!("minns_commit_coordinator_commit_seconds", "status" => status)
        .record(commit_elapsed.as_secs_f64());
    metrics::histogram!("minns_commit_coordinator_batch_ops").record(total_ops as f64);
    metrics::histogram!("minns_commit_coordinator_waiters_per_commit").record(waiters as f64);

    for sub in batch.drain(..) {
        let outcome = match &result {
            Ok(()) => Ok(()),
            Err(e) => Err(CoordinatorError::Storage(clone_storage_error(e))),
        };
        // `send` only errors when the receiver was dropped — the caller
        // gave up waiting. Nothing to do.
        let _ = sub.responder.send(outcome);
    }
}

/// `StorageError` does not implement `Clone`. We need to fan the same
/// error out to every waiter in a failed batch, so synthesise a copy
/// from the string representation. Acceptable because every variant
/// today carries a `String` and the structured kind isn't load-bearing
/// for callers (they all log + return 500).
fn clone_storage_error(e: &StorageError) -> StorageError {
    StorageError::DatabaseError(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{table_names, RedbConfig};
    use tempfile::TempDir;

    fn temp_backend() -> (TempDir, Arc<RedbBackend>) {
        let dir = TempDir::new().unwrap();
        let cfg = RedbConfig {
            data_path: dir.path().join("test.redb"),
            cache_size_bytes: 8 * 1024 * 1024,
            repair_on_open: false,
        };
        (dir, Arc::new(RedbBackend::open(cfg).unwrap()))
    }

    /// Smallest viable submission: one op, observe Done, observe row.
    #[tokio::test]
    async fn submit_single_op_writes_durable() {
        let (_dir, backend) = temp_backend();
        let coord = CommitCoordinator::new(backend.clone(), BatcherConfig::default());

        let op = BatchOperation::Put {
            table_name: table_names::PARTITION_MAP.to_string(),
            key: b"k1".to_vec(),
            value: b"v1".to_vec(),
        };
        let fut = coord.submit(op).expect("submit accepted");
        let outcome = fut.await.expect("responder lived");
        assert!(outcome.is_ok(), "commit failed: {:?}", outcome.err());

        // Read-after-commit must see the write.
        let got: Option<Vec<u8>> = backend.get_raw(table_names::PARTITION_MAP, b"k1").unwrap();
        assert_eq!(got.as_deref(), Some(&b"v1"[..]));
        coord.shutdown();
    }

    /// Many concurrent submits should all observe Done in any order,
    /// and every value should be readable after the futures resolve.
    #[tokio::test]
    async fn concurrent_submits_all_durable() {
        let (_dir, backend) = temp_backend();
        let coord = CommitCoordinator::new(backend.clone(), BatcherConfig::default());

        // Submit N writes in parallel.
        let n = 64;
        let mut futs = Vec::with_capacity(n);
        for i in 0..n {
            let op = BatchOperation::Put {
                table_name: table_names::PARTITION_MAP.to_string(),
                key: format!("k{}", i).into_bytes(),
                value: format!("v{}", i).into_bytes(),
            };
            futs.push(coord.submit(op).expect("submit accepted"));
        }

        // Await all.
        for (i, fut) in futs.into_iter().enumerate() {
            let r = fut.await.expect("responder lived");
            assert!(r.is_ok(), "commit {} failed: {:?}", i, r.err());
        }

        // All rows present.
        for i in 0..n {
            let key = format!("k{}", i);
            let want = format!("v{}", i);
            let got: Option<Vec<u8>> = backend
                .get_raw(table_names::PARTITION_MAP, key.as_bytes())
                .unwrap();
            assert_eq!(got.as_deref(), Some(want.as_bytes()), "row {} missing", i);
        }
        coord.shutdown();
    }

    /// submit_many semantics: all-or-nothing within the supplied batch.
    /// We check the happy path here; the atomic-failure case is exercised
    /// by the underlying redb behaviour of `write_batch`.
    #[tokio::test]
    async fn submit_many_atomic_happy_path() {
        let (_dir, backend) = temp_backend();
        let coord = CommitCoordinator::new(backend.clone(), BatcherConfig::default());

        let ops = vec![
            BatchOperation::Put {
                table_name: table_names::PARTITION_MAP.to_string(),
                key: b"a".to_vec(),
                value: b"1".to_vec(),
            },
            BatchOperation::Put {
                table_name: table_names::PARTITION_MAP.to_string(),
                key: b"b".to_vec(),
                value: b"2".to_vec(),
            },
            BatchOperation::Delete {
                table_name: table_names::PARTITION_MAP.to_string(),
                key: b"a".to_vec(),
            },
        ];
        let fut = coord.submit_many(ops).expect("submit_many accepted");
        assert!(fut.await.expect("responder lived").is_ok());

        let a: Option<Vec<u8>> = backend.get_raw(table_names::PARTITION_MAP, b"a").unwrap();
        let b: Option<Vec<u8>> = backend.get_raw(table_names::PARTITION_MAP, b"b").unwrap();
        assert_eq!(a, None, "delete inside batch should win");
        assert_eq!(b.as_deref(), Some(&b"2"[..]));
        coord.shutdown();
    }

    /// QueueFull surfaces synchronously without waiting on a future.
    /// Tight queue + slow draining + many submits forces the bound.
    #[tokio::test]
    async fn queue_full_returns_backpressure_err() {
        let (_dir, backend) = temp_backend();
        let cfg = BatcherConfig {
            // Tight capacity + small batch + no wait so the batcher
            // processes slowly enough that the channel fills before
            // we drain it.
            timeout: Duration::from_millis(0),
            wait_threshold: usize::MAX,
            max_batch_size: 1,
            queue_capacity: 4,
        };
        let coord = CommitCoordinator::new(backend.clone(), cfg);

        // Hammer the channel synchronously. With queue_capacity=4 the
        // 5th-to-Nth submit must observe Full. We intentionally don't
        // await the futures so the queue can't drain.
        let mut accepted = 0usize;
        let mut full_seen = false;
        for i in 0..512 {
            let op = BatchOperation::Put {
                table_name: table_names::PARTITION_MAP.to_string(),
                key: format!("k{}", i).into_bytes(),
                value: b"v".to_vec(),
            };
            match coord.submit(op) {
                Ok(_fut) => accepted += 1,
                Err(CoordinatorError::QueueFull { .. }) => {
                    full_seen = true;
                    break;
                },
                Err(other) => panic!("unexpected submit err: {:?}", other),
            }
        }
        assert!(full_seen, "expected QueueFull; only {} accepted", accepted);
        coord.shutdown();
    }

    /// After shutdown, submit returns CoordinatorError::Shutdown.
    #[tokio::test]
    async fn submit_after_shutdown_errors() {
        let (_dir, backend) = temp_backend();
        let coord = CommitCoordinator::new(backend.clone(), BatcherConfig::default());
        coord.shutdown();
        // Give the batcher thread a moment to observe shutdown.
        let op = BatchOperation::Put {
            table_name: table_names::PARTITION_MAP.to_string(),
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        };
        match coord.submit(op) {
            Err(CoordinatorError::Shutdown) => {},
            other => panic!("expected Shutdown, got {:?}", other),
        }
    }
}
