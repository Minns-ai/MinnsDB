//! Async job store with content-hash dedupe.
//!
//! Three concerns kept in one place because they share state:
//!
//! 1. **Dedupe** — identical conversation payloads (same content hash)
//!    reuse an existing job rather than re-running the 30 s LLM pipeline.
//!    Demo "click again" → cached response in <50 ms.
//!
//! 2. **Async lifecycle** — POST /api/conversations/ingest returns 202 +
//!    job_id immediately; processing happens off the request task. Status
//!    transitions broadcast on a per-job channel so WebSocket subscribers
//!    and HTTP pollers see the same source of truth.
//!
//! 3. **Backpressure** — a bounded in-flight counter rejects new work with
//!    429 + Retry-After when the system is saturated. No silent queueing,
//!    no unbounded memory growth, no timeout traps for the caller.
//!
//! Multi-agent scaling target: 100s of concurrent agents must NOT block.
//! The dedupe + 202 response combo lets us absorb traffic spikes without
//! any agent waiting on another agent's LLM round-trips.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

/// Opaque job identifier. Hex-encoded UUIDv4. Treated as a string by all
/// callers — the format is internal.
pub type JobId = String;

/// A 64-bit content hash. Sized for speed (`DefaultHasher`, SipHash13)
/// rather than collision resistance — two distinct payloads colliding
/// would dedupe falsely, which is a correctness issue not a security
/// one. We accept the trade because callers should not be feeding
/// adversarial payloads to an in-memory ingest queue.
pub type ContentHash = u64;

/// Lifecycle states surfaced to status pollers and subscribers. Done +
/// Failed are terminal; transitions stop after either.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobState {
    /// Submitted but not yet picked up by a worker.
    Queued,
    /// A worker has started processing.
    Processing,
    /// Completed successfully. `response` is the JSON body the synchronous
    /// path would have returned.
    Done { response: serde_json::Value },
    /// Failed. `error` is the message; callers may retry with a new
    /// submission (which will dedupe and observe the failure).
    Failed { error: String },
}

impl JobState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Failed { .. })
    }
}

/// One job's record. Broadcast capacity is small because subscribers
/// receive at most one state transition per phase (queued → processing
/// → done/failed) plus optional progress events — slow consumers can
/// miss transitions and the polling endpoint fills the gap.
struct JobEntry {
    state: JobState,
    created_at: Instant,
    tx: broadcast::Sender<JobState>,
}

/// Errors callers should map to HTTP responses.
#[derive(Debug, Clone)]
pub enum JobError {
    /// In-flight queue is at capacity. Caller should retry after the
    /// suggested delay; HTTP layer should emit 429 + Retry-After.
    QueueFull { retry_after: Duration },
    /// Requested JobId is not in the store (expired, evicted, or never
    /// existed). HTTP layer maps to 404.
    NotFound,
}

/// Default in-flight cap. Sits above WRITE_LANES so backpressure is
/// triggered by genuine pipeline saturation rather than incidental
/// queueing inside the lanes. Override via `JOB_QUEUE_MAX_INFLIGHT`.
const DEFAULT_MAX_INFLIGHT: usize = 128;

/// Default retention for terminal jobs. Long enough for a status poll
/// or subscribe-after-the-fact; short enough not to leak memory.
const TERMINAL_TTL: Duration = Duration::from_secs(60 * 60);

/// Maximum jobs retained in the store regardless of state. Hard
/// safety net against runaway producers; in practice the TTL evicts
/// terminal jobs first.
const MAX_TOTAL_JOBS: usize = 10_000;

/// The shared job store. Cheap to clone (one Arc).
#[derive(Clone)]
pub struct JobStore {
    inner: Arc<JobStoreInner>,
}

struct JobStoreInner {
    jobs: Mutex<HashMap<JobId, JobEntry>>,
    /// Maps content hash → existing JobId for dedupe. Cleared when the
    /// job is evicted.
    by_hash: Mutex<HashMap<ContentHash, JobId>>,
    max_inflight: usize,
}

impl JobStore {
    pub fn new() -> Self {
        let max_inflight = std::env::var("JOB_QUEUE_MAX_INFLIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_INFLIGHT);
        Self {
            inner: Arc::new(JobStoreInner {
                jobs: Mutex::new(HashMap::new()),
                by_hash: Mutex::new(HashMap::new()),
                max_inflight,
            }),
        }
    }

    /// Result of submitting a new job. The caller decides what to do
    /// based on which variant comes back.
    pub fn submit(&self, hash: ContentHash) -> Result<SubmitOutcome, JobError> {
        // Sweep expired terminal jobs before touching counts. Cheap when
        // the map is small; bounded by MAX_TOTAL_JOBS otherwise.
        self.evict_expired();

        // Dedupe path. If a job already exists for this content hash,
        // reuse it regardless of state — callers that want the response
        // can fetch it from the cached terminal state.
        if let Some(existing_id) = self.inner.by_hash.lock().get(&hash).cloned() {
            if let Some(entry) = self.inner.jobs.lock().get(&existing_id) {
                return Ok(SubmitOutcome::Existing {
                    id: existing_id,
                    state: entry.state.clone(),
                });
            }
            // The by_hash entry pointed at a now-evicted job. Fall
            // through to create a fresh one.
        }

        // Backpressure check. We only count non-terminal jobs against
        // the cap because terminal records are cheap (no LLM work
        // pending) and pruned by TTL.
        let inflight = self
            .inner
            .jobs
            .lock()
            .values()
            .filter(|e| !e.state.is_terminal())
            .count();
        if inflight >= self.inner.max_inflight {
            return Err(JobError::QueueFull {
                retry_after: Duration::from_secs(5),
            });
        }

        // Create the new job. Broadcast channel capacity 16 covers
        // queued → processing → done plus headroom for progress events
        // without blocking the producer if a subscriber is slow.
        let id = new_job_id();
        let (tx, _rx) = broadcast::channel(16);
        let entry = JobEntry {
            state: JobState::Queued,
            created_at: Instant::now(),
            tx,
        };
        self.inner.jobs.lock().insert(id.clone(), entry);
        self.inner.by_hash.lock().insert(hash, id.clone());
        Ok(SubmitOutcome::Created { id })
    }

    /// Transition a job to a new state and broadcast it. Idempotent:
    /// transitions on missing or terminal jobs are no-ops, which keeps
    /// the worker code free of "what if it raced eviction" branches.
    pub fn transition(&self, id: &str, state: JobState) {
        let mut jobs = self.inner.jobs.lock();
        if let Some(entry) = jobs.get_mut(id) {
            if entry.state.is_terminal() {
                return;
            }
            entry.state = state.clone();
            // send returns Err only when there are no subscribers,
            // which is the common case (polling, not subscribing). The
            // broadcast channel still queues the value for late
            // subscribers up to its capacity.
            let _ = entry.tx.send(state);
        }
    }

    /// Read the current state of a job without subscribing.
    pub fn get(&self, id: &str) -> Result<JobState, JobError> {
        self.inner
            .jobs
            .lock()
            .get(id)
            .map(|e| e.state.clone())
            .ok_or(JobError::NotFound)
    }

    /// Subscribe to state transitions on a job. Late subscribers
    /// receive the current state on connect via the polling layer;
    /// they will see any subsequent transitions via this receiver.
    pub fn subscribe(&self, id: &str) -> Result<broadcast::Receiver<JobState>, JobError> {
        self.inner
            .jobs
            .lock()
            .get(id)
            .map(|e| e.tx.subscribe())
            .ok_or(JobError::NotFound)
    }

    /// Sweep terminal jobs older than `TERMINAL_TTL` and enforce
    /// `MAX_TOTAL_JOBS`. Called inline on submit; cheap because the
    /// map is small in steady state.
    fn evict_expired(&self) {
        let now = Instant::now();
        let mut jobs = self.inner.jobs.lock();
        let mut by_hash = self.inner.by_hash.lock();

        // TTL sweep on terminal jobs.
        let expired: Vec<String> = jobs
            .iter()
            .filter(|(_, e)| {
                e.state.is_terminal() && now.duration_since(e.created_at) > TERMINAL_TTL
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            jobs.remove(id);
        }

        // Hard cap. If we're still over the limit, drop oldest terminal
        // jobs first; non-terminal entries are spared because they
        // represent work in flight that the client expects to find.
        if jobs.len() > MAX_TOTAL_JOBS {
            let mut by_age: Vec<(String, Instant)> = jobs
                .iter()
                .filter(|(_, e)| e.state.is_terminal())
                .map(|(id, e)| (id.clone(), e.created_at))
                .collect();
            by_age.sort_by_key(|(_, ts)| *ts);
            let to_drop = jobs.len() - MAX_TOTAL_JOBS;
            for (id, _) in by_age.into_iter().take(to_drop) {
                jobs.remove(&id);
            }
        }

        // Drop dangling by_hash entries whose JobId no longer exists.
        by_hash.retain(|_, id| jobs.contains_key(id));
    }
}

impl Default for JobStore {
    fn default() -> Self {
        Self::new()
    }
}

/// What happened on submit.
pub enum SubmitOutcome {
    /// First time this hash was submitted; caller should spawn the work.
    Created { id: JobId },
    /// Hash matched an existing job; caller should serve the existing
    /// `state` (cached response if Done, "still in flight" if not).
    Existing { id: JobId, state: JobState },
}

/// 64-bit content hash via the standard `DefaultHasher`. Not stable
/// across Rust versions — fine for in-memory dedupe, not for persistent
/// keys.
pub fn content_hash<T: std::hash::Hash>(value: &T) -> ContentHash {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// New random JobId via the already-vendored `uuid` v4 crate.
fn new_job_id() -> JobId {
    uuid::Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_first_time_returns_created() {
        let store = JobStore::new();
        let outcome = store.submit(123).expect("submit");
        match outcome {
            SubmitOutcome::Created { id } => assert!(!id.is_empty()),
            _ => panic!("expected Created"),
        }
    }

    #[test]
    fn submit_same_hash_returns_existing() {
        let store = JobStore::new();
        let first = match store.submit(456).expect("first") {
            SubmitOutcome::Created { id } => id,
            _ => panic!(),
        };
        match store.submit(456).expect("second") {
            SubmitOutcome::Existing { id, state } => {
                assert_eq!(id, first);
                assert!(matches!(state, JobState::Queued));
            },
            _ => panic!("expected Existing"),
        }
    }

    #[test]
    fn transition_to_done_caches_response() {
        let store = JobStore::new();
        let id = match store.submit(789).expect("submit") {
            SubmitOutcome::Created { id } => id,
            _ => panic!(),
        };
        store.transition(
            &id,
            JobState::Done {
                response: serde_json::json!({"ok": true}),
            },
        );
        match store.get(&id).expect("get") {
            JobState::Done { response } => {
                assert_eq!(response, serde_json::json!({"ok": true}))
            },
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn terminal_transitions_are_idempotent() {
        let store = JobStore::new();
        let id = match store.submit(1).expect("submit") {
            SubmitOutcome::Created { id } => id,
            _ => panic!(),
        };
        store.transition(
            &id,
            JobState::Failed {
                error: "first".to_string(),
            },
        );
        // Re-transitioning a terminal job is a no-op; the first error
        // sticks. Workers may double-fire on shutdown races.
        store.transition(
            &id,
            JobState::Done {
                response: serde_json::json!({}),
            },
        );
        assert!(matches!(
            store.get(&id).expect("get"),
            JobState::Failed { .. }
        ));
    }

    #[test]
    fn queue_full_returns_backpressure_err() {
        // Tiny store via env override would change global state in
        // tests; instead saturate the default cap directly. 128 cheap
        // submits to hit the limit; 129th should be rejected.
        let store = JobStore::new();
        for i in 0..DEFAULT_MAX_INFLIGHT {
            store.submit(1_000_000 + i as u64).expect("each within cap");
        }
        match store.submit(99_999_999) {
            Err(JobError::QueueFull { retry_after }) => {
                assert!(retry_after.as_secs() > 0);
            },
            _ => panic!("expected QueueFull"),
        }
    }

    #[test]
    fn get_unknown_id_returns_not_found() {
        let store = JobStore::new();
        match store.get("doesnotexist") {
            Err(JobError::NotFound) => {},
            _ => panic!("expected NotFound"),
        }
    }

    #[tokio::test]
    async fn subscribe_receives_transitions() {
        let store = JobStore::new();
        let id = match store.submit(42).expect("submit") {
            SubmitOutcome::Created { id } => id,
            _ => panic!(),
        };
        let mut rx = store.subscribe(&id).expect("subscribe");
        store.transition(&id, JobState::Processing);
        let received = rx.recv().await.expect("recv");
        assert!(matches!(received, JobState::Processing));
    }
}
