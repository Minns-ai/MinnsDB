//! Read Gate — semaphore-gated concurrent read admission.
//!
//! Limits the number of concurrent read operations (NLQ queries, search, etc.)
//! to prevent unbounded resource exhaustion. Reads that cannot acquire a permit
//! within the timeout receive HTTP 503.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Observability counters for the read gate.
pub struct ReadGateMetrics {
    pub in_flight: AtomicU64,
    pub completed: AtomicU64,
    pub rejected: AtomicU64,
    pub read_latency: parking_lot::Mutex<crate::write_lanes::LatencyTracker>,
}

/// Semaphore-gated read admission controller.
pub struct ReadGate {
    semaphore: Arc<Semaphore>,
    pub permits_total: usize,
    pub metrics: Arc<ReadGateMetrics>,
}

/// RAII permit — tracks in-flight count and records latency on drop.
pub struct ReadPermit {
    _permit: OwnedSemaphorePermit,
    start: std::time::Instant,
    metrics: Arc<ReadGateMetrics>,
}

impl Drop for ReadPermit {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        self.metrics.in_flight.fetch_sub(1, Relaxed);
        self.metrics.completed.fetch_add(1, Relaxed);
        self.metrics.read_latency.lock().record(elapsed_us);

        let in_flight = self.metrics.in_flight.load(Relaxed) as f64;
        metrics::gauge!("minns_read_gate_in_flight").set(in_flight);
        metrics::counter!("minns_read_gate_completed_total").increment(1);
        metrics::histogram!("minns_read_duration_seconds").record(elapsed.as_secs_f64());
    }
}

impl ReadGate {
    /// Create a new read gate with `permits` concurrent read slots.
    pub fn new(permits: usize) -> Self {
        // Seed the gauge so scrapes prior to the first acquire still see a
        // value. `permits_total` is exposed as a separate gauge so dashboards
        // can compute utilisation.
        metrics::gauge!("minns_read_gate_in_flight").set(0.0);
        metrics::gauge!("minns_read_gate_permits_total").set(permits as f64);
        Self {
            semaphore: Arc::new(Semaphore::new(permits)),
            permits_total: permits,
            metrics: Arc::new(ReadGateMetrics {
                in_flight: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                rejected: AtomicU64::new(0),
                read_latency: parking_lot::Mutex::new(crate::write_lanes::LatencyTracker::new(
                    1000,
                )),
            }),
        }
    }

    /// Acquire a read permit. Returns `Err` if the semaphore is exhausted
    /// after a 10-second timeout.
    pub async fn acquire(&self) -> Result<ReadPermit, String> {
        let acquire_start = std::time::Instant::now();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            self.semaphore.clone().acquire_owned(),
        )
        .await;
        let wait = acquire_start.elapsed();
        metrics::histogram!("minns_read_gate_acquire_seconds").record(wait.as_secs_f64());

        match result {
            Ok(Ok(permit)) => {
                self.metrics.in_flight.fetch_add(1, Relaxed);
                let in_flight = self.metrics.in_flight.load(Relaxed) as f64;
                metrics::gauge!("minns_read_gate_in_flight").set(in_flight);
                Ok(ReadPermit {
                    _permit: permit,
                    start: std::time::Instant::now(),
                    metrics: self.metrics.clone(),
                })
            },
            Ok(Err(_)) => {
                self.metrics.rejected.fetch_add(1, Relaxed);
                metrics::counter!("minns_read_gate_rejected_total", "reason" => "closed")
                    .increment(1);
                Err("Read gate semaphore closed".to_string())
            },
            Err(_) => {
                self.metrics.rejected.fetch_add(1, Relaxed);
                metrics::counter!("minns_read_gate_rejected_total", "reason" => "timeout")
                    .increment(1);
                Err("Read gate exhausted (timeout after 10s)".to_string())
            },
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release_permit() {
        let gate = ReadGate::new(2);
        assert_eq!(gate.metrics.in_flight.load(Relaxed), 0);

        let permit = gate.acquire().await.unwrap();
        assert_eq!(gate.metrics.in_flight.load(Relaxed), 1);

        drop(permit);
        assert_eq!(gate.metrics.in_flight.load(Relaxed), 0);
        assert_eq!(gate.metrics.completed.load(Relaxed), 1);
    }

    #[tokio::test]
    async fn metrics_track_multiple_permits() {
        let gate = ReadGate::new(4);
        let p1 = gate.acquire().await.unwrap();
        let p2 = gate.acquire().await.unwrap();
        assert_eq!(gate.metrics.in_flight.load(Relaxed), 2);

        drop(p1);
        assert_eq!(gate.metrics.in_flight.load(Relaxed), 1);
        assert_eq!(gate.metrics.completed.load(Relaxed), 1);

        drop(p2);
        assert_eq!(gate.metrics.in_flight.load(Relaxed), 0);
        assert_eq!(gate.metrics.completed.load(Relaxed), 2);
    }
}
