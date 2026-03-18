//! Per-module usage recording for billing.
//!
//! Tracks cumulative compute/memory/IO per module. Resettable monthly.

use std::sync::atomic::{AtomicU64, Ordering};

use agent_db_core::types::Timestamp;
use serde::{Deserialize, Serialize};

/// Cumulative usage counters for a module. Thread-safe via atomics.
pub struct ModuleUsageCounters {
    pub total_life_consumed_lo: AtomicU64,
    pub total_life_consumed_hi: AtomicU64,
    pub total_calls: AtomicU64,
    pub total_rows_read: AtomicU64,
    pub total_rows_written: AtomicU64,
    pub total_graph_queries: AtomicU64,
    pub total_http_requests: AtomicU64,
    pub total_http_bytes: AtomicU64,
    pub total_subscription_events: AtomicU64,
}

impl ModuleUsageCounters {
    pub fn new() -> Self {
        ModuleUsageCounters {
            total_life_consumed_lo: AtomicU64::new(0),
            total_life_consumed_hi: AtomicU64::new(0),
            total_calls: AtomicU64::new(0),
            total_rows_read: AtomicU64::new(0),
            total_rows_written: AtomicU64::new(0),
            total_graph_queries: AtomicU64::new(0),
            total_http_requests: AtomicU64::new(0),
            total_http_bytes: AtomicU64::new(0),
            total_subscription_events: AtomicU64::new(0),
        }
    }

    /// Record life consumed from a single call (u64 portion).
    pub fn record_life(&self, consumed: u64) {
        let prev = self
            .total_life_consumed_lo
            .fetch_add(consumed, Ordering::Relaxed);
        // Handle overflow into hi word
        if prev.checked_add(consumed).is_none() {
            self.total_life_consumed_hi.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_call(&self) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rows_read(&self, count: u64) {
        self.total_rows_read.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_rows_written(&self, count: u64) {
        self.total_rows_written.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_graph_query(&self) {
        self.total_graph_queries.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_http_request(&self, response_bytes: u64) {
        self.total_http_requests.fetch_add(1, Ordering::Relaxed);
        self.total_http_bytes
            .fetch_add(response_bytes, Ordering::Relaxed);
    }

    pub fn record_subscription_event(&self) {
        self.total_subscription_events
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot current counters into a serialisable record.
    pub fn snapshot(&self, module_name: &str, period_start: Timestamp) -> ModuleUsage {
        ModuleUsage {
            module_name: module_name.to_string(),
            total_life_consumed_lo: self.total_life_consumed_lo.load(Ordering::Relaxed),
            total_life_consumed_hi: self.total_life_consumed_hi.load(Ordering::Relaxed),
            total_calls: self.total_calls.load(Ordering::Relaxed),
            total_rows_read: self.total_rows_read.load(Ordering::Relaxed),
            total_rows_written: self.total_rows_written.load(Ordering::Relaxed),
            total_graph_queries: self.total_graph_queries.load(Ordering::Relaxed),
            total_http_requests: self.total_http_requests.load(Ordering::Relaxed),
            total_http_bytes: self.total_http_bytes.load(Ordering::Relaxed),
            total_subscription_events: self.total_subscription_events.load(Ordering::Relaxed),
            period_start,
            last_updated: agent_db_core::types::current_timestamp(),
        }
    }

    /// Reset all counters to zero. Returns the snapshot before reset.
    pub fn reset(&self, module_name: &str, period_start: Timestamp) -> ModuleUsage {
        let snapshot = self.snapshot(module_name, period_start);
        self.total_life_consumed_lo.store(0, Ordering::Relaxed);
        self.total_life_consumed_hi.store(0, Ordering::Relaxed);
        self.total_calls.store(0, Ordering::Relaxed);
        self.total_rows_read.store(0, Ordering::Relaxed);
        self.total_rows_written.store(0, Ordering::Relaxed);
        self.total_graph_queries.store(0, Ordering::Relaxed);
        self.total_http_requests.store(0, Ordering::Relaxed);
        self.total_http_bytes.store(0, Ordering::Relaxed);
        self.total_subscription_events.store(0, Ordering::Relaxed);
        snapshot
    }

    /// Restore counters from a persisted snapshot (on load).
    pub fn restore(&self, usage: &ModuleUsage) {
        self.total_life_consumed_lo
            .store(usage.total_life_consumed_lo, Ordering::Relaxed);
        self.total_life_consumed_hi
            .store(usage.total_life_consumed_hi, Ordering::Relaxed);
        self.total_calls.store(usage.total_calls, Ordering::Relaxed);
        self.total_rows_read
            .store(usage.total_rows_read, Ordering::Relaxed);
        self.total_rows_written
            .store(usage.total_rows_written, Ordering::Relaxed);
        self.total_graph_queries
            .store(usage.total_graph_queries, Ordering::Relaxed);
        self.total_http_requests
            .store(usage.total_http_requests, Ordering::Relaxed);
        self.total_http_bytes
            .store(usage.total_http_bytes, Ordering::Relaxed);
        self.total_subscription_events
            .store(usage.total_subscription_events, Ordering::Relaxed);
    }
}

/// Serialisable usage record for persistence and API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleUsage {
    pub module_name: String,
    /// Total WASM instructions consumed (low 64 bits of u128).
    pub total_life_consumed_lo: u64,
    /// Total WASM instructions consumed (high 64 bits of u128).
    pub total_life_consumed_hi: u64,
    pub total_calls: u64,
    pub total_rows_read: u64,
    pub total_rows_written: u64,
    pub total_graph_queries: u64,
    pub total_http_requests: u64,
    pub total_http_bytes: u64,
    pub total_subscription_events: u64,
    pub period_start: Timestamp,
    pub last_updated: Timestamp,
}

impl ModuleUsage {
    /// Total life consumed as u128.
    pub fn total_life_consumed(&self) -> u128 {
        ((self.total_life_consumed_hi as u128) << 64) | (self.total_life_consumed_lo as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_snapshot() {
        let counters = ModuleUsageCounters::new();
        counters.record_call();
        counters.record_call();
        counters.record_life(1000);
        counters.record_rows_read(50);
        counters.record_rows_written(3);
        counters.record_graph_query();
        counters.record_http_request(4096);

        let snap = counters.snapshot("test-module", 0);
        assert_eq!(snap.total_calls, 2);
        assert_eq!(snap.total_life_consumed(), 1000);
        assert_eq!(snap.total_rows_read, 50);
        assert_eq!(snap.total_rows_written, 3);
        assert_eq!(snap.total_graph_queries, 1);
        assert_eq!(snap.total_http_requests, 1);
        assert_eq!(snap.total_http_bytes, 4096);
    }

    #[test]
    fn test_reset_returns_previous() {
        let counters = ModuleUsageCounters::new();
        counters.record_call();
        counters.record_life(500);

        let prev = counters.reset("test", 0);
        assert_eq!(prev.total_calls, 1);
        assert_eq!(prev.total_life_consumed(), 500);

        // After reset, everything is zero
        let after = counters.snapshot("test", 0);
        assert_eq!(after.total_calls, 0);
        assert_eq!(after.total_life_consumed(), 0);
    }

    #[test]
    fn test_restore() {
        let counters = ModuleUsageCounters::new();
        let usage = ModuleUsage {
            module_name: "test".into(),
            total_life_consumed_lo: 999,
            total_life_consumed_hi: 0,
            total_calls: 42,
            total_rows_read: 100,
            total_rows_written: 10,
            total_graph_queries: 5,
            total_http_requests: 2,
            total_http_bytes: 8192,
            total_subscription_events: 7,
            period_start: 0,
            last_updated: 0,
        };
        counters.restore(&usage);
        let snap = counters.snapshot("test", 0);
        assert_eq!(snap.total_calls, 42);
        assert_eq!(snap.total_life_consumed(), 999);
    }
}
