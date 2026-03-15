//! Streaming query results: QueryContext, CancelHandle, StreamingQuery.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Execution context for a streaming query.
///
/// `items_yielded` is auto-updated by `StreamingQuery`. Other metrics
/// (`nodes_visited`, `edges_traversed`, `max_depth_seen`) are updated
/// by the caller or iterator wrapper if desired.
pub struct QueryContext {
    cancelled: Arc<AtomicBool>,
    limit: u64,
    items_yielded: AtomicU64,
    pub nodes_visited: AtomicU64,
    pub edges_traversed: AtomicU64,
    pub max_depth_seen: AtomicU32,
}

impl QueryContext {
    /// Create a new context with the given item limit.
    pub fn new(limit: u64) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            limit,
            items_yielded: AtomicU64::new(0),
            nodes_visited: AtomicU64::new(0),
            edges_traversed: AtomicU64::new(0),
            max_depth_seen: AtomicU32::new(0),
        }
    }

    /// Returns true if the query should stop (limit reached or cancelled).
    pub fn is_done(&self) -> bool {
        self.cancelled.load(AtomicOrdering::Relaxed)
            || self.items_yielded.load(AtomicOrdering::Relaxed) >= self.limit
    }

    /// Obtain a cancellation handle that can stop this query from
    /// another thread or async task.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            cancelled: Arc::clone(&self.cancelled),
        }
    }

    pub fn items_yielded(&self) -> u64 {
        self.items_yielded.load(AtomicOrdering::Relaxed)
    }
}

/// A clonable handle for cancelling a streaming query.
#[derive(Clone)]
pub struct CancelHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancelHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, AtomicOrdering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(AtomicOrdering::Relaxed)
    }
}

/// A streaming query that yields results in batches, respecting a global
/// item limit and a cancel handle.
pub struct StreamingQuery<I: Iterator> {
    iter: I,
    context: QueryContext,
    batch_size: usize,
}

impl<I: Iterator> StreamingQuery<I> {
    pub fn new(iter: I, limit: u64, batch_size: usize) -> Self {
        Self {
            iter,
            context: QueryContext::new(limit),
            batch_size,
        }
    }

    /// Access the query context (for metrics or cancel handle).
    pub fn context(&self) -> &QueryContext {
        &self.context
    }

    /// Yield the next batch of items (up to `batch_size`).
    pub fn next_batch(&mut self) -> Option<Vec<I::Item>> {
        if self.context.is_done() {
            return None;
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if self.context.is_done() {
                break;
            }
            match self.iter.next() {
                Some(item) => {
                    self.context
                        .items_yielded
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    batch.push(item);
                },
                None => break,
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Collect all remaining items (up to the limit).
    pub fn collect_all(&mut self) -> Vec<I::Item> {
        let mut all = Vec::new();
        while let Some(batch) = self.next_batch() {
            all.extend(batch);
        }
        all
    }
}
