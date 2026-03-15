//! LRU query cache with TTL and graph-generation-based invalidation.

use super::types::QueryResult;
use crate::structures::NodeId;
use std::cmp::Ordering;

// ============================================================================
// Priority queue entry for Dijkstra / A*
// ============================================================================

/// Priority queue entry for pathfinding algorithms.
/// Stores only the node and cost — path is reconstructed from `came_from` map.
#[derive(Debug, Clone)]
pub(crate) struct PathEntry {
    pub node_id: NodeId,
    pub cost: f32,
}

impl PartialEq for PathEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
    }
}

impl Eq for PathEntry {}

impl PartialOrd for PathEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior; NaN-safe via total_cmp
        other.cost.total_cmp(&self.cost)
    }
}

// ============================================================================
// LRU Query Cache
// ============================================================================

/// Single cache entry with value and insertion time for TTL.
struct CacheEntry {
    result: QueryResult,
    inserted_at: std::time::Instant,
}

/// LRU-evicting query cache with TTL expiration.
/// Keys are u64 hashes of `GraphQuery` — no allocation per lookup.
pub(crate) struct QueryCache {
    entries: lru::LruCache<u64, CacheEntry>,
    ttl_secs: u64,
    /// Graph generation when the cache was last validated. Entries are
    /// implicitly stale when `graph.generation() > known_generation`.
    known_generation: u64,
    hits: u64,
    misses: u64,
}

impl QueryCache {
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            entries: lru::LruCache::new(
                std::num::NonZeroUsize::new(capacity)
                    .unwrap_or(std::num::NonZeroUsize::new(1).unwrap()),
            ),
            ttl_secs,
            known_generation: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Lookup a cached result. Returns None if missing or expired.
    pub fn get(&mut self, key: u64) -> Option<&QueryResult> {
        // LruCache::get promotes the entry; we must also check TTL.
        if let Some(entry) = self.entries.get(&key) {
            if entry.inserted_at.elapsed().as_secs() < self.ttl_secs {
                self.hits += 1;
                // Re-borrow to satisfy the borrow checker
                return self.entries.peek(&key).map(|e| &e.result);
            }
            // Expired — remove
            self.entries.pop(&key);
        }
        self.misses += 1;
        None
    }

    pub fn insert(&mut self, key: u64, result: QueryResult) {
        self.entries.push(
            key,
            CacheEntry {
                result,
                inserted_at: std::time::Instant::now(),
            },
        );
    }

    /// Invalidate all entries (called after graph mutations).
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }

    /// Check graph generation and auto-invalidate if the graph has mutated.
    pub fn check_generation(&mut self, graph_generation: u64) {
        if graph_generation > self.known_generation {
            self.entries.clear();
            self.known_generation = graph_generation;
        }
    }

    /// Remove expired entries without blocking normal lookups.
    pub fn evict_expired(&mut self) {
        let now = std::time::Instant::now();
        let keys_to_remove: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.inserted_at).as_secs() >= self.ttl_secs)
            .map(|(k, _)| *k)
            .collect();
        for key in keys_to_remove {
            self.entries.pop(&key);
        }
    }
}
