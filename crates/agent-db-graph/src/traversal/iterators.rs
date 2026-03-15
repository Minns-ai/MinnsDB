//! Lazy iterators for graph traversal: BFS, DFS, and Dijkstra.
//!
//! These iterators yield elements one at a time, enabling early termination
//! via `.take(N)` or `.take_while(...)` without expanding the full frontier.

use super::cache::PathEntry;
use super::edge_cost::edge_cost_between;
use crate::structures::{Graph, NodeId};
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, HashSet, VecDeque};

// ============================================================================
// Lazy Iterators
// ============================================================================

/// Lazy BFS iterator that yields `(node_id, depth)` pairs one at a time.
///
/// Consumers can `.take(N)` for early termination — only nodes actually
/// dequeued trigger neighbor expansion, so `.take(10)` on a graph with
/// 100K reachable nodes touches at most ~10 * max_fanout nodes.
///
/// ```ignore
/// let bfs = BfsIter::new(&graph, start, 5);
/// let first_ten: Vec<_> = bfs.take(10).collect();
/// ```
pub struct BfsIter<'a> {
    graph: &'a Graph,
    queue: VecDeque<(NodeId, u32)>,
    visited: HashSet<NodeId>,
    max_depth: u32,
}

impl<'a> BfsIter<'a> {
    /// Create a BFS iterator rooted at `start`, bounded by `max_depth` hops.
    pub fn new(graph: &'a Graph, start: NodeId, max_depth: u32) -> Self {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back((start, 0));
        Self {
            graph,
            queue,
            visited,
            max_depth,
        }
    }
}

impl<'a> Iterator for BfsIter<'a> {
    type Item = (NodeId, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let (current, depth) = self.queue.pop_front()?;

        // Expand neighbors only if we haven't exhausted depth budget.
        if depth < self.max_depth {
            for neighbor in self.graph.get_neighbors(current) {
                if self.visited.insert(neighbor) {
                    self.queue.push_back((neighbor, depth + 1));
                }
            }
        }
        Some((current, depth))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Lower bound: items already in the queue.
        (self.queue.len(), None)
    }
}

/// Lazy DFS (pre-order) iterator that yields `(node_id, depth)` pairs.
///
/// Uses an explicit stack to avoid recursion limits on deep graphs.
pub struct DfsIter<'a> {
    graph: &'a Graph,
    stack: Vec<(NodeId, u32)>,
    visited: HashSet<NodeId>,
    max_depth: u32,
}

impl<'a> DfsIter<'a> {
    /// Create a DFS iterator rooted at `start`, bounded by `max_depth` hops.
    pub fn new(graph: &'a Graph, start: NodeId, max_depth: u32) -> Self {
        let mut visited = HashSet::new();
        visited.insert(start);
        Self {
            graph,
            stack: vec![(start, 0)],
            visited,
            max_depth,
        }
    }
}

impl<'a> Iterator for DfsIter<'a> {
    type Item = (NodeId, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let (current, depth) = self.stack.pop()?;

        if depth < self.max_depth {
            for neighbor in self.graph.get_neighbors(current) {
                if self.visited.insert(neighbor) {
                    self.stack.push((neighbor, depth + 1));
                }
            }
        }
        Some((current, depth))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }
}

/// Lazy Dijkstra iterator that yields `(node_id, cost)` pairs in order
/// of increasing traversal cost. Consumers can `.take(N)` or
/// `.take_while(|(_, c)| *c < threshold)` for bounded exploration.
pub struct DijkstraIter<'a> {
    graph: &'a Graph,
    heap: BinaryHeap<PathEntry>,
    dist: FxHashMap<NodeId, f32>,
}

impl<'a> DijkstraIter<'a> {
    /// Create a Dijkstra iterator rooted at `start`.
    /// Yields every reachable node in cost-ascending order.
    pub fn new(graph: &'a Graph, start: NodeId) -> Self {
        let mut dist = FxHashMap::default();
        let mut heap = BinaryHeap::new();
        dist.insert(start, 0.0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });
        Self { graph, heap, dist }
    }
}

impl<'a> Iterator for DijkstraIter<'a> {
    type Item = (NodeId, f32);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(PathEntry { node_id, cost }) = self.heap.pop() {
            // Skip stale entries (a cheaper path was already processed).
            if cost > *self.dist.get(&node_id).unwrap_or(&f32::INFINITY) {
                continue;
            }

            // Expand neighbors
            for &neighbor in self.graph.get_neighbors(node_id).iter() {
                let w = edge_cost_between(self.graph, node_id, neighbor);
                let new_cost = cost + w;
                if new_cost < *self.dist.get(&neighbor).unwrap_or(&f32::INFINITY) {
                    self.dist.insert(neighbor, new_cost);
                    self.heap.push(PathEntry {
                        node_id: neighbor,
                        cost: new_cost,
                    });
                }
            }

            return Some((node_id, cost));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (if self.heap.is_empty() { 0 } else { 1 }, None)
    }
}
