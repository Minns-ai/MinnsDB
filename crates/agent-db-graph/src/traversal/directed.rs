//! Direction-aware iterators: BFS, DFS, and Dijkstra with configurable
//! traversal direction (Out, In, Both) and depth specifications.

use super::cache::PathEntry;
use super::edge_cost::edge_cost;
use crate::structures::{Depth, Direction, Graph, NodeId};
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, HashSet, VecDeque};

/// Helper: expand neighbors + costs for a given direction.
fn expand_directed_costs(
    graph: &Graph,
    node_id: NodeId,
    direction: Direction,
) -> Vec<(NodeId, f32)> {
    match direction {
        Direction::Out => graph
            .get_edges_from(node_id)
            .into_iter()
            .map(|e| (e.target, edge_cost(e)))
            .collect(),
        Direction::In => graph
            .get_edges_to(node_id)
            .into_iter()
            .map(|e| (e.source, edge_cost(e)))
            .collect(),
        Direction::Both => {
            let mut best: FxHashMap<NodeId, f32> = FxHashMap::default();
            for e in graph.get_edges_from(node_id) {
                let c = edge_cost(e);
                let entry = best.entry(e.target).or_insert(f32::INFINITY);
                if c < *entry {
                    *entry = c;
                }
            }
            for e in graph.get_edges_to(node_id) {
                let c = edge_cost(e);
                let entry = best.entry(e.source).or_insert(f32::INFINITY);
                if c < *entry {
                    *entry = c;
                }
            }
            best.into_iter().collect()
        },
    }
}

/// Directed BFS iterator yielding `(NodeId, depth)` for nodes in `[min_depth, max_depth]`.
pub struct DirectedBfsIter<'a> {
    graph: &'a Graph,
    queue: VecDeque<(NodeId, u32)>,
    visited: HashSet<NodeId>,
    direction: Direction,
    min_depth: u32,
    max_depth: Option<u32>,
}

impl<'a> DirectedBfsIter<'a> {
    pub fn new(graph: &'a Graph, start: NodeId, direction: Direction, depth: Depth) -> Self {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back((start, 0));
        Self {
            graph,
            queue,
            visited,
            direction,
            min_depth: depth.min_depth(),
            max_depth: depth.max_depth(),
        }
    }
}

impl<'a> Iterator for DirectedBfsIter<'a> {
    type Item = (NodeId, u32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (current, depth) = self.queue.pop_front()?;

            // Expand neighbors if within depth budget
            let should_expand = self.max_depth.is_none_or(|max| depth < max);
            if should_expand {
                for neighbor in self.graph.neighbors_directed(current, self.direction) {
                    if self.visited.insert(neighbor) {
                        self.queue.push_back((neighbor, depth + 1));
                    }
                }
            }

            // Only yield if depth >= min_depth
            if depth >= self.min_depth {
                return Some((current, depth));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.queue.len(), None)
    }
}

/// Directed DFS (pre-order) iterator yielding `(NodeId, depth)`.
pub struct DirectedDfsIter<'a> {
    graph: &'a Graph,
    stack: Vec<(NodeId, u32)>,
    visited: HashSet<NodeId>,
    direction: Direction,
    min_depth: u32,
    max_depth: Option<u32>,
}

impl<'a> DirectedDfsIter<'a> {
    pub fn new(graph: &'a Graph, start: NodeId, direction: Direction, depth: Depth) -> Self {
        let mut visited = HashSet::new();
        visited.insert(start);
        Self {
            graph,
            stack: vec![(start, 0)],
            visited,
            direction,
            min_depth: depth.min_depth(),
            max_depth: depth.max_depth(),
        }
    }
}

impl<'a> Iterator for DirectedDfsIter<'a> {
    type Item = (NodeId, u32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (current, depth) = self.stack.pop()?;

            let should_expand = self.max_depth.is_none_or(|max| depth < max);
            if should_expand {
                for neighbor in self.graph.neighbors_directed(current, self.direction) {
                    if self.visited.insert(neighbor) {
                        self.stack.push((neighbor, depth + 1));
                    }
                }
            }

            if depth >= self.min_depth {
                return Some((current, depth));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }
}

/// Directed Dijkstra iterator yielding `(NodeId, cost)` in cost-ascending order.
pub struct DirectedDijkstraIter<'a> {
    graph: &'a Graph,
    heap: BinaryHeap<PathEntry>,
    dist: FxHashMap<NodeId, f32>,
    direction: Direction,
}

impl<'a> DirectedDijkstraIter<'a> {
    pub fn new(graph: &'a Graph, start: NodeId, direction: Direction) -> Self {
        let mut dist = FxHashMap::default();
        let mut heap = BinaryHeap::new();
        dist.insert(start, 0.0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
        });
        Self {
            graph,
            heap,
            dist,
            direction,
        }
    }
}

impl<'a> Iterator for DirectedDijkstraIter<'a> {
    type Item = (NodeId, f32);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(PathEntry { node_id, cost }) = self.heap.pop() {
            if cost > *self.dist.get(&node_id).unwrap_or(&f32::INFINITY) {
                continue;
            }

            for (neighbor, w) in expand_directed_costs(self.graph, node_id, self.direction) {
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
