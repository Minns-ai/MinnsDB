//! Variable-length path expansion for incremental subscriptions.
//!
//! Tracks all reachable (source, target) pairs via BFS.
//! Uses a reverse index (node → sources that reach it) for efficient
//! edge-add lookup instead of scanning all reachability entries.

use std::collections::{HashSet, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::query_lang::ast::Direction as AstDirection;
use crate::query_lang::executor::{edge_matches_type_standalone, edge_visible_standalone};
use crate::query_lang::planner::{SlotIdx, TemporalViewport};
use crate::structures::{EdgeId, Graph, NodeId};

use super::incremental::{BoundEntityId, RowDelta, RowId, ScanState};

/// Maximum nodes visited during BFS expansion (matches executor's limit).
const MAX_BFS_VISITED: usize = 10_000;

/// Variable-length expand operator state.
pub struct VarLengthExpandState {
    pub from_var: SlotIdx,
    pub to_var: SlotIdx,
    pub edge_type_filter: Option<String>,
    pub direction: AstDirection,
    pub min_hops: u32,
    pub max_hops: u32,

    /// source_node → set of reachable target nodes at valid distances.
    pub reachability: FxHashMap<NodeId, FxHashSet<NodeId>>,

    /// edge_id → set of source nodes whose reachability paths use this edge.
    /// Deduplicated: each (edge, source) pair stored at most once.
    edge_to_sources: FxHashMap<EdgeId, FxHashSet<NodeId>>,

    /// Reverse index: target_node → set of source nodes that can reach it.
    /// Used for efficient EdgeAdded lookup (instead of scanning all reachability).
    node_reached_by: FxHashMap<NodeId, FxHashSet<NodeId>>,
}

impl VarLengthExpandState {
    pub fn init(
        from_var: SlotIdx,
        to_var: SlotIdx,
        edge_type: &Option<String>,
        direction: &AstDirection,
        min_hops: u32,
        max_hops: Option<u32>,
        scan_state: &ScanState,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Self {
        let max_hops = max_hops.unwrap_or(10);
        let mut state = Self {
            from_var,
            to_var,
            edge_type_filter: edge_type.clone(),
            direction: direction.clone(),
            min_hops,
            max_hops,
            reachability: FxHashMap::default(),
            edge_to_sources: FxHashMap::default(),
            node_reached_by: FxHashMap::default(),
        };

        for &source in &scan_state.active_nodes {
            state.expand_from_source(source, graph, viewport, txn_cutoff);
        }

        state
    }

    /// BFS expand from a source node. Records reachability, edge usage, and reverse index.
    fn expand_from_source(
        &mut self,
        source: NodeId,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) {
        let mut visited: HashSet<NodeId> = HashSet::new();
        visited.insert(source);

        let mut queue: VecDeque<(NodeId, u32)> = VecDeque::new();
        queue.push_back((source, 0));

        let targets = self.reachability.entry(source).or_default();

        while let Some((node, depth)) = queue.pop_front() {
            if visited.len() > MAX_BFS_VISITED {
                break;
            }
            if depth >= self.max_hops {
                continue;
            }

            let edges = match self.direction {
                AstDirection::Out => graph.get_edges_from(node),
                AstDirection::In => graph.get_edges_to(node),
            };

            for edge in edges {
                if !edge_visible_standalone(edge, viewport, txn_cutoff) {
                    continue;
                }
                if !edge_matches_type_standalone(edge, &self.edge_type_filter) {
                    continue;
                }

                let next = match self.direction {
                    AstDirection::Out => edge.target,
                    AstDirection::In => edge.source,
                };

                let new_depth = depth + 1;

                // Record edge usage (deduplicated via FxHashSet).
                self.edge_to_sources
                    .entry(edge.id)
                    .or_default()
                    .insert(source);

                if new_depth >= self.min_hops {
                    targets.insert(next);
                    // Reverse index: next is reachable from source.
                    self.node_reached_by.entry(next).or_default().insert(source);
                }

                if !visited.contains(&next) {
                    visited.insert(next);
                    queue.push_back((next, new_depth));
                }
            }
        }
    }

    /// Remove all index entries for a source node.
    fn remove_source(&mut self, source: NodeId) {
        // Remove from reachability.
        if let Some(targets) = self.reachability.remove(&source) {
            // Clean reverse index.
            for target in &targets {
                if let Some(sources) = self.node_reached_by.get_mut(target) {
                    sources.remove(&source);
                    if sources.is_empty() {
                        self.node_reached_by.remove(target);
                    }
                }
            }
        }
        // Clean edge_to_sources.
        self.edge_to_sources.retain(|_, sources| {
            sources.remove(&source);
            !sources.is_empty()
        });
    }

    /// Re-expand a source and diff against old targets. Returns row deltas.
    fn reexpand_and_diff(
        &mut self,
        source: NodeId,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Vec<RowDelta> {
        let old_targets: FxHashSet<NodeId> = self.reachability.remove(&source).unwrap_or_default();

        // Clean indexes for this source before re-expanding.
        for target in &old_targets {
            if let Some(sources) = self.node_reached_by.get_mut(target) {
                sources.remove(&source);
            }
        }
        self.edge_to_sources.retain(|_, sources| {
            sources.remove(&source);
            !sources.is_empty()
        });

        self.expand_from_source(source, graph, viewport, txn_cutoff);

        let new_targets = self.reachability.get(&source);
        let mut out = Vec::new();

        if let Some(new) = new_targets {
            for &t in new {
                if !old_targets.contains(&t) {
                    out.push(RowDelta::insert(self.make_row_id(source, t)));
                }
            }
        }
        for t in &old_targets {
            if !new_targets.is_some_and(|n| n.contains(t)) {
                out.push(RowDelta::delete(self.make_row_id(source, *t)));
            }
        }

        out
    }

    fn make_row_id(&self, from_node: NodeId, to_node: NodeId) -> RowId {
        RowId::new(smallvec::smallvec![
            (self.from_var, BoundEntityId::Node(from_node)),
            (self.to_var, BoundEntityId::Node(to_node)),
        ])
    }

    pub fn apply_upstream_deltas(
        &mut self,
        scan_deltas: &[RowDelta],
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Vec<RowDelta> {
        let mut out = Vec::new();

        for delta in scan_deltas {
            match delta {
                RowDelta::Insert { row_id } => {
                    if let Some(BoundEntityId::Node(source)) = row_id.get(self.from_var) {
                        let source = *source;
                        self.expand_from_source(source, graph, viewport, txn_cutoff);
                        if let Some(targets) = self.reachability.get(&source) {
                            for &target in targets {
                                out.push(RowDelta::insert(self.make_row_id(source, target)));
                            }
                        }
                    }
                },
                RowDelta::Delete { row_id } => {
                    if let Some(BoundEntityId::Node(source)) = row_id.get(self.from_var) {
                        let source = *source;
                        if let Some(targets) = self.reachability.get(&source) {
                            for &target in targets.iter() {
                                out.push(RowDelta::delete(self.make_row_id(source, target)));
                            }
                        }
                        self.remove_source(source);
                    }
                },
            }
        }

        out
    }

    pub fn apply_edge_delta(
        &mut self,
        delta: &super::delta::GraphDelta,
        scan_state: &ScanState,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Vec<RowDelta> {
        use super::delta::GraphDelta;

        match delta {
            GraphDelta::EdgeAdded { source, target, .. } => {
                let (from, _to) = match self.direction {
                    AstDirection::Out => (*source, *target),
                    AstDirection::In => (*target, *source),
                };

                // Use reverse index: find sources that already reach `from`.
                let mut sources_to_reexpand: FxHashSet<NodeId> =
                    self.node_reached_by.get(&from).cloned().unwrap_or_default();

                // Also check if `from` is itself an active source.
                if scan_state.active_nodes.contains(&from) {
                    sources_to_reexpand.insert(from);
                }

                if sources_to_reexpand.is_empty() {
                    return Vec::new();
                }

                let mut out = Vec::new();
                for source in sources_to_reexpand {
                    out.extend(self.reexpand_and_diff(source, graph, viewport, txn_cutoff));
                }
                out
            },
            GraphDelta::EdgeRemoved { edge_id, .. } => {
                let affected_sources: FxHashSet<NodeId> =
                    self.edge_to_sources.remove(edge_id).unwrap_or_default();

                if affected_sources.is_empty() {
                    return Vec::new();
                }

                let mut out = Vec::new();
                for source in affected_sources {
                    out.extend(self.reexpand_and_diff(source, graph, viewport, txn_cutoff));
                }
                out
            },
            GraphDelta::EdgeSuperseded {
                edge_id,
                source,
                target,
                ..
            } => {
                if let Some(edge) = graph.get_edge(*edge_id) {
                    let is_visible = edge_visible_standalone(edge, viewport, txn_cutoff);
                    let was_used = self.edge_to_sources.contains_key(edge_id);

                    if was_used && !is_visible {
                        // Edge left viewport — re-expand affected sources.
                        let affected = self.edge_to_sources.remove(edge_id).unwrap_or_default();
                        let mut out = Vec::new();
                        for src in affected {
                            out.extend(self.reexpand_and_diff(src, graph, viewport, txn_cutoff));
                        }
                        return out;
                    } else if !was_used && is_visible {
                        // Edge entered viewport — treat as addition.
                        return self.apply_edge_delta(
                            &GraphDelta::EdgeAdded {
                                edge_id: *edge_id,
                                source: *source,
                                target: *target,
                                edge_type_tag: String::new(),
                                generation: 0,
                            },
                            scan_state,
                            graph,
                            viewport,
                            txn_cutoff,
                        );
                    }
                }
                Vec::new()
            },
            _ => Vec::new(),
        }
    }
}
