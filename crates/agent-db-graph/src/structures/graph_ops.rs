// crates/agent-db-graph/src/structures/graph_ops.rs
//
// Graph mutation methods: add_node, add_edge, remove_edge, remove_node,
// merge_nodes, reindex_node_with_edges, invalidate_edge.

use rustc_hash::FxHashMap;

use super::adj_list::AdjList;
use super::edge::EdgeType;
use super::graph::{edge_text_for_bm25, Graph};
use super::node::{GraphNode, NodeType};
use super::types::{EdgeId, NodeId};
use crate::subscription::delta::{DeltaBatch, GraphDelta};

/// Extract a compact string tag from an EdgeType for delta emission.
fn edge_type_tag(edge_type: &EdgeType) -> String {
    match edge_type {
        EdgeType::Association {
            association_type, ..
        } => association_type.clone(),
        EdgeType::Causality { .. } => "Causality".to_string(),
        EdgeType::Temporal { .. } => "Temporal".to_string(),
        EdgeType::Contextual { .. } => "Contextual".to_string(),
        EdgeType::Interaction { .. } => "Interaction".to_string(),
        EdgeType::GoalRelation { .. } => "GoalRelation".to_string(),
        EdgeType::Communication { .. } => "Communication".to_string(),
        EdgeType::DerivedFrom { .. } => "DerivedFrom".to_string(),
        EdgeType::SupportedBy { .. } => "SupportedBy".to_string(),
        EdgeType::CodeStructure { relation_kind, .. } => relation_kind.clone(),
        EdgeType::About {
            predicate: Some(p), ..
        } => p.clone(),
        EdgeType::About { .. } => "About".to_string(),
    }
}

impl Graph {
    /// Add a node to the graph.
    ///
    /// Returns `GraphError::CapacityExceeded` when `nodes.len() >= max_graph_size`.
    /// Let the graph pruner free space before retrying.
    pub fn add_node(&mut self, mut node: GraphNode) -> crate::GraphResult<NodeId> {
        if self.nodes.len() >= self.max_graph_size {
            return Err(crate::GraphError::CapacityExceeded(format!(
                "graph has {} nodes (max {})",
                self.nodes.len(),
                self.max_graph_size
            )));
        }

        // SlotVec manages ID assignment (monotonic, starts at 1).
        // We peek at next_id and then insert to get that ID.
        let node_id = self.nodes.next_id();

        node.id = node_id;

        // Capture discriminant before potential move
        let disc = node.node_type.discriminant();

        // Update type index (u8 discriminant key)
        self.type_index.entry(disc).or_default().insert(node_id);

        // Update temporal index
        self.temporal_index
            .entry(node.created_at)
            .or_default()
            .push(node_id);

        // Bump generation for cache invalidation
        self.generation += 1;

        // Update specialized indices
        self.update_specialized_indices_for_node(&node);

        // Index text content with BM25 for full-text search
        self.index_node_bm25(&node);

        let inserted_id = self.nodes.insert(node);
        debug_assert_eq!(inserted_id, node_id);
        // Initialize adjacency lists for this node
        self.adjacency_out.insert_at(node_id, AdjList::new());
        self.adjacency_in.insert_at(node_id, AdjList::new());
        self.dirty_nodes.insert(node_id);
        self.adjacency_dirty = true;
        self.update_stats();

        // Emit subscription delta
        if let Some(tx) = &self.delta_tx {
            let _ = tx.send(DeltaBatch {
                deltas: vec![GraphDelta::NodeAdded {
                    node_id,
                    node_type_disc: disc,
                    generation: self.generation,
                }],
                generation_range: (self.generation, self.generation),
            });
        }

        Ok(node_id)
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, mut edge: super::edge::GraphEdge) -> Option<EdgeId> {
        // Verify both nodes exist
        if !self.nodes.contains_key(edge.source) || !self.nodes.contains_key(edge.target) {
            return None;
        }

        let edge_id = self.edges.next_id();

        edge.id = edge_id;

        // Update adjacency lists
        self.adjacency_out.get_mut(edge.source)?.push(edge_id);
        self.adjacency_in.get_mut(edge.target)?.push(edge_id);

        // Update node degrees
        if let Some(source_node) = self.nodes.get_mut(edge.source) {
            source_node.degree += 1;
            self.total_degree += 1;
            source_node.touch();
        }
        if let Some(target_node) = self.nodes.get_mut(edge.target) {
            target_node.degree += 1;
            self.total_degree += 1;
            target_node.touch();
        }

        let source = edge.source;
        let target = edge.target;
        // Defer edge_type_tag computation to avoid String allocation when no subscribers.
        let tag = if self.delta_tx.is_some() {
            Some(edge_type_tag(&edge.edge_type))
        } else {
            None
        };
        let inserted_id = self.edges.insert(edge);
        debug_assert_eq!(inserted_id, edge_id);
        self.dirty_edges.insert(edge_id);
        self.dirty_nodes.insert(source);
        self.dirty_nodes.insert(target);
        self.adjacency_dirty = true;

        // Index edge metadata into BM25 for the source node only.
        // The source's BM25 document includes outgoing edge text, which is
        // sufficient for relationship search. Reindexing both endpoints on
        // every edge add was O(degree) x 2; this halves the cost.
        if let Some(edge_ref) = self.edges.get(edge_id) {
            let edge_text = edge_text_for_bm25(edge_ref);
            if !edge_text.is_empty() {
                self.reindex_node_with_edges(source);
            }
        }

        // Bump generation for cache invalidation
        self.generation += 1;

        // Emit subscription delta
        if let Some(tx) = &self.delta_tx {
            let _ = tx.send(DeltaBatch {
                deltas: vec![GraphDelta::EdgeAdded {
                    edge_id,
                    source,
                    target,
                    edge_type_tag: tag.unwrap(),
                    generation: self.generation,
                }],
                generation_range: (self.generation, self.generation),
            });
        }

        self.update_stats();

        Some(edge_id)
    }

    /// Rebuild BM25 index for a node, including text from all its edges.
    pub(crate) fn reindex_node_with_edges(&mut self, node_id: NodeId) {
        // Use a single String buffer instead of Vec<String> + join to avoid
        // intermediate allocations. Pre-allocate with a reasonable capacity.
        let mut combined = String::with_capacity(512);

        // Collect node text
        if let Some(node) = self.nodes.get(node_id) {
            let primary_text = match &node.node_type {
                NodeType::Claim { claim_text, .. } => Some(claim_text.as_str()),
                NodeType::Goal { description, .. } => Some(description.as_str()),
                NodeType::Strategy { name, .. } => Some(name.as_str()),
                NodeType::Result { summary, .. } => Some(summary.as_str()),
                NodeType::Concept { concept_name, .. } => Some(concept_name.as_str()),
                NodeType::Tool { tool_name, .. } => Some(tool_name.as_str()),
                NodeType::Episode { outcome, .. } => Some(outcome.as_str()),
                _ => None,
            };
            if let Some(text) = primary_text {
                combined.push_str(text);
            }

            // Extract searchable text from properties (avoid to_lowercase per key
            // by checking ASCII bytes directly for common patterns)
            for (key, value) in &node.properties {
                if Self::is_searchable_key(key) {
                    if let Some(text) = value.as_str() {
                        if !combined.is_empty() {
                            combined.push(' ');
                        }
                        combined.push_str(text);
                    } else {
                        let flat = Self::flatten_json_to_text(value);
                        if !flat.is_empty() {
                            if !combined.is_empty() {
                                combined.push(' ');
                            }
                            combined.push_str(&flat);
                        }
                    }
                }
            }
        }

        // Collect edge text from outgoing edges only (incoming edges are
        // covered by the source node's BM25 document for the reverse direction)
        if let Some(out_edges) = self.adjacency_out.get(node_id) {
            for &eid in out_edges.iter() {
                if let Some(edge) = self.edges.get(eid) {
                    let et = edge_text_for_bm25(edge);
                    if !et.is_empty() {
                        if !combined.is_empty() {
                            combined.push(' ');
                        }
                        combined.push_str(&et);
                    }
                }
            }
        }

        if !combined.is_empty() {
            self.bm25_index.remove_document(node_id);
            self.bm25_index.index_document(node_id, &combined);
        }
    }

    /// Fast check whether a property key is searchable. Zero allocations.
    /// Uses case-insensitive byte window matching directly on the key bytes.
    #[inline]
    fn is_searchable_key(key: &str) -> bool {
        let k = key.as_bytes();
        // Exact matches (most common, fastest path)
        if k.eq_ignore_ascii_case(b"data")
            || k.eq_ignore_ascii_case(b"category")
            || k.eq_ignore_ascii_case(b"metadata_text")
        {
            return true;
        }
        // Substring patterns via case-insensitive window scan (no allocation)
        Self::contains_ascii_ci(k, b"text")
            || Self::contains_ascii_ci(k, b"name")
            || Self::contains_ascii_ci(k, b"content")
            || Self::contains_ascii_ci(k, b"summary")
            || Self::contains_ascii_ci(k, b"description")
    }

    /// Case-insensitive ASCII substring check without allocation.
    #[inline]
    fn contains_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.len() > haystack.len() {
            return false;
        }
        haystack
            .windows(needle.len())
            .any(|w| w.eq_ignore_ascii_case(needle))
    }

    /// Remove an edge by ID. Returns the removed edge, or None if not found.
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> Option<super::edge::GraphEdge> {
        let edge = self.edges.remove(edge_id)?;

        // Remove from source's outgoing adjacency list
        if let Some(out_list) = self.adjacency_out.get_mut(edge.source) {
            out_list.retain(|eid| *eid != edge_id);
        }
        // Remove from target's incoming adjacency list
        if let Some(in_list) = self.adjacency_in.get_mut(edge.target) {
            in_list.retain(|eid| *eid != edge_id);
        }

        // Update degrees
        if let Some(source) = self.nodes.get_mut(edge.source) {
            source.degree = source.degree.saturating_sub(1);
            self.total_degree = self.total_degree.saturating_sub(1);
            self.dirty_nodes.insert(edge.source);
        }
        if let Some(target) = self.nodes.get_mut(edge.target) {
            target.degree = target.degree.saturating_sub(1);
            self.total_degree = self.total_degree.saturating_sub(1);
            self.dirty_nodes.insert(edge.target);
        }

        // Track deletion for delta persistence
        self.deleted_edges.insert(edge_id);
        self.dirty_edges.remove(&edge_id);
        self.adjacency_dirty = true;
        self.generation += 1;

        // Emit subscription delta
        if let Some(tx) = &self.delta_tx {
            let _ = tx.send(DeltaBatch {
                deltas: vec![GraphDelta::EdgeRemoved {
                    edge_id,
                    source: edge.source,
                    target: edge.target,
                    edge_type_tag: edge_type_tag(&edge.edge_type),
                    generation: self.generation,
                }],
                generation_range: (self.generation, self.generation),
            });
        }

        self.update_stats();
        Some(edge)
    }

    /// Remove a node and all its incident edges from the graph.
    ///
    /// Cleans up: nodes map, type_index, all 11 specialized indices,
    /// bm25_index, outgoing edges, incoming edges, adjacency lists,
    /// degree updates on neighbors, and stats.
    pub fn remove_node(&mut self, node_id: NodeId) -> Option<GraphNode> {
        let node = self.nodes.remove(node_id)?;

        // Subtract the removed node's degree from the running total
        self.total_degree = self.total_degree.saturating_sub(node.degree as u64);

        // Remove from type index (u8 discriminant key)
        let disc = node.node_type.discriminant();
        if let Some(set) = self.type_index.get_mut(&disc) {
            set.remove(&node_id);
            if set.is_empty() {
                self.type_index.remove(&disc);
            }
        }

        // Remove from temporal index
        if let Some(ids) = self.temporal_index.get_mut(&node.created_at) {
            ids.retain(|nid| *nid != node_id);
            if ids.is_empty() {
                self.temporal_index.remove(&node.created_at);
            }
        }

        // Bump generation for cache invalidation
        self.generation += 1;

        // Remove from specialized indices
        match &node.node_type {
            NodeType::Agent { agent_id, .. } => {
                self.agent_index.remove(agent_id);
            },
            NodeType::Event { event_id, .. } => {
                self.event_index.remove(event_id);
            },
            NodeType::Context { context_hash, .. } => {
                self.context_index.remove(context_hash);
            },
            NodeType::Goal { goal_id, .. } => {
                self.goal_index.remove(goal_id);
            },
            NodeType::Episode { episode_id, .. } => {
                self.episode_index.remove(episode_id);
            },
            NodeType::Memory { memory_id, .. } => {
                self.memory_index.remove(memory_id);
            },
            NodeType::Strategy { strategy_id, .. } => {
                self.strategy_index.remove(strategy_id);
            },
            NodeType::Tool { tool_name, .. } => {
                self.tool_index.remove(tool_name.as_str());
            },
            NodeType::Result { result_key, .. } => {
                self.result_index.remove(result_key.as_str());
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.remove(claim_id);
            },
            NodeType::Concept { concept_name, .. } => {
                self.concept_index.remove(concept_name.as_str());
            },
        }

        // Remove from BM25 index
        self.bm25_index.remove_document(node_id);

        // Collect edge IDs to remove (outgoing + incoming)
        let outgoing_edge_ids = self.adjacency_out.remove(node_id).unwrap_or_default();
        let incoming_edge_ids = self.adjacency_in.remove(node_id).unwrap_or_default();

        // Accumulate edge removal deltas for subscription broadcast
        let has_subs = self.delta_tx.is_some();
        let mut edge_deltas: Vec<GraphDelta> = Vec::new();

        // Remove outgoing edges and update targets' incoming adjacency + degree
        for edge_id in &outgoing_edge_ids {
            if let Some(edge) = self.edges.remove(*edge_id) {
                if has_subs {
                    edge_deltas.push(GraphDelta::EdgeRemoved {
                        edge_id: *edge_id,
                        source: edge.source,
                        target: edge.target,
                        edge_type_tag: edge_type_tag(&edge.edge_type),
                        generation: self.generation,
                    });
                }
                self.deleted_edges.insert(*edge_id);
                self.dirty_edges.remove(edge_id);
                if edge.target != node_id {
                    if let Some(in_list) = self.adjacency_in.get_mut(edge.target) {
                        in_list.retain(|eid| *eid != *edge_id);
                    }
                    if let Some(target) = self.nodes.get_mut(edge.target) {
                        target.degree = target.degree.saturating_sub(1);
                        self.total_degree = self.total_degree.saturating_sub(1);
                        self.dirty_nodes.insert(edge.target);
                    }
                }
            }
        }

        // Remove incoming edges and update sources' outgoing adjacency + degree
        for edge_id in &incoming_edge_ids {
            if let Some(edge) = self.edges.remove(*edge_id) {
                if has_subs {
                    edge_deltas.push(GraphDelta::EdgeRemoved {
                        edge_id: *edge_id,
                        source: edge.source,
                        target: edge.target,
                        edge_type_tag: edge_type_tag(&edge.edge_type),
                        generation: self.generation,
                    });
                }
                self.deleted_edges.insert(*edge_id);
                self.dirty_edges.remove(edge_id);
                if edge.source != node_id {
                    if let Some(out_list) = self.adjacency_out.get_mut(edge.source) {
                        out_list.retain(|eid| *eid != *edge_id);
                    }
                    if let Some(source) = self.nodes.get_mut(edge.source) {
                        source.degree = source.degree.saturating_sub(1);
                        self.total_degree = self.total_degree.saturating_sub(1);
                        self.dirty_nodes.insert(edge.source);
                    }
                }
            }
        }

        // Track deletion for delta persistence
        self.deleted_nodes.insert(node_id);
        self.dirty_nodes.remove(&node_id);
        self.adjacency_dirty = true;

        // Emit subscription deltas: NodeRemoved + cascaded EdgeRemoved
        if let Some(tx) = &self.delta_tx {
            let mut deltas = vec![GraphDelta::NodeRemoved {
                node_id,
                node_type_disc: disc,
                generation: self.generation,
            }];
            deltas.append(&mut edge_deltas);
            let _ = tx.send(DeltaBatch {
                generation_range: (self.generation, self.generation),
                deltas,
            });
        }

        self.update_stats();
        Some(node)
    }

    /// Merge two nodes of the same type variant. The survivor keeps its data
    /// with a signal boost; all edges from the absorbed node are redirected
    /// to the survivor (strengthening existing, skipping self-loops).
    /// Returns the updated survivor node.
    pub fn merge_nodes(
        &mut self,
        survivor_id: NodeId,
        absorbed_id: NodeId,
    ) -> Result<GraphNode, String> {
        // Verify both exist
        if !self.nodes.contains_key(survivor_id) {
            return Err(format!("Survivor node {} not found", survivor_id));
        }
        if !self.nodes.contains_key(absorbed_id) {
            return Err(format!("Absorbed node {} not found", absorbed_id));
        }

        // Same variant check
        {
            let survivor = self.nodes.get(survivor_id).unwrap();
            let absorbed = self.nodes.get(absorbed_id).unwrap();
            if std::mem::discriminant(&survivor.node_type)
                != std::mem::discriminant(&absorbed.node_type)
            {
                return Err(format!(
                    "Cannot merge different node types: {} vs {}",
                    survivor.type_name(),
                    absorbed.type_name()
                ));
            }
        }

        // Collect absorbed edges before mutation
        let absorbed_out = self
            .adjacency_out
            .get(absorbed_id)
            .cloned()
            .unwrap_or_default();
        let absorbed_in = self
            .adjacency_in
            .get(absorbed_id)
            .cloned()
            .unwrap_or_default();

        // Build O(1) lookup maps of the survivor's existing edge targets/sources.
        // This replaces the O(degree_absorbed * degree_survivor) linear scan
        // with O(degree_absorbed + degree_survivor).
        let mut survivor_out_targets: FxHashMap<NodeId, EdgeId> = FxHashMap::default();
        if let Some(out_ids) = self.adjacency_out.get(survivor_id) {
            for &eid in out_ids.iter() {
                if let Some(edge) = self.edges.get(eid) {
                    survivor_out_targets.insert(edge.target, eid);
                }
            }
        }
        let mut survivor_in_sources: FxHashMap<NodeId, EdgeId> = FxHashMap::default();
        if let Some(in_ids) = self.adjacency_in.get(survivor_id) {
            for &eid in in_ids.iter() {
                if let Some(edge) = self.edges.get(eid) {
                    survivor_in_sources.insert(edge.source, eid);
                }
            }
        }

        // Redirect outgoing edges: absorbed -> X becomes survivor -> X
        let mut out_redirects: Vec<(EdgeId, NodeId, f32)> = Vec::new();
        for edge_id in &absorbed_out {
            if let Some(edge) = self.edges.get(*edge_id) {
                let target = edge.target;
                if target == survivor_id || target == absorbed_id {
                    continue;
                }
                out_redirects.push((*edge_id, target, edge.weight));
            }
        }

        for (edge_id, target, weight) in out_redirects {
            // O(1) lookup instead of linear scan
            if let Some(&existing_eid) = survivor_out_targets.get(&target) {
                if let Some(existing_edge) = self.edges.get_mut(existing_eid) {
                    existing_edge.strengthen(weight * 0.5);
                }
            } else {
                if let Some(edge) = self.edges.get_mut(edge_id) {
                    edge.source = survivor_id;
                }
                self.adjacency_out
                    .ensure_at(survivor_id, AdjList::new())
                    .push(edge_id);
                survivor_out_targets.insert(target, edge_id);
            }
        }

        // Redirect incoming edges: X -> absorbed becomes X -> survivor
        let mut in_redirects: Vec<(EdgeId, NodeId, f32)> = Vec::new();
        for edge_id in &absorbed_in {
            if let Some(edge) = self.edges.get(*edge_id) {
                let source = edge.source;
                if source == survivor_id || source == absorbed_id {
                    continue;
                }
                in_redirects.push((*edge_id, source, edge.weight));
            }
        }

        for (edge_id, source, weight) in in_redirects {
            // O(1) lookup instead of linear scan
            if let Some(&existing_eid) = survivor_in_sources.get(&source) {
                if let Some(existing_edge) = self.edges.get_mut(existing_eid) {
                    existing_edge.strengthen(weight * 0.5);
                }
            } else {
                if let Some(edge) = self.edges.get_mut(edge_id) {
                    edge.target = survivor_id;
                }
                self.adjacency_in
                    .ensure_at(survivor_id, AdjList::new())
                    .push(edge_id);
                survivor_in_sources.insert(source, edge_id);
            }
        }

        // Boost survivor's signal via properties
        if let Some(survivor) = self.nodes.get_mut(survivor_id) {
            let current_boost = survivor
                .properties
                .get("merge_boost")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            survivor.properties.insert(
                "merge_boost".to_string(),
                serde_json::Value::from(current_boost + 0.1),
            );
            survivor.touch();
        }

        // Remove absorbed node (this cleans up all remaining references)
        self.remove_node(absorbed_id);

        // Update survivor degree and total_degree
        if let Some(survivor) = self.nodes.get_mut(survivor_id) {
            let old_degree = survivor.degree as u64;
            let out_count = self.adjacency_out.get(survivor_id).map_or(0, |v| v.len());
            let in_count = self.adjacency_in.get(survivor_id).map_or(0, |v| v.len());
            survivor.degree = (out_count + in_count) as u32;
            let new_degree = survivor.degree as u64;
            // Adjust total_degree for the difference
            self.total_degree = self.total_degree.saturating_sub(old_degree) + new_degree;
        }

        // Emit subscription delta for node merge
        if let Some(tx) = &self.delta_tx {
            let _ = tx.send(DeltaBatch {
                deltas: vec![GraphDelta::NodeMerged {
                    survivor_id,
                    absorbed_id,
                    generation: self.generation,
                }],
                generation_range: (self.generation, self.generation),
            });
        }

        Ok(self.nodes.get(survivor_id).unwrap().clone())
    }

    /// Soft-delete an edge by ID, marking it invalid with a reason.
    pub fn invalidate_edge(&mut self, edge_id: EdgeId, reason: &str) -> bool {
        if let Some(edge) = self.edges.get_mut(edge_id) {
            // Capture pre-mutation state for delta emission
            let old_valid_until = edge.valid_until;
            let source = edge.source;
            let target = edge.target;
            // Defer tag computation to avoid allocation when no subscribers
            let tag = if self.delta_tx.is_some() {
                Some(edge_type_tag(&edge.edge_type))
            } else {
                None
            };

            edge.invalidate(reason);

            let new_valid_until = edge.valid_until;

            self.dirty_edges.insert(edge_id);
            self.generation += 1;

            // Emit subscription delta
            if let Some(tx) = &self.delta_tx {
                let tag = tag.unwrap();
                let delta = if old_valid_until != new_valid_until {
                    GraphDelta::EdgeSuperseded {
                        edge_id,
                        source,
                        target,
                        edge_type_tag: tag,
                        old_valid_until,
                        new_valid_until,
                        generation: self.generation,
                    }
                } else {
                    GraphDelta::EdgeMutated {
                        edge_id,
                        source,
                        target,
                        edge_type_tag: tag,
                        generation: self.generation,
                    }
                };
                let _ = tx.send(DeltaBatch {
                    deltas: vec![delta],
                    generation_range: (self.generation, self.generation),
                });
            }

            true
        } else {
            false
        }
    }

    /// Helper: update all specialized indices for a node being added.
    fn update_specialized_indices_for_node(&mut self, node: &GraphNode) {
        let node_id = node.id;
        match &node.node_type {
            NodeType::Agent { agent_id, .. } => {
                self.agent_index.insert(*agent_id, node_id);
            },
            NodeType::Event { event_id, .. } => {
                self.event_index.insert(*event_id, node_id);
            },
            NodeType::Context { context_hash, .. } => {
                self.context_index.insert(*context_hash, node_id);
            },
            NodeType::Goal { goal_id, .. } => {
                self.goal_index.insert(*goal_id, node_id);
            },
            NodeType::Episode { episode_id, .. } => {
                self.episode_index.insert(*episode_id, node_id);
            },
            NodeType::Memory { memory_id, .. } => {
                self.memory_index.insert(*memory_id, node_id);
            },
            NodeType::Strategy { strategy_id, .. } => {
                self.strategy_index.insert(*strategy_id, node_id);
            },
            NodeType::Tool { tool_name, .. } => {
                let key = self.interner.intern(tool_name);
                self.tool_index.insert(key, node_id);
            },
            NodeType::Result { result_key, .. } => {
                let key = self.interner.intern(result_key);
                self.result_index.insert(key, node_id);
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.insert(*claim_id, node_id);
            },
            NodeType::Concept { concept_name, .. } => {
                let key = self.interner.intern(concept_name);
                self.concept_index.insert(key, node_id);
            },
        }
    }

    /// Helper: index a node's text content into BM25.
    fn index_node_bm25(&mut self, node: &GraphNode) {
        let node_id = node.id;
        let mut text_parts = Vec::new();

        // Extract searchable text from NodeType fields
        match &node.node_type {
            NodeType::Claim { claim_text, .. } => {
                text_parts.push(claim_text.as_str());
            },
            NodeType::Goal { description, .. } => {
                text_parts.push(description.as_str());
            },
            NodeType::Strategy { name, .. } => {
                text_parts.push(name.as_str());
            },
            NodeType::Result { summary, .. } => {
                text_parts.push(summary.as_str());
            },
            NodeType::Concept { concept_name, .. } => {
                text_parts.push(concept_name.as_str());
            },
            NodeType::Tool { tool_name, .. } => {
                text_parts.push(tool_name.as_str());
            },
            NodeType::Episode { outcome, .. } => {
                text_parts.push(outcome.as_str());
            },
            _ => {}, // Other node types don't have text in NodeType
        }

        // Extract searchable text from common property keys
        let mut found_code_key = false;
        let mut owned_parts: Vec<String> = Vec::new();
        for (key, value) in &node.properties {
            let key_lower = key.to_lowercase();
            if key_lower.contains("text")
                || key_lower.contains("description")
                || key_lower.contains("content")
                || key_lower.contains("name")
                || key_lower.contains("summary")
                || key_lower == "data"
                || key_lower == "code"
                || key_lower == "source"
                || key_lower == "source_code"
                || key_lower == "snippet"
                || key_lower == "body"
                || key_lower == "function_name"
                || key_lower == "class_name"
                || key_lower == "message"
                || key_lower == "query"
                || key_lower == "result"
                || key_lower == "error"
                || key_lower == "prompt"
                || key_lower == "answer"
                || key_lower == "response"
                || key_lower == "output"
                || key_lower == "category"
                || key_lower == "metadata_text"
            {
                // Track whether a code-specific key was found
                if matches!(
                    key_lower.as_str(),
                    "code" | "source" | "source_code" | "snippet" | "function_name" | "class_name"
                ) {
                    found_code_key = true;
                }
                if let Some(text) = value.as_str() {
                    text_parts.push(text);
                } else {
                    // Flatten nested JSON objects/arrays into searchable text
                    let flat = Self::flatten_json_to_text(value);
                    if !flat.is_empty() {
                        owned_parts.push(flat);
                    }
                }
            }
        }

        // Index combined text if available, routing to code or natural tokenizer
        if !text_parts.is_empty() || !owned_parts.is_empty() {
            let mut combined_text = text_parts.join(" ");
            if !owned_parts.is_empty() {
                if !combined_text.is_empty() {
                    combined_text.push(' ');
                }
                combined_text.push_str(&owned_parts.join(" "));
            }
            let is_code_content = node
                .properties
                .get("content_type")
                .and_then(|v| v.as_str())
                .map(|v| v == "code")
                .unwrap_or(false);

            if is_code_content || found_code_key {
                self.bm25_index.index_document_code(node_id, &combined_text);
            } else {
                self.bm25_index.index_document(node_id, &combined_text);
            }
        }
    }
}
