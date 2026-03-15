// crates/agent-db-graph/src/structures/graph_persistence.rs
//
// insert_existing_node/edge (preserves original IDs) and delta persistence helpers.

use super::adj_list::AdjList;
use super::edge::GraphEdge;
use super::graph::Graph;
use super::node::{GraphNode, NodeType};
use super::types::EdgeId;

impl Graph {
    /// Insert a node preserving its original ID. Updates all indices.
    /// Advances next_node_id if needed.
    pub fn insert_existing_node(&mut self, node: GraphNode) {
        let node_id = node.id;

        // Update type index (u8 discriminant key)
        self.type_index
            .entry(node.node_type.discriminant())
            .or_default()
            .insert(node_id);

        // Update temporal index
        self.temporal_index
            .entry(node.created_at)
            .or_default()
            .push(node_id);

        // Bump generation for cache invalidation
        self.generation += 1;

        // Update specialized indices
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

        // BM25 indexing
        let mut text_parts = Vec::new();
        match &node.node_type {
            NodeType::Claim { claim_text, .. } => text_parts.push(claim_text.as_str()),
            NodeType::Goal { description, .. } => text_parts.push(description.as_str()),
            NodeType::Strategy { name, .. } => text_parts.push(name.as_str()),
            NodeType::Result { summary, .. } => text_parts.push(summary.as_str()),
            NodeType::Concept { concept_name, .. } => text_parts.push(concept_name.as_str()),
            NodeType::Tool { tool_name, .. } => text_parts.push(tool_name.as_str()),
            NodeType::Episode { outcome, .. } => text_parts.push(outcome.as_str()),
            _ => {},
        }
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
        if !text_parts.is_empty() || !owned_parts.is_empty() {
            let mut combined = text_parts.join(" ");
            if !owned_parts.is_empty() {
                if !combined.is_empty() {
                    combined.push(' ');
                }
                combined.push_str(&owned_parts.join(" "));
            }
            let is_code_content = node
                .properties
                .get("content_type")
                .and_then(|v| v.as_str())
                .map(|v| v == "code")
                .unwrap_or(false);

            if is_code_content || found_code_key {
                self.bm25_index.index_document_code(node_id, &combined);
            } else {
                self.bm25_index.index_document(node_id, &combined);
            }
        }

        // Initialize adjacency if needed (ensure_at is a no-op if already occupied)
        self.adjacency_out.ensure_at(node_id, AdjList::new());
        self.adjacency_in.ensure_at(node_id, AdjList::new());

        self.nodes.insert_at(node_id, node);
        self.dirty_nodes.insert(node_id);
        self.adjacency_dirty = true;
        self.update_stats();
    }

    /// Insert an edge preserving its original ID. Both endpoints must exist.
    pub fn insert_existing_edge(&mut self, edge: GraphEdge) -> Option<EdgeId> {
        if !self.nodes.contains_key(edge.source) || !self.nodes.contains_key(edge.target) {
            return None;
        }

        let edge_id = edge.id;

        self.adjacency_out
            .ensure_at(edge.source, AdjList::new())
            .push(edge_id);
        self.adjacency_in
            .ensure_at(edge.target, AdjList::new())
            .push(edge_id);

        if let Some(source) = self.nodes.get_mut(edge.source) {
            source.degree += 1;
            self.total_degree += 1;
        }
        if let Some(target) = self.nodes.get_mut(edge.target) {
            target.degree += 1;
            self.total_degree += 1;
        }

        let source = edge.source;
        let target = edge.target;
        self.edges.insert_at(edge_id, edge);
        self.dirty_edges.insert(edge_id);
        self.dirty_nodes.insert(source);
        self.dirty_nodes.insert(target);
        self.adjacency_dirty = true;
        self.update_stats();

        Some(edge_id)
    }

    // ========================================================================
    // Delta persistence helpers
    // ========================================================================

    /// Whether there are any pending changes to persist.
    pub fn has_pending_changes(&self) -> bool {
        !self.dirty_nodes.is_empty()
            || !self.dirty_edges.is_empty()
            || !self.deleted_nodes.is_empty()
            || !self.deleted_edges.is_empty()
            || self.adjacency_dirty
    }

    /// Clear all dirty tracking state after a successful persist.
    pub fn clear_dirty(&mut self) {
        self.dirty_nodes.clear();
        self.dirty_edges.clear();
        self.deleted_nodes.clear();
        self.deleted_edges.clear();
        self.adjacency_dirty = false;
    }
}
