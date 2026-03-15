// crates/agent-db-graph/src/structures/vector_index.rs
//
// Brute-force vector index for graph node embeddings.

use super::types::NodeId;

/// Entry in the node vector index.
#[derive(Debug, Clone)]
struct NodeVectorEntry {
    node_id: NodeId,
    embedding: Vec<f32>,
}

/// Brute-force vector index for graph node embeddings.
///
/// Used for semantic entity resolution and hybrid search.
/// At small scale (<10k nodes), brute-force is fast enough.
#[derive(Debug, Clone, Default)]
pub struct NodeVectorIndex {
    entries: Vec<NodeVectorEntry>,
}

impl NodeVectorIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Number of entries in the index.
    pub fn entries_count(&self) -> usize {
        self.entries.len()
    }

    /// Insert or update a node's embedding.
    pub fn insert(&mut self, node_id: NodeId, embedding: Vec<f32>) {
        if embedding.is_empty() {
            return;
        }
        // Update existing if present
        for entry in &mut self.entries {
            if entry.node_id == node_id {
                entry.embedding = embedding;
                return;
            }
        }
        self.entries.push(NodeVectorEntry { node_id, embedding });
    }

    /// Search for the top-k most similar nodes by cosine similarity.
    pub fn search(&self, query: &[f32], top_k: usize, min_sim: f32) -> Vec<(NodeId, f32)> {
        if query.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        let query_norm = dot(query, query).sqrt();
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scored: Vec<(NodeId, f32)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let entry_norm = dot(&entry.embedding, &entry.embedding).sqrt();
                if entry_norm == 0.0 {
                    return None;
                }
                let sim = dot(query, &entry.embedding) / (query_norm * entry_norm);
                if sim >= min_sim {
                    Some((entry.node_id, sim))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Dot product of two f32 slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
