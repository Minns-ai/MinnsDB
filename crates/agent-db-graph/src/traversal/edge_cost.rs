//! Edge cost derivation from edge types and safety bound constants.

use crate::structures::{EdgeType, Graph, GraphEdge, NodeId};

// ============================================================================
// Safety bounds — prevent OOM and stack overflow on large/dense graphs
// ============================================================================

/// Maximum number of nodes collected from a single BFS/DFS/subgraph traversal.
pub(crate) const MAX_TRAVERSAL_NODES: usize = 50_000;

/// Maximum number of edges collected from subgraph extraction.
pub(crate) const MAX_TRAVERSAL_EDGES: usize = 100_000;

/// Maximum iteration count for bidirectional Dijkstra before forced termination.
pub(crate) const MAX_DIJKSTRA_ITERATIONS: usize = 200_000;

/// Maximum PageRank iterations (even if caller passes higher).
pub(crate) const MAX_PAGERANK_ITERATIONS: usize = 200;

/// Maximum number of nodes for PageRank (skip on huge graphs).
pub(crate) const MAX_PAGERANK_NODES: usize = 500_000;

// ============================================================================
// Edge cost derivation from edge types
// ============================================================================

/// Derive traversal cost from an edge's type-specific metadata.
///
/// Lower cost = "closer" in the graph sense. Strength/confidence/similarity
/// values are inverted so that stronger relationships are cheaper to traverse.
/// The minimum cost is clamped to 0.001 to prevent zero-cost cycles.
#[inline]
pub fn edge_cost(edge: &GraphEdge) -> f32 {
    let raw = match &edge.edge_type {
        EdgeType::Causality { strength, .. } => 1.0 - strength,
        EdgeType::Temporal {
            sequence_confidence,
            ..
        } => 1.0 - sequence_confidence,
        EdgeType::Contextual { similarity, .. } => 1.0 - similarity,
        EdgeType::Interaction { success_rate, .. } => 1.0 - success_rate,
        EdgeType::GoalRelation {
            dependency_strength,
            ..
        } => 1.0 - dependency_strength,
        EdgeType::Association {
            statistical_significance,
            ..
        } => 1.0 - statistical_significance,
        EdgeType::Communication { reliability, .. } => 1.0 - reliability,
        EdgeType::DerivedFrom {
            extraction_confidence,
            ..
        } => 1.0 - extraction_confidence,
        EdgeType::SupportedBy {
            evidence_strength, ..
        } => 1.0 - evidence_strength,
        EdgeType::About {
            relevance_score, ..
        } => 1.0 - relevance_score,
        EdgeType::CodeStructure { confidence, .. } => 1.0 - confidence,
    };
    // Clamp to prevent zero-cost cycles while preserving ordering
    raw.max(0.001)
}

/// Get the edge cost between two specific nodes using the graph's edge data.
/// Falls back to 1.0 if no edge exists (for BFS-like behavior).
#[inline]
pub(crate) fn edge_cost_between(graph: &Graph, from: NodeId, to: NodeId) -> f32 {
    graph
        .get_edge_between(from, to)
        .map(edge_cost)
        .unwrap_or(1.0)
}
