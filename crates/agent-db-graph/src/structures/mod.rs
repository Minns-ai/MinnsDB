//! Core graph data structures
//!
//! Implements the graph data structures used for modeling relationships
//! between agents, events, and contexts in the agentic database.

mod adj_list;
mod edge;
mod graph;
mod graph_ops;
mod graph_persistence;
mod graph_query;
mod node;
mod types;
mod vector_index;

#[cfg(test)]
mod tests;

// ── Re-exports ──

// Types & constants
pub use types::{Depth, Direction, EdgeId, EdgeWeight, GoalBucketId, NodeId, NUM_SHARDS};

// Adjacency list
pub use adj_list::{AdjList, AdjListIter};

// Node types
pub use node::{
    node_type_discriminant_from_name, ConceptType, GoalStatus, GraphNode, NodeType, NODE_TYPE_COUNT,
};

// Edge types
pub use edge::{EdgeType, GoalRelationType, GraphEdge, InteractionType};

// Vector index
pub use vector_index::NodeVectorIndex;

// Graph
pub(crate) use graph::edge_text_for_bm25;
pub use graph::{Graph, GraphStats};
