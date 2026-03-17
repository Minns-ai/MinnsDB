use crate::structures::{EdgeId, NodeId};
use agent_db_core::types::Timestamp;
use smallvec::SmallVec;

/// A single atomic change to the graph, captured at mutation time.
#[derive(Debug, Clone)]
pub enum GraphDelta {
    NodeAdded {
        node_id: NodeId,
        node_type_disc: u8,
        generation: u64,
    },
    NodeRemoved {
        node_id: NodeId,
        node_type_disc: u8,
        generation: u64,
    },
    EdgeAdded {
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type_tag: String,
        generation: u64,
    },
    EdgeRemoved {
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type_tag: String,
        generation: u64,
    },
    /// Temporal supersession: valid_until changed. NOT a delete+insert pair.
    EdgeSuperseded {
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type_tag: String,
        old_valid_until: Option<Timestamp>,
        new_valid_until: Option<Timestamp>,
        generation: u64,
    },
    /// Edge property mutation (weight, confidence, observation_count).
    EdgeMutated {
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type_tag: String,
        generation: u64,
    },
    /// Node merge: absorbed_id removed, edges redirected to survivor_id.
    NodeMerged {
        survivor_id: NodeId,
        absorbed_id: NodeId,
        generation: u64,
    },
}

impl GraphDelta {
    /// All node IDs touched by this delta (for trigger set intersection).
    pub fn touched_nodes(&self) -> SmallVec<[NodeId; 2]> {
        match self {
            GraphDelta::NodeAdded { node_id, .. } | GraphDelta::NodeRemoved { node_id, .. } => {
                smallvec::smallvec![*node_id]
            }
            GraphDelta::EdgeAdded {
                source, target, ..
            }
            | GraphDelta::EdgeRemoved {
                source, target, ..
            }
            | GraphDelta::EdgeSuperseded {
                source, target, ..
            }
            | GraphDelta::EdgeMutated {
                source, target, ..
            } => smallvec::smallvec![*source, *target],
            GraphDelta::NodeMerged {
                survivor_id,
                absorbed_id,
                ..
            } => smallvec::smallvec![*survivor_id, *absorbed_id],
        }
    }

    pub fn generation(&self) -> u64 {
        match self {
            GraphDelta::NodeAdded { generation, .. }
            | GraphDelta::NodeRemoved { generation, .. }
            | GraphDelta::EdgeAdded { generation, .. }
            | GraphDelta::EdgeRemoved { generation, .. }
            | GraphDelta::EdgeSuperseded { generation, .. }
            | GraphDelta::EdgeMutated { generation, .. }
            | GraphDelta::NodeMerged { generation, .. } => *generation,
        }
    }
}

/// A batch of deltas from a single mutation or write-lock scope.
#[derive(Debug, Clone)]
pub struct DeltaBatch {
    pub deltas: Vec<GraphDelta>,
    pub generation_range: (u64, u64),
}
