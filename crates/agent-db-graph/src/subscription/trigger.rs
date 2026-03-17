use std::collections::HashSet;

use crate::query_lang::planner::{ExecutionPlan, PlanStep};
use crate::structures::NodeId;

use super::delta::{DeltaBatch, GraphDelta};

/// Identifies which graph changes could affect a subscription's results.
#[derive(Debug, Clone)]
pub enum TriggerSet {
    /// Specific node IDs — changes to these nodes (or edges incident to them) trigger re-eval.
    Nodes(HashSet<NodeId>),
    /// Specific node type discriminants — new/removed nodes of these types trigger re-eval.
    NodeTypes(HashSet<u8>),
    /// Edge types that the query traverses.
    EdgeTypes(HashSet<String>),
    /// Combined: node types for scans + edge types for traversals.
    Combined {
        node_types: HashSet<u8>,
        edge_types: HashSet<String>,
    },
    /// Any change triggers (fallback for queries we can't analyze).
    Any,
}

impl TriggerSet {
    /// Check if a delta batch could affect a subscription with this trigger set.
    pub fn overlaps(&self, batch: &DeltaBatch) -> bool {
        match self {
            TriggerSet::Any => true,
            TriggerSet::Nodes(set) => batch
                .deltas
                .iter()
                .any(|d| d.touched_nodes().iter().any(|nid| set.contains(nid))),
            TriggerSet::NodeTypes(types) => batch.deltas.iter().any(|d| match d {
                GraphDelta::NodeAdded {
                    node_type_disc, ..
                }
                | GraphDelta::NodeRemoved {
                    node_type_disc, ..
                } => types.contains(node_type_disc),
                // Edge changes could affect results if they connect nodes of watched types.
                // Conservatively return true for edge changes — refined in Phase 3.
                GraphDelta::EdgeAdded { .. }
                | GraphDelta::EdgeRemoved { .. }
                | GraphDelta::EdgeSuperseded { .. }
                | GraphDelta::EdgeMutated { .. } => true,
                GraphDelta::NodeMerged { .. } => true,
            }),
            TriggerSet::EdgeTypes(types) => batch.deltas.iter().any(|d| match d {
                GraphDelta::EdgeAdded {
                    edge_type_tag, ..
                }
                | GraphDelta::EdgeRemoved {
                    edge_type_tag, ..
                }
                | GraphDelta::EdgeSuperseded {
                    edge_type_tag, ..
                }
                | GraphDelta::EdgeMutated {
                    edge_type_tag, ..
                } => types.contains(edge_type_tag.as_str()),
                GraphDelta::NodeAdded { .. } | GraphDelta::NodeRemoved { .. } => false,
                GraphDelta::NodeMerged { .. } => true,
            }),
            TriggerSet::Combined {
                node_types,
                edge_types,
            } => batch.deltas.iter().any(|d| match d {
                GraphDelta::NodeAdded {
                    node_type_disc, ..
                }
                | GraphDelta::NodeRemoved {
                    node_type_disc, ..
                } => node_types.contains(node_type_disc),
                GraphDelta::EdgeAdded {
                    edge_type_tag, ..
                }
                | GraphDelta::EdgeRemoved {
                    edge_type_tag, ..
                }
                | GraphDelta::EdgeSuperseded {
                    edge_type_tag, ..
                }
                | GraphDelta::EdgeMutated {
                    edge_type_tag, ..
                } => edge_types.contains(edge_type_tag.as_str()),
                GraphDelta::NodeMerged { .. } => true,
            }),
        }
    }
}

/// Compile a trigger set from an execution plan.
///
/// Walks the plan steps and extracts:
/// - Node type discriminants from ScanNodes label filters
/// - Edge type strings from Expand edge_type filters
pub fn compile_trigger_set(plan: &ExecutionPlan) -> TriggerSet {
    let mut node_types: HashSet<u8> = HashSet::new();
    let mut edge_types: HashSet<String> = HashSet::new();
    let mut has_unfiltered_scan = false;
    let mut has_unfiltered_expand = false;

    for step in &plan.steps {
        match step {
            PlanStep::ScanNodes { labels, .. } => {
                if labels.is_empty() {
                    has_unfiltered_scan = true;
                } else {
                    for label in labels {
                        if let Some(disc) = label_to_discriminant(label) {
                            node_types.insert(disc);
                        } else {
                            // Unknown label — can't filter by type
                            has_unfiltered_scan = true;
                        }
                    }
                }
            }
            PlanStep::Expand { edge_type, .. } => {
                if let Some(et) = edge_type {
                    edge_types.insert(et.clone());
                } else {
                    has_unfiltered_expand = true;
                }
            }
            PlanStep::Filter(_) => {
                // Filters don't change what we watch, they narrow results.
            }
        }
    }

    // If any step is unfiltered, fall back to Any.
    if has_unfiltered_scan && has_unfiltered_expand {
        return TriggerSet::Any;
    }

    match (node_types.is_empty(), edge_types.is_empty()) {
        (true, true) => TriggerSet::Any,
        (false, true) => {
            if has_unfiltered_expand {
                TriggerSet::Any // Can't scope edge changes
            } else {
                TriggerSet::NodeTypes(node_types)
            }
        }
        (true, false) => {
            if has_unfiltered_scan {
                TriggerSet::Any
            } else {
                TriggerSet::EdgeTypes(edge_types)
            }
        }
        (false, false) => TriggerSet::Combined {
            node_types,
            edge_types,
        },
    }
}

/// Compute the maximum graph radius (hops) that a query pattern covers.
/// Used for k-hop pruning: changes outside this radius from anchor nodes can be skipped.
pub fn compute_max_pattern_radius(plan: &ExecutionPlan) -> u32 {
    let mut total = 0u32;
    for step in &plan.steps {
        if let PlanStep::Expand { range, .. } = step {
            match range {
                Some((_, Some(max))) => total = total.saturating_add(*max),
                Some((_, None)) => return u32::MAX, // unbounded
                None => total = total.saturating_add(1),
            }
        }
    }
    total
}

/// Map MinnsQL label strings to NodeType discriminants.
fn label_to_discriminant(label: &str) -> Option<u8> {
    match label.to_lowercase().as_str() {
        "agent" => Some(0),
        "event" => Some(1),
        "context" => Some(2),
        "concept" => Some(3),
        "goal" => Some(4),
        "episode" => Some(5),
        "memory" => Some(6),
        "strategy" => Some(7),
        "tool" => Some(8),
        "result" => Some(9),
        "claim" => Some(10),
        // Labels that map to Concept nodes (used in MinnsQL as `:Person`, `:Location`, etc.)
        "person" | "location" | "organization" | "thing" => Some(3),
        _ => None,
    }
}
