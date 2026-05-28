// crates/agent-db-graph/src/integration/pipeline/contradiction.rs
//
// Update classification: distinguish *evolution* (state legitimately changed
// over time), *contradiction* (two facts disagree at overlapping times),
// *reaffirmation* (new fact restates an active fact), and *independent*
// (nothing to reconcile). The EvoKG distinction in one place.
//
// Both `state_tracking.rs` and `fact_writing.rs` call `classify_update`
// before they mutate the graph, then dispatch on the returned `UpdateKind`.
// Keeping the classifier pure (read-only on the graph) makes the four
// cases trivially testable without spinning up an engine.

use crate::ontology::OntologyRegistry;
use crate::structures::{EdgeId, EdgeType, Graph, NodeId};

/// Why a new fact was classified as a contradiction. Kept on the edge as
/// a property and surfaced to the NLQ layer so answers can flag disputed
/// state to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContradictionReason {
    /// New fact's `valid_from` predates an existing active edge but its
    /// target differs. Either a retroactive correction or a contradiction.
    /// We flag both edges and let downstream code resolve.
    OutOfOrderTemporal,
    /// Property is `owl:FunctionalProperty` (at most one value) but the
    /// new value would coexist with an active edge of the same property.
    FunctionalPropertyViolation,
    /// LLM verdict flagged a logical conflict (not produced by
    /// `classify_update` directly; reserved for the LLM path in
    /// `conflict_detection.rs`).
    LogicalConflict,
}

impl ContradictionReason {
    /// Short tag stored on the edge property `disputed_reason` so the
    /// classification survives serialisation and is queryable later.
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            Self::OutOfOrderTemporal => "out_of_order_temporal",
            Self::FunctionalPropertyViolation => "functional_property_violation",
            Self::LogicalConflict => "logical_conflict",
        }
    }
}

/// Outcome of comparing a new fact to the existing state of its subject.
/// Each variant carries the edge IDs the caller needs to touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UpdateKind {
    /// No active edge of this property exists; just create the new one.
    Independent,
    /// An active edge with the same target already exists; skip creation
    /// and refresh `last_confirmed_at` on the existing edge.
    Reaffirmation { existing_eid: EdgeId },
    /// State legitimately changed over time. The listed edges are the
    /// prior-active values that should be sealed with `valid_until`.
    Evolution { superseded_eids: Vec<EdgeId> },
    /// Two facts disagree without a clear temporal ordering. The listed
    /// edges should be flagged `disputed` rather than superseded.
    Contradiction {
        reason: ContradictionReason,
        conflicting_eids: Vec<EdgeId>,
    },
}

/// Classify a new fact against the existing active edges from `entity_nid`.
///
/// Pure-temporal logic only. The LLM-driven verdict (used when statements
/// look superficially different but mean the same thing) lives in
/// `conflict_detection.rs` and operates on a different layer.
///
/// Decision order:
/// 1. No active same-property edges → `Independent`.
/// 2. Any active edge with the same target → `Reaffirmation`.
/// 3. `new_valid_from` strictly before every active edge's `valid_from` →
///    `Contradiction(OutOfOrderTemporal)`. (We learnt about a *past*
///    state but it disagrees with what we currently believe.)
/// 4. Ontology marks the property functional AND the existing values
///    don't all precede the new one in time → `Contradiction(
///    FunctionalPropertyViolation)`.
/// 5. Otherwise → `Evolution`, sealing every existing active edge.
pub(crate) fn classify_update(
    graph: &Graph,
    ontology: &OntologyRegistry,
    entity_nid: NodeId,
    new_target_nid: NodeId,
    new_assoc_type: &str,
    new_valid_from: u64,
    category: &str,
) -> UpdateKind {
    // 1. Gather active edges from this subject that share the property.
    //    `assoc_type` is the full `category:predicate` string; we match
    //    on it exactly here because supersession across sibling
    //    predicates is the caller's responsibility (see fact_writing.rs
    //    `supersession_group` logic).
    let active_same_type: Vec<(EdgeId, NodeId, Option<u64>)> = graph
        .get_edges_from(entity_nid)
        .iter()
        .filter_map(|e| {
            if let EdgeType::Association {
                association_type, ..
            } = &e.edge_type
            {
                if association_type == new_assoc_type && e.valid_until.is_none() {
                    return Some((e.id, e.target, e.valid_from));
                }
            }
            None
        })
        .collect();

    if active_same_type.is_empty() {
        return UpdateKind::Independent;
    }

    // 2. Reaffirmation: an active edge already points at the new target.
    if let Some((existing_eid, _, _)) = active_same_type
        .iter()
        .find(|(_, tgt, _)| *tgt == new_target_nid)
    {
        return UpdateKind::Reaffirmation {
            existing_eid: *existing_eid,
        };
    }

    // 3. Out-of-order temporal contradiction. We use the *earliest*
    //    active valid_from as the cutoff so a single late-arriving fact
    //    in the middle of an evolution chain doesn't trip the rule.
    //    Edges without an explicit valid_from are treated as t=0 (always
    //    before the new fact), so they fall through to the evolution
    //    case below — they can't anchor an out-of-order claim.
    if active_same_type
        .iter()
        .filter_map(|(_, _, vf)| *vf)
        .all(|vf| new_valid_from < vf)
        && active_same_type.iter().any(|(_, _, vf)| vf.is_some())
    {
        let conflicting_eids = active_same_type.iter().map(|(eid, _, _)| *eid).collect();
        return UpdateKind::Contradiction {
            reason: ContradictionReason::OutOfOrderTemporal,
            conflicting_eids,
        };
    }

    // 4. Functional property violation. If the ontology says this
    //    property is single-valued and the new fact's valid_from does
    //    NOT strictly post-date every active edge, two distinct values
    //    would be simultaneously believed true.
    if ontology.is_functional(category) {
        let strictly_after_all = active_same_type.iter().all(|(_, _, vf)| match vf {
            Some(existing_vf) => new_valid_from > *existing_vf,
            None => true,
        });
        if !strictly_after_all {
            let conflicting_eids = active_same_type.iter().map(|(eid, _, _)| *eid).collect();
            return UpdateKind::Contradiction {
                reason: ContradictionReason::FunctionalPropertyViolation,
                conflicting_eids,
            };
        }
    }

    // 5. Plain evolution. Every active edge gets sealed when the caller
    //    writes the new one.
    let superseded_eids = active_same_type.iter().map(|(eid, _, _)| *eid).collect();
    UpdateKind::Evolution { superseded_eids }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{PropertyCharacteristic, PropertyDescriptor};
    use crate::structures::{ConceptType, EdgeType, EdgeWeight, GraphEdge, GraphNode, NodeType};

    /// Insert a concept node and return its NodeId. Wraps the boilerplate
    /// used by every test in this module.
    fn add_concept(graph: &mut Graph, name: &str) -> NodeId {
        let node = GraphNode::new(NodeType::Concept {
            concept_name: name.to_string(),
            concept_type: ConceptType::ContextualAssociation,
            confidence: 1.0,
        });
        graph.add_node(node).expect("add_node")
    }

    /// Insert an active association edge from `src` to `tgt` with the
    /// given `assoc_type` and `valid_from`. Returns its EdgeId.
    fn add_active_edge(
        graph: &mut Graph,
        src: NodeId,
        tgt: NodeId,
        assoc_type: &str,
        valid_from: u64,
    ) -> EdgeId {
        let mut edge = GraphEdge::new(
            src,
            tgt,
            EdgeType::Association {
                association_type: assoc_type.to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            EdgeWeight::from(0.9),
        );
        edge.valid_from = Some(valid_from);
        graph.add_edge(edge).expect("add_edge returned None")
    }

    /// Bare ontology registry with no properties registered. Defaults are
    /// non-functional and not append-only, matching unknown predicates.
    fn empty_ontology() -> OntologyRegistry {
        OntologyRegistry::new()
    }

    /// Ontology with `location` registered as a functional property.
    fn functional_ontology() -> OntologyRegistry {
        let onto = OntologyRegistry::new();
        let mut desc = PropertyDescriptor::default_for("location");
        desc.characteristics
            .push(PropertyCharacteristic::Functional);
        desc.max_cardinality = Some(1);
        onto.register_property(desc);
        onto
    }

    #[test]
    fn no_active_edges_is_independent() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let nyc = add_concept(&mut g, "NYC");
        let kind = classify_update(
            &g,
            &empty_ontology(),
            alice,
            nyc,
            "location:lives_in",
            1_000,
            "location",
        );
        assert_eq!(kind, UpdateKind::Independent);
    }

    #[test]
    fn same_target_active_is_reaffirmation() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let nyc = add_concept(&mut g, "NYC");
        let existing = add_active_edge(&mut g, alice, nyc, "location:lives_in", 1_000);
        let kind = classify_update(
            &g,
            &empty_ontology(),
            alice,
            nyc,
            "location:lives_in",
            2_000,
            "location",
        );
        assert_eq!(
            kind,
            UpdateKind::Reaffirmation {
                existing_eid: existing
            }
        );
    }

    #[test]
    fn monotonic_change_is_evolution() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let london = add_concept(&mut g, "London");
        let nyc = add_concept(&mut g, "NYC");
        let old_eid = add_active_edge(&mut g, alice, london, "location:lives_in", 1_000);
        let kind = classify_update(
            &g,
            &empty_ontology(),
            alice,
            nyc,
            "location:lives_in",
            2_000,
            "location",
        );
        assert_eq!(
            kind,
            UpdateKind::Evolution {
                superseded_eids: vec![old_eid]
            }
        );
    }

    #[test]
    fn earlier_than_all_active_is_out_of_order_contradiction() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let london = add_concept(&mut g, "London");
        let nyc = add_concept(&mut g, "NYC");
        let old_eid = add_active_edge(&mut g, alice, london, "location:lives_in", 5_000);
        let kind = classify_update(
            &g,
            &empty_ontology(),
            alice,
            nyc,
            "location:lives_in",
            1_000,
            "location",
        );
        assert_eq!(
            kind,
            UpdateKind::Contradiction {
                reason: ContradictionReason::OutOfOrderTemporal,
                conflicting_eids: vec![old_eid],
            }
        );
    }

    #[test]
    fn functional_property_simultaneous_is_contradiction() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let london = add_concept(&mut g, "London");
        let nyc = add_concept(&mut g, "NYC");
        // Existing edge at t=1000; new fact at t=1000 (equal, not strictly
        // after) with a different target and a functional property.
        let old_eid = add_active_edge(&mut g, alice, london, "location:lives_in", 1_000);
        let kind = classify_update(
            &g,
            &functional_ontology(),
            alice,
            nyc,
            "location:lives_in",
            1_000,
            "location",
        );
        assert_eq!(
            kind,
            UpdateKind::Contradiction {
                reason: ContradictionReason::FunctionalPropertyViolation,
                conflicting_eids: vec![old_eid],
            }
        );
    }

    #[test]
    fn functional_property_later_in_time_is_evolution() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let london = add_concept(&mut g, "London");
        let nyc = add_concept(&mut g, "NYC");
        let old_eid = add_active_edge(&mut g, alice, london, "location:lives_in", 1_000);
        let kind = classify_update(
            &g,
            &functional_ontology(),
            alice,
            nyc,
            "location:lives_in",
            2_000,
            "location",
        );
        assert_eq!(
            kind,
            UpdateKind::Evolution {
                superseded_eids: vec![old_eid]
            }
        );
    }

    #[test]
    fn different_property_yields_independent() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let london = add_concept(&mut g, "London");
        let coffee = add_concept(&mut g, "coffee");
        // Active lives_in edge exists, but we're adding a likes edge.
        let _ = add_active_edge(&mut g, alice, london, "location:lives_in", 1_000);
        let kind = classify_update(
            &g,
            &empty_ontology(),
            alice,
            coffee,
            "preference:likes",
            2_000,
            "preference",
        );
        assert_eq!(kind, UpdateKind::Independent);
    }

    #[test]
    fn edge_with_no_valid_from_does_not_anchor_out_of_order() {
        let mut g = Graph::new();
        let alice = add_concept(&mut g, "Alice");
        let london = add_concept(&mut g, "London");
        let nyc = add_concept(&mut g, "NYC");
        // Existing active edge has no explicit valid_from. New fact at
        // t=1 with a different target should evolve, not contradict.
        let mut edge = GraphEdge::new(
            alice,
            london,
            EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            EdgeWeight::from(0.9),
        );
        edge.valid_from = None;
        let old_eid = g.add_edge(edge).expect("add_edge");
        let kind = classify_update(
            &g,
            &empty_ontology(),
            alice,
            nyc,
            "location:lives_in",
            1,
            "location",
        );
        assert_eq!(
            kind,
            UpdateKind::Evolution {
                superseded_eids: vec![old_eid]
            }
        );
    }
}
