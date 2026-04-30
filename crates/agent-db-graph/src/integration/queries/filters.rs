// crates/agent-db-graph/src/integration/queries/filters.rs
//
// Temporal validity filter, state anchor filter, and epoch-based filter
// for removing stale/superseded results from the NLQ pipeline.

/// Temporal validity filter: remove results linked to superseded `state:*` edges.
///
/// Returns the set of superseded target names for downstream filtering.
///
/// For multi-transition scenarios (2tr, 3tr, 4tr+), this is the critical gate
/// that prevents historical facts from polluting the synthesis context.
/// Superseded results are zeroed out (removed), not just soft-demoted.
pub(crate) fn apply_temporal_validity_filter(
    results: &mut Vec<(u64, f32)>,
    graph: &crate::structures::Graph,
) -> std::collections::HashSet<String> {
    use crate::conversation::graph_projection::concept_name_of;
    use crate::structures::{EdgeType, NodeType};

    // Phase 1: Collect superseded state target names.
    let mut superseded_targets: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    // Any structured "category:predicate" edge is potentially stateful.
    // The LLM generates arbitrary categories so we accept all of them —
    // supersession (valid_until) already distinguishes current vs old.
    let is_stateful = |assoc: &str| -> bool { assoc.contains(':') };

    for (_eid, edge) in graph.edges.iter() {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if is_stateful(association_type) && edge.valid_until.is_some() {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    superseded_targets.insert(name.to_lowercase());
                }
            }
        }
    }

    // Phase 1d: depends_on cascade — edges whose depends_on property references
    // a superseded target should also have their targets added to superseded_targets.
    // E.g., edge with depends_on: "User lives in Tokyo" → if "tokyo" is in
    // superseded_targets, add this edge's target too.
    if !superseded_targets.is_empty() {
        let mut dep_cascade_targets: Vec<String> = Vec::new();
        for (_eid, edge) in graph.edges.iter() {
            if edge.valid_until.is_some() {
                continue; // already superseded
            }
            if let Some(serde_json::Value::String(dep)) = edge.properties.get("depends_on") {
                let dep_lower = dep.to_lowercase();
                let is_dep_stale = superseded_targets
                    .iter()
                    .any(|t| dep_lower.contains(t.as_str()));
                if is_dep_stale {
                    if let Some(name) = concept_name_of(graph, edge.target) {
                        dep_cascade_targets.push(name.to_lowercase());
                    }
                }
            }
        }
        for t in dep_cascade_targets {
            superseded_targets.insert(t);
        }
    }

    // Phase 1c: Also collect current targets so we never accidentally filter them
    let mut current_targets: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_eid, edge) in graph.edges.iter() {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if is_stateful(association_type) && edge.valid_until.is_none() {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    current_targets.insert(name.to_lowercase());
                }
            }
        }
    }
    // Remove anything that's also a current target (safety: don't filter current state)
    for ct in &current_targets {
        superseded_targets.remove(ct);
    }

    // Phase 2: Zero out nodes that are direct targets of superseded edges.
    // Covers both old "state:*" edges and new "category:predicate" edges
    // (e.g. location:lives_in, routine:visits).
    for (node_id, score) in results.iter_mut() {
        for edge in graph.get_edges_to(*node_id) {
            if let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            {
                if is_stateful(association_type) || association_type.starts_with("state:") {
                    if edge.valid_until.is_some() {
                        *score = 0.0; // Hard remove — superseded target
                        break;
                    }
                    // Check if a newer edge exists (implicit supersession)
                    let edge_vf = edge.valid_from.unwrap_or(0);
                    let source = edge.source;
                    let assoc = association_type.clone();

                    let is_superseded = graph.get_edges_from(source).iter().any(|other| {
                        if let EdgeType::Association {
                            association_type: ref at,
                            ..
                        } = other.edge_type
                        {
                            at == &assoc
                                && other.target != *node_id
                                && other.valid_from.unwrap_or(0) > edge_vf
                        } else {
                            false
                        }
                    });

                    if is_superseded {
                        *score = 0.0; // Hard remove
                        break;
                    }
                }
            }
        }
    }

    // Phase 3: Zero out Claim nodes that mention superseded targets
    if !superseded_targets.is_empty() {
        for (node_id, score) in results.iter_mut() {
            if *score == 0.0 {
                continue; // Already removed
            }
            if let Some(n) = graph.get_node(*node_id) {
                if let NodeType::Claim { claim_text, .. } = &n.node_type {
                    let text_lower = claim_text.to_lowercase();
                    for target in &superseded_targets {
                        if target.len() < 3 {
                            continue;
                        }
                        if let Some(pos) = text_lower.find(target.as_str()) {
                            let before_ok =
                                pos == 0 || !text_lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
                            let end = pos + target.len();
                            let after_ok = end >= text_lower.len()
                                || !text_lower.as_bytes()[end].is_ascii_alphanumeric();
                            if before_ok && after_ok {
                                *score = 0.0;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 3b: Also zero out Memory and Strategy nodes that mention superseded targets
    if !superseded_targets.is_empty() {
        for (node_id, score) in results.iter_mut() {
            if *score == 0.0 {
                continue;
            }
            if let Some(n) = graph.get_node(*node_id) {
                let text_to_check = match &n.node_type {
                    NodeType::Memory { .. } => Some(n.label().to_lowercase()),
                    NodeType::Strategy { .. } => Some(n.label().to_lowercase()),
                    _ => None,
                };
                if let Some(text_lower) = text_to_check {
                    for target in &superseded_targets {
                        if target.len() < 3 {
                            continue;
                        }
                        if let Some(pos) = text_lower.find(target.as_str()) {
                            let before_ok =
                                pos == 0 || !text_lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
                            let end = pos + target.len();
                            let after_ok = end >= text_lower.len()
                                || !text_lower.as_bytes()[end].is_ascii_alphanumeric();
                            if before_ok && after_ok {
                                *score = 0.0;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Remove zeroed entries and re-sort
    results.retain(|(_, score)| *score > 0.0);
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    superseded_targets
}

/// Filter claims whose state_anchor metadata disagrees with the current projected state.
///
/// Claims ingested while the user was in Amsterdam will have
/// `state_anchor:location:lives_in = Amsterdam`. If the current state is Vancouver,
/// those claims are zeroed out. Claims without state anchors pass through.
pub(crate) fn apply_state_anchor_filter(
    results: &mut Vec<(u64, f32)>,
    graph: &crate::structures::Graph,
    claim_store: &crate::claims::ClaimStore,
    ontology: &crate::ontology::OntologyRegistry,
) {
    use crate::conversation::graph_projection;
    use crate::structures::NodeType;

    // Project current state for "user" entity (the primary entity in personal knowledge graphs)
    let projected = graph_projection::project_entity_state(graph, "user", u64::MAX, Some(ontology));
    if projected.slots.is_empty() {
        return; // No state to filter against
    }

    // Build map of current state values: "location:lives_in" → "Vancouver"
    let mut current_state: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for slot in projected.slots.values() {
        let category = slot.association_type.split(':').next().unwrap_or("");
        if ontology.is_single_valued(category) {
            let val = slot.value.as_deref().unwrap_or(&slot.target_name);
            current_state.insert(
                format!("state_anchor:{}", slot.association_type),
                val.to_lowercase(),
            );
        }
    }

    if current_state.is_empty() {
        return;
    }

    for (node_id, score) in results.iter_mut() {
        if *score == 0.0 {
            continue;
        }
        // Only filter Claim nodes
        let claim_id = if let Some(node) = graph.get_node(*node_id) {
            if let NodeType::Claim { claim_id, .. } = &node.node_type {
                *claim_id
            } else {
                continue;
            }
        } else {
            continue;
        };

        // Look up claim metadata from store
        if let Ok(Some(claim)) = claim_store.get(claim_id) {
            for (anchor_key, anchor_val) in &claim.metadata {
                if !anchor_key.starts_with("state_anchor:") {
                    continue;
                }
                // Check if current state has a different value for this anchor
                if let Some(current_val) = current_state.get(anchor_key) {
                    if *current_val != anchor_val.to_lowercase() {
                        // Claim was anchored to a different state — filter it out
                        tracing::debug!(
                            "State anchor filter: claim {} anchored to {}={}, current={}",
                            claim_id,
                            anchor_key,
                            anchor_val,
                            current_val
                        );
                        *score = 0.0;
                        break;
                    }
                }
            }
        }
    }

    // Remove zeroed entries
    results.retain(|(_, score)| *score > 0.0);
}

/// Epoch-based filter: for single-valued categories, any claim created before
/// the current epoch's `valid_from` timestamp gets zeroed out.
///
/// This catches claims that escaped both the temporal validity filter (because
/// their graph nodes aren't direct targets of superseded edges) and the state
/// anchor filter (because they were never anchor-stamped).
pub(crate) fn apply_epoch_filter(
    results: &mut Vec<(u64, f32)>,
    graph: &crate::structures::Graph,
    claim_store: &crate::claims::ClaimStore,
    ontology: &crate::ontology::OntologyRegistry,
) {
    use crate::conversation::graph_projection;
    use crate::structures::NodeType;

    // Project current state for "user" entity
    let projected = graph_projection::project_entity_state(graph, "user", u64::MAX, Some(ontology));
    if projected.slots.is_empty() {
        return;
    }

    // Build epoch boundaries: category → valid_from timestamp
    let mut epoch_boundaries: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for slot in projected.slots.values() {
        let category = slot.association_type.split(':').next().unwrap_or("");
        if ontology.is_single_valued(category) {
            if let Some(vf) = slot.valid_from {
                epoch_boundaries.insert(category.to_string(), vf);
            }
        }
    }

    if epoch_boundaries.is_empty() {
        return;
    }

    for (node_id, score) in results.iter_mut() {
        if *score == 0.0 {
            continue;
        }
        // Only filter Claim, Memory, and Strategy nodes
        let (claim_id, is_claim_node) = if let Some(node) = graph.get_node(*node_id) {
            match &node.node_type {
                NodeType::Claim { claim_id, .. } => (Some(*claim_id), true),
                NodeType::Memory { .. } | NodeType::Strategy { .. } => (None, false),
                _ => continue,
            }
        } else {
            continue;
        };

        if let Some(cid) = claim_id {
            if is_claim_node {
                if let Ok(Some(claim)) = claim_store.get(cid) {
                    // Skip claims that already have state anchors (handled by anchor filter)
                    let has_anchors = claim
                        .metadata
                        .keys()
                        .any(|k| k.starts_with("state_anchor:"));
                    if has_anchors {
                        continue;
                    }

                    // Check if this claim's category matches any single-valued epoch
                    // and its created_at predates the epoch boundary
                    if let Some(ref cat) = claim.category {
                        if let Some(&epoch_start) = epoch_boundaries.get(cat.as_str()) {
                            if claim.created_at < epoch_start {
                                tracing::debug!(
                                    "Epoch filter: claim {} (cat={}) created_at={} < epoch={}",
                                    cid,
                                    cat,
                                    claim.created_at,
                                    epoch_start
                                );
                                *score = 0.0;
                            }
                        }
                    }
                }
            }
        } else {
            // Memory/Strategy nodes: check created_at property from the node
            if let Some(node) = graph.get_node(*node_id) {
                let created_at = node.properties.get("created_at").and_then(|v| v.as_u64());
                if let Some(ts) = created_at {
                    for &epoch_start in epoch_boundaries.values() {
                        if ts < epoch_start {
                            *score = 0.0;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Remove zeroed entries
    results.retain(|(_, score)| *score > 0.0);
}
