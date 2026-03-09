//! Graph projection functions for querying structured data from graph edges.
//!
//! These functions walk the graph to project structured state (current location,
//! net balances, relationships, preferences) directly from Association edges,
//! replacing StructuredMemoryStore lookups where graph data is available.
//!
//! Temporal state is resolved using a dual check: the edge with the highest
//! `valid_from` for a given entity+category AND `valid_until.is_none()` is
//! definitionally current. This unifies with graph edge supersession.

use crate::structures::{EdgeType, Graph, NodeId, NodeType};
use std::collections::HashMap;

/// A single fact about an entity, derived from an outgoing Association edge.
#[derive(Debug, Clone)]
pub struct EntityFact {
    pub association_type: String,
    pub target_name: String,
    pub value: Option<String>,
    pub valid_from: Option<u64>,
    pub valid_until: Option<u64>,
    pub is_current: bool,
    /// Sentiment polarity for preference facts (-1.0 to 1.0, default 0.5).
    pub sentiment: f32,
}

/// Build a temporal timeline for entity+category.
///
/// Returns all edges for the given entity and association type, sorted by `valid_from` ASC.
/// This IS the temporal ledger — constructed dynamically from the graph.
pub fn build_temporal_timeline(
    graph: &Graph,
    entity_name: &str,
    category: &str,
) -> Vec<EntityFact> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return Vec::new(),
    };

    let old_style = format!("state:{}", category);
    let new_prefix = format!("{}:", category);
    let mut timeline: Vec<EntityFact> = Vec::new();

    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            // Match both old-style "state:location" and new-style "location:*" edges
            if association_type == &old_style || association_type.starts_with(&new_prefix) {
                let target_name = concept_name_of(graph, edge.target).unwrap_or_default();
                let value = edge
                    .properties
                    .get("value")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let sentiment = edge
                    .properties
                    .get("sentiment")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;

                timeline.push(EntityFact {
                    association_type: association_type.clone(),
                    target_name,
                    value,
                    valid_from: edge.valid_from,
                    valid_until: edge.valid_until,
                    is_current: false, // will be set below
                    sentiment,
                });
            }
        }
    }

    // Sort by valid_from ASC
    timeline.sort_by_key(|f| f.valid_from.unwrap_or(0));

    // Mark the last entry that is NOT explicitly superseded as current
    if let Some(last) = timeline.iter_mut().rev().find(|f| f.valid_until.is_none()) {
        last.is_current = true;
    }

    timeline
}

/// Get current state by picking the latest entry in the temporal timeline.
///
/// The edge with the MAX `valid_from` is definitionally current — no reliance
/// on supersession being correct.
pub fn resolve_current_state(
    graph: &Graph,
    entity_name: &str,
    category: &str,
) -> Option<EntityFact> {
    let timeline = build_temporal_timeline(graph, entity_name, category);
    timeline.into_iter().last()
}

/// Look up the current state value for an entity + attribute from graph edges.
///
/// Uses the dynamic temporal model: the edge with the highest `valid_from`
/// for the given association type is current.
pub fn state_current_from_graph(
    graph: &Graph,
    entity_name: &str,
    attribute: &str,
) -> Option<String> {
    resolve_current_state(graph, entity_name, attribute).map(|fact| fact.target_name)
}

/// Build a formatted timeline summary across all stateful categories for an entity.
///
/// Returns a human-readable timeline string showing the progression of state changes,
/// with each entry marked as `(superseded)` or `(CURRENT)`. Returns `None` if the
/// entity has no state edges.
pub fn build_entity_timeline_summary(graph: &Graph, entity_name: &str) -> Option<String> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return None,
    };

    // Collect all stateful association edges grouped by category
    let mut category_edges: HashMap<String, Vec<(String, Option<String>, Option<u64>, bool)>> =
        HashMap::new();

    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if !association_type.contains(':') {
                continue;
            }
            // Extract category (before the colon)
            let category = if let Some(cat) = association_type.strip_prefix("state:") {
                cat.to_string()
            } else {
                association_type.split(':').next().unwrap_or("").to_string()
            };

            // Skip append-only categories (e.g., financial) — they're not state
            let has_amount = edge.properties.contains_key("amount");
            if category.is_empty() || has_amount {
                continue;
            }

            let target_name = concept_name_of(graph, edge.target).unwrap_or_default();
            let value = edge
                .properties
                .get("value")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let display = value.unwrap_or(target_name);
            let is_current = edge.valid_until.is_none();

            category_edges.entry(category).or_default().push((
                display,
                edge.valid_from.map(|_| association_type.clone()),
                edge.valid_from,
                is_current,
            ));
        }
    }

    if category_edges.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    lines.push(format!("State timeline for {}:", entity_name));

    // Sort categories for deterministic output
    let mut cats: Vec<_> = category_edges.into_iter().collect();
    cats.sort_by(|a, b| a.0.cmp(&b.0));

    for (category, mut entries) in cats {
        // Sort by valid_from ASC
        entries.sort_by_key(|e| e.2.unwrap_or(0));

        // Only show timeline for categories with >0 entries
        if entries.len() == 1 && entries[0].3 {
            // Single current entry — just show as current state, no timeline needed
            lines.push(format!("- {}: {} (CURRENT)", category, entries[0].0));
        } else {
            // Collapse superseded entries into a count to avoid leaking old names
            // into the LLM context (which causes weaker models to hallucinate).
            let superseded_count = entries.iter().filter(|e| !e.3).count();
            let current_entry = entries.iter().find(|e| e.3);
            if let Some((display, _, _, _)) = current_entry {
                if superseded_count > 0 {
                    lines.push(format!(
                        "- {}: {} (CURRENT) — {} previous value(s) superseded",
                        category, display, superseded_count
                    ));
                } else {
                    lines.push(format!("- {}: {} (CURRENT)", category, display));
                }
            } else if superseded_count > 0 {
                // All superseded, no current — show last as most recent
                if let Some((display, _, _, _)) = entries.last() {
                    lines.push(format!(
                        "- {}: {} (most recent, superseded)",
                        category, display
                    ));
                }
            }
        }
    }

    Some(lines.join("\n"))
}

/// Compute net balances from financial Association edges in the graph.
///
/// Identifies financial edges by the presence of an `"amount"` property
/// (ontology-driven — no hardcoded edge type names).
///
/// Returns a map of `(payer_name, beneficiary_name) -> net_amount`.
pub fn compute_net_balances_from_graph(graph: &Graph) -> HashMap<(String, String), f64> {
    let mut balances: HashMap<(String, String), f64> = HashMap::new();

    // Iterate all edges looking for associations with an "amount" property
    for (_eid, edge) in graph.edges.iter() {
        if let EdgeType::Association { .. } = &edge.edge_type {
            if let Some(amount) = edge
                .properties
                .get("amount")
                .and_then(|v: &serde_json::Value| v.as_f64())
            {
                let from_name = match concept_name_of(graph, edge.source) {
                    Some(n) => n,
                    None => continue,
                };
                let to_name = match concept_name_of(graph, edge.target) {
                    Some(n) => n,
                    None => continue,
                };

                *balances.entry((from_name, to_name)).or_insert(0.0) += amount;
            }
        }
    }

    balances
}

/// BFS to find a path between two entities following `"relationship:{relation_type}"` edges.
pub fn find_relationship_path_from_graph(
    graph: &Graph,
    from: &str,
    to: &str,
    relation_type: &str,
) -> Option<Vec<String>> {
    let &from_id = graph.concept_index.get(from)?;
    let &to_id = graph.concept_index.get(to)?;

    if from_id == to_id {
        return Some(vec![from.to_string()]);
    }

    let assoc_type = format!("relationship:{}", relation_type);

    // BFS
    let mut visited: HashMap<NodeId, NodeId> = HashMap::new(); // child -> parent
    visited.insert(from_id, from_id);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(from_id);

    while let Some(current) = queue.pop_front() {
        for edge in graph.get_edges_from(current) {
            if let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            {
                if association_type == &assoc_type && !visited.contains_key(&edge.target) {
                    visited.insert(edge.target, current);
                    if edge.target == to_id {
                        // Reconstruct path
                        let mut path = Vec::new();
                        let mut cur = to_id;
                        while cur != from_id {
                            if let Some(name) = concept_name_of(graph, cur) {
                                path.push(name);
                            }
                            cur = visited[&cur];
                        }
                        if let Some(name) = concept_name_of(graph, from_id) {
                            path.push(name);
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(edge.target);
                }
            }
        }
    }

    None
}

/// Rank preferences for an entity in a given category from graph edges.
///
/// Returns `(item_name, sentiment)` pairs sorted by most recent first.
pub fn rank_preferences_from_graph(
    graph: &Graph,
    entity_name: &str,
    category: &str,
) -> Vec<(String, f32)> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return Vec::new(),
    };

    let assoc_prefix = format!("preference:{}", category);
    let mut prefs: Vec<(String, f32, u64)> = Vec::new();

    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if association_type == &assoc_prefix {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    let sentiment = edge
                        .properties
                        .get("sentiment")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5) as f32;
                    let vf = edge.valid_from.unwrap_or(0);
                    prefs.push((name, sentiment, vf));
                }
            }
        }
    }

    // Sort by valid_from descending (most recent first)
    prefs.sort_by(|a, b| b.2.cmp(&a.2));
    prefs
        .into_iter()
        .map(|(name, sent, _)| (name, sent))
        .collect()
}

/// Collect all structured facts about an entity from graph edges.
///
/// Walks ALL outgoing Association edges from the entity and returns
/// structured facts usable for entity summaries and general fact retrieval.
///
/// `is_current` is computed using `valid_until.is_none()` as primary check,
/// with max `valid_from` per `(association_type, target_name)` as tiebreaker.
pub fn collect_entity_facts(graph: &Graph, entity_name: &str) -> Vec<EntityFact> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return Vec::new(),
    };

    let mut facts = Vec::new();

    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            // Collect all structured Association edges (any "category:predicate" format).
            // The LLM generates arbitrary categories so we accept all of them.
            if association_type.contains(':') {
                let target_name = concept_name_of(graph, edge.target).unwrap_or_default();
                let value = edge
                    .properties
                    .get("value")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let sentiment = edge
                    .properties
                    .get("sentiment")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;

                facts.push(EntityFact {
                    association_type: association_type.clone(),
                    target_name,
                    value,
                    valid_from: edge.valid_from,
                    valid_until: edge.valid_until,
                    is_current: false, // computed below
                    sentiment,
                });
            }
        }
    }

    // Compute is_current: group by (full association_type, target_name) so that
    // independent facts like routine:saturday_morning and routine:saturday_evening
    // can both be current. Use valid_until.is_none() as primary check, max valid_from
    // as tiebreaker only for duplicate edges with the same type+target.
    let mut max_vf_per_key: HashMap<(String, String), u64> = HashMap::new();
    for fact in &facts {
        let key = (fact.association_type.clone(), fact.target_name.clone());
        let vf = fact.valid_from.unwrap_or(0);
        let entry = max_vf_per_key.entry(key).or_insert(0);
        if vf > *entry {
            *entry = vf;
        }
    }

    for fact in &mut facts {
        let key = (fact.association_type.clone(), fact.target_name.clone());
        let max_vf = max_vf_per_key.get(&key).copied().unwrap_or(0);
        fact.is_current = fact.valid_until.is_none() && fact.valid_from.unwrap_or(0) == max_vf;
    }

    facts
}

/// Project the direct children (neighbours) of an entity via relationship edges.
///
/// This is the graph-projection equivalent of `StructuredMemoryStore::tree_children`.
/// Walks outgoing symmetric-property edges from the entity and returns neighbour names.
/// Uses ontology to identify symmetric properties; falls back to "relationship" sub-property
/// expansion when ontology is available.
pub fn tree_children_from_graph(graph: &Graph, entity_name: &str) -> Vec<String> {
    tree_children_from_graph_with_ontology(graph, entity_name, None)
}

/// Ontology-aware version of `tree_children_from_graph`.
pub fn tree_children_from_graph_with_ontology(
    graph: &Graph,
    entity_name: &str,
    ontology: Option<&crate::ontology::OntologyRegistry>,
) -> Vec<String> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return Vec::new(),
    };

    let mut children = Vec::new();
    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            let cat = association_type.split(':').next().unwrap_or("");
            // Check if edge category is a symmetric property (ontology-driven).
            // Without ontology, include all Association edges as potential relationships.
            let include = match ontology {
                Some(onto) => onto.is_symmetric(cat),
                None => association_type.contains(':'),
            };
            if include {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    if !children.contains(&name) {
                        children.push(name);
                    }
                }
            }
        }
    }

    children
}

/// BFS to find a path between two entities following ANY symmetric-property edges.
///
/// Unlike `find_relationship_path_from_graph` which requires an exact relation_type,
/// this walks all symmetric-property edges (ontology-driven).
pub fn find_any_relationship_path_from_graph(
    graph: &Graph,
    from: &str,
    to: &str,
) -> Option<Vec<String>> {
    find_any_relationship_path_with_ontology(graph, from, to, None)
}

/// Ontology-aware version of `find_any_relationship_path_from_graph`.
pub fn find_any_relationship_path_with_ontology(
    graph: &Graph,
    from: &str,
    to: &str,
    ontology: Option<&crate::ontology::OntologyRegistry>,
) -> Option<Vec<String>> {
    let &from_id = graph.concept_index.get(from)?;
    let &to_id = graph.concept_index.get(to)?;

    if from_id == to_id {
        return Some(vec![from.to_string()]);
    }

    let mut visited: HashMap<NodeId, NodeId> = HashMap::new();
    visited.insert(from_id, from_id);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(from_id);

    while let Some(current) = queue.pop_front() {
        for edge in graph.get_edges_from(current) {
            if let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            {
                let cat = association_type.split(':').next().unwrap_or("");
                // Check if edge category is a symmetric property (ontology-driven).
                // Without ontology, include all Association edges as potential paths.
                let is_relationship = match ontology {
                    Some(onto) => onto.is_symmetric(cat),
                    None => association_type.contains(':'),
                };

                if is_relationship && !visited.contains_key(&edge.target) {
                    visited.insert(edge.target, current);
                    if edge.target == to_id {
                        let mut path = Vec::new();
                        let mut cur = to_id;
                        while cur != from_id {
                            if let Some(name) = concept_name_of(graph, cur) {
                                path.push(name);
                            }
                            cur = visited[&cur];
                        }
                        if let Some(name) = concept_name_of(graph, from_id) {
                            path.push(name);
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(edge.target);
                }
            }
        }
    }

    None
}

/// Helper: get the concept_name of a node, if it's a Concept node.
pub(crate) fn concept_name_of(graph: &Graph, node_id: NodeId) -> Option<String> {
    graph.get_node(node_id).and_then(|node| {
        if let NodeType::Concept { concept_name, .. } = &node.node_type {
            Some(concept_name.clone())
        } else {
            None
        }
    })
}

/// A structured multi-hop view of an entity's current state.
///
/// Facts are categorized generically by their association type prefix (e.g.
/// "location", "routine", "relationship", "preference:food").  No domain-specific
/// hardcoding — every edge category discovered in the graph gets its own bucket.
#[derive(Debug, Clone, Default)]
pub struct EntityStateView {
    pub entity_name: String,
    /// Facts grouped by category prefix (e.g. "location", "routine", "relationship", "preference:food")
    pub categories: HashMap<String, Vec<EntityFact>>,
    /// financial:payment net balances: (payer, beneficiary) -> net amount
    pub balances: HashMap<(String, String), f64>,
}

/// Walk entity state from graph edges — multi-hop structured projection.
///
/// Collects all current facts about an entity, categorizes them, includes
/// financial balances, and does a second hop to get relationship targets'
/// current locations.
pub fn walk_entity_state(graph: &Graph, entity_name: &str) -> EntityStateView {
    let mut view = EntityStateView {
        entity_name: entity_name.to_string(),
        ..Default::default()
    };

    let facts = collect_entity_facts(graph, entity_name);

    for fact in facts {
        if !fact.is_current {
            continue;
        }

        // Derive category key from the association type prefix
        let assoc = &fact.association_type;
        let category = if let Some(pos) = assoc.find(':') {
            &assoc[..pos]
        } else {
            assoc.as_str()
        };

        view.categories
            .entry(category.to_string())
            .or_default()
            .push(fact);
    }

    // Add financial balances involving this entity
    let all_balances = compute_net_balances_from_graph(graph);
    let entity_lower = entity_name.to_lowercase();
    for ((payer, beneficiary), amount) in &all_balances {
        if payer.to_lowercase() == entity_lower || beneficiary.to_lowercase() == entity_lower {
            view.balances
                .insert((payer.clone(), beneficiary.clone()), *amount);
        }
    }

    // Second hop: for all category targets, enrich with their current state.
    // Try all single-valued state categories from the entity's projected state
    // to find useful context (e.g., location).
    let all_cats: Vec<String> = view.categories.keys().cloned().collect();
    for cat in &all_cats {
        if let Some(rels) = view.categories.get_mut(cat.as_str()) {
            for rel in rels.iter_mut() {
                if rel.value.is_some() {
                    continue;
                }
                // Enrich with the first single-valued state of the target
                let target_projected =
                    project_entity_state(graph, &rel.target_name, u64::MAX, None);
                for slot in target_projected.slots.values() {
                    let slot_cat = slot.association_type.split(':').next().unwrap_or("");
                    // Pick the first slot that looks like a state value
                    if !slot.target_name.is_empty() && !slot_cat.is_empty() {
                        rel.value = Some(format!("({}: {})", slot_cat, slot.target_name));
                        break;
                    }
                }
            }
        }
    }

    view
}

impl EntityStateView {
    /// Format a human-readable summary suitable for LLM synthesis context.
    ///
    /// Iterates categories generically — no hardcoded domain knowledge. Each
    /// category is rendered as `"Category: item1, item2"` with optional value
    /// annotations (e.g. relationship location enrichment).
    pub fn format_summary(&self) -> String {
        let mut parts = Vec::new();

        // Sort categories for deterministic output
        let mut cats: Vec<_> = self.categories.keys().collect();
        cats.sort();

        for cat in cats {
            let facts = &self.categories[cat];
            if facts.is_empty() {
                continue;
            }
            let items: Vec<String> = facts
                .iter()
                .map(|f| {
                    let label = f
                        .association_type
                        .split(':')
                        .nth(1)
                        .unwrap_or(&f.target_name);
                    match &f.value {
                        Some(v) => format!("{}: {} {}", label, f.target_name, v),
                        None => format!("{}: {}", label, f.target_name),
                    }
                })
                .collect();
            // Capitalize category name for readability
            let cat_display = {
                let mut c = cat.chars();
                match c.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().to_string() + c.as_str(),
                }
            };
            parts.push(format!("{}: {}", cat_display, items.join("; ")));
        }

        if !self.balances.is_empty() {
            let bals: Vec<String> = self
                .balances
                .iter()
                .map(|((p, b), amt)| format!("{} → {}: {:.2}", p, b, amt))
                .collect();
            parts.push(format!("Balances: {}", bals.join("; ")));
        }

        if parts.is_empty() {
            format!("No current facts for {}", self.entity_name)
        } else {
            format!("Entity: {}\n{}", self.entity_name, parts.join("\n"))
        }
    }
}

// ---------------------------------------------------------------------------
// Successor-state projection
// ---------------------------------------------------------------------------

/// A single projected slot in an entity's current state.
#[derive(Debug, Clone)]
pub struct ProjectedSlot {
    /// Canonical "category:predicate" (e.g., "location:lives_in")
    pub association_type: String,
    pub target_name: String,
    pub value: Option<String>,
    pub valid_from: Option<u64>,
    pub edge_id: u64,
}

/// A hint that an edge's stored `valid_until` disagrees with the projection.
#[derive(Debug, Clone)]
pub struct RepairHint {
    pub edge_id: u64,
    /// The `valid_until` value the edge *should* have. `None` means "should be current".
    pub set_valid_until: Option<u64>,
    pub reason: String,
}

/// The projected current state of an entity, computed from successor-state rules.
#[derive(Debug, Clone, Default)]
pub struct ProjectedState {
    pub entity_name: String,
    /// Current value per canonical domain key. For Single: one entry per category.
    /// For Multi: one entry per (category, sub-key).
    pub slots: HashMap<String, ProjectedSlot>,
    /// Edges whose stored valid_until disagrees with the projection.
    pub repair_hints: Vec<RepairHint>,
}

/// Internal edge record used during projection.
struct EdgeRecord {
    edge_id: u64,
    association_type: String,
    target_name: String,
    value: Option<String>,
    valid_from: u64,
    valid_until: Option<u64>,
}

/// Compute the successor-state projection of an entity's current state.
///
/// Walks ALL outgoing Association edges, normalizes each to a canonical domain
/// key, groups by domain, and applies successor rules based on cardinality:
///
/// - **Single**: last edge (by `valid_from`) wins; all prior active edges generate RepairHints.
/// - **Multi**: group by sub-key (predicate suffix), apply Single logic within each sub-key.
/// - **Append**: all edges are current, no successor logic.
///
/// Pass `as_of = u64::MAX` for current state, or a timestamp for state-at-time-T.
pub fn project_entity_state(
    graph: &Graph,
    entity_name: &str,
    as_of: u64,
    ontology: Option<&crate::ontology::OntologyRegistry>,
) -> ProjectedState {
    use crate::domain_schema::Cardinality;

    let mut state = ProjectedState {
        entity_name: entity_name.to_string(),
        ..Default::default()
    };

    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return state,
    };

    // 1. Collect all outgoing Association edges
    let mut records: Vec<EdgeRecord> = Vec::new();
    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if !association_type.contains(':') {
                continue;
            }
            let vf = edge.valid_from.unwrap_or(0);
            if vf > as_of {
                continue; // Future edge, skip for state-at-T
            }
            let target_name = concept_name_of(graph, edge.target).unwrap_or_default();
            let value = edge
                .properties
                .get("value")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            records.push(EdgeRecord {
                edge_id: edge.id,
                association_type: association_type.clone(),
                target_name,
                value,
                valid_from: vf,
                valid_until: edge.valid_until,
            });
        }
    }

    // 2. Normalize each edge to a canonical domain key
    //    and determine cardinality
    struct NormalizedRecord {
        domain_key: String, // grouping key (e.g., "location" for Single, "routine:morning" for Multi)
        canonical_assoc: String, // full canonical assoc_type for the slot
        cardinality: Cardinality,
        rec: EdgeRecord,
    }

    let mut normalized: Vec<NormalizedRecord> = Vec::new();
    for rec in records {
        let (category, predicate) = if let Some(rest) = rec.association_type.strip_prefix("state:")
        {
            // Legacy "state:location" → ("location", "lives_in") via ontology decode
            if let Some((cat, cpred)) =
                ontology.and_then(|o| o.decode_legacy_state_assoc(&rec.association_type))
            {
                (cat, cpred)
            } else {
                (rest.to_string(), String::new())
            }
        } else if let Some(idx) = rec.association_type.find(':') {
            let cat = &rec.association_type[..idx];
            let pred = &rec.association_type[idx + 1..];
            let canonical = if let Some(onto) = ontology {
                onto.canonicalize_predicate(cat, pred).into_owned()
            } else {
                pred.to_string()
            };
            (cat.to_string(), canonical)
        } else {
            continue;
        };

        let cardinality = if let Some(onto) = ontology {
            if onto.is_append_only(&category) {
                Cardinality::Append
            } else if onto.is_single_valued(&category) {
                Cardinality::Single
            } else {
                Cardinality::Multi
            }
        } else {
            Cardinality::Multi
        };

        let (domain_key, canonical_assoc) = match cardinality {
            Cardinality::Single => {
                // Group by category alone — only one active value
                let resolved_cpred = if let Some(onto) = ontology {
                    onto.resolve(&category)
                        .map(|d| d.canonical_predicate.clone())
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                let cpred = if resolved_cpred.is_empty() {
                    &predicate
                } else {
                    &resolved_cpred
                };
                (category.clone(), format!("{}:{}", category, cpred))
            },
            Cardinality::Multi => {
                // Group by category + sub-key
                let sub_key = if predicate.is_empty() {
                    rec.association_type
                        .split(':')
                        .nth(1)
                        .unwrap_or("")
                        .to_string()
                } else {
                    predicate.clone()
                };
                let dk = format!("{}:{}", category, sub_key);
                let ca = format!("{}:{}", category, sub_key);
                (dk, ca)
            },
            Cardinality::Append => {
                // Every edge is current — unique key per edge
                let dk = format!("append:{}:{}", rec.edge_id, rec.association_type);
                let ca = rec.association_type.clone();
                (dk, ca)
            },
        };

        normalized.push(NormalizedRecord {
            domain_key,
            canonical_assoc,
            cardinality,
            rec,
        });
    }

    // 3. Group by domain_key
    let mut groups: HashMap<String, Vec<NormalizedRecord>> = HashMap::new();
    for nr in normalized {
        groups.entry(nr.domain_key.clone()).or_default().push(nr);
    }

    // 4. For each group, apply successor rules
    for (_key, mut group) in groups {
        if group.is_empty() {
            continue;
        }

        let cardinality = group[0].cardinality;

        match cardinality {
            Cardinality::Append => {
                // All edges are current
                for nr in group {
                    state.slots.insert(
                        format!("append:{}:{}", nr.rec.edge_id, nr.canonical_assoc),
                        ProjectedSlot {
                            association_type: nr.canonical_assoc,
                            target_name: nr.rec.target_name,
                            value: nr.rec.value,
                            valid_from: Some(nr.rec.valid_from),
                            edge_id: nr.rec.edge_id,
                        },
                    );
                }
            },
            Cardinality::Single | Cardinality::Multi => {
                // Sort by valid_from ASC — last one wins
                group.sort_by_key(|nr| nr.rec.valid_from);

                let winner_idx = group.len() - 1;
                for (i, nr) in group.iter().enumerate() {
                    if i == winner_idx {
                        // Winner: should be current (valid_until = None)
                        if nr.rec.valid_until.is_some() {
                            state.repair_hints.push(RepairHint {
                                edge_id: nr.rec.edge_id,
                                set_valid_until: None,
                                reason: format!(
                                    "newest edge for '{}' has valid_until set but should be current",
                                    nr.canonical_assoc
                                ),
                            });
                        }
                    } else {
                        // Loser: should have valid_until set
                        if nr.rec.valid_until.is_none() {
                            // Compute what valid_until should be: the next edge's valid_from
                            let next_vf = group[i + 1].rec.valid_from;
                            state.repair_hints.push(RepairHint {
                                edge_id: nr.rec.edge_id,
                                set_valid_until: Some(next_vf),
                                reason: format!(
                                    "older edge for '{}' missing valid_until (superseded at t={})",
                                    nr.canonical_assoc, next_vf
                                ),
                            });
                        }
                    }
                }

                // Insert winner into slots
                let winner = &group[winner_idx];
                state.slots.insert(
                    winner.domain_key.clone(),
                    ProjectedSlot {
                        association_type: winner.canonical_assoc.clone(),
                        target_name: winner.rec.target_name.clone(),
                        value: winner.rec.value.clone(),
                        valid_from: Some(winner.rec.valid_from),
                        edge_id: winner.rec.edge_id,
                    },
                );
            },
        }
    }

    state
}

/// Apply repair hints to a mutable graph. NOT called automatically.
///
/// Intended for future maintenance task or explicit repair endpoint.
/// Caller must hold the inference write lock.
pub fn apply_repair_hints(graph: &mut Graph, hints: &[RepairHint]) {
    for hint in hints {
        if let Some(edge) = graph.edges.get_mut(hint.edge_id) {
            edge.valid_until = hint.set_valid_until;
            graph.dirty_edges.insert(hint.edge_id);
            tracing::info!(
                "apply_repair_hint edge_id={} set valid_until={:?} reason='{}'",
                hint.edge_id,
                hint.set_valid_until,
                hint.reason
            );
        }
    }
}

/// Rank preferences for an entity using PersonalizedPageRank for graph-structural weighting.
///
/// Combines sentiment scores with PPR proximity scores from the entity node.
/// Items connected through multiple paths or shared with similar entities rank higher.
/// Falls back to the recency-based `rank_preferences_from_graph()` if PPR fails.
pub fn rank_preferences_with_ppr(
    graph: &Graph,
    entity_name: &str,
    category: &str,
) -> Vec<(String, f32)> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return Vec::new(),
    };

    // Only use PPR for graphs with enough structure
    if graph.node_count() < 5 {
        return rank_preferences_from_graph(graph, entity_name, category);
    }

    // Run PPR from the entity node
    let walker = crate::algorithms::random_walk::RandomWalker::with_config(
        crate::algorithms::random_walk::RandomWalkConfig {
            walk_length: 40,
            restart_probability: 0.15,
            num_walks: 50,
            weighted: true,
            seed: Some(42),
        },
    );
    let ppr_scores = match walker.personalized_pagerank(graph, node_id) {
        Ok(scores) => scores,
        Err(_) => return rank_preferences_from_graph(graph, entity_name, category),
    };

    let assoc_prefix = format!("preference:{}", category);
    let mut prefs: Vec<(String, f32)> = Vec::new();

    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if association_type == &assoc_prefix && edge.valid_until.is_none() {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    let sentiment = edge
                        .properties
                        .get("sentiment")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5) as f32;
                    // Weight sentiment by PPR score of target node
                    let ppr = *ppr_scores.get(&edge.target).unwrap_or(&0.001);
                    let combined = sentiment * (1.0 + (ppr as f32).ln().max(-5.0));
                    prefs.push((name, combined));
                }
            }
        }
    }

    prefs.sort_by(|a, b| b.1.total_cmp(&a.1));
    prefs
}

/// Filter preferences by sentiment range.
///
/// Enables queries like "what are my negative reviews" (min=-1.0, max=0.0)
/// and "how many positive preferences" (min=0.0, max=1.0).
pub fn filter_preferences_by_sentiment(
    graph: &Graph,
    entity_name: &str,
    category: &str,
    min_sentiment: f32,
    max_sentiment: f32,
) -> Vec<(String, f32)> {
    let node_id = match graph.concept_index.get(entity_name) {
        Some(&nid) => nid,
        None => return Vec::new(),
    };

    let assoc_prefix = format!("preference:{}", category);
    let mut prefs: Vec<(String, f32)> = Vec::new();

    for edge in graph.get_edges_from(node_id) {
        if let EdgeType::Association {
            association_type, ..
        } = &edge.edge_type
        {
            if association_type == &assoc_prefix && edge.valid_until.is_none() {
                if let Some(name) = concept_name_of(graph, edge.target) {
                    let sentiment = edge
                        .properties
                        .get("sentiment")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5) as f32;
                    if sentiment >= min_sentiment && sentiment <= max_sentiment {
                        prefs.push((name, sentiment));
                    }
                }
            }
        }
    }

    prefs.sort_by(|a, b| b.1.total_cmp(&a.1));
    prefs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{ConceptType, EdgeType, Graph, GraphEdge, GraphNode, NodeType};

    fn test_ontology() -> crate::ontology::OntologyRegistry {
        let path = std::path::PathBuf::from("../../data/ontology");
        crate::ontology::OntologyRegistry::load_from_directory(&path).unwrap_or_else(|_| {
            let alt = std::path::PathBuf::from("data/ontology");
            crate::ontology::OntologyRegistry::load_from_directory(&alt)
                .unwrap_or_else(|_| crate::ontology::OntologyRegistry::new())
        })
    }

    fn make_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add entity node "User"
        let user_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "User".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        // Add target nodes
        let sat_morning_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "morning_yoga".to_string(),
                concept_type: ConceptType::NamedEntity,
                confidence: 1.0,
            }))
            .unwrap();

        let sat_evening_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "evening_run".to_string(),
                concept_type: ConceptType::NamedEntity,
                confidence: 1.0,
            }))
            .unwrap();

        let tokyo_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Tokyo".to_string(),
                concept_type: ConceptType::Location,
                confidence: 1.0,
            }))
            .unwrap();

        let anna_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Anna".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        // Add routine:saturday_morning edge
        let mut edge1 = GraphEdge::new(
            user_id,
            sat_morning_id,
            EdgeType::Association {
                association_type: "routine:saturday_morning".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        edge1.valid_from = Some(100);
        graph.add_edge(edge1);

        // Add routine:saturday_evening edge
        let mut edge2 = GraphEdge::new(
            user_id,
            sat_evening_id,
            EdgeType::Association {
                association_type: "routine:saturday_evening".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        edge2.valid_from = Some(100);
        graph.add_edge(edge2);

        // Add location edge
        let mut edge3 = GraphEdge::new(
            user_id,
            tokyo_id,
            EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        edge3.valid_from = Some(200);
        graph.add_edge(edge3);

        // Add relationship edge
        let mut edge4 = GraphEdge::new(
            user_id,
            anna_id,
            EdgeType::Association {
                association_type: "relationship:sister".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        edge4.valid_from = Some(50);
        graph.add_edge(edge4);

        graph
    }

    #[test]
    fn test_independent_routines_both_current() {
        let graph = make_test_graph();
        let facts = collect_entity_facts(&graph, "User");

        let sat_morning: Vec<_> = facts
            .iter()
            .filter(|f| f.association_type == "routine:saturday_morning")
            .collect();
        let sat_evening: Vec<_> = facts
            .iter()
            .filter(|f| f.association_type == "routine:saturday_evening")
            .collect();

        assert_eq!(sat_morning.len(), 1);
        assert_eq!(sat_evening.len(), 1);
        assert!(
            sat_morning[0].is_current,
            "saturday_morning should be current"
        );
        assert!(
            sat_evening[0].is_current,
            "saturday_evening should be current"
        );
    }

    #[test]
    fn test_walk_entity_state_categorizes() {
        let graph = make_test_graph();
        let view = walk_entity_state(&graph, "User");

        assert_eq!(view.entity_name, "User");

        // The fixture creates edges with prefixes "location:", "routine:", "relationship:".
        // Verify the generic categorization puts each prefix into its own bucket
        // and preserves the correct number of items and target names.
        assert_eq!(
            view.categories.len(),
            3,
            "should have 3 distinct category prefixes"
        );

        // Collect all target names across all categories
        let all_targets: Vec<&str> = view
            .categories
            .values()
            .flat_map(|facts| facts.iter().map(|f| f.target_name.as_str()))
            .collect();
        assert!(
            all_targets.contains(&"Tokyo"),
            "should contain location target"
        );
        assert!(
            all_targets.contains(&"Anna"),
            "should contain relationship target"
        );

        // Total current facts: 1 location + 2 routines + 1 relationship = 4
        let total: usize = view.categories.values().map(|v| v.len()).sum();
        assert_eq!(total, 4, "should have 4 total categorized facts");
    }

    #[test]
    fn test_walk_entity_state_format_summary() {
        let graph = make_test_graph();
        let view = walk_entity_state(&graph, "User");
        let summary = view.format_summary();

        // Verify targets appear in the summary (not category names)
        assert!(summary.contains("Tokyo"), "summary should mention Tokyo");
        assert!(summary.contains("Anna"), "summary should mention Anna");
        // Verify it's structured with "Entity:" header
        assert!(
            summary.starts_with("Entity: User"),
            "summary should start with entity header"
        );
    }

    #[test]
    fn test_walk_entity_state_empty_entity() {
        let graph = make_test_graph();
        let view = walk_entity_state(&graph, "NonExistent");
        assert!(
            view.categories.is_empty(),
            "unknown entity should have no categories"
        );
        let summary = view.format_summary();
        assert!(summary.contains("No current facts"));
    }

    // ── Projection walker tests ──

    fn make_projection_graph() -> Graph {
        let mut graph = Graph::new();

        let user_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "User".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        let tokyo_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Tokyo".to_string(),
                concept_type: ConceptType::Location,
                confidence: 1.0,
            }))
            .unwrap();

        let nyc_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "NYC".to_string(),
                concept_type: ConceptType::Location,
                confidence: 1.0,
            }))
            .unwrap();

        let yoga_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "morning_yoga".to_string(),
                concept_type: ConceptType::NamedEntity,
                confidence: 1.0,
            }))
            .unwrap();

        let run_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "evening_run".to_string(),
                concept_type: ConceptType::NamedEntity,
                confidence: 1.0,
            }))
            .unwrap();

        // Location edge 1: Tokyo at t=100 (no valid_until — should be repaired)
        let mut e1 = GraphEdge::new(
            user_id,
            tokyo_id,
            EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        e1.valid_from = Some(100);
        graph.add_edge(e1);

        // Location edge 2: NYC at t=200
        let mut e2 = GraphEdge::new(
            user_id,
            nyc_id,
            EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        e2.valid_from = Some(200);
        graph.add_edge(e2);

        // Routine: morning_yoga at t=100
        let mut e3 = GraphEdge::new(
            user_id,
            yoga_id,
            EdgeType::Association {
                association_type: "routine:morning".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        e3.valid_from = Some(100);
        graph.add_edge(e3);

        // Routine: evening_run at t=100
        let mut e4 = GraphEdge::new(
            user_id,
            run_id,
            EdgeType::Association {
                association_type: "routine:evening".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        e4.valid_from = Some(100);
        graph.add_edge(e4);

        graph
    }

    #[test]
    fn test_projection_single_valued_last_wins() {
        let graph = make_projection_graph();
        let onto = test_ontology();
        let state = project_entity_state(&graph, "User", u64::MAX, Some(&onto));

        let loc = state.slots.get("location").expect("location slot missing");
        assert_eq!(loc.target_name, "NYC");
        assert_eq!(loc.valid_from, Some(200));
    }

    #[test]
    fn test_projection_repair_hint_missing_valid_until() {
        let graph = make_projection_graph();
        let onto = test_ontology();
        let state = project_entity_state(&graph, "User", u64::MAX, Some(&onto));

        // The older Tokyo edge lacks valid_until → should generate a repair hint
        let tokyo_hint = state
            .repair_hints
            .iter()
            .find(|h| h.reason.contains("older edge") && h.reason.contains("location"));
        assert!(
            tokyo_hint.is_some(),
            "expected repair hint for old Tokyo edge"
        );
        assert_eq!(tokyo_hint.unwrap().set_valid_until, Some(200));
    }

    #[test]
    fn test_projection_repair_hint_wrong_valid_until_on_winner() {
        let mut graph = make_projection_graph();

        // Set valid_until on the NYC edge (the winner) — this is wrong
        for (_eid, edge) in graph.edges.iter_mut() {
            if let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            {
                if association_type == "location:lives_in" && edge.valid_from == Some(200) {
                    edge.valid_until = Some(999);
                }
            }
        }

        let onto = test_ontology();
        let state = project_entity_state(&graph, "User", u64::MAX, Some(&onto));
        let winner_hint = state
            .repair_hints
            .iter()
            .find(|h| h.reason.contains("newest edge") && h.set_valid_until.is_none());
        assert!(
            winner_hint.is_some(),
            "expected repair hint to clear valid_until on winner"
        );
    }

    #[test]
    fn test_projection_multi_valued_independent() {
        let graph = make_projection_graph();
        let onto = test_ontology();
        let state = project_entity_state(&graph, "User", u64::MAX, Some(&onto));

        // Both routine sub-keys should be current
        let morning = state.slots.get("routine:morning");
        let evening = state.slots.get("routine:evening");
        assert!(morning.is_some(), "routine:morning should be present");
        assert!(evening.is_some(), "routine:evening should be present");
        assert_eq!(morning.unwrap().target_name, "morning_yoga");
        assert_eq!(evening.unwrap().target_name, "evening_run");
    }

    #[test]
    fn test_projection_state_at_time_t() {
        let graph = make_projection_graph();
        let onto = test_ontology();
        let state = project_entity_state(&graph, "User", 150, Some(&onto));

        // At t=150, only Tokyo edge (t=100) is visible
        let loc = state
            .slots
            .get("location")
            .expect("location slot missing at t=150");
        assert_eq!(loc.target_name, "Tokyo");
    }

    #[test]
    fn test_projection_legacy_state_assoc() {
        let mut graph = Graph::new();

        let user_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "User".to_string(),
                concept_type: ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();

        let london_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "London".to_string(),
                concept_type: ConceptType::Location,
                confidence: 1.0,
            }))
            .unwrap();

        let mut e = GraphEdge::new(
            user_id,
            london_id,
            EdgeType::Association {
                association_type: "state:location".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            1.0,
        );
        e.valid_from = Some(50);
        graph.add_edge(e);

        let onto = test_ontology();
        let state = project_entity_state(&graph, "User", u64::MAX, Some(&onto));
        let loc = state
            .slots
            .get("location")
            .expect("location slot missing for legacy edge");
        assert_eq!(loc.target_name, "London");
    }

    #[test]
    fn test_projection_empty_entity() {
        let graph = make_projection_graph();
        let state = project_entity_state(&graph, "NonExistent", u64::MAX, None);
        assert!(state.slots.is_empty());
        assert!(state.repair_hints.is_empty());
    }

    #[test]
    fn test_apply_repair_hints() {
        let mut graph = make_projection_graph();
        let state = project_entity_state(&graph, "User", u64::MAX, None);

        // Should have at least one repair hint (the old Tokyo edge)
        assert!(!state.repair_hints.is_empty());

        apply_repair_hints(&mut graph, &state.repair_hints);

        // Re-project — should have no repair hints now
        let state2 = project_entity_state(&graph, "User", u64::MAX, None);
        assert!(
            state2.repair_hints.is_empty(),
            "after repair, no hints should remain; got: {:?}",
            state2.repair_hints
        );
    }
}
