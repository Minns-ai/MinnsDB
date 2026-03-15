// crates/agent-db-graph/src/integration/pipeline/fact_writing.rs
//
// Write extracted facts to the graph as edges. Handles financial facts,
// general facts, deduplication, supersession, cascade deletion, and symmetric properties.

use super::*;
use crate::structures::ConceptType;

impl GraphEngine {
    /// Write a single extracted fact directly to the graph ledger.
    ///
    /// This is the correct path for compaction-extracted facts:
    ///   1. Resolve entity node (fuzzy match or create)
    ///   2. Walk graph: any active edge for this entity+category?
    ///   3. Same value → skip (idempotent)
    ///   4. Different value → supersede old edge, create new edge
    ///   5. No existing edge → create new edge
    ///
    /// One fact, one graph walk, one edge write. No re-processing.
    /// Write a fact to the graph as an edge. Uses open-ended edge types from LLM extraction.
    ///
    /// Edge type = `{category}:{predicate}` (e.g. `location:lives_in`, `work:works_with`).
    /// Financial facts use special `financial:payment` type with structured amount data.
    ///
    /// No hardcoded supersession — conflict detection is handled separately by
    /// `detect_edge_conflicts` which uses semantic search + LLM comparison
    pub async fn write_fact_to_graph(
        &self,
        fact: &crate::conversation::compaction::ExtractedFact,
        timestamp: u64,
    ) {
        let group_id = &fact.group_id;
        let category = fact.category.as_deref().unwrap_or("other");
        let is_financial = self.ontology.is_append_only(category);

        // 2. Edge type: canonicalize predicate BEFORE taking the write lock.
        // Try embedding-based canonicalizer first, fall back to sync version.
        let assoc_type = if is_financial {
            let cp = self.ontology.canonical_predicate(category);
            let pred = if cp.is_empty() { "payment" } else { &cp };
            self.ontology.build_edge_type(category, pred)
        } else {
            let canonical = if let Some(ref pc) = self.predicate_canonicalizer {
                pc.canonicalize(category, &fact.predicate).await
            } else {
                self.ontology
                    .canonicalize_predicate(category, &fact.predicate)
                    .into_owned()
            };
            let base = self.ontology.build_edge_type(category, &canonical);
            // Fallback: when Multi-valued category has generic predicate, derive sub-key from object
            if !self.ontology.is_single_valued(category) && canonical == category {
                let sub_key = derive_sub_key(&fact.object);
                if !sub_key.is_empty() {
                    self.ontology.build_edge_type(category, &sub_key)
                } else {
                    base
                }
            } else {
                base
            }
        };

        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // 1. Resolve entity node
        let entity_nid = match resolve_or_create_entity(
            graph,
            &fact.subject,
            ConceptType::NamedEntity,
            group_id,
        ) {
            Some(nid) => nid,
            None => return,
        };

        let target_ctype = self.ontology.default_target_concept_type(category);

        // 3. Resolve the target value as a concept node
        let target_nid = match resolve_or_create_entity(graph, &fact.object, target_ctype, group_id)
        {
            Some(nid) => nid,
            None => return,
        };

        if is_financial {
            // ── Financial: use LLM-extracted amount + split_with fields ──
            let total_amount = match fact.amount {
                Some(a) if a > 0.0 => a,
                _ => {
                    tracing::debug!(
                        "write_fact_to_graph SKIP financial (no amount) stmt='{}'",
                        fact.statement.get(..80).unwrap_or(&fact.statement)
                    );
                    return;
                },
            };

            let payer_name = normalize_entity_name(&fact.subject);
            let split_with = fact.split_with.as_deref().unwrap_or(&[]);

            let is_all = split_with.iter().any(|s| s.to_lowercase() == "all");
            let beneficiaries: Vec<String> = if is_all {
                let mut participants: Vec<String> = Vec::new();
                for (name, _) in graph.concept_index.iter() {
                    if let Some(&nid) = graph.concept_index.get(name) {
                        let has_financial = graph.get_edges_from(nid).iter().any(|e| {
                            matches!(&e.edge_type, crate::structures::EdgeType::Association { association_type, .. }
                                if crate::ontology::OntologyRegistry::parse_edge_category(association_type)
                                    .map(|c| self.ontology.is_append_only(c)).unwrap_or(false))
                        }) || graph.get_edges_to(nid).iter().any(|e| {
                            matches!(&e.edge_type, crate::structures::EdgeType::Association { association_type, .. }
                                if crate::ontology::OntologyRegistry::parse_edge_category(association_type)
                                    .map(|c| self.ontology.is_append_only(c)).unwrap_or(false))
                        });
                        if has_financial {
                            let norm = normalize_entity_name(name);
                            if norm != payer_name && !participants.contains(&norm) {
                                participants.push(norm);
                            }
                        }
                    }
                }
                participants
            } else {
                split_with
                    .iter()
                    .map(|s| normalize_entity_name(s))
                    .collect()
            };

            if beneficiaries.is_empty() {
                let mut edge = GraphEdge::new(
                    entity_nid,
                    target_nid,
                    crate::structures::EdgeType::Association {
                        association_type: self.ontology.build_edge_type("financial", "payment"),
                        evidence_count: 1,
                        statistical_significance: 0.9,
                    },
                    0.9,
                );
                edge.valid_from = Some(timestamp);
                edge.group_id = group_id.clone();
                edge.properties
                    .insert("amount".to_string(), serde_json::json!(total_amount));
                edge.properties.insert(
                    "statement".to_string(),
                    serde_json::Value::String(fact.statement.clone()),
                );
                for (k, v) in &fact.ingest_metadata {
                    edge.properties
                        .entry(k.clone())
                        .or_insert_with(|| v.clone());
                }
                let new_eid = graph.add_edge(edge);
                tracing::info!(
                    "write_fact_to_graph financial eid={:?} '{}' -> '{}' amount={:.2}",
                    new_eid,
                    fact.subject,
                    fact.object,
                    total_amount
                );
            } else {
                let num_people = beneficiaries.len() as f64 + 1.0;
                let per_person = total_amount / num_people;
                for beneficiary_name in &beneficiaries {
                    let ben_nid = match resolve_or_create_entity(
                        graph,
                        beneficiary_name,
                        ConceptType::NamedEntity,
                        group_id,
                    ) {
                        Some(nid) => nid,
                        None => continue,
                    };
                    let mut edge = GraphEdge::new(
                        entity_nid,
                        ben_nid,
                        crate::structures::EdgeType::Association {
                            association_type: self.ontology.build_edge_type("financial", "payment"),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        0.9,
                    );
                    edge.valid_from = Some(timestamp);
                    edge.group_id = group_id.clone();
                    edge.properties
                        .insert("amount".to_string(), serde_json::json!(per_person));
                    edge.properties.insert(
                        "statement".to_string(),
                        serde_json::Value::String(fact.statement.clone()),
                    );
                    for (k, v) in &fact.ingest_metadata {
                        edge.properties
                            .entry(k.clone())
                            .or_insert_with(|| v.clone());
                    }
                    let new_eid = graph.add_edge(edge);
                    tracing::info!("write_fact_to_graph financial eid={:?} '{}' -> '{}' amount={:.2} (total={:.2}/{})",
                        new_eid, fact.subject, beneficiary_name, per_person, total_amount, num_people as u64);
                }
            }
        } else {
            // ── General fact: create edge, dedup exact matches ──

            // Check for exact duplicate (same entity → same target, same edge type, still active)
            let already_exists = graph.get_edges_from(entity_nid).iter().any(|e| {
                if let crate::structures::EdgeType::Association {
                    association_type, ..
                } = &e.edge_type
                {
                    association_type == &assoc_type
                        && e.target == target_nid
                        && e.valid_until.is_none()
                } else {
                    false
                }
            });
            if already_exists {
                tracing::debug!(
                    "write_fact_to_graph SKIP duplicate '{}' '{}'->'{}'",
                    assoc_type,
                    fact.subject,
                    fact.object
                );
                return;
            }

            // ── Supersede active edges, scoped by exact property ──
            // Single-valued properties: supersede edges of the SAME property
            // (not all edges sharing a parent category prefix).
            // E.g., location:lives_in supersedes location:lives_in, but NOT
            // workplace_location:works_in — those are separate properties.
            // Multi-valued: only supersede when is_update is explicitly true.
            let single_valued = self.ontology.is_single_valued(category);
            if single_valued || fact.is_update.unwrap_or(false) {
                let supersession_prefixes: Vec<String> = self
                    .ontology
                    .supersession_group(category)
                    .iter()
                    .map(|c| format!("{}:", c))
                    .collect();
                let edges_to_supersede: Vec<crate::structures::EdgeId> = graph
                    .get_edges_from(entity_nid)
                    .iter()
                    .filter(|e| {
                        if let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &e.edge_type
                        {
                            if e.valid_until.is_some() || e.target == target_nid {
                                return false;
                            }
                            if single_valued {
                                // Single-valued: supersede edges matching any prefix
                                // in the supersession group
                                supersession_prefixes
                                    .iter()
                                    .any(|p| association_type.starts_with(p.as_str()))
                            } else {
                                // Multi-valued: only supersede edges with exact same assoc_type
                                association_type == &assoc_type
                            }
                        } else {
                            false
                        }
                    })
                    .map(|e| e.id)
                    .collect();

                // Collect target names before mutation (read-only pass)
                let supersede_info: Vec<(crate::structures::EdgeId, String)> = edges_to_supersede
                    .iter()
                    .filter_map(|eid| {
                        graph.edges.get(*eid).map(|e| {
                            let name = crate::conversation::graph_projection::concept_name_of(
                                graph, e.target,
                            )
                            .unwrap_or_default();
                            (*eid, name)
                        })
                    })
                    .collect();

                for (eid, old_target) in &supersede_info {
                    if let Some(e) = graph.edges.get_mut(*eid) {
                        e.valid_until = Some(timestamp);
                        graph.dirty_edges.insert(*eid);
                        tracing::info!(
                            "write_fact_to_graph is_update=true superseded '{}' -> '{}' (replaced by '{}')",
                            assoc_type, old_target, fact.object
                        );
                    }
                }

                // ── depends_on cascade: supersede edges whose depends_on references
                // the superseded target. E.g., "visits Meiji Shrine" depends_on
                // "User lives in Tokyo" — when Tokyo is superseded, Meiji Shrine goes too.
                if !edges_to_supersede.is_empty() {
                    // Collect superseded target names (read-only pass)
                    let superseded_targets: Vec<String> = edges_to_supersede
                        .iter()
                        .filter_map(|eid| graph.edges.get(*eid))
                        .filter_map(|e| {
                            crate::conversation::graph_projection::concept_name_of(graph, e.target)
                        })
                        .collect();

                    if !superseded_targets.is_empty() {
                        // Collect edges to cascade-supersede and their info (read-only pass)
                        let dep_edges_info: Vec<(crate::structures::EdgeId, String, String)> =
                            graph
                                .get_edges_from(entity_nid)
                                .iter()
                                .filter(|e| {
                                    if e.valid_until.is_some() {
                                        return false;
                                    }
                                    if let Some(serde_json::Value::String(dep)) =
                                        e.properties.get("depends_on")
                                    {
                                        let dep_lower = dep.to_lowercase();
                                        superseded_targets
                                            .iter()
                                            .any(|t| dep_lower.contains(&t.to_lowercase()))
                                    } else {
                                        false
                                    }
                                })
                                .map(|e| {
                                    let target_name =
                                        crate::conversation::graph_projection::concept_name_of(
                                            graph, e.target,
                                        )
                                        .unwrap_or_default();
                                    let dep_text = e
                                        .properties
                                        .get("depends_on")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?")
                                        .to_string();
                                    (e.id, target_name, dep_text)
                                })
                                .collect();

                        // Mutation pass
                        for (eid, dep_target, dep_text) in &dep_edges_info {
                            if let Some(e) = graph.edges.get_mut(*eid) {
                                e.valid_until = Some(timestamp);
                                graph.dirty_edges.insert(*eid);
                                tracing::info!(
                                    "write_fact_to_graph depends_on cascade: superseded edge -> '{}' (depends_on: '{}')",
                                    dep_target, dep_text
                                );
                            }
                        }
                    }
                }
            }

            // Create new edge
            let mut edge = GraphEdge::new(
                entity_nid,
                target_nid,
                crate::structures::EdgeType::Association {
                    association_type: assoc_type.clone(),
                    evidence_count: 1,
                    statistical_significance: 0.9,
                },
                0.9,
            );
            edge.valid_from = Some(timestamp);
            edge.group_id = group_id.clone();
            edge.properties.insert(
                "statement".to_string(),
                serde_json::Value::String(fact.statement.clone()),
            );
            edge.properties.insert(
                "value".to_string(),
                serde_json::Value::String(fact.object.clone()),
            );
            if let Some(cat) = &fact.category {
                edge.properties.insert(
                    "category".to_string(),
                    serde_json::Value::String(cat.clone()),
                );
            }
            if let Some(ref dep) = fact.depends_on {
                edge.properties.insert(
                    "depends_on".to_string(),
                    serde_json::Value::String(dep.clone()),
                );
            }
            if let Some(ref ts) = fact.temporal_signal {
                edge.properties.insert(
                    "temporal_signal".to_string(),
                    serde_json::Value::String(ts.clone()),
                );
            }
            // Wire sentiment from extraction to edge properties (preference facts)
            if let Some(sent) = fact.sentiment {
                edge.properties
                    .insert("sentiment".to_string(), serde_json::json!(sent));
            }
            // Propagate ingest-level metadata to edge properties
            for (k, v) in &fact.ingest_metadata {
                edge.properties
                    .entry(k.clone())
                    .or_insert_with(|| v.clone());
            }
            let new_eid = graph.add_edge(edge);
            tracing::info!(
                "write_fact_to_graph edge eid={:?} '{}' '{}'->'{}'",
                new_eid,
                assoc_type,
                fact.subject,
                fact.object
            );

            // Auto-register unknown categories in the domain registry
            if self.ontology.resolve(category).is_none() && category != "other" {
                let cardinality = fact
                    .cardinality_hint
                    .as_deref()
                    .map(|h| match h {
                        "single" => crate::domain_schema::Cardinality::Single,
                        "append" => crate::domain_schema::Cardinality::Append,
                        _ => crate::domain_schema::Cardinality::Multi,
                    })
                    .unwrap_or(crate::domain_schema::Cardinality::Multi);
                if self
                    .ontology
                    .register_category(category, cardinality, "", false)
                {
                    tracing::info!(
                        "Auto-registered domain category '{}' with cardinality {:?}",
                        category,
                        cardinality
                    );
                    // Persist as concept node (backward compat)
                    if let Some(slot) = self
                        .ontology
                        .learned_slots()
                        .iter()
                        .find(|s| s.category == category)
                    {
                        crate::domain_schema::persist_domain_slot(graph, slot);
                    }
                }
            } else {
                self.ontology.record_usage(category);
            }

            // For symmetric properties, also create the reverse edge
            if self.ontology.is_symmetric(category) {
                let rev_exists = graph.get_edges_from(target_nid).iter().any(|e| {
                    if let crate::structures::EdgeType::Association {
                        association_type, ..
                    } = &e.edge_type
                    {
                        association_type == &assoc_type
                            && e.target == entity_nid
                            && e.valid_until.is_none()
                    } else {
                        false
                    }
                });
                if !rev_exists {
                    let mut rev_edge = GraphEdge::new(
                        target_nid,
                        entity_nid,
                        crate::structures::EdgeType::Association {
                            association_type: assoc_type.clone(),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        0.9,
                    );
                    rev_edge.valid_from = Some(timestamp);
                    rev_edge.properties.insert(
                        "statement".to_string(),
                        serde_json::Value::String(fact.statement.clone()),
                    );
                    graph.add_edge(rev_edge);
                }
            }
        }
    }
}
