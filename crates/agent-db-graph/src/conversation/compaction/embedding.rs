//! Post-compaction embedding and claim creation for vector/hybrid search.

use super::types::ExtractedFact;
use crate::conversation::graph_projection;
use std::collections::HashMap;

/// Build rich embedding text for an edge that goes beyond bare triplets.
///
/// Instead of "user location tokyo", produces something like:
/// "User currently lives in Tokyo. (recently moved, depends on: User relocated for work)"
///
/// Uses the original `statement` property (a self-contained natural language
/// proposition) when available, and appends qualifiers: temporal signal,
/// dependency, current/historical status.
fn build_rich_edge_text(
    source: &str,
    association_type: &str,
    target: &str,
    properties: &HashMap<String, serde_json::Value>,
    is_current: bool,
) -> String {
    // Prefer the original statement (richest representation)
    let base = if let Some(serde_json::Value::String(stmt)) = properties.get("statement") {
        stmt.clone()
    } else {
        // Fallback: reconstruct from triplet
        let predicate = association_type
            .split(':')
            .nth(1)
            .unwrap_or(association_type)
            .replace('_', " ");
        format!("{} {} {}", source, predicate, target)
    };

    let mut parts = vec![base];

    // Temporal qualifier
    if let Some(serde_json::Value::String(ts)) = properties.get("temporal_signal") {
        if !ts.is_empty() {
            parts.push(format!("({})", ts));
        }
    }

    // Dependency qualifier — what condition must hold for this fact
    if let Some(serde_json::Value::String(dep)) = properties.get("depends_on") {
        if !dep.is_empty() {
            parts.push(format!("[while: {}]", dep));
        }
    }

    // Current vs historical status
    if is_current {
        parts.push("[current]".to_string());
    } else {
        parts.push("[historical/superseded]".to_string());
    }

    parts.join(" ")
}

/// Embed all un-embedded concept nodes and create claims from extracted facts.
///
/// This ensures that node vector search and claim hybrid search (BM25+vector)
/// can find compaction-created artifacts. Must complete BEFORE queries execute.
pub(crate) async fn embed_nodes_and_create_claims(
    engine: &crate::integration::GraphEngine,
    facts: &[ExtractedFact],
    base_ts: u64,
) {
    let embedding_client = match &engine.embedding_client {
        Some(ec) => ec.clone(),
        None => {
            tracing::debug!("COMPACTION: no embedding client — skipping node/claim embedding");
            return;
        },
    };

    // 1. Embed un-embedded concept nodes
    let nodes_to_embed: Vec<(u64, String)> = {
        let inference = engine.inference.read().await;
        let graph = inference.graph();
        graph
            .concept_index
            .iter()
            .filter_map(|(name, &nid)| {
                graph.get_node(nid).and_then(|node| {
                    if node.embedding.is_empty() {
                        Some((nid, name.to_string()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    };

    if !nodes_to_embed.is_empty() {
        tracing::info!(
            "COMPACTION: embedding {} concept nodes for vector search",
            nodes_to_embed.len()
        );
        for (nid, text) in &nodes_to_embed {
            match embedding_client
                .embed(crate::claims::EmbeddingRequest {
                    text: text.clone(),
                    context: None,
                })
                .await
            {
                Ok(resp) if !resp.embedding.is_empty() => {
                    let mut inf = engine.inference.write().await;
                    let graph = inf.graph_mut();
                    if let Some(node) = graph.get_node_mut(*nid) {
                        node.embedding = resp.embedding.clone();
                    }
                    graph.node_vector_index.insert(*nid, resp.embedding);
                },
                Ok(_) => {},
                Err(e) => {
                    tracing::debug!("COMPACTION: node embedding failed for '{}': {}", text, e);
                },
            }
        }
    }

    // 2. Embed triplets (subject + predicate + object) for edge vector search.
    // This enables triplet scoring where the query is compared against
    // the full context of each edge, not just individual nodes.
    {
        let edges_to_embed: Vec<(u64, String)> = {
            let inference = engine.inference.read().await;
            let graph = inference.graph();
            let mut edges = Vec::new();
            for (eid, edge) in graph.edges.iter() {
                if let crate::structures::EdgeType::Association {
                    association_type, ..
                } = &edge.edge_type
                {
                    let source_name =
                        graph_projection::concept_name_of(graph, edge.source).unwrap_or_default();
                    let target_name =
                        graph_projection::concept_name_of(graph, edge.target).unwrap_or_default();
                    if !source_name.is_empty() && !target_name.is_empty() {
                        let rich_text = build_rich_edge_text(
                            &source_name,
                            association_type,
                            &target_name,
                            &edge.properties,
                            edge.valid_until.is_none(),
                        );
                        edges.push((eid, rich_text));
                    }
                }
            }
            edges
        };

        if !edges_to_embed.is_empty() {
            tracing::info!(
                "COMPACTION: embedding {} edge triplets for vector search",
                edges_to_embed.len()
            );
            for (eid, text) in &edges_to_embed {
                match embedding_client
                    .embed(crate::claims::EmbeddingRequest {
                        text: text.clone(),
                        context: None,
                    })
                    .await
                {
                    Ok(resp) if !resp.embedding.is_empty() => {
                        let mut inf = engine.inference.write().await;
                        let graph = inf.graph_mut();
                        graph.edge_vector_index.insert(*eid, resp.embedding);
                    },
                    Ok(_) => {},
                    Err(e) => {
                        tracing::debug!("COMPACTION: edge embedding failed for '{}': {}", text, e);
                    },
                }
            }
        }
    }

    // 3. Create claims from extracted facts (for claim hybrid search)
    let claim_store = match &engine.claim_store {
        Some(cs) => cs.clone(),
        None => return,
    };

    let mut claims_created = 0usize;
    for (i, fact) in facts.iter().enumerate() {
        // Embed the fact statement
        let embedding = match embedding_client
            .embed(crate::claims::EmbeddingRequest {
                text: fact.statement.clone(),
                context: Some(format!(
                    "{} {} {}",
                    fact.subject, fact.predicate, fact.object
                )),
            })
            .await
        {
            Ok(resp) => resp.embedding,
            Err(_) => vec![], // store claim without embedding — BM25 still works
        };

        let claim_id = match claim_store.next_id() {
            Ok(id) => id,
            Err(_) => continue,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let claim_type = match fact.category.as_deref() {
            Some("preference") => crate::claims::ClaimType::Preference,
            Some("intention") | Some("goal") => crate::claims::ClaimType::Intention,
            Some("belief") | Some("opinion") => crate::claims::ClaimType::Belief,
            _ => crate::claims::ClaimType::Fact,
        };

        let claim = crate::claims::DerivedClaim {
            id: claim_id,
            claim_text: fact.statement.clone(),
            supporting_evidence: vec![crate::claims::EvidenceSpan::new(
                0,
                fact.statement.len(),
                &fact.statement,
            )],
            confidence: fact.confidence,
            embedding,
            source_event_id: 0,
            episode_id: None,
            thread_id: None,
            user_id: None,
            workspace_id: None,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            status: crate::claims::ClaimStatus::Active,
            support_count: 1,
            metadata: HashMap::new(),
            claim_type,
            subject_entity: Some(fact.subject.to_lowercase()),
            predicate: Some(fact.predicate.clone()),
            object_entity: Some(fact.object.to_lowercase()),
            expires_at: None,
            superseded_by: None,
            entities: vec![],
            category: fact.category.clone(),
            temporal_type: crate::claims::TemporalType::Dynamic,
            valid_from: Some(base_ts + i as u64 * 1_000),
            valid_until: None,
        };

        if let Ok(()) = claim_store.store(&claim) {
            // ── State anchor stamping ──
            // Mirror the anchor logic from integration_claims.rs so that
            // apply_state_anchor_filter() can detect stale compaction claims.
            if let Some(ref subj) = claim.subject_entity {
                let subj_normalized = subj.to_lowercase();
                let inf = engine.inference.read().await;
                let projected = graph_projection::project_entity_state(
                    inf.graph(),
                    subj_normalized.trim(),
                    u64::MAX,
                    Some(engine.ontology.as_ref()),
                );
                drop(inf);
                let mut anchor_meta: HashMap<String, String> = HashMap::new();
                for slot in projected.slots.values() {
                    let cat = slot.association_type.split(':').next().unwrap_or("");
                    if engine.ontology.is_single_valued(cat) {
                        let key = format!("state_anchor:{}", slot.association_type);
                        let val = slot.value.as_deref().unwrap_or(&slot.target_name);
                        anchor_meta.insert(key, val.to_string());
                    }
                }
                if !anchor_meta.is_empty() {
                    let _ = claim_store.update_metadata(claim.id, &anchor_meta);
                    tracing::debug!(
                        "COMPACTION: stamped {} state anchors on claim {}",
                        anchor_meta.len(),
                        claim.id,
                    );
                }
            }

            // Also add a Claim node to the graph for unified retrieval
            let mut inf = engine.inference.write().await;
            let graph = inf.graph_mut();
            let claim_node =
                crate::structures::GraphNode::new(crate::structures::NodeType::Claim {
                    claim_id,
                    claim_text: fact.statement.clone(),
                    confidence: fact.confidence,
                    source_event_id: 0,
                });
            if let Ok(node_id) = graph.add_node(claim_node) {
                graph.claim_index.insert(claim_id, node_id);
                // Index in BM25
                graph.bm25_index.index_document(node_id, &fact.statement);
            }
            claims_created += 1;
        }
    }

    if claims_created > 0 {
        tracing::info!(
            "COMPACTION: created {} claims from extracted facts for hybrid search",
            claims_created
        );
    }
}
