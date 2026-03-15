// crates/agent-db-graph/src/integration/pipeline/state_tracking.rs
//
// Automatic state/transaction/relationship/preference extraction from episodes.
// Scans events for state-change and transaction signals, resolves entities,
// updates structured memory, and creates graph edges.

use super::*;
use crate::metadata_normalize::{metadata_value_preview, MetadataRole};
use crate::structures::ConceptType;
use std::collections::HashSet;

impl GraphEngine {
    /// Process episode for automatic state tracking and ledger extraction.
    ///
    /// Scans episode events for state-change and transaction signals, then
    /// updates the structured memory store.
    pub(crate) async fn process_episode_for_state_tracking(
        &self,
        episode: &Episode,
    ) -> GraphResult<()> {
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|id| store.get(id).cloned())
                .collect()
        };

        tracing::info!(
            "STATE_TRACKING episode_id={} event_count={} event_ids={:?}",
            episode.id,
            events.len(),
            episode.events.iter().take(5).collect::<Vec<_>>()
        );

        if events.is_empty() {
            tracing::info!(
                "STATE_TRACKING episode_id={} SKIPPED (no events found in store)",
                episode.id
            );
            return Ok(());
        }

        // Pre-resolve all entity names to NodeIds while holding inference lock,
        // then drop it before acquiring structured_memory write lock to avoid deadlock.
        struct StateCandidate {
            entity: String,
            attribute: String,
            new_state: String,
            node_id: NodeId,
            timestamp: u64,
            /// Semantic category for supersession grouping (e.g. "location", "routine").
            /// Facts with the same entity + category supersede each other.
            category: Option<String>,
        }
        struct LedgerCandidate {
            from: String,
            to: String,
            from_id: NodeId,
            to_id: NodeId,
            amount: f64,
            description: String,
            direction: crate::structured_memory::LedgerDirection,
            timestamp: u64,
        }
        struct RelationshipCandidate {
            subject: String,
            object: String,
            relation_type: String,
        }
        struct PreferenceCandidate {
            entity: String,
            item: String,
            category: String,
            sentiment: f64,
        }

        // Track Event→Concept mentions for creating About edges later.
        // Maps minns_node_id → Vec<concept_node_id>.
        let mut event_concept_mentions: HashSet<(NodeId, NodeId)> = HashSet::new();

        let (
            mut state_candidates,
            ledger_candidates,
            relationship_candidates,
            preference_candidates,
        ) = {
            let mut inference = self.inference.write().await;

            // Helper: ensure a concept node exists in the graph, creating it if missing.
            // Uses fuzzy entity resolution to match "coffee shop" = "Coffee Shop" etc.
            let ensure_node =
                |graph: &mut Graph, name: &str, ctype: ConceptType| -> Option<NodeId> {
                    resolve_or_create_entity(graph, name, ctype, "")
                };

            let graph = inference.graph_mut();

            let mut state_candidates = Vec::new();
            let mut ledger_candidates = Vec::new();
            let mut relationship_candidates = Vec::new();
            let mut preference_candidates = Vec::new();

            for event in &events {
                // Look up this event's graph node ID for creating About edges
                let minns_nid = graph.event_index.get(&event.id).copied();

                // Scan event content for known concept names → track for About edges.
                // This connects Conversation events to entity Concept nodes.
                if let Some(egnid) = minns_nid {
                    let content = match &event.event_type {
                        EventType::Conversation { content, .. } => Some(content.as_str()),
                        EventType::Action { action_name, .. } => Some(action_name.as_str()),
                        _ => None,
                    };
                    if let Some(text) = content {
                        let lower = text.to_lowercase();
                        for (concept_name, &concept_nid) in graph.concept_index.iter() {
                            if lower.contains(&concept_name.to_lowercase()) {
                                event_concept_mentions.insert((egnid, concept_nid));
                            }
                        }
                    }
                }

                // ---- Normalize metadata keys via alias matching ----
                let event_type_name = match &event.event_type {
                    EventType::Conversation { .. } => "Conversation",
                    EventType::Observation {
                        observation_type, ..
                    } => observation_type.as_str(),
                    EventType::Action { action_name, .. } => action_name.as_str(),
                    EventType::Context { context_type, .. } => context_type.as_str(),
                    _ => "other",
                };
                let meta_keys: Vec<&str> = event.metadata.keys().map(|k| k.as_str()).collect();
                tracing::info!(
                    "STATE_TRACKING event_id={} type={} graph_nid={:?} metadata_keys={:?}",
                    event.id,
                    event_type_name,
                    minns_nid,
                    meta_keys
                );
                let mut normalized = self.metadata_normalizer.normalize(&event.metadata);

                // LLM fallback: if alias resolved < 2 roles and event has >= 2 metadata keys
                if normalized.roles.len() < 2 && event.metadata.len() >= 2 {
                    if let Some(ref llm) = self.metadata_llm_normalizer {
                        let pairs: Vec<(String, String)> = event
                            .metadata
                            .iter()
                            .map(|(k, v)| (k.clone(), metadata_value_preview(v)))
                            .collect();
                        match tokio::time::timeout(
                            Duration::from_millis(self.config.metadata_normalization_timeout_ms),
                            llm.normalize_keys(&pairs),
                        )
                        .await
                        {
                            Ok(Ok(Some(mappings))) => {
                                normalized = self.metadata_normalizer.apply_llm_mappings(
                                    &mappings,
                                    &event.metadata,
                                    &normalized,
                                );
                            },
                            Ok(Ok(None)) => {},
                            Ok(Err(e)) => {
                                tracing::warn!("Metadata LLM normalizer error: {}", e)
                            },
                            Err(_) => {
                                tracing::warn!("Metadata LLM normalizer timed out")
                            },
                        }
                    }
                }

                // ---- State change detection ----
                let entity_name = normalized.get_str(MetadataRole::Entity, &event.metadata);
                let new_state = normalized.get_str(MetadataRole::NewState, &event.metadata);
                let _old_state = normalized.get_str(MetadataRole::OldState, &event.metadata);

                // Recognized event patterns for state changes
                let is_state_event = matches!(&event.event_type, EventType::Context { context_type, .. } if context_type == "state_update")
                    || event.metadata.contains_key("entity_state")
                    || event.metadata.contains_key("new_state")
                    || (normalized.has_role(MetadataRole::Entity)
                        && normalized.has_role(MetadataRole::NewState))
                    || match &event.event_type {
                        EventType::Action { action_name, .. } => {
                            action_name.contains("update_status")
                                || action_name.contains("set_state")
                                || action_name.contains("transition")
                        },
                        EventType::Observation {
                            observation_type, ..
                        } => observation_type == "state_change",
                        _ => false,
                    };

                tracing::info!(
                    "STATE_TRACKING event_id={} entity={:?} new_state={:?} is_state_event={}",
                    event.id,
                    entity_name.as_deref().unwrap_or("NONE"),
                    new_state.as_deref().unwrap_or("NONE"),
                    is_state_event
                );

                // Section E guards: ALL must pass
                if let (Some(ref entity), Some(ref new_st)) = (&entity_name, &new_state) {
                    if !entity.is_empty()
                        && !new_st.is_empty()
                        && is_state_event
                        && event.timestamp > 0
                    {
                        if let Some(node_id) = ensure_node(graph, entity, ConceptType::NamedEntity)
                        {
                            if let Some(egnid) = minns_nid {
                                event_concept_mentions.insert((egnid, node_id));
                            }
                            let _trigger = match &event.event_type {
                                EventType::Action { action_name, .. } => action_name.clone(),
                                _ => "auto".to_string(),
                            };
                            // Derive attribute from metadata if available, otherwise
                            // infer from context_type or event metadata keys.
                            let attribute = helpers::metadata_str(&event.metadata, "attribute")
                                .or_else(|| match &event.event_type {
                                    EventType::Context { context_type, .. } => {
                                        Some(context_type.clone())
                                    },
                                    _ => None,
                                })
                                .unwrap_or_else(|| "other".to_string());
                            let category = helpers::metadata_str(&event.metadata, "category");
                            tracing::info!(
                                "STATE_TRACKING CANDIDATE entity='{}' attr='{}' new_state='{}' category={:?} node_id={}",
                                entity, attribute, new_st, category, node_id
                            );
                            state_candidates.push(StateCandidate {
                                entity: entity.clone(),
                                attribute,
                                new_state: new_st.clone(),
                                node_id,
                                timestamp: event.timestamp,
                                category,
                            });
                        }
                    } else {
                        tracing::info!(
                            "STATE_TRACKING REJECTED entity_empty={} state_empty={} is_state_event={} ts={}",
                            entity.is_empty(),
                            new_st.is_empty(),
                            is_state_event,
                            event.timestamp
                        );
                    }
                }

                // ---- Ledger/transaction detection ----
                let amount = normalized.get_f64(MetadataRole::Amount, &event.metadata);
                let from_entity = normalized.get_str(MetadataRole::From, &event.metadata);
                let to_entity = normalized.get_str(MetadataRole::To, &event.metadata);

                let is_transaction_event = event.metadata.contains_key("amount")
                    || event.metadata.contains_key("transaction")
                    || (normalized.has_role(MetadataRole::Amount)
                        && (normalized.has_role(MetadataRole::From)
                            || normalized.has_role(MetadataRole::To)));

                if let (Some(amt), Some(ref from), Some(ref to)) =
                    (amount, &from_entity, &to_entity)
                {
                    if !from.is_empty()
                        && !to.is_empty()
                        && is_transaction_event
                        && event.timestamp > 0
                        && amt.is_finite()
                    {
                        let from_id = ensure_node(graph, from, ConceptType::NamedEntity);
                        let to_id = ensure_node(graph, to, ConceptType::NamedEntity);

                        if let (Some(fid), Some(tid)) = (from_id, to_id) {
                            if let Some(egnid) = minns_nid {
                                event_concept_mentions.insert((egnid, fid));
                                event_concept_mentions.insert((egnid, tid));
                            }
                            let description = normalized
                                .get_str(MetadataRole::Description, &event.metadata)
                                .unwrap_or_else(|| "auto-extracted".to_string());

                            // Detect direction from metadata, default to Credit
                            let direction = normalized
                                .get_str(MetadataRole::Direction, &event.metadata)
                                .map(|d| {
                                    if d.eq_ignore_ascii_case("debit") {
                                        crate::structured_memory::LedgerDirection::Debit
                                    } else {
                                        crate::structured_memory::LedgerDirection::Credit
                                    }
                                })
                                .unwrap_or(crate::structured_memory::LedgerDirection::Credit);

                            ledger_candidates.push(LedgerCandidate {
                                from: from.clone(),
                                to: to.clone(),
                                from_id: fid,
                                to_id: tid,
                                amount: amt,
                                description,
                                direction,
                                timestamp: event.timestamp,
                            });
                        } else {
                            tracing::debug!(
                                "Ledger tracking: entities '{}' or '{}' not found in graph",
                                from,
                                to
                            );
                        }
                    }
                }

                // ---- Relationship detection ----
                if event.metadata.contains_key("relationship") {
                    let subject = event
                        .metadata
                        .get("subject")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let object = event
                        .metadata
                        .get("object")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let relation_type = event
                        .metadata
                        .get("relation_type")
                        .and_then(crate::metadata_normalize::metadata_as_str);

                    if let (Some(subj), Some(obj), Some(rel)) = (subject, object, relation_type) {
                        if !subj.is_empty() && !obj.is_empty() && !rel.is_empty() {
                            // Track mentions for About edges (concept IDs resolved in Phase B)
                            if let Some(egnid) = minns_nid {
                                if let Some(&snid) = graph.concept_index.get(subj.as_str()) {
                                    event_concept_mentions.insert((egnid, snid));
                                }
                                if let Some(&onid) = graph.concept_index.get(obj.as_str()) {
                                    event_concept_mentions.insert((egnid, onid));
                                }
                            }
                            relationship_candidates.push(RelationshipCandidate {
                                subject: subj,
                                object: obj,
                                relation_type: rel,
                            });
                        }
                    }
                }

                // ---- Preference detection ----
                if event.metadata.contains_key("preference") {
                    let entity = event
                        .metadata
                        .get("entity")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let item = event
                        .metadata
                        .get("item")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let category = event
                        .metadata
                        .get("category")
                        .and_then(crate::metadata_normalize::metadata_as_str);
                    let sentiment = event
                        .metadata
                        .get("sentiment")
                        .and_then(crate::metadata_normalize::metadata_as_f64)
                        .unwrap_or(0.5);

                    if let (Some(ent), Some(itm), Some(cat)) = (entity, item, category) {
                        if !ent.is_empty() && !itm.is_empty() {
                            if let Some(egnid) = minns_nid {
                                if let Some(&enid) = graph.concept_index.get(ent.as_str()) {
                                    event_concept_mentions.insert((egnid, enid));
                                }
                                if let Some(&inid) = graph.concept_index.get(itm.as_str()) {
                                    event_concept_mentions.insert((egnid, inid));
                                }
                            }
                            preference_candidates.push(PreferenceCandidate {
                                entity: ent,
                                item: itm,
                                category: cat,
                                sentiment,
                            });
                        }
                    }
                }
            }

            (
                state_candidates,
                ledger_candidates,
                relationship_candidates,
                preference_candidates,
            )
        };
        // inference lock is now dropped

        // Apply state changes, ledger entries, relationships, and preferences
        let has_work = !state_candidates.is_empty()
            || !ledger_candidates.is_empty()
            || !relationship_candidates.is_empty()
            || !preference_candidates.is_empty();
        if has_work {
            // ---- Phase A: Write to StructuredMemoryStore (iterate by reference) ----
            {
                let mut sm = self.structured_memory.write().await;

                // State transitions are now tracked exclusively via graph edges (Phase B).
                // StructuredMemoryStore state machines are no longer populated here.
                // The graph IS the state machine — transitions are edges with valid_from timestamps.

                for lc in &ledger_candidates {
                    let key = crate::structured_memory::ledger_key(lc.from_id, lc.to_id);

                    if sm.get(&key).is_none() {
                        sm.upsert(
                            &key,
                            crate::structured_memory::MemoryTemplate::Ledger {
                                entity_pair: (lc.from.clone(), lc.to.clone()),
                                entries: vec![],
                                balance: 0.0,
                                provenance:
                                    crate::structured_memory::MemoryProvenance::EpisodePipeline,
                            },
                        );
                    }

                    let entry = crate::structured_memory::LedgerEntry {
                        timestamp: lc.timestamp,
                        amount: lc.amount,
                        description: lc.description.clone(),
                        direction: lc.direction.clone(),
                    };

                    if let Err(e) = sm.ledger_append(&key, entry) {
                        tracing::debug!("Ledger append failed for {}: {}", key, e);
                    } else {
                        tracing::debug!(
                            "Auto ledger entry: {} -> {} amount={} (key={})",
                            lc.from,
                            lc.to,
                            lc.amount,
                            key
                        );
                    }
                }

                // Relationships are stored as graph edges only — no StructuredMemoryStore tree.
                // Tree projections are built at query time by walking graph edges (graph_projection.rs).
                for rc in &relationship_candidates {
                    tracing::debug!(
                        "Auto relationship: {} <-> {} ({})",
                        rc.subject,
                        rc.object,
                        rc.relation_type
                    );
                }

                for pc in &preference_candidates {
                    let key = format!("prefs:{}:{}", pc.entity, pc.category);
                    if sm.get(&key).is_none() {
                        sm.upsert(
                            &key,
                            crate::structured_memory::MemoryTemplate::PreferenceList {
                                entity: pc.entity.clone(),
                                ranked_items: vec![],
                                provenance:
                                    crate::structured_memory::MemoryProvenance::EpisodePipeline,
                            },
                        );
                    }
                    let rank =
                        if let Some(crate::structured_memory::MemoryTemplate::PreferenceList {
                            ranked_items,
                            ..
                        }) = sm.get(&key)
                        {
                            ranked_items.len()
                        } else {
                            0
                        };
                    if let Err(e) = sm.preference_update(&key, &pc.item, rank, Some(pc.sentiment)) {
                        tracing::debug!("Preference update failed for {}: {}", key, e);
                    } else {
                        tracing::debug!(
                            "Auto preference: {} -> {} (category={}, sentiment={})",
                            pc.entity,
                            pc.item,
                            pc.category,
                            pc.sentiment
                        );
                    }
                }
            }
            // structured_memory lock dropped

            tracing::info!(
                "STATE_TRACKING PHASE_A_DONE episode_id={} state_candidates={} ledger_candidates={} relationship_candidates={} preference_candidates={} about_edge_mentions={}",
                episode.id,
                state_candidates.len(),
                ledger_candidates.len(),
                relationship_candidates.len(),
                preference_candidates.len(),
                event_concept_mentions.len()
            );

            // ---- Phase B: Create graph edges for each candidate ----
            {
                let mut inference = self.inference.write().await;

                tracing::info!(
                    "STATE_TRACKING PHASE_B_START episode_id={} state={} ledger={} relationship={} preference={} about={}",
                    episode.id,
                    state_candidates.len(),
                    ledger_candidates.len(),
                    relationship_candidates.len(),
                    preference_candidates.len(),
                    event_concept_mentions.len()
                );

                // Helper: ensure a concept node exists, returning its NodeId.
                // Uses fuzzy entity resolution to avoid duplicate nodes.
                let ensure_concept =
                    |graph: &mut Graph, name: &str, ctype: ConceptType| -> Option<NodeId> {
                        resolve_or_create_entity(graph, name, ctype, "")
                    };

                // ---- State changes → graph edges ----
                // Use category for supersession: facts with the same entity + category
                // supersede each other (latest wins). Falls back to attribute if no category.
                //
                // Deduplicate: keep only the LATEST candidate per (entity, supersession_key).
                // This prevents edge bloat when the episode is reprocessed with accumulated events.
                {
                    let mut best: std::collections::HashMap<(String, String), usize> =
                        std::collections::HashMap::new();
                    for (i, sc) in state_candidates.iter().enumerate() {
                        let key = (
                            sc.entity.to_lowercase(),
                            sc.category.as_deref().unwrap_or(&sc.attribute).to_string(),
                        );
                        match best.entry(key) {
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(i);
                            },
                            std::collections::hash_map::Entry::Occupied(mut e) => {
                                if sc.timestamp > state_candidates[*e.get()].timestamp {
                                    e.insert(i);
                                }
                            },
                        }
                    }
                    let keep: std::collections::HashSet<usize> = best.into_values().collect();
                    let deduped: Vec<_> = state_candidates
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| keep.contains(i))
                        .map(|(_, sc)| sc)
                        .collect();
                    state_candidates = deduped;
                }
                tracing::info!(
                    "PHASE_B state candidates after dedup: {}",
                    state_candidates.len()
                );

                let mut state_edges_created = 0usize;
                let mut state_edges_superseded = 0usize;
                for (idx, sc) in state_candidates.iter().enumerate() {
                    let graph = inference.graph_mut();
                    let entity_nid = sc.node_id;

                    let attribute = &sc.attribute;
                    // Edge type uses category when available, attribute as fallback
                    let supersession_key = sc.category.as_deref().unwrap_or(attribute);
                    let cp = self.ontology.canonical_predicate(supersession_key);
                    let predicate = if cp.is_empty() { supersession_key } else { &cp };
                    let assoc_type = self.ontology.build_edge_type(supersession_key, predicate);

                    tracing::info!(
                        "PHASE_B state[{}/{}] entity='{}' nid={} attr='{}' category={:?} supersession_key='{}' new_state='{}' ts={}",
                        idx + 1,
                        state_candidates.len(),
                        sc.entity,
                        entity_nid,
                        attribute,
                        sc.category,
                        supersession_key,
                        sc.new_state,
                        sc.timestamp
                    );

                    // Ensure concept node for the new state value
                    let state_ctype = self.ontology.default_target_concept_type(supersession_key);
                    let state_nid = match ensure_concept(graph, &sc.new_state, state_ctype) {
                        Some(nid) => nid,
                        None => continue,
                    };

                    // Check for idempotency: if there's already an active edge
                    // with the same type pointing to the same target, skip it.
                    // This prevents edge bloat when the same episode is reprocessed.
                    let is_cascade_trigger = self.ontology.triggers_cascade(supersession_key);
                    let mut already_exists = false;
                    let mut edges_to_supersede: Vec<EdgeId> = Vec::new();
                    for edge in graph.get_edges_from(entity_nid).iter() {
                        if let EdgeType::Association {
                            association_type, ..
                        } = &edge.edge_type
                        {
                            if edge.valid_until.is_none() {
                                if association_type == &assoc_type {
                                    if edge.target == state_nid {
                                        // Exact same active edge already exists — idempotent skip
                                        already_exists = true;
                                    } else {
                                        // Different target — must supersede
                                        edges_to_supersede.push(edge.id);
                                    }
                                } else if is_cascade_trigger {
                                    // Ontology-driven cascade: supersede edges whose
                                    // category is cascade-dependent.
                                    let edge_category: String = if let Some(rest) =
                                        association_type.strip_prefix("state:")
                                    {
                                        self.ontology
                                            .decode_legacy_state_assoc(association_type)
                                            .map(|(cat, _)| cat)
                                            .unwrap_or_else(|| rest.to_string())
                                    } else {
                                        association_type.split(':').next().unwrap_or("").to_string()
                                    };
                                    if self.ontology.is_cascade_dependent(&edge_category) {
                                        edges_to_supersede.push(edge.id);
                                    }
                                }
                            }
                        }
                    }

                    if already_exists && edges_to_supersede.is_empty() {
                        tracing::info!(
                            "PHASE_B SKIP (idempotent) '{}' entity={} -> state='{}' already active",
                            assoc_type,
                            sc.entity,
                            sc.new_state
                        );
                        continue;
                    }

                    if !edges_to_supersede.is_empty() {
                        tracing::info!(
                            "PHASE_B superseding {} active '{}' edges from entity nid={}",
                            edges_to_supersede.len(),
                            assoc_type,
                            entity_nid
                        );
                    }
                    for eid in &edges_to_supersede {
                        if let Some(e) = graph.edges.get_mut(*eid) {
                            e.valid_until = Some(sc.timestamp);
                            graph.dirty_edges.insert(*eid);
                            state_edges_superseded += 1;
                        }
                    }

                    // Create new state edge
                    let mut edge = GraphEdge::new(
                        entity_nid,
                        state_nid,
                        EdgeType::Association {
                            association_type: assoc_type.clone(),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        0.9,
                    );
                    edge.valid_from = Some(sc.timestamp);
                    edge.properties.insert(
                        "attribute".to_string(),
                        serde_json::Value::String(attribute.to_string()),
                    );
                    edge.properties.insert(
                        "value".to_string(),
                        serde_json::Value::String(sc.new_state.clone()),
                    );
                    if let Some(cat) = &sc.category {
                        edge.properties.insert(
                            "category".to_string(),
                            serde_json::Value::String(cat.clone()),
                        );
                    }
                    let new_eid = graph.add_edge(edge);
                    state_edges_created += 1;

                    tracing::info!(
                        "PHASE_B created edge eid={:?} '{}' {} -> {} (entity='{}' -> state='{}')",
                        new_eid,
                        assoc_type,
                        entity_nid,
                        state_nid,
                        sc.entity,
                        sc.new_state
                    );
                }

                tracing::info!(
                    "PHASE_B state edges done: created={} superseded={}",
                    state_edges_created,
                    state_edges_superseded
                );

                // ---- Transactions → graph edges (append-only, each payment is a separate edge) ----
                for lc in &ledger_candidates {
                    let graph = inference.graph_mut();
                    let mut edge = GraphEdge::new(
                        lc.from_id,
                        lc.to_id,
                        EdgeType::Association {
                            association_type: self.ontology.build_edge_type("financial", "payment"),
                            evidence_count: 1,
                            statistical_significance: 0.9,
                        },
                        (lc.amount as f32).clamp(0.0, 1.0),
                    );
                    edge.valid_from = Some(lc.timestamp);
                    edge.properties
                        .insert("amount".to_string(), serde_json::json!(lc.amount));
                    edge.properties.insert(
                        "description".to_string(),
                        serde_json::Value::String(lc.description.clone()),
                    );
                    graph.add_edge(edge);

                    tracing::info!(
                        "PHASE_B ledger edge: {} -> {} amount={} (ts={})",
                        lc.from,
                        lc.to,
                        lc.amount,
                        lc.timestamp
                    );
                }

                // ---- Relationships → bidirectional graph edges ----
                for rc in &relationship_candidates {
                    let graph = inference.graph_mut();
                    let subj_nid =
                        match ensure_concept(graph, &rc.subject, ConceptType::NamedEntity) {
                            Some(nid) => nid,
                            None => continue,
                        };
                    let obj_nid = match ensure_concept(graph, &rc.object, ConceptType::NamedEntity)
                    {
                        Some(nid) => nid,
                        None => continue,
                    };

                    let assoc_type = self
                        .ontology
                        .build_edge_type("relationship", &rc.relation_type);
                    let props = serde_json::json!({
                        "relation_type": rc.relation_type,
                    });
                    // Use add_or_strengthen for idempotency (bidirectional)
                    Self::add_or_strengthen_association(
                        graph,
                        subj_nid,
                        obj_nid,
                        &assoc_type,
                        0.8,
                        props.clone(),
                    );
                    Self::add_or_strengthen_association(
                        graph,
                        obj_nid,
                        subj_nid,
                        &assoc_type,
                        0.8,
                        props,
                    );

                    tracing::info!(
                        "PHASE_B relationship edge: {} <-> {} (type={})",
                        rc.subject,
                        rc.object,
                        rc.relation_type
                    );
                }

                // ---- Preferences → graph edges ----
                for pc in &preference_candidates {
                    let graph = inference.graph_mut();
                    let entity_nid =
                        match ensure_concept(graph, &pc.entity, ConceptType::NamedEntity) {
                            Some(nid) => nid,
                            None => continue,
                        };
                    let item_nid = match ensure_concept(graph, &pc.item, ConceptType::NamedEntity) {
                        Some(nid) => nid,
                        None => continue,
                    };

                    let assoc_type = self.ontology.build_edge_type("preference", &pc.category);
                    let props = serde_json::json!({
                        "category": pc.category,
                        "sentiment": pc.sentiment,
                    });
                    Self::add_or_strengthen_association(
                        graph,
                        entity_nid,
                        item_nid,
                        &assoc_type,
                        pc.sentiment as f32,
                        props,
                    );

                    tracing::info!(
                        "PHASE_B preference edge: {} -> {} (category={} sentiment={})",
                        pc.entity,
                        pc.item,
                        pc.category,
                        pc.sentiment
                    );
                }

                // ---- Event → Concept About edges ----
                // Connect event nodes to the concept nodes they mention.
                // This enables entity resolution to find relevant events via neighbor traversal.
                {
                    let graph = inference.graph_mut();
                    for (event_nid, concept_nid) in &event_concept_mentions {
                        let edge = GraphEdge::new(
                            *event_nid,
                            *concept_nid,
                            EdgeType::About {
                                relevance_score: 0.9,
                                mention_count: 1,
                                entity_role: crate::claims::types::EntityRole::default(),
                                predicate: Some("mentions".to_string()),
                            },
                            0.8,
                        );
                        graph.add_edge(edge);
                    }
                    if !event_concept_mentions.is_empty() {
                        tracing::info!("PHASE_B about edges: {}", event_concept_mentions.len());
                    }
                }

                // Final summary: count graph nodes and edges
                {
                    let graph = inference.graph_mut();
                    let total_nodes = graph.nodes.len();
                    let total_edges = graph.edges.len();
                    let concept_count = graph.concept_index.len();
                    let active_state_edges = graph
                        .edges
                        .iter()
                        .filter(|(_, e)| {
                            if let EdgeType::Association {
                                association_type, ..
                            } = &e.edge_type
                            {
                                association_type.starts_with("state:") && e.valid_until.is_none()
                            } else {
                                false
                            }
                        })
                        .count();
                    let superseded_state_edges = graph
                        .edges
                        .iter()
                        .filter(|(_, e)| {
                            if let EdgeType::Association {
                                association_type, ..
                            } = &e.edge_type
                            {
                                association_type.starts_with("state:") && e.valid_until.is_some()
                            } else {
                                false
                            }
                        })
                        .count();
                    tracing::info!(
                        "STATE_TRACKING PHASE_B_DONE episode_id={} total_nodes={} concept_nodes={} total_edges={} active_state_edges={} superseded_state_edges={}",
                        episode.id,
                        total_nodes,
                        concept_count,
                        total_edges,
                        active_state_edges,
                        superseded_state_edges
                    );
                }
            }
            // inference lock dropped

            // Embed newly created concept nodes asynchronously
            let nodes_to_embed: Vec<(NodeId, String)> = {
                let inference = self.inference.read().await;
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
                if let Some(ref ec) = self.embedding_client {
                    let ec = ec.clone();
                    let inference_ref = self.inference.clone();
                    tokio::spawn(async move {
                        for (nid, text) in nodes_to_embed {
                            match ec
                                .embed(crate::claims::EmbeddingRequest {
                                    text,
                                    context: None,
                                })
                                .await
                            {
                                Ok(resp) if !resp.embedding.is_empty() => {
                                    let mut inf = inference_ref.write().await;
                                    let graph = inf.graph_mut();
                                    if let Some(node) = graph.get_node_mut(nid) {
                                        node.embedding = resp.embedding.clone();
                                    }
                                    graph.node_vector_index.insert(nid, resp.embedding);
                                },
                                Ok(_) => {},
                                Err(e) => {
                                    tracing::debug!("Node embedding failed for nid={}: {}", nid, e);
                                },
                            }
                        }
                    });
                }
            }
        }

        Ok(())
    }
}
