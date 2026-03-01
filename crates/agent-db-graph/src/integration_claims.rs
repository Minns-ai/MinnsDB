//! Claim graph integration
//!
//! This module handles integration of semantic memory claims into the graph structure.
//! Claims are extracted from events and represented as nodes with edges showing:
//! - DERIVED_FROM: Links claim to source event
//! - SUPPORTED_BY: Links claim to evidence spans/entities
//! - ABOUT: Links claim to entities/concepts it mentions

use crate::claims::types::{ClaimExtractionRequest, DerivedClaim};
use crate::integration::GraphEngine;
use crate::structures::{EdgeType, Graph, GraphEdge, GraphNode, NodeType};
use agent_db_core::types::EventId;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

impl Graph {
    /// Create a claim node in the graph from a derived claim
    ///
    /// This creates:
    /// 1. A Claim node
    /// 2. A DERIVED_FROM edge to the source event node
    /// 3. SUPPORTED_BY edges to relevant entity/context nodes (if they exist)
    ///
    /// Returns the NodeId of the created claim node
    pub fn add_claim_node(&mut self, claim: &DerivedClaim) -> Option<crate::structures::NodeId> {
        // Create claim node
        let claim_node = GraphNode::new(NodeType::Claim {
            claim_id: claim.id,
            claim_text: claim.claim_text.clone(),
            confidence: claim.confidence,
            source_event_id: claim.source_event_id,
        });

        let claim_node_id = self.add_node(claim_node).ok()?;

        debug!(
            "Created claim node {} for claim {} (confidence: {})",
            claim_node_id, claim.id, claim.confidence
        );

        // Create DERIVED_FROM edge to source event
        if let Some(event_node_id) = self.get_event_node(claim.source_event_id).map(|n| n.id) {
            let mut derived_edge = GraphEdge::new(
                claim_node_id,
                event_node_id,
                EdgeType::DerivedFrom {
                    extraction_confidence: claim.confidence,
                    extraction_timestamp: claim.created_at,
                },
                claim.confidence,
            );
            // Bi-temporal: set valid-time from claim lifecycle
            derived_edge.valid_from = Some(claim.created_at * 1_000_000_000);
            if let Some(exp) = claim.expires_at {
                derived_edge.valid_until = Some(exp * 1_000_000_000);
            }

            if let Some(edge_id) = self.add_edge(derived_edge) {
                debug!(
                    "Created DERIVED_FROM edge {} from claim {} to event {}",
                    edge_id, claim.id, claim.source_event_id
                );
            }
        } else {
            debug!(
                "No event node found for event {} when creating claim {}",
                claim.source_event_id, claim.id
            );
        }

        // Create SUPPORTED_BY edges for evidence spans
        // Note: This creates edges to the source event node with span metadata
        // In a more advanced implementation, you could create separate evidence nodes
        for (idx, evidence) in claim.supporting_evidence.iter().enumerate() {
            if let Some(event_node_id) = self.get_event_node(claim.source_event_id).map(|n| n.id) {
                let mut evidence_edge = GraphEdge::new(
                    claim_node_id,
                    event_node_id,
                    EdgeType::SupportedBy {
                        evidence_strength: claim.confidence,
                        span_offset: (evidence.start_offset, evidence.end_offset),
                    },
                    0.8, // Evidence support weight
                );
                // Bi-temporal: evidence edges share claim's validity window
                evidence_edge.valid_from = Some(claim.created_at * 1_000_000_000);
                if let Some(exp) = claim.expires_at {
                    evidence_edge.valid_until = Some(exp * 1_000_000_000);
                }

                if let Some(edge_id) = self.add_edge(evidence_edge) {
                    debug!(
                        "Created SUPPORTED_BY edge {} for evidence span {} of claim {}",
                        edge_id, idx, claim.id
                    );
                }
            }
        }

        info!(
            "Added claim {} to graph with {} evidence spans",
            claim.id,
            claim.supporting_evidence.len()
        );

        // Auto-link to NER entities attached to the claim (with proper labels and roles)
        for entity in &claim.entities {
            use crate::claims::types::EntityRole;

            // Find or create concept node for entity, using the NER label
            let concept_type = crate::structures::ConceptType::from_ner_label(&entity.label);

            let entity_node_id = if let Some(node) = self.get_concept_node(&entity.text) {
                Some(node.id)
            } else {
                let new_node = GraphNode::new(NodeType::Concept {
                    concept_name: entity.text.clone(),
                    concept_type,
                    confidence: entity.confidence,
                });
                self.add_node(new_node).ok()
            };

            // Assign relevance by role: Subject=0.95, Object=0.90, Mentioned=0.70
            let relevance = match entity.role {
                EntityRole::Subject => 0.95,
                EntityRole::Object => 0.90,
                EntityRole::Mentioned => 0.70,
            };

            // Only pass predicate for Subject/Object roles
            let pred = match entity.role {
                EntityRole::Subject | EntityRole::Object => claim.predicate.clone(),
                EntityRole::Mentioned => None,
            };

            // Link claim to entity with role
            if let Some(eid) = entity_node_id {
                self.link_claim_to_entity(claim.id, eid, relevance, entity.role, pred);
            }
        }

        // Backward-compat: if no structured entities, fall back to metadata
        if claim.entities.is_empty() {
            if let Some(entities_str) = claim.metadata.get("found_entities") {
                for entity_name in entities_str.split(',') {
                    let entity_name = entity_name.trim();
                    if entity_name.is_empty() {
                        continue;
                    }

                    let entity_node_id = if let Some(node) = self.get_concept_node(entity_name) {
                        Some(node.id)
                    } else {
                        let new_node = GraphNode::new(NodeType::Concept {
                            concept_name: entity_name.to_string(),
                            concept_type: crate::structures::ConceptType::ContextualAssociation,
                            confidence: claim.confidence,
                        });
                        self.add_node(new_node).ok()
                    };

                    if let Some(eid) = entity_node_id {
                        self.link_claim_to_entity(
                            claim.id,
                            eid,
                            0.70,
                            crate::claims::types::EntityRole::Mentioned,
                            None,
                        );
                    }
                }
            }
        }

        Some(claim_node_id)
    }

    /// Link a claim to an entity/concept node via ABOUT edge
    ///
    /// This creates semantic relationships between claims and the entities they mention,
    /// with role information (Subject/Object/Mentioned) and optional predicate.
    pub fn link_claim_to_entity(
        &mut self,
        claim_id: u64,
        entity_node_id: crate::structures::NodeId,
        relevance_score: f32,
        entity_role: crate::claims::types::EntityRole,
        predicate: Option<String>,
    ) -> Option<crate::structures::EdgeId> {
        let claim_node_id = self.get_claim_node(claim_id).map(|n| n.id)?;

        let about_edge = GraphEdge::new(
            claim_node_id,
            entity_node_id,
            EdgeType::About {
                relevance_score,
                mention_count: 1,
                entity_role,
                predicate,
            },
            relevance_score,
        );
        // Note: ABOUT edges don't carry bi-temporal fields since the entity
        // relationship itself is atemporal (a claim is always "about" its entity).

        let edge_id = self.add_edge(about_edge)?;

        debug!(
            "Created ABOUT edge {} linking claim {} to entity node {} (role: {})",
            edge_id, claim_id, entity_node_id, entity_role
        );

        Some(edge_id)
    }

    /// Get all claims derived from a specific event
    pub fn get_claims_from_event(&self, event_id: EventId) -> Vec<&GraphNode> {
        // Get event node
        let event_node = match self.get_event_node(event_id) {
            Some(node) => node,
            None => return Vec::new(),
        };

        // Find all incoming DERIVED_FROM edges to the event
        self.get_edges_to(event_node.id)
            .into_iter()
            .filter_map(|edge| {
                // Check if it's a DERIVED_FROM edge
                if matches!(edge.edge_type, EdgeType::DerivedFrom { .. }) {
                    self.get_node(edge.source)
                } else {
                    None
                }
            })
            .filter(|node| matches!(node.node_type, NodeType::Claim { .. }))
            .collect()
    }

    /// Get all claims about a specific entity/concept
    pub fn get_claims_about_entity(
        &self,
        entity_node_id: crate::structures::NodeId,
    ) -> Vec<&GraphNode> {
        // Find all incoming ABOUT edges to the entity
        self.get_edges_to(entity_node_id)
            .into_iter()
            .filter_map(|edge| {
                // Check if it's an ABOUT edge
                if matches!(edge.edge_type, EdgeType::About { .. }) {
                    self.get_node(edge.source)
                } else {
                    None
                }
            })
            .filter(|node| matches!(node.node_type, NodeType::Claim { .. }))
            .collect()
    }

    /// Get all claims about a specific entity/concept filtered by entity role
    pub fn get_claims_about_entity_with_role(
        &self,
        entity_node_id: crate::structures::NodeId,
        role: crate::claims::types::EntityRole,
    ) -> Vec<&GraphNode> {
        self.get_edges_to(entity_node_id)
            .into_iter()
            .filter_map(|edge| {
                if let EdgeType::About { entity_role, .. } = &edge.edge_type {
                    if *entity_role == role {
                        return self.get_node(edge.source);
                    }
                }
                None
            })
            .filter(|node| matches!(node.node_type, NodeType::Claim { .. }))
            .collect()
    }

    /// Get evidence span offsets for a claim
    pub fn get_claim_evidence_spans(&self, claim_id: u64) -> Vec<(usize, usize)> {
        let claim_node = match self.get_claim_node(claim_id) {
            Some(node) => node,
            None => return Vec::new(),
        };

        self.get_edges_from(claim_node.id)
            .into_iter()
            .filter_map(|edge| {
                if let EdgeType::SupportedBy { span_offset, .. } = edge.edge_type {
                    Some(span_offset)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::types::EvidenceSpan;
    use crate::structures::NodeType;

    fn create_test_claim(id: u64, event_id: EventId, text: &str) -> DerivedClaim {
        let evidence = vec![EvidenceSpan::new(0, 10, "test span")];
        DerivedClaim::new(
            id,
            text.to_string(),
            evidence,
            0.9,
            vec![0.1, 0.2, 0.3],
            event_id,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn test_add_claim_node() {
        let mut graph = Graph::new();

        // Create an event node first
        let event_node = GraphNode::new(NodeType::Event {
            event_id: 123,
            event_type: "Context".to_string(),
            significance: 0.8,
        });
        let _event_node_id = graph.add_node(event_node).unwrap();

        // Create and add claim
        let claim = create_test_claim(1, 123, "Test claim");
        let claim_node_id = graph.add_claim_node(&claim);

        assert!(claim_node_id.is_some());

        // Verify claim node exists
        let claim_node = graph.get_claim_node(1);
        assert!(claim_node.is_some());

        // Verify DERIVED_FROM edge was created
        let claims_from_event = graph.get_claims_from_event(123);
        assert_eq!(claims_from_event.len(), 1);
    }

    #[test]
    fn test_link_claim_to_entity() {
        let mut graph = Graph::new();

        // Create event, claim, and concept nodes
        let event_node = GraphNode::new(NodeType::Event {
            event_id: 123,
            event_type: "Context".to_string(),
            significance: 0.8,
        });
        graph.add_node(event_node).unwrap();

        let claim = create_test_claim(1, 123, "John works at Google");
        graph.add_claim_node(&claim);

        let concept_node = GraphNode::new(NodeType::Concept {
            concept_name: "Google".to_string(),
            concept_type: crate::structures::ConceptType::ContextualAssociation,
            confidence: 0.9,
        });
        let concept_node_id = graph.add_node(concept_node).unwrap();

        // Link claim to entity with role
        let edge_id = graph.link_claim_to_entity(
            1,
            concept_node_id,
            0.95,
            crate::claims::types::EntityRole::Object,
            Some("works at".to_string()),
        );
        assert!(edge_id.is_some());

        // Verify ABOUT edge was created
        let claims_about_entity = graph.get_claims_about_entity(concept_node_id);
        assert_eq!(claims_about_entity.len(), 1);
    }

    #[test]
    fn test_get_claim_evidence_spans() {
        let mut graph = Graph::new();

        // Create event and claim
        let event_node = GraphNode::new(NodeType::Event {
            event_id: 123,
            event_type: "Context".to_string(),
            significance: 0.8,
        });
        graph.add_node(event_node).unwrap();

        let claim = create_test_claim(1, 123, "Test claim with evidence");
        graph.add_claim_node(&claim);

        // Get evidence spans
        let spans = graph.get_claim_evidence_spans(1);
        assert!(!spans.is_empty());
        assert_eq!(spans[0], (0, 10));
    }

    /// Verify that extracted_fact observations produce clean NL canonical text
    /// instead of JSON-in-text, enabling proper NER + distance scoring.
    #[test]
    fn test_extracted_fact_canonical_text() {
        use agent_db_events::EventType;

        // Simulate the canonical text extraction logic from extract_claims_async
        let event_type = EventType::Observation {
            observation_type: "extracted_fact".to_string(),
            data: serde_json::json!({
                "statement": "Alice lives in Paris",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Paris",
            }),
            confidence: 0.9,
            source: "conversation_compaction".to_string(),
        };

        let canonical_text = match &event_type {
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                if observation_type == "extracted_fact" {
                    if let Some(stmt) = data.get("statement").and_then(|s| s.as_str()) {
                        stmt.to_string()
                    } else {
                        format!(
                            "Observation [{}] from source '{}' (confidence {:.2}): {}",
                            observation_type, source, confidence, data
                        )
                    }
                } else {
                    format!(
                        "Observation [{}] from source '{}' (confidence {:.2}): {}",
                        observation_type, source, confidence, data
                    )
                }
            },
            _ => unreachable!(),
        };

        // Should be clean NL text, not JSON-in-text
        assert_eq!(canonical_text, "Alice lives in Paris");
        assert!(!canonical_text.contains("Observation ["));
        assert!(!canonical_text.contains("confidence"));

        // Non-extracted_fact observations should still use the old format
        let regular_obs = EventType::Observation {
            observation_type: "sensor_reading".to_string(),
            data: serde_json::json!({"temperature": 22.5}),
            confidence: 0.95,
            source: "thermometer".to_string(),
        };

        let regular_text = match &regular_obs {
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                if observation_type == "extracted_fact" {
                    if let Some(stmt) = data.get("statement").and_then(|s| s.as_str()) {
                        stmt.to_string()
                    } else {
                        format!(
                            "Observation [{}] from source '{}' (confidence {:.2}): {}",
                            observation_type, source, confidence, data
                        )
                    }
                } else {
                    format!(
                        "Observation [{}] from source '{}' (confidence {:.2}): {}",
                        observation_type, source, confidence, data
                    )
                }
            },
            _ => unreachable!(),
        };

        assert!(regular_text.contains("Observation [sensor_reading]"));
        assert!(regular_text.contains("thermometer"));
    }

    #[test]
    fn test_link_claim_to_entity_with_role_and_predicate() {
        let mut graph = Graph::new();

        let event_node = GraphNode::new(NodeType::Event {
            event_id: 200,
            event_type: "Context".to_string(),
            significance: 0.8,
        });
        graph.add_node(event_node).unwrap();

        let claim = create_test_claim(10, 200, "User works at Google");
        graph.add_claim_node(&claim);

        let concept_node = GraphNode::new(NodeType::Concept {
            concept_name: "Google".to_string(),
            concept_type: crate::structures::ConceptType::ContextualAssociation,
            confidence: 0.9,
        });
        let concept_id = graph.add_node(concept_node).unwrap();

        // Link with Subject role
        let edge_id = graph.link_claim_to_entity(
            10,
            concept_id,
            0.95,
            crate::claims::types::EntityRole::Subject,
            Some("works at".to_string()),
        );
        assert!(edge_id.is_some());

        // Verify the edge has the correct role and predicate
        let edge = graph.get_edge(edge_id.unwrap()).unwrap();
        if let EdgeType::About {
            entity_role,
            predicate,
            ..
        } = &edge.edge_type
        {
            assert_eq!(*entity_role, crate::claims::types::EntityRole::Subject);
            assert_eq!(predicate.as_deref(), Some("works at"));
        } else {
            panic!("Expected About edge type");
        }
    }

    #[test]
    fn test_get_claims_about_entity_with_role_filters() {
        use crate::claims::types::EntityRole;

        let mut graph = Graph::new();

        let event_node = GraphNode::new(NodeType::Event {
            event_id: 300,
            event_type: "Context".to_string(),
            significance: 0.8,
        });
        graph.add_node(event_node).unwrap();

        // Create two claims
        let claim1 = create_test_claim(20, 300, "John works at Google");
        graph.add_claim_node(&claim1);
        let claim2 = create_test_claim(21, 300, "Alice visited Google");
        graph.add_claim_node(&claim2);

        let concept_node = GraphNode::new(NodeType::Concept {
            concept_name: "Google".to_string(),
            concept_type: crate::structures::ConceptType::ContextualAssociation,
            confidence: 0.9,
        });
        let concept_id = graph.add_node(concept_node).unwrap();

        // Link claim1 as Object, claim2 as Subject
        graph.link_claim_to_entity(
            20,
            concept_id,
            0.90,
            EntityRole::Object,
            Some("works at".to_string()),
        );
        graph.link_claim_to_entity(21, concept_id, 0.95, EntityRole::Subject, None);

        // All claims about entity
        let all = graph.get_claims_about_entity(concept_id);
        assert_eq!(all.len(), 2);

        // Only Subject role
        let subjects = graph.get_claims_about_entity_with_role(concept_id, EntityRole::Subject);
        assert_eq!(subjects.len(), 1);

        // Only Object role
        let objects = graph.get_claims_about_entity_with_role(concept_id, EntityRole::Object);
        assert_eq!(objects.len(), 1);

        // Mentioned role (none linked)
        let mentioned = graph.get_claims_about_entity_with_role(concept_id, EntityRole::Mentioned);
        assert_eq!(mentioned.len(), 0);
    }

    #[test]
    fn test_add_claim_node_with_entity_roles() {
        use crate::claims::types::{ClaimEntity, EntityRole};

        let mut graph = Graph::new();

        let event_node = GraphNode::new(NodeType::Event {
            event_id: 400,
            event_type: "Context".to_string(),
            significance: 0.8,
        });
        graph.add_node(event_node).unwrap();

        let mut claim = create_test_claim(30, 400, "John works at Google");
        claim.subject_entity = Some("john".to_string());
        claim.predicate = Some("works at".to_string());
        claim.object_entity = Some("google".to_string());
        claim.entities = vec![
            ClaimEntity {
                text: "John".to_string(),
                label: "PERSON".to_string(),
                normalized: "john".to_string(),
                confidence: 0.95,
                role: EntityRole::Subject,
            },
            ClaimEntity {
                text: "Google".to_string(),
                label: "ORG".to_string(),
                normalized: "google".to_string(),
                confidence: 0.90,
                role: EntityRole::Object,
            },
        ];

        let claim_node_id = graph.add_claim_node(&claim);
        assert!(claim_node_id.is_some());

        // Find the Google concept node
        let google_node = graph.get_concept_node("Google");
        assert!(google_node.is_some());
        let google_id = google_node.unwrap().id;

        // Verify role filtering works
        let object_claims = graph.get_claims_about_entity_with_role(google_id, EntityRole::Object);
        assert_eq!(object_claims.len(), 1);

        // John should be linked as Subject
        let john_node = graph.get_concept_node("John");
        assert!(john_node.is_some());
        let john_id = john_node.unwrap().id;
        let subject_claims = graph.get_claims_about_entity_with_role(john_id, EntityRole::Subject);
        assert_eq!(subject_claims.len(), 1);
    }
}

// GraphEngine integration methods
impl GraphEngine {
    /// Extract claims from an event and integrate into graph (async, non-blocking)
    ///
    /// This method:
    /// 1. Checks if event qualifies for claim extraction
    /// 2. Retrieves NER features if available
    /// 3. Submits to claim extraction queue
    /// 4. Creates graph nodes and edges for accepted claims
    pub(crate) async fn extract_claims_async(&self, event: &agent_db_events::Event) {
        use agent_db_events::EventType;
        use tracing::{debug, warn};

        // Check if claim queue is available
        let claim_queue: Arc<crate::claims::ClaimExtractionQueue> = match &self.claim_queue {
            Some(q) => q.clone(),
            None => {
                debug!(
                    "Claim extraction queue not initialized, skipping for event {}",
                    event.id
                );
                return;
            },
        };

        // Explicitly clone only the Arcs we need for the background task
        // This avoids cloning the entire GraphEngine struct
        let ner_queue: Option<Arc<agent_db_ner::NerExtractionQueue>> = self.ner_queue.clone();
        let ner_store: Option<Arc<agent_db_ner::NerFeatureStore>> = self.ner_store.clone();
        let inference: Arc<RwLock<crate::inference::GraphInference>> = self.inference.clone();
        let ner_promotion_threshold = self.config.ner_promotion_threshold;

        // Snapshot rolling summary for context enrichment (best-effort)
        let rolling_summary: Option<String> = if self.config.enable_rolling_summary {
            let summaries = self.conversation_summaries.read().await;
            summaries
                .values()
                .max_by_key(|s| s.last_updated)
                .map(|s| s.summary.clone())
        } else {
            None
        };

        // Extract canonical text from event.
        // Context events carry inline text; other event types synthesise a
        // textual representation from their structured fields so the claims
        // pipeline can extract factual knowledge from *any* event.
        let canonical_text = match &event.event_type {
            EventType::Context { text, .. } => {
                if text.is_empty() {
                    return;
                }
                text.clone()
            },

            // ── Action events ───────────────────────────────────────────
            EventType::Action {
                action_name,
                parameters,
                outcome,
                duration_ns,
            } => {
                use agent_db_events::core::ActionOutcome;
                let outcome_text = match outcome {
                    ActionOutcome::Success { result } => {
                        format!("succeeded with result: {}", result)
                    },
                    ActionOutcome::Failure { error, error_code } => {
                        format!("failed with error code {}: {}", error_code, error)
                    },
                    ActionOutcome::Partial { result, issues } => {
                        format!(
                            "partially succeeded ({}), issues: {}",
                            result,
                            issues.join("; ")
                        )
                    },
                };
                format!(
                    "Action '{}' with parameters {} {} (took {} ns)",
                    action_name, parameters, outcome_text, duration_ns
                )
            },

            // ── Observation events ──────────────────────────────────────
            EventType::Observation {
                observation_type,
                data,
                confidence,
                source,
            } => {
                // For compaction-extracted facts, use the statement directly
                // so the claims pipeline gets clean NL text for NER + scoring
                if observation_type == "extracted_fact" {
                    if let Some(stmt) = data.get("statement").and_then(|s| s.as_str()) {
                        stmt.to_string()
                    } else {
                        format!(
                            "Observation [{}] from source '{}' (confidence {:.2}): {}",
                            observation_type, source, confidence, data
                        )
                    }
                } else {
                    format!(
                        "Observation [{}] from source '{}' (confidence {:.2}): {}",
                        observation_type, source, confidence, data
                    )
                }
            },

            // ── Cognitive / Decision events ─────────────────────────────
            EventType::Cognitive {
                process_type,
                input,
                output,
                reasoning_trace,
            } => {
                let trace = if reasoning_trace.is_empty() {
                    String::new()
                } else {
                    format!(" Reasoning: {}", reasoning_trace.join(" → "))
                };
                format!(
                    "Cognitive process {:?}: input={} output={}.{}",
                    process_type, input, output, trace
                )
            },

            // ── Learning telemetry events ───────────────────────────────
            EventType::Learning { event: learning } => {
                use agent_db_events::core::LearningEvent;
                match learning {
                    LearningEvent::MemoryRetrieved {
                        query_id,
                        memory_ids,
                    } => {
                        format!(
                            "Learning: retrieved memories {:?} for query '{}'",
                            memory_ids, query_id
                        )
                    },
                    LearningEvent::MemoryUsed {
                        query_id,
                        memory_id,
                    } => {
                        format!(
                            "Learning: used memory {} for query '{}'",
                            memory_id, query_id
                        )
                    },
                    LearningEvent::StrategyServed {
                        query_id,
                        strategy_ids,
                    } => {
                        format!(
                            "Learning: served strategies {:?} for query '{}'",
                            strategy_ids, query_id
                        )
                    },
                    LearningEvent::StrategyUsed {
                        query_id,
                        strategy_id,
                    } => {
                        format!(
                            "Learning: used strategy {} for query '{}'",
                            strategy_id, query_id
                        )
                    },
                    LearningEvent::Outcome { query_id, success } => {
                        format!(
                            "Learning outcome for query '{}': success={}",
                            query_id, success
                        )
                    },
                    LearningEvent::ClaimRetrieved {
                        query_id,
                        claim_ids,
                    } => {
                        format!(
                            "Learning: retrieved claims {:?} for query '{}'",
                            claim_ids, query_id
                        )
                    },
                    LearningEvent::ClaimUsed { query_id, claim_id } => {
                        format!("Learning: used claim {} for query '{}'", claim_id, query_id)
                    },
                }
            },

            // ── Communication events ────────────────────────────────────
            EventType::Communication {
                message_type,
                sender,
                recipient,
                content,
            } => {
                format!(
                    "Communication [{}] from agent {} to agent {}: {}",
                    message_type, sender, recipient, content
                )
            },

            // ── Conversation events ─────────────────────────────────────
            EventType::Conversation {
                speaker, content, ..
            } => {
                format!("{}: {}", speaker, content)
            },
        };

        // Skip empty synthesised text
        if canonical_text.is_empty() {
            return;
        }

        // Promotion threshold: if the event's raw context exceeds the
        // threshold, we prefer the segment-stored content (future: dereference
        // segment_pointer).  For now, fall through to use the synthesised text
        // above — the LLM can still extract claims from it.
        if event.context_size_bytes >= ner_promotion_threshold {
            debug!(
                "Event {} exceeds promotion threshold ({} >= {}), using synthesised text ({} bytes)",
                event.id, event.context_size_bytes, ner_promotion_threshold, canonical_text.len()
            );
        }

        debug!(
            "Submitting event {} for claim extraction ({} bytes)",
            event.id,
            canonical_text.len()
        );

        let event_id = event.id;
        let agent_id = event.agent_id;
        let session_id = event.session_id;

        // Spawn background task
        tokio::spawn(async move {
            // Step 1: Wait for NER features to be ready
            let ner_features = if let Some(queue) = ner_queue {
                debug!("Waiting for NER features for event {}...", event_id);
                match queue.extract(event_id, canonical_text.clone()).await {
                    Ok(features) => {
                        debug!(
                            "NER features ready for event {}: {} entities",
                            event_id,
                            features.entity_spans.len()
                        );

                        // Store features if store is available
                        if let Some(ref store) = ner_store {
                            let _ = store.store(&features);
                        }

                        Some(features)
                    },
                    Err(e) => {
                        warn!(
                            "NER extraction failed during claim pipeline for event {}: {}",
                            event_id, e
                        );
                        None
                    },
                }
            } else {
                None
            };

            // Build claim extraction request
            let request = ClaimExtractionRequest {
                canonical_text,
                ner_features,
                context_embedding: None, // Will be filled if available or triggered later
                event_id,
                episode_id: None,
                thread_id: None,
                user_id: Some(agent_id.to_string()),
                workspace_id: Some(format!("session_{}", session_id)),
                source_role: crate::claims::types::SourceRole::default(),
                rolling_summary,
            };

            // Submit to claim extraction queue
            match claim_queue.extract(request).await {
                Ok(result) => {
                    if !result.accepted_claims.is_empty() || !result.rejected_claims.is_empty() {
                        tracing::info!(
                            "Claims extracted for event {}: {} accepted, {} rejected",
                            event_id,
                            result.accepted_claims.len(),
                            result.rejected_claims.len()
                        );
                    } else {
                        debug!(
                            "Claim extraction complete for event {}: no claims produced",
                            event_id,
                        );
                    }

                    // Step 2: Integrate accepted claims into the graph
                    for claim in result.accepted_claims {
                        // We do the integration directly using the cloned inference Arc
                        let mut inference_lock = inference.write().await;
                        let graph = inference_lock.graph_mut();

                        if let Some(node_id) = graph.add_claim_node(&claim) {
                            debug!(
                                "Integrated claim {} into graph as node {}",
                                claim.id, node_id
                            );
                        }
                    }
                },
                Err(e) => {
                    warn!("Claim extraction failed for event {}: {}", event_id, e);
                },
            }
        });
    }

    /// Create graph nodes and edges for a claim
    ///
    /// This should be called after a claim has been stored
    pub async fn integrate_claim_to_graph(
        &self,
        claim: &DerivedClaim,
    ) -> Result<(), crate::GraphError> {
        use tracing::info;

        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // Add claim node to graph
        if let Some(claim_node_id) = graph.add_claim_node(claim) {
            info!(
                "Integrated claim {} into graph as node {}",
                claim.id, claim_node_id
            );
        }

        Ok(())
    }

    /// Process pending embeddings for claims
    ///
    /// Generates embeddings for all claims that don't have them yet
    pub async fn process_pending_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<usize, crate::GraphError> {
        use tracing::info;

        let embedding_queue = match &self.embedding_queue {
            Some(q) => q,
            None => {
                return Err(crate::GraphError::InvalidOperation(
                    "Embedding queue not initialized".to_string(),
                ));
            },
        };

        let claim_store = match &self.claim_store {
            Some(s) => s,
            None => {
                return Err(crate::GraphError::InvalidOperation(
                    "Claim store not initialized".to_string(),
                ));
            },
        };

        let embedding_client = match &self.embedding_client {
            Some(c) => c,
            None => {
                return Err(crate::GraphError::InvalidOperation(
                    "Embedding client not initialized".to_string(),
                ));
            },
        };

        let count = embedding_queue
            .process_pending_embeddings(claim_store, &**embedding_client, batch_size)
            .await
            .map_err(|e| {
                crate::GraphError::OperationError(format!("Failed to process embeddings: {}", e))
            })?;

        info!("Processed {} claims for embedding generation", count);

        Ok(count)
    }

    /// Search for claims using hybrid BM25 + semantic search with graph-enhanced
    /// retrieval (Graph RAG).
    ///
    /// Runs up to three retrieval legs:
    /// 1. **BM25 keyword** — full-text search on claim text
    /// 2. **Vector semantic** — cosine similarity on embeddings (skipped when no
    ///    embedding client is configured; degrades gracefully to keyword-only)
    /// 3. **Graph entity traversal** — finds claims linked to the same entities
    ///    via ABOUT edges in the knowledge graph
    ///
    /// Results are fused via Reciprocal Rank Fusion (RRF) and re-ranked with
    /// temporal decay weighting.
    pub async fn search_similar_claims(
        &self,
        query_text: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(crate::claims::DerivedClaim, f32)>, crate::GraphError> {
        use crate::claims::types::ClaimId;
        use tracing::{debug, info};

        let claim_store = match &self.claim_store {
            Some(s) => s,
            None => {
                return Err(crate::GraphError::InvalidOperation(
                    "Claim store not initialized".to_string(),
                ));
            },
        };

        // Determine search mode based on embedding client availability
        let query_embedding = if let Some(ref embedding_client) = self.embedding_client {
            let request = crate::claims::EmbeddingRequest {
                text: query_text.to_string(),
                context: None,
            };

            match embedding_client.embed(request).await {
                Ok(response) => {
                    debug!(
                        "Generated query embedding ({} dimensions)",
                        response.embedding.len()
                    );
                    Some(response.embedding)
                },
                Err(e) => {
                    info!(
                        "Embedding generation failed, falling back to keyword-only search: {}",
                        e
                    );
                    None
                },
            }
        } else {
            debug!("No embedding client available, using keyword-only search");
            None
        };

        let search_mode = if query_embedding.is_some() {
            crate::indexing::SearchMode::Hybrid
        } else {
            crate::indexing::SearchMode::Keyword
        };

        let hybrid_config = crate::claims::hybrid_search::HybridSearchConfig {
            mode: search_mode,
            min_similarity,
            ..Default::default()
        };

        // Request extra candidates so temporal re-ranking can still fill top_k
        let fetch_k = (top_k * 3).max(20);
        let empty_embedding: Vec<f32> = Vec::new();
        let search_embedding = query_embedding.as_deref().unwrap_or(&empty_embedding);

        let similar_ids = crate::claims::hybrid_search::HybridClaimSearch::search(
            query_text,
            search_embedding,
            claim_store,
            fetch_k,
            &hybrid_config,
        )
        .map_err(|e| {
            crate::GraphError::OperationError(format!("Failed to search claims: {}", e))
        })?;

        debug!(
            "Found {} candidate claims from hybrid search",
            similar_ids.len()
        );

        // ── Graph RAG: entity-based claim discovery ──────────────────────
        // Find claims linked to the same entities via ABOUT edges in the
        // knowledge graph.  This surfaces claims that share entities with
        // the query even when there is zero keyword or embedding overlap.
        let graph_claim_ids: Vec<(ClaimId, f32)> = {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            self.graph_entity_claims(query_text, graph, fetch_k)
        };

        if !graph_claim_ids.is_empty() {
            debug!(
                "Graph RAG found {} additional claims via entity traversal",
                graph_claim_ids.len()
            );
        }

        // ── Fuse all retrieval legs via RRF ──────────────────────────────
        let fused_ids = if graph_claim_ids.is_empty() {
            similar_ids
        } else {
            let lists: Vec<Vec<(ClaimId, f32)>> = vec![similar_ids, graph_claim_ids];
            let mut fused = crate::retrieval::multi_list_rrf(&lists, 60.0);
            fused.truncate(fetch_k);
            fused
        };

        debug!(
            "Fused {} candidate claims for re-ranking",
            fused_ids.len()
        );

        // Retrieve full claims and compute temporally-weighted retrieval score
        let mut results: Vec<(crate::claims::DerivedClaim, f32)> = Vec::new();
        for (claim_id, similarity) in fused_ids {
            if let Some(claim) = claim_store.get(claim_id).map_err(|e| {
                crate::GraphError::OperationError(format!("Failed to retrieve claim: {}", e))
            })? {
                let score = claim.retrieval_score(similarity);
                results.push((claim, score));
            }
        }

        // Re-sort by temporally-weighted score (descending) and truncate
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        debug!(
            "Returning {} claims after temporal re-ranking",
            results.len()
        );

        Ok(results)
    }

    /// Graph RAG: discover claims related to the query via entity traversal.
    ///
    /// 1. Tokenizes the query into words
    /// 2. Looks up each word as a concept node in the graph
    /// 3. Traverses ABOUT edges from matching concept nodes to find claim nodes
    /// 4. Returns (ClaimId, relevance_score) pairs
    fn graph_entity_claims(
        &self,
        query_text: &str,
        graph: &crate::structures::Graph,
        limit: usize,
    ) -> Vec<(crate::claims::types::ClaimId, f32)> {
        use crate::structures::{EdgeType, NodeType};
        use std::collections::HashMap;

        let mut claim_scores: HashMap<u64, f32> = HashMap::new();

        // Tokenize query into candidate entity names (single words + bigrams)
        let words: Vec<&str> = query_text.split_whitespace().collect();
        let mut candidates: Vec<String> = words.iter().map(|w| w.to_string()).collect();

        // Add bigrams for multi-word entity names (e.g. "Alice Chen")
        for window in words.windows(2) {
            candidates.push(format!("{} {}", window[0], window[1]));
        }

        // Also try the full query as a concept name
        candidates.push(query_text.to_string());

        for candidate in &candidates {
            // Try exact match and case variations
            let variations = [
                candidate.clone(),
                capitalize_first(candidate),
                candidate.to_lowercase(),
            ];

            for name in &variations {
                if let Some(concept_node) = graph.get_concept_node(name) {
                    // Found a concept node — traverse ABOUT edges to find claims
                    for edge in graph.get_edges_to(concept_node.id) {
                        if let EdgeType::About {
                            relevance_score, ..
                        } = &edge.edge_type
                        {
                            if let Some(source_node) = graph.get_node(edge.source) {
                                if let NodeType::Claim { claim_id, .. } = &source_node.node_type {
                                    // Accumulate score — claims linked via multiple entities
                                    // get a higher score
                                    let entry =
                                        claim_scores.entry(*claim_id).or_insert(0.0);
                                    *entry += relevance_score;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize and sort by score
        let mut results: Vec<(u64, f32)> = claim_scores
            .into_iter()
            .map(|(id, score)| {
                // Clamp to [0, 1] — multiple entity matches can sum above 1.0
                (id, score.min(1.0))
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}

/// Capitalize the first letter of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}
