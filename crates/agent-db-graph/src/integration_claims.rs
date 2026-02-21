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
            let derived_edge = GraphEdge::new(
                claim_node_id,
                event_node_id,
                EdgeType::DerivedFrom {
                    extraction_confidence: claim.confidence,
                    extraction_timestamp: claim.created_at,
                },
                claim.confidence,
            );

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
                let evidence_edge = GraphEdge::new(
                    claim_node_id,
                    event_node_id,
                    EdgeType::SupportedBy {
                        evidence_strength: claim.confidence,
                        span_offset: (evidence.start_offset, evidence.end_offset),
                    },
                    0.8, // Evidence support weight
                );

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

        // Auto-link to NER entities attached to the claim (with proper labels)
        for entity in &claim.entities {
            // Find or create concept node for entity, using the NER label
            let concept_type = crate::structures::ConceptType::from_ner_label(&entity.label);

            let entity_node_id = if let Some(node) = self.get_concept_node(&entity.text) {
                Some(node.id)
            } else {
                let new_node = GraphNode::new(NodeType::Concept {
                    concept_name: entity.text.clone(),
                    concept_type,
                    confidence: claim.confidence,
                });
                self.add_node(new_node).ok()
            };

            // Link claim to entity
            if let Some(eid) = entity_node_id {
                self.link_claim_to_entity(claim.id, eid, 0.9);
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
                        self.link_claim_to_entity(claim.id, eid, 0.9);
                    }
                }
            }
        }

        Some(claim_node_id)
    }

    /// Link a claim to an entity/concept node via ABOUT edge
    ///
    /// This creates semantic relationships between claims and the entities they mention
    pub fn link_claim_to_entity(
        &mut self,
        claim_id: u64,
        entity_node_id: crate::structures::NodeId,
        relevance_score: f32,
    ) -> Option<crate::structures::EdgeId> {
        let claim_node_id = self.get_claim_node(claim_id).map(|n| n.id)?;

        let about_edge = GraphEdge::new(
            claim_node_id,
            entity_node_id,
            EdgeType::About {
                relevance_score,
                mention_count: 1,
            },
            relevance_score,
        );

        let edge_id = self.add_edge(about_edge)?;

        debug!(
            "Created ABOUT edge {} linking claim {} to entity node {}",
            edge_id, claim_id, entity_node_id
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

        // Link claim to entity
        let edge_id = graph.link_claim_to_entity(1, concept_node_id, 0.95);
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
                format!(
                    "Observation [{}] from source '{}' (confidence {:.2}): {}",
                    observation_type, source, confidence, data
                )
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
            };

            // Submit to claim extraction queue
            match claim_queue.extract(request).await {
                Ok(result) => {
                    debug!(
                        "Claim extraction complete for event {}: {} accepted, {} rejected",
                        event_id,
                        result.accepted_claims.len(),
                        result.rejected_claims.len()
                    );

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

    /// Search for similar claims using semantic search
    ///
    /// Returns claims ranked by similarity to the query text
    pub async fn search_similar_claims(
        &self,
        query_text: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(crate::claims::DerivedClaim, f32)>, crate::GraphError> {
        use tracing::debug;

        let embedding_client = match &self.embedding_client {
            Some(c) => c,
            None => {
                return Err(crate::GraphError::InvalidOperation(
                    "Embedding client not initialized".to_string(),
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

        // Generate embedding for query
        let request = crate::claims::EmbeddingRequest {
            text: query_text.to_string(),
            context: None,
        };

        let response = embedding_client.embed(request).await.map_err(|e| {
            crate::GraphError::OperationError(format!("Failed to generate query embedding: {}", e))
        })?;

        debug!(
            "Generated query embedding ({} dimensions)",
            response.embedding.len()
        );

        // Search for similar claims — request extra candidates so temporal
        // re-ranking can still fill top_k after reordering.
        let fetch_k = (top_k * 3).max(20);
        let similar_ids = claim_store
            .find_similar(&response.embedding, fetch_k, min_similarity)
            .map_err(|e| {
                crate::GraphError::OperationError(format!("Failed to search claims: {}", e))
            })?;

        debug!(
            "Found {} candidate claims for re-ranking",
            similar_ids.len()
        );

        // Retrieve full claims and compute temporally-weighted retrieval score
        let mut results: Vec<(crate::claims::DerivedClaim, f32)> = Vec::new();
        for (claim_id, similarity) in similar_ids {
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
}
