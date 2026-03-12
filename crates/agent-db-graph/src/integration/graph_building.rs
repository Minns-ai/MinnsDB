use super::*;

impl GraphEngine {
    /// Embed a list of nodes asynchronously (fire-and-forget).
    ///
    /// Takes a list of `(node_id, text_to_embed)` pairs.
    /// Each node gets its embedding stored on the node and indexed
    /// in the `node_vector_index` for semantic search.
    ///
    /// This uses the same embedding client and pattern as the pipeline.
    /// Safe to call from both `&self` (pipeline) and `Arc<Self>` (server handlers).
    pub fn embed_nodes_async(&self, nodes: Vec<(NodeId, String)>) {
        if nodes.is_empty() {
            return;
        }
        let ec = match &self.embedding_client {
            Some(ec) => ec.clone(),
            None => return,
        };
        let inference_ref = self.inference.clone();
        tokio::spawn(async move {
            for (nid, text) in nodes {
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

    /// Get graph structure for visualization
    pub async fn get_graph_structure(
        &self,
        limit: usize,
        session_id: Option<SessionId>,
        agent_type: Option<AgentType>,
    ) -> GraphStructure {
        let agent_type = agent_type.and_then(|value| {
            if value.trim().is_empty() {
                None
            } else {
                Some(value)
            }
        });

        // Snapshot event labels quickly, then release the event_store read lock
        // before acquiring the inference read lock. Holding both simultaneously
        // blocks event_store writers for the entire graph iteration.
        let event_labels: Vec<(EventId, String)> = {
            let event_store = self.event_store.read().await;
            event_store
                .iter()
                .filter(|(_, event)| {
                    if let Some(sid) = session_id {
                        if event.session_id != sid {
                            return false;
                        }
                    }
                    if let Some(ref at) = agent_type {
                        if event.agent_type != *at {
                            return false;
                        }
                    }
                    true
                })
                .take(limit)
                .map(|(event_id, event)| {
                    let label = match &event.event_type {
                        EventType::Action { action_name, .. } => action_name.clone(),
                        EventType::Observation {
                            observation_type, ..
                        } => observation_type.clone(),
                        EventType::Cognitive { process_type, .. } => {
                            format!("{:?}", process_type)
                        },
                        EventType::Learning { .. } => "Learning".to_string(),
                        _ => format!("Event {}", event.id),
                    };
                    (*event_id, label)
                })
                .collect()
            // event_store read lock dropped here
        };

        // Now acquire only the inference read lock for graph iteration
        if let (Some(session_id), Some(ref agent_type)) = (session_id, &agent_type) {
            let scope = crate::scoped_inference::InferenceScope {
                agent_type: agent_type.clone(),
                session_id,
            };
            if let Ok(scope_engine) = self.scoped_inference.get_scope_engine(&scope).await {
                let inference = scope_engine.read().await;
                let graph = inference.graph();
                return Self::build_graph_structure(graph, &event_labels, limit);
            }
            return GraphStructure {
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        let inference = self.inference.read().await;
        let graph = inference.graph();
        Self::build_graph_structure(graph, &event_labels, limit)
    }

    /// Get graph structure centered around a context hash
    pub async fn get_graph_structure_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
        session_id: Option<SessionId>,
        agent_type: Option<AgentType>,
    ) -> GraphStructure {
        let agent_type = agent_type.and_then(|value| {
            if value.trim().is_empty() {
                None
            } else {
                Some(value)
            }
        });

        // Snapshot event labels quickly, then release event_store lock
        let event_labels: HashMap<EventId, String> = {
            let event_store = self.event_store.read().await;
            event_store
                .iter()
                .filter(|(_, event)| {
                    if let Some(sid) = session_id {
                        if event.session_id != sid {
                            return false;
                        }
                    }
                    if let Some(ref at) = agent_type {
                        if event.agent_type != *at {
                            return false;
                        }
                    }
                    true
                })
                .take(limit)
                .map(|(event_id, event)| {
                    let label = match &event.event_type {
                        EventType::Action { action_name, .. } => action_name.clone(),
                        EventType::Observation {
                            observation_type, ..
                        } => observation_type.clone(),
                        EventType::Cognitive { process_type, .. } => {
                            format!("{:?}", process_type)
                        },
                        EventType::Learning { .. } => "Learning".to_string(),
                        _ => format!("Event {}", event.id),
                    };
                    (*event_id, label)
                })
                .collect()
            // event_store read lock dropped here
        };

        if let (Some(session_id), Some(ref agent_type)) = (session_id, &agent_type) {
            let scope = crate::scoped_inference::InferenceScope {
                agent_type: agent_type.clone(),
                session_id,
            };
            if let Ok(scope_engine) = self.scoped_inference.get_scope_engine(&scope).await {
                let inference = scope_engine.read().await;
                let graph = inference.graph();
                return Self::build_context_graph_structure(
                    graph,
                    context_hash,
                    &event_labels,
                    limit,
                );
            }
            return GraphStructure {
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        let inference = self.inference.read().await;
        let graph = inference.graph();
        Self::build_context_graph_structure(graph, context_hash, &event_labels, limit)
    }

    fn build_graph_structure(
        graph: &Graph,
        event_labels: &[(EventId, String)],
        limit: usize,
    ) -> GraphStructure {
        let mut nodes: HashMap<NodeId, GraphNodeData> = HashMap::new();
        let mut edges: Vec<GraphEdgeData> = Vec::new();

        // Phase 1: Traverse from event nodes using pre-computed labels
        for (event_id, label) in event_labels.iter().take(limit) {
            if let Some(node) = graph.get_event_node(*event_id) {
                nodes.insert(
                    node.id,
                    Self::build_graph_node_data(node, Some(label.clone())),
                );

                // Get edges from this node
                let outgoing_edges = graph.get_edges_from(node.id);
                for edge in outgoing_edges.into_iter().take(5) {
                    edges.push(Self::build_edge_data(edge));

                    if let Some(target_node) = graph.get_node(edge.target) {
                        nodes
                            .entry(target_node.id)
                            .or_insert_with(|| Self::build_graph_node_data(target_node, None));
                    }
                }
            }
        }

        // Phase 2: Include non-event nodes (Concept, Memory, Strategy, Episode, etc.)
        // that the event traversal above would miss.
        let remaining = limit.saturating_sub(nodes.len());
        if remaining > 0 {
            for (node_id, node) in graph.nodes.iter() {
                if nodes.len() >= limit {
                    break;
                }
                if nodes.contains_key(&node_id) {
                    continue;
                }
                // Skip bare Event nodes not already discovered — they lack labels
                if matches!(node.node_type, NodeType::Event { .. }) {
                    continue;
                }
                nodes.insert(node_id, Self::build_graph_node_data(node, None));

                // Include edges from/to this node
                for edge in graph.get_edges_from(node_id) {
                    edges.push(Self::build_edge_data(edge));
                    if let Some(target) = graph.get_node(edge.target) {
                        nodes
                            .entry(target.id)
                            .or_insert_with(|| Self::build_graph_node_data(target, None));
                    }
                }
                for edge in graph.get_edges_to(node_id) {
                    edges.push(Self::build_edge_data(edge));
                    if let Some(source) = graph.get_node(edge.source) {
                        nodes
                            .entry(source.id)
                            .or_insert_with(|| Self::build_graph_node_data(source, None));
                    }
                }
            }
        }

        // Deduplicate edges by id
        edges.sort_by_key(|e| e.id);
        edges.dedup_by_key(|e| e.id);

        GraphStructure {
            nodes: nodes.into_values().collect(),
            edges,
        }
    }

    fn build_edge_data(edge: &GraphEdge) -> GraphEdgeData {
        let edge_type = match &edge.edge_type {
            EdgeType::Temporal { .. } => "Temporal",
            EdgeType::Causality { .. } => "Causal",
            EdgeType::Contextual { .. } => "Contextual",
            EdgeType::Interaction { .. } => "Interaction",
            EdgeType::Association { .. } => "Association",
            EdgeType::GoalRelation { .. } => "GoalRelation",
            EdgeType::Communication { .. } => "Communication",
            EdgeType::DerivedFrom { .. } => "DerivedFrom",
            EdgeType::SupportedBy { .. } => "SupportedBy",
            EdgeType::About { .. } => "About",
            EdgeType::CodeStructure { .. } => "CodeStructure",
        };
        GraphEdgeData {
            id: edge.id,
            from: edge.source,
            to: edge.target,
            edge_type: edge_type.to_string(),
            weight: edge.weight,
            confidence: edge.confidence,
        }
    }

    fn build_graph_node_data(node: &GraphNode, label_override: Option<String>) -> GraphNodeData {
        let (node_type, label) = match &node.node_type {
            NodeType::Event { event_type, .. } => {
                (format!("Event::{}", event_type), label_override)
            },
            NodeType::Context { .. } => ("Context".to_string(), label_override),
            NodeType::Agent { .. } => ("Agent".to_string(), label_override),
            NodeType::Goal { description, .. } => ("Goal".to_string(), Some(description.clone())),
            NodeType::Episode { episode_id, .. } => (
                "Episode".to_string(),
                Some(format!("Episode {}", episode_id)),
            ),
            NodeType::Memory { memory_id, .. } => {
                ("Memory".to_string(), Some(format!("Memory {}", memory_id)))
            },
            NodeType::Strategy { name, .. } => ("Strategy".to_string(), Some(name.clone())),
            NodeType::Tool { tool_name, .. } => ("Tool".to_string(), Some(tool_name.clone())),
            NodeType::Result { summary, .. } => ("Result".to_string(), Some(summary.clone())),
            NodeType::Concept { concept_name, .. } => {
                ("Concept".to_string(), Some(concept_name.clone()))
            },
            NodeType::Claim { claim_text, .. } => ("Claim".to_string(), Some(claim_text.clone())),
        };

        GraphNodeData {
            id: node.id,
            label,
            node_type,
            created_at: node.created_at,
            properties: serde_json::to_value(&node.properties)
                .unwrap_or_else(|_| serde_json::json!({})),
        }
    }

    fn build_context_graph_structure(
        graph: &Graph,
        context_hash: ContextHash,
        event_labels: &HashMap<EventId, String>,
        limit: usize,
    ) -> GraphStructure {
        let context_node = match graph.get_context_node(context_hash) {
            Some(node) => node,
            None => {
                return GraphStructure {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                }
            },
        };

        let mut nodes: HashMap<NodeId, GraphNodeData> = HashMap::new();
        let mut edges: Vec<GraphEdgeData> = Vec::new();

        nodes.insert(
            context_node.id,
            Self::build_graph_node_data(context_node, None),
        );

        // Phase 1: edges from/to context node
        let mut candidate_edges = Vec::new();
        candidate_edges.extend(graph.get_edges_from(context_node.id));
        candidate_edges.extend(graph.get_edges_to(context_node.id));

        for (edge_count, edge) in candidate_edges.into_iter().enumerate() {
            if edge_count >= limit {
                break;
            }

            edges.push(Self::build_edge_data(edge));

            for node_id in [edge.source, edge.target] {
                if nodes.contains_key(&node_id) {
                    continue;
                }

                if let Some(node) = graph.get_node(node_id) {
                    let label = if let NodeType::Event { event_id, .. } = node.node_type {
                        event_labels.get(&event_id).cloned()
                    } else {
                        None
                    };
                    nodes.insert(node.id, Self::build_graph_node_data(node, label));
                }
            }
        }

        // Phase 2: Include non-event nodes not yet discovered
        let remaining = limit.saturating_sub(nodes.len());
        if remaining > 0 {
            for (node_id, node) in graph.nodes.iter() {
                if nodes.len() >= limit {
                    break;
                }
                if nodes.contains_key(&node_id) {
                    continue;
                }
                if matches!(node.node_type, NodeType::Event { .. }) {
                    continue;
                }
                nodes.insert(node_id, Self::build_graph_node_data(node, None));
                for edge in graph.get_edges_from(node_id) {
                    edges.push(Self::build_edge_data(edge));
                    if let Some(target) = graph.get_node(edge.target) {
                        nodes
                            .entry(target.id)
                            .or_insert_with(|| Self::build_graph_node_data(target, None));
                    }
                }
                for edge in graph.get_edges_to(node_id) {
                    edges.push(Self::build_edge_data(edge));
                    if let Some(source) = graph.get_node(edge.source) {
                        nodes
                            .entry(source.id)
                            .or_insert_with(|| Self::build_graph_node_data(source, None));
                    }
                }
            }
        }

        // Deduplicate edges
        edges.sort_by_key(|e| e.id);
        edges.dedup_by_key(|e| e.id);

        GraphStructure {
            nodes: nodes.into_values().collect(),
            edges,
        }
    }

    pub(super) async fn attach_memory_to_graph(
        &self,
        episode: &Episode,
        memory: &Memory,
    ) -> GraphResult<()> {
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        let episode_node_id = Self::ensure_episode_node(graph, episode)?;
        let memory_node_id = Self::ensure_memory_node(graph, memory)?;

        Self::add_or_strengthen_association(
            graph,
            episode_node_id,
            memory_node_id,
            "FormedMemory",
            0.8,
            json!({
                "episode_id": episode.id,
                "memory_id": memory.id,
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
            }),
        );

        if let Some(context_node) = graph.get_context_node(memory.context.fingerprint) {
            Self::add_or_strengthen_association(
                graph,
                memory_node_id,
                context_node.id,
                "MemoryContext",
                0.7,
                json!({
                    "context_hash": memory.context.fingerprint,
                }),
            );
        }

        for goal in &episode.context.active_goals {
            let goal_node_id = Self::ensure_goal_node(graph, goal)?;
            Self::add_or_strengthen_association(
                graph,
                memory_node_id,
                goal_node_id,
                "MemoryGoal",
                0.6,
                json!({
                    "goal_id": goal.id,
                }),
            );
        }

        for event_id in episode.events.iter().take(5) {
            if let Some(event_node) = graph.get_event_node(*event_id) {
                Self::add_or_strengthen_association(
                    graph,
                    episode_node_id,
                    event_node.id,
                    "EpisodeContainsEvent",
                    0.4,
                    json!({
                        "event_id": event_id.to_string(),
                    }),
                );
            }
        }

        Ok(())
    }

    pub(super) async fn attach_strategy_to_graph(
        &self,
        episode: &Episode,
        strategy: &Strategy,
    ) -> GraphResult<()> {
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        let episode_node_id = Self::ensure_episode_node(graph, episode)?;
        let strategy_node_id = Self::ensure_strategy_node(graph, strategy)?;

        Self::add_or_strengthen_association(
            graph,
            episode_node_id,
            strategy_node_id,
            "DerivedStrategy",
            0.8,
            json!({
                "episode_id": episode.id,
                "strategy_id": strategy.id,
            }),
        );

        for goal in &episode.context.active_goals {
            let goal_node_id = Self::ensure_goal_node(graph, goal)?;
            Self::add_or_strengthen_association(
                graph,
                strategy_node_id,
                goal_node_id,
                "StrategyGoal",
                0.7,
                json!({
                    "goal_id": goal.id,
                }),
            );
        }

        let tool_names = strategy
            .metadata
            .get("tool_names")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default();
        for tool_name in tool_names {
            let tool_node_id = Self::ensure_tool_node(graph, &tool_name)?;
            Self::add_or_strengthen_association(
                graph,
                strategy_node_id,
                tool_node_id,
                "StrategyUsesTool",
                0.6,
                json!({
                    "tool_name": tool_name,
                }),
            );
        }

        let result_types = strategy
            .metadata
            .get("result_types")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default();
        for result_type in result_types {
            let result_key = format!("strategy:{}:{}", strategy.id, result_type);
            let result_node_id =
                Self::ensure_result_node(graph, &result_key, &result_type, &result_type)?;
            Self::add_or_strengthen_association(
                graph,
                strategy_node_id,
                result_node_id,
                "StrategyProducesResult",
                0.6,
                json!({
                    "result_type": result_type,
                }),
            );
        }

        Ok(())
    }

    fn ensure_episode_node(graph: &mut Graph, episode: &Episode) -> GraphResult<NodeId> {
        if let Some(node) = graph.get_episode_node(episode.id) {
            Ok(node.id)
        } else {
            let outcome = episode
                .outcome
                .as_ref()
                .map(|value| format!("{:?}", value))
                .unwrap_or_else(|| "Unknown".to_string());
            let mut node = GraphNode::new(NodeType::Episode {
                episode_id: episode.id,
                agent_id: episode.agent_id,
                session_id: episode.session_id,
                outcome: outcome.clone(),
            });
            node.properties
                .insert("outcome".to_string(), json!(outcome));
            node.properties
                .insert("event_count".to_string(), json!(episode.events.len()));
            node.properties
                .insert("significance".to_string(), json!(episode.significance));
            node.properties
                .insert("salience_score".to_string(), json!(episode.salience_score));
            graph.add_node(node)
        }
    }

    fn ensure_memory_node(graph: &mut Graph, memory: &Memory) -> GraphResult<NodeId> {
        if let Some(node) = graph.get_memory_node(memory.id) {
            Ok(node.id)
        } else {
            let mut node = GraphNode::new(NodeType::Memory {
                memory_id: memory.id,
                agent_id: memory.agent_id,
                session_id: memory.session_id,
            });
            node.properties
                .insert("strength".to_string(), json!(memory.strength));
            node.properties
                .insert("relevance_score".to_string(), json!(memory.relevance_score));
            node.properties.insert(
                "context_hash".to_string(),
                json!(memory.context.fingerprint),
            );
            node.properties
                .insert("formed_at".to_string(), json!(memory.formed_at));
            node.properties.insert(
                "memory_type".to_string(),
                json!(format!("{:?}", memory.memory_type)),
            );
            // Store actual memory content for graph-based retrieval
            if !memory.summary.is_empty() {
                node.properties
                    .insert("summary".to_string(), json!(memory.summary));
            }
            if !memory.takeaway.is_empty() {
                node.properties
                    .insert("takeaway".to_string(), json!(memory.takeaway));
            }
            if !memory.causal_note.is_empty() {
                node.properties
                    .insert("causal_note".to_string(), json!(memory.causal_note));
            }
            node.properties
                .insert("tier".to_string(), json!(format!("{:?}", memory.tier)));
            node.properties.insert(
                "consolidation_status".to_string(),
                json!(format!("{:?}", memory.consolidation_status)),
            );
            node.properties.insert(
                "outcome".to_string(),
                json!(format!("{:?}", memory.outcome)),
            );
            graph.add_node(node)
        }
    }

    fn ensure_strategy_node(graph: &mut Graph, strategy: &Strategy) -> GraphResult<NodeId> {
        if let Some(node) = graph.get_strategy_node(strategy.id) {
            Ok(node.id)
        } else {
            let mut node = GraphNode::new(NodeType::Strategy {
                strategy_id: strategy.id,
                agent_id: strategy.agent_id,
                name: strategy.name.clone(),
            });
            node.properties
                .insert("quality_score".to_string(), json!(strategy.quality_score));
            node.properties
                .insert("success_count".to_string(), json!(strategy.success_count));
            node.properties
                .insert("failure_count".to_string(), json!(strategy.failure_count));
            node.properties
                .insert("version".to_string(), json!(strategy.version));
            node.properties.insert(
                "strategy_type".to_string(),
                json!(format!("{:?}", strategy.strategy_type)),
            );
            node.properties
                .insert("support_count".to_string(), json!(strategy.support_count));
            node.properties.insert(
                "expected_success".to_string(),
                json!(strategy.expected_success),
            );
            node.properties
                .insert("expected_cost".to_string(), json!(strategy.expected_cost));
            node.properties
                .insert("expected_value".to_string(), json!(strategy.expected_value));
            node.properties
                .insert("confidence".to_string(), json!(strategy.confidence));
            node.properties
                .insert("goal_bucket_id".to_string(), json!(strategy.goal_bucket_id));
            node.properties.insert(
                "behavior_signature".to_string(),
                json!(strategy.behavior_signature),
            );
            node.properties
                .insert("precondition".to_string(), json!(strategy.precondition));
            node.properties
                .insert("action_hint".to_string(), json!(strategy.action_hint));
            graph.add_node(node)
        }
    }

    fn ensure_goal_node(
        graph: &mut Graph,
        goal: &agent_db_events::core::Goal,
    ) -> GraphResult<NodeId> {
        if let Some(node) = graph.get_goal_node(goal.id) {
            Ok(node.id)
        } else {
            let status = if goal.progress >= 1.0 {
                GoalStatus::Completed
            } else {
                GoalStatus::Active
            };
            let mut node = GraphNode::new(NodeType::Goal {
                goal_id: goal.id,
                description: goal.description.clone(),
                priority: goal.priority,
                status,
            });
            node.properties
                .insert("progress".to_string(), json!(goal.progress));
            if let Some(deadline) = goal.deadline {
                node.properties
                    .insert("deadline".to_string(), json!(deadline));
            }
            graph.add_node(node)
        }
    }

    fn ensure_tool_node(graph: &mut Graph, tool_name: &str) -> GraphResult<NodeId> {
        if let Some(node) = graph.get_tool_node(tool_name) {
            Ok(node.id)
        } else {
            let mut node = GraphNode::new(NodeType::Tool {
                tool_name: tool_name.to_string(),
                tool_type: "external".to_string(),
            });
            node.properties.insert("usage_count".to_string(), json!(1));
            graph.add_node(node)
        }
    }

    fn ensure_result_node(
        graph: &mut Graph,
        result_key: &str,
        result_type: &str,
        summary: &str,
    ) -> GraphResult<NodeId> {
        if let Some(node) = graph.get_result_node(result_key) {
            Ok(node.id)
        } else {
            let mut node = GraphNode::new(NodeType::Result {
                result_key: result_key.to_string(),
                result_type: result_type.to_string(),
                summary: summary.to_string(),
            });
            node.properties
                .insert("summary".to_string(), json!(summary));
            graph.add_node(node)
        }
    }

    pub(crate) fn add_or_strengthen_association(
        graph: &mut Graph,
        source: NodeId,
        target: NodeId,
        association_type: &str,
        weight: EdgeWeight,
        properties: serde_json::Value,
    ) {
        add_or_strengthen_association(graph, source, target, association_type, weight, properties);
    }
}

/// Create or strengthen an Association edge between two nodes.
///
/// If an edge already exists between `source` and `target`, strengthens it.
/// Otherwise creates a new `EdgeType::Association` edge.
pub(super) fn add_or_strengthen_association(
    graph: &mut Graph,
    source: NodeId,
    target: NodeId,
    association_type: &str,
    weight: EdgeWeight,
    properties: serde_json::Value,
) {
    if let Some(edge) = graph.get_edge_between_mut(source, target) {
        edge.strengthen(weight * 0.1);
        return;
    }

    let mut edge = GraphEdge::new(
        source,
        target,
        EdgeType::Association {
            association_type: association_type.to_string(),
            evidence_count: 1,
            statistical_significance: weight,
        },
        weight,
    );
    edge.properties.insert("details".to_string(), properties);
    graph.add_edge(edge);
}
