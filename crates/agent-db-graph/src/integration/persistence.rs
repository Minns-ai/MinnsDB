use super::*;

impl GraphEngine {
    /// Persist current graph state to redb storage.
    ///
    /// Serializes every node, edge, and graph metadata into an atomic
    /// `write_batch` so the on-disk representation is always consistent.
    /// Returns the number of nodes and edges persisted.
    pub async fn persist_graph_state(&self) -> GraphResult<(usize, usize)> {
        let backend = match self.redb_backend {
            Some(ref b) => b.clone(),
            None => {
                // No persistent backend — nothing to do
                return Ok((0, 0));
            },
        };

        let inference = self.inference.read().await;
        let graph = inference.graph();

        let mut ops: Vec<BatchOperation> = Vec::with_capacity(
            graph.nodes.len() + graph.edges.len() + 1, // +1 for metadata
        );

        // Serialize each node: key = "n" + node_id (big-endian)
        // Uses MessagePack for schema-evolution safety (handles added/removed fields,
        // serde_json::Value in properties, and new enum variants across upgrades)
        for (id, node) in &graph.nodes {
            let mut key = Vec::with_capacity(9);
            key.push(b'n');
            key.extend_from_slice(&id.to_be_bytes());
            let value = rmp_serde::to_vec(node).map_err(|e| {
                GraphError::OperationError(format!("Failed to serialize node {}: {}", id, e))
            })?;
            ops.push(BatchOperation::Put {
                table_name: table_names::GRAPH_NODES.to_string(),
                key,
                value,
            });
        }

        // Serialize each edge: key = "e" + edge_id (big-endian)
        for (id, edge) in &graph.edges {
            let mut key = Vec::with_capacity(9);
            key.push(b'e');
            key.extend_from_slice(&id.to_be_bytes());
            let value = rmp_serde::to_vec(edge).map_err(|e| {
                GraphError::OperationError(format!("Failed to serialize edge {}: {}", id, e))
            })?;
            ops.push(BatchOperation::Put {
                table_name: table_names::GRAPH_EDGES.to_string(),
                key,
                value,
            });
        }

        // Persist adjacency lists and graph metadata in a single metadata blob.
        // This keeps the batch atomic and avoids needing extra tables.
        #[derive(serde::Serialize, serde::Deserialize)]
        struct GraphMeta {
            adjacency_out: HashMap<NodeId, Vec<EdgeId>>,
            adjacency_in: HashMap<NodeId, Vec<EdgeId>>,
            next_node_id: NodeId,
            next_edge_id: EdgeId,
            stats: GraphStats,
        }

        let meta = GraphMeta {
            adjacency_out: graph.adjacency_out.clone(),
            adjacency_in: graph.adjacency_in.clone(),
            next_node_id: graph.next_node_id,
            next_edge_id: graph.next_edge_id,
            stats: graph.stats.clone(),
        };
        let meta_value = rmp_serde::to_vec(&meta).map_err(|e| {
            GraphError::OperationError(format!("Failed to serialize graph metadata: {}", e))
        })?;
        ops.push(BatchOperation::Put {
            table_name: table_names::GRAPH_ADJACENCY.to_string(),
            key: b"__meta__".to_vec(),
            value: meta_value,
        });

        let node_count = graph.nodes.len();
        let edge_count = graph.edges.len();

        // Release the read lock before the blocking I/O
        drop(inference);

        backend.write_batch(ops).map_err(|e| {
            GraphError::OperationError(format!("Failed to persist graph state: {:?}", e))
        })?;

        // Update persistence checkpoint
        let engine_stats = self.stats.read().await;
        *self.last_persistence.write().await = engine_stats.total_events_processed;

        tracing::info!(
            "Graph persisted: {} nodes, {} edges written to redb",
            node_count,
            edge_count
        );

        Ok((node_count, edge_count))
    }

    /// Restore graph state from redb on startup.
    ///
    /// Reads all persisted nodes, edges, and metadata and rebuilds the
    /// in-memory `Graph`, including all secondary indexes.
    pub async fn restore_graph_state(&self) -> GraphResult<(usize, usize)> {
        let backend = match self.redb_backend {
            Some(ref b) => b.clone(),
            None => return Ok((0, 0)),
        };

        // Load metadata first — if it doesn't exist, there's nothing to restore
        #[derive(serde::Serialize, serde::Deserialize)]
        struct GraphMeta {
            adjacency_out: HashMap<NodeId, Vec<EdgeId>>,
            adjacency_in: HashMap<NodeId, Vec<EdgeId>>,
            next_node_id: NodeId,
            next_edge_id: EdgeId,
            stats: GraphStats,
        }

        let meta: GraphMeta = match backend
            .get_raw(table_names::GRAPH_ADJACENCY, b"__meta__")
            .map_err(|e| {
                GraphError::OperationError(format!("Failed to read graph metadata: {:?}", e))
            })? {
            Some(bytes) => rmp_serde::from_slice(&bytes).map_err(|e| {
                GraphError::OperationError(format!("Failed to deserialize graph metadata: {}", e))
            })?,
            None => {
                tracing::info!("No persisted graph state found — starting fresh");
                return Ok((0, 0));
            },
        };

        // Load all nodes (keys prefixed with 'n')
        let raw_node_bytes = backend
            .scan_prefix_raw(table_names::GRAPH_NODES, b"n")
            .map_err(|e| {
                GraphError::OperationError(format!("Failed to scan graph nodes: {:?}", e))
            })?;
        let mut raw_nodes = Vec::with_capacity(raw_node_bytes.len());
        for (key, value) in raw_node_bytes {
            let node: GraphNode = rmp_serde::from_slice(&value).map_err(|e| {
                GraphError::OperationError(format!("Failed to deserialize graph node: {}", e))
            })?;
            raw_nodes.push((key, node));
        }

        // Load all edges (keys prefixed with 'e')
        let raw_edge_bytes = backend
            .scan_prefix_raw(table_names::GRAPH_EDGES, b"e")
            .map_err(|e| {
                GraphError::OperationError(format!("Failed to scan graph edges: {:?}", e))
            })?;
        let mut raw_edges = Vec::with_capacity(raw_edge_bytes.len());
        for (key, value) in raw_edge_bytes {
            let edge: GraphEdge = rmp_serde::from_slice(&value).map_err(|e| {
                GraphError::OperationError(format!("Failed to deserialize graph edge: {}", e))
            })?;
            raw_edges.push((key, edge));
        }

        let node_count = raw_nodes.len();
        let edge_count = raw_edges.len();

        // Rebuild the graph under the inference write lock
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // Clear existing data
        graph.nodes.clear();
        graph.edges.clear();
        graph.adjacency_out = meta.adjacency_out;
        graph.adjacency_in = meta.adjacency_in;
        graph.next_node_id = meta.next_node_id;
        graph.next_edge_id = meta.next_edge_id;
        graph.stats = meta.stats;

        // Clear all secondary indexes before rebuilding
        graph.type_index.clear();
        graph.context_index.clear();
        graph.agent_index.clear();
        graph.event_index.clear();
        graph.goal_index.clear();
        graph.episode_index.clear();
        graph.memory_index.clear();
        graph.strategy_index.clear();
        graph.tool_index.clear();
        graph.result_index.clear();
        graph.claim_index.clear();
        graph.concept_index.clear();

        // Insert nodes and rebuild indexes
        for (_key, node) in raw_nodes {
            let node_id = node.id;

            // Rebuild type index
            let type_name = node.type_name();
            graph
                .type_index
                .entry(type_name)
                .or_default()
                .insert(node_id);

            // Rebuild specialized indexes
            match &node.node_type {
                NodeType::Agent { agent_id, .. } => {
                    graph.agent_index.insert(*agent_id, node_id);
                },
                NodeType::Event { event_id, .. } => {
                    graph.event_index.insert(*event_id, node_id);
                },
                NodeType::Context { context_hash, .. } => {
                    graph.context_index.insert(*context_hash, node_id);
                },
                NodeType::Goal { goal_id, .. } => {
                    graph.goal_index.insert(*goal_id, node_id);
                },
                NodeType::Episode { episode_id, .. } => {
                    graph.episode_index.insert(*episode_id, node_id);
                },
                NodeType::Memory { memory_id, .. } => {
                    graph.memory_index.insert(*memory_id, node_id);
                },
                NodeType::Strategy { strategy_id, .. } => {
                    graph.strategy_index.insert(*strategy_id, node_id);
                },
                NodeType::Tool { tool_name, .. } => {
                    graph.tool_index.insert(tool_name.clone(), node_id);
                },
                NodeType::Result { result_key, .. } => {
                    graph.result_index.insert(result_key.clone(), node_id);
                },
                NodeType::Claim { claim_id, .. } => {
                    graph.claim_index.insert(*claim_id, node_id);
                },
                NodeType::Concept { concept_name, .. } => {
                    graph.concept_index.insert(concept_name.clone(), node_id);
                },
            }

            // Rebuild BM25 index
            let mut text_parts = Vec::new();
            match &node.node_type {
                NodeType::Claim { claim_text, .. } => text_parts.push(claim_text.as_str()),
                NodeType::Goal { description, .. } => text_parts.push(description.as_str()),
                NodeType::Strategy { name, .. } => text_parts.push(name.as_str()),
                NodeType::Result { summary, .. } => text_parts.push(summary.as_str()),
                NodeType::Concept { concept_name, .. } => text_parts.push(concept_name.as_str()),
                NodeType::Tool { tool_name, .. } => text_parts.push(tool_name.as_str()),
                NodeType::Episode { outcome, .. } => text_parts.push(outcome.as_str()),
                _ => {},
            }
            for (key, value) in &node.properties {
                let key_lower = key.to_lowercase();
                if key_lower.contains("text")
                    || key_lower.contains("description")
                    || key_lower.contains("content")
                    || key_lower.contains("name")
                    || key_lower.contains("summary")
                    || key_lower == "data"
                {
                    if let Some(text) = value.as_str() {
                        text_parts.push(text);
                    }
                }
            }
            if !text_parts.is_empty() {
                let combined_text = text_parts.join(" ");
                graph.bm25_index.index_document(node_id, &combined_text);
            }

            graph.nodes.insert(node_id, node);
        }

        // Insert edges
        for (_key, edge) in raw_edges {
            graph.edges.insert(edge.id, edge);
        }

        tracing::info!(
            "Graph restored from redb: {} nodes, {} edges",
            node_count,
            edge_count
        );

        Ok((node_count, edge_count))
    }
}
