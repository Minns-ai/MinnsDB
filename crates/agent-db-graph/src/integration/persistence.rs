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

        // BUG 1 fix: Delete stale nodes/edges from disk that no longer exist in memory.
        // Without this, deleted nodes resurrect as zombies on restart.
        {
            let disk_node_keys = backend
                .scan_prefix_raw(table_names::GRAPH_NODES, b"n")
                .unwrap_or_default();
            let memory_node_ids: std::collections::HashSet<NodeId> =
                graph.nodes.keys().copied().collect();
            let mut stale_nodes = 0usize;
            for (key, _) in disk_node_keys {
                if key.len() >= 9 {
                    if let Ok(id_bytes) = key[1..9].try_into() {
                        let id = u64::from_be_bytes(id_bytes);
                        if !memory_node_ids.contains(&id) {
                            ops.push(BatchOperation::Delete {
                                table_name: table_names::GRAPH_NODES.to_string(),
                                key,
                            });
                            stale_nodes += 1;
                        }
                    }
                }
            }

            let disk_edge_keys = backend
                .scan_prefix_raw(table_names::GRAPH_EDGES, b"e")
                .unwrap_or_default();
            let memory_edge_ids: std::collections::HashSet<EdgeId> =
                graph.edges.keys().copied().collect();
            let mut stale_edges = 0usize;
            for (key, _) in disk_edge_keys {
                if key.len() >= 9 {
                    if let Ok(id_bytes) = key[1..9].try_into() {
                        let id = u64::from_be_bytes(id_bytes);
                        if !memory_edge_ids.contains(&id) {
                            ops.push(BatchOperation::Delete {
                                table_name: table_names::GRAPH_EDGES.to_string(),
                                key,
                            });
                            stale_edges += 1;
                        }
                    }
                }
            }

            if stale_nodes > 0 || stale_edges > 0 {
                tracing::info!(
                    "persist_graph_state: deleting {} stale nodes, {} stale edges from disk",
                    stale_nodes,
                    stale_edges
                );
            }
        }

        // Serialize each node: key = "n" + node_id (big-endian)
        // Uses versioned MessagePack for schema-evolution safety (handles added/removed fields,
        // serde_json::Value in properties, and new enum variants across upgrades)
        for (id, node) in &graph.nodes {
            let mut key = Vec::with_capacity(9);
            key.push(b'n');
            key.extend_from_slice(&id.to_be_bytes());
            let value = agent_db_storage::serialize_versioned(node).map_err(|e| {
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
            let value = agent_db_storage::serialize_versioned(edge).map_err(|e| {
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
        let meta_value = agent_db_storage::serialize_versioned(&meta).map_err(|e| {
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
            Some(bytes) => agent_db_storage::deserialize_versioned(&bytes).map_err(|e| {
                GraphError::OperationError(format!("Failed to deserialize graph metadata: {}", e))
            })?,
            None => {
                tracing::info!("No persisted graph state found — starting fresh");
                return Ok((0, 0));
            },
        };

        // Rebuild the graph under the inference write lock
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();
        let max_graph_size = graph.max_graph_size;

        // Streaming load of all nodes (keys prefixed with 'n').
        // Uses for_each_prefix_raw to avoid materializing a Vec of all (key, value) pairs.
        // When persisted node count exceeds max_graph_size, we keep only the newest nodes
        // using a bounded min-heap keyed by created_at.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // First pass: stream nodes into a bounded collection
        let mut all_nodes: Vec<GraphNode> = Vec::new();
        let mut total_persisted_nodes = 0usize;
        let mut deser_errors = 0usize;
        let scan_result: Result<(), agent_db_storage::ForEachError<std::convert::Infallible>> =
            backend.for_each_prefix_raw(table_names::GRAPH_NODES, b"n".to_vec(), |_key, value| {
                total_persisted_nodes += 1;
                match agent_db_storage::deserialize_versioned::<GraphNode>(value) {
                    Ok(node) => all_nodes.push(node),
                    Err(e) => {
                        deser_errors += 1;
                        tracing::warn!("Skipping corrupt graph node during restore: {}", e);
                    },
                }
                Ok(())
            });
        if let Err(e) = scan_result {
            return Err(GraphError::OperationError(format!(
                "Failed to stream graph nodes: {:?}",
                e
            )));
        }

        // If we have more nodes than max_graph_size, keep the N newest by created_at
        if all_nodes.len() > max_graph_size {
            tracing::warn!(
                "restore_graph_state: {} persisted nodes exceed max_graph_size {}, keeping newest",
                all_nodes.len(),
                max_graph_size
            );
            // Use a bounded min-heap of size max_graph_size keyed by created_at
            let mut heap: BinaryHeap<Reverse<(agent_db_core::types::Timestamp, usize)>> =
                BinaryHeap::with_capacity(max_graph_size + 1);
            for (idx, node) in all_nodes.iter().enumerate() {
                heap.push(Reverse((node.created_at, idx)));
                if heap.len() > max_graph_size {
                    heap.pop(); // remove oldest
                }
            }
            let keep_indices: std::collections::HashSet<usize> =
                heap.into_iter().map(|Reverse((_, idx))| idx).collect();
            let mut kept = Vec::with_capacity(max_graph_size);
            for (idx, node) in all_nodes.into_iter().enumerate() {
                if keep_indices.contains(&idx) {
                    kept.push(node);
                }
            }
            all_nodes = kept;
        }

        // Stream edges
        let mut raw_edges: Vec<GraphEdge> = Vec::new();
        let edge_scan: Result<(), agent_db_storage::ForEachError<std::convert::Infallible>> =
            backend.for_each_prefix_raw(table_names::GRAPH_EDGES, b"e".to_vec(), |_key, value| {
                match agent_db_storage::deserialize_versioned::<GraphEdge>(value) {
                    Ok(edge) => raw_edges.push(edge),
                    Err(e) => {
                        tracing::warn!("Skipping corrupt graph edge during restore: {}", e);
                    },
                }
                Ok(())
            });
        if let Err(e) = edge_scan {
            return Err(GraphError::OperationError(format!(
                "Failed to stream graph edges: {:?}",
                e
            )));
        }

        let node_count = all_nodes.len();
        let edge_count = raw_edges.len();

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

        // Build set of kept node IDs for edge filtering
        let kept_node_ids: std::collections::HashSet<NodeId> =
            all_nodes.iter().map(|n| n.id).collect();

        // Insert nodes and rebuild indexes
        for node in all_nodes {
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

        // Insert edges (only those whose source and target are in kept nodes)
        for edge in raw_edges {
            if kept_node_ids.contains(&edge.source) && kept_node_ids.contains(&edge.target) {
                graph.edges.insert(edge.id, edge);
            }
        }

        // BUG 8 fix: Validate adjacency lists — remove references to edges that don't exist.
        {
            let mut orphan_count = 0usize;
            for edge_ids in graph.adjacency_out.values_mut() {
                let before = edge_ids.len();
                edge_ids.retain(|eid| graph.edges.contains_key(eid));
                orphan_count += before - edge_ids.len();
            }
            for edge_ids in graph.adjacency_in.values_mut() {
                let before = edge_ids.len();
                edge_ids.retain(|eid| graph.edges.contains_key(eid));
                orphan_count += before - edge_ids.len();
            }
            if orphan_count > 0 {
                tracing::warn!(
                    "restore_graph_state: removed {} orphaned adjacency references",
                    orphan_count
                );
            }
        }

        // Collect all restored node IDs for property index rebuild (BUG 7 fix).
        let all_node_ids: Vec<NodeId> = graph.nodes.keys().copied().collect();

        tracing::info!(
            "Graph restored from redb: {} nodes, {} edges",
            node_count,
            edge_count
        );

        // Must drop inference write lock before auto_index_nodes (which needs read lock)
        drop(inference);

        // BUG 7 fix: Rebuild property indexes after restore.
        if !all_node_ids.is_empty() {
            if let Err(e) = self.auto_index_nodes(&all_node_ids).await {
                tracing::warn!("Failed to rebuild property indexes after restore: {}", e);
            }
        }

        Ok((node_count, edge_count))
    }
}
