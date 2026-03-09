use super::*;
use crate::slot_vec::SlotVec;
use crate::structures::AdjList;

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

        // Delete stale nodes/edges from disk that no longer exist in memory.
        // Without this, deleted nodes resurrect as zombies on restart.
        // Safety: skip mass-deletion if the stale count exceeds 50% of disk
        // entries — that likely means the in-memory set was only partially
        // loaded (e.g. max_graph_size cap), not that nodes were truly deleted.
        {
            let disk_node_keys = backend
                .scan_prefix_raw(table_names::GRAPH_NODES, b"n")
                .unwrap_or_default();
            let disk_node_count = disk_node_keys.len();
            let memory_node_ids: std::collections::HashSet<NodeId> = graph.nodes.keys().collect();
            let mut stale_node_ops: Vec<BatchOperation> = Vec::new();
            for (key, _) in disk_node_keys {
                if key.len() >= 9 {
                    if let Ok(id_bytes) = key[1..9].try_into() {
                        let id = u64::from_be_bytes(id_bytes);
                        if !memory_node_ids.contains(&id) {
                            stale_node_ops.push(BatchOperation::Delete {
                                table_name: table_names::GRAPH_NODES.to_string(),
                                key,
                            });
                        }
                    }
                }
            }

            let disk_edge_keys = backend
                .scan_prefix_raw(table_names::GRAPH_EDGES, b"e")
                .unwrap_or_default();
            let disk_edge_count = disk_edge_keys.len();
            let memory_edge_ids: std::collections::HashSet<EdgeId> = graph.edges.keys().collect();
            let mut stale_edge_ops: Vec<BatchOperation> = Vec::new();
            for (key, _) in disk_edge_keys {
                if key.len() >= 9 {
                    if let Ok(id_bytes) = key[1..9].try_into() {
                        let id = u64::from_be_bytes(id_bytes);
                        if !memory_edge_ids.contains(&id) {
                            stale_edge_ops.push(BatchOperation::Delete {
                                table_name: table_names::GRAPH_EDGES.to_string(),
                                key,
                            });
                        }
                    }
                }
            }

            // Safety threshold: if more than 50% of disk entries would be
            // deleted, the in-memory graph is likely a partial load — skip
            // the deletion to prevent data loss from evicted nodes.
            let node_ratio = if disk_node_count > 0 {
                stale_node_ops.len() as f64 / disk_node_count as f64
            } else {
                0.0
            };
            let edge_ratio = if disk_edge_count > 0 {
                stale_edge_ops.len() as f64 / disk_edge_count as f64
            } else {
                0.0
            };

            if node_ratio > 0.5 || edge_ratio > 0.5 {
                tracing::warn!(
                    "persist_graph_state: skipping stale cleanup — would delete {}/{} nodes, {}/{} edges (likely partial load)",
                    stale_node_ops.len(), disk_node_count,
                    stale_edge_ops.len(), disk_edge_count
                );
            } else {
                if !stale_node_ops.is_empty() || !stale_edge_ops.is_empty() {
                    tracing::info!(
                        "persist_graph_state: deleting {} stale nodes, {} stale edges from disk",
                        stale_node_ops.len(),
                        stale_edge_ops.len()
                    );
                }
                ops.extend(stale_node_ops);
                ops.extend(stale_edge_ops);
            }
        }

        // Serialize each node: key = "n" + node_id (big-endian)
        // Uses versioned MessagePack for schema-evolution safety (handles added/removed fields,
        // serde_json::Value in properties, and new enum variants across upgrades)
        for (id, node) in graph.nodes.iter() {
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
        for (id, edge) in graph.edges.iter() {
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
            adjacency_out: SlotVec<AdjList>,
            adjacency_in: SlotVec<AdjList>,
            #[serde(default)]
            next_node_id: NodeId,
            #[serde(default)]
            next_edge_id: EdgeId,
            stats: GraphStats,
        }

        let meta = GraphMeta {
            adjacency_out: graph.adjacency_out.clone(),
            adjacency_in: graph.adjacency_in.clone(),
            next_node_id: graph.nodes.next_id(),
            next_edge_id: graph.edges.next_id(),
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

        // Clear dirty state after full persist (all data is now on disk)
        {
            let mut inference = self.inference.write().await;
            inference.graph_mut().clear_dirty();
        }

        // Update persistence checkpoint
        let total = self
            .stats
            .total_events_processed
            .load(AtomicOrdering::Relaxed);
        self.last_persistence.store(total, AtomicOrdering::Relaxed);

        tracing::info!(
            "Graph persisted: {} nodes, {} edges written to redb",
            node_count,
            edge_count
        );

        Ok((node_count, edge_count))
    }

    /// Persist only changed graph state (delta) to redb storage.
    ///
    /// Only writes dirty nodes/edges and deletes removed ones, avoiding
    /// the full-graph scan of `persist_graph_state`. Falls back to full
    /// persistence if there are no pending changes or if the dirty set
    /// covers more than 50% of the graph (full persist is simpler).
    ///
    /// Returns the number of operations written.
    pub async fn persist_graph_delta(&self) -> GraphResult<usize> {
        let backend = match self.redb_backend {
            Some(ref b) => b.clone(),
            None => return Ok(0),
        };

        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        if !graph.has_pending_changes() {
            return Ok(0);
        }

        // If dirty set is very large relative to graph size, fall back to full persist
        let dirty_ratio = (graph.dirty_nodes.len() + graph.dirty_edges.len()) as f64
            / (graph.nodes.len() + graph.edges.len()).max(1) as f64;
        if dirty_ratio > 0.5 {
            drop(inference);
            let (n, e) = self.persist_graph_state().await?;
            // Clear dirty state after full persist
            let mut inference = self.inference.write().await;
            inference.graph_mut().clear_dirty();
            return Ok(n + e);
        }

        let mut ops: Vec<BatchOperation> = Vec::new();

        // Delete removed nodes from disk
        for node_id in &graph.deleted_nodes {
            let mut key = Vec::with_capacity(9);
            key.push(b'n');
            key.extend_from_slice(&node_id.to_be_bytes());
            ops.push(BatchOperation::Delete {
                table_name: table_names::GRAPH_NODES.to_string(),
                key,
            });
        }

        // Delete removed edges from disk
        for edge_id in &graph.deleted_edges {
            let mut key = Vec::with_capacity(9);
            key.push(b'e');
            key.extend_from_slice(&edge_id.to_be_bytes());
            ops.push(BatchOperation::Delete {
                table_name: table_names::GRAPH_EDGES.to_string(),
                key,
            });
        }

        // Serialize only dirty nodes
        for &node_id in &graph.dirty_nodes {
            if let Some(node) = graph.nodes.get(node_id) {
                let mut key = Vec::with_capacity(9);
                key.push(b'n');
                key.extend_from_slice(&node_id.to_be_bytes());
                let value = agent_db_storage::serialize_versioned(node).map_err(|e| {
                    GraphError::OperationError(format!(
                        "Failed to serialize node {}: {}",
                        node_id, e
                    ))
                })?;
                ops.push(BatchOperation::Put {
                    table_name: table_names::GRAPH_NODES.to_string(),
                    key,
                    value,
                });
            }
        }

        // Serialize only dirty edges
        for &edge_id in &graph.dirty_edges {
            if let Some(edge) = graph.edges.get(edge_id) {
                let mut key = Vec::with_capacity(9);
                key.push(b'e');
                key.extend_from_slice(&edge_id.to_be_bytes());
                let value = agent_db_storage::serialize_versioned(edge).map_err(|e| {
                    GraphError::OperationError(format!(
                        "Failed to serialize edge {}: {}",
                        edge_id, e
                    ))
                })?;
                ops.push(BatchOperation::Put {
                    table_name: table_names::GRAPH_EDGES.to_string(),
                    key,
                    value,
                });
            }
        }

        // Re-persist adjacency metadata if changed
        if graph.adjacency_dirty {
            #[derive(serde::Serialize, serde::Deserialize)]
            struct GraphMeta {
                adjacency_out: SlotVec<AdjList>,
                adjacency_in: SlotVec<AdjList>,
                #[serde(default)]
                next_node_id: NodeId,
                #[serde(default)]
                next_edge_id: EdgeId,
                stats: GraphStats,
            }

            let meta = GraphMeta {
                adjacency_out: graph.adjacency_out.clone(),
                adjacency_in: graph.adjacency_in.clone(),
                next_node_id: graph.nodes.next_id(),
                next_edge_id: graph.edges.next_id(),
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
        }

        let op_count = ops.len();
        let dirty_n = graph.dirty_nodes.len();
        let dirty_e = graph.dirty_edges.len();
        let del_n = graph.deleted_nodes.len();
        let del_e = graph.deleted_edges.len();

        // Clear dirty state before I/O (safe: ops already built)
        graph.clear_dirty();

        // Release lock before blocking I/O
        drop(inference);

        if op_count > 0 {
            backend.write_batch(ops).map_err(|e| {
                GraphError::OperationError(format!("Failed to persist graph delta: {:?}", e))
            })?;
        }

        tracing::info!(
            "Graph delta persisted: {} ops ({}N dirty, {}E dirty, {}N deleted, {}E deleted)",
            op_count,
            dirty_n,
            dirty_e,
            del_n,
            del_e,
        );

        Ok(op_count)
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
            adjacency_out: SlotVec<AdjList>,
            adjacency_in: SlotVec<AdjList>,
            #[serde(default)]
            next_node_id: NodeId,
            #[serde(default)]
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
            backend.for_each_prefix_raw(table_names::GRAPH_NODES, vec![b'n'], |_key, value| {
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
            backend.for_each_prefix_raw(table_names::GRAPH_EDGES, vec![b'e'], |_key, value| {
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
        // next_node_id / next_edge_id are now managed by SlotVec internally;
        // insert_at() during restore will advance next_id automatically.
        graph.stats = meta.stats;

        // Clear all secondary indexes before rebuilding
        graph.type_index.clear();
        graph.temporal_index.clear();
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

            // Rebuild type index (u8 discriminant key)
            graph
                .type_index
                .entry(node.node_type.discriminant())
                .or_default()
                .insert(node_id);

            // Rebuild temporal index
            graph
                .temporal_index
                .entry(node.created_at)
                .or_insert_with(smallvec::SmallVec::new)
                .push(node_id);

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
                    let key = graph.interner.intern(tool_name);
                    graph.tool_index.insert(key, node_id);
                },
                NodeType::Result { result_key, .. } => {
                    let key = graph.interner.intern(result_key);
                    graph.result_index.insert(key, node_id);
                },
                NodeType::Claim { claim_id, .. } => {
                    graph.claim_index.insert(*claim_id, node_id);
                },
                NodeType::Concept { concept_name, .. } => {
                    let key = graph.interner.intern(concept_name);
                    graph.concept_index.insert(key, node_id);
                },
            }

            // Rebuild BM25 index (must stay in sync with Graph::add_node whitelist)
            let mut text_parts: Vec<&str> = Vec::new();
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
            let mut found_code_key = false;
            let mut owned_parts: Vec<String> = Vec::new();
            for (key, value) in &node.properties {
                let key_lower = key.to_lowercase();
                if key_lower.contains("text")
                    || key_lower.contains("description")
                    || key_lower.contains("content")
                    || key_lower.contains("name")
                    || key_lower.contains("summary")
                    || key_lower == "data"
                    || key_lower == "code"
                    || key_lower == "source"
                    || key_lower == "source_code"
                    || key_lower == "snippet"
                    || key_lower == "body"
                    || key_lower == "function_name"
                    || key_lower == "class_name"
                    || key_lower == "message"
                    || key_lower == "query"
                    || key_lower == "result"
                    || key_lower == "error"
                    || key_lower == "prompt"
                    || key_lower == "answer"
                    || key_lower == "response"
                    || key_lower == "output"
                    || key_lower == "category"
                    || key_lower == "metadata_text"
                {
                    if matches!(
                        key_lower.as_str(),
                        "code"
                            | "source"
                            | "source_code"
                            | "snippet"
                            | "function_name"
                            | "class_name"
                    ) {
                        found_code_key = true;
                    }
                    if let Some(text) = value.as_str() {
                        text_parts.push(text);
                    } else {
                        let flat = Graph::flatten_json_to_text(value);
                        if !flat.is_empty() {
                            owned_parts.push(flat);
                        }
                    }
                }
            }
            if !text_parts.is_empty() || !owned_parts.is_empty() {
                let mut combined_text = text_parts.join(" ");
                if !owned_parts.is_empty() {
                    if !combined_text.is_empty() {
                        combined_text.push(' ');
                    }
                    combined_text.push_str(&owned_parts.join(" "));
                }
                let is_code_content = node
                    .properties
                    .get("content_type")
                    .and_then(|v| v.as_str())
                    .map(|v| v == "code")
                    .unwrap_or(false);
                if is_code_content || found_code_key {
                    graph
                        .bm25_index
                        .index_document_code(node_id, &combined_text);
                } else {
                    graph.bm25_index.index_document(node_id, &combined_text);
                }
            }

            graph.nodes.insert_at(node_id, node);
        }

        // Insert edges (only those whose source and target are in kept nodes)
        for edge in raw_edges {
            if kept_node_ids.contains(&edge.source) && kept_node_ids.contains(&edge.target) {
                graph.edges.insert_at(edge.id, edge);
            }
        }

        // BUG 8 fix: Validate adjacency lists — remove references to edges that don't exist.
        {
            let mut orphan_count = 0usize;
            for edge_ids in graph.adjacency_out.values_mut() {
                let before = edge_ids.len();
                edge_ids.retain(|eid| graph.edges.contains_key(*eid));
                orphan_count += before - edge_ids.len();
            }
            for edge_ids in graph.adjacency_in.values_mut() {
                let before = edge_ids.len();
                edge_ids.retain(|eid| graph.edges.contains_key(*eid));
                orphan_count += before - edge_ids.len();
            }
            if orphan_count > 0 {
                tracing::warn!(
                    "restore_graph_state: removed {} orphaned adjacency references",
                    orphan_count
                );
            }
        }

        // Re-index BM25 for nodes that have edges with searchable metadata.
        // This appends edge label text (e.g., "preference food", "relationship neighbor")
        // to the node's BM25 document so edge types are discoverable via search.
        {
            let mut reindexed = 0usize;
            let node_ids: Vec<NodeId> = graph.nodes.keys().collect();
            for nid in node_ids {
                let has_edge_text = graph
                    .adjacency_out
                    .get(nid)
                    .map(|edges| {
                        edges.iter().any(|&eid| {
                            graph.edges.get(eid).is_some_and(|e| {
                                !crate::structures::edge_text_for_bm25(e).is_empty()
                            })
                        })
                    })
                    .unwrap_or(false)
                    || graph
                        .adjacency_in
                        .get(nid)
                        .map(|edges| {
                            edges.iter().any(|&eid| {
                                graph.edges.get(eid).is_some_and(|e| {
                                    !crate::structures::edge_text_for_bm25(e).is_empty()
                                })
                            })
                        })
                        .unwrap_or(false);
                if has_edge_text {
                    graph.reindex_node_with_edges(nid);
                    reindexed += 1;
                }
            }
            if reindexed > 0 {
                tracing::info!("BM25 edge-text reindex: {} nodes updated", reindexed);
            }
        }

        // Collect all restored node IDs for property index rebuild (BUG 7 fix).
        let all_node_ids: Vec<NodeId> = graph.nodes.keys().collect();

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
