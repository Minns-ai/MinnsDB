use super::*;

impl GraphEngine {
    /// Auto-index newly created nodes
    pub(super) async fn auto_index_nodes(&self, node_ids: &[NodeId]) -> GraphResult<()> {
        tracing::info!(
            "auto_index_nodes acquiring inference read lock (nodes={})",
            node_ids.len()
        );
        let inference = match timeout(Duration::from_secs(2), self.inference.read()).await {
            Ok(lock) => lock,
            Err(_) => {
                tracing::info!("auto_index_nodes timeout acquiring inference read lock");
                return Err(GraphError::OperationError(
                    "auto_index_nodes timeout acquiring inference read lock".to_string(),
                ));
            },
        };
        tracing::info!("auto_index_nodes acquired inference read lock");
        let graph = inference.graph();
        tracing::info!("auto_index_nodes acquiring index manager write lock");
        let mut idx_mgr = self.index_manager.write().await;
        tracing::info!("auto_index_nodes acquired index manager write lock");

        for &node_id in node_ids {
            if let Some(node) = graph.get_node(node_id) {
                tracing::info!("auto_index_nodes indexing node_id={}", node_id);
                // Index common properties
                for (key, value) in &node.properties {
                    // Track property queries for auto-indexing
                    idx_mgr.record_property_query(key);

                    // Find or auto-create index for this property
                    if let Some(index) = idx_mgr.find_index_for_property(key) {
                        index.insert(node_id, value);
                    }
                }

                // Index node type
                let node_type_str = node.type_name();
                let node_type_value = serde_json::json!(node_type_str);
                if let Some(index) = idx_mgr.find_index_for_property("node_type") {
                    index.insert(node_id, &node_type_value);
                }
            }
        }

        tracing::info!("auto_index_nodes finished");
        Ok(())
    }

    /// Run Louvain community detection and update memory clusters
    pub(super) async fn run_community_detection(&self) -> GraphResult<()> {
        tracing::info!("Acquiring inference write lock for community detection");
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // Detect communities using Louvain
        let communities = self.louvain.detect_communities(graph)?;

        for (node_id, community_id) in communities.node_communities {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.properties
                    .insert("community_id".to_string(), json!(community_id));
                node.touch();
            }
        }

        Ok(())
    }

    /// Get graph analytics including learning metrics
    pub async fn get_analytics(&self) -> GraphResult<crate::analytics::GraphMetrics> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        let analytics = GraphAnalytics::from_ref(graph);
        analytics.calculate_all_metrics()
    }

    /// Get property index statistics
    pub async fn get_index_stats(&self) -> Vec<crate::indexing::IndexStats> {
        let idx_mgr = self.index_manager.read().await;
        idx_mgr.get_all_stats().into_values().collect()
    }

    /// Manually trigger community detection
    pub async fn detect_communities(
        &self,
    ) -> GraphResult<crate::algorithms::CommunityDetectionResult> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        self.louvain.detect_communities(graph)
    }

    /// Get centrality scores for all nodes
    pub async fn get_all_centrality_scores(
        &self,
    ) -> GraphResult<crate::algorithms::AllCentralities> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        self.centrality.all_centralities(graph)
    }
}
