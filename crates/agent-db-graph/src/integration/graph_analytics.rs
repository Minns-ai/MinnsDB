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

    /// Run community detection and update memory clusters.
    /// Respects `config.community_algorithm` to choose between Louvain and Label Propagation.
    pub(super) async fn run_community_detection(&self) -> GraphResult<()> {
        tracing::info!("Acquiring inference write lock for community detection");
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        let algo = self.config.community_algorithm.trim().to_lowercase();
        match algo.as_str() {
            "label_propagation" | "label-propagation" | "labelprop" | "lp" => {
                tracing::info!("Running Label Propagation community detection");
                let result = self.label_propagation.detect_communities(graph)?;
                for (&node_id, &label) in &result.node_labels {
                    if let Some(node) = graph.get_node_mut(node_id) {
                        node.properties
                            .insert("community_id".to_string(), json!(label));
                        node.touch();
                    }
                }
            },
            _ => {
                // Default: Louvain
                let communities = self.louvain.detect_communities(graph)?;
                for (node_id, community_id) in communities.node_communities {
                    if let Some(node) = graph.get_node_mut(node_id) {
                        node.properties
                            .insert("community_id".to_string(), json!(community_id));
                        node.touch();
                    }
                }
            },
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

    /// Compute PersonalizedPageRank scores from a source node.
    pub async fn personalized_pagerank(&self, source: NodeId) -> GraphResult<HashMap<NodeId, f64>> {
        let inference = self.inference.read().await;
        let graph = inference.graph();
        self.random_walker.personalized_pagerank(graph, source)
    }

    /// Compute temporal reachability from a source node.
    ///
    /// If `max_hops > 0`, creates a config with that limit.
    /// If `max_hops == 0`, uses default config (unlimited hops).
    pub async fn temporal_reachability_from(
        &self,
        source: NodeId,
        max_hops: usize,
    ) -> GraphResult<crate::algorithms::TemporalReachabilityResult> {
        if max_hops > 1000 {
            return Err(GraphError::InvalidQuery(
                "max_hops must be <= 1000".to_string(),
            ));
        }
        let inference = self.inference.read().await;
        let graph = inference.graph();

        if max_hops > 0 {
            let tr = crate::algorithms::TemporalReachability::with_config(
                crate::algorithms::TemporalReachabilityConfig {
                    max_hops,
                    ..Default::default()
                },
            );
            tr.propagate(graph, source)
        } else {
            self.temporal_reachability.propagate(graph, source)
        }
    }

    /// Find the causal path from source to target using temporal reachability.
    pub async fn causal_path(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Vec<NodeId>>> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        let result = self.temporal_reachability.propagate(graph, source)?;
        Ok(self.temporal_reachability.causal_path(&result, target))
    }

    /// Detect communities using a specified algorithm.
    ///
    /// Algorithm precedence: explicit param > `config.community_algorithm` > "louvain" fallback.
    pub async fn detect_communities_with_algorithm(
        &self,
        algorithm: Option<&str>,
    ) -> GraphResult<crate::algorithms::CommunityDetectionResult> {
        let algo = algorithm
            .unwrap_or(&self.config.community_algorithm)
            .trim()
            .to_lowercase();

        let inference = self.inference.read().await;
        let graph = inference.graph();

        match algo.as_str() {
            "louvain" => {
                tracing::info!("Running Louvain community detection");
                self.louvain.detect_communities(graph)
            }
            "label_propagation" | "label-propagation" | "labelprop" | "lp" => {
                tracing::info!("Running Label Propagation community detection");
                let lp_result = self.label_propagation.detect_communities(graph)?;
                // Convert LabelPropagationResult → CommunityDetectionResult
                Ok(crate::algorithms::CommunityDetectionResult {
                    communities: lp_result.communities,
                    node_communities: lp_result.node_labels,
                    modularity: -1.0, // sentinel: not computed by LP
                    community_count: lp_result.community_count,
                    iterations: lp_result.iterations,
                })
            }
            _ => Err(GraphError::InvalidQuery(format!(
                "Unknown community algorithm '{}'. Accepted values: louvain, label_propagation, label-propagation, labelprop, lp",
                algo
            ))),
        }
    }
}
