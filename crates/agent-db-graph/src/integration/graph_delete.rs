use super::*;

impl GraphEngine {
    /// Hard-delete a node and all its incident edges from the graph.
    ///
    /// Returns `true` if the node existed and was removed, `false` if not found.
    /// Emits `NodeRemoved` and cascaded `EdgeRemoved` deltas for subscriptions.
    pub async fn delete_node(&self, node_id: NodeId) -> bool {
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();
        graph.remove_node(node_id).is_some()
    }

    /// Hard-delete a single edge from the graph.
    ///
    /// Returns `true` if the edge existed and was removed, `false` if not found.
    /// Emits `EdgeRemoved` delta for subscriptions.
    pub async fn delete_edge(&self, edge_id: EdgeId) -> bool {
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();
        graph.remove_edge(edge_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{ConceptType, GraphEdge, GraphNode, NodeType};
    use crate::subscription::delta::GraphDelta;
    use serial_test::serial;

    fn concept(name: &str) -> GraphNode {
        GraphNode::new(NodeType::Concept {
            concept_name: name.to_string(),
            concept_type: ConceptType::Person,
            confidence: 1.0,
        })
    }

    fn assoc(source: NodeId, target: NodeId, ty: &str) -> GraphEdge {
        GraphEdge::new(
            source,
            target,
            EdgeType::Association {
                association_type: ty.to_string(),
                evidence_count: 1,
                statistical_significance: 1.0,
            },
            1.0,
        )
    }

    /// Seed two nodes + one edge, subscribe, then drain any deltas emitted by
    /// setup so the test receiver only sees post-subscription activity.
    async fn seed(
        engine: &GraphEngine,
    ) -> (
        NodeId,
        NodeId,
        EdgeId,
        tokio::sync::broadcast::Receiver<crate::subscription::delta::DeltaBatch>,
    ) {
        let mut inf = engine.inference.write().await;
        let graph = inf.graph_mut();
        let n1 = graph.add_node(concept("Alice")).unwrap();
        let n2 = graph.add_node(concept("Bob")).unwrap();
        let eid = graph.add_edge(assoc(n1, n2, "KNOWS")).unwrap();
        let rx = graph.enable_subscriptions();
        (n1, n2, eid, rx)
    }

    #[tokio::test]
    #[serial]
    async fn delete_edge_emits_edge_removed_delta() {
        let engine = GraphEngine::new().await.unwrap();
        let (_n1, _n2, eid, mut rx) = seed(&engine).await;

        assert!(engine.delete_edge(eid).await);

        let batch = rx.try_recv().expect("expected a delta batch");
        assert_eq!(batch.deltas.len(), 1);
        match &batch.deltas[0] {
            GraphDelta::EdgeRemoved {
                edge_id,
                edge_type_tag,
                ..
            } => {
                assert_eq!(*edge_id, eid);
                assert_eq!(edge_type_tag, "KNOWS");
            },
            other => panic!("expected EdgeRemoved, got {:?}", other),
        }
    }

    #[tokio::test]
    #[serial]
    async fn delete_edge_missing_returns_false_and_no_delta() {
        let engine = GraphEngine::new().await.unwrap();
        let (_n1, _n2, _eid, mut rx) = seed(&engine).await;

        assert!(!engine.delete_edge(9_999).await);
        assert!(rx.try_recv().is_err(), "no delta expected on miss");
    }

    #[tokio::test]
    #[serial]
    async fn delete_node_emits_node_and_cascaded_edge_deltas() {
        let engine = GraphEngine::new().await.unwrap();
        let (n1, _n2, eid, mut rx) = seed(&engine).await;

        assert!(engine.delete_node(n1).await);

        let batch = rx.try_recv().expect("expected a delta batch");
        let node_removed = batch
            .deltas
            .iter()
            .any(|d| matches!(d, GraphDelta::NodeRemoved { node_id, .. } if *node_id == n1));
        let edge_removed = batch
            .deltas
            .iter()
            .any(|d| matches!(d, GraphDelta::EdgeRemoved { edge_id, .. } if *edge_id == eid));
        assert!(
            node_removed,
            "expected NodeRemoved for n1 in {:?}",
            batch.deltas
        );
        assert!(
            edge_removed,
            "expected cascaded EdgeRemoved in {:?}",
            batch.deltas
        );
    }

    #[tokio::test]
    #[serial]
    async fn delete_node_missing_returns_false() {
        let engine = GraphEngine::new().await.unwrap();
        assert!(!engine.delete_node(9_999).await);
    }
}
