// crates/agent-db-graph/src/integration/queries/tests.rs
//
// Unit tests for temporal validity filter, state anchor filter, and epoch filter.

use super::*;
use crate::structures::{EdgeType, Graph, GraphEdge, GraphNode, NodeType};

/// Helper: create a concept node and return its ID.
fn add_concept(graph: &mut Graph, name: &str) -> u64 {
    graph
        .add_node(GraphNode::new(NodeType::Concept {
            concept_name: name.to_string(),
            concept_type: crate::structures::ConceptType::Person,
            confidence: 1.0,
        }))
        .unwrap()
}

/// Helper: create a claim node and return its ID.
fn add_claim(graph: &mut Graph, claim_id: u64, text: &str) -> u64 {
    graph
        .add_node(GraphNode::new(NodeType::Claim {
            claim_id,
            claim_text: text.to_string(),
            confidence: 0.9,
            source_event_id: 1u128,
        }))
        .unwrap()
}

/// Helper: add an association edge with optional valid_until.
fn add_edge(
    graph: &mut Graph,
    source: u64,
    target: u64,
    assoc_type: &str,
    valid_from: Option<u64>,
    valid_until: Option<u64>,
) {
    let mut edge = GraphEdge::new(
        source,
        target,
        EdgeType::Association {
            association_type: assoc_type.to_string(),
            evidence_count: 1,
            statistical_significance: 1.0,
        },
        1.0,
    );
    edge.valid_from = valid_from;
    edge.valid_until = valid_until;
    let _ = graph.add_edge(edge);
}

#[test]
fn test_temporal_validity_filter_supersedes_old_state() {
    let mut graph = Graph::new();

    let user = add_concept(&mut graph, "user");
    let london = add_concept(&mut graph, "London");
    let tokyo = add_concept(&mut graph, "Tokyo");

    // London superseded, Tokyo current
    add_edge(
        &mut graph,
        user,
        london,
        "location:lives_in",
        Some(100),
        Some(200),
    );
    add_edge(
        &mut graph,
        user,
        tokyo,
        "location:lives_in",
        Some(200),
        None,
    );

    // Claim referencing London
    let claim_london = add_claim(&mut graph, 1, "User lives in London and enjoys the parks");

    let mut results = vec![(london, 0.9), (tokyo, 0.8), (claim_london, 0.7)];

    let superseded = apply_temporal_validity_filter(&mut results, &graph);

    // London should be superseded
    assert!(superseded.contains("london"));
    // Tokyo should survive
    assert!(results.iter().any(|(id, s)| *id == tokyo && *s > 0.0));
    // London node and claim mentioning London should be filtered
    assert!(!results.iter().any(|(id, _)| *id == london));
    assert!(!results.iter().any(|(id, _)| *id == claim_london));
}

#[test]
fn test_temporal_validity_filter_memory_strategy_nodes() {
    let mut graph = Graph::new();

    let user = add_concept(&mut graph, "user");
    let berlin = add_concept(&mut graph, "Berlin");
    let tokyo = add_concept(&mut graph, "Tokyo");

    // Berlin superseded, Tokyo current
    add_edge(
        &mut graph,
        user,
        berlin,
        "location:lives_in",
        Some(100),
        Some(200),
    );
    add_edge(
        &mut graph,
        user,
        tokyo,
        "location:lives_in",
        Some(200),
        None,
    );

    // Memory node mentioning Berlin (via label)
    let mem_node = graph
        .add_node(GraphNode::new(NodeType::Memory {
            memory_id: 42,
            agent_id: 1,
            session_id: 1,
        }))
        .unwrap();
    // The label() method returns "Memory 42" but Phase 3b checks node labels.
    // In practice, memory nodes get BM25-indexed by summary text.
    // For this test, create a claim that mentions Berlin instead.
    let claim_berlin = add_claim(&mut graph, 10, "Had a great time in Berlin");

    let mut results = vec![(claim_berlin, 0.8), (tokyo, 0.5), (mem_node, 0.3)];

    let superseded = apply_temporal_validity_filter(&mut results, &graph);
    assert!(superseded.contains("berlin"));
    // Claim mentioning Berlin should be filtered
    assert!(!results.iter().any(|(id, _)| *id == claim_berlin));
    // Tokyo should remain
    assert!(results.iter().any(|(id, _)| *id == tokyo));
}

#[test]
fn test_temporal_validity_filter_multi_transition() {
    let mut graph = Graph::new();

    let user = add_concept(&mut graph, "user");
    let london = add_concept(&mut graph, "London");
    let berlin = add_concept(&mut graph, "Berlin");
    let tokyo = add_concept(&mut graph, "Tokyo");

    // London → Berlin → Tokyo (3 transitions)
    add_edge(
        &mut graph,
        user,
        london,
        "location:lives_in",
        Some(100),
        Some(200),
    );
    add_edge(
        &mut graph,
        user,
        berlin,
        "location:lives_in",
        Some(200),
        Some(300),
    );
    add_edge(
        &mut graph,
        user,
        tokyo,
        "location:lives_in",
        Some(300),
        None,
    );

    let claim_london = add_claim(&mut graph, 1, "User enjoys London weather");
    let claim_berlin = add_claim(&mut graph, 2, "User likes Berlin nightlife");
    let claim_tokyo = add_claim(&mut graph, 3, "User explores Tokyo temples");

    let mut results = vec![
        (claim_london, 0.9),
        (claim_berlin, 0.8),
        (claim_tokyo, 0.7),
        (tokyo, 0.5),
    ];

    let superseded = apply_temporal_validity_filter(&mut results, &graph);

    // Both London and Berlin should be superseded
    assert!(superseded.contains("london"));
    assert!(superseded.contains("berlin"));
    // Only Tokyo claim should survive
    assert!(!results.iter().any(|(id, _)| *id == claim_london));
    assert!(!results.iter().any(|(id, _)| *id == claim_berlin));
    assert!(results.iter().any(|(id, _)| *id == claim_tokyo));
}

#[test]
fn test_current_target_not_filtered() {
    let mut graph = Graph::new();

    let user = add_concept(&mut graph, "user");
    let tokyo = add_concept(&mut graph, "Tokyo");

    // Only current edge, no supersession
    add_edge(
        &mut graph,
        user,
        tokyo,
        "location:lives_in",
        Some(100),
        None,
    );

    let claim = add_claim(&mut graph, 1, "User loves Tokyo sushi");

    let mut results = vec![(claim, 0.9), (tokyo, 0.8)];

    let superseded = apply_temporal_validity_filter(&mut results, &graph);

    assert!(superseded.is_empty());
    assert_eq!(results.len(), 2);
}
