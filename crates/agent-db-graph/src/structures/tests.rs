// crates/agent-db-graph/src/structures/tests.rs

use super::*;
use std::collections::HashMap;

#[test]
fn test_concept_type_from_ner_label_code_types() {
    assert!(matches!(
        ConceptType::from_ner_label("FUNCTION"),
        ConceptType::Function
    ));
    assert!(matches!(
        ConceptType::from_ner_label("METHOD"),
        ConceptType::Function
    ));
    assert!(matches!(
        ConceptType::from_ner_label("FUNC"),
        ConceptType::Function
    ));
    assert!(matches!(
        ConceptType::from_ner_label("CLASS"),
        ConceptType::Class
    ));
    assert!(matches!(
        ConceptType::from_ner_label("STRUCT"),
        ConceptType::Class
    ));
    assert!(matches!(
        ConceptType::from_ner_label("TYPE"),
        ConceptType::Class
    ));
    assert!(matches!(
        ConceptType::from_ner_label("INTERFACE"),
        ConceptType::Interface
    ));
    assert!(matches!(
        ConceptType::from_ner_label("MODULE"),
        ConceptType::Module
    ));
    assert!(matches!(
        ConceptType::from_ner_label("PACKAGE"),
        ConceptType::Module
    ));
    assert!(matches!(
        ConceptType::from_ner_label("CRATE"),
        ConceptType::Module
    ));
    assert!(matches!(
        ConceptType::from_ner_label("NAMESPACE"),
        ConceptType::Module
    ));
    assert!(matches!(
        ConceptType::from_ner_label("VARIABLE"),
        ConceptType::Variable
    ));
    assert!(matches!(
        ConceptType::from_ner_label("CONST"),
        ConceptType::Variable
    ));
    assert!(matches!(
        ConceptType::from_ner_label("VAR"),
        ConceptType::Variable
    ));
    assert!(matches!(
        ConceptType::from_ner_label("PARAM"),
        ConceptType::Variable
    ));
}

#[test]
fn test_concept_type_from_ner_label_existing() {
    // Ensure existing NER labels still work
    assert!(matches!(
        ConceptType::from_ner_label("PERSON"),
        ConceptType::Person
    ));
    assert!(matches!(
        ConceptType::from_ner_label("ORG"),
        ConceptType::Organization
    ));
}

#[test]
fn test_node_indexing_routing_code_content_type() {
    let mut graph = Graph::new();
    let mut node = GraphNode::new(NodeType::Tool {
        tool_name: "myFunction".to_string(),
        tool_type: "code".to_string(),
    });
    node.properties
        .insert("content_type".to_string(), serde_json::json!("code"));
    node.properties
        .insert("code".to_string(), serde_json::json!("fn getUserName()"));

    let node_id = graph.add_node(node).unwrap();

    // Should be indexed via code tokenizer — query with snake_case
    let results = graph.bm25_index.search_code("get_user_name", 10);
    assert!(
        results.iter().any(|(id, _)| *id == node_id),
        "Node with content_type=code should be findable via code search"
    );
}

#[test]
fn test_node_indexing_routing_code_key_fallback() {
    let mut graph = Graph::new();
    let mut node = GraphNode::new(NodeType::Concept {
        concept_name: "helper".to_string(),
        concept_type: ConceptType::Function,
        confidence: 0.9,
    });
    // No content_type, but has code-specific key "snippet"
    node.properties
        .insert("snippet".to_string(), serde_json::json!("fn parseJSON()"));

    let node_id = graph.add_node(node).unwrap();

    // Should be indexed via code tokenizer due to code-specific key
    let results = graph.bm25_index.search_code("parse", 10);
    assert!(
        results.iter().any(|(id, _)| *id == node_id),
        "Node with code-specific key should be findable via code search"
    );
}

#[test]
fn test_node_indexing_routing_natural_default() {
    let mut graph = Graph::new();
    let mut node = GraphNode::new(NodeType::Goal {
        goal_id: 1,
        description: "improve system performance".to_string(),
        priority: 0.5,
        status: GoalStatus::Active,
    });
    node.properties.insert(
        "description".to_string(),
        serde_json::json!("improve system performance"),
    );

    let node_id = graph.add_node(node).unwrap();

    // Should be indexed via natural tokenizer
    let results = graph.bm25_index.search("performance", 10);
    assert!(
        results.iter().any(|(id, _)| *id == node_id),
        "Node without code signals should be findable via natural search"
    );
}

// ── Soft-Delete Graph Edges Tests ──

#[test]
fn test_edge_is_valid_default() {
    let edge = GraphEdge::new(
        1,
        2,
        EdgeType::Temporal {
            average_interval_ms: 100,
            sequence_confidence: 0.9,
        },
        0.5,
    );
    assert!(edge.is_valid(), "New edge should be valid by default");
    assert!(edge.invalidated_at().is_none());
}

#[test]
fn test_edge_invalidate() {
    let mut edge = GraphEdge::new(
        1,
        2,
        EdgeType::Temporal {
            average_interval_ms: 100,
            sequence_confidence: 0.9,
        },
        0.5,
    );
    edge.invalidate("superseded by newer info");
    assert!(!edge.is_valid(), "Invalidated edge should not be valid");
    assert_eq!(
        edge.properties
            .get("invalidated_reason")
            .and_then(|v| v.as_str()),
        Some("superseded by newer info")
    );
}

#[test]
fn test_edge_invalidated_at_timestamp() {
    let mut edge = GraphEdge::new(
        1,
        2,
        EdgeType::Temporal {
            average_interval_ms: 100,
            sequence_confidence: 0.9,
        },
        0.5,
    );
    assert!(edge.invalidated_at().is_none());
    edge.invalidate("test");
    let ts = edge.invalidated_at();
    assert!(ts.is_some(), "Invalidated edge should have a timestamp");
    assert!(ts.unwrap() > 0);
}

#[test]
fn test_graph_invalidate_edge() {
    let mut graph = Graph::new();
    let n1 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 1,
            agent_type: "test".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n2 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 2,
            agent_type: "test".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let eid = graph
        .add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();

    let gen_before = graph.generation();
    assert!(graph.invalidate_edge(eid, "outdated"));
    assert!(
        graph.generation() > gen_before,
        "Generation should bump after invalidation"
    );
    assert!(!graph.get_edge(eid).unwrap().is_valid());
    assert!(graph.dirty_edges.contains(&eid));
}

#[test]
fn test_graph_invalidate_edge_nonexistent() {
    let mut graph = Graph::new();
    assert!(
        !graph.invalidate_edge(9999, "missing"),
        "Should return false for nonexistent edge"
    );
}

#[test]
fn test_get_valid_edges_from_filters() {
    let mut graph = Graph::new();
    let n1 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 1,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n2 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 2,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n3 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 3,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();

    let e1 = graph
        .add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 0,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();
    let _e2 = graph
        .add_edge(GraphEdge::new(
            n1,
            n3,
            EdgeType::Temporal {
                average_interval_ms: 0,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();

    // All edges valid initially
    assert_eq!(graph.get_valid_edges_from(n1).len(), 2);

    // Invalidate one
    graph.invalidate_edge(e1, "test");
    assert_eq!(graph.get_valid_edges_from(n1).len(), 1);
    assert_eq!(graph.get_valid_edges_from(n1)[0].target, n3);
}

#[test]
fn test_get_valid_neighbors_filters() {
    let mut graph = Graph::new();
    let n1 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 1,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n2 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 2,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n3 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 3,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();

    let e1 = graph
        .add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 0,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();
    let _e2 = graph
        .add_edge(GraphEdge::new(
            n1,
            n3,
            EdgeType::Temporal {
                average_interval_ms: 0,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();

    assert_eq!(graph.get_valid_neighbors(n1).len(), 2);
    graph.invalidate_edge(e1, "test");
    let valid = graph.get_valid_neighbors(n1);
    assert_eq!(valid.len(), 1);
    assert_eq!(valid[0], n3);
}

#[test]
fn test_get_invalidated_edges_from() {
    let mut graph = Graph::new();
    let n1 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 1,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n2 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 2,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();
    let n3 = graph
        .add_node(GraphNode::new(NodeType::Agent {
            agent_id: 3,
            agent_type: "t".into(),
            capabilities: vec![],
        }))
        .unwrap();

    let e1 = graph
        .add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 0,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();
    let _e2 = graph
        .add_edge(GraphEdge::new(
            n1,
            n3,
            EdgeType::Temporal {
                average_interval_ms: 0,
                sequence_confidence: 0.9,
            },
            0.5,
        ))
        .unwrap();

    // No invalidated edges initially
    assert!(graph.get_invalidated_edges_from(n1).is_empty());

    graph.invalidate_edge(e1, "outdated");
    let invalidated = graph.get_invalidated_edges_from(n1);
    assert_eq!(invalidated.len(), 1);
    assert_eq!(invalidated[0].target, n2);
}

#[test]
fn test_backward_compat_no_properties() {
    // Edges deserialized from old format without is_valid property should be treated as valid
    let edge = GraphEdge {
        id: 1,
        source: 10,
        target: 20,
        edge_type: EdgeType::Temporal {
            average_interval_ms: 100,
            sequence_confidence: 0.9,
        },
        weight: 0.5,
        created_at: 0,
        updated_at: 0,
        valid_from: None,
        valid_until: None,
        observation_count: 1,
        confidence: 0.5,
        properties: HashMap::new(), // No is_valid property
        confidence_history: crate::tcell::TCell::Empty,
        weight_history: crate::tcell::TCell::Empty,
        group_id: String::new(),
    };
    assert!(
        edge.is_valid(),
        "Edge without is_valid property should be treated as valid"
    );
}

// ── Bi-temporal edge tests ───────────────────────────────────────────

fn make_bitemporal_edge(valid_from: Option<u64>, valid_until: Option<u64>) -> GraphEdge {
    GraphEdge {
        id: 1,
        source: 10,
        target: 20,
        edge_type: EdgeType::Temporal {
            average_interval_ms: 100,
            sequence_confidence: 0.9,
        },
        weight: 0.5,
        created_at: 1000,
        updated_at: 1000,
        valid_from,
        valid_until,
        observation_count: 1,
        confidence: 0.9,
        properties: HashMap::new(),
        confidence_history: crate::tcell::TCell::Empty,
        weight_history: crate::tcell::TCell::Empty,
        group_id: String::new(),
    }
}

#[test]
fn test_valid_at_open_interval() {
    // No valid_from/valid_until → valid at all times
    let edge = make_bitemporal_edge(None, None);
    assert!(edge.valid_at(0));
    assert!(edge.valid_at(999_999));
}

#[test]
fn test_valid_at_with_from() {
    let edge = make_bitemporal_edge(Some(100), None);
    assert!(!edge.valid_at(50), "Before valid_from");
    assert!(edge.valid_at(100), "At valid_from");
    assert!(edge.valid_at(200), "After valid_from");
}

#[test]
fn test_valid_at_with_until() {
    let edge = make_bitemporal_edge(None, Some(200));
    assert!(edge.valid_at(100), "Before valid_until");
    assert!(!edge.valid_at(200), "At valid_until (half-open)");
    assert!(!edge.valid_at(300), "After valid_until");
}

#[test]
fn test_valid_at_closed_range() {
    let edge = make_bitemporal_edge(Some(100), Some(200));
    assert!(!edge.valid_at(50));
    assert!(edge.valid_at(100));
    assert!(edge.valid_at(150));
    assert!(!edge.valid_at(200));
    assert!(!edge.valid_at(300));
}

#[test]
fn test_valid_during_overlap() {
    let edge = make_bitemporal_edge(Some(100), Some(300));
    // Query range fully inside
    assert!(edge.valid_during(150, 250));
    // Query range overlaps start
    assert!(edge.valid_during(50, 150));
    // Query range overlaps end
    assert!(edge.valid_during(250, 350));
    // Query range fully outside before
    assert!(!edge.valid_during(10, 50));
    // Query range fully outside after
    assert!(!edge.valid_during(400, 500));
    // Query range exactly at boundary (no overlap)
    assert!(!edge.valid_during(300, 400));
}

#[test]
fn test_valid_during_open_edge() {
    let edge = make_bitemporal_edge(None, None);
    assert!(edge.valid_during(0, 999_999));
}

#[test]
fn test_set_valid_time() {
    let mut edge = make_bitemporal_edge(None, None);
    assert!(edge.valid_from.is_none());
    assert!(edge.valid_until.is_none());

    edge.set_valid_time(Some(100), Some(200));
    assert_eq!(edge.valid_from, Some(100));
    assert_eq!(edge.valid_until, Some(200));
}

#[test]
fn test_is_currently_valid_fact() {
    let edge = make_bitemporal_edge(Some(100), Some(300));
    assert!(edge.is_currently_valid_fact(150));
    assert!(!edge.is_currently_valid_fact(50));
    assert!(!edge.is_currently_valid_fact(350));
}

#[test]
fn test_is_currently_valid_fact_soft_deleted() {
    let mut edge = make_bitemporal_edge(Some(100), Some(300));
    edge.invalidate("superseded");
    // Even though point_in_time is within valid range, edge is soft-deleted
    assert!(!edge.is_currently_valid_fact(150));
}

#[test]
fn test_default_valid_from_until_is_none() {
    let edge = GraphEdge::new(
        10,
        20,
        EdgeType::Temporal {
            average_interval_ms: 100,
            sequence_confidence: 0.9,
        },
        0.5,
    );
    assert!(edge.valid_from.is_none());
    assert!(edge.valid_until.is_none());
}

#[test]
fn test_bitemporal_serde_round_trip() {
    let edge = make_bitemporal_edge(Some(100_000), Some(200_000));
    let json = serde_json::to_string(&edge).unwrap();
    let deserialized: GraphEdge = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.valid_from, Some(100_000));
    assert_eq!(deserialized.valid_until, Some(200_000));
}

#[test]
fn test_bitemporal_serde_backward_compat() {
    // Old JSON without valid_from/valid_until should deserialize with None
    let json = r#"{"id":1,"source":10,"target":20,"edge_type":{"Temporal":{"average_interval_ms":100,"sequence_confidence":0.9}},"weight":0.5,"created_at":1000,"updated_at":1000,"observation_count":1,"confidence":0.9,"properties":{}}"#;
    let edge: GraphEdge = serde_json::from_str(json).unwrap();
    assert!(edge.valid_from.is_none());
    assert!(edge.valid_until.is_none());
}
