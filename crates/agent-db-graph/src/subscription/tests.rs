use std::collections::HashSet;

use crate::query_lang::types::Value;
use crate::structures::{ConceptType, EdgeType, Graph, GraphEdge, GraphNode, NodeType};

use super::delta::GraphDelta;
use super::trigger::{compile_trigger_set, compute_max_pattern_radius, TriggerSet};

fn concept_node(name: &str) -> GraphNode {
    GraphNode::new(NodeType::Concept {
        concept_name: name.to_string(),
        concept_type: ConceptType::Person,
        confidence: 0.9,
    })
}

fn assoc_edge(source: u64, target: u64, assoc_type: &str) -> GraphEdge {
    GraphEdge::new(
        source,
        target,
        EdgeType::Association {
            association_type: assoc_type.to_string(),
            evidence_count: 1,
            statistical_significance: 0.8,
        },
        1.0,
    )
}

// ── Phase 1: Delta capture tests ──

#[test]
fn test_delta_node_add() {
    let mut graph = Graph::new();
    let mut rx = graph.enable_subscriptions();
    let node = concept_node("Alice");
    let node_id = graph.add_node(node).unwrap();
    let batch = rx.try_recv().unwrap();
    assert_eq!(batch.deltas.len(), 1);
    match &batch.deltas[0] {
        GraphDelta::NodeAdded {
            node_id: nid,
            node_type_disc,
            ..
        } => {
            assert_eq!(nid, &node_id);
            assert_eq!(*node_type_disc, 3); // Concept
        },
        other => panic!("Expected NodeAdded, got {:?}", other),
    }
}

#[test]
fn test_delta_edge_add() {
    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let mut rx = graph.enable_subscriptions();
    let edge = assoc_edge(n1, n2, "KNOWS");
    let eid = graph.add_edge(edge).unwrap();
    let batch = rx.try_recv().unwrap();
    assert_eq!(batch.deltas.len(), 1);
    match &batch.deltas[0] {
        GraphDelta::EdgeAdded {
            edge_id,
            source,
            target,
            edge_type_tag,
            ..
        } => {
            assert_eq!(*edge_id, eid);
            assert_eq!(*source, n1);
            assert_eq!(*target, n2);
            assert_eq!(edge_type_tag, "KNOWS");
        },
        other => panic!("Expected EdgeAdded, got {:?}", other),
    }
}

#[test]
fn test_delta_edge_remove() {
    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    let mut rx = graph.enable_subscriptions();
    let removed = graph.remove_edge(eid);
    assert!(removed.is_some());
    let batch = rx.try_recv().unwrap();
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
        other => panic!("Expected EdgeRemoved, got {:?}", other),
    }
}

#[test]
fn test_delta_node_remove_cascades_edges() {
    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let _eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    let mut rx = graph.enable_subscriptions();
    graph.remove_node(n1);
    let batch = rx.try_recv().unwrap();
    // Should have NodeRemoved + at least one EdgeRemoved
    let has_node_removed = batch
        .deltas
        .iter()
        .any(|d| matches!(d, GraphDelta::NodeRemoved { .. }));
    let has_edge_removed = batch
        .deltas
        .iter()
        .any(|d| matches!(d, GraphDelta::EdgeRemoved { .. }));
    assert!(has_node_removed, "Expected NodeRemoved delta");
    assert!(
        has_edge_removed,
        "Expected EdgeRemoved delta for cascaded edge"
    );
}

#[test]
fn test_delta_invalidate_edge_emits_mutated() {
    // invalidate() sets properties but doesn't change valid_until,
    // so we expect EdgeMutated (not EdgeSuperseded).
    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    let mut rx = graph.enable_subscriptions();
    assert!(graph.invalidate_edge(eid, "test reason"));
    let batch = rx.try_recv().unwrap();
    assert_eq!(batch.deltas.len(), 1);
    // invalidate() doesn't change valid_until, so we get EdgeMutated
    assert!(
        matches!(&batch.deltas[0], GraphDelta::EdgeMutated { edge_type_tag, .. } if edge_type_tag == "KNOWS"),
        "Expected EdgeMutated, got {:?}",
        batch.deltas[0]
    );
}

#[test]
fn test_no_overhead_when_disabled() {
    let mut graph = Graph::new(); // delta_tx = None
                                  // All mutations should work without panics
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    graph.invalidate_edge(eid, "test");
    graph.remove_edge(eid);
    graph.remove_node(n1);
}

#[test]
fn test_delta_merge_nodes() {
    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Alice2")).unwrap();
    let mut rx = graph.enable_subscriptions();
    let result = graph.merge_nodes(n1, n2);
    assert!(result.is_ok());
    // We should receive deltas — at minimum a NodeMerged and the NodeRemoved from remove_node
    let mut found_merge = false;
    // Drain all batches from the channel
    loop {
        match rx.try_recv() {
            Ok(batch) => {
                for d in &batch.deltas {
                    if matches!(d, GraphDelta::NodeMerged { survivor_id, absorbed_id, .. }
                        if *survivor_id == n1 && *absorbed_id == n2)
                    {
                        found_merge = true;
                    }
                }
            },
            Err(_) => break,
        }
    }
    assert!(found_merge, "Expected NodeMerged delta");
}

#[test]
fn test_enable_subscriptions_idempotent() {
    let mut graph = Graph::new();
    let mut rx1 = graph.enable_subscriptions();
    let mut rx2 = graph.enable_subscriptions();
    // Both receivers should get deltas
    let _n = graph.add_node(concept_node("Alice")).unwrap();
    assert!(rx1.try_recv().is_ok());
    assert!(rx2.try_recv().is_ok());
}

// ── Phase 2: Trigger set tests ──

#[test]
fn test_trigger_set_edge_type_overlap_match() {
    let ts = TriggerSet::EdgeTypes(HashSet::from(["KNOWS".to_string()]));
    let batch = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::EdgeAdded {
            edge_id: 1,
            source: 1,
            target: 2,
            edge_type_tag: "KNOWS".to_string(),
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    assert!(ts.overlaps(&batch));
}

#[test]
fn test_trigger_set_edge_type_overlap_miss() {
    let ts = TriggerSet::EdgeTypes(HashSet::from(["KNOWS".to_string()]));
    let batch = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::EdgeAdded {
            edge_id: 1,
            source: 1,
            target: 2,
            edge_type_tag: "WORKS_AT".to_string(),
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    assert!(!ts.overlaps(&batch));
}

#[test]
fn test_trigger_set_node_type_overlap_match() {
    let ts = TriggerSet::NodeTypes(HashSet::from([3])); // Concept
    let batch = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::NodeAdded {
            node_id: 1,
            node_type_disc: 3,
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    assert!(ts.overlaps(&batch));
}

#[test]
fn test_trigger_set_node_type_overlap_miss() {
    let ts = TriggerSet::NodeTypes(HashSet::from([3])); // Concept
    let batch = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::NodeAdded {
            node_id: 1,
            node_type_disc: 1, // Event
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    // NodeAdded with wrong disc doesn't match
    assert!(!ts.overlaps(&batch));
}

#[test]
fn test_trigger_set_combined() {
    let ts = TriggerSet::Combined {
        node_types: HashSet::from([3]),
        edge_types: HashSet::from(["KNOWS".to_string()]),
    };
    // Node match
    let batch_node = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::NodeAdded {
            node_id: 1,
            node_type_disc: 3,
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    assert!(ts.overlaps(&batch_node));
    // Edge match
    let batch_edge = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::EdgeAdded {
            edge_id: 1,
            source: 1,
            target: 2,
            edge_type_tag: "KNOWS".to_string(),
            generation: 2,
        }],
        generation_range: (2, 2),
    };
    assert!(ts.overlaps(&batch_edge));
    // Neither match
    let batch_miss = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::NodeAdded {
            node_id: 2,
            node_type_disc: 1,
            generation: 3,
        }],
        generation_range: (3, 3),
    };
    assert!(!ts.overlaps(&batch_miss));
}

#[test]
fn test_trigger_set_any_always_overlaps() {
    let ts = TriggerSet::Any;
    let batch = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::NodeAdded {
            node_id: 1,
            node_type_disc: 0,
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    assert!(ts.overlaps(&batch));
}

#[test]
fn test_trigger_set_nodes_overlap() {
    let ts = TriggerSet::Nodes(HashSet::from([42]));
    let batch_hit = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::EdgeAdded {
            edge_id: 1,
            source: 42,
            target: 99,
            edge_type_tag: "X".to_string(),
            generation: 1,
        }],
        generation_range: (1, 1),
    };
    assert!(ts.overlaps(&batch_hit));
    let batch_miss = super::delta::DeltaBatch {
        deltas: vec![GraphDelta::EdgeAdded {
            edge_id: 2,
            source: 10,
            target: 20,
            edge_type_tag: "X".to_string(),
            generation: 2,
        }],
        generation_range: (2, 2),
    };
    assert!(!ts.overlaps(&batch_miss));
}

#[test]
fn test_compile_trigger_set_label_scan() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };
    let ts = compile_trigger_set(&plan);
    match ts {
        TriggerSet::NodeTypes(types) => {
            assert!(types.contains(&3)); // Person → Concept disc
        },
        other => panic!("Expected NodeTypes, got {:?}", other),
    }
}

#[test]
fn test_compile_trigger_set_edge_type() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec!["Concept".to_string()],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: Some(1),
                to_var: 2,
                edge_type: Some("KNOWS".to_string()),
                direction: crate::query_lang::ast::Direction::Out,
                range: None,
            },
        ],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 3,
    };
    let ts = compile_trigger_set(&plan);
    match ts {
        TriggerSet::Combined {
            node_types,
            edge_types,
        } => {
            assert!(node_types.contains(&3));
            assert!(edge_types.contains("KNOWS"));
        },
        other => panic!("Expected Combined, got {:?}", other),
    }
}

#[test]
fn test_compile_trigger_set_unfiltered_fallback() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec![],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: None,
                to_var: 1,
                edge_type: None,
                direction: crate::query_lang::ast::Direction::Out,
                range: None,
            },
        ],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 2,
    };
    let ts = compile_trigger_set(&plan);
    assert!(matches!(ts, TriggerSet::Any));
}

#[test]
fn test_max_pattern_radius_single_hop() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::Expand {
            from_var: 0,
            edge_var: None,
            to_var: 1,
            edge_type: None,
            direction: crate::query_lang::ast::Direction::Out,
            range: None,
        }],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 2,
    };
    assert_eq!(compute_max_pattern_radius(&plan), 1);
}

#[test]
fn test_max_pattern_radius_bounded() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::Expand {
            from_var: 0,
            edge_var: None,
            to_var: 1,
            edge_type: None,
            direction: crate::query_lang::ast::Direction::Out,
            range: Some((1, Some(3))),
        }],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 2,
    };
    assert_eq!(compute_max_pattern_radius(&plan), 3);
}

#[test]
fn test_max_pattern_radius_unbounded() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::Expand {
            from_var: 0,
            edge_var: None,
            to_var: 1,
            edge_type: None,
            direction: crate::query_lang::ast::Direction::Out,
            range: Some((1, None)),
        }],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 2,
    };
    assert_eq!(compute_max_pattern_radius(&plan), u32::MAX);
}

#[test]
fn test_delta_generation() {
    let d = GraphDelta::NodeAdded {
        node_id: 1,
        node_type_disc: 3,
        generation: 42,
    };
    assert_eq!(d.generation(), 42);
}

#[test]
fn test_delta_touched_nodes() {
    let d = GraphDelta::EdgeAdded {
        edge_id: 1,
        source: 10,
        target: 20,
        edge_type_tag: "X".to_string(),
        generation: 1,
    };
    let nodes = d.touched_nodes();
    assert_eq!(nodes.len(), 2);
    assert!(nodes.contains(&10));
    assert!(nodes.contains(&20));
}

// ── Phase 3A: Incremental view maintenance tests ──

use super::incremental::*;

#[test]
fn test_row_id_structural_equality() {
    let r1 = RowId::new(smallvec::smallvec![
        (0, BoundEntityId::Node(1)),
        (1, BoundEntityId::Node(2))
    ]);
    let r2 = RowId::new(smallvec::smallvec![
        (1, BoundEntityId::Node(2)),
        (0, BoundEntityId::Node(1))
    ]);
    // Same bindings, different order → should be equal after sorting.
    assert_eq!(r1, r2);
}

#[test]
fn test_row_id_get_slot() {
    let r = RowId::new(smallvec::smallvec![
        (0, BoundEntityId::Node(42)),
        (2, BoundEntityId::Edge(7))
    ]);
    assert_eq!(r.get(0), Some(&BoundEntityId::Node(42)));
    assert_eq!(r.get(2), Some(&BoundEntityId::Edge(7)));
    assert_eq!(r.get(1), None);
}

#[test]
fn test_row_id_extend() {
    let r1 = RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(1))]);
    let r2 = r1.extend(&[(1, BoundEntityId::Edge(5)), (2, BoundEntityId::Node(3))]);
    assert_eq!(r2.slots().len(), 3);
    assert_eq!(r2.get(1), Some(&BoundEntityId::Edge(5)));
}

#[test]
fn test_plan_classification_simple_scan() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };
    let ip = IncrementalPlan::analyze(plan);
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_plan_classification_scan_expand() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec!["Concept".to_string()],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: Some(1),
                to_var: 2,
                edge_type: Some("KNOWS".to_string()),
                direction: crate::query_lang::ast::Direction::Out,
                range: None,
            },
        ],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 3,
    };
    let ip = IncrementalPlan::analyze(plan);
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_plan_classification_variable_length_incremental() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec!["Concept".to_string()],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: None,
                to_var: 1,
                edge_type: None,
                direction: crate::query_lang::ast::Direction::Out,
                range: Some((1, Some(3))),
            },
        ],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 2,
    };
    let ip = IncrementalPlan::analyze(plan);
    // Variable-length paths now supported incrementally (Feature 6).
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_plan_classification_ungrouped_count_incremental() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![],
        aggregations: vec![Aggregation {
            function: AggregateFunction::Count,
            input_expr: RExpr::Star,
            output_alias: "count".to_string(),
        }],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };
    let ip = IncrementalPlan::analyze(plan);
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_plan_classification_grouped_count_incremental() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![],
        aggregations: vec![Aggregation {
            function: AggregateFunction::Count,
            input_expr: RExpr::Star,
            output_alias: "count".to_string(),
        }],
        group_by_keys: vec!["type".to_string()],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };
    let ip = IncrementalPlan::analyze(plan);
    // Grouped aggregation now supported incrementally (Feature 5).
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_plan_classification_sum_incremental() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![],
        aggregations: vec![Aggregation {
            function: AggregateFunction::Sum,
            input_expr: RExpr::Property(0, "weight".to_string()),
            output_alias: "total".to_string(),
        }],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };
    let ip = IncrementalPlan::analyze(plan);
    // Sum is now supported as incremental (Feature 4).
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_plan_classification_point_in_time_incremental() {
    use crate::query_lang::planner::*;
    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec![],
            props: vec![],
        }],
        projections: vec![],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::PointInTime(12345),
        transaction_cutoff: None,
        var_count: 1,
    };
    let ip = IncrementalPlan::analyze(plan);
    // PointInTime now supported incrementally (Feature 7).
    assert!(matches!(ip.strategy, MaintenanceStrategy::Incremental));
}

#[test]
fn test_scan_state_add_remove_nodes() {
    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();

    let mut scan = ScanState::init(0, &["Concept".to_string()], &[], &graph);
    assert!(scan.active_nodes.contains(&n1));

    // Add a new node.
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let delta = GraphDelta::NodeAdded {
        node_id: n2,
        node_type_disc: 3,
        generation: 2,
    };
    let row_deltas = scan.apply_delta(&delta, &graph);
    assert_eq!(row_deltas.len(), 1);
    assert!(row_deltas[0].is_insert());
    assert!(scan.active_nodes.contains(&n2));

    // Remove a node.
    let delta = GraphDelta::NodeRemoved {
        node_id: n1,
        node_type_disc: 3,
        generation: 3,
    };
    let row_deltas = scan.apply_delta(&delta, &graph);
    assert_eq!(row_deltas.len(), 1);
    assert!(!row_deltas[0].is_insert());
    assert!(!scan.active_nodes.contains(&n1));
}

#[test]
fn test_scan_state_type_filter() {
    let mut graph = Graph::new();

    let scan = ScanState::init(0, &["Concept".to_string()], &[], &graph);

    // Add Event node — should be ignored by Concept scan.
    let event_node = GraphNode::new(NodeType::Event {
        event_id: 999,
        event_type: "test".to_string(),
        significance: 0.5,
    });
    let n_event = graph.add_node(event_node).unwrap();
    let delta = GraphDelta::NodeAdded {
        node_id: n_event,
        node_type_disc: 1, // Event
        generation: 1,
    };
    let mut scan = scan;
    let row_deltas = scan.apply_delta(&delta, &graph);
    assert_eq!(row_deltas.len(), 0);
    assert!(!scan.active_nodes.contains(&n_event));
}

#[test]
fn test_expand_state_edge_add_remove() {
    use crate::query_lang::planner::TemporalViewport;

    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();

    let scan = ScanState::init(0, &["Concept".to_string()], &[], &graph);
    let mut expand = ExpandState::init(
        0,
        Some(1),
        2,
        &Some("KNOWS".to_string()),
        &crate::query_lang::ast::Direction::Out,
        &scan,
        &graph,
        &TemporalViewport::ActiveOnly,
        None,
    );

    // Add an edge.
    let eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    let delta = GraphDelta::EdgeAdded {
        edge_id: eid,
        source: n1,
        target: n2,
        edge_type_tag: "KNOWS".to_string(),
        generation: 3,
    };
    let deltas =
        expand.apply_edge_delta(&delta, &scan, &graph, &TemporalViewport::ActiveOnly, None);
    assert_eq!(deltas.len(), 1);
    assert!(deltas[0].is_insert());

    // Remove the edge.
    let delta = GraphDelta::EdgeRemoved {
        edge_id: eid,
        source: n1,
        target: n2,
        edge_type_tag: "KNOWS".to_string(),
        generation: 4,
    };
    let deltas =
        expand.apply_edge_delta(&delta, &scan, &graph, &TemporalViewport::ActiveOnly, None);
    assert_eq!(deltas.len(), 1);
    assert!(!deltas[0].is_insert());
}

#[test]
fn test_expand_state_upstream_propagation() {
    use crate::query_lang::planner::TemporalViewport;

    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let _eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();

    let scan = ScanState::init(0, &["Concept".to_string()], &[], &graph);
    let mut expand = ExpandState::init(
        0,
        Some(1),
        2,
        &Some("KNOWS".to_string()),
        &crate::query_lang::ast::Direction::Out,
        &scan,
        &graph,
        &TemporalViewport::ActiveOnly,
        None,
    );

    // Verify Alice has expansions.
    assert!(expand.expansions.contains_key(&n1));

    // Simulate removing Alice from scan.
    let scan_delete = vec![RowDelta::delete(RowId::new(smallvec::smallvec![(
        0,
        BoundEntityId::Node(n1)
    )]))];
    let deltas =
        expand.apply_upstream_deltas(&scan_delete, &graph, &TemporalViewport::ActiveOnly, None);
    assert_eq!(deltas.len(), 1);
    assert!(!deltas[0].is_insert());
    assert!(!expand.expansions.contains_key(&n1));
}

#[test]
fn test_count_state() {
    let mut cs = CountState::new(5);
    let deltas = vec![
        RowDelta::insert(RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(1))])),
        RowDelta::insert(RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(2))])),
        RowDelta::delete(RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(3))])),
    ];
    let count = cs.apply_deltas(&deltas);
    assert_eq!(count, 6); // 5 + 2 - 1
}

// test_active_rows removed — ActiveRows type eliminated in favor of cached_output.rows.

// ── Phase 3A: Diff tests ──

use super::diff::*;

#[test]
fn test_diff_outputs_insert() {
    let old = CachedOutput::default();
    let mut new_rows = rustc_hash::FxHashMap::default();
    let r1 = RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(1))]);
    new_rows.insert(r1.clone(), vec![Value::String("Alice".to_string())]);

    let diff = diff_outputs(&old, &new_rows);
    assert_eq!(diff.inserts.len(), 1);
    assert_eq!(diff.deletes.len(), 0);
    assert_eq!(diff.inserts[0].0, r1);
}

#[test]
fn test_diff_outputs_delete() {
    let r1 = RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(1))]);
    let mut old_rows = rustc_hash::FxHashMap::default();
    old_rows.insert(r1.clone(), vec![Value::String("Alice".to_string())]);
    let old = CachedOutput {
        columns: vec!["name".to_string()],
        rows: old_rows,
    };

    let new_rows = rustc_hash::FxHashMap::default();
    let diff = diff_outputs(&old, &new_rows);
    assert_eq!(diff.inserts.len(), 0);
    assert_eq!(diff.deletes.len(), 1);
    assert_eq!(diff.deletes[0], r1);
}

#[test]
fn test_diff_outputs_value_change() {
    let r1 = RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(1))]);
    let mut old_rows = rustc_hash::FxHashMap::default();
    old_rows.insert(r1.clone(), vec![Value::String("Alice".to_string())]);
    let old = CachedOutput {
        columns: vec!["name".to_string()],
        rows: old_rows,
    };

    let mut new_rows = rustc_hash::FxHashMap::default();
    new_rows.insert(r1.clone(), vec![Value::String("Alice Smith".to_string())]);

    let diff = diff_outputs(&old, &new_rows);
    // Same RowId, different values → Delete + Insert
    assert_eq!(diff.deletes.len(), 1);
    assert_eq!(diff.inserts.len(), 1);
    assert_eq!(diff.deletes[0], r1);
    assert_eq!(diff.inserts[0].0, r1);
}

#[test]
fn test_diff_outputs_no_change() {
    let r1 = RowId::new(smallvec::smallvec![(0, BoundEntityId::Node(1))]);
    let mut old_rows = rustc_hash::FxHashMap::default();
    old_rows.insert(r1.clone(), vec![Value::String("Alice".to_string())]);
    let old = CachedOutput {
        columns: vec!["name".to_string()],
        rows: old_rows,
    };

    let mut new_rows = rustc_hash::FxHashMap::default();
    new_rows.insert(r1.clone(), vec![Value::String("Alice".to_string())]);

    let diff = diff_outputs(&old, &new_rows);
    assert_eq!(diff.deletes.len(), 0);
    assert_eq!(diff.inserts.len(), 0);
}

// ── Phase 3A: Manager integration tests ──

use super::manager::*;

#[test]
fn test_manager_subscribe_scan_only() {
    use crate::ontology::OntologyRegistry;
    use crate::query_lang::planner::*;

    let mut graph = Graph::new();
    let _n1 = graph.add_node(concept_node("Alice")).unwrap();
    let _n2 = graph.add_node(concept_node("Bob")).unwrap();
    let rx = graph.enable_subscriptions();

    let ontology = OntologyRegistry::new();

    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![Projection {
            expr: RExpr::Property(0, "name".to_string()),
            alias: "name".to_string(),
            distinct: false,
        }],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };

    let mut mgr = SubscriptionManager::new(rx);
    let (sub_id, initial) = mgr.subscribe(plan, &graph, &ontology).unwrap();
    assert_eq!(sub_id, 1);
    assert_eq!(initial.rows.len(), 2); // Alice and Bob
    assert_eq!(mgr.subscription_count(), 1);

    // Add a new node and process.
    let _n3 = graph.add_node(concept_node("Carol")).unwrap();
    mgr.drain_and_process(&graph, &ontology);
    let updates: Vec<_> = mgr.take_all_pending();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].subscription_id, sub_id);
    assert_eq!(updates[0].inserts.len(), 1);
    assert_eq!(updates[0].deletes.len(), 0);
    assert!(!updates[0].was_full_rerun);

    // Unsubscribe.
    assert!(mgr.unsubscribe(sub_id));
    assert_eq!(mgr.subscription_count(), 0);
}

#[test]
fn test_manager_subscribe_scan_expand() {
    use crate::ontology::OntologyRegistry;
    use crate::query_lang::planner::*;

    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let _eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    let rx = graph.enable_subscriptions();

    let ontology = OntologyRegistry::new();

    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec!["Concept".to_string()],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: Some(1),
                to_var: 2,
                edge_type: Some("KNOWS".to_string()),
                direction: crate::query_lang::ast::Direction::Out,
                range: None,
            },
        ],
        projections: vec![
            Projection {
                expr: RExpr::Property(0, "name".to_string()),
                alias: "from".to_string(),
                distinct: false,
            },
            Projection {
                expr: RExpr::Property(2, "name".to_string()),
                alias: "to".to_string(),
                distinct: false,
            },
        ],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 3,
    };

    let mut mgr = SubscriptionManager::new(rx);
    let (sub_id, initial) = mgr.subscribe(plan, &graph, &ontology).unwrap();
    assert_eq!(initial.rows.len(), 1); // Alice -> Bob

    // Add another edge.
    let n3 = graph.add_node(concept_node("Carol")).unwrap();
    let _eid2 = graph.add_edge(assoc_edge(n1, n3, "KNOWS")).unwrap();
    mgr.drain_and_process(&graph, &ontology);
    let updates: Vec<_> = mgr.take_all_pending();

    // Should have at least one update with the new edge.
    let total_inserts: usize = updates.iter().map(|u| u.inserts.len()).sum();
    assert!(
        total_inserts >= 1,
        "Expected at least 1 insert for new edge, got {}",
        total_inserts
    );
}

#[test]
fn test_manager_edge_remove_propagates() {
    use crate::ontology::OntologyRegistry;
    use crate::query_lang::planner::*;

    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Bob")).unwrap();
    let eid = graph.add_edge(assoc_edge(n1, n2, "KNOWS")).unwrap();
    let rx = graph.enable_subscriptions();

    let ontology = OntologyRegistry::new();

    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec!["Concept".to_string()],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: Some(1),
                to_var: 2,
                edge_type: Some("KNOWS".to_string()),
                direction: crate::query_lang::ast::Direction::Out,
                range: None,
            },
        ],
        projections: vec![
            Projection {
                expr: RExpr::Property(0, "name".to_string()),
                alias: "from".to_string(),
                distinct: false,
            },
            Projection {
                expr: RExpr::Property(2, "name".to_string()),
                alias: "to".to_string(),
                distinct: false,
            },
        ],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 3,
    };

    let mut mgr = SubscriptionManager::new(rx);
    let (sub_id, initial) = mgr.subscribe(plan, &graph, &ontology).unwrap();
    assert_eq!(initial.rows.len(), 1);

    // Remove the edge.
    graph.remove_edge(eid);
    mgr.drain_and_process(&graph, &ontology);
    let updates: Vec<_> = mgr.take_all_pending();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].deletes.len(), 1);
    assert_eq!(updates[0].inserts.len(), 0);
}

#[test]
fn test_manager_count_aggregation() {
    use crate::ontology::OntologyRegistry;
    use crate::query_lang::planner::*;

    let mut graph = Graph::new();
    let _n1 = graph.add_node(concept_node("Alice")).unwrap();
    let _n2 = graph.add_node(concept_node("Bob")).unwrap();
    let rx = graph.enable_subscriptions();

    let ontology = OntologyRegistry::new();

    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![Projection {
            expr: RExpr::Star,
            alias: "count".to_string(),
            distinct: false,
        }],
        aggregations: vec![Aggregation {
            function: AggregateFunction::Count,
            input_expr: RExpr::Star,
            output_alias: "count".to_string(),
        }],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };

    let mut mgr = SubscriptionManager::new(rx);
    let (sub_id, _initial) = mgr.subscribe(plan, &graph, &ontology).unwrap();

    // Add a node.
    let _n3 = graph.add_node(concept_node("Carol")).unwrap();
    mgr.drain_and_process(&graph, &ontology);
    let updates: Vec<_> = mgr.take_all_pending();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].count, Some(3)); // 2 + 1
}

#[test]
fn test_manager_no_update_when_irrelevant() {
    use crate::ontology::OntologyRegistry;
    use crate::query_lang::planner::*;

    let mut graph = Graph::new();
    let _n1 = graph.add_node(concept_node("Alice")).unwrap();
    let rx = graph.enable_subscriptions();

    let ontology = OntologyRegistry::new();

    // Subscribe to KNOWS edges only.
    let plan = ExecutionPlan {
        steps: vec![
            PlanStep::ScanNodes {
                var: 0,
                labels: vec!["Concept".to_string()],
                props: vec![],
            },
            PlanStep::Expand {
                from_var: 0,
                edge_var: Some(1),
                to_var: 2,
                edge_type: Some("KNOWS".to_string()),
                direction: crate::query_lang::ast::Direction::Out,
                range: None,
            },
        ],
        projections: vec![Projection {
            expr: RExpr::Property(0, "name".to_string()),
            alias: "name".to_string(),
            distinct: false,
        }],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 3,
    };

    let mut mgr = SubscriptionManager::new(rx);
    let (_sub_id, _) = mgr.subscribe(plan, &graph, &ontology).unwrap();

    // Add a WORKS_AT edge — should not trigger update for KNOWS subscription.
    let n2 = graph.add_node(concept_node("Acme")).unwrap();
    let _eid = graph.add_edge(assoc_edge(_n1, n2, "WORKS_AT")).unwrap();
    mgr.drain_and_process(&graph, &ontology);
    let updates: Vec<_> = mgr.take_all_pending();

    // The node add for "Acme" may trigger a scan delta but expand won't match,
    // so we should get no meaningful output deltas (or an empty update at most).
    let total_inserts: usize = updates.iter().map(|u| u.inserts.len()).sum();
    let total_deletes: usize = updates.iter().map(|u| u.deletes.len()).sum();
    assert_eq!(
        total_inserts, 0,
        "Should not have inserts for WORKS_AT edge on KNOWS subscription"
    );
    assert_eq!(
        total_deletes, 0,
        "Should not have deletes for WORKS_AT edge on KNOWS subscription"
    );
}

#[test]
fn test_manager_node_merge_forces_rerun() {
    use crate::ontology::OntologyRegistry;
    use crate::query_lang::planner::*;

    let mut graph = Graph::new();
    let n1 = graph.add_node(concept_node("Alice")).unwrap();
    let n2 = graph.add_node(concept_node("Alice2")).unwrap();
    let rx = graph.enable_subscriptions();

    let ontology = OntologyRegistry::new();

    let plan = ExecutionPlan {
        steps: vec![PlanStep::ScanNodes {
            var: 0,
            labels: vec!["Concept".to_string()],
            props: vec![],
        }],
        projections: vec![Projection {
            expr: RExpr::Property(0, "name".to_string()),
            alias: "name".to_string(),
            distinct: false,
        }],
        aggregations: vec![],
        group_by_keys: vec![],
        ordering: vec![],
        limit: None,
        temporal_viewport: TemporalViewport::ActiveOnly,
        transaction_cutoff: None,
        var_count: 1,
    };

    let mut mgr = SubscriptionManager::new(rx);
    let (sub_id, initial) = mgr.subscribe(plan, &graph, &ontology).unwrap();
    assert_eq!(initial.rows.len(), 2);

    // Merge nodes.
    let _ = graph.merge_nodes(n1, n2);
    mgr.drain_and_process(&graph, &ontology);
    let updates: Vec<_> = mgr.take_all_pending();

    // The merge causes NodeRemoved (absorbed) then NodeMerged.
    // NodeRemoved is processed incrementally (deletes absorbed row).
    // NodeMerged forces a full rerun, but since the state is now correct
    // (only survivor remains), the rerun diff may be empty.
    // The key invariant is correctness: after processing, subscription
    // should reflect the merged state.
    let total_deletes: usize = updates.iter().map(|u| u.deletes.len()).sum();
    // At least one delete should have occurred (the absorbed node).
    assert!(
        total_deletes >= 1,
        "Expected at least 1 delete from merge, got {}",
        total_deletes
    );
}
