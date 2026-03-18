//! Tests for graph traversal and query engine.

use super::*;
use crate::structures::{EdgeId, EdgeType, Graph, GraphEdge, GraphNode, NodeId, NodeType};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::sync::Arc;

fn make_node_at(id: NodeId, created_at: u64) -> GraphNode {
    GraphNode {
        id,
        node_type: NodeType::Event {
            event_id: id as u128,
            event_type: format!("type_{}", id),
            significance: 0.5,
        },
        created_at,
        updated_at: created_at,
        properties: HashMap::new(),
        degree: 0,
        embedding: Vec::new(),
        group_id: String::new(),
    }
}

fn make_edge_at(source: NodeId, target: NodeId, created_at: u64) -> GraphEdge {
    GraphEdge {
        id: 0,
        source,
        target,
        edge_type: EdgeType::Causality {
            strength: 0.8,
            lag_ms: 100,
        },
        weight: 1.0,
        created_at,
        updated_at: created_at,
        valid_from: None,
        valid_until: None,
        observation_count: 1,
        confidence: 0.9,
        properties: HashMap::new(),
        confidence_history: crate::tcell::TCell::Empty,
        weight_history: crate::tcell::TCell::Empty,
        group_id: String::new(),
    }
}

/// Build: 1 -> 2 -> 3, 1 -> 4
fn build_directed_graph() -> Graph {
    let mut g = Graph::new();
    let mut n = make_node_at(0, 100);
    let id1 = g.add_node(n.clone()).unwrap();
    n.created_at = 200;
    let id2 = g.add_node(n.clone()).unwrap();
    n.created_at = 300;
    let id3 = g.add_node(n.clone()).unwrap();
    n.created_at = 400;
    let id4 = g.add_node(n.clone()).unwrap();

    g.add_edge(make_edge_at(id1, id2, 150));
    g.add_edge(make_edge_at(id2, id3, 250));
    g.add_edge(make_edge_at(id1, id4, 350));
    g
}

// ── Direction / Depth unit tests ──

#[test]
fn depth_validate_valid() {
    use crate::structures::Depth;
    assert!(Depth::Fixed(5).validate().is_ok());
    assert!(Depth::Range(1, 5).validate().is_ok());
    assert!(Depth::Range(3, 3).validate().is_ok());
    assert!(Depth::Unbounded.validate().is_ok());
}

#[test]
fn depth_validate_invalid() {
    use crate::structures::Depth;
    assert!(Depth::Range(5, 2).validate().is_err());
}

#[test]
fn depth_min_max() {
    use crate::structures::Depth;
    assert_eq!(Depth::Fixed(3).min_depth(), 3);
    assert_eq!(Depth::Fixed(3).max_depth(), Some(3));
    assert_eq!(Depth::Range(1, 5).min_depth(), 1);
    assert_eq!(Depth::Range(1, 5).max_depth(), Some(5));
    assert_eq!(Depth::Unbounded.min_depth(), 0);
    assert_eq!(Depth::Unbounded.max_depth(), None);
}

#[test]
fn neighbors_directed_out() {
    use crate::structures::Direction;
    let g = build_directed_graph();
    let mut n = g.neighbors_directed(1, Direction::Out);
    n.sort();
    assert_eq!(n, vec![2, 4]);
}

#[test]
fn neighbors_directed_in() {
    use crate::structures::Direction;
    let g = build_directed_graph();
    let n = g.neighbors_directed(2, Direction::In);
    assert_eq!(n, vec![1]);
}

#[test]
fn neighbors_directed_both() {
    use crate::structures::Direction;
    let g = build_directed_graph();
    // Node 2: out=[3], in=[1] → both=[3,1] or [1,3]
    let n = g.neighbors_directed(2, Direction::Both);
    assert_eq!(n.len(), 2);
    assert!(n.contains(&1));
    assert!(n.contains(&3));
}

#[test]
fn edges_directed_out() {
    use crate::structures::Direction;
    let g = build_directed_graph();
    let edges = g.edges_directed(1, Direction::Out);
    assert_eq!(edges.len(), 2);
    let targets: Vec<NodeId> = edges.iter().map(|e| e.target).collect();
    assert!(targets.contains(&2));
    assert!(targets.contains(&4));
}

#[test]
fn edges_directed_in() {
    use crate::structures::Direction;
    let g = build_directed_graph();
    let edges = g.edges_directed(2, Direction::In);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].source, 1);
}

#[test]
fn edges_directed_both_dedup() {
    use crate::structures::{Direction, EdgeId};
    let g = build_directed_graph();
    let edges = g.edges_directed(2, Direction::Both);
    // out: edge to 3; in: edge from 1 → 2 edges total, all distinct
    assert_eq!(edges.len(), 2);
    let mut ids: Vec<EdgeId> = edges.iter().map(|e| e.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 2);
}

#[test]
fn latest_timestamp_empty() {
    let g = Graph::new();
    assert_eq!(g.latest_timestamp(), None);
}

#[test]
fn latest_timestamp_nonempty() {
    let g = build_directed_graph();
    // Nodes created at 100, 200, 300, 400
    assert_eq!(g.latest_timestamp(), Some(400));
}

// ── DirectedBfsIter tests ──

#[test]
fn directed_bfs_out() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let nodes: Vec<NodeId> = DirectedBfsIter::new(&g, 1, Direction::Out, Depth::Range(0, 2))
        .map(|(id, _)| id)
        .collect();
    assert!(nodes.contains(&1));
    assert!(nodes.contains(&2));
    assert!(nodes.contains(&3));
    assert!(nodes.contains(&4));
}

#[test]
fn directed_bfs_in() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    // From node 3, going In: 3 ← 2 ← 1
    let nodes: Vec<NodeId> = DirectedBfsIter::new(&g, 3, Direction::In, Depth::Range(0, 2))
        .map(|(id, _)| id)
        .collect();
    assert!(nodes.contains(&3));
    assert!(nodes.contains(&2));
    assert!(nodes.contains(&1));
}

#[test]
fn directed_bfs_depth_range() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    // Depth Range(1, 2): skip depth 0 (start node), collect depths 1 and 2
    let nodes: Vec<NodeId> = DirectedBfsIter::new(&g, 1, Direction::Out, Depth::Range(1, 2))
        .map(|(id, _)| id)
        .collect();
    assert!(!nodes.contains(&1)); // depth 0, skipped
    assert!(nodes.contains(&2)); // depth 1
    assert!(nodes.contains(&4)); // depth 1
    assert!(nodes.contains(&3)); // depth 2
}

// ── DirectedDfsIter tests ──

#[test]
fn directed_dfs_both() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let nodes: Vec<NodeId> = DirectedDfsIter::new(&g, 2, Direction::Both, Depth::Range(0, 1))
        .map(|(id, _)| id)
        .collect();
    assert!(nodes.contains(&2)); // start
    assert!(nodes.len() >= 2); // at least one neighbor
}

// ── DirectedDijkstraIter tests ──

#[test]
fn directed_dijkstra_out() {
    use crate::structures::Direction;
    let g = build_directed_graph();
    let results: Vec<(NodeId, f32)> = DirectedDijkstraIter::new(&g, 1, Direction::Out).collect();
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1); // start node first
    assert_eq!(results[0].1, 0.0); // zero cost to self
}

// ── execute_traversal tests ──

#[test]
fn execute_collect_basic() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Range(0, 2),
        instruction: Instruction::Collect,
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: None,
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(nodes.contains(&1));
            assert!(nodes.contains(&2));
            assert!(nodes.contains(&3));
            assert!(nodes.contains(&4));
        },
        _ => panic!("Expected Nodes"),
    }
}

#[test]
fn execute_collect_with_depth_range() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Range(1, 1),
        instruction: Instruction::Collect,
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: None,
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(!nodes.contains(&1)); // depth 0
            assert!(nodes.contains(&2)); // depth 1
            assert!(nodes.contains(&4)); // depth 1
            assert!(!nodes.contains(&3)); // depth 2
        },
        _ => panic!("Expected Nodes"),
    }
}

#[test]
fn execute_collect_with_node_filter() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Range(0, 2),
        instruction: Instruction::Collect,
        node_filter: Some(Arc::new(|node: &GraphNode| node.id != 4)),
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: None,
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(!nodes.contains(&4));
            assert!(nodes.contains(&2));
        },
        _ => panic!("Expected Nodes"),
    }
}

#[test]
fn execute_collect_with_budget() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Unbounded,
        instruction: Instruction::Collect,
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: Some(2),
        max_edges_traversed: None,
        time_window: None,
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(nodes.len() <= 2);
        },
        _ => panic!("Expected Nodes"),
    }
}

#[test]
fn execute_collect_with_time_window() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    // Only allow edges/nodes in [100, 200]
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Range(0, 3),
        instruction: Instruction::Collect,
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: Some((100, 200)),
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(nodes.contains(&1)); // t=100, in range
            assert!(nodes.contains(&2)); // t=200, in range; edge t=150 in range
            assert!(!nodes.contains(&3)); // t=300, out of range
        },
        _ => panic!("Expected Nodes"),
    }
}

#[test]
fn execute_paths_simple() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Range(0, 3),
        instruction: Instruction::Path { max_paths: 10 },
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: None,
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Paths(paths) => {
            assert!(!paths.is_empty());
            // All paths should start with node 1
            for path in &paths {
                assert_eq!(path[0], 1);
            }
        },
        _ => panic!("Expected Paths"),
    }
}

#[test]
fn execute_shortest_directed() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Unbounded,
        instruction: Instruction::Shortest(3),
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: None,
    };
    let result = execute_traversal(&g, &spec).unwrap();
    match result {
        QueryResult::Path(path) => {
            assert_eq!(path[0], 1);
            assert_eq!(*path.last().unwrap(), 3);
        },
        _ => panic!("Expected Path"),
    }
}

#[test]
fn execute_shortest_no_path() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    // Going In from node 1 cannot reach node 3
    let spec = TraversalSpec {
        start: 1,
        direction: Direction::In,
        depth: Depth::Unbounded,
        instruction: Instruction::Shortest(3),
        node_filter: None,
        edge_filter: None,
        max_nodes_visited: None,
        max_edges_traversed: None,
        time_window: None,
    };
    assert!(execute_traversal(&g, &spec).is_err());
}

// ── TraversalRequest compile tests ──

#[test]
fn traversal_request_compile() {
    use crate::structures::{Depth, Direction};
    let req = TraversalRequest {
        start: 1,
        direction: Direction::Out,
        depth: Depth::Fixed(3),
        instruction: Instruction::Collect,
        node_filters: vec![NodeFilterExpr::ByType("Event".to_string())],
        edge_filters: vec![EdgeFilterExpr::MinWeight(OrderedFloat(0.5))],
        max_nodes_visited: Some(100),
        max_edges_traversed: None,
        time_window: None,
    };
    let spec = req.compile();
    assert_eq!(spec.start, 1);
    assert!(spec.node_filter.is_some());
    assert!(spec.edge_filter.is_some());
}

// ── GraphQuery dispatch tests ──

#[test]
fn query_directed_traversal() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let engine = GraphTraversal::new();
    let result = engine
        .execute_query(
            &g,
            GraphQuery::DirectedTraversal {
                start: 1,
                direction: Direction::Out,
                depth: Depth::Range(0, 1),
            },
        )
        .unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(nodes.contains(&1));
            assert!(nodes.contains(&2));
            assert!(nodes.contains(&4));
        },
        _ => panic!("Expected Nodes"),
    }
}

#[test]
fn query_recursive_traversal() {
    use crate::structures::{Depth, Direction};
    let g = build_directed_graph();
    let engine = GraphTraversal::new();
    let result = engine
        .execute_query(
            &g,
            GraphQuery::RecursiveTraversal(TraversalRequest {
                start: 1,
                direction: Direction::Out,
                depth: Depth::Range(0, 2),
                instruction: Instruction::Collect,
                node_filters: vec![],
                edge_filters: vec![],
                max_nodes_visited: None,
                max_edges_traversed: None,
                time_window: None,
            }),
        )
        .unwrap();
    match result {
        QueryResult::Nodes(nodes) => {
            assert!(nodes.len() >= 3);
        },
        _ => panic!("Expected Nodes"),
    }
}

// ── QueryContext tests ──

#[test]
fn query_context_new() {
    let ctx = QueryContext::new(100);
    assert!(!ctx.is_done());
    assert_eq!(ctx.items_yielded(), 0);
}

#[test]
fn query_context_limit() {
    let ctx = QueryContext::new(0);
    assert!(ctx.is_done()); // limit=0 means immediately done
}

#[test]
fn query_context_cancel() {
    let ctx = QueryContext::new(100);
    let handle = ctx.cancel_handle();
    assert!(!ctx.is_done());
    handle.cancel();
    assert!(ctx.is_done());
}

#[test]
fn cancel_handle_clone() {
    let ctx = QueryContext::new(100);
    let h1 = ctx.cancel_handle();
    let h2 = h1.clone();
    h2.cancel();
    assert!(h1.is_cancelled());
    assert!(ctx.is_done());
}

// ── StreamingQuery tests ──

#[test]
fn streaming_next_batch() {
    let data = vec![1, 2, 3, 4, 5];
    let mut sq = StreamingQuery::new(data.into_iter(), 100, 2);
    let b1 = sq.next_batch().unwrap();
    assert_eq!(b1, vec![1, 2]);
    let b2 = sq.next_batch().unwrap();
    assert_eq!(b2, vec![3, 4]);
    let b3 = sq.next_batch().unwrap();
    assert_eq!(b3, vec![5]);
    assert!(sq.next_batch().is_none());
}

#[test]
fn streaming_collect_all() {
    let data = vec![10, 20, 30];
    let mut sq = StreamingQuery::new(data.into_iter(), 100, 2);
    let all = sq.collect_all();
    assert_eq!(all, vec![10, 20, 30]);
}

#[test]
fn streaming_respects_limit() {
    let data = vec![1, 2, 3, 4, 5];
    let mut sq = StreamingQuery::new(data.into_iter(), 3, 10);
    let all = sq.collect_all();
    assert_eq!(all.len(), 3);
}

#[test]
fn streaming_respects_cancel() {
    let data = vec![1, 2, 3, 4, 5];
    let mut sq = StreamingQuery::new(data.into_iter(), 100, 2);
    let handle = sq.context().cancel_handle();
    let b1 = sq.next_batch().unwrap();
    assert_eq!(b1.len(), 2);
    handle.cancel();
    assert!(sq.next_batch().is_none());
}

#[test]
fn streaming_empty_iterator() {
    let data: Vec<i32> = vec![];
    let mut sq = StreamingQuery::new(data.into_iter(), 100, 10);
    assert!(sq.next_batch().is_none());
}

#[test]
fn streaming_items_yielded() {
    let data = vec![1, 2, 3];
    let mut sq = StreamingQuery::new(data.into_iter(), 100, 10);
    sq.collect_all();
    assert_eq!(sq.context().items_yielded(), 3);
}
