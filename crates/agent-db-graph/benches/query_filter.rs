//! Criterion benchmarks for MinnsQL query filter optimizations.
//!
//! Measures the impact of pre-compiled IN filter sets on query execution time.
//! Run with: cargo bench --bench query_filter

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;

use agent_db_graph::ontology::OntologyRegistry;
use agent_db_graph::query_lang::executor::Executor;
use agent_db_graph::query_lang::parser::Parser;
use agent_db_graph::query_lang::planner::plan;
use agent_db_graph::structures::{ConceptType, Graph, GraphNode, NodeType};

/// Create a graph with N concept nodes, each having a "status" property
/// drawn from a set of 10 possible values.
fn build_test_graph(node_count: usize) -> Graph {
    let mut graph = Graph::with_max_size(node_count + 100);
    let statuses = [
        "active",
        "pending",
        "review",
        "escalated",
        "waiting",
        "closed",
        "archived",
        "draft",
        "blocked",
        "cancelled",
    ];

    for i in 0..node_count {
        let status = statuses[i % statuses.len()];
        let name = format!("entity_{}", i);
        let mut props = HashMap::new();
        props.insert(
            "status".to_string(),
            serde_json::Value::String(status.to_string()),
        );
        props.insert("name".to_string(), serde_json::Value::String(name.clone()));

        let mut node = GraphNode::new(NodeType::Concept {
            concept_name: name,
            concept_type: ConceptType::NamedEntity,
            confidence: 1.0,
        });
        node.properties = props;
        node.group_id = "bench".to_string();
        let _ = graph.add_node(node);
    }
    graph
}

/// Benchmark: MATCH (n:Concept) WHERE n.status IN (...5 values...) RETURN n.name
fn bench_in_filter(c: &mut Criterion) {
    let ontology = OntologyRegistry::new();
    let mut group = c.benchmark_group("in_filter_5_values");

    for &node_count in &[1_000, 10_000, 50_000] {
        let graph = build_test_graph(node_count);

        let query_str = r#"MATCH (n:Concept) WHERE n.status IN ("active", "pending", "review", "escalated", "waiting") RETURN n.name"#;
        let ast = Parser::parse(query_str).expect("parse failed");
        let execution_plan = plan(ast).expect("plan failed");

        group.bench_with_input(
            BenchmarkId::new("precompiled_in", node_count),
            &node_count,
            |b, _| {
                b.iter(|| {
                    Executor::execute(
                        black_box(&graph),
                        black_box(&ontology),
                        black_box(execution_plan.clone()),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: same query shape but with only 1 IN value (minimal benefit from precompilation)
fn bench_in_filter_1_value(c: &mut Criterion) {
    let ontology = OntologyRegistry::new();
    let mut group = c.benchmark_group("in_filter_1_value");

    for &node_count in &[1_000, 10_000, 50_000] {
        let graph = build_test_graph(node_count);

        let query_str = r#"MATCH (n:Concept) WHERE n.status IN ("active") RETURN n.name"#;
        let ast = Parser::parse(query_str).expect("parse failed");
        let execution_plan = plan(ast).expect("plan failed");

        group.bench_with_input(
            BenchmarkId::new("precompiled_in_1", node_count),
            &node_count,
            |b, _| {
                b.iter(|| {
                    Executor::execute(
                        black_box(&graph),
                        black_box(&ontology),
                        black_box(execution_plan.clone()),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: equality filter (baseline comparison, no IN clause)
fn bench_equality_filter(c: &mut Criterion) {
    let ontology = OntologyRegistry::new();
    let mut group = c.benchmark_group("equality_filter");

    for &node_count in &[1_000, 10_000, 50_000] {
        let graph = build_test_graph(node_count);

        let query_str = r#"MATCH (n:Concept) WHERE n.status = "active" RETURN n.name"#;
        let ast = Parser::parse(query_str).expect("parse failed");
        let execution_plan = plan(ast).expect("plan failed");

        group.bench_with_input(
            BenchmarkId::new("equality", node_count),
            &node_count,
            |b, _| {
                b.iter(|| {
                    Executor::execute(
                        black_box(&graph),
                        black_box(&ontology),
                        black_box(execution_plan.clone()),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: scan without any filter (raw scan cost baseline)
fn bench_scan_only(c: &mut Criterion) {
    let ontology = OntologyRegistry::new();
    let mut group = c.benchmark_group("scan_no_filter");

    for &node_count in &[1_000, 10_000, 50_000] {
        let graph = build_test_graph(node_count);

        let query_str = r#"MATCH (n:Concept) RETURN n.name LIMIT 50000"#;
        let ast = Parser::parse(query_str).expect("parse failed");
        let execution_plan = plan(ast).expect("plan failed");

        group.bench_with_input(
            BenchmarkId::new("scan_only", node_count),
            &node_count,
            |b, _| {
                b.iter(|| {
                    Executor::execute(
                        black_box(&graph),
                        black_box(&ontology),
                        black_box(execution_plan.clone()),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_in_filter,
    bench_in_filter_1_value,
    bench_equality_filter,
    bench_scan_only,
);
criterion_main!(benches);
