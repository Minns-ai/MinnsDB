//! Smoke test for the shared Qdrant fixture. Verifies the fixture starts a
//! container and returns a usable [`VectorsConfig`]. Ignored by default
//! because it needs Docker; run with `cargo test -- --ignored`.

mod common;

use common::qdrant_fixture::{qdrant_url, test_vectors_config};

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn fixture_starts_a_container_and_returns_a_url() {
    let url = qdrant_url().await;
    assert!(url.starts_with("http://127.0.0.1:"));
    let again = qdrant_url().await;
    assert_eq!(
        url, again,
        "shared fixture returns the same URL across calls"
    );
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn each_call_to_test_vectors_config_uses_a_distinct_prefix() {
    let a = test_vectors_config().await;
    let b = test_vectors_config().await;
    assert_ne!(a.collection_prefix, b.collection_prefix);
    assert_eq!(a.url, b.url);
    assert_eq!(a.dim, agent_db_graph::vectors::DEFAULT_DIM);
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn graph_engine_constructs_against_the_fixture() {
    let vc = test_vectors_config().await;
    let config = agent_db_graph::GraphEngineConfig {
        vectors_config: vc,
        ..Default::default()
    };
    let _engine = agent_db_graph::GraphEngine::with_config(config)
        .await
        .expect("engine opens against shared qdrant");
}
