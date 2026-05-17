//! Integration tests that exercise `QdrantStore` against a real Qdrant
//! container started by `testcontainers`.
//!
//! Every test is marked `#[ignore]` so a fresh `cargo test` is fast and works
//! on machines without Docker. Run the full suite with:
//!
//! ```sh
//! cargo test -p minns-vectors -- --ignored
//! ```
//!
//! Each test spins up its own container so failure in one cannot corrupt the
//! state another observes. Container teardown happens when the handle is
//! dropped at the end of each test.

use minns_vectors::{
    Distance, Filter, FilterValue, Payload, Point, QdrantConfig, QdrantStore, Quantization, Query,
    ScalarQuantization, VectorError, VectorStore,
};
use testcontainers::core::{IntoContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};

const QDRANT_IMAGE: &str = "qdrant/qdrant";
const QDRANT_TAG: &str = "v1.13.0";
const QDRANT_GRPC: u16 = 6334;
const DIM: usize = 4;

/// Spin up a Qdrant container and return the container handle (the test must
/// keep this in scope to keep Qdrant alive) plus the gRPC URL pointing at it.
async fn start_qdrant() -> (ContainerAsync<GenericImage>, String) {
    let image = GenericImage::new(QDRANT_IMAGE, QDRANT_TAG)
        .with_exposed_port(QDRANT_GRPC.tcp())
        .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening"))
        .with_startup_timeout(std::time::Duration::from_secs(60));
    let container = image.start().await.expect("start qdrant container");
    let port = container
        .get_host_port_ipv4(QDRANT_GRPC)
        .await
        .expect("read mapped gRPC port");
    let url = format!("http://127.0.0.1:{port}");
    (container, url)
}

async fn open_store(url: &str, collection: &str) -> QdrantStore {
    QdrantStore::open(QdrantConfig::new(url, collection, DIM).with_distance(Distance::Cosine))
        .await
        .expect("open qdrant store")
}

fn point(id: u128, vector: Vec<f32>, payload: Payload) -> Point {
    Point::new(id, vector, payload)
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn upsert_then_fetch_round_trips_vectors_and_payloads() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "fetch_test").await;

    let original = point(
        1,
        vec![1.0, 0.0, 0.0, 0.0],
        Payload::new()
            .with("agent_id", 7u64)
            .with("tier", "episodic"),
    );
    store.upsert(vec![original.clone()]).await.unwrap();

    let fetched = store.fetch(&[1, 999]).await.unwrap();
    assert_eq!(fetched.len(), 2, "result aligned with input length");

    let got = fetched[0].as_ref().expect("point 1 present");
    assert_eq!(got.id, 1);
    assert_eq!(got.vector, original.vector);
    // u64 round-trips as i64 because Qdrant payloads use signed integers.
    assert_eq!(got.payload.get("agent_id"), Some(&FilterValue::I64(7)));
    assert_eq!(
        got.payload.get("tier"),
        Some(&FilterValue::Str("episodic".into()))
    );

    assert!(fetched[1].is_none(), "missing id reported as None");
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn search_returns_top_k_by_cosine_similarity() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "search_top_k").await;

    store
        .upsert(vec![
            point(1, vec![1.0, 0.0, 0.0, 0.0], Payload::EMPTY),
            point(2, vec![0.9, 0.1, 0.0, 0.0], Payload::EMPTY),
            point(3, vec![0.0, 0.0, 1.0, 0.0], Payload::EMPTY),
        ])
        .await
        .unwrap();

    let hits = store
        .search(&Query::builder(vec![1.0, 0.0, 0.0, 0.0]).top_k(2).build())
        .await
        .unwrap();

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, 1, "closest match first under cosine");
    assert_eq!(hits[1].id, 2);
    assert!(hits[0].score >= hits[1].score, "results ordered by score");
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn search_applies_filter_during_search() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "search_filtered").await;

    store
        .upsert(vec![
            point(
                1,
                vec![1.0, 0.0, 0.0, 0.0],
                Payload::new()
                    .with("agent_id", 1u64)
                    .with("status", "active"),
            ),
            point(
                2,
                vec![0.95, 0.05, 0.0, 0.0],
                Payload::new()
                    .with("agent_id", 2u64)
                    .with("status", "active"),
            ),
            point(
                3,
                vec![0.9, 0.1, 0.0, 0.0],
                Payload::new()
                    .with("agent_id", 1u64)
                    .with("status", "archived"),
            ),
        ])
        .await
        .unwrap();

    let hits = store
        .search(
            &Query::builder(vec![1.0, 0.0, 0.0, 0.0])
                .top_k(10)
                .filter(Filter::new().eq("agent_id", 1u64).neq("status", "archived"))
                .build(),
        )
        .await
        .unwrap();

    assert_eq!(hits.len(), 1, "filter excludes wrong agent and archived");
    assert_eq!(hits[0].id, 1);
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn search_respects_min_score_threshold() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "search_min_score").await;

    store
        .upsert(vec![
            point(1, vec![1.0, 0.0, 0.0, 0.0], Payload::EMPTY),
            point(2, vec![0.0, 0.0, 1.0, 0.0], Payload::EMPTY),
        ])
        .await
        .unwrap();

    let hits = store
        .search(
            &Query::builder(vec![1.0, 0.0, 0.0, 0.0])
                .top_k(10)
                .min_score(0.9)
                .build(),
        )
        .await
        .unwrap();

    assert_eq!(hits.len(), 1, "only the high-score match passes threshold");
    assert_eq!(hits[0].id, 1);
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn delete_removes_points_and_reports_actual_count() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "delete_test").await;

    store
        .upsert(vec![
            point(1, vec![1.0, 0.0, 0.0, 0.0], Payload::EMPTY),
            point(2, vec![0.0, 1.0, 0.0, 0.0], Payload::EMPTY),
        ])
        .await
        .unwrap();

    let removed = store.delete(&[1, 999]).await.unwrap();
    assert_eq!(removed, 1, "only the existing id counts as removed");

    let after = store.fetch(&[1, 2]).await.unwrap();
    assert!(after[0].is_none(), "id 1 was deleted");
    assert!(after[1].is_some(), "id 2 untouched");
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn count_reflects_collection_size() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "count_test").await;

    assert_eq!(store.count().await.unwrap(), 0);

    store
        .upsert(vec![
            point(1, vec![1.0, 0.0, 0.0, 0.0], Payload::EMPTY),
            point(2, vec![0.0, 1.0, 0.0, 0.0], Payload::EMPTY),
            point(3, vec![0.0, 0.0, 1.0, 0.0], Payload::EMPTY),
        ])
        .await
        .unwrap();

    assert_eq!(store.count().await.unwrap(), 3);
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn reopening_with_mismatched_dim_returns_dimension_mismatch() {
    let (_container, url) = start_qdrant().await;
    let _ = open_store(&url, "dim_test").await;

    match QdrantStore::open(QdrantConfig::new(&url, "dim_test", DIM + 1)).await {
        Ok(_) => panic!("opening with the wrong dim must fail"),
        Err(VectorError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, DIM + 1);
            assert_eq!(got, DIM);
        },
        Err(other) => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn empty_inputs_are_no_ops() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "empty_inputs").await;

    store.upsert(vec![]).await.expect("empty upsert no-ops");
    assert_eq!(
        store.delete(&[]).await.expect("empty delete no-ops"),
        0,
        "empty delete reports zero"
    );
    assert!(
        store
            .fetch(&[])
            .await
            .expect("empty fetch no-ops")
            .is_empty(),
        "empty fetch returns empty vec"
    );
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn search_with_zero_top_k_is_rejected() {
    let (_container, url) = start_qdrant().await;
    let store = open_store(&url, "invalid_query").await;

    let err = store
        .search(&Query::builder(vec![1.0, 0.0, 0.0, 0.0]).top_k(0).build())
        .await
        .expect_err("top_k = 0 must be rejected");
    assert!(matches!(err, VectorError::InvalidQuery(_)));
}

#[tokio::test]
#[ignore = "requires Docker; run with `cargo test -- --ignored`"]
async fn quantization_configured_at_creation_is_accepted_and_searchable() {
    let (_container, url) = start_qdrant().await;
    let store = QdrantStore::open(
        QdrantConfig::new(&url, "quant_test", DIM).with_quantization(Quantization::Scalar(
            ScalarQuantization {
                quantile: None,
                always_ram: true,
            },
        )),
    )
    .await
    .expect("scalar quantization config accepted");

    store
        .upsert(vec![
            point(1, vec![1.0, 0.0, 0.0, 0.0], Payload::EMPTY),
            point(2, vec![0.0, 1.0, 0.0, 0.0], Payload::EMPTY),
        ])
        .await
        .unwrap();

    let hits = store
        .search(&Query::builder(vec![1.0, 0.0, 0.0, 0.0]).top_k(1).build())
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 1);
}
