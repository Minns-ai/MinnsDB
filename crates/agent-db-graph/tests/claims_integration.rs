//! Real LLM claims pipeline integration test
//!
//! Requires LLM_API_KEY to be set in the environment.
//! Run with: cargo test -p agent-db-graph --test claims_integration -- --ignored

use agent_db_events::core::{EventContext, EventType};
use agent_db_events::Event;
use agent_db_graph::claims::EmbeddingClient;
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use tempfile::tempdir;

/// Build a Context event with rich, factual text that the LLM can extract claims from.
fn build_context_event(text: &str) -> Event {
    Event {
        id: 1001,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        agent_id: 42,
        agent_type: "test-agent".to_string(),
        session_id: 100,
        event_type: EventType::Context {
            text: text.to_string(),
            context_type: "document".to_string(),
            language: Some("en".to_string()),
        },
        causality_chain: Vec::new(),
        context: EventContext::default(),
        metadata: Default::default(),
        context_size_bytes: text.len(),
        segment_pointer: None,
    }
}

/// End-to-end test: send a Context event through the full claims pipeline
/// with a real OpenAI API key and verify claims are extracted and stored.
#[tokio::test]
#[ignore = "requires LLM_API_KEY — run with: cargo test --ignored"]
async fn test_real_claims_pipeline_end_to_end() {
    let api_key = match std::env::var("LLM_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            eprintln!("Skipping: LLM_API_KEY not set");
            return;
        },
    };

    let dir = tempdir().unwrap();

    let config = GraphEngineConfig {
        enable_semantic_memory: true,
        openai_api_key: Some(api_key),
        llm_model: "gpt-4o-mini".to_string(),
        claim_workers: 1,
        embedding_workers: 1,
        ner_workers: 0, // No NER service in CI — claims still work without it
        claim_storage_path: Some(dir.path().join("claims.redb")),
        ner_storage_path: Some(dir.path().join("ner.redb")),
        redb_path: dir.path().join("graph.redb"),
        claim_min_confidence: 0.5, // Lower threshold so we're more likely to get claims
        claim_max_per_input: 5,
        enable_embedding_generation: true,
        ..Default::default()
    };

    let engine = GraphEngine::with_config(config).await.unwrap();

    // Rich, factual text that should produce multiple atomic claims
    let text = "\
        Google was founded by Larry Page and Sergey Brin in September 1998 while they were \
        PhD students at Stanford University in California. The company's initial public offering \
        took place on August 19, 2004, at a price of $85 per share. By 2024, Alphabet Inc., \
        Google's parent company, had more than 180,000 employees worldwide. The company's \
        headquarters, known as the Googleplex, is located in Mountain View, California.";

    let event = build_context_event(text);

    // Process with semantic = true → triggers claim extraction
    let result = engine
        .process_event_with_options(event, Some(true))
        .await
        .unwrap();

    assert!(
        result.errors.is_empty(),
        "Event processing produced errors: {:?}",
        result.errors
    );

    // The claim extraction runs in a background tokio::spawn.
    // Give it time to complete (LLM call + embedding generation).
    tokio::time::sleep(std::time::Duration::from_secs(15)).await;

    // Verify claims were stored in the ClaimStore (redb)
    let claim_store = engine
        .claim_store()
        .expect("Claim store should be initialized");
    let claim_count = claim_store.count().unwrap();

    println!("=== Claims pipeline produced {} claims ===", claim_count);

    // We expect at least 1 claim from this rich factual text
    assert!(
        claim_count >= 1,
        "Expected at least 1 claim from factual text, got {}",
        claim_count
    );

    // Retrieve claims and validate structure
    let claims = claim_store.get_all_active(10).unwrap();
    for claim in &claims {
        println!(
            "  Claim {}: \"{}\" (confidence: {:.2})",
            claim.id, claim.claim_text, claim.confidence
        );
        assert!(
            !claim.claim_text.is_empty(),
            "Claim text should not be empty"
        );
        assert!(
            claim.confidence > 0.0 && claim.confidence <= 1.0,
            "Confidence {} should be in (0, 1]",
            claim.confidence
        );
        assert!(
            !claim.supporting_evidence.is_empty(),
            "Claim {} should have at least one evidence span",
            claim.id
        );
        // Verify evidence span has valid offsets
        for span in &claim.supporting_evidence {
            assert!(
                span.start_offset < span.end_offset,
                "Evidence span should have start < end"
            );
            assert!(
                !span.text_snippet.is_empty(),
                "Evidence text snippet should not be empty"
            );
        }
    }

    // Verify graph grew (claims add nodes + edges to the graph)
    let stats = engine.get_graph_stats().await;
    println!(
        "  Graph: {} nodes, {} edges",
        stats.node_count, stats.edge_count
    );
    assert!(
        stats.node_count > 0,
        "Graph should have nodes after claim integration"
    );

    println!("=== Claims pipeline integration test PASSED ===");
}

/// Test that embedding generation works end-to-end with real OpenAI API.
#[tokio::test]
#[ignore = "requires LLM_API_KEY — run with: cargo test --ignored"]
async fn test_real_embedding_generation() {
    let api_key = match std::env::var("LLM_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            eprintln!("Skipping: LLM_API_KEY not set");
            return;
        },
    };

    // Test the embedding client directly
    let client = agent_db_graph::claims::OpenAiEmbeddingClient::new(
        api_key,
        "text-embedding-3-small".to_string(),
    );

    let response = client
        .embed(agent_db_graph::claims::EmbeddingRequest {
            text: "Google was founded by Larry Page and Sergey Brin".to_string(),
            context: None,
        })
        .await
        .unwrap();

    assert_eq!(response.embedding.len(), 1536); // text-embedding-3-small = 1536 dims
    assert!(response.tokens_used > 0, "Should report token usage");

    // Verify the embedding is normalized (unit vector)
    let magnitude: f32 = response.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.05,
        "Embedding should be approximately unit length, got {}",
        magnitude
    );

    println!(
        "=== Embedding generation test PASSED (dims={}, tokens={}) ===",
        response.embedding.len(),
        response.tokens_used
    );
}
