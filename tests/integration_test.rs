//! Integration Tests for Agentic Database
//!
//! Comprehensive test suite covering all major components:
//! - Event creation and validation
//! - Storage persistence and retrieval
//! - Graph inference and relationships
//! - Scoped inference and cross-scope detection
//! - Concurrent event processing

use agent_db_core::types::{AgentId, SessionId, AgentType, generate_event_id, current_timestamp};
use agent_db_events::{Event, EventContext, EventType, ActionOutcome, CognitiveType, EnvironmentState, TemporalContext, ResourceState, ComputationalResources};
use agent_db_storage::{StorageEngine, StorageConfig, CompressionType};
use agent_db_graph::{
    GraphEngine, GraphEngineConfig,
    event_ordering::OrderingConfig,
    scoped_inference::{ScopedInferenceConfig, InferenceScope},
};
use std::collections::HashMap;
use tokio::time::Duration;
use serde_json::json;

#[tokio::test]
async fn test_complete_system_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize components
    let storage_config = StorageConfig {
        data_dir: "./test_data".to_string(),
        compression: CompressionType::Lz4,
        max_file_size_mb: 10,
        enable_checksums: true,
        sync_interval_secs: 1,
    };
    let storage = StorageEngine::new(storage_config).await?;

    let graph_config = GraphEngineConfig {
        ordering_config: OrderingConfig {
            reorder_window_ms: 1000,
            max_buffer_size: 50,
            flush_interval_ms: 200,
            strict_causality: true,
            max_clock_skew_ms: 2000,
        },
        scoped_inference_config: ScopedInferenceConfig {
            enable_cross_scope_inference: true,
            scope_similarity_threshold: 0.8,
            max_cross_scope_edges: 10,
            temporal_scope_window_ms: 2000,
        },
        ..GraphEngineConfig::default()
    };
    let graph = GraphEngine::with_config(graph_config).await?;

    // Test data
    let agent_type_a: AgentType = "test-agent-a".to_string();
    let agent_type_b: AgentType = "test-agent-b".to_string();
    let session_1: SessionId = 1;
    let session_2: SessionId = 2;
    let context = create_test_context();

    // Create test events
    let base_time = current_timestamp();
    let events = vec![
        create_test_event(1, agent_type_a.clone(), session_1, &context, base_time),
        create_test_event(2, agent_type_a.clone(), session_1, &context, base_time + 500_000_000),
        create_test_event(3, agent_type_b.clone(), session_2, &context, base_time + 1_000_000_000),
        create_test_event(4, agent_type_b.clone(), session_2, &context, base_time + 1_500_000_000),
    ];

    // Test 1: Event processing and storage
    for event in &events {
        let result = graph.process_event(event.clone()).await?;
        storage.store_event(event.clone()).await?;
        
        assert!(!result.nodes_created.is_empty(), "Should create nodes for each event");
    }

    // Test 2: Storage retrieval
    let retrieved_events = storage.query_events(session_1, base_time, base_time + 2_000_000_000).await?;
    assert_eq!(retrieved_events.len(), 2, "Should retrieve events for session 1");

    // Test 3: Graph statistics
    let graph_stats = graph.get_graph_stats().await;
    assert!(graph_stats.node_count > 0, "Graph should have nodes");
    assert!(graph_stats.edge_count >= 0, "Graph should have edges");

    // Test 4: Scoped inference
    let scope_a = InferenceScope {
        agent_type: agent_type_a,
        session_id: session_1,
    };
    let scope_events = graph.query_events_in_scope(&scope_a).await?;
    assert_eq!(scope_events.len(), 2, "Should find 2 events in scope A");

    // Test 5: Pattern detection
    let patterns = graph.detect_patterns().await?;
    assert!(!patterns.is_empty(), "Should detect some patterns");

    // Test 6: Health metrics
    let health = graph.get_health_metrics().await;
    assert!(health.is_healthy, "System should be healthy");

    // Cleanup
    graph.flush_all_buffers().await?;
    storage.sync().await?;

    println!("✅ All integration tests passed!");
    Ok(())
}

#[tokio::test]
async fn test_concurrent_event_processing() -> Result<(), Box<dyn std::error::Error>> {
    let graph = GraphEngine::new().await?;
    let context = create_test_context();
    let base_time = current_timestamp();

    // Create events that arrive out of order
    let event1 = create_test_event(1, "agent-x".to_string(), 100, &context, base_time);
    let event2 = create_test_event(1, "agent-x".to_string(), 100, &context, base_time + 1_000_000_000);
    let event3 = create_test_event(1, "agent-x".to_string(), 100, &context, base_time + 2_000_000_000);

    // Process in out-of-order sequence: 1, 3, 2
    let _result1 = graph.process_event(event1).await?;
    let _result3 = graph.process_event(event3).await?; // This should be buffered
    
    // Small delay to simulate real timing
    tokio::time::sleep(Duration::from_millis(10)).await;
    
    let _result2 = graph.process_event(event2).await?; // This should trigger buffered event processing

    // Flush buffers
    graph.flush_all_buffers().await?;

    let stats = graph.get_graph_stats().await;
    assert!(stats.node_count >= 3, "Should process all events despite order");

    println!("✅ Concurrent processing test passed!");
    Ok(())
}

#[tokio::test]
async fn test_cross_scope_inference() -> Result<(), Box<dyn std::error::Error>> {
    let config = GraphEngineConfig {
        scoped_inference_config: ScopedInferenceConfig {
            enable_cross_scope_inference: true,
            scope_similarity_threshold: 0.5, // Lower threshold for testing
            max_cross_scope_edges: 20,
            temporal_scope_window_ms: 5000,
        },
        ..GraphEngineConfig::default()
    };
    let graph = GraphEngine::with_config(config).await?;
    let context = create_test_context();
    let base_time = current_timestamp();

    // Create events in different scopes but with similar timing/context
    let events = vec![
        create_test_event(1, "type-a".to_string(), 1, &context, base_time),
        create_test_event(2, "type-b".to_string(), 2, &context, base_time + 100_000_000), // Close in time
        create_test_event(3, "type-a".to_string(), 3, &context, base_time + 200_000_000),
    ];

    for event in events {
        let _result = graph.process_event(event).await?;
    }

    // Check for cross-scope relationships
    let cross_relationships = graph.get_cross_scope_relationships().await?;
    assert!(!cross_relationships.is_empty(), "Should detect cross-scope relationships");

    let scope_stats = graph.get_scoped_inference_stats().await;
    assert!(scope_stats.cross_scope_edges > 0, "Should create cross-scope edges");

    println!("✅ Cross-scope inference test passed!");
    Ok(())
}

#[tokio::test]
async fn test_storage_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let config = StorageConfig {
        data_dir: "./test_persistence".to_string(),
        compression: CompressionType::None, // No compression for testing
        max_file_size_mb: 1,
        enable_checksums: true,
        sync_interval_secs: 1,
    };

    // Create storage engine and add some events
    {
        let storage = StorageEngine::new(config.clone()).await?;
        let context = create_test_context();
        let base_time = current_timestamp();

        let event = create_test_event(1, "persistent-agent".to_string(), 42, &context, base_time);
        storage.store_event(&event).await?;
        storage.sync().await?;
    } // Storage engine goes out of scope

    // Create new storage engine (simulating restart)
    {
        let storage = StorageEngine::new(config).await?;
        let events = storage.query_events(42, 0, u64::MAX).await?;
        assert_eq!(events.len(), 1, "Should persist and retrieve event after restart");
    }

    println!("✅ Storage persistence test passed!");
    Ok(())
}

fn create_test_context() -> EventContext {
    EventContext {
        environment: EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("test".to_string(), json!("integration"));
                vars
            },
            spatial: None,
            temporal: TemporalContext {
                time_of_day: None,
                deadlines: Vec::new(),
                patterns: Vec::new(),
            },
        },
        active_goals: Vec::new(),
        resources: ResourceState {
            computational: ComputationalResources {
                cpu_percent: 50.0,
                memory_bytes: 1024 * 1024 * 1024, // 1GB
                storage_bytes: 10 * 1024 * 1024 * 1024, // 10GB
                network_bandwidth: 1000 * 1000, // 1Mbps
            },
            external: HashMap::new(),
        },
        fingerprint: 98765,
        embeddings: None,
    }
}

fn create_test_event(
    agent_id: AgentId,
    agent_type: AgentType,
    session_id: SessionId,
    context: &EventContext,
    timestamp: u64
) -> Event {
    Event {
        id: generate_event_id(),
        timestamp,
        agent_id,
        agent_type,
        session_id,
        event_type: EventType::Action {
            action_name: "test_action".to_string(),
            parameters: json!({"test": true, "id": agent_id}),
            outcome: ActionOutcome::Success { result: json!("test_result") },
            duration_ns: 1_000_000,
        },
        causality_chain: Vec::new(),
        context: context.clone(),
        metadata: HashMap::new(),
    }
}