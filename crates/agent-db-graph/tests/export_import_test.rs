//! Export/Import Integration Test
//!
//! Creates a persistent GraphEngine, processes events to build up
//! memories, strategies, and graph state, then exports to binary v2,
//! imports into a fresh engine, and verifies all data survived.

use agent_db_core::types::{current_timestamp, generate_event_id, AgentId};
use agent_db_events::{
    ActionOutcome, CognitiveType, ComputationalResources, EnvironmentState, Event, EventContext,
    EventType, ResourceState, TemporalContext,
};
use agent_db_graph::integration::StorageBackend;
use agent_db_graph::{GraphEngine, GraphEngineConfig, ImportMode};
use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;

fn create_test_context(fingerprint: u64) -> EventContext {
    EventContext {
        environment: EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("task".to_string(), json!("debug_code"));
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
                memory_bytes: 1024 * 1024 * 1024,
                storage_bytes: 10 * 1024 * 1024 * 1024,
                network_bandwidth: 1000 * 1000,
            },
            external: HashMap::new(),
        },
        fingerprint,
        goal_bucket_id: 0,
        embeddings: None,
    }
}

fn create_cognitive_event(
    agent_id: AgentId,
    timestamp: u64,
    reasoning: Vec<String>,
    fingerprint: u64,
) -> Event {
    Event {
        id: generate_event_id(),
        timestamp,
        agent_id,
        agent_type: "ai-code-assistant".to_string(),
        session_id: 1,
        event_type: EventType::Cognitive {
            process_type: CognitiveType::Reasoning,
            input: json!({"error": "TypeError"}),
            output: json!({"fix": "add null check"}),
            reasoning_trace: reasoning,
        },
        causality_chain: Vec::new(),
        context: create_test_context(fingerprint),
        metadata: HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    }
}

fn create_action_event(
    agent_id: AgentId,
    timestamp: u64,
    success: bool,
    fingerprint: u64,
) -> Event {
    Event {
        id: generate_event_id(),
        timestamp,
        agent_id,
        agent_type: "ai-code-assistant".to_string(),
        session_id: 1,
        event_type: EventType::Action {
            action_name: "apply_fix".to_string(),
            parameters: json!({"fix_type": "null_check"}),
            outcome: if success {
                ActionOutcome::Success {
                    result: json!("fixed"),
                }
            } else {
                ActionOutcome::Failure {
                    error: "failed".to_string(),
                    error_code: 500,
                }
            },
            duration_ns: 1_000_000,
        },
        causality_chain: Vec::new(),
        context: create_test_context(fingerprint),
        metadata: HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    }
}

fn persistent_config(dir: &std::path::Path) -> GraphEngineConfig {
    GraphEngineConfig {
        storage_backend: StorageBackend::Persistent,
        redb_path: dir.join("graph.redb"),
        enable_semantic_memory: false,
        enable_louvain: false,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_export_import_full_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    // ================================================================
    // Step 1: Create source engine with persistent storage
    // ================================================================
    let source_dir = TempDir::new()?;
    let source_config = persistent_config(source_dir.path());
    let source_engine = GraphEngine::with_config(source_config).await?;

    // ================================================================
    // Step 2: Process events to build up data
    // ================================================================
    let base_time = current_timestamp();
    let fingerprint = 12345u64;

    // Process several events to generate graph nodes, episodes, memories, strategies
    let events = vec![
        create_cognitive_event(
            1,
            base_time,
            vec![
                "Analyze TypeError: undefined property".to_string(),
                "Identify null reference issue".to_string(),
                "Consider solutions: null check vs optional chaining".to_string(),
                "Select null check approach for clarity".to_string(),
            ],
            fingerprint,
        ),
        create_action_event(1, base_time + 500_000_000, true, fingerprint),
        create_cognitive_event(
            1,
            base_time + 3_000_000_000,
            vec![
                "Analyze similar TypeError".to_string(),
                "Recognize pattern from previous fix".to_string(),
                "Apply null check strategy".to_string(),
            ],
            fingerprint,
        ),
        create_action_event(1, base_time + 3_500_000_000, true, fingerprint),
    ];

    for event in &events {
        source_engine.process_event(event.clone()).await?;
    }

    // Collect source stats before export
    let source_stats = source_engine.get_engine_stats().await;
    let source_graph_stats = source_engine.get_graph_stats().await;
    let source_memories = source_engine.get_agent_memories(1, 1000).await;
    let source_strategies = source_engine.get_agent_strategies(1, 1000).await;

    println!("Source engine state:");
    println!(
        "  events processed:  {}",
        source_stats.total_events_processed
    );
    println!("  graph nodes:       {}", source_graph_stats.node_count);
    println!("  graph edges:       {}", source_graph_stats.edge_count);
    println!("  memories:          {}", source_memories.len());
    println!("  strategies:        {}", source_strategies.len());

    assert!(
        source_stats.total_events_processed >= 4,
        "Expected at least 4 events processed"
    );
    assert!(
        source_graph_stats.node_count > 0,
        "Expected graph nodes to be created"
    );

    // ================================================================
    // Step 3: Export to binary v2
    // ================================================================
    let mut export_buf = Vec::new();
    let record_count = source_engine.export(&mut export_buf).await?;
    println!(
        "\nExported {} records, {} bytes of binary",
        record_count,
        export_buf.len()
    );
    assert!(export_buf.len() > 0, "Expected non-empty export");

    // ================================================================
    // Step 4: Import into a fresh engine
    // ================================================================
    let target_dir = TempDir::new()?;
    let target_config = persistent_config(target_dir.path());
    let target_engine = GraphEngine::with_config(target_config).await?;

    let reader = std::io::Cursor::new(&export_buf);
    let import_stats = target_engine.import(reader, ImportMode::Replace).await?;

    println!("\nImport stats:");
    println!(
        "  memories imported:     {}",
        import_stats.memories_imported
    );
    println!(
        "  strategies imported:   {}",
        import_stats.strategies_imported
    );
    println!(
        "  graph nodes imported:  {}",
        import_stats.graph_nodes_imported
    );
    println!(
        "  graph edges imported:  {}",
        import_stats.graph_edges_imported
    );
    println!("  total records:         {}", import_stats.total_records);

    // ================================================================
    // Step 5: Verify data survived the roundtrip
    // ================================================================
    let target_graph_stats = target_engine.get_graph_stats().await;
    let target_memories = target_engine.get_agent_memories(1, 1000).await;
    let target_strategies = target_engine.get_agent_strategies(1, 1000).await;

    println!("\nTarget engine state after import:");
    println!("  graph nodes:       {}", target_graph_stats.node_count);
    println!("  graph edges:       {}", target_graph_stats.edge_count);
    println!("  memories:          {}", target_memories.len());
    println!("  strategies:        {}", target_strategies.len());

    // Graph structure must match
    assert_eq!(
        source_graph_stats.node_count, target_graph_stats.node_count,
        "Node count mismatch: source={} target={}",
        source_graph_stats.node_count, target_graph_stats.node_count
    );
    assert_eq!(
        source_graph_stats.edge_count, target_graph_stats.edge_count,
        "Edge count mismatch: source={} target={}",
        source_graph_stats.edge_count, target_graph_stats.edge_count
    );

    // Memory count must match
    assert_eq!(
        source_memories.len(),
        target_memories.len(),
        "Memory count mismatch: source={} target={}",
        source_memories.len(),
        target_memories.len()
    );

    // Strategy count must match
    assert_eq!(
        source_strategies.len(),
        target_strategies.len(),
        "Strategy count mismatch: source={} target={}",
        source_strategies.len(),
        target_strategies.len()
    );

    // Verify individual memory contents match
    for (src, tgt) in source_memories.iter().zip(target_memories.iter()) {
        assert_eq!(src.id, tgt.id, "Memory ID mismatch");
        assert_eq!(
            src.summary, tgt.summary,
            "Memory summary mismatch for id={}",
            src.id
        );
        assert_eq!(
            src.strength, tgt.strength,
            "Memory strength mismatch for id={}",
            src.id
        );
    }

    // Verify individual strategy contents match
    for (src, tgt) in source_strategies.iter().zip(target_strategies.iter()) {
        assert_eq!(src.id, tgt.id, "Strategy ID mismatch");
        assert_eq!(
            src.name, tgt.name,
            "Strategy name mismatch for id={}",
            src.id
        );
        assert_eq!(
            src.quality_score, tgt.quality_score,
            "Strategy quality mismatch for id={}",
            src.id
        );
    }

    println!("\nAll assertions passed - export/import roundtrip is correct.");

    Ok(())
}

/// Verify that new data can be written after an import without ID collisions.
#[tokio::test]
async fn test_import_then_write_no_id_collision() -> Result<(), Box<dyn std::error::Error>> {
    // Build source engine with data
    let source_dir = TempDir::new()?;
    let source_config = persistent_config(source_dir.path());
    let source_engine = GraphEngine::with_config(source_config).await?;

    let base_time = current_timestamp();
    let fingerprint = 99999u64;

    for i in 0..4 {
        let event = create_cognitive_event(
            1,
            base_time + (i * 1_000_000_000),
            vec![format!("Step {}", i)],
            fingerprint,
        );
        source_engine.process_event(event).await?;
    }

    // Export
    let mut export_buf = Vec::new();
    source_engine.export(&mut export_buf).await?;

    let source_memories = source_engine.get_agent_memories(1, 1000).await;
    let source_strategies = source_engine.get_agent_strategies(1, 1000).await;

    // Import into fresh engine
    let target_dir = TempDir::new()?;
    let target_config = persistent_config(target_dir.path());
    let target_engine = GraphEngine::with_config(target_config).await?;

    let reader = std::io::Cursor::new(&export_buf);
    target_engine.import(reader, ImportMode::Replace).await?;

    // Now process MORE events on the target engine
    let new_base = current_timestamp();
    for i in 0..4 {
        let event = create_action_event(1, new_base + (i * 1_000_000_000), true, fingerprint + 1);
        target_engine.process_event(event).await?;
    }

    // Verify: all original memories still exist, and new ones (if any) have higher IDs
    let target_memories = target_engine.get_agent_memories(1, 1000).await;
    let target_strategies = target_engine.get_agent_strategies(1, 1000).await;

    println!("After import + new events:");
    println!("  source memories:  {}", source_memories.len());
    println!("  target memories:  {}", target_memories.len());
    println!("  source strategies: {}", source_strategies.len());
    println!("  target strategies: {}", target_strategies.len());

    // Target must have at least as many as source (new events may create more)
    assert!(
        target_memories.len() >= source_memories.len(),
        "Target should have at least as many memories as source"
    );
    assert!(
        target_strategies.len() >= source_strategies.len(),
        "Target should have at least as many strategies as source"
    );

    // Verify no duplicate IDs (all memory IDs should be unique)
    let mut seen_mem_ids: Vec<u64> = target_memories.iter().map(|m| m.id).collect();
    seen_mem_ids.sort();
    seen_mem_ids.dedup();
    assert_eq!(
        seen_mem_ids.len(),
        target_memories.len(),
        "Duplicate memory IDs found after import + write"
    );

    // Verify no duplicate strategy IDs
    let mut seen_strat_ids: Vec<u64> = target_strategies.iter().map(|s| s.id).collect();
    seen_strat_ids.sort();
    seen_strat_ids.dedup();
    assert_eq!(
        seen_strat_ids.len(),
        target_strategies.len(),
        "Duplicate strategy IDs found after import + write"
    );

    println!("No ID collisions - import + continued operation is correct.");

    Ok(())
}

/// Verify export produces valid binary v2 structure.
#[tokio::test]
async fn test_export_is_valid_binary() -> Result<(), Box<dyn std::error::Error>> {
    let source_dir = TempDir::new()?;
    let source_config = persistent_config(source_dir.path());
    let engine = GraphEngine::with_config(source_config).await?;

    // Add some data
    let base_time = current_timestamp();
    engine
        .process_event(create_cognitive_event(
            1,
            base_time,
            vec!["test step".to_string()],
            55555,
        ))
        .await?;
    engine
        .process_event(create_action_event(1, base_time + 500_000_000, true, 55555))
        .await?;

    // Export
    let mut buf = Vec::new();
    engine.export(&mut buf).await?;

    // Validate header
    assert!(buf.len() >= 21, "Export too short for header");
    assert_eq!(&buf[0..4], b"EGDB", "Bad magic");
    assert_eq!(buf[4], 0x02, "Bad version");

    // Parse all records to verify structure
    let mut cursor = std::io::Cursor::new(&buf);
    agent_db_graph::wire_v2::read_header(&mut cursor)?;

    let known_tags: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    let mut record_count = 0u64;

    loop {
        let tag = agent_db_graph::wire_v2::read_record_tag(&mut cursor)?;
        if tag == 0xFF {
            let (footer_count, _checksum) = agent_db_graph::wire_v2::read_footer(&mut cursor)?;
            assert_eq!(
                footer_count, record_count,
                "Footer count {} doesn't match parsed records {}",
                footer_count, record_count
            );
            break;
        }
        assert!(
            known_tags.contains(&tag),
            "Unknown tag 0x{:02X} at record {}",
            tag,
            record_count
        );
        let (_key, _value) = agent_db_graph::wire_v2::read_record_body(&mut cursor)?;
        record_count += 1;
    }

    // Verify we consumed the entire stream
    assert_eq!(
        cursor.position() as usize,
        buf.len(),
        "Leftover bytes after footer"
    );

    println!(
        "Export binary is valid: {} bytes, {} records",
        buf.len(),
        record_count
    );

    Ok(())
}
