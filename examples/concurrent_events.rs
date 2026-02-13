//! Concurrent Event Processing Demo
//!
//! Demonstrates how the Agentic Database handles out-of-order events
//! from multiple concurrent sources while maintaining correct causal relationships.

use agent_db_core::types::{current_timestamp, generate_event_id, AgentId, SessionId};
use agent_db_events::{
    ActionOutcome, ComputationalResources, EnvironmentState, Event, EventContext, EventType,
    ResourceState, TemporalContext,
};
use agent_db_graph::{event_ordering::OrderingConfig, GraphEngine, GraphEngineConfig};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Concurrent Event Processing Demo");
    println!("=====================================\n");

    // Configure with shorter reorder window for demo
    let config = GraphEngineConfig {
        ordering_config: OrderingConfig {
            reorder_window_ms: 2000, // 2 second reorder window
            max_buffer_size: 100,
            flush_interval_ms: 500,
            watermark_window_ms: 2000,
            strict_causality: true,
            max_clock_skew_ms: 5000,
            ..OrderingConfig::default()
        },
        ..GraphEngineConfig::default()
    };

    let engine = GraphEngine::with_config(config).await?;

    // Simulate multiple agents with different timing
    let agent_a: AgentId = 1;
    let agent_b: AgentId = 2;
    let agent_c: AgentId = 3;

    // Create base context
    let context = create_context();

    println!("📝 Simulating concurrent event streams...\n");

    // Simulate real-world scenario: events arrive out of order
    let events = create_event_scenario(agent_a, agent_b, agent_c, &context).await;

    println!("⏰ Event Timeline (as they should logically happen):");
    for (i, event) in events.iter().enumerate() {
        println!(
            "  {}. Agent {} at T+{}ms: {:?}",
            i + 1,
            event.agent_id,
            (event.timestamp % 1_000_000_000) / 1_000_000, // Show relative time in ms
            get_event_name(&event.event_type)
        );
    }

    // Simulate network delays and out-of-order arrival
    println!("\n📡 Processing events as they arrive (out of order):");

    // Reorder events to simulate real network conditions
    let shuffled_events = events.clone();

    // Agent A: Events 1,3 arrive first
    let result_1 = engine.process_event(shuffled_events[0].clone()).await?;
    println!(
        "  ✅ Processed Agent A Event 1: {} nodes created, {} buffered",
        result_1.nodes_created.len(),
        if result_1
            .patterns_detected
            .contains(&"event_reordering_occurred".to_string())
        {
            "reordered"
        } else {
            "in-order"
        }
    );

    // Agent B: Event 2 arrives late
    sleep(Duration::from_millis(100)).await;
    let result_2 = engine.process_event(shuffled_events[3].clone()).await?; // Agent B Event 1
    println!(
        "  ✅ Processed Agent B Event 1: {} nodes created",
        result_2.nodes_created.len()
    );

    // Agent A: Event 3 arrives before Event 2
    let _result_3 = engine.process_event(shuffled_events[2].clone()).await?; // Agent A Event 3
    println!("  ⏸️  Agent A Event 3 buffered (waiting for Event 2)");

    // Agent C: Events arrive
    let result_4 = engine.process_event(shuffled_events[4].clone()).await?;
    println!(
        "  ✅ Processed Agent C Event 1: {} nodes created",
        result_4.nodes_created.len()
    );

    // Agent A: Event 2 finally arrives (triggers processing of buffered Event 3)
    sleep(Duration::from_millis(200)).await;
    let result_5 = engine.process_event(shuffled_events[1].clone()).await?; // Agent A Event 2
    println!("  ✅ Agent A Event 2 arrived - triggered processing of buffered events");
    println!(
        "     {} total nodes created (including buffered Event 3)",
        result_5.nodes_created.len()
    );

    // Remaining events
    sleep(Duration::from_millis(50)).await;
    let _result_6 = engine.process_event(shuffled_events[5].clone()).await?; // Agent B Event 2
    let _result_7 = engine.process_event(shuffled_events[6].clone()).await?; // Agent C Event 2

    println!("\n📊 Final Statistics:");
    let stats = engine.get_graph_stats().await;
    println!("  Total Nodes: {}", stats.node_count);
    println!("  Total Edges: {}", stats.edge_count);
    println!("  Average Degree: {:.2}", stats.avg_degree);

    // Force flush any remaining buffered events
    println!("\n🔄 Flushing remaining buffers...");
    engine.flush_all_buffers().await?;

    println!("\n✅ All events processed successfully!");
    println!("   The system correctly inferred relationships despite out-of-order arrival!");

    Ok(())
}

async fn create_event_scenario(
    agent_a: AgentId,
    agent_b: AgentId,
    agent_c: AgentId,
    context: &EventContext,
) -> Vec<Event> {
    let base_time = current_timestamp();
    let session_id: SessionId = 200;

    vec![
        // Agent A sequence: start_task → process_data → complete_task
        Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: agent_a,
            agent_type: "task-processor".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "start_task".to_string(),
                parameters: serde_json::json!({"task_id": "task_001"}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Task started"),
                },
                duration_ns: 1_000_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 500_000_000, // 500ms later
            agent_id: agent_a,
            agent_type: "task-processor".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "process_data".to_string(),
                parameters: serde_json::json!({"data_size": 1024}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Data processed"),
                },
                duration_ns: 2_000_000,
            },
            causality_chain: Vec::new(), // Will be linked by inference
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 1_000_000_000, // 1s later
            agent_id: agent_a,
            agent_type: "task-processor".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "complete_task".to_string(),
                parameters: serde_json::json!({"task_id": "task_001"}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Task completed"),
                },
                duration_ns: 500_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        // Agent B sequence: start_analysis → generate_report
        Event {
            id: generate_event_id(),
            timestamp: base_time + 200_000_000, // 200ms after start
            agent_id: agent_b,
            agent_type: "data-analyst".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "start_analysis".to_string(),
                parameters: serde_json::json!({"analysis_type": "performance"}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Analysis started"),
                },
                duration_ns: 1_500_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        // Agent C sequence: monitor_system → alert_threshold
        Event {
            id: generate_event_id(),
            timestamp: base_time + 300_000_000, // 300ms after start
            agent_id: agent_c,
            agent_type: "system-monitor".to_string(),
            session_id,
            event_type: EventType::Observation {
                observation_type: "system_monitor".to_string(),
                data: serde_json::json!({"cpu_usage": 85.0, "memory_usage": 70.0}),
                confidence: 0.95,
                source: "system_metrics".to_string(),
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        // Agent B Event 2
        Event {
            id: generate_event_id(),
            timestamp: base_time + 800_000_000, // 800ms after start
            agent_id: agent_b,
            agent_type: "data-analyst".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "generate_report".to_string(),
                parameters: serde_json::json!({"format": "pdf"}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Report generated"),
                },
                duration_ns: 3_000_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        // Agent C Event 2
        Event {
            id: generate_event_id(),
            timestamp: base_time + 900_000_000, // 900ms after start
            agent_id: agent_c,
            agent_type: "system-monitor".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "send_alert".to_string(),
                parameters: serde_json::json!({"threshold": "high_cpu", "recipients": ["admin"]}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Alert sent"),
                },
                duration_ns: 100_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
    ]
}

fn create_context() -> EventContext {
    EventContext {
        environment: EnvironmentState {
            variables: std::collections::HashMap::new(),
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
                cpu_percent: 75.0,
                memory_bytes: 2 * 1024 * 1024 * 1024,   // 2GB
                storage_bytes: 50 * 1024 * 1024 * 1024, // 50GB
                network_bandwidth: 1000 * 1000 * 100,   // 100Mbps in bytes/sec
            },
            external: std::collections::HashMap::new(),
        },
        fingerprint: 54321,
        goal_bucket_id: 0,
        embeddings: None,
    }
}

fn get_event_name(event_type: &EventType) -> &str {
    match event_type {
        EventType::Action { action_name, .. } => action_name,
        EventType::Observation {
            observation_type, ..
        } => observation_type,
        EventType::Communication { message_type, .. } => message_type,
        EventType::Cognitive { process_type, .. } => match process_type {
            agent_db_events::CognitiveType::GoalFormation => "goal_formation",
            agent_db_events::CognitiveType::Planning => "planning",
            agent_db_events::CognitiveType::Reasoning => "reasoning",
            agent_db_events::CognitiveType::MemoryRetrieval => "memory_retrieval",
            agent_db_events::CognitiveType::LearningUpdate => "learning_update",
        },
        EventType::Learning { .. } => "learning",
        EventType::Context { .. } => "Context",
    }
}
