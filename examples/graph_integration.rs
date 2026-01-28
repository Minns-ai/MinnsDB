//! Graph Integration Example
//!
//! Demonstrates the complete graph system in action:
//! - Event processing and node creation
//! - Automatic relationship inference
//! - Graph queries and traversal
//! - Pattern detection

use agent_db_core::types::{current_timestamp, generate_event_id, AgentId, SessionId};
use agent_db_events::{
    ActionOutcome, ComputationalResources, EnvironmentState, Event, EventContext, EventType,
    ResourceState, TemporalContext,
};
use agent_db_graph::{GraphEngine, GraphEngineConfig, GraphQuery, QueryResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Agentic Database - Graph Integration Demo");
    println!("============================================\n");

    // Initialize graph engine
    let config = GraphEngineConfig::default();
    let engine = GraphEngine::with_config(config).await?;

    // Create sample events to process
    let agent1_id: AgentId = 1;
    let agent2_id: AgentId = 2;
    let session_id: SessionId = 100;

    // Create a simple context for all events
    let context = EventContext {
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
                cpu_percent: 20.0,
                memory_bytes: 1024 * 1024 * 1024,       // 1GB
                storage_bytes: 10 * 1024 * 1024 * 1024, // 10GB
                network_bandwidth: 1000 * 1000,         // 1Mbps in bytes/sec
            },
            external: std::collections::HashMap::new(),
        },
        fingerprint: 12345,
        embeddings: None,
    };

    // Event 1: Agent 1 performs communication
    let event1 = Event {
        id: generate_event_id(),
        timestamp: current_timestamp(),
        agent_id: agent1_id,
        agent_type: "chat-assistant".to_string(),
        session_id,
        event_type: EventType::Communication {
            message_type: "greeting".to_string(),
            sender: agent1_id,
            recipient: agent2_id,
            content: serde_json::json!("Hello, how are you?"),
        },
        causality_chain: Vec::new(),
        context: context.clone(),
        metadata: std::collections::HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    };

    // Event 2: Agent 2 responds (with slight delay)
    let event2 = Event {
        id: generate_event_id(),
        timestamp: current_timestamp() + 1_000_000_000, // 1 second later
        agent_id: agent2_id,
        agent_type: "chat-assistant".to_string(),
        session_id,
        event_type: EventType::Communication {
            message_type: "response".to_string(),
            sender: agent2_id,
            recipient: agent1_id,
            content: serde_json::json!("I'm doing well, thank you!"),
        },
        causality_chain: vec![event1.id],
        context: context.clone(),
        metadata: std::collections::HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    };

    // Event 3: Agent 1 performs an action
    let event3 = Event {
        id: generate_event_id(),
        timestamp: current_timestamp() + 2_000_000_000, // 2 seconds later
        agent_id: agent1_id,
        agent_type: "file-assistant".to_string(),
        session_id,
        event_type: EventType::Action {
            action_name: "file_access".to_string(),
            parameters: serde_json::json!({"filename": "document.txt", "mode": "read"}),
            outcome: ActionOutcome::Success {
                result: serde_json::json!("File read successfully"),
            },
            duration_ns: 1_000_000, // 1ms
        },
        causality_chain: vec![event2.id],
        context: context.clone(),
        metadata: std::collections::HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    };

    println!("📝 Processing events through graph engine...");

    // Process events one by one to see inference in action
    let result1 = engine.process_event(event1).await?;
    println!(
        "   Event 1 processed: {} nodes created",
        result1.nodes_created.len()
    );

    let result2 = engine.process_event(event2).await?;
    println!(
        "   Event 2 processed: {} nodes created, {} relationships discovered",
        result2.nodes_created.len(),
        result2.relationships_discovered
    );

    let result3 = engine.process_event(event3).await?;
    println!(
        "   Event 3 processed: {} nodes created, {} relationships discovered",
        result3.nodes_created.len(),
        result3.relationships_discovered
    );

    // Get graph statistics
    let stats = engine.get_graph_stats().await;
    println!("\n📊 Graph Statistics:");
    println!("   Nodes: {}", stats.node_count);
    println!("   Edges: {}", stats.edge_count);
    println!("   Average degree: {:.2}", stats.avg_degree);
    println!("   Largest component: {}", stats.largest_component_size);

    // Get engine statistics
    let engine_stats = engine.get_engine_stats().await;
    println!("\n⚙️  Engine Statistics:");
    println!(
        "   Events processed: {}",
        engine_stats.total_events_processed
    );
    println!(
        "   Relationships created: {}",
        engine_stats.total_relationships_created
    );
    println!(
        "   Average processing time: {:.2}ms",
        engine_stats.average_processing_time_ms
    );

    // Demonstrate query capabilities
    println!("\n🔍 Executing graph queries...");

    // Query for agent nodes
    let agent_query = GraphQuery::NodesByType("Agent".to_string());
    let agent_results = engine.execute_query(agent_query).await?;
    match agent_results {
        QueryResult::Nodes(nodes) => {
            println!("   Found {} agent nodes", nodes.len());
        },
        _ => println!("   Unexpected query result format"),
    }

    // Query for event nodes
    let event_query = GraphQuery::NodesByType("Event".to_string());
    let event_results = engine.execute_query(event_query).await?;
    match event_results {
        QueryResult::Nodes(nodes) => {
            println!("   Found {} event nodes", nodes.len());
        },
        _ => println!("   Unexpected query result format"),
    }

    // Query for context nodes
    let context_query = GraphQuery::NodesByType("Context".to_string());
    let context_results = engine.execute_query(context_query).await?;
    match context_results {
        QueryResult::Nodes(nodes) => {
            println!("   Found {} context nodes", nodes.len());
        },
        _ => println!("   Unexpected query result format"),
    }

    // Detect patterns
    println!("\n🔮 Detecting patterns...");
    let patterns = engine.detect_patterns().await?;
    println!("   Detected {} patterns:", patterns.len());
    for pattern in &patterns {
        println!("     - {}", pattern);
    }

    // Health metrics
    let health = engine.get_health_metrics().await;
    println!("\n🏥 Health Metrics:");
    println!(
        "   Processing rate: {:.2} events/sec",
        health.processing_rate
    );
    println!("   Memory estimate: {} bytes", health.memory_usage_estimate);
    println!(
        "   System health: {}",
        if health.is_healthy {
            "✅ Healthy"
        } else {
            "⚠️  Needs attention"
        }
    );

    println!("\n✅ Graph integration demo completed successfully!");
    Ok(())
}
