//! End-to-End Agentic Database Demo
//!
//! Comprehensive demonstration of the complete Agentic Database system:
//! - Event creation and processing
//! - Storage layer with persistence
//! - Graph operations and relationship inference  
//! - Scoped inference with agent types
//! - Concurrent event handling
//! - Pattern detection and analytics

use agent_db_core::types::{AgentId, SessionId, AgentType, generate_event_id, current_timestamp};
use agent_db_events::{Event, EventContext, EventType, ActionOutcome, CognitiveType, EnvironmentState, TemporalContext, ResourceState, ComputationalResources, Goal, TimeOfDay};
use agent_db_storage::{StorageEngine, StorageConfig, CompressionType};
use agent_db_graph::{
    GraphEngine, GraphEngineConfig, 
    event_ordering::OrderingConfig,
    scoped_inference::{ScopedInferenceConfig, InferenceScope},
    GraphQuery, QueryResult
};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Agentic Database - End-to-End System Demo");
    println!("===============================================\n");

    // Initialize storage engine
    println!("📁 Initializing storage layer...");
    let storage_config = StorageConfig {
        data_dir: "./demo_data".to_string(),
        compression: CompressionType::Lz4,
        max_file_size_mb: 100,
        enable_checksums: true,
        sync_interval_secs: 30,
    };
    let storage = StorageEngine::new(storage_config).await?;
    println!("   ✅ Storage engine initialized");

    // Initialize graph engine with comprehensive config
    println!("🧠 Initializing graph engine...");
    let graph_config = GraphEngineConfig {
        ordering_config: OrderingConfig {
            reorder_window_ms: 3000,
            max_buffer_size: 200,
            flush_interval_ms: 1000,
            strict_causality: true,
            max_clock_skew_ms: 10000,
        },
        scoped_inference_config: ScopedInferenceConfig {
            enable_cross_scope_inference: true,
            scope_similarity_threshold: 0.7,
            max_cross_scope_edges: 50,
            temporal_scope_window_ms: 5000,
        },
        ..GraphEngineConfig::default()
    };
    let graph = GraphEngine::with_config(graph_config).await?;
    println!("   ✅ Graph engine initialized with scoped inference");

    println!("\n🎭 Setting up multi-agent scenario...");
    
    // Define agent types and sessions
    let coding_assistant: AgentType = "coding-assistant".to_string();
    let data_analyst: AgentType = "data-analyst".to_string();
    let system_monitor: AgentType = "system-monitor".to_string();
    let task_manager: AgentType = "task-manager".to_string();
    
    let session_a: SessionId = 1001; // Development session
    let session_b: SessionId = 1002; // Analysis session
    let session_c: SessionId = 1003; // Monitoring session

    // Create realistic contexts for different scenarios
    let dev_context = create_development_context();
    let analysis_context = create_analysis_context();
    let monitoring_context = create_monitoring_context();

    println!("   👥 4 agent types across 3 sessions configured");

    // Phase 1: Development Session Events
    println!("\n📊 Phase 1: Development Session Events");
    let base_time = current_timestamp();
    
    let dev_events = create_development_events(
        coding_assistant.clone(), 
        session_a, 
        &dev_context, 
        base_time
    );

    for (i, event) in dev_events.iter().enumerate() {
        let result = graph.process_event(event.clone()).await?;
        storage.store_event(event.clone()).await?;
        
        println!("   Event {}: {} ({}ms) - {} nodes, {} relationships",
                 i + 1,
                 get_event_description(&event.event_type),
                 (event.timestamp - base_time) / 1_000_000,
                 result.nodes_created.len(),
                 result.relationships_discovered);
        
        sleep(Duration::from_millis(100)).await; // Simulate processing delay
    }

    // Phase 2: Analysis Session Events (concurrent with some development)
    println!("\n📈 Phase 2: Analysis Session Events");
    
    let analysis_events = create_analysis_events(
        data_analyst.clone(),
        session_b,
        &analysis_context,
        base_time + 1_000_000_000 // Start 1 second after dev session
    );

    for (i, event) in analysis_events.iter().enumerate() {
        let result = graph.process_event(event.clone()).await?;
        storage.store_event(event.clone()).await?;
        
        println!("   Event {}: {} ({}ms) - {} nodes, {} relationships",
                 i + 1,
                 get_event_description(&event.event_type),
                 (event.timestamp - base_time) / 1_000_000,
                 result.nodes_created.len(),
                 result.relationships_discovered);
    }

    // Phase 3: System Monitoring Events (concurrent)
    println!("\n📡 Phase 3: System Monitoring Events");
    
    let monitoring_events = create_monitoring_events(
        system_monitor.clone(),
        session_c,
        &monitoring_context,
        base_time + 2_000_000_000 // Start 2 seconds after dev session
    );

    for (i, event) in monitoring_events.iter().enumerate() {
        let result = graph.process_event(event.clone()).await?;
        storage.store_event(event.clone()).await?;
        
        println!("   Event {}: {} ({}ms) - {} nodes, {} relationships",
                 i + 1,
                 get_event_description(&event.event_type),
                 (event.timestamp - base_time) / 1_000_000,
                 result.nodes_created.len(),
                 result.relationships_discovered);
    }

    // Phase 4: Task Management Events (cross-session coordination)
    println!("\n🎯 Phase 4: Task Management Events (cross-session)");
    
    let task_events = create_task_management_events(
        task_manager.clone(),
        session_a, // Same session as development for coordination
        &dev_context,
        base_time + 3_000_000_000 // Start 3 seconds after dev session
    );

    for (i, event) in task_events.iter().enumerate() {
        let result = graph.process_event(event.clone()).await?;
        storage.store_event(event.clone()).await?;
        
        println!("   Event {}: {} ({}ms) - {} nodes, {} relationships",
                 i + 1,
                 get_event_description(&event.event_type),
                 (event.timestamp - base_time) / 1_000_000,
                 result.nodes_created.len(),
                 result.relationships_discovered);
    }

    // Flush any remaining buffered events
    println!("\n🔄 Flushing event buffers...");
    graph.flush_all_buffers().await?;

    // Storage analytics
    println!("\n💾 Storage Analytics:");
    let storage_stats = storage.get_storage_stats().await;
    println!("   Total events stored: {}", storage_stats.total_events);
    println!("   Storage size: {:.2} MB", storage_stats.total_size_bytes as f64 / (1024.0 * 1024.0));
    println!("   Compression ratio: {:.2}x", storage_stats.compression_ratio);

    // Graph analytics
    println!("\n🕸️  Graph Analytics:");
    let graph_stats = graph.get_graph_stats().await;
    println!("   Total nodes: {}", graph_stats.node_count);
    println!("   Total edges: {}", graph_stats.edge_count);
    println!("   Average degree: {:.2}", graph_stats.avg_degree);
    println!("   Largest component: {}", graph_stats.largest_component_size);

    // Engine performance metrics
    let engine_stats = graph.get_engine_stats().await;
    println!("\n⚡ Engine Performance:");
    println!("   Events processed: {}", engine_stats.total_events_processed);
    println!("   Relationships discovered: {}", engine_stats.total_relationships_created);
    println!("   Avg processing time: {:.2}ms", engine_stats.average_processing_time_ms);

    // Scoped inference statistics
    println!("\n🎯 Scoped Inference Results:");
    let scope_stats = graph.get_scoped_inference_stats().await;
    println!("   Active scopes: {}", scope_stats.scope_count);
    println!("   Cross-scope relationships: {}", scope_stats.cross_scope_edges);
    println!("   Most active scope: {:?}", scope_stats.most_active_scope);

    // Pattern detection
    println!("\n🔮 Pattern Detection:");
    let patterns = graph.detect_patterns().await?;
    println!("   Detected {} patterns:", patterns.len());
    for (i, pattern) in patterns.iter().enumerate().take(5) {
        println!("     {}. {}", i + 1, pattern);
    }

    // Demonstrate queries across different scopes
    println!("\n🔍 Cross-Scope Query Examples:");
    
    // Query for coding events
    let coding_scope = InferenceScope {
        agent_type: coding_assistant.clone(),
        session_id: session_a,
    };
    let coding_events = graph.query_events_in_scope(&coding_scope).await?;
    println!("   Coding events in session {}: {}", session_a, coding_events.len());

    // Query for analysis events
    let analysis_scope = InferenceScope {
        agent_type: data_analyst.clone(),
        session_id: session_b,
    };
    let analysis_events = graph.query_events_in_scope(&analysis_scope).await?;
    println!("   Analysis events in session {}: {}", session_b, analysis_events.len());

    // Cross-scope relationship analysis
    let cross_relationships = graph.get_cross_scope_relationships().await?;
    println!("   Cross-scope relationships found: {}", cross_relationships.len());

    // Health check
    println!("\n🏥 System Health Check:");
    let health = graph.get_health_metrics().await;
    println!("   Processing rate: {:.2} events/sec", health.processing_rate);
    println!("   Memory usage: {:.2} MB", health.memory_usage_estimate as f64 / (1024.0 * 1024.0));
    println!("   System status: {}", if health.is_healthy { "✅ Healthy" } else { "⚠️  Needs attention" });

    // Final synchronization
    println!("\n💫 Synchronizing storage...");
    storage.sync().await?;
    
    println!("\n🎉 End-to-End Demo Completed Successfully!");
    println!("   The Agentic Database successfully processed events from multiple");
    println!("   agent types across different sessions, automatically inferring");
    println!("   relationships while maintaining proper scope isolation.");

    Ok(())
}

fn create_development_context() -> EventContext {
    EventContext {
        environment: EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("project".to_string(), json!("rust-agentic-db"));
                vars.insert("branch".to_string(), json!("feature/graph-inference"));
                vars.insert("ide".to_string(), json!("vscode"));
                vars
            },
            spatial: None,
            temporal: TemporalContext {
                time_of_day: Some(TimeOfDay {
                    hour: 14,
                    minute: 30,
                    timezone: "UTC".to_string(),
                }),
                deadlines: Vec::new(),
                patterns: Vec::new(),
            },
        },
        active_goals: vec![
            Goal {
                id: 1,
                description: "Implement graph inference engine".to_string(),
                priority: 0.9,
                deadline: None,
                progress: 0.7,
                subgoals: vec![2, 3],
            },
            Goal {
                id: 2,
                description: "Add scoped inference".to_string(),
                priority: 0.8,
                deadline: None,
                progress: 0.5,
                subgoals: Vec::new(),
            },
        ],
        resources: ResourceState {
            computational: ComputationalResources {
                cpu_percent: 45.0,
                memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB
                storage_bytes: 100 * 1024 * 1024 * 1024, // 100GB
                network_bandwidth: 1000 * 1000 * 10, // 10Mbps
            },
            external: HashMap::new(),
        },
        fingerprint: 11111,
        embeddings: None,
    }
}

fn create_analysis_context() -> EventContext {
    EventContext {
        environment: EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("dataset".to_string(), json!("user_events_q4"));
                vars.insert("analysis_type".to_string(), json!("performance"));
                vars.insert("tool".to_string(), json!("pandas"));
                vars
            },
            spatial: None,
            temporal: TemporalContext {
                time_of_day: Some(TimeOfDay {
                    hour: 15,
                    minute: 45,
                    timezone: "UTC".to_string(),
                }),
                deadlines: Vec::new(),
                patterns: Vec::new(),
            },
        },
        active_goals: vec![
            Goal {
                id: 10,
                description: "Analyze system performance patterns".to_string(),
                priority: 0.7,
                deadline: None,
                progress: 0.3,
                subgoals: Vec::new(),
            },
        ],
        resources: ResourceState {
            computational: ComputationalResources {
                cpu_percent: 75.0,
                memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                storage_bytes: 500 * 1024 * 1024 * 1024, // 500GB
                network_bandwidth: 1000 * 1000 * 100, // 100Mbps
            },
            external: HashMap::new(),
        },
        fingerprint: 22222,
        embeddings: None,
    }
}

fn create_monitoring_context() -> EventContext {
    EventContext {
        environment: EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("service".to_string(), json!("agentic-db-cluster"));
                vars.insert("region".to_string(), json!("us-west-2"));
                vars.insert("monitoring_tool".to_string(), json!("prometheus"));
                vars
            },
            spatial: None,
            temporal: TemporalContext {
                time_of_day: Some(TimeOfDay {
                    hour: 16,
                    minute: 0,
                    timezone: "UTC".to_string(),
                }),
                deadlines: Vec::new(),
                patterns: Vec::new(),
            },
        },
        active_goals: vec![
            Goal {
                id: 20,
                description: "Monitor system health".to_string(),
                priority: 0.95,
                deadline: None,
                progress: 1.0,
                subgoals: Vec::new(),
            },
        ],
        resources: ResourceState {
            computational: ComputationalResources {
                cpu_percent: 25.0,
                memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
                storage_bytes: 50 * 1024 * 1024 * 1024, // 50GB
                network_bandwidth: 1000 * 1000 * 50, // 50Mbps
            },
            external: HashMap::new(),
        },
        fingerprint: 33333,
        embeddings: None,
    }
}

fn create_development_events(agent_type: AgentType, session_id: SessionId, context: &EventContext, base_time: u64) -> Vec<Event> {
    vec![
        Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: 1001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Cognitive {
                process_type: CognitiveType::Planning,
                input: json!({"task": "implement graph inference"}),
                output: json!({"plan": "create structures, inference, integration modules"}),
                reasoning_trace: vec![
                    "Analyze requirements".to_string(),
                    "Design module structure".to_string(),
                    "Plan implementation phases".to_string(),
                ],
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 500_000_000,
            agent_id: 1001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Action {
                action_name: "create_module".to_string(),
                parameters: json!({"module": "graph/structures.rs", "lines": 350}),
                outcome: ActionOutcome::Success { result: json!("Module created successfully") },
                duration_ns: 45_000_000_000, // 45 seconds
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 1_200_000_000,
            agent_id: 1001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Action {
                action_name: "run_tests".to_string(),
                parameters: json!({"test_suite": "graph_structures", "coverage": true}),
                outcome: ActionOutcome::Success { result: json!({"tests_passed": 23, "coverage": 92.5}) },
                duration_ns: 8_000_000_000, // 8 seconds
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
    ]
}

fn create_analysis_events(agent_type: AgentType, session_id: SessionId, context: &EventContext, base_time: u64) -> Vec<Event> {
    vec![
        Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: 2001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Action {
                action_name: "load_dataset".to_string(),
                parameters: json!({"dataset": "user_events_q4.parquet", "rows": 1_250_000}),
                outcome: ActionOutcome::Success { result: json!("Dataset loaded") },
                duration_ns: 12_000_000_000, // 12 seconds
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 800_000_000,
            agent_id: 2001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                input: json!({"data_summary": {"unique_agents": 1250, "event_types": 8, "timespan": "90 days"}}),
                output: json!({"insights": ["Peak usage at 2-4 PM", "Graph inference correlates with complexity"]}),
                reasoning_trace: vec![
                    "Identify temporal patterns".to_string(),
                    "Correlate with system metrics".to_string(),
                    "Generate insights".to_string(),
                ],
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
    ]
}

fn create_monitoring_events(agent_type: AgentType, session_id: SessionId, context: &EventContext, base_time: u64) -> Vec<Event> {
    vec![
        Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: 3001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Observation {
                observation_type: "system_metrics".to_string(),
                data: json!({"cpu": 78.5, "memory": 65.2, "disk_io": 145.8, "network_rx": 25.4}),
                confidence: 0.99,
                source: "prometheus".to_string(),
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 300_000_000,
            agent_id: 3001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Action {
                action_name: "check_alert_thresholds".to_string(),
                parameters: json!({"thresholds": {"cpu": 80.0, "memory": 85.0, "disk_io": 200.0}}),
                outcome: ActionOutcome::Success { result: json!({"alerts": 0, "status": "normal"}) },
                duration_ns: 150_000_000, // 150ms
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
    ]
}

fn create_task_management_events(agent_type: AgentType, session_id: SessionId, context: &EventContext, base_time: u64) -> Vec<Event> {
    vec![
        Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: 4001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: json!({"project_status": "week_3_implementation", "completion": 0.85}),
                output: json!({"next_goals": ["performance_testing", "documentation", "integration_tests"]}),
                reasoning_trace: vec![
                    "Assess current progress".to_string(),
                    "Identify remaining tasks".to_string(),
                    "Prioritize by dependencies".to_string(),
                ],
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
        Event {
            id: generate_event_id(),
            timestamp: base_time + 600_000_000,
            agent_id: 4001,
            agent_type: agent_type.clone(),
            session_id,
            event_type: EventType::Communication {
                message_type: "task_assignment".to_string(),
                sender: 4001,
                recipient: 1001, // Send to coding assistant
                content: json!({
                    "task": "performance_optimization",
                    "priority": "high",
                    "deadline": "end_of_week",
                    "context": "graph_inference_bottlenecks"
                }),
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: HashMap::new(),
        },
    ]
}

fn get_event_description(event_type: &EventType) -> &str {
    match event_type {
        EventType::Action { action_name, .. } => action_name,
        EventType::Observation { observation_type, .. } => observation_type,
        EventType::Communication { message_type, .. } => message_type,
        EventType::Cognitive { process_type, .. } => match process_type {
            CognitiveType::Planning => "planning",
            CognitiveType::Reasoning => "reasoning",
            CognitiveType::GoalFormation => "goal_formation",
            CognitiveType::MemoryRetrieval => "memory_retrieval",
            CognitiveType::LearningUpdate => "learning_update",
        },
    }
}