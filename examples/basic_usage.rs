//! Basic usage example for the Agentic Database
//! 
//! This example demonstrates:
//! - Event creation and ingestion
//! - Context tracking
//! - Basic memory formation
//! - Simple queries

use agent_db_core::*;
use agent_db_events::*;
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Agentic Database - Basic Usage Example");
    println!("=========================================");
    
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Create database configuration
    let mut config = DatabaseConfig::default();
    config.data_directory = std::path::PathBuf::from("./example_data");
    
    // Clean up any existing data
    if config.data_directory.exists() {
        std::fs::remove_dir_all(&config.data_directory)?;
    }
    std::fs::create_dir_all(&config.data_directory)?;
    
    println!("📁 Database directory: {:?}", config.data_directory);
    println!("⚙️  Configuration: {}", serde_json::to_string_pretty(&config)?);
    
    // TODO: Initialize database when implemented
    // let mut db = AgentDatabase::new(config).await?;
    // println!("✅ Database initialized successfully");
    
    // Simulate agent workflow
    simulate_learning_agent().await?;
    simulate_task_automation().await?;
    
    // Clean up
    std::fs::remove_dir_all(&config.data_directory).ok();
    
    println!("🎉 Example completed successfully!");
    Ok(())
}

/// Simulate a learning agent performing navigation tasks
async fn simulate_learning_agent() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🤖 Simulating Learning Agent");
    println!("----------------------------");
    
    let agent_id = 1;
    let session_id = 100;
    let agent_type = "warehouse-robot".to_string();
    
    // Create initial context (robot in warehouse)
    let initial_context = create_warehouse_context("warehouse_a", (0.0, 0.0), 85.0);
    
    println!("📍 Agent starting at position: {:?}", 
        initial_context.environment.spatial.as_ref().unwrap().location);
    
    // Simulate navigation sequence
    let navigation_events = vec![
        // Goal formation
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: json!({"task": "navigate_to_shelf", "target": "shelf_a5"}),
                output: json!({"goal_id": 1, "priority": 0.8}),
                reasoning_trace: vec![
                    "Received navigation request".to_string(),
                    "Target shelf_a5 identified".to_string(),
                    "Route planning initiated".to_string(),
                ],
            },
            initial_context.clone(),
        ),
        
        // Planning phase
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Cognitive {
                process_type: CognitiveType::Planning,
                input: json!({"goal": "navigate_to_shelf", "current_pos": [0.0, 0.0]}),
                output: json!({"plan": ["move_forward", "turn_right", "move_forward"]}),
                reasoning_trace: vec![
                    "Analyzing current position".to_string(),
                    "Computing optimal path".to_string(),
                    "Generated 3-step plan".to_string(),
                ],
            },
            initial_context.clone(),
        ),
        
        // Execute movements
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Action {
                action_name: "move_forward".to_string(),
                parameters: json!({"distance": 5.0, "speed": 1.0}),
                outcome: ActionOutcome::Success {
                    result: json!({"new_position": [5.0, 0.0], "time_taken": 5.2}),
                },
                duration_ns: 5_200_000_000,
            },
            create_warehouse_context("warehouse_a", (5.0, 0.0), 87.0),
        ),
        
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Action {
                action_name: "turn_right".to_string(),
                parameters: json!({"angle": 90.0}),
                outcome: ActionOutcome::Success {
                    result: json!({"new_heading": 90.0, "time_taken": 1.8}),
                },
                duration_ns: 1_800_000_000,
            },
            create_warehouse_context("warehouse_a", (5.0, 0.0), 88.0),
        ),
        
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Action {
                action_name: "move_forward".to_string(),
                parameters: json!({"distance": 3.0, "speed": 1.0}),
                outcome: ActionOutcome::Success {
                    result: json!({"new_position": [5.0, 3.0], "time_taken": 3.1}),
                },
                duration_ns: 3_100_000_000,
            },
            create_warehouse_context("warehouse_a", (5.0, 3.0), 89.0),
        ),
    ];
    
    // Process each event
    for (i, event) in navigation_events.iter().enumerate() {
        println!("📝 Event {}: {} (Agent {})", 
            i + 1, 
            match &event.event_type {
                EventType::Action { action_name, .. } => format!("Action: {}", action_name),
                EventType::Cognitive { process_type, .. } => format!("Cognitive: {:?}", process_type),
                _ => "Other".to_string(),
            },
            event.agent_id
        );
        
        // Simulate event processing
        let serialized = bincode::serialize(event)?;
        println!("   💾 Serialized size: {} bytes", serialized.len());
        
        // Simulate context analysis
        let context_fingerprint = event.context.fingerprint;
        println!("   🔍 Context fingerprint: {}", context_fingerprint);
        
        // TODO: Actual database ingestion when implemented
        // db.ingest_event(event.clone()).await?;
    }
    
    println!("✅ Navigation sequence completed successfully");
    
    // Demonstrate memory formation (conceptual)
    println!("\n🧠 Memory Formation:");
    println!("   📚 Episodic memory: 'Successful navigation from (0,0) to shelf_a5'");
    println!("   🎯 Pattern learned: 'move_forward → turn_right → move_forward' for shelf navigation");
    println!("   💡 Context association: 'warehouse_a' environment linked to navigation success");
    
    Ok(())
}

/// Simulate task automation workflow
async fn simulate_task_automation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Simulating Task Automation");
    println!("-----------------------------");
    
    let agent_id = 2;
    let session_id = 200;
    
    // Create office context
    let office_context = create_office_context("office_b", 72.0);
    
    // Simulate automated task processing
    let task_events = vec![
        // Observation: New email received
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Observation {
                observation_type: "email_received".to_string(),
                data: json!({
                    "from": "user@company.com",
                    "subject": "Monthly Report Request",
                    "priority": "high"
                }),
                confidence: 0.95,
                source: "email_system".to_string(),
            },
            office_context.clone(),
        ),
        
        // Cognitive: Process and categorize
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                input: json!({"email_subject": "Monthly Report Request"}),
                output: json!({"category": "report_generation", "priority": 0.8}),
                reasoning_trace: vec![
                    "Email content analyzed".to_string(),
                    "Keywords matched: 'monthly', 'report'".to_string(),
                    "Categorized as report generation task".to_string(),
                ],
            },
            office_context.clone(),
        ),
        
        // Action: Generate report
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Action {
                action_name: "generate_monthly_report".to_string(),
                parameters: json!({"month": "December", "year": 2023}),
                outcome: ActionOutcome::Success {
                    result: json!({
                        "report_path": "/reports/december_2023.pdf",
                        "pages": 15,
                        "charts": 8
                    }),
                },
                duration_ns: 45_000_000_000, // 45 seconds
            },
            office_context.clone(),
        ),
        
        // Action: Send email response
        Event::new(
            agent_id,
            agent_type.clone(),
            session_id,
            EventType::Action {
                action_name: "send_email".to_string(),
                parameters: json!({
                    "to": "user@company.com",
                    "subject": "Re: Monthly Report Request - Completed",
                    "attachment": "/reports/december_2023.pdf"
                }),
                outcome: ActionOutcome::Success {
                    result: json!({"message_id": "msg_12345", "sent_at": "2023-12-15T14:30:00Z"}),
                },
                duration_ns: 2_500_000_000, // 2.5 seconds
            },
            office_context.clone(),
        ),
    ];
    
    // Process task events
    for (i, event) in task_events.iter().enumerate() {
        println!("📋 Task step {}: {}", 
            i + 1,
            match &event.event_type {
                EventType::Action { action_name, .. } => action_name.clone(),
                EventType::Observation { observation_type, .. } => format!("Observed: {}", observation_type),
                EventType::Cognitive { process_type, .. } => format!("Cognitive: {:?}", process_type),
                _ => "Other".to_string(),
            }
        );
        
        // Show outcome if available
        if let EventType::Action { outcome, .. } = &event.event_type {
            match outcome {
                ActionOutcome::Success { result } => {
                    println!("   ✅ Success: {}", result);
                }
                ActionOutcome::Failure { error, .. } => {
                    println!("   ❌ Failed: {}", error);
                }
                ActionOutcome::Partial { result, issues } => {
                    println!("   ⚠️  Partial: {}, Issues: {:?}", result, issues);
                }
            }
        }
        
        // TODO: Actual database ingestion when implemented
        // db.ingest_event(event.clone()).await?;
    }
    
    println!("✅ Task automation completed successfully");
    
    // Show learned patterns
    println!("\n🎯 Patterns Learned:");
    println!("   📧 Email pattern: 'report request' → generate_monthly_report → send_email");
    println!("   ⏱️  Timing pattern: Report generation takes ~45 seconds");
    println!("   📊 Context pattern: Office environment + high priority → automated response");
    
    Ok(())
}

// Helper functions for creating test contexts

fn create_warehouse_context(warehouse_id: &str, position: (f64, f64), battery_level: f64) -> EventContext {
    EventContext::new(
        EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("warehouse_id".to_string(), json!(warehouse_id));
                vars.insert("battery_level".to_string(), json!(battery_level));
                vars.insert("temperature".to_string(), json!(18.5));
                vars.insert("lighting".to_string(), json!("good"));
                vars
            },
            spatial: Some(SpatialContext {
                location: (position.0, position.1, 0.0),
                bounds: Some(BoundingBox {
                    min: (0.0, 0.0, 0.0),
                    max: (100.0, 50.0, 10.0),
                }),
                reference_frame: warehouse_id.to_string(),
            }),
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
        vec![Goal {
            id: 1,
            description: "Navigate to target location efficiently".to_string(),
            priority: 0.8,
            deadline: Some(current_timestamp() + 300_000_000_000), // 5 minutes
            progress: 0.0,
            subgoals: Vec::new(),
        }],
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 65.0,
                memory_bytes: 512 * 1024 * 1024, // 512MB
                storage_bytes: 32 * 1024 * 1024 * 1024, // 32GB
                network_bandwidth: 1000,
            },
            external: {
                let mut ext = HashMap::new();
                ext.insert("gps".to_string(), ResourceAvailability {
                    available: true,
                    capacity: 1.0,
                    current_usage: 0.1,
                    estimated_cost: None,
                });
                ext.insert("lidar".to_string(), ResourceAvailability {
                    available: true,
                    capacity: 1.0,
                    current_usage: 0.3,
                    estimated_cost: None,
                });
                ext
            },
        },
    )
}

fn create_office_context(office_id: &str, temperature: f64) -> EventContext {
    EventContext::new(
        EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("office_id".to_string(), json!(office_id));
                vars.insert("temperature".to_string(), json!(temperature));
                vars.insert("noise_level".to_string(), json!(35.0));
                vars.insert("occupancy".to_string(), json!(12));
                vars
            },
            spatial: Some(SpatialContext {
                location: (0.0, 0.0, 0.0),
                bounds: Some(BoundingBox {
                    min: (0.0, 0.0, 0.0),
                    max: (50.0, 30.0, 3.0),
                }),
                reference_frame: office_id.to_string(),
            }),
            temporal: TemporalContext {
                time_of_day: Some(TimeOfDay {
                    hour: 14,
                    minute: 30,
                    timezone: "EST".to_string(),
                }),
                deadlines: vec![Deadline {
                    goal_id: 1,
                    timestamp: current_timestamp() + 3600_000_000_000, // 1 hour
                    priority: 0.8,
                }],
                patterns: Vec::new(),
            },
        },
        vec![Goal {
            id: 1,
            description: "Process incoming requests efficiently".to_string(),
            priority: 0.9,
            deadline: Some(current_timestamp() + 3600_000_000_000),
            progress: 0.0,
            subgoals: Vec::new(),
        }],
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 45.0,
                memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                storage_bytes: 500 * 1024 * 1024 * 1024, // 500GB
                network_bandwidth: 10000, // 10 Gbps
            },
            external: {
                let mut ext = HashMap::new();
                ext.insert("email_system".to_string(), ResourceAvailability {
                    available: true,
                    capacity: 1.0,
                    current_usage: 0.2,
                    estimated_cost: Some(0.01),
                });
                ext.insert("report_generator".to_string(), ResourceAvailability {
                    available: true,
                    capacity: 1.0,
                    current_usage: 0.0,
                    estimated_cost: Some(0.05),
                });
                ext
            },
        },
    )
}