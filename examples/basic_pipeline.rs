//! Basic event pipeline example
//! 
//! Demonstrates creating events, validating them, and buffering them
//! to showcase the core functionality of the agentic database.

use agent_db_events::{
    Event, EventType, ActionOutcome, EventContext, EnvironmentState, Goal,
    ResourceState, ComputationalResources, TemporalContext, BasicEventValidator,
    EventBuffer
};
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("🚀 Agent Database - Basic Pipeline Example");
    println!("==========================================");
    
    // Create validator
    let validator = BasicEventValidator::new();
    println!("✅ Created event validator");
    
    // Create event buffer
    let mut buffer = EventBuffer::new(1000);
    println!("✅ Created event buffer with capacity 1000");
    
    // Performance tracking
    let start_time = Instant::now();
    let target_events = 10_000;
    
    println!("\n📊 Creating and processing {} events...", target_events);
    
    let mut successful_events = 0;
    let mut validation_errors = 0;
    
    for i in 0..target_events {
        // Create a test event
        let event = create_test_event(i as u64);
        
        // Validate the event
        match validator.validate_event(&event) {
            Ok(()) => {
                // Add to buffer
                match buffer.add(event) {
                    Ok(()) => successful_events += 1,
                    Err(e) => println!("Buffer error: {}", e),
                }
            }
            Err(e) => {
                validation_errors += 1;
                if validation_errors <= 5 {
                    println!("Validation error: {}", e);
                }
            }
        }
        
        // Progress indicator
        if i > 0 && i % 1000 == 0 {
            println!("  Processed: {} events", i);
        }
    }
    
    let duration = start_time.elapsed();
    let events_per_second = successful_events as f64 / duration.as_secs_f64();
    
    println!("\n📈 Performance Results:");
    println!("  Total events processed: {}", target_events);
    println!("  Successful events: {}", successful_events);
    println!("  Validation errors: {}", validation_errors);
    println!("  Total time: {:.2?}", duration);
    println!("  Events per second: {:.0}", events_per_second);
    
    // Display buffer statistics
    let stats = buffer.stats();
    println!("\n📊 Buffer Statistics:");
    println!("  Events in buffer: {}", buffer.len());
    println!("  Total added: {}", stats.total_added);
    println!("  Total dropped: {}", stats.total_dropped);
    
    if events_per_second > 10_000.0 {
        println!("\n🎉 SUCCESS: Achieved target of 10K+ events/second!");
    } else {
        println!("\n⚠️  Target not met, but basic pipeline working!");
    }
    
    println!("\n🔍 Testing context fingerprinting...");
    let event1 = create_test_event(1);
    let event2 = create_test_event(2);
    let same_context = event1.context.fingerprint == event2.context.fingerprint;
    println!("  Context fingerprint working: {}", same_context);
    
    println!("\n✅ Basic pipeline example completed successfully!");
}

fn create_test_event(sequence: u64) -> Event {
    Event::new(
        42 + (sequence % 10), // agent_id with some variation
        100, // session_id
        EventType::Action {
            action_name: format!("test_action_{}", sequence % 5),
            parameters: json!({
                "sequence": sequence,
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            }),
            outcome: ActionOutcome::Success {
                result: json!({
                    "processed": true,
                    "sequence": sequence
                })
            },
            duration_ns: 1_000_000 + (sequence * 1000), // Varying duration
        },
        create_test_context(sequence),
    )
}

fn create_test_context(sequence: u64) -> EventContext {
    EventContext::new(
        EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("sequence".to_string(), json!(sequence));
                vars.insert("test_mode".to_string(), json!(true));
                vars
            },
            spatial: None,
            temporal: TemporalContext {
                time_of_day: None,
                deadlines: Vec::new(),
                patterns: Vec::new(),
            },
        },
        vec![Goal {
            id: 1,
            description: "Process test events efficiently".to_string(),
            priority: 0.8,
            deadline: None,
            progress: (sequence as f32 / 10000.0).min(1.0),
            subgoals: Vec::new(),
        }],
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 50.0 + (sequence as f32 % 30.0),
                memory_bytes: 1024 * 1024 * (100 + sequence % 50),
                storage_bytes: 1024 * 1024 * 1024,
                network_bandwidth: 1000,
            },
            external: HashMap::new(),
        },
    )
}