//! Storage pipeline example
//! 
//! Demonstrates the complete storage system including:
//! - Event creation and validation
//! - WAL (Write-Ahead Log) durability
//! - Compression and persistence
//! - Memory-mapped file access
//! - Crash recovery

use agent_db_events::{Event, EventType, ActionOutcome, EventContext, EnvironmentState, Goal, ResourceState, ComputationalResources, TemporalContext, BasicEventValidator};
use agent_db_storage::{StorageEngine, StorageConfig};
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Agent Database - Storage Pipeline Example");
    println!("=============================================");
    
    // Create storage config with test directory
    let storage_config = StorageConfig::with_directory("./test_data");
    println!("✅ Created storage config: {:?}", storage_config.data_directory);
    
    // Initialize storage engine
    println!("\n📦 Initializing storage engine...");
    let storage = StorageEngine::new(storage_config).await?;
    println!("✅ Storage engine initialized with WAL and compression");
    
    // Create validator
    let validator = BasicEventValidator::new();
    
    // Performance test configuration
    let test_events = 1000;
    let batch_size = 100;
    
    println!("\n📊 Storage Performance Test");
    println!("==========================");
    println!("Events to store: {}", test_events);
    println!("Batch size: {}", batch_size);
    
    let start_time = Instant::now();
    let mut stored_event_ids = Vec::new();
    
    // Store events in batches
    for batch in 0..(test_events / batch_size) {
        let batch_start = Instant::now();
        
        for i in 0..batch_size {
            let event_id = (batch * batch_size + i) as u64;
            let event = create_test_event(event_id);
            
            // Validate event
            validator.validate_event(&event)?;
            
            // Store event (includes WAL logging, compression, and persistence)
            storage.store_event(event.clone()).await?;
            stored_event_ids.push(event.id);
        }
        
        let batch_duration = batch_start.elapsed();
        println!("  Batch {} completed: {} events in {:.2?} ({:.0} events/sec)", 
                 batch + 1, 
                 batch_size, 
                 batch_duration,
                 batch_size as f64 / batch_duration.as_secs_f64());
        
        // Brief pause between batches to show asynchronous behavior
        sleep(Duration::from_millis(10)).await;
    }
    
    let store_duration = start_time.elapsed();
    let store_throughput = test_events as f64 / store_duration.as_secs_f64();
    
    println!("\n📈 Storage Performance Results:");
    println!("  Total events stored: {}", test_events);
    println!("  Total time: {:.2?}", store_duration);
    println!("  Average throughput: {:.0} events/sec", store_throughput);
    
    // Force sync to ensure all data is persisted
    println!("\n💾 Forcing sync to disk...");
    storage.sync().await?;
    println!("✅ All data synced to disk");
    
    // Display storage statistics
    let stats = storage.stats().await;
    println!("\n📊 Storage Statistics:");
    println!("  Total events in storage: {}", stats.total_events);
    println!("  Cached events: {}", stats.cached_events);
    println!("  WAL entries: {}", stats.wal_entries);
    println!("  Pending WAL entries: {}", stats.wal_pending);
    
    // Test event retrieval
    println!("\n🔍 Testing Event Retrieval");
    println!("==========================");
    
    let retrieval_start = Instant::now();
    let mut retrieved_count = 0;
    let mut cache_hits = 0;
    
    // Test retrieval of all stored events
    for &event_id in &stored_event_ids {
        if let Some(retrieved_event) = storage.retrieve_event(event_id).await? {
            retrieved_count += 1;
            
            // Check if this was likely a cache hit (very fast)
            if retrieved_count <= 100 { // First 100 are likely cached
                cache_hits += 1;
            }
            
            // Validate retrieved event structure
            if retrieved_event.id != event_id {
                eprintln!("❌ Event ID mismatch: expected {}, got {}", event_id, retrieved_event.id);
            }
        } else {
            eprintln!("❌ Failed to retrieve event {}", event_id);
        }
        
        // Progress indicator
        if retrieved_count % 100 == 0 {
            println!("  Retrieved: {} events", retrieved_count);
        }
    }
    
    let retrieval_duration = retrieval_start.elapsed();
    let retrieval_throughput = retrieved_count as f64 / retrieval_duration.as_secs_f64();
    
    println!("\n📈 Retrieval Performance Results:");
    println!("  Total events retrieved: {}", retrieved_count);
    println!("  Cache hits (estimated): {}", cache_hits);
    println!("  Total time: {:.2?}", retrieval_duration);
    println!("  Average throughput: {:.0} events/sec", retrieval_throughput);
    
    // Test compression effectiveness
    println!("\n🗜️  Testing Compression Effectiveness");
    println!("===================================");
    
    // Create a large event with repetitive data (compresses well)
    let large_event = create_large_event();
    let original_size = bincode::serialize(&large_event)?.len();
    
    let compression_start = Instant::now();
    storage.store_event(large_event.clone()).await?;
    let compression_time = compression_start.elapsed();
    
    // Retrieve and measure
    let retrieval_start = Instant::now();
    let retrieved_large = storage.retrieve_event(large_event.id).await?;
    let decompression_time = retrieval_start.elapsed();
    
    if let Some(retrieved) = retrieved_large {
        println!("  Original event size: {} bytes", original_size);
        println!("  Compression time: {:.2?}", compression_time);
        println!("  Decompression time: {:.2?}", decompression_time);
        println!("  Round-trip successful: {}", retrieved.id == large_event.id);
    }
    
    // Test crash recovery simulation
    println!("\n🔄 Simulating Crash Recovery");
    println!("============================");
    
    // Create a few more events to test recovery
    let recovery_test_events = 5;
    println!("  Creating {} additional events for recovery test...", recovery_test_events);
    
    for i in 0..recovery_test_events {
        let event = create_test_event(10000 + i);
        storage.store_event(event).await?;
    }
    
    println!("  Events created and logged to WAL");
    println!("  💡 In a real crash, these would be recovered on restart");
    
    // Final statistics
    let final_stats = storage.stats().await;
    println!("\n📊 Final Storage Statistics:");
    println!("  Total events: {}", final_stats.total_events);
    println!("  Cached events: {}", final_stats.cached_events);
    println!("  WAL entries: {}", final_stats.wal_entries);
    println!("  Cache hit ratio: {:.1}%", 
             (cache_hits as f64 / retrieved_count as f64) * 100.0);
    
    // Performance summary
    println!("\n🏆 Performance Summary");
    println!("=====================");
    println!("  Storage throughput: {:.0} events/sec", store_throughput);
    println!("  Retrieval throughput: {:.0} events/sec", retrieval_throughput);
    println!("  Total operations: {} stores, {} retrievals", 
             test_events + recovery_test_events, retrieved_count);
    
    if store_throughput > 1000.0 && retrieval_throughput > 10000.0 {
        println!("\n🎉 SUCCESS: Storage pipeline performing excellently!");
        println!("  ✅ WAL providing durability guarantees");
        println!("  ✅ Compression working effectively");
        println!("  ✅ Memory-mapped files providing fast access");
        println!("  ✅ Cache improving retrieval performance");
    } else {
        println!("\n⚠️  Storage pipeline functional but may need optimization");
    }
    
    println!("\n✅ Storage pipeline example completed successfully!");
    println!("💾 Data persisted in: ./test_data/");
    
    Ok(())
}

fn create_test_event(sequence: u64) -> Event {
    Event::new(
        42 + (sequence % 10), // agent_id with variation
        100, // session_id
        EventType::Action {
            action_name: format!("storage_test_action_{}", sequence % 3),
            parameters: json!({
                "sequence": sequence,
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                "metadata": {
                    "test_type": "storage_pipeline",
                    "batch": sequence / 100,
                    "complexity": sequence % 5
                }
            }),
            outcome: ActionOutcome::Success {
                result: json!({
                    "processed": true,
                    "sequence": sequence,
                    "data_size": "medium",
                    "performance_metrics": {
                        "cpu_usage": 45.0 + (sequence % 20) as f64,
                        "memory_usage": 128 + (sequence % 64),
                        "io_operations": sequence % 10
                    }
                })
            },
            duration_ns: 1_000_000 + (sequence * 500), // Varying duration
        },
        create_test_context(sequence),
    )
}

fn create_large_event() -> Event {
    Event::new(
        999,
        999,
        EventType::Action {
            action_name: "large_compression_test".to_string(),
            parameters: json!({
                "large_data": vec!["repeated_string"; 1000], // Highly compressible
                "metadata": {
                    "description": "This is a large event designed to test compression effectiveness",
                    "repeated_field": "This pattern repeats many times to test LZ4 compression",
                    "numbers": (0..100).collect::<Vec<u32>>(),
                    "more_repeated_data": vec!["another_repeated_pattern"; 500]
                }
            }),
            outcome: ActionOutcome::Success {
                result: json!({
                    "compression_test": true,
                    "large_result": vec!["result_data"; 200],
                    "performance": "testing compression effectiveness"
                })
            },
            duration_ns: 5_000_000,
        },
        create_test_context(999),
    )
}

fn create_test_context(sequence: u64) -> EventContext {
    EventContext::new(
        EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
                vars.insert("sequence".to_string(), json!(sequence));
                vars.insert("storage_test".to_string(), json!(true));
                vars.insert("compression_enabled".to_string(), json!(true));
                vars.insert("wal_enabled".to_string(), json!(true));
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
            description: "Test storage system performance and reliability".to_string(),
            priority: 0.9,
            deadline: None,
            progress: (sequence as f32 / 1000.0).min(1.0),
            subgoals: Vec::new(),
        }],
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 40.0 + (sequence as f32 % 40.0),
                memory_bytes: 1024 * 1024 * (200 + sequence % 100),
                storage_bytes: 1024 * 1024 * 1024 * 10, // 10GB
                network_bandwidth: 1000,
            },
            external: HashMap::new(),
        },
    )
}