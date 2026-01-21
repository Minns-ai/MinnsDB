use agent_db_core::*;
use agent_db_events::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use serde_json::json;
use std::time::Duration;

fn create_test_event(agent_id: AgentId, session_id: SessionId) -> Event {
    let context = EventContext::new(
        EnvironmentState {
            variables: std::collections::HashMap::new(),
            spatial: None,
            temporal: TemporalContext {
                time_of_day: None,
                deadlines: Vec::new(),
                patterns: Vec::new(),
            },
        },
        Vec::new(),
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 50.0,
                memory_bytes: 1024 * 1024,
                storage_bytes: 1024 * 1024 * 1024,
                network_bandwidth: 1000,
            },
            external: std::collections::HashMap::new(),
        },
    );

    Event::new(
        agent_id,
        "benchmark_agent".to_string(), // agent_type
        session_id,
        EventType::Action {
            action_name: "test_action".to_string(),
            parameters: json!({"x": 10, "y": 20}),
            outcome: ActionOutcome::Success {
                result: json!({"success": true})
            },
            duration_ns: 1_000_000,
        },
        context,
    )
}

fn bench_event_creation(c: &mut Criterion) {
    c.bench_function("event_creation", |b| {
        b.iter(|| {
            let event = create_test_event(black_box(123), black_box(456));
            black_box(event);
        });
    });
}

fn bench_event_serialization(c: &mut Criterion) {
    let event = create_test_event(123, 456);
    
    c.bench_function("event_json_serialize", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(black_box(&event)).unwrap();
            black_box(serialized);
        });
    });
    
    c.bench_function("event_bincode_serialize", |b| {
        b.iter(|| {
            let serialized = bincode::serialize(black_box(&event)).unwrap();
            black_box(serialized);
        });
    });
}

fn bench_event_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_buffer");
    
    for buffer_size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("buffer_add", buffer_size),
            buffer_size,
            |b, &buffer_size| {
                let mut buffer = EventBuffer::new(buffer_size);
                let events: Vec<_> = (0..buffer_size)
                    .map(|i| create_test_event(i as u64, 1))
                    .collect();
                
                b.iter(|| {
                    for event in &events {
                        buffer.add(black_box(event.clone())).unwrap();
                    }
                    buffer.clear();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_context_fingerprinting(c: &mut Criterion) {
    let context = EventContext::new(
        EnvironmentState {
            variables: {
                let mut vars = std::collections::HashMap::new();
                vars.insert("temp".to_string(), json!(23.5));
                vars.insert("humidity".to_string(), json!(65.0));
                vars.insert("location".to_string(), json!("office_a"));
                vars
            },
            spatial: Some(SpatialContext {
                location: (10.0, 20.0, 0.0),
                bounds: None,
                reference_frame: "world".to_string(),
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
            description: "Complete task A".to_string(),
            priority: 0.8,
            deadline: None,
            progress: 0.5,
            subgoals: Vec::new(),
        }],
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 75.0,
                memory_bytes: 2048 * 1024,
                storage_bytes: 1024 * 1024 * 1024,
                network_bandwidth: 1000,
            },
            external: std::collections::HashMap::new(),
        },
    );
    
    c.bench_function("context_fingerprint", |b| {
        b.iter(|| {
            let fingerprint = black_box(&context).compute_fingerprint();
            black_box(fingerprint);
        });
    });
}

fn bench_event_validation(c: &mut Criterion) {
    let validator = BasicEventValidator::new();
    let valid_event = create_test_event(123, 456);
    
    c.bench_function("event_validation", |b| {
        b.iter(|| {
            let result = validator.validate_event(black_box(&valid_event));
            black_box(result);
        });
    });
}

// Simulate high-throughput event ingestion
fn bench_throughput_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(10));
    
    for events_per_batch in [1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("ingestion_simulation", events_per_batch),
            events_per_batch,
            |b, &count| {
                b.iter(|| {
                    let events: Vec<_> = (0..count)
                        .map(|i| create_test_event(i as u64 % 100, 1))
                        .collect();
                    
                    // Simulate processing pipeline
                    for event in events {
                        // Validation step
                        let _valid = event.id != 0;
                        
                        // Serialization step
                        let _serialized = bincode::serialize(&event).unwrap();
                        
                        // Context processing step
                        let _fingerprint = event.context.fingerprint;
                        
                        black_box(());
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_event_creation,
    bench_event_serialization,
    bench_event_buffer,
    bench_context_fingerprinting,
    bench_event_validation,
    bench_throughput_simulation
);

criterion_main!(benches);