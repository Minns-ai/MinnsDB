//! Performance benchmarks for event system
//! 
//! Benchmarks core event operations including creation, validation, and buffering.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use agent_db_events::{
    Event, EventType, ActionOutcome, EventContext, EnvironmentState, Goal,
    ResourceState, ComputationalResources, TemporalContext, BasicEventValidator,
    EventBuffer
};
use serde_json::json;
use std::collections::HashMap;

fn create_test_event(sequence: u64) -> Event {
    Event::new(
        42,
        100,
        EventType::Action {
            action_name: "benchmark_action".to_string(),
            parameters: json!({"sequence": sequence}),
            outcome: ActionOutcome::Success {
                result: json!({"processed": true})
            },
            duration_ns: 1_000_000,
        },
        create_test_context(),
    )
}

fn create_test_context() -> EventContext {
    EventContext::new(
        EnvironmentState {
            variables: {
                let mut vars = HashMap::new();
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
            description: "Benchmark goal".to_string(),
            priority: 0.8,
            deadline: None,
            progress: 0.5,
            subgoals: Vec::new(),
        }],
        ResourceState {
            computational: ComputationalResources {
                cpu_percent: 50.0,
                memory_bytes: 1024 * 1024 * 100,
                storage_bytes: 1024 * 1024 * 1024,
                network_bandwidth: 1000,
            },
            external: HashMap::new(),
        },
    )
}

fn bench_event_creation(c: &mut Criterion) {
    c.bench_function("event_creation", |b| {
        b.iter(|| {
            let event = create_test_event(black_box(42));
            black_box(event)
        })
    });
}

fn bench_event_validation(c: &mut Criterion) {
    let validator = BasicEventValidator::new();
    let event = create_test_event(42);
    
    c.bench_function("event_validation", |b| {
        b.iter(|| {
            validator.validate_event(black_box(&event)).unwrap()
        })
    });
}

fn bench_event_buffering(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_buffering");
    
    for buffer_size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("add_events", buffer_size),
            buffer_size,
            |b, &size| {
                b.iter(|| {
                    let mut buffer = EventBuffer::new(size);
                    for i in 0..size.min(1000) {
                        let event = create_test_event(i as u64);
                        buffer.add(event).unwrap();
                    }
                    buffer
                })
            },
        );
    }
    
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    
    for event_count in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_validate_buffer", event_count),
            event_count,
            |b, &count| {
                b.iter(|| {
                    let validator = BasicEventValidator::new();
                    let mut buffer = EventBuffer::new(count);
                    
                    for i in 0..count {
                        let event = create_test_event(black_box(i as u64));
                        validator.validate_event(&event).unwrap();
                        buffer.add(event).unwrap();
                    }
                    
                    (validator, buffer)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_context_fingerprinting(c: &mut Criterion) {
    c.bench_function("context_fingerprinting", |b| {
        b.iter(|| {
            let context = create_test_context();
            black_box(context.fingerprint)
        })
    });
}

criterion_group!(
    benches,
    bench_event_creation,
    bench_event_validation,
    bench_event_buffering,
    bench_full_pipeline,
    bench_context_fingerprinting
);
criterion_main!(benches);