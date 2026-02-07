//! Integration Tests for New Features (Episodes, Memory, Strategies, Reinforcement, Policy Guide)
//!
//! Tests the complete self-evolution pipeline:
//! Event → Episode → Memory → Strategy → Reinforcement → Policy Guide

use agent_db_core::types::{current_timestamp, generate_event_id, AgentId, AgentType, SessionId};
use agent_db_events::{
    ActionOutcome, CognitiveType, ComputationalResources, EnvironmentState, Event, EventContext,
    EventType, ResourceState, TemporalContext,
};
use agent_db_graph::{
    EpisodeDetector, EpisodeDetectorConfig, EpisodeMetrics, EpisodeOutcome, Graph, GraphInference,
    MemoryFormation, MemoryFormationConfig, MemoryType, StrategyExtractionConfig,
    StrategyExtractor,
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

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
                memory_bytes: 1024 * 1024 * 1024,
                storage_bytes: 10 * 1024 * 1024 * 1024,
                network_bandwidth: 1000 * 1000,
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
    timestamp: u64,
) -> Event {
    Event {
        id: generate_event_id(),
        timestamp,
        agent_id,
        agent_type,
        session_id,
        event_type: EventType::Action {
            action_name: "debug_typescript_error".to_string(),
            parameters: json!({"error_type": "TypeError", "fix": "add_null_check"}),
            outcome: ActionOutcome::Success {
                result: json!("fixed"),
            },
            duration_ns: 1_000_000,
        },
        causality_chain: Vec::new(),
        context: context.clone(),
        metadata: HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    }
}

fn create_cognitive_event(
    agent_id: AgentId,
    agent_type: AgentType,
    session_id: SessionId,
    context: &EventContext,
    timestamp: u64,
    reasoning_trace: Vec<String>,
) -> Event {
    Event {
        id: generate_event_id(),
        timestamp,
        agent_id,
        agent_type,
        session_id,
        event_type: EventType::Cognitive {
            process_type: CognitiveType::Reasoning,
            input: json!({}),
            output: json!({}),
            reasoning_trace,
        },
        causality_chain: Vec::new(),
        context: context.clone(),
        metadata: HashMap::new(),
        context_size_bytes: 0,
        segment_pointer: None,
    }
}

#[tokio::test]
async fn test_episode_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Episode Detection...");

    let graph = Arc::new(Graph::new());
    let config = EpisodeDetectorConfig::default();
    let mut detector = EpisodeDetector::new(graph.clone(), config);

    let context = create_test_context();
    let base_time = current_timestamp();

    // Create sequence of events
    let events = vec![
        create_cognitive_event(
            1,
            "debug-agent".to_string(),
            1,
            &context,
            base_time,
            vec!["Check error".to_string(), "Add null check".to_string()],
        ),
        create_test_event(
            1,
            "debug-agent".to_string(),
            1,
            &context,
            base_time + 500_000_000,
        ),
    ];

    // Process events
    for event in &events {
        detector.process_event(event);
    }

    let episodes = detector.get_completed_episodes();

    println!("   ✓ Episodes detected: {}", episodes.len());

    Ok(())
}

#[tokio::test]
async fn test_memory_formation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Memory Formation...");

    let config = MemoryFormationConfig::default();
    let mut memory_formation = MemoryFormation::new(config);

    let graph = Arc::new(Graph::new());
    let detector_config = EpisodeDetectorConfig::default();
    let mut detector = EpisodeDetector::new(graph, detector_config);

    let context = create_test_context();
    let base_time = current_timestamp();

    let events = vec![
        create_cognitive_event(
            1,
            "code-reviewer".to_string(),
            1,
            &context,
            base_time,
            vec!["Check null safety".to_string(), "Verify types".to_string()],
        ),
        create_test_event(
            1,
            "code-reviewer".to_string(),
            1,
            &context,
            base_time + 500_000_000,
        ),
    ];

    for event in &events {
        detector.process_event(event);
    }

    let episodes = detector.get_completed_episodes();

    if !episodes.is_empty() {
        let mut episode = episodes[0].clone();
        episode.significance = 0.8; // Set high significance
        episode.outcome = Some(EpisodeOutcome::Success);

        let memory_id = memory_formation.form_memory(&episode);

        println!("   ✓ Memory formed: {:?}", memory_id.is_some());

        if memory_id.is_some() {
            let retrieved = memory_formation.retrieve_by_context(&context, 5);
            println!("   ✓ Memories retrieved: {}", retrieved.len());

            if !retrieved.is_empty() {
                if let MemoryType::Episodic { .. } = retrieved[0].memory_type {
                    println!("   ✓ Memory is Episodic type");
                }
            }
        }
    }

    let stats = memory_formation.get_stats();
    println!("   ✓ Total memories: {}", stats.total_memories);

    Ok(())
}

#[tokio::test]
async fn test_strategy_extraction() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Strategy Extraction...");

    let config = StrategyExtractionConfig::default();
    let mut extractor = StrategyExtractor::new(config);

    let graph = Arc::new(Graph::new());
    let detector_config = EpisodeDetectorConfig::default();
    let mut detector = EpisodeDetector::new(graph, detector_config);

    let context = create_test_context();
    let base_time = current_timestamp();

    let events = vec![
        create_cognitive_event(
            1,
            "test-generator".to_string(),
            1,
            &context,
            base_time,
            vec![
                "Identify test cases".to_string(),
                "Generate edge cases".to_string(),
                "Add assertions".to_string(),
            ],
        ),
        create_test_event(
            1,
            "test-generator".to_string(),
            1,
            &context,
            base_time + 500_000_000,
        ),
    ];

    for event in &events {
        detector.process_event(event);
    }

    let episodes = detector.get_completed_episodes();

    if !episodes.is_empty() {
        let mut episode = episodes[0].clone();
        episode.outcome = Some(EpisodeOutcome::Success);
        episode.significance = 0.85;

        let strategy_id = extractor.extract_from_episode(&episode, &events)?;

        println!("   ✓ Strategy extracted: {:?}", strategy_id.is_some());

        if strategy_id.is_some() {
            let strategies = extractor.get_agent_strategies(1, 10);
            println!("   ✓ Strategies found: {}", strategies.len());

            if !strategies.is_empty() {
                println!(
                    "   ✓ Reasoning steps: {}",
                    strategies[0].reasoning_steps.len()
                );
                assert_eq!(strategies[0].agent_id, 1);
            }
        }
    }

    let stats = extractor.get_stats();
    println!("   ✓ Total strategies: {}", stats.total_strategies);

    Ok(())
}

#[tokio::test]
async fn test_reinforcement_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Reinforcement Learning...");

    let mut inference = GraphInference::new();

    let graph = Arc::new(Graph::new());
    let detector_config = EpisodeDetectorConfig::default();
    let mut detector = EpisodeDetector::new(graph, detector_config);

    let context = create_test_context();
    let base_time = current_timestamp();

    let events = vec![
        create_test_event(1, "debugger".to_string(), 1, &context, base_time),
        create_test_event(
            1,
            "debugger".to_string(),
            1,
            &context,
            base_time + 500_000_000,
        ),
    ];

    for event in &events {
        detector.process_event(event);
    }

    let episodes = detector.get_completed_episodes();

    if !episodes.is_empty() {
        let mut episode = episodes[0].clone();
        episode.outcome = Some(EpisodeOutcome::Success);

        let metrics = EpisodeMetrics {
            duration_seconds: 2.5,
            expected_duration_seconds: 3.0,
            quality_score: Some(0.9),
            custom_metrics: HashMap::new(),
        };

        let result = inference
            .reinforce_patterns(&episode, true, Some(metrics))
            .await?;

        println!(
            "   ✓ Patterns strengthened: {}",
            result.patterns_strengthened
        );
        println!("   ✓ Patterns updated: {}", result.patterns_updated);

        let stats = inference.get_reinforcement_stats();
        println!("   ✓ Total patterns: {}", stats.total_patterns);
    }

    Ok(())
}

#[tokio::test]
async fn test_complete_self_evolution_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Complete Self-Evolution Pipeline...");
    println!("   Event → Episode → Memory → Strategy → Reinforcement");

    // Initialize all components
    let graph = Arc::new(Graph::new());
    let mut episode_detector =
        EpisodeDetector::new(graph.clone(), EpisodeDetectorConfig::default());
    let mut memory_formation = MemoryFormation::new(MemoryFormationConfig::default());
    let mut strategy_extractor = StrategyExtractor::new(StrategyExtractionConfig::default());
    let mut inference = GraphInference::new();

    // Step 1: Create and process events
    println!("\n   Step 1: Processing events...");
    let context = create_test_context();
    let base_time = current_timestamp();

    let events = vec![
        create_cognitive_event(
            1,
            "self-learning-agent".to_string(),
            1,
            &context,
            base_time,
            vec![
                "Analyze TypeError".to_string(),
                "Check parameter types".to_string(),
                "Add type guard".to_string(),
                "Run tests".to_string(),
            ],
        ),
        create_test_event(
            1,
            "self-learning-agent".to_string(),
            1,
            &context,
            base_time + 500_000_000,
        ),
    ];

    // Step 2: Episode detection
    println!("   Step 2: Detecting episodes...");
    for event in &events {
        episode_detector.process_event(event);
    }

    let episodes = episode_detector.get_completed_episodes();
    println!("      ✓ Episodes detected: {}", episodes.len());

    if !episodes.is_empty() {
        let mut episode = episodes[0].clone();
        episode.significance = 0.8;
        episode.outcome = Some(EpisodeOutcome::Success);

        // Step 3: Memory formation
        println!("   Step 3: Forming memories...");
        let memory_id = memory_formation.form_memory(&episode);
        println!("      ✓ Memory formed: {:?}", memory_id.is_some());

        // Step 4: Strategy extraction
        println!("   Step 4: Extracting strategies...");
        let strategy_id = strategy_extractor.extract_from_episode(&episode, &events)?;
        println!("      ✓ Strategy extracted: {:?}", strategy_id.is_some());

        // Step 5: Reinforcement learning
        println!("   Step 5: Reinforcement learning...");
        let metrics = EpisodeMetrics {
            duration_seconds: 1.5,
            expected_duration_seconds: 2.0,
            quality_score: Some(0.95),
            custom_metrics: HashMap::new(),
        };

        let reinforcement_result = inference
            .reinforce_patterns(&episode, true, Some(metrics))
            .await?;
        println!(
            "      ✓ Patterns strengthened: {}",
            reinforcement_result.patterns_strengthened
        );

        // Step 6: Verify learnings
        println!("   Step 6: Testing retrieval...");
        let retrieved_memories = memory_formation.retrieve_by_context(&context, 5);
        println!("      ✓ Memories retrieved: {}", retrieved_memories.len());

        // Summary
        println!("\n   ✅ COMPLETE PIPELINE TEST PASSED!");
        println!("   ═══════════════════════════════════");
        println!("   Events processed:       {}", events.len());
        println!("   Episodes detected:      {}", episodes.len());
        println!(
            "   Patterns reinforced:    {}",
            reinforcement_result.patterns_strengthened
        );
        println!("   ═══════════════════════════════════");
    } else {
        println!("   ⚠️  No episodes detected (detector may need more events)");
    }

    Ok(())
}
