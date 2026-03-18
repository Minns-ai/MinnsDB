// crates/agent-db-graph/src/strategies/tests.rs

use super::synthesis::build_playbook;
use super::types::META_Q_VALUE;
use super::*;
use crate::episodes::{Episode, EpisodeOutcome};
use agent_db_core::types::current_timestamp;
use agent_db_events::core::{ActionOutcome, Event, EventType};
use std::collections::HashMap;

#[test]
fn test_goal_bucket_id_differs_for_different_goal_sets_with_same_min() {
    // Both sets share the same minimum goal id (1), but differ in the full set.
    let bucket_a = compute_goal_bucket_id_from_ids(&[1, 2, 3]);
    let bucket_b = compute_goal_bucket_id_from_ids(&[1, 4, 5]);

    assert_ne!(
        bucket_a, bucket_b,
        "Different goal sets with the same min id must produce different bucket ids"
    );
}

#[test]
fn test_goal_bucket_id_identical_for_same_goals() {
    let bucket_a = compute_goal_bucket_id_from_ids(&[3, 1, 2]);
    let bucket_b = compute_goal_bucket_id_from_ids(&[1, 2, 3]);

    assert_eq!(
        bucket_a, bucket_b,
        "Identical goal sets (regardless of order) must produce the same bucket id"
    );
}

#[test]
fn test_goal_bucket_id_empty_returns_zero() {
    assert_eq!(compute_goal_bucket_id_from_ids(&[]), 0);
}

#[test]
fn test_goal_bucket_id_matches_event_context() {
    use agent_db_events::core::{EventContext, Goal};

    let goals = vec![
        Goal {
            id: 10,
            description: String::new(),
            priority: 0.5,
            deadline: None,
            progress: 0.0,
            subgoals: Vec::new(),
        },
        Goal {
            id: 3,
            description: String::new(),
            priority: 0.9,
            deadline: None,
            progress: 0.0,
            subgoals: Vec::new(),
        },
    ];

    let ctx = EventContext::new(Default::default(), goals.clone(), Default::default());

    let strategy_bucket = compute_goal_bucket_id_from_ids(&[10, 3]);

    assert_eq!(
        ctx.goal_bucket_id, strategy_bucket,
        "Strategy bucket id must match EventContext.goal_bucket_id for the same goals"
    );
}

fn make_test_episode(outcome: EpisodeOutcome) -> Episode {
    use agent_db_core::types::Timestamp;
    use agent_db_events::core::EventContext;
    Episode {
        id: 1,
        episode_version: 1,
        agent_id: 1,
        start_event: 1u128,
        end_event: Some(2u128),
        events: vec![1u128, 2u128],
        session_id: 1,
        context_signature: 0,
        context: EventContext::default(),
        outcome: Some(outcome.clone()),
        start_timestamp: Timestamp::from(1000u64),
        end_timestamp: Some(Timestamp::from(2000u64)),
        significance: 0.8,
        prediction_error: 0.0,
        self_judged_quality: None,
        salience_score: 0.5,
        last_event_timestamp: Some(Timestamp::from(2000u64)),
        consecutive_outcome_count: 0,
    }
}

fn make_test_event(event_type: EventType) -> Event {
    use agent_db_core::types::Timestamp;
    Event {
        id: 1u128,
        timestamp: Timestamp::from(1000u64),
        agent_id: 1,
        agent_type: "test".to_string(),
        session_id: 1,
        event_type,
        causality_chain: vec![],
        context: agent_db_events::core::EventContext::default(),
        metadata: Default::default(),
        context_size_bytes: 0,
        segment_pointer: None,
        is_code: false,
    }
}

#[test]
fn test_playbook_not_execute_raw() {
    let events = vec![make_test_event(EventType::Action {
        action_name: "cognitive_plan".to_string(),
        parameters: serde_json::json!({"query": "plan deployment"}),
        outcome: ActionOutcome::Success {
            result: serde_json::json!({"text": "plan created"}),
        },
        duration_ns: 1000,
    })];

    let playbook = build_playbook(&events, &StrategyType::Positive);
    assert!(!playbook.is_empty());
    // Should NOT contain "Execute 'cognitive_plan'"
    assert!(
        !playbook[0].action.starts_with("Execute"),
        "Playbook action should not be raw 'Execute', got: {}",
        playbook[0].action
    );
    // Should contain humanized description
    assert!(
        playbook[0].action.contains("Cognitive Plan"),
        "Playbook action should contain humanized name, got: {}",
        playbook[0].action
    );
}

#[test]
fn test_summary_not_do_this() {
    let episode = make_test_episode(EpisodeOutcome::Success);
    let events = vec![make_test_event(EventType::Action {
        action_name: "search".to_string(),
        parameters: serde_json::json!({"query": "find users"}),
        outcome: ActionOutcome::Success {
            result: serde_json::json!({"text": "found 10 users"}),
        },
        duration_ns: 1000,
    })];

    let summary = synthesize_strategy_summary(
        &StrategyType::Positive,
        &EpisodeOutcome::Success,
        &episode,
        &events,
        &[],
        &[],
    );
    // Should NOT start with generic "DO this when applicable"
    assert!(
        !summary.contains("DO this when applicable"),
        "Summary should not contain generic template, got: {}",
        summary
    );
}

#[test]
fn test_cognitive_in_playbook() {
    let events = vec![make_test_event(EventType::Cognitive {
        process_type: agent_db_events::core::CognitiveType::Reasoning,
        input: serde_json::json!("analyze options"),
        output: serde_json::json!("chose option A"),
        reasoning_trace: vec!["step1".to_string()],
    })];

    let playbook = build_playbook(&events, &StrategyType::Positive);
    assert!(!playbook.is_empty());
    // Should NOT contain "Think (Reasoning)" — should have actual content
    assert!(
        !playbook[0].action.starts_with("Think ("),
        "Playbook cognitive step should not be generic 'Think', got: {}",
        playbook[0].action
    );
    assert!(
        playbook[0].action.contains("Reasoning"),
        "Should contain reasoning label, got: {}",
        playbook[0].action
    );
    assert!(
        playbook[0].action.contains("chose option A"),
        "Should contain cognitive output, got: {}",
        playbook[0].action
    );
}

#[test]
fn test_strategy_ema_q_value() {
    let mut extractor = StrategyExtractor::new(Default::default());
    // Create a minimal strategy
    let strategy = Strategy {
        id: 1,
        name: "test_strategy".to_string(),
        summary: String::new(),
        when_to_use: String::new(),
        when_not_to_use: String::new(),
        failure_modes: Vec::new(),
        playbook: Vec::new(),
        counterfactual: String::new(),
        supersedes: Vec::new(),
        applicable_domains: Vec::new(),
        lineage_depth: 0,
        summary_embedding: Vec::new(),
        agent_id: 1,
        reasoning_steps: Vec::new(),
        context_patterns: Vec::new(),
        success_indicators: Vec::new(),
        failure_patterns: Vec::new(),
        quality_score: 0.0,
        success_count: 0,
        failure_count: 0,
        support_count: 0,
        strategy_type: StrategyType::Positive,
        precondition: String::new(),
        action_hint: String::new(),
        expected_success: 0.5,
        expected_cost: 0.0,
        expected_value: 0.0,
        confidence: 0.0,
        contradictions: Vec::new(),
        goal_bucket_id: 1,
        behavior_signature: String::new(),
        source_episodes: Vec::new(),
        created_at: current_timestamp(),
        last_used: current_timestamp(),
        metadata: HashMap::new(),
        self_judged_quality: None,
        source_outcomes: Vec::new(),
        version: 1,
        parent_strategy: None,
    };
    extractor.strategies.insert(1, strategy);

    // Record 3 successes (Bayesian phase)
    extractor.update_strategy_outcome(1, true).unwrap();
    extractor.update_strategy_outcome(1, true).unwrap();
    extractor.update_strategy_outcome(1, true).unwrap();

    let s = extractor.strategies.get(&1).unwrap();
    assert_eq!(s.success_count, 3);
    assert_eq!(s.failure_count, 0);
    // Q-value should be in metadata
    assert!(s.metadata.contains_key(META_Q_VALUE));
    let q: f32 = s.metadata.get(META_Q_VALUE).unwrap().parse().unwrap();
    assert!(
        q > 0.5,
        "Q should be above 0.5 after 3 successes, got {}",
        q
    );
    // Confidence should reflect both sample size and outcome quality
    assert!(s.confidence > 0.0, "Confidence should be positive");

    // Record 3 more (crosses Q_KICK_IN=5)
    extractor.update_strategy_outcome(1, true).unwrap();
    extractor.update_strategy_outcome(1, true).unwrap();
    extractor.update_strategy_outcome(1, false).unwrap();

    let s = extractor.strategies.get(&1).unwrap();
    assert_eq!(s.success_count, 5);
    assert_eq!(s.failure_count, 1);
    // Now in Q phase — confidence should reflect outcome quality
    let q: f32 = s.metadata.get(META_Q_VALUE).unwrap().parse().unwrap();
    assert!(q > 0.5, "Q should still be >0.5, got {}", q);
    // Confidence = sample_confidence * piecewise_score, should be substantial
    assert!(
        s.confidence > 0.3,
        "Confidence should be substantial, got {}",
        s.confidence
    );
}
