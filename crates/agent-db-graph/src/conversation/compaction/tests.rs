//! Tests for conversation compaction.

use super::*;
use crate::conversation::types::{ConversationMessage, ConversationSession};
use crate::llm_client::parse_json_from_llm;
use turn_processing::MAX_TRANSCRIPT_CHARS;

fn make_ingest(messages: Vec<(&str, &str)>) -> crate::conversation::types::ConversationIngest {
    crate::conversation::types::ConversationIngest {
        case_id: Some("test_compaction".to_string()),
        sessions: vec![ConversationSession {
            session_id: "s1".to_string(),
            topic: None,
            messages: messages
                .into_iter()
                .map(|(role, content)| ConversationMessage {
                    role: role.to_string(),
                    content: content.to_string(),
                    metadata: Default::default(),
                })
                .collect(),
            timestamp: None,
            contains_fact: None,
            fact_id: None,
            fact_quote: None,
            answers: vec![],
        }],
        queries: vec![],
        group_id: Default::default(),
        metadata: Default::default(),
    }
}

#[test]
fn test_split_into_turns() {
    let data = crate::conversation::types::ConversationIngest {
        case_id: Some("test".to_string()),
        sessions: vec![
            ConversationSession {
                session_id: "s1".to_string(),
                topic: None,
                messages: vec![
                    ConversationMessage {
                        role: "user".to_string(),
                        content: "Hi".to_string(),
                        metadata: Default::default(),
                    },
                    ConversationMessage {
                        role: "assistant".to_string(),
                        content: "Hello!".to_string(),
                        metadata: Default::default(),
                    },
                    ConversationMessage {
                        role: "user".to_string(),
                        content: "Where am I?".to_string(),
                        metadata: Default::default(),
                    },
                    ConversationMessage {
                        role: "assistant".to_string(),
                        content: "Lisbon".to_string(),
                        metadata: Default::default(),
                    },
                ],
                timestamp: None,
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            },
            ConversationSession {
                session_id: "s2".to_string(),
                topic: None,
                messages: vec![
                    ConversationMessage {
                        role: "user".to_string(),
                        content: "I moved".to_string(),
                        metadata: Default::default(),
                    },
                    ConversationMessage {
                        role: "assistant".to_string(),
                        content: "Where?".to_string(),
                        metadata: Default::default(),
                    },
                ],
                timestamp: None,
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            },
        ],
        queries: vec![],
        group_id: Default::default(),
        metadata: Default::default(),
    };

    let turns = split_into_turns(&data);
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[0].turn_index, 0);
    assert_eq!(turns[0].session_index, 0);
    assert_eq!(turns[0].messages.len(), 2);
    assert_eq!(turns[1].turn_index, 1);
    assert_eq!(turns[1].session_index, 0);
    assert_eq!(turns[2].turn_index, 2);
    assert_eq!(turns[2].session_index, 1);
}

#[test]
fn test_format_turn_transcript() {
    let turn = ConversationTurn {
        session_timestamp: None,
        messages: vec![
            ConversationMessage {
                role: "user".to_string(),
                content: "I moved to NYC".to_string(),
                metadata: Default::default(),
            },
            ConversationMessage {
                role: "assistant".to_string(),
                content: "Great!".to_string(),
                metadata: Default::default(),
            },
        ],
        session_index: 0,
        turn_index: 1,
    };

    let transcript = format_turn_transcript(
        &turn,
        Some("User lives in Lisbon"),
        "User: state:location=Lisbon",
    );
    assert!(transcript.contains("User lives in Lisbon"));
    assert!(transcript.contains("state:location=Lisbon"));
    assert!(transcript.contains("user: I moved to NYC"));
}

// 1. test_format_transcript
#[test]
fn test_format_transcript() {
    let data = make_ingest(vec![
        ("user", "Hello, I want to plan a trip"),
        ("assistant", "Sure! Where would you like to go?"),
        ("user", "I want to visit Japan in April"),
    ]);

    let transcript = format_transcript(&data);
    assert!(transcript.contains("user: Hello, I want to plan a trip\n"));
    assert!(transcript.contains("assistant: Sure! Where would you like to go?\n"));
    assert!(transcript.contains("user: I want to visit Japan in April\n"));
}

#[test]
fn test_format_transcript_truncation() {
    // Create a very long transcript that exceeds MAX_TRANSCRIPT_CHARS
    let long_msg = "x".repeat(20_000);
    let data = make_ingest(vec![("user", &long_msg)]);

    let transcript = format_transcript(&data);
    assert!(transcript.len() <= MAX_TRANSCRIPT_CHARS);
}

// 2. test_parse_compaction_response
#[test]
fn test_parse_compaction_response() {
    let json = r#"{
        "facts": [
            {
                "statement": "Alice lives in Paris",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Paris",
                "confidence": 0.9
            }
        ],
        "goals": [
            {
                "description": "Plan a trip to Japan",
                "status": "active",
                "owner": "user"
            }
        ],
        "procedural_summary": {
            "objective": "Plan vacation",
            "progress_status": "in_progress",
            "steps": [
                {
                    "step_number": 1,
                    "action": "Choose destination",
                    "result": "Selected Japan",
                    "outcome": "success"
                }
            ],
            "overall_summary": "User is planning a trip to Japan",
            "takeaway": "User prefers travel to Asia"
        }
    }"#;

    let response: CompactionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.facts.len(), 1);
    assert_eq!(response.facts[0].subject, "Alice");
    assert_eq!(response.goals.len(), 1);
    assert_eq!(response.goals[0].status, "active");
    assert!(response.procedural_summary.is_some());
    let summary = response.procedural_summary.unwrap();
    assert_eq!(summary.steps.len(), 1);
    assert_eq!(summary.progress_status, "in_progress");
}

// 3. test_parse_compaction_response_minimal
#[test]
fn test_parse_compaction_response_minimal() {
    let json = r#"{
        "facts": [],
        "goals": [],
        "procedural_summary": null
    }"#;

    let response: CompactionResponse = serde_json::from_str(json).unwrap();
    assert!(response.facts.is_empty());
    assert!(response.goals.is_empty());
    assert!(response.procedural_summary.is_none());
}

// 4. test_parse_compaction_response_fenced
#[test]
fn test_parse_compaction_response_fenced() {
    let fenced = r#"```json
{
    "facts": [{"statement": "Bob is tall", "subject": "Bob", "predicate": "is", "object": "tall", "confidence": 0.8}],
    "goals": [],
    "procedural_summary": null
}
```"#;

    let value = parse_json_from_llm(fenced).unwrap();
    let response: CompactionResponse = serde_json::from_value(value).unwrap();
    assert_eq!(response.facts.len(), 1);
    assert_eq!(response.facts[0].statement, "Bob is tall");
}

// 5. test_compaction_to_events_facts
#[test]
fn test_compaction_to_events_facts() {
    use agent_db_events::core::EventType;

    let response = CompactionResponse {
        facts: vec![
            ExtractedFact {
                statement: "Alice lives in Paris".to_string(),
                subject: "Alice".to_string(),
                predicate: "lives_in".to_string(),
                object: "Paris".to_string(),
                confidence: 0.9,
                category: None,
                amount: None,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: None,
                cardinality_hint: None,
                sentiment: None,
                group_id: Default::default(),
                ingest_metadata: Default::default(),
            },
            ExtractedFact {
                statement: "Bob works at Google".to_string(),
                subject: "Bob".to_string(),
                predicate: "works_at".to_string(),
                object: "Google".to_string(),
                confidence: 0.85,
                category: None,
                amount: None,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: None,
                cardinality_hint: None,
                sentiment: None,
                group_id: Default::default(),
                ingest_metadata: Default::default(),
            },
            ExtractedFact {
                statement: "Alice and Bob are friends".to_string(),
                subject: "Alice".to_string(),
                predicate: "friends_with".to_string(),
                object: "Bob".to_string(),
                confidence: 0.75,
                category: None,
                amount: None,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: None,
                cardinality_hint: None,
                sentiment: None,
                group_id: Default::default(),
                ingest_metadata: Default::default(),
            },
        ],
        goals: vec![],
        procedural_summary: None,
    };

    let events = compaction_to_events(&response, "case1", 100, 200, 1000);
    assert_eq!(events.len(), 3);

    for evt in &events {
        assert_eq!(evt.agent_type, "conversation_compaction");
        match &evt.event_type {
            EventType::Observation {
                observation_type,
                source,
                ..
            } => {
                assert_eq!(observation_type, "extracted_fact");
                assert_eq!(source, "conversation_compaction");
            },
            other => panic!("Expected Observation, got {:?}", other),
        }
        assert!(evt.metadata.contains_key("compaction_fact"));
    }
}

// 6. test_compaction_to_events_goals
#[test]
fn test_compaction_to_events_goals() {
    use agent_db_events::core::{CognitiveType, EventType};

    let response = CompactionResponse {
        facts: vec![],
        goals: vec![
            ExtractedGoal {
                description: "Plan trip to Japan".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Learn Rust".to_string(),
                status: "completed".to_string(),
                owner: "user".to_string(),
            },
        ],
        procedural_summary: None,
    };

    let events = compaction_to_events(&response, "case1", 100, 200, 1000);
    assert_eq!(events.len(), 2);

    for evt in &events {
        match &evt.event_type {
            EventType::Cognitive {
                process_type,
                reasoning_trace,
                ..
            } => {
                assert_eq!(*process_type, CognitiveType::GoalFormation);
                assert_eq!(reasoning_trace[0], "LLM conversation compaction");
            },
            other => panic!("Expected Cognitive, got {:?}", other),
        }
        assert!(evt.metadata.contains_key("compaction_goal"));
    }
}

// 7. test_compaction_to_events_steps
#[test]
fn test_compaction_to_events_steps() {
    use agent_db_events::core::{ActionOutcome, EventType};

    let response = CompactionResponse {
        facts: vec![],
        goals: vec![],
        procedural_summary: Some(ProceduralSummary {
            objective: "Deploy app".to_string(),
            progress_status: "completed".to_string(),
            steps: vec![
                ProceduralStep {
                    step_number: 1,
                    action: "Build Docker image".to_string(),
                    result: "Image built successfully".to_string(),
                    outcome: "success".to_string(),
                },
                ProceduralStep {
                    step_number: 2,
                    action: "Push to registry".to_string(),
                    result: "Push failed due to auth".to_string(),
                    outcome: "failure".to_string(),
                },
                ProceduralStep {
                    step_number: 3,
                    action: "Retry with credentials".to_string(),
                    result: "Push succeeded".to_string(),
                    outcome: "success".to_string(),
                },
            ],
            overall_summary: "Deployed app after auth fix".to_string(),
            takeaway: "Always verify registry credentials".to_string(),
        }),
    };

    let events = compaction_to_events(&response, "case1", 100, 200, 1000);
    assert_eq!(events.len(), 3);

    // Check outcome mapping
    match &events[0].event_type {
        EventType::Action {
            action_name,
            outcome,
            ..
        } => {
            assert_eq!(action_name, "step_1");
            assert!(matches!(outcome, ActionOutcome::Success { .. }));
        },
        other => panic!("Expected Action, got {:?}", other),
    }

    match &events[1].event_type {
        EventType::Action { outcome, .. } => {
            assert!(matches!(outcome, ActionOutcome::Failure { .. }));
        },
        other => panic!("Expected Action, got {:?}", other),
    }

    for evt in &events {
        assert!(evt.metadata.contains_key("compaction_step"));
    }
}

// 8. test_compaction_to_events_mixed
#[test]
fn test_compaction_to_events_mixed() {
    let response = CompactionResponse {
        facts: vec![
            ExtractedFact {
                statement: "F1".to_string(),
                subject: "s".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
                confidence: 0.9,
                category: None,
                amount: None,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: None,
                cardinality_hint: None,
                sentiment: None,
                group_id: Default::default(),
                ingest_metadata: Default::default(),
            },
            ExtractedFact {
                statement: "F2".to_string(),
                subject: "s".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
                confidence: 0.8,
                category: None,
                amount: None,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: None,
                cardinality_hint: None,
                sentiment: None,
                group_id: Default::default(),
                ingest_metadata: Default::default(),
            },
        ],
        goals: vec![ExtractedGoal {
            description: "G1".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        }],
        procedural_summary: Some(ProceduralSummary {
            objective: "Obj".to_string(),
            progress_status: "completed".to_string(),
            steps: vec![ProceduralStep {
                step_number: 1,
                action: "A".to_string(),
                result: "R".to_string(),
                outcome: "success".to_string(),
            }],
            overall_summary: "S".to_string(),
            takeaway: "T".to_string(),
        }),
    };

    let events = compaction_to_events(&response, "case1", 100, 200, 1000);
    // 2 facts + 1 goal + 1 step = 4
    assert_eq!(events.len(), 4);

    // Timestamps must be monotonically increasing
    for window in events.windows(2) {
        assert!(
            window[1].timestamp > window[0].timestamp,
            "Timestamps not monotonic: {} <= {}",
            window[1].timestamp,
            window[0].timestamp
        );
    }
}

// 9. test_compaction_to_events_empty
#[test]
fn test_compaction_to_events_empty() {
    let response = CompactionResponse {
        facts: vec![],
        goals: vec![],
        procedural_summary: None,
    };

    let events = compaction_to_events(&response, "case1", 100, 200, 1000);
    assert!(events.is_empty());
}

// 10. test_build_procedural_memory
#[test]
fn test_build_procedural_memory() {
    use crate::episodes::EpisodeOutcome;
    use crate::memory::MemoryTier;

    let summary = ProceduralSummary {
        objective: "Deploy app".to_string(),
        progress_status: "completed".to_string(),
        steps: vec![],
        overall_summary: "Successfully deployed the application".to_string(),
        takeaway: "Always test in staging first".to_string(),
    };

    let memory = build_procedural_memory(&summary, 100, 200, 300);

    assert_eq!(memory.summary, "Successfully deployed the application");
    assert_eq!(memory.takeaway, "Always test in staging first");
    assert_eq!(
        memory.causal_note,
        "Objective: Deploy app. Status: completed"
    );
    assert_eq!(memory.tier, MemoryTier::Episodic);
    assert_eq!(memory.strength, 0.8);
    assert_eq!(memory.outcome, EpisodeOutcome::Success);
    assert_eq!(memory.agent_id, 100);
    assert_eq!(memory.session_id, 200);
    assert_eq!(memory.episode_id, 300);
    assert_eq!(
        memory.metadata.get("source").map(|s| s.as_str()),
        Some("compaction")
    );
    assert_eq!(
        memory.metadata.get("objective").map(|s| s.as_str()),
        Some("Deploy app")
    );
    assert_eq!(
        memory.metadata.get("progress_status").map(|s| s.as_str()),
        Some("completed")
    );
}

// 11. test_build_procedural_memory_outcome_mapping
#[test]
fn test_build_procedural_memory_outcome_mapping() {
    use crate::episodes::EpisodeOutcome;

    assert_eq!(
        map_progress_to_outcome("completed"),
        EpisodeOutcome::Success
    );
    assert_eq!(map_progress_to_outcome("blocked"), EpisodeOutcome::Failure);
    assert_eq!(
        map_progress_to_outcome("abandoned"),
        EpisodeOutcome::Failure
    );
    assert_eq!(
        map_progress_to_outcome("in_progress"),
        EpisodeOutcome::Partial
    );
    assert_eq!(
        map_progress_to_outcome("unknown_status"),
        EpisodeOutcome::Partial
    );
}

// 12. test_compaction_result_has_classifier_fields
#[test]
fn test_compaction_result_has_classifier_fields() {
    let result = CompactionResult::default();
    assert_eq!(result.memories_updated, 0);
    assert_eq!(result.memories_deleted, 0);
    assert_eq!(result.facts_extracted, 0);
    assert!(!result.procedural_memory_created);
    assert!(!result.llm_success);
}

// 13. test_goals_deduplicated_field_default
#[test]
fn test_goals_deduplicated_field_default() {
    let result = CompactionResult::default();
    assert_eq!(result.goals_deduplicated, 0);
}

// 14. test_filter_goals_add_keeps
#[test]
fn test_filter_goals_add_keeps() {
    use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

    let goals = vec![ExtractedGoal {
        description: "Visit Japan".to_string(),
        status: "active".to_string(),
        owner: "user".to_string(),
    }];
    let ops = vec![ClassifiedOperation {
        action: MemoryAction::Add,
        target_index: None,
        new_text: Some("Visit Japan".to_string()),
        fact_text: "Visit Japan".to_string(),
    }];

    let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
    assert_eq!(approved.len(), 1);
    assert_eq!(approved[0].description, "Visit Japan");
    assert_eq!(dedup, 0);
}

// 15. test_filter_goals_none_filters
#[test]
fn test_filter_goals_none_filters() {
    use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

    let goals = vec![ExtractedGoal {
        description: "Visit Japan".to_string(),
        status: "active".to_string(),
        owner: "user".to_string(),
    }];
    let ops = vec![ClassifiedOperation {
        action: MemoryAction::None,
        target_index: None,
        new_text: None,
        fact_text: "Visit Japan".to_string(),
    }];

    let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
    assert!(approved.is_empty());
    assert_eq!(dedup, 1);
}

// 16. test_filter_goals_mixed
#[test]
fn test_filter_goals_mixed() {
    use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

    let goals = vec![
        ExtractedGoal {
            description: "Visit Japan".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
        ExtractedGoal {
            description: "Learn Rust".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
        ExtractedGoal {
            description: "Buy groceries".to_string(),
            status: "completed".to_string(),
            owner: "user".to_string(),
        },
        ExtractedGoal {
            description: "Read a book".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
    ];
    let ops = vec![
        ClassifiedOperation {
            action: MemoryAction::Add,
            target_index: None,
            new_text: Some("Visit Japan".to_string()),
            fact_text: "Visit Japan".to_string(),
        },
        ClassifiedOperation {
            action: MemoryAction::None,
            target_index: None,
            new_text: None,
            fact_text: "Learn Rust".to_string(),
        },
        ClassifiedOperation {
            action: MemoryAction::Delete,
            target_index: Some(0),
            new_text: None,
            fact_text: "Buy groceries".to_string(),
        },
        ClassifiedOperation {
            action: MemoryAction::Update,
            target_index: Some(1),
            new_text: Some("Read a book regularly".to_string()),
            fact_text: "Read a book".to_string(),
        },
    ];

    let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
    // ADD + UPDATE kept, NONE + DELETE filtered
    assert_eq!(approved.len(), 2);
    assert_eq!(approved[0].description, "Visit Japan");
    assert_eq!(approved[1].description, "Read a book");
    assert_eq!(dedup, 2);
}

// 17. test_compaction_to_events_with_filtered_goals
#[test]
fn test_compaction_to_events_with_filtered_goals() {
    use agent_db_events::core::EventType;

    // An empty goals vec produces only fact+step events (no Cognitive)
    let response = CompactionResponse {
        facts: vec![ExtractedFact {
            statement: "F1".to_string(),
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
            confidence: 0.9,
            category: None,
            amount: None,
            split_with: None,
            temporal_signal: None,
            depends_on: None,
            is_update: None,
            cardinality_hint: None,
            sentiment: None,
            ingest_metadata: Default::default(),
            group_id: Default::default(),
        }],
        goals: vec![], // all goals filtered out
        procedural_summary: Some(ProceduralSummary {
            objective: "Obj".to_string(),
            progress_status: "completed".to_string(),
            steps: vec![ProceduralStep {
                step_number: 1,
                action: "A".to_string(),
                result: "R".to_string(),
                outcome: "success".to_string(),
            }],
            overall_summary: "S".to_string(),
            takeaway: "T".to_string(),
        }),
    };

    let events = compaction_to_events(&response, "case1", 100, 200, 1000);
    // 1 fact + 0 goals + 1 step = 2
    assert_eq!(events.len(), 2);

    // No Cognitive events
    for evt in &events {
        assert!(
            !matches!(&evt.event_type, EventType::Cognitive { .. }),
            "Expected no Cognitive events when goals are filtered"
        );
    }
}

// 18. test_fallback_keeps_all_goals
#[test]
fn test_fallback_keeps_all_goals() {
    // When no classification is provided (more goals than ops), extras default to ADD
    use crate::memory_classifier::ClassifiedOperation;

    let goals = vec![
        ExtractedGoal {
            description: "G1".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
        ExtractedGoal {
            description: "G2".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
        ExtractedGoal {
            description: "G3".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
    ];
    // Empty ops — all goals should pass through (fallback to Add)
    let ops: Vec<ClassifiedOperation> = vec![];

    let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
    assert_eq!(approved.len(), 3);
    assert_eq!(dedup, 0);
}

// 19. test_fast_goal_dedup_filters_duplicates
#[test]
fn test_fast_goal_dedup_filters_duplicates() {
    use crate::goal_store::{GoalDedupDecision, GoalStore};

    let mut store = GoalStore::new();

    // First time: all goals are new
    let goals = vec![
        ExtractedGoal {
            description: "Plan trip to Japan".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
        ExtractedGoal {
            description: "Learn Rust programming".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        },
    ];

    let mut new_goals = Vec::new();
    let mut dedup_count = 0usize;
    for goal in &goals {
        match store.store_or_dedup(&goal.description, &goal.status, &goal.owner, "case1") {
            GoalDedupDecision::NewGoal => new_goals.push(goal.clone()),
            GoalDedupDecision::Duplicate { .. } => dedup_count += 1,
            GoalDedupDecision::StatusUpdate {
                existing_id,
                new_status,
            } => {
                store.update_status(existing_id, new_status);
                dedup_count += 1;
            },
        }
    }
    assert_eq!(new_goals.len(), 2);
    assert_eq!(dedup_count, 0);

    // Second time: same goals → all duplicates
    let mut new_goals2 = Vec::new();
    let mut dedup_count2 = 0usize;
    for goal in &goals {
        match store.store_or_dedup(&goal.description, &goal.status, &goal.owner, "case2") {
            GoalDedupDecision::NewGoal => new_goals2.push(goal.clone()),
            GoalDedupDecision::Duplicate { .. } => dedup_count2 += 1,
            GoalDedupDecision::StatusUpdate {
                existing_id,
                new_status,
            } => {
                store.update_status(existing_id, new_status);
                dedup_count2 += 1;
            },
        }
    }
    assert!(new_goals2.is_empty(), "All goals should be deduplicated");
    assert_eq!(dedup_count2, 2);
}

// 20. test_goal_playbook_serde
#[test]
fn test_goal_playbook_serde() {
    let playbook = GoalPlaybook {
        goal_description: "Deploy the app".to_string(),
        what_worked: vec!["Docker build".to_string()],
        what_didnt_work: vec!["Manual deployment".to_string()],
        lessons_learned: vec!["Always use CI/CD".to_string()],
        steps_taken: vec![
            "Build".to_string(),
            "Push".to_string(),
            "Deploy".to_string(),
        ],
        confidence: 0.85,
    };

    let json = serde_json::to_string(&playbook).unwrap();
    let roundtrip: GoalPlaybook = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.goal_description, "Deploy the app");
    assert_eq!(roundtrip.what_worked.len(), 1);
    assert_eq!(roundtrip.what_didnt_work.len(), 1);
    assert_eq!(roundtrip.lessons_learned.len(), 1);
    assert_eq!(roundtrip.steps_taken.len(), 3);
    assert!((roundtrip.confidence - 0.85).abs() < f32::EPSILON);
}

// 21. test_playbook_extraction_response_serde
#[test]
fn test_playbook_extraction_response_serde() {
    let json = r#"{
        "playbooks": [
            {
                "goal_description": "Plan trip",
                "what_worked": ["Booked flights early"],
                "what_didnt_work": ["Waited too long for hotel"],
                "lessons_learned": ["Book everything 2 months ahead"],
                "steps_taken": ["Research", "Book flights", "Find hotel"],
                "confidence": 0.9
            }
        ]
    }"#;

    let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.playbooks.len(), 1);
    assert_eq!(response.playbooks[0].goal_description, "Plan trip");
    assert_eq!(response.playbooks[0].what_worked[0], "Booked flights early");
}

// 22. test_playbook_extraction_response_empty
#[test]
fn test_playbook_extraction_response_empty() {
    let json = r#"{"playbooks": []}"#;
    let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
    assert!(response.playbooks.is_empty());
}

// 23. test_playbook_extraction_response_partial
#[test]
fn test_playbook_extraction_response_partial() {
    // Missing optional fields (steps_taken, confidence) should use defaults
    let json = r#"{
        "playbooks": [
            {
                "goal_description": "Learn Rust",
                "what_worked": ["Read the book"],
                "what_didnt_work": [],
                "lessons_learned": ["Practice daily"]
            }
        ]
    }"#;

    let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.playbooks.len(), 1);
    let pb = &response.playbooks[0];
    assert!(
        pb.steps_taken.is_empty(),
        "steps_taken should default to empty"
    );
    assert!(
        (pb.confidence - 0.5).abs() < f32::EPSILON,
        "confidence should default to 0.5"
    );
}

// 24. test_compaction_result_playbooks_field
#[test]
fn test_compaction_result_playbooks_field() {
    let result = CompactionResult::default();
    assert_eq!(result.playbooks_extracted, 0);
}

// ── Rolling Summary Tests ──

// 25. test_conversation_rolling_summary_serde
#[test]
fn test_conversation_rolling_summary_serde() {
    let summary = ConversationRollingSummary {
        case_id: "case_123".to_string(),
        summary: "User wants to plan a trip to Japan".to_string(),
        last_updated: 1_000_000_000,
        turn_count: 5,
        token_estimate: 42,
    };

    let json = serde_json::to_string(&summary).unwrap();
    let roundtrip: ConversationRollingSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.case_id, "case_123");
    assert_eq!(roundtrip.summary, "User wants to plan a trip to Japan");
    assert_eq!(roundtrip.last_updated, 1_000_000_000);
    assert_eq!(roundtrip.turn_count, 5);
    assert_eq!(roundtrip.token_estimate, 42);
}

// 26. test_format_with_summary
#[test]
fn test_format_with_summary() {
    let summary = ConversationRollingSummary {
        case_id: "test".to_string(),
        summary: "User is planning a trip to Japan in April.".to_string(),
        last_updated: 0,
        turn_count: 3,
        token_estimate: 10,
    };
    let data = make_ingest(vec![
        ("user", "I want to visit Tokyo"),
        ("assistant", "Great choice!"),
        ("user", "What about Kyoto?"),
        ("assistant", "Kyoto is wonderful too"),
        ("user", "Let's add Osaka"),
    ]);

    let result = format_with_summary(&summary, &data, 2);
    assert!(result.contains("[Rolling Summary]"));
    assert!(result.contains("User is planning a trip to Japan in April."));
    assert!(result.contains("[Recent Messages]"));
    // Only last 2 messages
    assert!(result.contains("assistant: Kyoto is wonderful too"));
    assert!(result.contains("user: Let's add Osaka"));
    // Earlier messages should NOT be present
    assert!(!result.contains("I want to visit Tokyo"));
}

// 27. test_format_with_summary_few_messages
#[test]
fn test_format_with_summary_few_messages() {
    let summary = ConversationRollingSummary {
        case_id: "test".to_string(),
        summary: "Summary here.".to_string(),
        last_updated: 0,
        turn_count: 1,
        token_estimate: 5,
    };
    let data = make_ingest(vec![("user", "Hello"), ("assistant", "Hi there")]);

    // recent_count > actual messages: should include all
    let result = format_with_summary(&summary, &data, 10);
    assert!(result.contains("user: Hello"));
    assert!(result.contains("assistant: Hi there"));
}

// 28. test_update_rolling_summary_first_call
#[tokio::test]
async fn test_update_rolling_summary_first_call() {
    use crate::llm_client::LlmResponse;

    // Create a simple mock that returns a summary
    struct SimpleLlm;
    #[async_trait::async_trait]
    impl crate::llm_client::LlmClient for SimpleLlm {
        async fn complete(
            &self,
            req: crate::llm_client::LlmRequest,
        ) -> anyhow::Result<LlmResponse> {
            // First call has no existing summary → prompt starts with "Summarize"
            assert!(req.user_prompt.contains("Summarize this conversation"));
            Ok(LlmResponse {
                content: "User discussed trip plans to Japan.".to_string(),
                tokens_used: 10,
            })
        }
        fn model_name(&self) -> &str {
            "test"
        }
    }

    let messages = vec![crate::conversation::types::ConversationMessage {
        role: "user".to_string(),
        content: "I want to go to Japan".to_string(),
        metadata: Default::default(),
    }];

    let result = update_rolling_summary(&SimpleLlm, None, &messages).await;
    assert!(result.is_some());
    assert_eq!(result.unwrap(), "User discussed trip plans to Japan.");
}
