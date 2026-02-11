//! Scoped Inference Demo
//!
//! Demonstrates how agent_type + session_id scoping enables
//! better relationship inference and pattern detection.

use agent_db_core::types::{current_timestamp, generate_event_id};
use agent_db_events::{
    ActionOutcome, ComputationalResources, EnvironmentState, Event, EventContext, EventType,
    ResourceState, TemporalContext,
};
use agent_db_graph::scoped_inference::{
    InferenceScope, ScopeMetadata, ScopeQuery, ScopedEvent, ScopedInferenceConfig,
    ScopedInferenceEngine,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Agent Type + Session ID Scoped Inference Demo");
    println!("=================================================\n");

    let config = ScopedInferenceConfig::default();
    let engine = ScopedInferenceEngine::new(config).await?;

    // Simulate multiple scenarios with different agent types and sessions
    let scenarios = create_realistic_scenarios().await;

    println!("📝 Processing events from different scopes...\n");

    // Process events from different scopes
    for (scenario_name, events) in scenarios {
        println!("🎯 Scenario: {}", scenario_name);

        for event in events {
            let result = engine.process_scoped_event(event.clone()).await?;

            println!(
                "  ✅ Processed [{}/{}]: {} nodes, {} relationships",
                event.agent_type,
                event.event.session_id,
                result.nodes_created.len(),
                result.relationships_discovered
            );

            if !result.patterns_detected.is_empty() {
                println!("    📊 Patterns: {:?}", result.patterns_detected);
            }

            if !result.cross_scope_patterns.is_empty() {
                println!(
                    "    🔗 Cross-scope patterns: {}",
                    result.cross_scope_patterns.len()
                );
            }
        }
        println!();
    }

    // Demonstrate scope-specific queries
    println!("🔍 Querying specific scopes...\n");

    // Query coding assistant scope
    let coding_scope = InferenceScope {
        agent_type: "coding-assistant".to_string(),
        session_id: 1001,
    };

    let collaboration_result = engine
        .query_scope(&coding_scope, ScopeQuery::AgentCollaboration)
        .await?;
    println!(
        "🤝 Coding Assistant Collaborations: {:?}",
        collaboration_result
    );

    let workflow_result = engine
        .query_scope(&coding_scope, ScopeQuery::WorkflowSequences)
        .await?;
    println!("⚡ Coding Workflows: {:?}", workflow_result);

    // Query data analyst scope
    let analyst_scope = InferenceScope {
        agent_type: "data-analyst".to_string(),
        session_id: 2001,
    };

    let metrics_result = engine
        .query_scope(&analyst_scope, ScopeQuery::SessionMetrics)
        .await?;
    println!("📈 Data Analysis Metrics: {:?}", metrics_result);

    // Cross-scope insights
    println!("\n🌐 Cross-Scope Insights:");
    let insights = engine.get_cross_scope_insights().await;
    println!(
        "  Common Agent Patterns: {} types",
        insights.common_agent_patterns.len()
    );
    println!(
        "  Session Similarities: {} pairs",
        insights.session_similarities.len()
    );
    println!(
        "  Global Workflows: {} patterns",
        insights.global_workflows.len()
    );

    // Scope statistics
    println!("\n📊 Scope Statistics:");
    let stats = engine.get_scope_statistics().await;
    println!("  Total Scopes: {}", stats.total_scopes);
    println!("  Total Sessions: {}", stats.total_sessions);
    println!("  Total Agents: {}", stats.total_agents);

    for (agent_type, type_stats) in &stats.stats_by_agent_type {
        println!(
            "  {}: {} sessions, {} agents, {} events",
            agent_type, type_stats.session_count, type_stats.unique_agents, type_stats.total_events
        );
    }

    if let Some(ref most_active) = stats.most_active_agent_type {
        println!("  Most Active Agent Type: {}", most_active);
    }

    println!("\n✅ Scoped inference demo completed successfully!");
    println!("   The system correctly grouped and analyzed relationships within logical scopes!");

    Ok(())
}

async fn create_realistic_scenarios() -> Vec<(String, Vec<ScopedEvent>)> {
    let base_context = create_context();

    vec![
        // Scenario 1: Coding Assistant Session - Feature Development
        (
            "Feature Development Team".to_string(),
            create_coding_session_events(&base_context).await,
        ),
        // Scenario 2: Data Analysis Session - Report Generation
        (
            "Data Analysis Project".to_string(),
            create_data_analysis_events(&base_context).await,
        ),
        // Scenario 3: Task Management Session - Project Coordination
        (
            "Project Management".to_string(),
            create_task_management_events(&base_context).await,
        ),
        // Scenario 4: Different Coding Session - Bug Fixing
        (
            "Bug Fix Session".to_string(),
            create_bug_fix_events(&base_context).await,
        ),
    ]
}

async fn create_coding_session_events(context: &EventContext) -> Vec<ScopedEvent> {
    let base_time = current_timestamp();
    let session_id = 1001;

    vec![
        // Agent 1: Senior Developer
        ScopedEvent {
            event: Event {
                id: generate_event_id(),
                timestamp: base_time,
                agent_id: 101,
                agent_type: "coding-assistant".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: "analyze_requirements".to_string(),
                    parameters: serde_json::json!({"feature": "user_authentication", "complexity": "medium"}),
                    outcome: ActionOutcome::Success {
                        result: serde_json::json!("Requirements analyzed"),
                    },
                    duration_ns: 2_000_000,
                },
                causality_chain: Vec::new(),
                context: context.clone(),
                metadata: std::collections::HashMap::new(),
                context_size_bytes: 0,
                segment_pointer: None,
            },
            agent_type: "coding-assistant".to_string(),
            priority: 1.0,
            scope_metadata: ScopeMetadata {
                workspace_id: Some("project_alpha".to_string()),
                user_id: Some("dev_alice".to_string()),
                environment: Some("development".to_string()),
                tags: vec!["frontend".to_string(), "authentication".to_string()],
            },
        },
        // Agent 2: Junior Developer (collaborating)
        ScopedEvent {
            event: Event {
                id: generate_event_id(),
                timestamp: base_time + 500_000_000,
                agent_id: 102,
                agent_type: "coding-assistant".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: "implement_feature".to_string(),
                    parameters: serde_json::json!({"feature": "user_authentication", "approach": "oauth2"}),
                    outcome: ActionOutcome::Success {
                        result: serde_json::json!("Feature implemented"),
                    },
                    duration_ns: 5_000_000,
                },
                causality_chain: Vec::new(),
                context: context.clone(),
                metadata: std::collections::HashMap::new(),
                context_size_bytes: 0,
                segment_pointer: None,
            },
            agent_type: "coding-assistant".to_string(),
            priority: 0.8,
            scope_metadata: ScopeMetadata {
                workspace_id: Some("project_alpha".to_string()),
                user_id: Some("dev_bob".to_string()),
                environment: Some("development".to_string()),
                tags: vec!["backend".to_string(), "security".to_string()],
            },
        },
        // Agent 3: Code Reviewer
        ScopedEvent {
            event: Event {
                id: generate_event_id(),
                timestamp: base_time + 1_000_000_000,
                agent_id: 103,
                agent_type: "coding-assistant".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: "code_review".to_string(),
                    parameters: serde_json::json!({"pull_request": "PR_123", "focus": "security"}),
                    outcome: ActionOutcome::Success {
                        result: serde_json::json!("Review completed"),
                    },
                    duration_ns: 3_000_000,
                },
                causality_chain: Vec::new(),
                context: context.clone(),
                metadata: std::collections::HashMap::new(),
                context_size_bytes: 0,
                segment_pointer: None,
            },
            agent_type: "coding-assistant".to_string(),
            priority: 1.0,
            scope_metadata: ScopeMetadata {
                workspace_id: Some("project_alpha".to_string()),
                user_id: Some("dev_charlie".to_string()),
                environment: Some("development".to_string()),
                tags: vec!["review".to_string(), "security".to_string()],
            },
        },
    ]
}

async fn create_data_analysis_events(context: &EventContext) -> Vec<ScopedEvent> {
    let base_time = current_timestamp();
    let session_id = 2001;

    vec![
        ScopedEvent {
            event: Event {
                id: generate_event_id(),
                timestamp: base_time,
                agent_id: 201,
                agent_type: "data-analyst".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: "load_dataset".to_string(),
                    parameters: serde_json::json!({"source": "sales_data.csv", "size_mb": 150}),
                    outcome: ActionOutcome::Success {
                        result: serde_json::json!("Dataset loaded"),
                    },
                    duration_ns: 1_000_000,
                },
                causality_chain: Vec::new(),
                context: context.clone(),
                metadata: std::collections::HashMap::new(),
                context_size_bytes: 0,
                segment_pointer: None,
            },
            agent_type: "data-analyst".to_string(),
            priority: 1.0,
            scope_metadata: ScopeMetadata {
                workspace_id: Some("analytics_workspace".to_string()),
                user_id: Some("analyst_diana".to_string()),
                environment: Some("production".to_string()),
                tags: vec!["sales".to_string(), "quarterly_report".to_string()],
            },
        },
        ScopedEvent {
            event: Event {
                id: generate_event_id(),
                timestamp: base_time + 800_000_000,
                agent_id: 202,
                agent_type: "data-analyst".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: "generate_insights".to_string(),
                    parameters: serde_json::json!({"analysis_type": "trend_analysis", "period": "Q3_2024"}),
                    outcome: ActionOutcome::Success {
                        result: serde_json::json!("Insights generated"),
                    },
                    duration_ns: 4_000_000,
                },
                causality_chain: Vec::new(),
                context: context.clone(),
                metadata: std::collections::HashMap::new(),
                context_size_bytes: 0,
                segment_pointer: None,
            },
            agent_type: "data-analyst".to_string(),
            priority: 0.9,
            scope_metadata: ScopeMetadata {
                workspace_id: Some("analytics_workspace".to_string()),
                user_id: Some("analyst_eve".to_string()),
                environment: Some("production".to_string()),
                tags: vec!["trends".to_string(), "insights".to_string()],
            },
        },
    ]
}

async fn create_task_management_events(context: &EventContext) -> Vec<ScopedEvent> {
    let base_time = current_timestamp();
    let session_id = 3001;

    vec![ScopedEvent {
        event: Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: 301,
            agent_type: "task-manager".to_string(),
            session_id,
            event_type: EventType::Action {
                action_name: "create_milestone".to_string(),
                parameters: serde_json::json!({"project": "Q4_goals", "deadline": "2024-12-31"}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Milestone created"),
                },
                duration_ns: 500_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        agent_type: "task-manager".to_string(),
        priority: 1.0,
        scope_metadata: ScopeMetadata {
            workspace_id: Some("company_wide".to_string()),
            user_id: Some("pm_frank".to_string()),
            environment: Some("production".to_string()),
            tags: vec!["planning".to_string(), "q4".to_string()],
        },
    }]
}

async fn create_bug_fix_events(context: &EventContext) -> Vec<ScopedEvent> {
    let base_time = current_timestamp();
    let session_id = 1002; // Different session, same agent type

    vec![ScopedEvent {
        event: Event {
            id: generate_event_id(),
            timestamp: base_time,
            agent_id: 104,
            agent_type: "coding-assistant".to_string(), // Same type, different session
            session_id,
            event_type: EventType::Action {
                action_name: "debug_issue".to_string(),
                parameters: serde_json::json!({"bug_id": "BUG_456", "severity": "high"}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!("Bug identified"),
                },
                duration_ns: 2_500_000,
            },
            causality_chain: Vec::new(),
            context: context.clone(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        },
        agent_type: "coding-assistant".to_string(),
        priority: 1.0,
        scope_metadata: ScopeMetadata {
            workspace_id: Some("project_beta".to_string()),
            user_id: Some("dev_grace".to_string()),
            environment: Some("staging".to_string()),
            tags: vec!["bugfix".to_string(), "urgent".to_string()],
        },
    }]
}

fn create_context() -> EventContext {
    EventContext {
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
                cpu_percent: 60.0,
                memory_bytes: 4 * 1024 * 1024 * 1024,    // 4GB
                storage_bytes: 100 * 1024 * 1024 * 1024, // 100GB
                network_bandwidth: 1000 * 1000 * 10,     // 10Mbps
            },
            external: std::collections::HashMap::new(),
        },
        fingerprint: 98765,
        goal_bucket_id: 0,
        embeddings: None,
    }
}
