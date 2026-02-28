//! Conversation ingestion and query handlers.
//!
//! ## POST /api/conversations/ingest
//!
//! Ingest one or more conversation sessions into structured memory.
//! Messages are classified (transaction, state change, relationship, preference,
//! or chitchat) and bridged into the appropriate structured memory templates
//! (ledgers, state machines, trees, preference lists).
//!
//! When a unified LLM client is configured, messages are classified by the LLM
//! with Rust-side validation of numbers and currencies. Without an LLM, a
//! keyword-based classifier and parser pipeline is used as fallback.
//!
//! ### Request body
//!
//! ```json
//! {
//!   "case_id": "optional-string",       // auto-generated if omitted
//!   "sessions": [
//!     {
//!       "session_id": "session_01",
//!       "topic": "optional topic label",
//!       "messages": [
//!         { "role": "user",      "content": "Alice: Paid €50 for lunch - split with Bob" },
//!         { "role": "assistant", "content": "Got it!" }
//!       ],
//!       "contains_fact": false,          // benchmark metadata (optional)
//!       "fact_id": null,                 // benchmark metadata (optional)
//!       "fact_quote": null               // benchmark metadata (optional)
//!     }
//!   ],
//!   "include_assistant_facts": false     // if true, also extract facts from assistant messages
//! }
//! ```
//!
//! ### Response body
//!
//! ```json
//! {
//!   "case_id": "case_abc123",
//!   "messages_processed": 42,
//!   "transactions_found": 12,
//!   "state_changes_found": 3,
//!   "relationships_found": 5,
//!   "preferences_found": 8,
//!   "chitchat_skipped": 14
//! }
//! ```
//!
//! ## POST /api/conversations/query
//!
//! Query the structured memory populated by conversation ingestion.
//! First attempts conversation-specific classification (numeric/balance,
//! state, entity summary, preference, relationship path). Falls back to the
//! general NLQ pipeline if no conversation pattern matches.
//!
//! ### Request body
//!
//! ```json
//! {
//!   "question": "Who owes whom?",
//!   "session_id": "optional-session-for-nlq-context"
//! }
//! ```
//!
//! ### Response body
//!
//! ```json
//! {
//!   "answer": "Settlement: Alice -> Bob : 172.50 EUR, Charlie -> Bob : 60.00 EUR",
//!   "query_type": "numeric"
//! }
//! ```
//!
//! Query types: `"numeric"`, `"state"`, `"entity_summary"`, `"preference"`,
//! `"relationship"`, `"nlq"` (fallback).

use crate::errors::ApiError;
use crate::state::AppState;
use crate::write_lanes::WriteJob;
use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tracing::info;

// ---------------------------------------------------------------------------
// Request / Response models
// ---------------------------------------------------------------------------

/// Ingest request for `POST /api/conversations/ingest`.
#[derive(Debug, Deserialize)]
pub struct ConversationIngestRequest {
    /// Optional case identifier; auto-generated if omitted.
    #[serde(default)]
    pub case_id: Option<String>,
    /// One or more conversation sessions to ingest.
    pub sessions: Vec<SessionInput>,
    /// Whether to process assistant messages for facts.
    #[serde(default)]
    pub include_assistant_facts: bool,
}

/// A single conversation session within an ingest request.
#[derive(Debug, Deserialize)]
pub struct SessionInput {
    /// Unique session identifier.
    pub session_id: String,
    /// Optional topic label for context.
    #[serde(default)]
    pub topic: Option<String>,
    /// Ordered messages in the session.
    pub messages: Vec<MessageInput>,
    /// Benchmark metadata: whether this session contains a key fact.
    #[serde(default)]
    pub contains_fact: Option<bool>,
    /// Benchmark metadata: identifier of the fact.
    #[serde(default)]
    pub fact_id: Option<String>,
    /// Benchmark metadata: exact quote of the fact.
    #[serde(default)]
    pub fact_quote: Option<String>,
}

/// A single message in a conversation session.
#[derive(Debug, Deserialize)]
pub struct MessageInput {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// Query request for `POST /api/conversations/query`.
#[derive(Debug, Deserialize)]
pub struct ConversationQueryRequest {
    /// The question to answer against structured memory.
    pub question: String,
    /// Optional session ID for NLQ conversational context.
    #[serde(default)]
    pub session_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Maximum number of conversation states to keep in memory.
const MAX_CONVERSATION_STATES: usize = 1000;

/// POST /api/conversations/ingest — ingest conversation sessions.
pub async fn ingest_conversation(
    State(state): State<AppState>,
    Json(request): Json<ConversationIngestRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let case_id = request.case_id.clone().unwrap_or_else(uuid_v4_simple);

    info!("Ingesting conversation case_id={}", case_id);

    // Convert request into domain types (pure data transform, no engine access)
    let ingest_data = agent_db_graph::ConversationIngest {
        case_id: Some(case_id.clone()),
        sessions: request
            .sessions
            .into_iter()
            .map(|s| agent_db_graph::ConversationSession {
                session_id: s.session_id,
                topic: s.topic,
                messages: s
                    .messages
                    .into_iter()
                    .map(|m| agent_db_graph::ConversationMessage {
                        role: m.role,
                        content: m.content,
                    })
                    .collect(),
                contains_fact: s.contains_fact,
                fact_id: s.fact_id,
                fact_quote: s.fact_quote,
                answers: vec![],
            })
            .collect(),
        queries: vec![],
    };

    let options = agent_db_graph::IngestOptions {
        include_assistant_facts: request.include_assistant_facts,
        ..Default::default()
    };

    // 1. Convert conversation to Event structs via bridge (pure CPU, no engine)
    let (events, conv_state, result) =
        agent_db_graph::conversation::ingest_to_events(&ingest_data, &options);

    info!(
        "Bridge produced {} events for case_id={}",
        events.len(),
        case_id
    );

    // Use a hash of the case_id as routing key for write lane
    let routing_key = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        case_id.hash(&mut hasher);
        hasher.finish()
    };

    // Submit the entire write-side logic to a write lane
    let case_id_for_closure = case_id.clone();
    let ingest_data_clone = ingest_data.clone();
    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let case_id = case_id_for_closure;

                        // 2. Pre-create Concept nodes for all participants
                        let participants: Vec<String> =
                            conv_state.known_participants.iter().cloned().collect();
                        if let Err(e) =
                            engine.ensure_conversation_participants(&participants).await
                        {
                            tracing::info!("Failed to pre-create participants: {}", e);
                        }

                        // 3. Submit each event through the full pipeline
                        let mut events_processed = 0usize;
                        for event in events {
                            match engine
                                .process_event_with_options(event, Some(false))
                                .await
                            {
                                Ok(_) => events_processed += 1,
                                Err(e) => {
                                    tracing::debug!("Event pipeline error: {}", e);
                                }
                            }
                        }

                        // 4. Store conversation state + LRU eviction
                        {
                            let mut states = engine.conversation_states().lock().await;
                            states.insert(case_id.clone(), conv_state);

                            if states.len() > MAX_CONVERSATION_STATES {
                                if let Some(evict_key) = states
                                    .iter()
                                    .min_by_key(|(_, s)| s.processed_messages.len())
                                    .map(|(k, _)| k.clone())
                                {
                                    states.remove(&evict_key);
                                }
                            }
                        }

                        // 5. Fire-and-forget: rolling summary update (if enabled)
                        if engine.config.enable_rolling_summary {
                            let engine_rs = engine.clone();
                            let case_id_clone = case_id.clone();
                            let new_messages: Vec<agent_db_graph::ConversationMessage> =
                                ingest_data_clone
                                    .sessions
                                    .iter()
                                    .flat_map(|s| s.messages.clone())
                                    .collect();
                            tokio::spawn(async move {
                                let existing =
                                    engine_rs.conversation_summary(&case_id_clone).await;
                                let existing_text =
                                    existing.as_ref().map(|s| s.summary.as_str());
                                let prev_turn_count =
                                    existing.as_ref().map(|s| s.turn_count).unwrap_or(0);

                                if let Some(llm) = engine_rs.unified_llm_client() {
                                    if let Some(updated_text) =
                                        agent_db_graph::conversation::update_rolling_summary(
                                            llm.as_ref(),
                                            existing_text,
                                            &new_messages,
                                        )
                                        .await
                                    {
                                        let now = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_nanos()
                                            as u64;
                                        let token_est = (updated_text.len() / 4) as u32;
                                        let summary =
                                            agent_db_graph::ConversationRollingSummary {
                                                case_id: case_id_clone.clone(),
                                                summary: updated_text,
                                                last_updated: now,
                                                turn_count: prev_turn_count
                                                    + new_messages.len() as u32,
                                                token_estimate: token_est,
                                            };
                                        engine_rs.set_conversation_summary(summary).await;
                                        tracing::info!(
                                            "Rolling summary updated for case_id={}",
                                            case_id_clone,
                                        );
                                    }
                                }
                            });
                        }

                        // 6. Fire-and-forget: LLM compaction (if enabled)
                        let compaction_started =
                            if engine.config.enable_conversation_compaction {
                                let engine_cmp = engine.clone();
                                let ingest_cmp = ingest_data_clone.clone();
                                let case_id_cmp = case_id.clone();
                                tokio::spawn(async move {
                                    let result =
                                        agent_db_graph::conversation::run_compaction(
                                            &engine_cmp,
                                            &ingest_cmp,
                                            &case_id_cmp,
                                        )
                                        .await;
                                    tracing::info!(
                                        "Compaction case_id={}: facts={} goals={} goals_deduped={} steps={} memory={} updated={} deleted={} playbooks={}",
                                        case_id_cmp,
                                        result.facts_extracted,
                                        result.goals_extracted,
                                        result.goals_deduplicated,
                                        result.procedural_steps_extracted,
                                        result.procedural_memory_created,
                                        result.memories_updated,
                                        result.memories_deleted,
                                        result.playbooks_extracted,
                                    );
                                });
                                true
                            } else {
                                false
                            };

                        let rolling_summary_started = engine.config.enable_rolling_summary;

                        Ok(json!({
                            "case_id": result.case_id,
                            "messages_processed": result.messages_processed,
                            "transactions_found": result.transactions_found,
                            "state_changes_found": result.state_changes_found,
                            "relationships_found": result.relationships_found,
                            "preferences_found": result.preferences_found,
                            "chitchat_skipped": result.chitchat_skipped,
                            "events_submitted": events_processed,
                            "compaction_started": compaction_started,
                            "rolling_summary_started": rolling_summary_started,
                        }))
                    })
                }),
                result_tx: tx,
            })
        .await?;

    Ok(Json(result))
}

/// POST /api/conversations/query — query with conversation context.
pub async fn query_conversation(
    State(state): State<AppState>,
    Json(request): Json<ConversationQueryRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    info!("Conversation query: {}", request.question);

    // Retrieve related memories and strategies for context enrichment (BM25-only)
    let related_memories = retrieve_related_memories(&state, &request.question).await;
    let related_strategies = retrieve_related_strategies(&state, &request.question).await;

    // First try conversation-specific query classification
    let sm_guard = state.engine.structured_memory().read().await;

    if let Some(cq) = agent_db_graph::conversation::classify_conversation_query(&request.question) {
        let registry = build_registry_from_store(&sm_guard);

        let answer = agent_db_graph::conversation::execute_conversation_query(
            &cq,
            &sm_guard,
            &registry,
            &request.question,
        );
        let memory_context =
            agent_db_graph::conversation::gather_memory_context(&cq, &sm_guard, &registry);
        let query_type = match &cq {
            agent_db_graph::conversation::ConversationQueryType::Numeric { .. } => "numeric",
            agent_db_graph::conversation::ConversationQueryType::State { .. } => "state",
            agent_db_graph::conversation::ConversationQueryType::EntitySummary { .. } => {
                "entity_summary"
            },
            agent_db_graph::conversation::ConversationQueryType::Preference { .. } => "preference",
            agent_db_graph::conversation::ConversationQueryType::RelationshipPath { .. } => {
                "relationship"
            },
        };
        return Ok(Json(json!({
            "answer": answer,
            "query_type": query_type,
            "memory_context": memory_context,
            "related_memories": related_memories,
            "related_strategies": related_strategies,
        })));
    }

    drop(sm_guard);

    // Fall back to general NLQ pipeline
    let pagination = agent_db_graph::nlq::NlqPagination::default();
    match state
        .engine
        .natural_language_query(
            &request.question,
            &pagination,
            request.session_id.as_deref(),
        )
        .await
    {
        Ok(response) => Ok(Json(json!({
            "answer": response.answer,
            "query_type": "nlq",
            "related_memories": related_memories,
            "related_strategies": related_strategies,
        }))),
        Err(e) => Err(ApiError::Internal(format!("NLQ error: {}", e))),
    }
}

/// Retrieve related memories via BM25-only multi-signal retrieval.
async fn retrieve_related_memories(
    state: &AppState,
    question: &str,
) -> Vec<agent_db_graph::MemorySummary> {
    let query = agent_db_graph::MemoryRetrievalQuery {
        query_text: question.to_string(),
        query_embedding: vec![],
        context: None,
        anchor_node: None,
        agent_id: None,
        session_id: None,
        now: None,
        limit: 5,
    };
    let memories = state
        .engine
        .retrieve_memories_multi_signal(query, None)
        .await;
    memories
        .iter()
        .map(agent_db_graph::MemorySummary::from_memory)
        .collect()
}

/// Retrieve related strategies via BM25-only multi-signal retrieval.
async fn retrieve_related_strategies(
    state: &AppState,
    question: &str,
) -> Vec<agent_db_graph::StrategySummary> {
    let query = agent_db_graph::StrategyRetrievalQuery {
        query_text: question.to_string(),
        query_embedding: vec![],
        anchor_node: None,
        now: None,
        limit: 3,
    };
    let strategies = state
        .engine
        .retrieve_strategies_multi_signal(query, None)
        .await;
    strategies
        .iter()
        .map(agent_db_graph::StrategySummary::from_strategy)
        .collect()
}

/// Build a minimal NameRegistry from the structured memory store.
fn build_registry_from_store(
    store: &agent_db_graph::StructuredMemoryStore,
) -> agent_db_graph::NameRegistry {
    let mut registry = agent_db_graph::NameRegistry::new();

    // Extract names from ledger entity_pairs
    for key in store.list_keys("ledger:") {
        if let Some(agent_db_graph::MemoryTemplate::Ledger { entity_pair, .. }) = store.get(key) {
            registry.get_or_create(&entity_pair.0);
            registry.get_or_create(&entity_pair.1);
        }
    }

    // Extract entity names from state machines
    for key in store.list_keys("state:") {
        if let Some(agent_db_graph::MemoryTemplate::StateMachine { entity, .. }) = store.get(key) {
            registry.get_or_create(entity);
        }
    }

    // Extract entity names from preference lists
    for key in store.list_keys("prefs:") {
        if let Some(agent_db_graph::MemoryTemplate::PreferenceList { entity, .. }) = store.get(key)
        {
            registry.get_or_create(entity);
        }
    }

    registry
}

/// Generate a simple pseudo-UUID (no external dependency).
fn uuid_v4_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("case_{:x}", t)
}
