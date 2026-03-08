//! Conversation ingestion handler.
//!
//! ## POST /api/conversations/ingest
//!
//! Ingest one or more conversation sessions into the graph.
//! Messages are stored as raw Conversation events, then LLM compaction
//! extracts facts (subject/predicate/object triples), goals, and procedural
//! summaries. The extracted facts are fed back through the pipeline to create
//! Concept nodes and graph edges.
//!
//! **Requires a configured LLM client.** Without it, the system cannot extract
//! any structured information from conversations and will return an error.
//!
//! For querying, use `POST /api/nlq` which provides the unified multi-source
//! pipeline (BM25 + memory + claims + graph entity resolution with RRF fusion).

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
    // Hard fail: LLM is required for conversation ingest.
    // Without it, no facts can be extracted and the graph would contain only
    // raw Conversation events with no structure.
    if state.engine.unified_llm_client().is_none() {
        return Err(ApiError::BadRequest(
            "LLM client is required for conversation ingest. \
             Configure an OpenAI-compatible API key (OPENAI_API_KEY or LLM_API_KEY) \
             and restart the server."
                .to_string(),
        ));
    }

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
                        // Disable per-event claim extraction (semantic=false) because
                        // LLM compaction handles all fact extraction. Running both would
                        // flood the LLM API with redundant calls and cause rate-limit
                        // failures that make compaction drop extracted facts.
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

                        // 6. LLM compaction: extract facts, goals, procedural
                        //    summaries and feed them back through the pipeline.
                        //    Runs INLINE so extracted data is in the graph before
                        //    the response returns.
                        let compaction_result =
                            if engine.config.enable_conversation_compaction {
                                let cr = agent_db_graph::conversation::run_compaction(
                                    &engine,
                                    &ingest_data_clone,
                                    &case_id,
                                )
                                .await;
                                tracing::info!(
                                    "Compaction case_id={}: facts={} goals={} goals_deduped={} steps={} memory={} updated={} deleted={} playbooks={}",
                                    case_id,
                                    cr.facts_extracted,
                                    cr.goals_extracted,
                                    cr.goals_deduplicated,
                                    cr.procedural_steps_extracted,
                                    cr.procedural_memory_created,
                                    cr.memories_updated,
                                    cr.memories_deleted,
                                    cr.playbooks_extracted,
                                );
                                Some(cr)
                            } else {
                                None
                            };

                        let rolling_summary_started = engine.config.enable_rolling_summary;

                        Ok(json!({
                            "case_id": result.case_id,
                            "messages_processed": result.messages_processed,
                            "events_submitted": events_processed,
                            "compaction": compaction_result.as_ref().map(|cr| json!({
                                "facts_extracted": cr.facts_extracted,
                                "goals_extracted": cr.goals_extracted,
                                "goals_deduplicated": cr.goals_deduplicated,
                                "procedural_steps": cr.procedural_steps_extracted,
                                "memories_created": cr.procedural_memory_created,
                                "memories_updated": cr.memories_updated,
                                "memories_deleted": cr.memories_deleted,
                                "playbooks_extracted": cr.playbooks_extracted,
                                "llm_success": cr.llm_success,
                            })),
                            "rolling_summary_started": rolling_summary_started,
                        }))
                    })
                }),
                result_tx: tx,
            })
        .await?;

    Ok(Json(result))
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
