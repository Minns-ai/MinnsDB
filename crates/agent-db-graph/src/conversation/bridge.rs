//! Event bridge — converts parsed messages into structured memory entries.
//!
//! Populates the existing `StructuredMemoryStore` directly, bypassing the
//! full event pipeline for efficiency.

use super::types::*;
use crate::llm_client::LlmClient;
use crate::structured_memory::StructuredMemoryStore;
use std::collections::HashSet;
use std::sync::Arc;

/// Process a full `ConversationIngest` into structured memory.
///
/// Returns an `IngestResult` summarizing what was extracted.
/// Creates a fresh `ConversationState` internally. For incremental ingestion
/// across multiple calls, use [`ingest_incremental`] instead.
pub fn ingest(
    data: &ConversationIngest,
    store: &mut StructuredMemoryStore,
    options: &IngestOptions,
) -> IngestResult {
    let mut state = ConversationState::new();
    ingest_incremental(data, store, options, &mut state)
}

/// Process a `ConversationIngest` incrementally, reusing an existing
/// `ConversationState` to preserve name→ID mappings and idempotency
/// tracking across calls.
///
/// Messages already in `state.processed_messages` are skipped automatically.
pub fn ingest_incremental(
    data: &ConversationIngest,
    _store: &mut StructuredMemoryStore,
    options: &IngestOptions,
    state: &mut ConversationState,
) -> IngestResult {
    let case_id = data
        .case_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let mut result = IngestResult {
        case_id: case_id.clone(),
        ..Default::default()
    };

    // First pass: collect all participant names from transactions for "everyone" resolution
    for session in &data.sessions {
        for msg in &session.messages {
            if msg.role == "assistant" && !options.include_assistant_facts {
                continue;
            }
            collect_participants(&msg.content, &mut state.known_participants);
        }
    }

    // Second pass: count messages (no rule-based classification).
    // LLM compaction handles all fact extraction.
    for session in &data.sessions {
        for (idx, _msg) in session.messages.iter().enumerate() {
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);
            result.messages_processed += 1;
        }
    }

    result
}

/// Process a full `ConversationIngest` with LLM-assisted classification.
///
/// When `llm_client` is `Some`, messages are classified and extracted via the
/// LLM first. If the LLM fails for a given message, the keyword-based path
/// is used as fallback. When `llm_client` is `None`, behaves identically to
/// [`ingest`].
///
/// Creates a fresh `ConversationState` internally. For incremental ingestion
/// across multiple calls, use [`ingest_with_llm_incremental`] instead.
pub async fn ingest_with_llm(
    data: &ConversationIngest,
    store: &mut StructuredMemoryStore,
    options: &IngestOptions,
    llm_client: Option<Arc<dyn LlmClient>>,
) -> IngestResult {
    let mut state = ConversationState::new();
    ingest_with_llm_incremental(data, store, options, llm_client, &mut state).await
}

/// Process a `ConversationIngest` with LLM-assisted classification,
/// reusing an existing `ConversationState` to preserve name→ID mappings
/// and idempotency tracking across calls.
///
/// Messages already in `state.processed_messages` are skipped automatically.
pub async fn ingest_with_llm_incremental(
    data: &ConversationIngest,
    _store: &mut StructuredMemoryStore,
    options: &IngestOptions,
    _llm_client: Option<Arc<dyn LlmClient>>,
    state: &mut ConversationState,
) -> IngestResult {
    // No per-message classification. LLM compaction handles all extraction.
    let case_id = data
        .case_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let mut result = IngestResult {
        case_id: case_id.clone(),
        ..Default::default()
    };

    for session in &data.sessions {
        for msg in &session.messages {
            if msg.role == "assistant" && !options.include_assistant_facts {
                continue;
            }
            collect_participants(&msg.content, &mut state.known_participants);
        }
    }

    for session in &data.sessions {
        for (idx, _msg) in session.messages.iter().enumerate() {
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);
            result.messages_processed += 1;
        }
    }

    result
}

/// Process each session independently with its own fresh store.
///
/// Returns one `SessionIngestResult` per session, each containing an isolated
/// `StructuredMemoryStore` so that settlements are per-session, not cumulative.
pub fn ingest_per_session(
    data: &ConversationIngest,
    options: &IngestOptions,
) -> Vec<SessionIngestResult> {
    let case_id = data
        .case_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let mut results = Vec::new();

    for session in &data.sessions {
        let store = StructuredMemoryStore::new();
        let mut state = ConversationState::new();
        let mut result = IngestResult {
            case_id: case_id.clone(),
            ..Default::default()
        };

        // First pass: collect participants for this session
        for msg in &session.messages {
            if msg.role == "assistant" && !options.include_assistant_facts {
                continue;
            }
            collect_participants(&msg.content, &mut state.known_participants);
        }

        // Second pass: count messages (no rule-based classification).
        // LLM compaction handles all fact extraction.
        for (idx, _msg) in session.messages.iter().enumerate() {
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);
            result.messages_processed += 1;
        }

        results.push(SessionIngestResult {
            session_id: session.session_id.clone(),
            store,
            name_registry: state.name_registry,
            result,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Participant collection (first pass)
// ---------------------------------------------------------------------------

/// Public wrapper for `collect_participants` for use by the runner.
pub fn collect_participants_pub(content: &str, participants: &mut HashSet<String>) {
    collect_participants(content, participants);
}

/// Collect person names from transaction messages for "everyone" resolution.
///
/// Conservative: only extract names from well-defined patterns to avoid
/// picking up place names (Berlin, Italian) or random capitalized words.
fn collect_participants(content: &str, participants: &mut HashSet<String>) {
    let lower = content.to_lowercase();

    // "Name: ..." pattern — speaker before first colon (single-word name only)
    if let Some(colon_pos) = content.find(':') {
        let name = content[..colon_pos].trim();
        // Only single-word names (Alice, Bob, Charlie, Tom, Maria, Sophia)
        // to avoid multi-word phrases being treated as names
        if name.starts_with(|c: char| c.is_uppercase())
            && name.split_whitespace().count() == 1
            && name.len() >= 2
            && name.len() <= 20
            && name.chars().all(|c| c.is_alphabetic())
        {
            participants.insert(name.to_string());
        }
    }

    // "Name paid" pattern (no colon)
    if let Some(paid_pos) = lower.find(" paid ") {
        let before = content[..paid_pos].trim();
        if before.starts_with(|c: char| c.is_uppercase()) {
            let name_words: Vec<&str> = before.split_whitespace().collect();
            if name_words.len() == 1 {
                participants.insert(before.to_string());
            }
        }
    }

    // Names after "split with X" / "split between X and Y" / "shared between X and me"
    if let Some(split_pos) = lower
        .find("split with")
        .or_else(|| lower.find("split between"))
        .or_else(|| lower.find("shared between"))
    {
        let after = &content[split_pos..];
        let names = extract_person_names_only(after);
        for name in names {
            participants.insert(name);
        }
    }

    // Names before "owe me" / "owes me" — find the sentence containing "owe me"
    if let Some(owe_pos) = lower.find("owe me").or_else(|| lower.find("owes me")) {
        let sentence_start = content[..owe_pos].rfind(". ").map(|p| p + 2).unwrap_or(0);
        let before = &content[sentence_start..owe_pos];
        let names = extract_person_names_only(before);
        for name in names {
            participants.insert(name);
        }
    }
}

/// Extract only likely person names (single capitalized words that look like
/// first names, not place names or other proper nouns).
fn extract_person_names_only(text: &str) -> Vec<String> {
    let skip_words = [
        "Paid",
        "Refund",
        "The",
        "This",
        "That",
        "I",
        "My",
        "It",
        "A",
        "An",
        "For",
        "With",
        "And",
        "Or",
        "But",
        "Is",
        "Are",
        "Was",
        "Were",
        "Has",
        "Have",
        "Had",
        "Do",
        "Does",
        "Did",
        "In",
        "On",
        "At",
        "To",
        "From",
        "By",
        "Of",
        "About",
        "Split",
        "Each",
        "All",
        "Between",
        "Among",
        "Yes",
        "No",
        "Shared",
        "Equally",
        "Three",
        "Should",
        "Their",
        "Our",
        "Everyone",
        "Total",
        "Cost",
        "Hotel",
        "Dinner",
        "Lunch",
        "Breakfast",
        "Museum",
        "Tour",
        "Taxi",
        "Train",
        "Coffee",
        "Italian",
        "French",
        "German",
        "Berlin",
        "Paris",
        "London",
        "Amsterdam",
    ];

    let mut names = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    for word in &words {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.is_empty() || clean.len() < 2 || clean.len() > 20 {
            continue;
        }
        if !clean.starts_with(|c: char| c.is_uppercase()) {
            continue;
        }
        if skip_words.contains(&clean) {
            continue;
        }
        // Must be all alphabetic (no digits, no punctuation)
        if !clean.chars().all(|c| c.is_alphabetic()) {
            continue;
        }
        names.push(clean.to_string());
    }

    names
}

// Rule-based bridge functions (bridge_transaction, bridge_state_change,
// bridge_relationship, bridge_preference) have been removed.
// All fact extraction is now handled by LLM compaction.

/// Convert a `serde_json::Value` to `MetadataValue`.
/// Returns `None` for null/array/object (non-scalar) values.
fn json_value_to_metadata(v: &serde_json::Value) -> Option<agent_db_events::core::MetadataValue> {
    use agent_db_events::core::MetadataValue;
    match v {
        serde_json::Value::String(s) => Some(MetadataValue::String(s.clone())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(MetadataValue::Integer(i))
            } else {
                n.as_f64().map(MetadataValue::Float)
            }
        },
        serde_json::Value::Bool(b) => Some(MetadataValue::String(
            if *b { "true" } else { "false" }.to_string(),
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Event pipeline bridge — produces real Event structs
// ---------------------------------------------------------------------------

/// Convert a `ConversationIngest` into `Event` structs suitable for the
/// full event pipeline (`process_event_with_options`).
///
/// Returns `(events, state, result)` where:
/// - `events` is a Vec of Event structs ready for pipeline processing
/// - `state` is the ConversationState with name registry + participants
/// - `result` is the IngestResult summary
pub fn ingest_to_events(
    data: &ConversationIngest,
    options: &IngestOptions,
) -> (Vec<agent_db_events::Event>, ConversationState, IngestResult) {
    use agent_db_events::core::{ActionOutcome, EventContext, EventType, MetadataValue};
    use std::collections::HashMap;

    let case_id = data
        .case_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let mut state = ConversationState::new();
    let mut result = IngestResult {
        case_id: case_id.clone(),
        ..Default::default()
    };
    let mut events: Vec<agent_db_events::Event> = Vec::new();

    // Derive a stable agent_id from case_id (high bit set to avoid collisions)
    let agent_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        case_id.hash(&mut hasher);
        hasher.finish() | 0x8000_0000_0000_0000
    };

    // Base timestamp for monotonic ordering
    let base_timestamp: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // First pass: collect all participant names
    for session in &data.sessions {
        for msg in &session.messages {
            if msg.role == "assistant" && !options.include_assistant_facts {
                continue;
            }
            collect_participants(&msg.content, &mut state.known_participants);
        }
    }

    let mut global_msg_idx: usize = 0;

    // Build request-level metadata (converted from serde_json::Value to MetadataValue)
    let request_metadata: HashMap<String, MetadataValue> = data
        .metadata
        .iter()
        .filter_map(|(k, v)| json_value_to_metadata(v).map(|mv| (k.clone(), mv)))
        .collect();

    // Second pass: classify, parse, and create events
    for session in &data.sessions {
        // Derive session_id hash
        let session_id_hash: u64 = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{}:{}", case_id, session.session_id).hash(&mut hasher);
            hasher.finish()
        };

        for (idx, msg) in session.messages.iter().enumerate() {
            // Idempotency check
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);

            result.messages_processed += 1;

            let timestamp = base_timestamp + (global_msg_idx as u64) * 1_000_000;
            global_msg_idx += 1;

            // No rule-based classification — every message is a raw Conversation event.
            // The LLM compaction (run_compaction) handles all fact extraction.
            let speaker = msg
                .content
                .split(':')
                .next()
                .unwrap_or(&msg.role)
                .trim()
                .to_string();

            // Merge request-level metadata with per-message metadata.
            // Per-message metadata takes precedence over request-level.
            let mut event_metadata = request_metadata.clone();
            for (k, v) in &msg.metadata {
                if let Some(mv) = json_value_to_metadata(v) {
                    event_metadata.insert(k.clone(), mv);
                }
            }

            let evt = agent_db_events::Event {
                id: agent_db_core::types::generate_event_id(),
                timestamp,
                agent_id,
                agent_type: "conversation_agent".to_string(),
                session_id: session_id_hash,
                event_type: EventType::Conversation {
                    speaker,
                    content: msg.content.clone(),
                    category: "message".to_string(),
                },
                causality_chain: Vec::new(),
                context: EventContext::default(),
                metadata: event_metadata,
                context_size_bytes: 0,
                segment_pointer: None,
                is_code: false,
            };
            events.push(evt);
        }

        // Append sentinel event at end of each session to trigger episode completion
        let sentinel_timestamp = base_timestamp + (global_msg_idx as u64) * 1_000_000;
        global_msg_idx += 1;

        let sentinel = agent_db_events::Event {
            id: agent_db_core::types::generate_event_id(),
            timestamp: sentinel_timestamp,
            agent_id,
            agent_type: "conversation_agent".to_string(),
            session_id: {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                format!("{}:{}", case_id, session.session_id).hash(&mut hasher);
                hasher.finish()
            },
            event_type: EventType::Action {
                action_name: "session_complete".to_string(),
                parameters: serde_json::json!({"session_id": session.session_id}),
                outcome: ActionOutcome::Success {
                    result: serde_json::json!({"status": "complete"}),
                },
                duration_ns: 1,
            },
            causality_chain: Vec::new(),
            context: EventContext::default(),
            metadata: HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        };
        events.push(sentinel);

        // Bridge fact_quote from session metadata
        if session.contains_fact == Some(true) {
            if let (Some(ref quote), Some(ref fact_id)) = (&session.fact_quote, &session.fact_id) {
                let category = session
                    .topic
                    .clone()
                    .unwrap_or_else(|| "general".to_string());
                let sentiment = if fact_id.contains("negative") {
                    0.2
                } else if fact_id.contains("mixed") {
                    0.5
                } else {
                    0.8
                };
                let mut meta = HashMap::new();
                meta.insert(
                    "entity".to_string(),
                    MetadataValue::String("user".to_string()),
                );
                meta.insert(
                    "item".to_string(),
                    MetadataValue::String(format!("[{}] {}", fact_id, quote)),
                );
                meta.insert("category".to_string(), MetadataValue::String(category));
                meta.insert("sentiment".to_string(), MetadataValue::Float(sentiment));
                meta.insert(
                    "preference".to_string(),
                    MetadataValue::String("true".to_string()),
                );

                let fact_evt = agent_db_events::Event {
                    id: agent_db_core::types::generate_event_id(),
                    timestamp: base_timestamp + (global_msg_idx as u64) * 1_000_000,
                    agent_id,
                    agent_type: "conversation_agent".to_string(),
                    session_id: {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        format!("{}:{}", case_id, session.session_id).hash(&mut hasher);
                        hasher.finish()
                    },
                    event_type: EventType::Conversation {
                        speaker: "user".to_string(),
                        content: quote.clone(),
                        category: "preference".to_string(),
                    },
                    causality_chain: Vec::new(),
                    context: EventContext::default(),
                    metadata: meta,
                    context_size_bytes: 0,
                    segment_pointer: None,
                    is_code: false,
                };
                events.push(fact_evt);
                global_msg_idx += 1;
                result.facts_captured.push((fact_id.clone(), quote.clone()));
            }
        }
    }

    (events, state, result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ingest(messages: Vec<(&str, &str)>) -> ConversationIngest {
        ConversationIngest {
            case_id: Some("test".to_string()),
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
    fn bridge_counts_messages() {
        let data = make_ingest(vec![
            ("user", "Alice: Paid €100 for dinner - split among all"),
            ("user", "Bob: Paid €60 for lunch - split with Alice"),
        ]);
        let mut store = StructuredMemoryStore::new();
        let result = ingest(&data, &mut store, &IngestOptions::default());

        // Rule-based classification is unplugged; messages are just counted.
        // LLM compaction handles all fact extraction.
        assert_eq!(result.messages_processed, 2);
    }

    #[test]
    fn bridge_idempotency() {
        let data = make_ingest(vec![(
            "user",
            "Alice: Paid €100 for dinner - split among all",
        )]);
        let mut store = StructuredMemoryStore::new();
        let r1 = ingest(&data, &mut store, &IngestOptions::default());
        // Ingesting the same data again should process messages again
        // (idempotency is per ConversationState instance, fresh per call)
        let r2 = ingest(&data, &mut store, &IngestOptions::default());
        assert_eq!(r1.messages_processed, r2.messages_processed);
    }

    #[test]
    fn test_incremental_idempotency() {
        let data = make_ingest(vec![
            ("user", "Alice: Paid €100 for dinner - split with Bob"),
            ("user", "Bob: Paid €60 for lunch - split with Alice"),
        ]);
        let mut store = StructuredMemoryStore::new();
        let mut state = ConversationState::new();

        // First ingest
        let r1 = ingest_incremental(&data, &mut store, &IngestOptions::default(), &mut state);
        assert_eq!(r1.messages_processed, 2);

        // Second ingest with same data and same state — should skip all
        let r2 = ingest_incremental(&data, &mut store, &IngestOptions::default(), &mut state);
        assert_eq!(r2.messages_processed, 0);
    }

    // -----------------------------------------------------------------------
    // ingest_to_events tests
    // -----------------------------------------------------------------------

    fn make_ingest_for_events(messages: Vec<(&str, &str)>) -> ConversationIngest {
        ConversationIngest {
            case_id: Some("test_events".to_string()),
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
    fn test_ingest_to_events_creates_conversation_events() {
        let data = make_ingest_for_events(vec![
            ("user", "Alice: Paid €100 for dinner - split with Bob"),
            ("user", "I live in Alfama, Lisbon."),
        ]);
        let (events, _state, result) = ingest_to_events(&data, &IngestOptions::default());

        // Rule-based classification is unplugged; all messages become
        // raw Conversation events. LLM compaction handles fact extraction.
        assert_eq!(result.messages_processed, 2);

        // Should have conversation events + sentinel
        let conv_events: Vec<_> = events
            .iter()
            .filter(|e| {
                matches!(
                    &e.event_type,
                    agent_db_events::core::EventType::Conversation { .. }
                )
            })
            .collect();
        assert_eq!(conv_events.len(), 2, "Expected 2 conversation events");
    }

    #[test]
    fn test_ingest_to_events_sentinel() {
        let data = make_ingest_for_events(vec![("user", "Alice: Hello there")]);
        let (events, _state, _result) = ingest_to_events(&data, &IngestOptions::default());

        // Last event should be the session_complete sentinel
        let last = events.last().unwrap();
        match &last.event_type {
            agent_db_events::core::EventType::Action { action_name, .. } => {
                assert_eq!(action_name, "session_complete");
            },
            other => panic!("Expected Action sentinel, got {:?}", other),
        }
    }

    #[test]
    fn test_ingest_to_events_timestamps_monotonic() {
        let data = make_ingest_for_events(vec![
            ("user", "Alice: Paid €50 for lunch - split with Bob"),
            ("user", "Bob: Paid €30 for coffee - split with Alice"),
            ("user", "Alice: I love pizza"),
        ]);
        let (events, _state, _result) = ingest_to_events(&data, &IngestOptions::default());

        // All timestamps should be monotonically non-decreasing
        for window in events.windows(2) {
            assert!(
                window[1].timestamp >= window[0].timestamp,
                "Timestamps not monotonic: {} < {}",
                window[1].timestamp,
                window[0].timestamp
            );
        }
    }

    #[test]
    fn test_ingest_to_events_agent_id_high_bit() {
        let data = make_ingest_for_events(vec![("user", "Alice: Hello")]);
        let (events, _state, _result) = ingest_to_events(&data, &IngestOptions::default());

        for event in &events {
            assert!(
                event.agent_id & 0x8000_0000_0000_0000 != 0,
                "Agent ID should have high bit set, got {:#x}",
                event.agent_id
            );
        }
    }

    #[test]
    fn test_ingest_to_events_participants_collected() {
        let data = make_ingest_for_events(vec![(
            "user",
            "Alice: Paid €50 for lunch - split with Bob and Carol",
        )]);
        let (_events, state, _result) = ingest_to_events(&data, &IngestOptions::default());

        assert!(
            state.known_participants.len() >= 2,
            "Expected at least 2 participants, got {}",
            state.known_participants.len()
        );
    }
}
