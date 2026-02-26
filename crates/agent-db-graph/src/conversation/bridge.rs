//! Event bridge — converts parsed messages into structured memory entries.
//!
//! Populates the existing `StructuredMemoryStore` directly, bypassing the
//! full event pipeline for efficiency.

use super::parsers;
use super::types::*;
use crate::llm_client::LlmClient;
use crate::structured_memory::{
    LedgerDirection, LedgerEntry, MemoryProvenance, MemoryTemplate, StructuredMemoryStore,
};
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
    store: &mut StructuredMemoryStore,
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

    // Second pass: classify, parse, and bridge each message
    for session in &data.sessions {
        for (idx, msg) in session.messages.iter().enumerate() {
            // Idempotency check
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);

            let ctx = ConversationContext {
                case_id: case_id.clone(),
                session_id: session.session_id.clone(),
                message_index: idx,
                speaker_entity: None,
                ingest_options: options.clone(),
            };

            let parsed = parsers::classify_and_parse(
                &ctx,
                state,
                &msg.content,
                &msg.role,
                session.topic.as_deref(),
            );

            result.messages_processed += 1;

            match &parsed.parsed {
                ParsedPayload::Transaction(tx) => {
                    bridge_transaction(tx, state, store);
                    result.transactions_found += 1;
                },
                ParsedPayload::StateChange(sc) => {
                    bridge_state_change(sc, state, store);
                    result.state_changes_found += 1;
                },
                ParsedPayload::Relationship(rel) => {
                    bridge_relationship(rel, state, store);
                    result.relationships_found += 1;
                },
                ParsedPayload::Preference(pref) => {
                    bridge_preference(pref, state, store);
                    result.preferences_found += 1;
                },
                ParsedPayload::Chitchat(_) => {
                    result.chitchat_skipped += 1;
                },
            }
        }

        // Bridge fact_quote from session metadata into preferences
        if session.contains_fact == Some(true) {
            if let (Some(ref quote), Some(ref fact_id)) = (&session.fact_quote, &session.fact_id) {
                let category = parsers::infer_preference_category(quote, session.topic.as_deref());
                let sentiment = if fact_id.contains("negative") {
                    0.2
                } else if fact_id.contains("mixed") {
                    0.5
                } else {
                    0.8
                };
                let pref = PreferenceData {
                    entity: "user".to_string(),
                    item: format!("[{}] {}", fact_id, quote),
                    category,
                    sentiment,
                };
                bridge_preference(&pref, state, store);
                result.facts_captured.push((fact_id.clone(), quote.clone()));
            }
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
    store: &mut StructuredMemoryStore,
    options: &IngestOptions,
    llm_client: Option<Arc<dyn LlmClient>>,
    state: &mut ConversationState,
) -> IngestResult {
    let llm_classifier = llm_client.map(super::llm_classifier::ConversationLlmClassifier::new);

    let case_id = data
        .case_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let mut result = IngestResult {
        case_id: case_id.clone(),
        ..Default::default()
    };

    // First pass: collect all participant names
    for session in &data.sessions {
        for msg in &session.messages {
            if msg.role == "assistant" && !options.include_assistant_facts {
                continue;
            }
            collect_participants(&msg.content, &mut state.known_participants);
        }
    }

    // Second pass: classify, parse, and bridge each message
    for session in &data.sessions {
        for (idx, msg) in session.messages.iter().enumerate() {
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);

            let ctx = ConversationContext {
                case_id: case_id.clone(),
                session_id: session.session_id.clone(),
                message_index: idx,
                speaker_entity: None,
                ingest_options: options.clone(),
            };

            // Try LLM classification first, fall back to keyword path
            let parsed = if let Some(ref classifier) = llm_classifier {
                match classifier
                    .classify_and_extract(
                        &msg.content,
                        &state.known_participants,
                        session.topic.as_deref(),
                    )
                    .await
                {
                    Some(mut llm_msg) => {
                        llm_msg.session_id = ctx.session_id.clone();
                        llm_msg.message_index = ctx.message_index;
                        llm_msg.original_content = msg.content.clone();
                        llm_msg
                    },
                    None => {
                        tracing::debug!("LLM classify failed, falling back to keyword path");
                        parsers::classify_and_parse(
                            &ctx,
                            state,
                            &msg.content,
                            &msg.role,
                            session.topic.as_deref(),
                        )
                    },
                }
            } else {
                parsers::classify_and_parse(
                    &ctx,
                    state,
                    &msg.content,
                    &msg.role,
                    session.topic.as_deref(),
                )
            };

            result.messages_processed += 1;

            match &parsed.parsed {
                ParsedPayload::Transaction(tx) => {
                    bridge_transaction(tx, state, store);
                    result.transactions_found += 1;
                },
                ParsedPayload::StateChange(sc) => {
                    bridge_state_change(sc, state, store);
                    result.state_changes_found += 1;
                },
                ParsedPayload::Relationship(rel) => {
                    bridge_relationship(rel, state, store);
                    result.relationships_found += 1;
                },
                ParsedPayload::Preference(pref) => {
                    bridge_preference(pref, state, store);
                    result.preferences_found += 1;
                },
                ParsedPayload::Chitchat(_) => {
                    result.chitchat_skipped += 1;
                },
            }
        }

        // Bridge fact_quote from session metadata
        if session.contains_fact == Some(true) {
            if let (Some(ref quote), Some(ref fact_id)) = (&session.fact_quote, &session.fact_id) {
                let category = parsers::infer_preference_category(quote, session.topic.as_deref());
                let sentiment = if fact_id.contains("negative") {
                    0.2
                } else if fact_id.contains("mixed") {
                    0.5
                } else {
                    0.8
                };
                let pref = PreferenceData {
                    entity: "user".to_string(),
                    item: format!("[{}] {}", fact_id, quote),
                    category,
                    sentiment,
                };
                bridge_preference(&pref, state, store);
                result.facts_captured.push((fact_id.clone(), quote.clone()));
            }
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
        let mut store = StructuredMemoryStore::new();
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

        // Second pass: classify, parse, and bridge
        for (idx, msg) in session.messages.iter().enumerate() {
            let dedup_key = (case_id.clone(), session.session_id.clone(), idx);
            if state.processed_messages.contains(&dedup_key) {
                continue;
            }
            state.processed_messages.insert(dedup_key);

            let ctx = ConversationContext {
                case_id: case_id.clone(),
                session_id: session.session_id.clone(),
                message_index: idx,
                speaker_entity: None,
                ingest_options: options.clone(),
            };

            let parsed = parsers::classify_and_parse(
                &ctx,
                &state,
                &msg.content,
                &msg.role,
                session.topic.as_deref(),
            );

            result.messages_processed += 1;

            match &parsed.parsed {
                ParsedPayload::Transaction(tx) => {
                    bridge_transaction(tx, &mut state, &mut store);
                    result.transactions_found += 1;
                },
                ParsedPayload::StateChange(sc) => {
                    bridge_state_change(sc, &mut state, &mut store);
                    result.state_changes_found += 1;
                },
                ParsedPayload::Relationship(rel) => {
                    bridge_relationship(rel, &mut state, &mut store);
                    result.relationships_found += 1;
                },
                ParsedPayload::Preference(pref) => {
                    bridge_preference(pref, &mut state, &mut store);
                    result.preferences_found += 1;
                },
                ParsedPayload::Chitchat(_) => {
                    result.chitchat_skipped += 1;
                },
            }
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

// ---------------------------------------------------------------------------
// Transaction → Ledger
// ---------------------------------------------------------------------------

/// Sanitize a name: trim whitespace and trailing punctuation.
fn sanitize_name(name: &str) -> String {
    name.trim()
        .trim_end_matches(|c: char| !c.is_alphanumeric())
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

fn bridge_transaction(
    tx: &TransactionData,
    state: &mut ConversationState,
    store: &mut StructuredMemoryStore,
) {
    let payer_name = sanitize_name(&tx.payer);
    if payer_name.is_empty() {
        return;
    }
    let payer_id = state.name_registry.get_or_create(&payer_name);

    // Resolve beneficiaries
    let beneficiaries: Vec<String> = if tx.participants_scope == ParticipantsScope::EveryoneKnown {
        let mut all: Vec<String> = state
            .known_participants
            .iter()
            .map(|n| sanitize_name(n))
            .filter(|n| !n.is_empty())
            .collect();
        all.sort();
        all.dedup();
        all
    } else {
        tx.beneficiaries
            .iter()
            .map(|n| sanitize_name(n))
            .filter(|n| !n.is_empty())
            .collect()
    };

    // Handle refund: reverses the direction of a normal payment.
    // If Alice gave a refund of €24 each for Bob and Charlie, it means
    // Alice is REDUCING what Bob/Charlie owe her (or increasing what she owes them).
    // So the direction is INVERTED compared to a normal payment.
    if tx.kind == TransactionKind::Reimbursement {
        // Determine per-person refund amount
        let num_beneficiaries = beneficiaries.len().max(1) as f64;
        let per_person = tx.amount / num_beneficiaries;

        for beneficiary_name in &beneficiaries {
            if beneficiary_name == &payer_name {
                continue;
            }
            let beneficiary_id = state.name_registry.get_or_create(beneficiary_name);
            let key = crate::structured_memory::ledger_key(payer_id, beneficiary_id);

            // Entity pair names must match the key's ID ordering (lo_id, hi_id)
            let (entity_a_name, entity_b_name) = if payer_id <= beneficiary_id {
                (payer_name.clone(), beneficiary_name.clone())
            } else {
                (beneficiary_name.clone(), payer_name.clone())
            };
            ensure_ledger(store, &key, &entity_a_name, &entity_b_name);

            // INVERTED direction: refund reverses the normal Credit/Debit
            let direction = if payer_id <= beneficiary_id {
                LedgerDirection::Debit // was Credit for normal payment
            } else {
                LedgerDirection::Credit // was Debit for normal payment
            };

            if let Err(e) = store.ledger_append(
                &key,
                LedgerEntry {
                    timestamp: 0,
                    amount: per_person,
                    description: format!("refund: {}", tx.description),
                    direction,
                },
            ) {
                tracing::warn!("Failed to append refund ledger entry: {}", e);
            }
        }
        return;
    }

    // Normal payment: compute per-person share
    let num_beneficiaries = beneficiaries.len().max(1) as f64;
    let per_person = match &tx.split_mode {
        SplitMode::Equal => tx.amount / num_beneficiaries,
        SplitMode::Percentage(_) => {
            // Will be handled per-beneficiary below
            0.0 // placeholder
        },
        SplitMode::ExplicitShares(_) => 0.0,
        SplitMode::SoleBeneficiary => tx.amount,
        SplitMode::Unknown => tx.amount / num_beneficiaries,
    };

    for beneficiary_name in &beneficiaries {
        if beneficiary_name == &payer_name {
            continue; // Skip self — payer doesn't owe themselves
        }

        let beneficiary_id = state.name_registry.get_or_create(beneficiary_name);
        let key = crate::structured_memory::ledger_key(payer_id, beneficiary_id);

        // Entity pair names must match the key's ID ordering (lo_id, hi_id)
        let (entity_a_name, entity_b_name) = if payer_id <= beneficiary_id {
            (payer_name.clone(), beneficiary_name.clone())
        } else {
            (beneficiary_name.clone(), payer_name.clone())
        };
        ensure_ledger(store, &key, &entity_a_name, &entity_b_name);

        let share = match &tx.split_mode {
            SplitMode::Percentage(pcts) => {
                if let Some((_, pct)) = pcts.iter().find(|(n, _)| n == beneficiary_name) {
                    tx.amount * pct / 100.0
                } else {
                    per_person
                }
            },
            SplitMode::ExplicitShares(shares) => {
                if let Some((_, amt)) = shares.iter().find(|(n, _)| n == beneficiary_name) {
                    *amt
                } else {
                    per_person
                }
            },
            _ => per_person,
        };

        // Payer paid for beneficiary → beneficiary owes payer
        // In the ledger keyed by (lo_id, hi_id):
        // If payer_id <= beneficiary_id: payer is entity_a → Credit (entity_a is owed)
        // If payer_id > beneficiary_id: beneficiary is entity_a → Debit (entity_b is owed)
        let direction = if payer_id <= beneficiary_id {
            LedgerDirection::Credit
        } else {
            LedgerDirection::Debit
        };

        if let Err(e) = store.ledger_append(
            &key,
            LedgerEntry {
                timestamp: 0,
                amount: share,
                description: tx.description.clone(),
                direction,
            },
        ) {
            tracing::warn!("Failed to append ledger entry: {}", e);
        }
    }
}

/// Ensure a ledger exists for the given key.
fn ensure_ledger(store: &mut StructuredMemoryStore, key: &str, name_a: &str, name_b: &str) {
    if store.get(key).is_none() {
        store.upsert(
            key,
            MemoryTemplate::Ledger {
                entity_pair: (name_a.to_string(), name_b.to_string()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// StateChange → StateMachine / PreferenceList (for facts)
// ---------------------------------------------------------------------------

fn bridge_state_change(
    sc: &StateChangeData,
    state: &mut ConversationState,
    store: &mut StructuredMemoryStore,
) {
    let entity_id = state.name_registry.get_or_create(&sc.entity);

    if sc.attribute == "location" || sc.attribute == "status" {
        // Mutable state → StateMachine
        let key = format!("state:{}:{}", entity_id, sc.attribute);
        if store.get(&key).is_none() {
            store.upsert(
                &key,
                MemoryTemplate::StateMachine {
                    entity: sc.entity.clone(),
                    current_state: sc.new_value.clone(),
                    history: vec![],
                    provenance: MemoryProvenance::EpisodePipeline,
                },
            );
        } else if let Err(e) = store.state_transition(&key, &sc.new_value, "conversation", 0) {
            tracing::warn!("Failed to apply state transition: {}", e);
        }
    } else if sc.attribute.starts_with("routine:") || sc.attribute == "activity" {
        // Facts (routines, activities) → stored as PreferenceList items
        let key = format!("prefs:{}:facts", entity_id);
        if store.get(&key).is_none() {
            store.upsert(
                &key,
                MemoryTemplate::PreferenceList {
                    entity: sc.entity.clone(),
                    ranked_items: vec![],
                    provenance: MemoryProvenance::EpisodePipeline,
                },
            );
        }
        if let Err(e) = store.preference_update(&key, &sc.new_value, 0, Some(1.0)) {
            tracing::warn!("Failed to update preference (facts): {}", e);
        }
    } else if sc.attribute == "landmark" {
        // Landmarks → stored as PreferenceList items
        let key = format!("prefs:{}:landmarks", entity_id);
        if store.get(&key).is_none() {
            store.upsert(
                &key,
                MemoryTemplate::PreferenceList {
                    entity: sc.entity.clone(),
                    ranked_items: vec![],
                    provenance: MemoryProvenance::EpisodePipeline,
                },
            );
        }
        if let Err(e) = store.preference_update(&key, &sc.new_value, 0, Some(1.0)) {
            tracing::warn!("Failed to update preference (landmarks): {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Relationship → Tree (adjacency)
// ---------------------------------------------------------------------------

fn bridge_relationship(
    rel: &RelationshipData,
    state: &mut ConversationState,
    store: &mut StructuredMemoryStore,
) {
    let _subject_id = state.name_registry.get_or_create(&rel.subject);
    let _object_id = state.name_registry.get_or_create(&rel.object);

    // Store relationships in a tree keyed by relation type
    let key = format!("tree:relations:{}", rel.relation_type);
    if store.get(&key).is_none() {
        store.upsert(
            &key,
            MemoryTemplate::Tree {
                root: rel.relation_type.clone(),
                children: std::collections::HashMap::new(),
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
    }

    // Add bidirectional edges: subject → object and object → subject
    if let Err(e) = store.tree_add_child(&key, &rel.subject, &rel.object) {
        tracing::warn!(
            "Failed to add tree child ({} → {}): {}",
            rel.subject,
            rel.object,
            e
        );
    }
    if let Err(e) = store.tree_add_child(&key, &rel.object, &rel.subject) {
        tracing::warn!(
            "Failed to add tree child ({} → {}): {}",
            rel.object,
            rel.subject,
            e
        );
    }
}

// ---------------------------------------------------------------------------
// Preference → PreferenceList
// ---------------------------------------------------------------------------

fn bridge_preference(
    pref: &PreferenceData,
    state: &mut ConversationState,
    store: &mut StructuredMemoryStore,
) {
    let entity_id = state.name_registry.get_or_create(&pref.entity);
    let key = crate::structured_memory::prefs_key(entity_id, &pref.category);

    if store.get(&key).is_none() {
        store.upsert(
            &key,
            MemoryTemplate::PreferenceList {
                entity: pref.entity.clone(),
                ranked_items: vec![],
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
    }

    // Rank by insertion order (lower = earlier = higher preference)
    // Score represents sentiment
    let rank = if let Some(MemoryTemplate::PreferenceList { ranked_items, .. }) = store.get(&key) {
        ranked_items.len()
    } else {
        0
    };

    if let Err(e) = store.preference_update(&key, &pref.item, rank, Some(pref.sentiment as f64)) {
        tracing::warn!("Failed to update preference ({}): {}", pref.category, e);
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

            let ctx = ConversationContext {
                case_id: case_id.clone(),
                session_id: session.session_id.clone(),
                message_index: idx,
                speaker_entity: None,
                ingest_options: options.clone(),
            };

            let parsed = parsers::classify_and_parse(
                &ctx,
                &state,
                &msg.content,
                &msg.role,
                session.topic.as_deref(),
            );

            result.messages_processed += 1;

            let timestamp = base_timestamp + (global_msg_idx as u64) * 1_000_000;
            global_msg_idx += 1;

            // Determine category string and build metadata
            let (category_str, metadata) = match &parsed.parsed {
                ParsedPayload::Transaction(tx) => {
                    result.transactions_found += 1;
                    // Emit one event per beneficiary pair
                    let payer_name = sanitize_name(&tx.payer);
                    let beneficiaries: Vec<String> =
                        if tx.participants_scope == ParticipantsScope::EveryoneKnown {
                            let mut all: Vec<String> = state
                                .known_participants
                                .iter()
                                .map(|n| sanitize_name(n))
                                .filter(|n| !n.is_empty())
                                .collect();
                            all.sort();
                            all.dedup();
                            all
                        } else {
                            tx.beneficiaries
                                .iter()
                                .map(|n| sanitize_name(n))
                                .filter(|n| !n.is_empty())
                                .collect()
                        };

                    let num_beneficiaries = beneficiaries.len().max(1) as f64;
                    let per_person = tx.amount / num_beneficiaries;

                    for beneficiary_name in &beneficiaries {
                        if beneficiary_name == &payer_name {
                            continue;
                        }
                        let share = per_person;
                        let mut meta = HashMap::new();
                        meta.insert(
                            "from".to_string(),
                            MetadataValue::String(payer_name.clone()),
                        );
                        meta.insert(
                            "to".to_string(),
                            MetadataValue::String(beneficiary_name.clone()),
                        );
                        meta.insert("amount".to_string(), MetadataValue::Float(share));
                        meta.insert(
                            "transaction".to_string(),
                            MetadataValue::String("true".to_string()),
                        );
                        meta.insert(
                            "description".to_string(),
                            MetadataValue::String(tx.description.clone()),
                        );

                        let evt = agent_db_events::Event {
                            id: agent_db_core::types::generate_event_id(),
                            timestamp,
                            agent_id,
                            agent_type: "conversation_agent".to_string(),
                            session_id: session_id_hash,
                            event_type: EventType::Conversation {
                                speaker: parsed
                                    .original_content
                                    .split(':')
                                    .next()
                                    .unwrap_or("unknown")
                                    .trim()
                                    .to_string(),
                                content: msg.content.clone(),
                                category: "transaction".to_string(),
                            },
                            causality_chain: Vec::new(),
                            context: EventContext::default(),
                            metadata: meta,
                            context_size_bytes: 0,
                            segment_pointer: None,
                            is_code: false,
                        };
                        events.push(evt);
                    }
                    continue; // Already pushed events per-beneficiary
                },
                ParsedPayload::StateChange(sc) => {
                    result.state_changes_found += 1;
                    let mut meta = HashMap::new();
                    meta.insert(
                        "entity".to_string(),
                        MetadataValue::String(sc.entity.clone()),
                    );
                    meta.insert(
                        "new_state".to_string(),
                        MetadataValue::String(sc.new_value.clone()),
                    );
                    meta.insert(
                        "entity_state".to_string(),
                        MetadataValue::String("true".to_string()),
                    );
                    ("state_change".to_string(), meta)
                },
                ParsedPayload::Relationship(rel) => {
                    result.relationships_found += 1;
                    let mut meta = HashMap::new();
                    meta.insert(
                        "subject".to_string(),
                        MetadataValue::String(rel.subject.clone()),
                    );
                    meta.insert(
                        "object".to_string(),
                        MetadataValue::String(rel.object.clone()),
                    );
                    meta.insert(
                        "relation_type".to_string(),
                        MetadataValue::String(rel.relation_type.clone()),
                    );
                    meta.insert(
                        "relationship".to_string(),
                        MetadataValue::String("true".to_string()),
                    );
                    ("relationship".to_string(), meta)
                },
                ParsedPayload::Preference(pref) => {
                    result.preferences_found += 1;
                    let mut meta = HashMap::new();
                    meta.insert(
                        "entity".to_string(),
                        MetadataValue::String(pref.entity.clone()),
                    );
                    meta.insert("item".to_string(), MetadataValue::String(pref.item.clone()));
                    meta.insert(
                        "category".to_string(),
                        MetadataValue::String(pref.category.clone()),
                    );
                    meta.insert(
                        "sentiment".to_string(),
                        MetadataValue::Float(pref.sentiment as f64),
                    );
                    meta.insert(
                        "preference".to_string(),
                        MetadataValue::String("true".to_string()),
                    );
                    ("preference".to_string(), meta)
                },
                ParsedPayload::Chitchat(_) => {
                    result.chitchat_skipped += 1;
                    ("chitchat".to_string(), HashMap::new())
                },
            };

            // Extract speaker from message content
            let speaker = msg
                .content
                .split(':')
                .next()
                .unwrap_or(&msg.role)
                .trim()
                .to_string();

            let evt = agent_db_events::Event {
                id: agent_db_core::types::generate_event_id(),
                timestamp,
                agent_id,
                agent_type: "conversation_agent".to_string(),
                session_id: session_id_hash,
                event_type: EventType::Conversation {
                    speaker,
                    content: msg.content.clone(),
                    category: category_str,
                },
                causality_chain: Vec::new(),
                context: EventContext::default(),
                metadata,
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
                let category = parsers::infer_preference_category(quote, session.topic.as_deref());
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
                    })
                    .collect(),
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
        }
    }

    #[test]
    fn bridge_simple_transaction() {
        let data = make_ingest(vec![
            ("user", "Alice: Paid €100 for dinner - split among all"),
            ("user", "Bob: Paid €60 for lunch - split with Alice"),
        ]);
        let mut store = StructuredMemoryStore::new();
        let result = ingest(&data, &mut store, &IngestOptions::default());

        assert_eq!(result.transactions_found, 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn bridge_relationship_chain() {
        let data = make_ingest(vec![
            ("user", "Johnny Fisher works with Christopher Peterson."),
            (
                "user",
                "Christopher Peterson is a colleague of Kathleen Herrera.",
            ),
        ]);
        let mut store = StructuredMemoryStore::new();
        let result = ingest(&data, &mut store, &IngestOptions::default());

        assert_eq!(result.relationships_found, 2);

        // Check tree has both relationships
        let key = "tree:relations:colleague";
        let children = store.tree_children(key, "Johnny Fisher");
        assert!(children.is_some());
        assert!(children
            .unwrap()
            .contains(&"Christopher Peterson".to_string()));

        // Check reverse direction
        let children = store.tree_children(key, "Christopher Peterson");
        assert!(children.is_some());
        let c = children.unwrap();
        assert!(c.contains(&"Johnny Fisher".to_string()));
        assert!(c.contains(&"Kathleen Herrera".to_string()));
    }

    #[test]
    fn bridge_state_change_location() {
        let data = make_ingest(vec![("user", "I live in Alfama, Lisbon.")]);
        let mut store = StructuredMemoryStore::new();
        let result = ingest(&data, &mut store, &IngestOptions::default());

        assert!(result.state_changes_found >= 1);
        // Should have a state machine for location
        let keys = store.list_keys("state:");
        assert!(!keys.is_empty());
    }

    #[test]
    fn bridge_preference() {
        let data = ConversationIngest {
            case_id: Some("test".to_string()),
            sessions: vec![ConversationSession {
                session_id: "s1".to_string(),
                topic: Some("Monet's Water Lilies".to_string()),
                messages: vec![ConversationMessage {
                    role: "user".to_string(),
                    content:
                        "Monet's Water Lilies series captures light in a way that feels alive."
                            .to_string(),
                }],
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
        };
        let mut store = StructuredMemoryStore::new();
        let result = ingest(&data, &mut store, &IngestOptions::default());

        assert!(result.preferences_found >= 1);
        let keys = store.list_keys("prefs:");
        assert!(!keys.is_empty());
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
    fn bridge_skips_assistant() {
        let data = make_ingest(vec![
            ("assistant", "That's a beautiful area!"),
            ("user", "Alice: Paid €50 for coffee - split among all"),
        ]);
        let mut store = StructuredMemoryStore::new();
        let result = ingest(&data, &mut store, &IngestOptions::default());

        assert_eq!(result.chitchat_skipped, 1);
        assert_eq!(result.transactions_found, 1);
    }

    #[test]
    fn test_incremental_preserves_registry() {
        // First batch: Alice and Bob
        let data1 = make_ingest(vec![(
            "user",
            "Alice: Paid €100 for dinner - split with Bob",
        )]);
        let mut store = StructuredMemoryStore::new();
        let mut state = ConversationState::new();
        let r1 = ingest_incremental(&data1, &mut store, &IngestOptions::default(), &mut state);
        assert_eq!(r1.transactions_found, 1);

        // Record Alice and Bob's IDs
        let alice_id = state.name_registry.id_for_name("Alice").unwrap();
        let bob_id = state.name_registry.id_for_name("Bob").unwrap();

        // Second batch: Charlie and Dave (new session to avoid dedup)
        let data2 = ConversationIngest {
            case_id: Some("test".to_string()),
            sessions: vec![ConversationSession {
                session_id: "s2".to_string(),
                topic: None,
                messages: vec![ConversationMessage {
                    role: "user".to_string(),
                    content: "Charlie: Paid €80 for lunch - split with Dave".to_string(),
                }],
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
        };
        let r2 = ingest_incremental(&data2, &mut store, &IngestOptions::default(), &mut state);
        assert_eq!(r2.transactions_found, 1);

        // Alice and Bob should retain their original IDs
        assert_eq!(state.name_registry.id_for_name("Alice").unwrap(), alice_id);
        assert_eq!(state.name_registry.id_for_name("Bob").unwrap(), bob_id);

        // Charlie and Dave should have new distinct IDs
        let charlie_id = state.name_registry.id_for_name("Charlie").unwrap();
        let dave_id = state.name_registry.id_for_name("Dave").unwrap();
        assert_ne!(charlie_id, alice_id);
        assert_ne!(charlie_id, bob_id);
        assert_ne!(dave_id, alice_id);
        assert_ne!(dave_id, bob_id);

        // Ledger keys should be distinct (no collisions)
        let ledger_keys = store.list_keys("ledger:");
        assert!(
            ledger_keys.len() >= 2,
            "Should have at least 2 distinct ledgers"
        );
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
        assert_eq!(r1.transactions_found, 2);

        // Second ingest with same data and same state — should skip all
        let r2 = ingest_incremental(&data, &mut store, &IngestOptions::default(), &mut state);
        assert_eq!(r2.messages_processed, 0);
        assert_eq!(r2.transactions_found, 0);
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
                    })
                    .collect(),
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
        }
    }

    #[test]
    fn test_ingest_to_events_transaction() {
        let data = make_ingest_for_events(vec![(
            "user",
            "Alice: Paid €100 for dinner - split with Bob",
        )]);
        let (events, _state, result) = ingest_to_events(&data, &IngestOptions::default());

        assert_eq!(result.transactions_found, 1);
        // Should have at least 1 transaction event + 1 sentinel
        assert!(
            events.len() >= 2,
            "Expected at least 2 events, got {}",
            events.len()
        );

        // Find a transaction event
        let tx_event = events
            .iter()
            .find(|e| e.metadata.contains_key("transaction"));
        assert!(tx_event.is_some(), "Expected a transaction event");
        let tx = tx_event.unwrap();
        assert!(tx.metadata.contains_key("from"));
        assert!(tx.metadata.contains_key("to"));
        assert!(tx.metadata.contains_key("amount"));
    }

    #[test]
    fn test_ingest_to_events_state_change() {
        let data = make_ingest_for_events(vec![("user", "Alice: I'm moving to Paris")]);
        let (events, _state, result) = ingest_to_events(&data, &IngestOptions::default());

        assert_eq!(result.state_changes_found, 1);
        let state_event = events
            .iter()
            .find(|e| e.metadata.contains_key("entity_state"));
        assert!(state_event.is_some(), "Expected a state change event");
        let se = state_event.unwrap();
        assert!(se.metadata.contains_key("entity"));
        assert!(se.metadata.contains_key("new_state"));
    }

    #[test]
    fn test_ingest_to_events_relationship() {
        let data = make_ingest_for_events(vec![("user", "Alice works with Bob")]);
        let (events, _state, result) = ingest_to_events(&data, &IngestOptions::default());

        assert_eq!(result.relationships_found, 1);
        let rel_event = events
            .iter()
            .find(|e| e.metadata.contains_key("relationship"));
        assert!(rel_event.is_some(), "Expected a relationship event");
        let re = rel_event.unwrap();
        assert!(re.metadata.contains_key("subject"));
        assert!(re.metadata.contains_key("object"));
        assert!(re.metadata.contains_key("relation_type"));
    }

    #[test]
    fn test_ingest_to_events_preference() {
        let data = make_ingest_for_events(vec![("user", "Alice: I love sushi")]);
        let (events, _state, result) = ingest_to_events(&data, &IngestOptions::default());

        assert_eq!(result.preferences_found, 1);
        let pref_event = events
            .iter()
            .find(|e| e.metadata.contains_key("preference"));
        assert!(pref_event.is_some(), "Expected a preference event");
        let pe = pref_event.unwrap();
        assert!(pe.metadata.contains_key("entity"));
        assert!(pe.metadata.contains_key("item"));
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
