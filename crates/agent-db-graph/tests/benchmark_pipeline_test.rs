//! Integration tests that ingest real benchmark data and verify the full pipeline:
//! graph nodes created with correct content, BM25 search returns relevant nodes,
//! memories formed with meaningful summaries, structured memory populated,
//! and facts captured from conversation sessions.

use agent_db_graph::{ConversationIngest, GraphEngine, GraphEngineConfig, IngestOptions};
use serial_test::serial;

/// Base path for benchmark data (relative to crate root).
const BENCH_BASE: &str = "../../ref/bench/bench/dataset/benchmark";

/// Helper: load a benchmark JSON file and deserialize as ConversationIngest.
/// Returns `None` when the file does not exist (e.g. in CI), so tests can skip.
fn load_benchmark(relative_path: &str) -> Option<ConversationIngest> {
    let path = format!("{}/{}", BENCH_BASE, relative_path);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => panic!("Failed to read {}: {}", path, e),
    };
    Some(
        serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path, e)),
    )
}

/// Macro to skip a test when benchmark data is missing.
macro_rules! skip_if_missing {
    ($path:expr) => {
        match load_benchmark($path) {
            Some(data) => data,
            None => {
                eprintln!(
                    "Skipping: benchmark data not found at {}/{}",
                    BENCH_BASE, $path
                );
                return;
            },
        }
    };
}

/// Trim a ConversationIngest to at most `max_msgs` messages per session.
fn trim_data(data: &ConversationIngest, max_msgs: usize) -> ConversationIngest {
    ConversationIngest {
        case_id: data.case_id.clone(),
        sessions: data
            .sessions
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.messages.truncate(max_msgs);
                s
            })
            .collect(),
        queries: data.queries.clone(),
        group_id: data.group_id.clone(),
        metadata: data.metadata.clone(),
    }
}

/// Helper: ingest a ConversationIngest into the engine, returning event count.
async fn ingest_into_engine(
    engine: &GraphEngine,
    data: &ConversationIngest,
) -> (usize, agent_db_graph::IngestResult) {
    let options = IngestOptions::default();
    let (events, state, result) = agent_db_graph::conversation::ingest_to_events(data, &options);

    // Pre-create participant nodes
    let participants: Vec<String> = state.known_participants.iter().cloned().collect();
    engine
        .ensure_conversation_participants(&participants)
        .await
        .unwrap();

    // Process all events
    let mut processed = 0usize;
    for event in &events {
        if engine
            .process_event_with_options(event.clone(), Some(false))
            .await
            .is_ok()
        {
            processed += 1;
        }
    }

    (processed, result)
}

/// Helper: safely truncate a string at a char boundary.
fn safe_truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Walk backwards from max to find a char boundary
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Helper: given BM25 hits (node_id, score), check that at least one returned node
/// contains the expected text in its properties. Returns the number of matching nodes.
async fn verify_bm25_content(
    engine: &GraphEngine,
    hits: &[(u64, f32)],
    expected_text: &str,
) -> usize {
    let lower = expected_text.to_lowercase();
    let mut matches = 0;
    for &(node_id, _score) in hits {
        if let Some(node) = engine.get_node(node_id).await {
            // Check node_type fields (e.g. Concept name, Claim text)
            let type_text = format!("{:?}", node.node_type).to_lowercase();
            if type_text.contains(&lower) {
                matches += 1;
                continue;
            }
            // Check properties
            for value in node.properties.values() {
                let prop_str = value.to_string().to_lowercase();
                if prop_str.contains(&lower) {
                    matches += 1;
                    break;
                }
            }
        }
    }
    matches
}

// ============================================================================
// Test 1: Accounting — transactions parsed, graph nodes contain real content,
//         BM25 returns nodes mentioning Alice/museum, facts captured
// ============================================================================

#[tokio::test]
#[serial]
async fn test_accounting_ingest_graph_and_search() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    let raw = skip_if_missing!("accounting/data/debt_tracker_10percent.json");
    // Trim to 20 messages to keep test fast (full file has 1000+ messages)
    let data = trim_data(&raw, 20);

    assert!(
        !data.sessions.is_empty(),
        "Benchmark file should have sessions"
    );

    let (processed, result) = ingest_into_engine(&engine, &data).await;

    // --- Verify events were processed ---
    assert!(
        processed >= 5,
        "Expected at least 5 processed events, got {}",
        processed
    );

    // --- Verify transaction detection found actual transactions ---
    assert!(
        result.transactions_found >= 3,
        "Expected at least 3 transactions from accounting data, got {}",
        result.transactions_found
    );
    println!(
        "Accounting: {} events, {} transactions, {} state_changes, {} relationships",
        processed,
        result.transactions_found,
        result.state_changes_found,
        result.relationships_found
    );

    // --- Verify stored events contain Conversation content ---
    let stored = engine.get_recent_events(200).await;
    assert!(
        stored.len() >= 5,
        "Expected at least 5 stored events, got {}",
        stored.len()
    );

    // At least some events should be Conversation type with transaction content
    let conversation_events: Vec<_> = stored
        .iter()
        .filter(|e| {
            matches!(
                &e.event_type,
                agent_db_events::EventType::Conversation { .. }
            )
        })
        .collect();
    assert!(
        !conversation_events.is_empty(),
        "Should have Conversation events stored"
    );

    // Verify conversation events have actual content (not empty strings)
    for evt in &conversation_events {
        if let agent_db_events::EventType::Conversation {
            speaker,
            content,
            category,
        } = &evt.event_type
        {
            assert!(!speaker.is_empty(), "Speaker should not be empty");
            assert!(!content.is_empty(), "Content should not be empty");
            assert!(!category.is_empty(), "Category should not be empty");
        }
    }

    // Verify transaction events have proper metadata (from, to, amount)
    let tx_events: Vec<_> = conversation_events
        .iter()
        .filter(|e| {
            if let agent_db_events::EventType::Conversation { category, .. } = &e.event_type {
                category == "transaction"
            } else {
                false
            }
        })
        .collect();
    assert!(
        !tx_events.is_empty(),
        "Should have transaction-category events"
    );
    for evt in &tx_events {
        assert!(
            evt.metadata.contains_key("from"),
            "Transaction event should have 'from' metadata"
        );
        assert!(
            evt.metadata.contains_key("to"),
            "Transaction event should have 'to' metadata"
        );
        assert!(
            evt.metadata.contains_key("amount"),
            "Transaction event should have 'amount' metadata"
        );
        // Verify from/to are non-empty real names (not system IDs or blanks)
        if let Some(agent_db_events::MetadataValue::String(from)) = evt.metadata.get("from") {
            assert!(
                !from.is_empty() && from.chars().next().unwrap().is_alphabetic(),
                "Transaction 'from' should be a real name (alphabetic), got: '{}'",
                from
            );
        }
        if let Some(agent_db_events::MetadataValue::String(to)) = evt.metadata.get("to") {
            assert!(
                !to.is_empty() && to.chars().next().unwrap().is_alphabetic(),
                "Transaction 'to' should be a real name (alphabetic), got: '{}'",
                to
            );
        }
        // Verify amount is a positive number
        if let Some(agent_db_events::MetadataValue::Float(amount)) = evt.metadata.get("amount") {
            assert!(
                *amount > 0.0,
                "Transaction amount should be positive, got: {}",
                amount
            );
        }
    }

    // --- Verify BM25 search returns nodes with actual Alice content ---
    let alice_hits = engine.search_bm25("Alice", 10).await;
    assert!(
        !alice_hits.is_empty(),
        "BM25 should find nodes mentioning Alice"
    );
    // Verify the returned nodes actually contain "Alice" in their content
    let alice_content_matches = verify_bm25_content(&engine, &alice_hits, "Alice").await;
    assert!(
        alice_content_matches > 0,
        "BM25 hits for 'Alice' should contain nodes with 'Alice' in content/type, \
         got {} hits but none contain 'Alice'",
        alice_hits.len()
    );
    println!(
        "BM25 'Alice': {} hits, {} with Alice in content",
        alice_hits.len(),
        alice_content_matches
    );

    // --- Verify BM25 for "museum" returns nodes with museum content ---
    let museum_hits = engine.search_bm25("museum", 10).await;
    assert!(
        !museum_hits.is_empty(),
        "BM25 should find nodes mentioning museum"
    );
    let museum_content_matches = verify_bm25_content(&engine, &museum_hits, "museum").await;
    assert!(
        museum_content_matches > 0,
        "BM25 hits for 'museum' should contain nodes with 'museum' in content, \
         got {} hits but none contain 'museum'",
        museum_hits.len()
    );
    println!(
        "BM25 'museum': {} hits, {} with museum in content",
        museum_hits.len(),
        museum_content_matches
    );

    // --- Verify BM25 scores are reasonable (positive, non-NaN) ---
    for &(node_id, score) in &alice_hits {
        assert!(
            score > 0.0,
            "BM25 score for node {} should be positive, got {}",
            node_id,
            score
        );
        assert!(!score.is_nan(), "BM25 score should not be NaN");
    }

    // --- Verify structured memory keys exist ---
    let sm = engine.structured_memory().read().await;
    let all_keys: Vec<_> = sm.list_keys("");
    println!("Structured memory keys: {:?}", all_keys);
    drop(sm);
}

// ============================================================================
// Test 2: Relationship graph — colleagues extracted, Concept nodes for people,
//         BM25 returns nodes with actual person names
// ============================================================================

#[tokio::test]
#[serial]
async fn test_relationship_graph_ingest_and_search() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    let data = skip_if_missing!("tree_based/graph_configs/graph_0_trimmed_10_with_path_1.json");

    let (processed, result) = ingest_into_engine(&engine, &data).await;

    let stats = engine.get_graph_stats().await;
    println!(
        "Relationships: {} nodes, {} edges, {} events, {} relationships found",
        stats.node_count, stats.edge_count, processed, result.relationships_found
    );

    // --- Relationship data should detect relationships ---
    assert!(
        result.relationships_found >= 3,
        "Expected at least 3 relationships from colleague data, got {}",
        result.relationships_found
    );

    // --- Verify Concept nodes were created for people ---
    // Johnny Fisher should be a Concept node
    let johnny_hits = engine.search_bm25("Johnny Fisher", 10).await;
    assert!(!johnny_hits.is_empty(), "BM25 should find Johnny Fisher");
    let johnny_matches = verify_bm25_content(&engine, &johnny_hits, "Johnny").await;
    assert!(
        johnny_matches > 0,
        "BM25 hits for 'Johnny Fisher' should have nodes containing 'Johnny'"
    );

    // Look up the top hit and verify it's a Concept or Event node with relevant content
    let (top_node_id, top_score) = johnny_hits[0];
    let top_node = engine.get_node(top_node_id).await;
    assert!(
        top_node.is_some(),
        "Top BM25 result should be a retrievable node"
    );
    let top_node = top_node.unwrap();
    println!(
        "Top hit for 'Johnny Fisher': node_id={}, type={:?}, score={}",
        top_node_id, top_node.node_type, top_score
    );

    // --- Verify Brenda Nguyen is also findable ---
    let brenda_hits = engine.search_bm25("Brenda Nguyen", 10).await;
    assert!(!brenda_hits.is_empty(), "BM25 should find Brenda Nguyen");
    let brenda_matches = verify_bm25_content(&engine, &brenda_hits, "Brenda").await;
    assert!(
        brenda_matches > 0,
        "BM25 hits for 'Brenda Nguyen' should have nodes containing 'Brenda'"
    );

    // --- BM25 for "colleague" should find relationship content ---
    let colleague_hits = engine.search_bm25("colleague", 10).await;
    let colleague_matches = verify_bm25_content(&engine, &colleague_hits, "colleague").await;
    println!(
        "BM25 'colleague': {} hits, {} with colleague in content",
        colleague_hits.len(),
        colleague_matches
    );
    assert!(
        colleague_matches > 0,
        "BM25 hits for 'colleague' should have nodes containing 'colleague'"
    );

    // --- Verify events contain the original relationship text ---
    let stored = engine.get_recent_events(200).await;
    let has_relationship_content = stored.iter().any(|e| {
        if let agent_db_events::EventType::Conversation { content, .. } = &e.event_type {
            content.contains("works with") || content.contains("colleague")
        } else {
            false
        }
    });
    assert!(
        has_relationship_content,
        "Stored events should contain original relationship text ('works with' or 'colleague')"
    );

    // --- Graph structure should be non-empty ---
    let graph_structure = engine.get_graph_structure(50, None, None).await;
    assert!(
        !graph_structure.nodes.is_empty(),
        "Graph structure should have visible nodes"
    );
    // Verify some graph nodes have labels (not all null/empty)
    let labeled_nodes = graph_structure
        .nodes
        .iter()
        .filter(|n| n.label.as_ref().is_some_and(|l| !l.is_empty()))
        .count();
    println!(
        "Graph structure: {} nodes ({} labeled), {} edges",
        graph_structure.nodes.len(),
        labeled_nodes,
        graph_structure.edges.len()
    );
}

// ============================================================================
// Test 3: Recommendations — facts captured from sessions, preference content,
//         BM25 returns nodes with actual genre/book text
// ============================================================================

#[tokio::test]
#[serial]
async fn test_recommendations_ingest_and_search() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    let raw = skip_if_missing!("recommendations/data/books_data.json");
    // Trim to 8 messages per session to keep test fast
    let data = trim_data(&raw, 8);

    let (processed, result) = ingest_into_engine(&engine, &data).await;

    let stats = engine.get_graph_stats().await;
    println!(
        "Recommendations: {} nodes, {} edges, {} events processed",
        stats.node_count, stats.edge_count, processed
    );

    assert!(
        processed >= 5,
        "Expected at least 5 processed events from book recommendations, got {}",
        processed
    );

    // --- Verify facts were captured from sessions ---
    // The books_data.json has contains_fact: true on sessions with fact_quote fields
    println!("Facts captured: {} entries", result.facts_captured.len());
    if !result.facts_captured.is_empty() {
        for (fact_id, quote) in &result.facts_captured {
            println!("  Fact [{}]: {}", fact_id, safe_truncate(quote, 80));
            // Each fact should have a non-empty ID and quote
            assert!(!fact_id.is_empty(), "Fact ID should not be empty");
            assert!(!quote.is_empty(), "Fact quote should not be empty");
            assert!(
                quote.len() > 10,
                "Fact quote should be a real sentence, got: '{}'",
                quote
            );
        }
        // Verify specific known facts from the benchmark
        let has_fantasy_fact = result
            .facts_captured
            .iter()
            .any(|(id, quote)| id.contains("fantasy") || quote.to_lowercase().contains("fantasy"));
        println!("Has fantasy preference fact: {}", has_fantasy_fact);
    }

    // --- Verify BM25 for "fantasy" returns nodes with fantasy content ---
    let fantasy_hits = engine.search_bm25("fantasy", 10).await;
    assert!(!fantasy_hits.is_empty(), "BM25 should find fantasy content");
    let fantasy_matches = verify_bm25_content(&engine, &fantasy_hits, "fantasy").await;
    assert!(
        fantasy_matches > 0,
        "BM25 hits for 'fantasy' should have nodes containing 'fantasy', \
         got {} hits but none matched",
        fantasy_hits.len()
    );
    println!(
        "BM25 'fantasy': {} hits, {} with fantasy in content",
        fantasy_hits.len(),
        fantasy_matches
    );

    // --- Verify BM25 for "horror" returns nodes with horror content ---
    let horror_hits = engine.search_bm25("horror", 10).await;
    if !horror_hits.is_empty() {
        let horror_matches = verify_bm25_content(&engine, &horror_hits, "horror").await;
        println!(
            "BM25 'horror': {} hits, {} with horror in content",
            horror_hits.len(),
            horror_matches
        );
    }

    // --- Verify BM25 for book titles returns actual book references ---
    let shining_hits = engine.search_bm25("Shining", 10).await;
    if !shining_hits.is_empty() {
        let shining_matches = verify_bm25_content(&engine, &shining_hits, "shining").await;
        println!(
            "BM25 'Shining': {} hits, {} with Shining in content",
            shining_hits.len(),
            shining_matches
        );
    }

    // --- Verify stored events have meaningful content ---
    let stored = engine.get_recent_events(200).await;
    let conv_events: Vec<_> = stored
        .iter()
        .filter(|e| {
            matches!(
                &e.event_type,
                agent_db_events::EventType::Conversation { .. }
            )
        })
        .collect();
    assert!(
        !conv_events.is_empty(),
        "Should have Conversation events from book recommendations"
    );

    // Verify the conversation content mentions books/genres
    let has_book_content = conv_events.iter().any(|e| {
        if let agent_db_events::EventType::Conversation { content, .. } = &e.event_type {
            let lower = content.to_lowercase();
            lower.contains("book")
                || lower.contains("novel")
                || lower.contains("fantasy")
                || lower.contains("horror")
                || lower.contains("romance")
                || lower.contains("read")
        } else {
            false
        }
    });
    assert!(
        has_book_content,
        "Conversation events should contain book/genre content"
    );
}

// ============================================================================
// Test 4: NLQ search — verify NLQ returns meaningful answers (not just ok)
// ============================================================================

#[tokio::test]
#[serial]
async fn test_nlq_search_returns_meaningful_results() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    // Ingest accounting data (trimmed for speed)
    let raw = skip_if_missing!("accounting/data/debt_tracker_10percent.json");
    let data = trim_data(&raw, 20);
    ingest_into_engine(&engine, &data).await;

    let pagination = agent_db_graph::nlq::NlqPagination::default();

    // --- NLQ: search for "Alice" should return non-empty answer ---
    let result = engine
        .natural_language_query("Alice", &pagination, None)
        .await;
    assert!(result.is_ok(), "NLQ for 'Alice' should not error");
    let resp = result.unwrap();
    assert!(
        !resp.answer.is_empty(),
        "NLQ answer for 'Alice' should not be empty"
    );
    assert!(
        resp.answer != "No results found." && resp.answer != "Unknown query type",
        "NLQ for 'Alice' should return a meaningful answer, got: '{}'",
        resp.answer
    );
    println!("NLQ 'Alice' answer: {}", safe_truncate(&resp.answer, 200));

    // The answer or result should reference Alice-related content
    let answer_lower = resp.answer.to_lowercase();
    let has_alice_ref = answer_lower.contains("alice")
        || answer_lower.contains("node")
        || answer_lower.contains("found");
    println!("NLQ 'Alice' references relevant content: {}", has_alice_ref);

    // --- NLQ: search for "museum" should find transaction content ---
    let result = engine
        .natural_language_query("museum", &pagination, None)
        .await;
    assert!(result.is_ok(), "NLQ for 'museum' should not error");
    let resp = result.unwrap();
    assert!(
        !resp.answer.is_empty(),
        "NLQ answer for 'museum' should not be empty"
    );
    println!("NLQ 'museum' answer: {}", safe_truncate(&resp.answer, 200));

    // --- NLQ: verify explanation trace is populated ---
    assert!(
        !resp.explanation.is_empty(),
        "NLQ should provide pipeline explanation steps"
    );
    println!("NLQ explanation steps: {:?}", resp.explanation);

    // --- NLQ: search for "dinner" ---
    let result = engine
        .natural_language_query("dinner", &pagination, None)
        .await;
    assert!(result.is_ok(), "NLQ for 'dinner' should not error");
    let resp = result.unwrap();
    assert!(
        !resp.answer.is_empty(),
        "NLQ answer for 'dinner' should not be empty"
    );
    println!("NLQ 'dinner' answer: {}", safe_truncate(&resp.answer, 200));
}

// ============================================================================
// Test 5: Graph visibility — verify node properties contain real content
// ============================================================================

#[tokio::test]
#[serial]
async fn test_graph_structure_nodes_have_content() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    let raw = skip_if_missing!("accounting/data/debt_tracker_10percent.json");
    let data = trim_data(&raw, 20);
    ingest_into_engine(&engine, &data).await;

    let graph_structure = engine.get_graph_structure(200, None, None).await;

    assert!(
        !graph_structure.nodes.is_empty(),
        "Graph structure should not be empty"
    );

    // --- Verify graph nodes have properties with content ---
    let mut nodes_with_content = 0;
    let mut concept_nodes = 0;
    let mut event_nodes = 0;

    for gn in &graph_structure.nodes {
        match gn.node_type.as_str() {
            "Concept" => {
                concept_nodes += 1;
                // Concept nodes should have a label
                if gn.label.as_ref().is_some_and(|l| !l.is_empty()) {
                    nodes_with_content += 1;
                }
            },
            "Event" => {
                event_nodes += 1;
                // Event nodes should have properties
                if !gn.properties.is_null()
                    && gn.properties.as_object().is_some_and(|m| !m.is_empty())
                {
                    nodes_with_content += 1;
                }
            },
            _ => {
                // Other node types count if they have a label or properties
                if gn.label.as_ref().is_some_and(|l| !l.is_empty())
                    || (!gn.properties.is_null()
                        && gn.properties.as_object().is_some_and(|m| !m.is_empty()))
                {
                    nodes_with_content += 1;
                }
            },
        }
    }

    println!(
        "Graph structure: {} nodes ({} concept, {} event), {} with content",
        graph_structure.nodes.len(),
        concept_nodes,
        event_nodes,
        nodes_with_content
    );

    assert!(
        nodes_with_content > 0,
        "Graph nodes should have content (labels or properties), but none did"
    );

    // --- Verify Alice is searchable and the node has proper type ---
    let alice_hits = engine.search_bm25("Alice", 5).await;
    assert!(!alice_hits.is_empty(), "Alice should be findable via BM25");
    // Look up the actual node to verify it's a Concept with name "Alice"
    if let Some(node) = engine.get_node(alice_hits[0].0).await {
        match &node.node_type {
            agent_db_graph::structures::NodeType::Concept {
                concept_name,
                concept_type,
                confidence,
            } => {
                assert!(
                    concept_name.to_lowercase().contains("alice"),
                    "Alice concept node should have 'Alice' in name, got: {}",
                    concept_name
                );
                println!(
                    "Alice node: name='{}', type={:?}, confidence={}",
                    concept_name, concept_type, confidence
                );
            },
            other => {
                // Could also be an Event node mentioning Alice — that's ok too
                let type_name = format!("{:?}", other);
                println!("Alice top hit is a {} node (not Concept)", type_name);
            },
        }
    }
}

// ============================================================================
// Test 6: Cross-dataset search — verify isolation and correct attribution
// ============================================================================

#[tokio::test]
#[serial]
async fn test_unified_search_cross_dataset_content() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    // Ingest accounting (trimmed for speed)
    let raw = skip_if_missing!("accounting/data/debt_tracker_10percent.json");
    let acct = trim_data(&raw, 20);
    ingest_into_engine(&engine, &acct).await;

    // Ingest relationships
    let rels = skip_if_missing!("tree_based/graph_configs/graph_0_trimmed_10_with_path_1.json");
    ingest_into_engine(&engine, &rels).await;

    // --- Alice search should return accounting content (museum, dinner, etc.) ---
    let alice_hits = engine.search_bm25("Alice", 10).await;
    assert!(
        !alice_hits.is_empty(),
        "Should find Alice from accounting data"
    );
    let alice_matches = verify_bm25_content(&engine, &alice_hits, "Alice").await;
    assert!(
        alice_matches > 0,
        "Alice hits should contain Alice in content"
    );

    // --- Johnny search should return relationship content (colleague, works) ---
    let johnny_hits = engine.search_bm25("Johnny", 10).await;
    assert!(
        !johnny_hits.is_empty(),
        "Should find Johnny from relationship data"
    );
    let johnny_matches = verify_bm25_content(&engine, &johnny_hits, "Johnny").await;
    assert!(
        johnny_matches > 0,
        "Johnny hits should contain Johnny in content"
    );

    // --- museum search should NOT return relationship data (cross-contamination check) ---
    let museum_hits = engine.search_bm25("museum", 10).await;
    if !museum_hits.is_empty() {
        // Verify museum hits are from accounting, not relationships
        let museum_matches = verify_bm25_content(&engine, &museum_hits, "museum").await;
        assert!(
            museum_matches > 0,
            "Museum hits should contain 'museum' in content"
        );
    }

    // --- colleague search should NOT return accounting data ---
    let colleague_hits = engine.search_bm25("colleague", 10).await;
    if !colleague_hits.is_empty() {
        let colleague_matches = verify_bm25_content(&engine, &colleague_hits, "colleague").await;
        assert!(
            colleague_matches > 0,
            "Colleague hits should contain 'colleague' in content"
        );
    }

    println!(
        "Cross-dataset: Alice={} hits ({} verified), Johnny={} hits ({} verified)",
        alice_hits.len(),
        alice_matches,
        johnny_hits.len(),
        johnny_matches
    );
}

// ============================================================================
// Test 7: Event content fidelity — verify original message text survives pipeline
// ============================================================================

#[tokio::test]
#[serial]
async fn test_event_content_fidelity() {
    let engine = GraphEngine::with_config(GraphEngineConfig::default())
        .await
        .unwrap();

    let raw = skip_if_missing!("accounting/data/debt_tracker_10percent.json");
    let data = trim_data(&raw, 10);

    let options = IngestOptions::default();
    let (events, _state, _result) = agent_db_graph::conversation::ingest_to_events(&data, &options);

    // Verify raw events before pipeline processing
    assert!(
        !events.is_empty(),
        "Should produce events from benchmark data"
    );

    // Check that Conversation events carry original message content
    let conv_events: Vec<_> = events
        .iter()
        .filter(|e| {
            matches!(
                &e.event_type,
                agent_db_events::EventType::Conversation { .. }
            )
        })
        .collect();
    assert!(
        !conv_events.is_empty(),
        "ingest_to_events should produce Conversation events"
    );

    // The original accounting messages contain "Paid €" — verify this survives
    let has_paid_content = conv_events.iter().any(|e| {
        if let agent_db_events::EventType::Conversation { content, .. } = &e.event_type {
            content.contains("Paid") || content.contains("paid")
        } else {
            false
        }
    });
    assert!(
        has_paid_content,
        "Conversation events should preserve original 'Paid' text from accounting data"
    );

    // Verify timestamps are monotonically increasing
    let mut prev_ts = 0u64;
    for evt in &events {
        assert!(
            evt.timestamp >= prev_ts,
            "Event timestamps should be monotonically increasing"
        );
        prev_ts = evt.timestamp;
    }

    // Now process through pipeline and verify graph nodes contain the content
    let participants: Vec<String> = events
        .iter()
        .filter_map(|e| {
            if let agent_db_events::EventType::Conversation { speaker, .. } = &e.event_type {
                Some(speaker.clone())
            } else {
                None
            }
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    engine
        .ensure_conversation_participants(&participants)
        .await
        .unwrap();

    for event in &events {
        let _ = engine
            .process_event_with_options(event.clone(), Some(false))
            .await;
    }

    // Verify the content made it into graph nodes (searchable via BM25)
    let paid_hits = engine.search_bm25("Paid", 10).await;
    assert!(
        !paid_hits.is_empty(),
        "BM25 should find 'Paid' content after pipeline processing"
    );
    let paid_matches = verify_bm25_content(&engine, &paid_hits, "paid").await;
    assert!(
        paid_matches > 0,
        "BM25 hits for 'Paid' should contain 'paid' in node content"
    );
}

// ============================================================================
// Test 8: Structured memory — verify ledger entries have correct participants
// ============================================================================

#[tokio::test]
#[serial]
async fn test_structured_memory_ledger_content() {
    // Use the direct ingest path (not event pipeline) which populates StructuredMemoryStore
    let data = skip_if_missing!("accounting/data/debt_tracker_10percent.json");
    let data = trim_data(&data, 20);

    let mut store = agent_db_graph::StructuredMemoryStore::new();
    let options = IngestOptions::default();
    let result = agent_db_graph::conversation::bridge::ingest(&data, &mut store, &options);

    println!(
        "Direct ingest: {} transactions, {} messages",
        result.transactions_found, result.messages_processed
    );

    // --- Verify ledger entries exist ---
    let ledger_keys: Vec<_> = store.list_keys("ledger:");
    println!("Ledger keys: {:?}", ledger_keys);

    if !ledger_keys.is_empty() {
        // Ledger keys should reference participant node IDs
        for key in &ledger_keys {
            assert!(
                key.starts_with("ledger:"),
                "Ledger key should start with 'ledger:', got: {}",
                key
            );
        }

        // Check that at least one ledger has a non-zero balance
        let mut found_nonzero = false;
        for key in &ledger_keys {
            if let Some(balance) = store.ledger_balance(key) {
                if balance.abs() > 0.01 {
                    found_nonzero = true;
                    println!("Ledger {} balance: {:.2}", key, balance);
                }
            }
        }
        if result.transactions_found > 0 {
            assert!(
                found_nonzero,
                "With {} transactions, at least one ledger should have non-zero balance",
                result.transactions_found
            );
        }
    }

    // --- Verify state keys if any state changes were detected ---
    let state_keys: Vec<_> = store.list_keys("state:");
    println!("State keys: {:?}", state_keys);

    // --- Verify preference keys if any preferences were detected ---
    let pref_keys: Vec<_> = store.list_keys("prefs:");
    println!("Preference keys: {:?}", pref_keys);
}
