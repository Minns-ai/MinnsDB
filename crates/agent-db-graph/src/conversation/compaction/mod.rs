//! LLM-driven conversation compaction.
//!
//! Runs AFTER the existing rule-based pipeline to extract:
//! - **Facts**: Cross-message inferences the rule-based classifier misses
//! - **Goals**: User objectives/intentions embedded in conversation flow
//! - **Procedural summary**: Structured session summary with steps and outcomes
//!
//! Extracted data is converted into Events (Observation, Cognitive, Action)
//! and a procedural Memory, then fed back through the pipeline.
//!
//! # Module structure
//!
//! - [`types`] — Core data types (ExtractedFact, CompactionResult, etc.)
//! - [`rolling_summary`] — Rolling conversation summary
//! - [`prompts`] — LLM prompt templates
//! - [`turn_processing`] — Turn splitting, formatting, graph context
//! - [`extraction`] — LLM fact/financial/playbook extraction
//! - [`cascade`] — 3-call cascade extraction pipeline
//! - [`events`] — Event conversion and goal dedup
//! - [`procedural`] — Procedural memory creation and updates
//! - [`embedding`] — Post-compaction embedding and claim creation

mod cascade;
pub(crate) mod embedding;
pub mod events;
pub mod extraction;
pub mod procedural;
mod prompts;
pub mod rolling_summary;
pub mod turn_processing;
pub mod types;

#[cfg(test)]
mod tests;

// ────────── Re-exports ──────────

pub use events::{compaction_to_events, filter_goals_by_classification};
pub use extraction::{extract_compaction, extract_playbooks};
pub use procedural::{build_procedural_memory, map_progress_to_outcome};
pub use rolling_summary::{
    format_with_summary, update_rolling_summary, ConversationRollingSummary,
};
pub use turn_processing::{
    build_graph_context_for_turn, format_transcript, format_turn_transcript, split_into_turns,
    ConversationTurn,
};
pub use types::{
    CompactionResponse, CompactionResult, ExtractedFact, ExtractedGoal, GoalPlaybook,
    PlaybookExtractionResponse, ProceduralStep, ProceduralSummary,
};

// ────────── Top-Level Entry Points ──────────

use crate::conversation::graph_projection;
use crate::conversation::types::ConversationIngest;
use crate::memory::Memory;
use crate::memory_classifier::{classify_memory_updates, resolve_target, MemoryAction};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Maximum number of batches whose LLM extraction runs concurrently. The
/// per-batch cascade is 3 sequential calls + 1 financial; at 4 batches that
/// puts up to 16 in-flight LLM requests, which is comfortably under the
/// provider rate limits we've observed in practice while still giving the
/// ~4× wall-clock win on a multi-batch ingest.
const MAX_PARALLEL_BATCHES: usize = 4;

use cascade::extract_turn_facts_cascade;
use embedding::embed_nodes_and_create_claims;

/// Truncate a string to at most `max_bytes` bytes, ensuring the cut
/// falls on a valid UTF-8 character boundary.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
use extraction::{extract_compaction_from_transcript, extract_financial_facts_llm};
// attach_playbooks is intentionally not imported here — the per-ingest
// playbook extraction has been removed (see the long comment in
// run_compaction_with_context). The function is preserved in `procedural`
// for the future trigger that should drive it.
use procedural::{handle_procedural_memory, handle_procedural_memory_fallback};
use turn_processing::{format_messages, TURN_GAP};
// MIN_PLAYBOOK_CONFIDENCE is preserved in `types` for the eventual
// re-introduction of playbook extraction on a non-ingest trigger.

/// Run LLM-driven compaction on a conversation ingest.
///
/// This is the top-level entry point called after the rule-based pipeline.
/// It extracts facts, goals, and a procedural summary via a single LLM call,
/// then feeds the results back through the event pipeline and stores a
/// procedural memory.
///
/// Fail-open: returns an empty `CompactionResult` on any failure.
/// Run compaction without prior context (used by batch ingest).
pub async fn run_compaction(
    engine: &crate::integration::GraphEngine,
    data: &ConversationIngest,
    case_id: &str,
) -> CompactionResult {
    run_compaction_with_context(engine, data, case_id, None).await
}

/// Run compaction with optional prior context from a rolling summary.
/// When `prior_summary` is provided, it seeds the rolling facts so the LLM
/// can resolve coreferences (pronouns, "the restaurant", etc.) across buffered
/// single-message batches.
pub async fn run_compaction_with_context(
    engine: &crate::integration::GraphEngine,
    data: &ConversationIngest,
    case_id: &str,
    prior_summary: Option<&str>,
) -> CompactionResult {
    let mut result = CompactionResult::default();

    // Need an LLM client
    let llm = match engine.unified_llm_client() {
        Some(client) => Arc::clone(client),
        None => return result,
    };

    // Derive agent_id and session_id using the same formula as ingest_to_events
    let agent_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        case_id.hash(&mut hasher);
        hasher.finish() | 0x8000_0000_0000_0000
    };

    let session_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{}:compaction", case_id).hash(&mut hasher);
        hasher.finish()
    };

    let wall_clock_ts: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // Parse a session timestamp string into nanoseconds since epoch.
    // Supports formats like "2023/05/28 (Sun) 21:04" or "2023-05-28T21:04:00".
    fn parse_session_ts(s: &str) -> Option<u64> {
        // Strip day-of-week in parens: "2023/05/28 (Sun) 21:04" → "2023/05/28 21:04"
        let cleaned = s.replace('/', "-").replace(['(', ')'], "");
        // Remove day names
        let cleaned = cleaned
            .replace("Mon", "")
            .replace("Tue", "")
            .replace("Wed", "")
            .replace("Thu", "")
            .replace("Fri", "")
            .replace("Sat", "")
            .replace("Sun", "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        // Try "YYYY-MM-DD HH:MM"
        let parts: Vec<&str> = cleaned.splitn(2, ' ').collect();
        let date_parts: Vec<u32> = parts
            .first()?
            .split('-')
            .filter_map(|p| p.parse().ok())
            .collect();
        if date_parts.len() != 3 {
            return None;
        }
        let (y, m, d) = (date_parts[0], date_parts[1], date_parts[2]);

        let (hour, min) = if let Some(time_str) = parts.get(1) {
            let tp: Vec<u32> = time_str.split(':').filter_map(|p| p.parse().ok()).collect();
            (
                tp.first().copied().unwrap_or(0),
                tp.get(1).copied().unwrap_or(0),
            )
        } else {
            (0, 0)
        };

        // Days from epoch (rough but consistent for ordering)
        let days = (y as u64) * 365 + (m as u64) * 30 + d as u64;
        let secs = days * 86400 + hour as u64 * 3600 + min as u64 * 60;
        Some(secs * 1_000_000_000)
    }

    // Compute per-session base timestamps
    let session_base_timestamps: Vec<u64> = data
        .sessions
        .iter()
        .enumerate()
        .map(|(i, session)| {
            session
                .timestamp
                .as_deref()
                .and_then(parse_session_ts)
                .unwrap_or(wall_clock_ts + i as u64 * 60 * 1_000_000_000)
        })
        .collect();

    let base_ts = session_base_timestamps
        .first()
        .copied()
        .unwrap_or(wall_clock_ts);

    // ── Phase 1: Per-turn extraction with rolling context + graph state ──
    // Batch adjacent turns in pairs (2 user+assistant exchanges per LLM call)
    // to reduce API calls while maintaining temporal ordering between batches.
    let turns = split_into_turns(data);
    let mut rolling_facts: Vec<String> = match prior_summary {
        Some(s) if !s.is_empty() => vec![format!("[Prior conversation context]\n{}", s)],
        _ => Vec::new(),
    };
    let mut known_entities: HashSet<String> = HashSet::new();
    let mut all_extracted_facts: Vec<ExtractedFact> = Vec::new();

    // Capture turn count before move below — the tracing line and the
    // later `goal_base_ts` calculation both need it.
    let total_turn_count = turns.len();

    // Adaptive batching by token budget (see turn_processing.rs::
    // batch_turns_by_token_budget). Replaces the historical fixed
    // `.chunks(2)` pairing — which over-batched dense conversations
    // (forcing many small cascade calls per 8-session scenario) and
    // under-packed sparse ones. Token-greedy matches the LLM context
    // budget directly: dense scenarios produce ~2-3 larger batches
    // instead of 4-5 tiny ones, halving the cascade-prompt overhead.
    //
    // Owned form (Vec<Vec<ConversationTurn>>) so the downstream JoinSet
    // tasks can move each batch into the spawned future. Avoids
    // borrow/index gymnastics.
    let batches: Vec<Vec<ConversationTurn>> = turn_processing::batch_turns_by_token_budget(turns);

    tracing::info!(
        "COMPACTION per-turn extraction: {} turns in {} batches for case_id={}",
        total_turn_count,
        batches.len(),
        case_id
    );

    // Phase-level timers. Surfaced as structured fields on the existing
    // summary lines so we can tell *where* the synchronous wait went —
    // parallel LLM extraction vs sequential graph writes vs the post-batch
    // goals+summary call. Each phase logs its own duration; the handler
    // logs the overall total.
    let phase1_started = std::time::Instant::now();

    // ── Phase 1: extract all batches in parallel ─────────────────────────
    //
    // Previously this was a `for batch in &batches` loop with the extraction
    // serialised one batch at a time, so a 3-batch payload paid 3× the
    // per-batch latency (≈3 cascade calls + 1 financial each) end to end.
    // Each batch's LLM calls are independent — the only cross-batch signals
    // that mattered were `rolling_facts` and `known_entities`, which fed
    // into the cascade prompt of the NEXT batch. We trade that incremental
    // context for parallelism: every batch sees the same initial rolling
    // facts + known entities (whatever flowed in via `prior_summary`), and
    // the cascade is told from one snapshot of the graph. The quality loss
    // is small in practice (batches within a single ingest are usually
    // about the same conversation) and the wall-clock win is roughly the
    // batch count, capped at MAX_PARALLEL_BATCHES.
    let initial_known: Vec<String> = known_entities.iter().cloned().collect();
    let initial_rolling_ctx = if rolling_facts.is_empty() {
        None
    } else {
        Some(rolling_facts.join("\n"))
    };
    let initial_graph_ctx = {
        let inf = engine.inference.read().await;
        build_graph_context_for_turn(inf.graph(), &initial_known)
    };
    let cat_block = engine.domain_registry.prompt_category_block();
    let cat_enum = engine.domain_registry.prompt_category_enum();

    // Build a (batch_idx, merged_turn, messages_text, label) tuple per batch
    // so the parallel async block has everything it needs without borrowing
    // anything across the await.
    struct BatchInput {
        idx: usize,
        merged_turn: ConversationTurn,
        messages_text: String,
        label: String,
    }
    let batch_inputs: Vec<BatchInput> = batches
        .iter()
        .enumerate()
        .map(|(idx, batch)| {
            let batch_start_idx = batch[0].turn_index;
            let mut merged_turn = ConversationTurn {
                messages: Vec::new(),
                session_index: batch[0].session_index,
                turn_index: batch_start_idx,
                session_timestamp: batch[0].session_timestamp.clone(),
            };
            for turn in batch {
                merged_turn.messages.extend(turn.messages.clone());
            }
            let mut messages_text = String::new();
            if let Some(ref ts) = merged_turn.session_timestamp {
                messages_text.push_str(&format!("[Session date: {}]\n", ts));
            }
            messages_text.push_str(&format_messages(&merged_turn));
            let label = match batch.last() {
                Some(last) if batch.len() > 1 => {
                    format!("batch {}-{}", batch_start_idx, last.turn_index)
                },
                _ => format!("turn {}", batch_start_idx),
            };
            BatchInput {
                idx,
                merged_turn,
                messages_text,
                label,
            }
        })
        .collect();

    // Drive the extractions with a JoinSet bounded by a semaphore. JoinSet
    // gives us tokio-native fan-out; the semaphore caps in-flight LLM calls
    // at MAX_PARALLEL_BATCHES * (cascade + financial) so we don't surge past
    // the provider's rate limit on payloads with many batches.
    let extraction_semaphore = Arc::new(Semaphore::new(MAX_PARALLEL_BATCHES));
    let mut join_set: JoinSet<(
        usize,
        ConversationTurn,
        String,
        Option<Vec<ExtractedFact>>,
        Vec<ExtractedFact>,
    )> = JoinSet::new();
    for bi in batch_inputs.into_iter() {
        let llm = Arc::clone(&llm);
        let rolling = initial_rolling_ctx.clone();
        let known = initial_known.clone();
        let graph_ctx_owned = initial_graph_ctx.clone();
        let cb = cat_block.clone();
        let ce = cat_enum.clone();
        let permit_arc = Arc::clone(&extraction_semaphore);
        join_set.spawn(async move {
            // Permit is held for the lifetime of this task. acquire_owned()
            // can only fail if the semaphore is closed, which we never do.
            let _permit = permit_arc.acquire_owned().await.expect("semaphore closed");
            let (turn_facts, ner_financial_facts) = tokio::join!(
                extract_turn_facts_cascade(
                    llm.as_ref(),
                    &bi.messages_text,
                    rolling.as_deref(),
                    &graph_ctx_owned,
                    &known,
                    &cb,
                    &ce,
                ),
                extract_financial_facts_llm(llm.as_ref(), &bi.messages_text)
            );
            (
                bi.idx,
                bi.merged_turn,
                bi.label,
                turn_facts,
                ner_financial_facts,
            )
        });
    }

    let mut extracted: Vec<(
        usize,
        ConversationTurn,
        String,
        Option<Vec<ExtractedFact>>,
        Vec<ExtractedFact>,
    )> = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok(tuple) => extracted.push(tuple),
            Err(e) => {
                tracing::warn!("COMPACTION extraction task panicked: {}", e);
            },
        }
    }

    // JoinSet returns in completion order; restore submission order so the
    // sequential write phase below sees batches in the order the user
    // uttered them (depends_on stamping reads the prior batch's writes).
    extracted.sort_by_key(|(idx, _, _, _, _)| *idx);

    tracing::info!(
        target: "compaction",
        case_id = %case_id,
        phase = "extract",
        batches = batches.len(),
        ms = phase1_started.elapsed().as_millis() as u64,
        "compaction.phase done"
    );
    let phase2_started = std::time::Instant::now();

    // ── Phase 2: apply writes sequentially per batch ─────────────────────
    //
    // The graph writes (Phase A → B → C inside each batch) must remain in
    // order: a later batch's depends_on stamping reads single-valued state
    // written by the earlier batch, and supersession cascades fire on
    // monotonic timestamps. So we collect extractions concurrently above
    // but apply them one at a time here.
    for (_batch_idx, merged_turn, batch_label, turn_facts, ner_financial_facts) in extracted {
        let batch_start_idx = merged_turn.turn_index;

        match &turn_facts {
            None => {
                tracing::warn!(
                    "COMPACTION {} returned None (extraction failed)",
                    batch_label
                );
            },
            Some(f) if f.is_empty() => {
                tracing::info!(
                    "COMPACTION {} extracted 0 facts (empty response)",
                    batch_label
                );
            },
            _ => {},
        }

        // Merge LLM facts with NER financial facts, deduplicating by
        // normalized (subject, object, category=financial).
        let mut facts_combined = turn_facts.unwrap_or_default();
        if !ner_financial_facts.is_empty() {
            let existing_financial: HashSet<(String, String)> = facts_combined
                .iter()
                .filter(|f| {
                    f.category
                        .as_deref()
                        .is_some_and(|c| engine.ontology.is_append_only(c))
                })
                .map(|f| (f.subject.to_lowercase(), f.object.to_lowercase()))
                .collect();

            for nf in ner_financial_facts {
                let key = (nf.subject.to_lowercase(), nf.object.to_lowercase());
                if !existing_financial.contains(&key) {
                    facts_combined.push(nf);
                }
            }
        }

        if !facts_combined.is_empty() {
            let mut facts = facts_combined;
            let session_ts = session_base_timestamps
                .get(merged_turn.session_index)
                .copied()
                .unwrap_or(base_ts);
            let turn_base = session_ts + batch_start_idx as u64 * TURN_GAP;

            // Inject group_id and request-level metadata into each fact so they flow to graph edges.
            for fact in facts.iter_mut() {
                if !data.group_id.is_empty() {
                    fact.group_id = data.group_id.clone();
                }
                if !data.metadata.is_empty() {
                    fact.ingest_metadata = data.metadata.clone();
                }
            }

            tracing::info!(
                "COMPACTION {} extracted {} facts, writing directly to graph",
                batch_label,
                facts.len(),
            );

            // ── Two-phase write: single-valued first, then multi-valued ──
            // Single-valued facts (location, work, education) establish entity
            // state. Writing them first ensures that:
            //   1. The depends_on stamp on multi-valued facts sees the correct
            //      current location (not the previous one or nothing)
            //   2. The supersession cascade triggers correctly when a new
            //      location replaces an old one, catching old depends_on edges

            // Phase A: write single-valued facts (they establish state).
            //
            // Writes are kept sequential so Phase B's depends_on stamping sees
            // the accumulated graph state — write_fact_to_graph mutates the
            // graph and the next fact may need to read those mutations.
            // Conflict detection, however, is per-fact and idempotent: each
            // call reads candidates filtered by the fact's category and
            // invalidates only the edges that contradict THAT fact. Different
            // facts touch disjoint edge sets, so the LLM calls can fan out
            // in parallel with `join_all` (in-task concurrency — no spawn).
            let mut sv_facts_with_ts: Vec<(&ExtractedFact, u64)> = Vec::new();
            let mut sv_offset = 0u64;
            for fact in facts.iter() {
                let cat = fact.category.as_deref().unwrap_or("");
                if engine.ontology.is_single_valued(cat) {
                    let fact_ts = turn_base + sv_offset * 1_000;
                    engine.write_fact_to_graph(fact, fact_ts).await;
                    sv_facts_with_ts.push((fact, fact_ts));
                    sv_offset += 1;
                }
            }
            if !sv_facts_with_ts.is_empty() {
                let detections = sv_facts_with_ts
                    .iter()
                    .map(|(fact, ts)| engine.detect_edge_conflicts(fact, *ts));
                futures::future::join_all(detections).await;
            }

            // Phase B: stamp depends_on on multi-valued facts using
            // the NOW-UPDATED graph state (after single-valued writes)
            {
                let inf = engine.inference.read().await;
                let graph = inf.graph();
                for fact in facts.iter_mut() {
                    // Skip if already set by the LLM
                    if fact.depends_on.is_some() {
                        continue;
                    }
                    // Skip single-valued categories — they ARE the state
                    let cat = match &fact.category {
                        Some(c) => c.as_str(),
                        None => continue,
                    };
                    if engine.ontology.is_single_valued(cat) {
                        continue;
                    }
                    // For multi-valued facts: check if the subject has a current location.
                    // Normalize the subject name (lowercase, strip articles) to match
                    // the concept_index key, which uses the same normalization.
                    let subject_normalized = fact.subject.to_lowercase();
                    let subject_normalized = subject_normalized.trim();
                    let projected = graph_projection::project_entity_state(
                        graph,
                        subject_normalized,
                        u64::MAX,
                        Some(engine.ontology.as_ref()),
                    );
                    // Find the first single-valued cascade-triggering slot (e.g., location)
                    // to stamp as depends_on context. Ontology-driven — no hardcoded category.
                    let cascade_slot = projected.slots.values().find(|s| {
                        let cat = s.association_type.split(':').next().unwrap_or("");
                        engine.ontology.triggers_cascade(cat)
                    });
                    if let Some(loc_slot) = cascade_slot {
                        let loc_val = loc_slot.value.as_deref().unwrap_or(&loc_slot.target_name);
                        let _cat = loc_slot
                            .association_type
                            .split(':')
                            .next()
                            .unwrap_or("state");
                        let pred = loc_slot.association_type.split(':').nth(1).unwrap_or("in");
                        let dep = format!("{} {} {}", fact.subject, pred, loc_val);
                        tracing::debug!(
                            "COMPACTION: auto-stamped depends_on='{}' on fact '{}'",
                            dep,
                            fact.statement,
                        );
                        fact.depends_on = Some(dep);
                    }
                }
            }

            // Phase C: write multi-valued facts (with correct depends_on).
            // Same shape as Phase A — writes sequential, conflict detection
            // fanned out via join_all afterwards.
            let mut mv_facts_with_ts: Vec<(&ExtractedFact, u64)> = Vec::new();
            let mut mv_offset = sv_offset;
            for fact in facts.iter() {
                let cat = fact.category.as_deref().unwrap_or("");
                if !engine.ontology.is_single_valued(cat) {
                    let fact_ts = turn_base + mv_offset * 1_000;
                    engine.write_fact_to_graph(fact, fact_ts).await;
                    mv_facts_with_ts.push((fact, fact_ts));
                    mv_offset += 1;
                }
            }
            if !mv_facts_with_ts.is_empty() {
                let detections = mv_facts_with_ts
                    .iter()
                    .map(|(fact, ts)| engine.detect_edge_conflicts(fact, *ts));
                futures::future::join_all(detections).await;
            }

            // Update rolling context
            for fact in &facts {
                rolling_facts.push(fact.statement.clone());
                known_entities.insert(fact.subject.clone());
                known_entities.insert(fact.object.clone());
            }
            const MAX_ROLLING_FACTS: usize = 30;
            if rolling_facts.len() > MAX_ROLLING_FACTS {
                rolling_facts.drain(..rolling_facts.len() - MAX_ROLLING_FACTS);
            }

            result.facts_extracted += facts.len();
            all_extracted_facts.extend(facts);
        }
    }

    result.llm_success = true;

    tracing::info!(
        target: "compaction",
        case_id = %case_id,
        phase = "write",
        facts = result.facts_extracted,
        ms = phase2_started.elapsed().as_millis() as u64,
        "compaction.phase done"
    );

    // ── Run community detection so DRIFT search has data ──
    if engine.config.enable_louvain && !all_extracted_facts.is_empty() {
        tracing::info!("COMPACTION: triggering community detection after per-turn extraction");
        if let Err(e) = engine.run_community_detection().await {
            tracing::warn!("COMPACTION: community detection failed: {}", e);
        }
    }

    // ── Embed concept nodes + create claims for extracted facts ──
    // Without this, node vector search and claim hybrid search find nothing
    // for compaction-created artifacts. This must complete BEFORE queries.
    if !all_extracted_facts.is_empty() {
        embed_nodes_and_create_claims(engine, &all_extracted_facts, base_ts).await;
    }

    // ── Active Retrieval Testing (ART): spawn background validation ──
    // Non-blocking — doesn't delay ingest response.
    if engine.config.art_config.enabled {
        if let (Some(ref llm_client), Some(ref embedding_client)) =
            (&engine.unified_llm_client, &engine.embedding_client)
        {
            // Collect recently embedded concept node IDs
            let candidate_node_ids: Vec<u64> = {
                let inf = engine.inference.read().await;
                let graph = inf.graph();
                graph
                    .concept_index
                    .values()
                    .copied()
                    .filter(|&nid| {
                        graph
                            .get_node(nid)
                            .map(|n| n.has_embedding)
                            .unwrap_or(false)
                    })
                    .collect()
            };

            if !candidate_node_ids.is_empty() {
                let inference = engine.inference.clone();
                let vectors = engine.vectors.clone();
                let llm = llm_client.clone();
                let embedder = embedding_client.clone();
                let art_config = engine.config.art_config.clone();

                tokio::spawn(async move {
                    let result = crate::active_retrieval_test::run_art_pass(
                        candidate_node_ids,
                        &inference,
                        &vectors,
                        llm.as_ref(),
                        embedder.as_ref(),
                        &art_config,
                    )
                    .await;
                    tracing::info!(
                        "ART background pass: tested={}, enhanced={}, hits={}, misses={}",
                        result.nodes_tested,
                        result.nodes_enhanced,
                        result.total_hits,
                        result.total_misses,
                    );
                });
            }
        }
    }

    // ── Goals + procedural summary: ONE call at end using full transcript ──
    let full_transcript = format_transcript(data);

    // Enrich with community context if enabled
    let enriched_transcript = if engine.config.enable_context_enrichment {
        let summaries = engine.community_summaries.read().await;
        if summaries.is_empty() {
            full_transcript.clone()
        } else {
            let topic_slice = safe_truncate(&full_transcript, 500);
            let ctx = crate::context_enrichment::community_context_for_topic(
                topic_slice,
                &summaries,
                &engine.config.enrichment_config,
            );
            if ctx.is_empty() {
                full_transcript.clone()
            } else {
                format!("{}\n\n[Knowledge Context]\n{}", full_transcript, ctx)
            }
        }
    } else {
        full_transcript.clone()
    };

    // Extract goals + procedural summary from full transcript
    let goals_started = std::time::Instant::now();
    let goal_extraction = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        extract_compaction_from_transcript(
            llm.as_ref(),
            &enriched_transcript,
            &engine.domain_registry.prompt_category_block(),
            &engine.domain_registry.prompt_category_enum(),
        ),
    )
    .await;
    tracing::info!(
        target: "compaction",
        case_id = %case_id,
        phase = "goals_summary",
        ok = goal_extraction.is_ok(),
        ms = goals_started.elapsed().as_millis() as u64,
        "compaction.phase done"
    );

    if let Ok(Some(response)) = goal_extraction {
        result.goals_extracted = response.goals.len();
        result.procedural_steps_extracted = response
            .procedural_summary
            .as_ref()
            .map(|s| s.steps.len())
            .unwrap_or(0);

        // ── Fast goal dedup via GoalStore (no LLM needed) ──
        let mut fast_dedup_count = 0usize;
        let pre_filtered_goals: Vec<ExtractedGoal> = {
            let mut goal_store = engine.goal_store.write().await;
            let mut kept = Vec::new();
            for goal in &response.goals {
                match goal_store.store_or_dedup(
                    &goal.description,
                    &goal.status,
                    &goal.owner,
                    case_id,
                ) {
                    crate::goal_store::GoalDedupDecision::NewGoal => {
                        kept.push(goal.clone());
                    },
                    crate::goal_store::GoalDedupDecision::Duplicate { .. } => {
                        fast_dedup_count += 1;
                    },
                    crate::goal_store::GoalDedupDecision::StatusUpdate {
                        existing_id,
                        new_status,
                    } => {
                        goal_store.update_status(existing_id, new_status);
                        fast_dedup_count += 1;
                    },
                }
            }
            kept
        };
        result.goals_deduplicated += fast_dedup_count;

        let response = CompactionResponse {
            facts: vec![], // Facts already processed per-turn above
            goals: pre_filtered_goals,
            procedural_summary: response.procedural_summary,
        };

        let has_goals = !response.goals.is_empty();
        let has_summary = response.procedural_summary.is_some();

        if has_goals || has_summary {
            let goal_count = response.goals.len();
            let mut classifiable: Vec<String> = response
                .goals
                .iter()
                .map(|g| g.description.clone())
                .collect();
            if let Some(ref summary) = response.procedural_summary {
                classifiable.push(summary.overall_summary.clone());
            }
            let classifiable_refs: Vec<&str> = classifiable.iter().map(|s| s.as_str()).collect();

            let similar_memories = {
                let bm25 = engine.memory_bm25_index.read().await;
                let mut seen_ids = HashSet::new();
                let mut all_memories = Vec::new();

                for item in &classifiable_refs {
                    let hits = bm25.search(item, 10);
                    for (id, _score) in &hits {
                        seen_ids.insert(*id);
                    }
                }
                drop(bm25);

                let store = engine.memory_store.read().await;
                for id in &seen_ids {
                    if let Some(mem) = store.get_memory(*id) {
                        all_memories.push(mem);
                    }
                }
                all_memories
            };

            let similar_refs: Vec<&Memory> = similar_memories.iter().collect();

            let classifier_ctx = if engine.config.enable_context_enrichment {
                let summaries = engine.community_summaries.read().await;
                if summaries.is_empty() {
                    None
                } else {
                    let topic = classifiable.join(" ");
                    let ctx = crate::context_enrichment::community_context_for_topic(
                        safe_truncate(&topic, 500),
                        &summaries,
                        &engine.config.enrichment_config,
                    );
                    if ctx.is_empty() {
                        None
                    } else {
                        Some(ctx)
                    }
                }
            } else {
                None
            };

            let classification = classify_memory_updates(
                llm.as_ref(),
                &classifiable_refs,
                &similar_refs,
                classifier_ctx.as_deref(),
            )
            .await;

            match classification {
                Ok(class_result) => {
                    let goal_ops =
                        &class_result.operations[..goal_count.min(class_result.operations.len())];
                    let proc_ops = if class_result.operations.len() > goal_count {
                        &class_result.operations[goal_count..]
                    } else {
                        &[]
                    };

                    let (approved_goals, dedup_count) =
                        filter_goals_by_classification(&response.goals, goal_ops);
                    result.goals_deduplicated = dedup_count;

                    for (i, op) in goal_ops.iter().enumerate() {
                        if op.action == MemoryAction::Update {
                            if let Some(target_id) = resolve_target(op, &similar_refs) {
                                let store = engine.memory_store.read().await;
                                let existing = store.get_memory(target_id);
                                drop(store);

                                if let Some(existing_mem) = existing {
                                    let goal_desc = response
                                        .goals
                                        .get(i)
                                        .map(|g| g.description.as_str())
                                        .unwrap_or("");
                                    let mut audit = engine.memory_audit_log.write().await;
                                    audit.record_update(
                                        target_id,
                                        &existing_mem.summary,
                                        goal_desc,
                                        &existing_mem.takeaway,
                                        goal_desc,
                                        crate::memory_audit::MutationActor::LlmClassifier,
                                        Some(format!(
                                            "Compaction goal UPDATE for case {}",
                                            case_id
                                        )),
                                    );
                                }
                            }
                        } else if op.action == MemoryAction::Delete {
                            if let Some(target_id) = resolve_target(op, &similar_refs) {
                                let store = engine.memory_store.read().await;
                                let existing = store.get_memory(target_id);
                                drop(store);

                                if let Some(existing_mem) = existing {
                                    let old_summary = existing_mem.summary.clone();
                                    let old_takeaway = existing_mem.takeaway.clone();

                                    let mut store = engine.memory_store.write().await;
                                    store.delete_memories_batch(vec![target_id]);
                                    drop(store);

                                    result.memories_deleted += 1;

                                    let mut audit = engine.memory_audit_log.write().await;
                                    audit.record_delete(
                                        target_id,
                                        &old_summary,
                                        &old_takeaway,
                                        crate::memory_audit::MutationActor::LlmClassifier,
                                        Some(format!(
                                            "Compaction goal DELETE for case {}",
                                            case_id
                                        )),
                                    );
                                }
                            }
                        }
                    }

                    // Process goal events (no facts — those were already processed per-turn)
                    let filtered_response = CompactionResponse {
                        facts: vec![],
                        goals: approved_goals,
                        procedural_summary: response.procedural_summary.clone(),
                    };

                    let goal_base_ts = base_ts + (total_turn_count as u64 + 1) * TURN_GAP;
                    let events = compaction_to_events(
                        &filtered_response,
                        case_id,
                        agent_id,
                        session_id,
                        goal_base_ts,
                    );
                    for event in events {
                        if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                            tracing::info!("COMPACTION goal event pipeline error: {}", e);
                        }
                    }

                    if let Some(ref summary) = response.procedural_summary {
                        if let Some(proc_op) = proc_ops.first() {
                            handle_procedural_memory(
                                engine,
                                summary,
                                proc_op,
                                &similar_refs,
                                case_id,
                                agent_id,
                                session_id,
                                &mut result,
                            )
                            .await;
                        } else {
                            handle_procedural_memory_fallback(
                                engine,
                                summary,
                                case_id,
                                agent_id,
                                session_id,
                                &mut result,
                            )
                            .await;
                        }
                    }
                },
                Err(_) => {
                    let events =
                        compaction_to_events(&response, case_id, agent_id, session_id, base_ts);
                    for event in events {
                        if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                            tracing::debug!("Compaction event pipeline error: {}", e);
                        }
                    }

                    if let Some(ref summary) = response.procedural_summary {
                        handle_procedural_memory_fallback(
                            engine,
                            summary,
                            case_id,
                            agent_id,
                            session_id,
                            &mut result,
                        )
                        .await;
                    }
                },
            }
        }

        // ── Playbook extraction: REMOVED from the per-ingest path ──
        //
        // Previously, every conversation ingest that produced any goals
        // fired a synchronous LLM call here to extract action-sequence
        // playbooks (with a 30-second internal timeout — already double the
        // edge proxy's 15s budget). That was a structural mistake: most
        // ingests produce no new playbook patterns, and even when they do,
        // playbooks are a cross-goal cross-case abstraction that doesn't
        // need to materialise in the same response cycle as the ingest
        // that contributed one data point.
        //
        // `extract_playbooks` and `attach_playbooks` still exist — they
        // should be re-invoked from a separate trigger (a goal-store
        // threshold, a scheduled rollup pass, or an explicit endpoint),
        // not from inside the ingest hot path. `result.playbooks_extracted`
        // stays in the response schema for wire compatibility; it just
        // always reports 0 until the new trigger lands.
    }

    result
}
