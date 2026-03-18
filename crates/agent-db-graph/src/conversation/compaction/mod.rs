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
mod embedding;
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
use procedural::{attach_playbooks, handle_procedural_memory, handle_procedural_memory_fallback};
use turn_processing::{format_messages, TURN_GAP};
use types::MIN_PLAYBOOK_CONFIDENCE;

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

    let base_ts: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

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

    // Batch turns in pairs: [0,1], [2,3], [4,5], ...
    let batches: Vec<Vec<&ConversationTurn>> = turns
        .iter()
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| c.to_vec())
        .collect();

    tracing::info!(
        "COMPACTION per-turn extraction: {} turns in {} batches for case_id={}",
        turns.len(),
        batches.len(),
        case_id
    );

    for batch in &batches {
        let batch_start_idx = batch[0].turn_index;

        // Query graph for current entity states
        let known_entities_vec: Vec<String> = known_entities.iter().cloned().collect();
        let graph_ctx = {
            let inf = engine.inference.read().await;
            build_graph_context_for_turn(inf.graph(), &known_entities_vec)
        };

        let rolling_ctx = if rolling_facts.is_empty() {
            None
        } else {
            Some(rolling_facts.join("\n"))
        };

        // Merge batch turns into a single transcript
        let mut merged_turn = ConversationTurn {
            messages: Vec::new(),
            session_index: batch[0].session_index,
            turn_index: batch_start_idx,
        };
        for turn in batch {
            merged_turn.messages.extend(turn.messages.clone());
        }

        let messages_text = format_messages(&merged_turn);
        let cat_block = engine.domain_registry.prompt_category_block();
        let cat_enum = engine.domain_registry.prompt_category_enum();

        // Extract facts: LLM cascade + LLM financial extraction in parallel
        let (turn_facts, ner_financial_facts) = tokio::join!(
            extract_turn_facts_cascade(
                llm.as_ref(),
                &messages_text,
                rolling_ctx.as_deref(),
                &graph_ctx,
                &known_entities_vec,
                &cat_block,
                &cat_enum,
            ),
            extract_financial_facts_llm(llm.as_ref(), &messages_text)
        );

        let batch_label = match batch.last() {
            Some(last) if batch.len() > 1 => {
                format!("batch {}-{}", batch_start_idx, last.turn_index)
            },
            _ => format!("turn {}", batch_start_idx),
        };

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
            let turn_base = base_ts + batch_start_idx as u64 * TURN_GAP;

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

            // Phase A: write single-valued facts (they establish state)
            let mut sv_offset = 0u64;
            for fact in facts.iter() {
                let cat = fact.category.as_deref().unwrap_or("");
                if engine.ontology.is_single_valued(cat) {
                    let fact_ts = turn_base + sv_offset * 1_000;
                    engine.write_fact_to_graph(fact, fact_ts).await;
                    engine.detect_edge_conflicts(fact, fact_ts).await;
                    sv_offset += 1;
                }
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

            // Phase C: write multi-valued facts (with correct depends_on)
            let mut mv_offset = sv_offset;
            for fact in facts.iter() {
                let cat = fact.category.as_deref().unwrap_or("");
                if !engine.ontology.is_single_valued(cat) {
                    let fact_ts = turn_base + mv_offset * 1_000;
                    engine.write_fact_to_graph(fact, fact_ts).await;
                    engine.detect_edge_conflicts(fact, fact_ts).await;
                    mv_offset += 1;
                }
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
                            .map(|n| !n.embedding.is_empty())
                            .unwrap_or(false)
                    })
                    .collect()
            };

            if !candidate_node_ids.is_empty() {
                let inference = engine.inference.clone();
                let llm = llm_client.clone();
                let embedder = embedding_client.clone();
                let art_config = engine.config.art_config.clone();

                tokio::spawn(async move {
                    let result = crate::active_retrieval_test::run_art_pass(
                        candidate_node_ids,
                        &inference,
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

                    let goal_base_ts = base_ts + (turns.len() as u64 + 1) * TURN_GAP;
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

        // ── Playbook extraction ──
        if !response.goals.is_empty() {
            let enriched_pb_transcript = if engine.config.enable_context_enrichment {
                let goal_store = engine.goal_store.read().await;
                let mut existing = Vec::new();
                for goal in &response.goals {
                    for (id, _score) in goal_store.find_similar(&goal.description, 3) {
                        if let Some(entry) = goal_store.get(id) {
                            if let Some(ref pb) = entry.playbook {
                                existing.push((entry.description.clone(), pb.clone()));
                            }
                        }
                    }
                }
                drop(goal_store);
                if existing.is_empty() {
                    full_transcript.clone()
                } else {
                    let ctx = crate::context_enrichment::build_playbook_context(
                        &existing,
                        engine.config.enrichment_config.max_similar_playbooks,
                    );
                    format!(
                        "{}\n\n[Prior Playbook Experience]\n{}",
                        full_transcript, ctx
                    )
                }
            } else {
                full_transcript.clone()
            };

            if let Ok(Some(pb_response)) = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                extract_playbooks(llm.as_ref(), &enriched_pb_transcript, &response.goals),
            )
            .await
            {
                let quality_playbooks: Vec<_> = pb_response
                    .playbooks
                    .into_iter()
                    .filter(|pb| {
                        pb.confidence >= MIN_PLAYBOOK_CONFIDENCE && !pb.steps_taken.is_empty()
                    })
                    .collect();
                result.playbooks_extracted = quality_playbooks.len();
                attach_playbooks(engine, &quality_playbooks, case_id).await;
            }
        }
    }

    result
}
