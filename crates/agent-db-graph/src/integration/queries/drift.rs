// crates/agent-db-graph/src/integration/queries/drift.rs
//
// DRIFT search: primer → follow-up → temporal context → fact summary → synthesis.

use super::*;

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

impl GraphEngine {
    /// Run DRIFT search: primer → follow-up → temporal context → fact summary → synthesis.
    pub(crate) async fn run_drift_search(
        &self,
        question: &str,
        community_summaries: &HashMap<u64, crate::community_summary::CommunitySummary>,
        initial_fused: &[(u64, f32)],
        llm_client: &dyn crate::llm_client::LlmClient,
        temporal_frame: &crate::nlq::llm_hint::TemporalFrame,
    ) -> Result<crate::nlq::drift::DriftResult, String> {
        // Use synthesis client for final answer composition if available
        let synthesis_client: &dyn crate::llm_client::LlmClient = self
            .synthesis_llm_client
            .as_ref()
            .map(|c| c.as_ref())
            .unwrap_or(llm_client);
        let config = &self.config.drift_config;

        // Phase 1: Primer — score communities + generate follow-up queries
        let (primer_communities, followup_queries) =
            crate::nlq::drift::drift_primer(llm_client, question, community_summaries, config)
                .await;

        // Phase 2: Follow-up — execute each query and collect results
        let mut results_per_query = Vec::new();
        for q in &followup_queries {
            let bm25 = self.search_bm25(q, config.max_followup_results).await;
            results_per_query.push(bm25);
        }
        let mut merged = crate::nlq::drift::drift_followup_merge(&results_per_query);

        // Apply temporal filters to DRIFT results (same as main pipeline)
        if *temporal_frame == crate::nlq::llm_hint::TemporalFrame::Current {
            {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                apply_temporal_validity_filter(&mut merged, graph);
            }
            if let Some(ref store) = self.claim_store {
                let inference = self.inference.read().await;
                let graph = inference.graph();
                apply_state_anchor_filter(&mut merged, graph, store, &self.ontology);
                apply_epoch_filter(&mut merged, graph, store, &self.ontology);
            }
        }

        // Collect text snippets for synthesis
        let mut community_context = Vec::new();
        for &cid in &primer_communities {
            if let Some(summary) = community_summaries.get(&cid) {
                community_context.push(format!(
                    "{} (entities: {})",
                    summary.summary,
                    summary.key_entities.join(", ")
                ));
            }
        }

        let mut retrieved_snippets = Vec::new();
        // Phase 3: Temporal Context Assembly — build entity timelines + current state
        let temporal_context = {
            let inference = self.inference.read().await;
            let graph = inference.graph();
            let mut seen = std::collections::HashSet::new();
            // Add snippets from initial fused + DRIFT follow-up results
            let combined_iter = initial_fused.iter().take(10).chain(merged.iter().take(10));
            for &(node_id, _score) in combined_iter {
                let label = graph
                    .get_node(node_id)
                    .map(|n| n.label())
                    .unwrap_or_else(|| format!("Node {}", node_id));
                if seen.insert(label.clone()) {
                    retrieved_snippets.push(label);
                }
            }

            // Resolve entity mentions from question and build temporal context
            let entities = extract_entity_names_from_question(graph, question);
            let mut entity_timelines = Vec::new();
            let mut current_state = Vec::new();

            for entity in &entities {
                if let Some(timeline) =
                    crate::conversation::graph_projection::build_entity_timeline_summary(
                        graph, entity,
                    )
                {
                    entity_timelines.push(timeline);
                }
                let projected = crate::conversation::graph_projection::project_entity_state(
                    graph,
                    entity,
                    u64::MAX,
                    None,
                );
                for slot in projected.slots.values() {
                    let value = slot.value.as_deref().unwrap_or(&slot.target_name);
                    current_state
                        .push(format!("{}: {} = {}", entity, slot.association_type, value));
                }
            }

            crate::nlq::drift::DriftTemporalContext {
                entity_timelines,
                current_state,
                temporal_frame: format!("{:?}", temporal_frame),
            }
        };

        let total_items = merged.len();

        // ── DRIFT debug logging ──
        tracing::info!(
            "DRIFT community context ({} items):",
            community_context.len()
        );
        for (i, c) in community_context.iter().enumerate() {
            tracing::info!("  community[{}]: {}", i, safe_truncate(c, 300));
        }
        tracing::info!(
            "DRIFT retrieved snippets ({} items):",
            retrieved_snippets.len()
        );
        for (i, s) in retrieved_snippets.iter().enumerate() {
            tracing::info!("  snippet[{}]: {}", i, safe_truncate(s, 300));
        }
        tracing::info!(
            "DRIFT temporal context: frame={}",
            temporal_context.temporal_frame
        );
        for s in &temporal_context.current_state {
            tracing::info!("  current_state: {}", s);
        }
        for t in &temporal_context.entity_timelines {
            tracing::info!("  timeline: {}", safe_truncate(t, 500));
        }

        // Phase 4: Fact Summary — LLM verifies facts against temporal context
        let fact_summary = if !retrieved_snippets.is_empty() {
            let summary = crate::nlq::drift::drift_fact_summary(
                llm_client,
                question,
                &retrieved_snippets,
                &temporal_context,
                config,
            )
            .await;
            if summary.is_empty() {
                tracing::info!("DRIFT fact_summary: (empty)");
                None
            } else {
                tracing::info!("DRIFT fact_summary:\n{}", safe_truncate(&summary, 1000));
                Some(summary)
            }
        } else {
            None
        };

        // Phase 5: Synthesis — with temporal awareness and fact sheet (uses higher-quality model)
        let answer = crate::nlq::drift::drift_synthesis(
            synthesis_client,
            question,
            &community_context,
            &retrieved_snippets,
            &temporal_context,
            fact_summary.as_deref(),
            config,
        )
        .await
        .map_err(|e| {
            tracing::warn!("DRIFT synthesis failed: {}", e);
            e
        })?;

        Ok(crate::nlq::drift::DriftResult {
            answer,
            primer_communities_used: primer_communities,
            followup_queries,
            total_items_retrieved: total_items,
            fact_summary,
        })
    }
}
