// crates/agent-db-graph/src/integration/pipeline/conflict_detection.rs
//
// LLM-based fact conflict detection and resolution.
// Finds existing edges that contradict a new fact and invalidates them.

use super::*;

impl GraphEngine {
    /// Detect and resolve conflicts between a new fact and existing edges.
    ///
    /// Uses semantic search on the entity's existing edges to find potentially
    /// conflicting facts, then asks the LLM if the new fact contradicts/supersedes
    /// any of them. Conflicting edges get `valid_until` set.
    pub async fn detect_edge_conflicts(
        &self,
        fact: &crate::conversation::compaction::ExtractedFact,
        timestamp: u64,
    ) {
        // Skip append-only/skip-conflict-detection properties
        let category = fact.category.as_deref().unwrap_or("other");
        if self.ontology.skip_conflict_detection(category) {
            return;
        }

        let entity_name = normalize_entity_name(&fact.subject);
        tracing::debug!(
            "detect_edge_conflicts category='{}' entity='{}' stmt='{}'",
            category,
            entity_name,
            fact.statement.get(..60).unwrap_or(&fact.statement)
        );

        // 1. Collect existing active edges from this entity with statement text
        let candidates: Vec<(crate::structures::EdgeId, String)> = {
            let inference = self.inference.read().await;
            let graph = inference.graph();

            let entity_nid = match graph.concept_index.get(&*entity_name) {
                Some(&nid) => nid,
                None => {
                    tracing::debug!(
                        "detect_edge_conflicts: entity '{}' not in concept_index",
                        entity_name
                    );
                    return;
                },
            };

            graph
                .get_edges_from(entity_nid)
                .iter()
                .filter(|e| e.valid_until.is_none()) // only active edges
                .filter(|e| {
                    matches!(
                        &e.edge_type,
                        crate::structures::EdgeType::Association { .. }
                    )
                })
                .filter_map(|e| {
                    let stmt = e
                        .properties
                        .get("statement")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())?;
                    Some((e.id, stmt))
                })
                .collect()
        };

        if candidates.is_empty() {
            tracing::debug!(
                "detect_edge_conflicts: no candidates for entity '{}'",
                entity_name
            );
            return;
        }
        tracing::info!(
            "detect_edge_conflicts: {} candidates for '{}' category='{}'",
            candidates.len(),
            entity_name,
            category
        );

        // 2. Find semantically similar existing facts using simple heuristic first:
        //    same category prefix = potential conflict candidate
        let new_statement = &fact.statement;
        let conflict_candidates: Vec<(crate::structures::EdgeId, String)> = {
            let inference = self.inference.read().await;
            let graph = inference.graph();

            candidates
                .iter()
                .filter(|(eid, _stmt)| {
                    if let Some(e) = graph.edges.get(*eid) {
                        if let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &e.edge_type
                        {
                            // Same category prefix = potential conflict
                            let prefix = association_type.split(':').next().unwrap_or("");
                            prefix == category
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                })
                .cloned()
                .collect()
        };

        if conflict_candidates.is_empty() {
            return;
        }

        // 3. Use LLM to check if new fact contradicts/supersedes any existing fact
        let llm = match &self.unified_llm_client {
            Some(llm) => llm.clone(),
            None => {
                // No LLM available — fall back to simple negation check
                let mut inference = self.inference.write().await;
                let graph = inference.graph_mut();
                for (eid, old_stmt) in &conflict_candidates {
                    if crate::maintenance::is_contradiction(new_statement, old_stmt) {
                        if let Some(e) = graph.edges.get_mut(*eid) {
                            e.valid_until = Some(timestamp);
                            graph.dirty_edges.insert(*eid);
                        }
                        tracing::info!("write_fact_to_graph CONFLICT (negation) invalidated eid={} old='{}' new='{}'",
                            eid, old_stmt.get(..60).unwrap_or(old_stmt), new_statement.get(..60).unwrap_or(new_statement));
                    }
                }
                return;
            },
        };

        // Build LLM prompt with all candidates
        let existing_facts: String = conflict_candidates
            .iter()
            .enumerate()
            .map(|(i, (_, stmt))| format!("{}. {}", i + 1, stmt))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Given a NEW fact and EXISTING facts about the same person, identify which existing facts are \
            CONTRADICTED, SUPERSEDED, or NO LONGER TRUE because of the new fact.\n\n\
            EXISTING FACTS:\n{existing}\n\n\
            NEW FACT: {new_fact}\n\n\
            A fact is superseded when:\n\
            - It describes the SAME attribute/role/activity but with a different value \
              (e.g., new location replaces old location)\n\
            - It describes a routine, habit, or activity tied to a previous location/context \
              that has changed (e.g., \"visits Tsukiji Market\" is superseded when the person \
              moves away from Tokyo)\n\
            - It directly contradicts the new fact (e.g., \"likes X\" vs \"dislikes X\")\n\
            - The new fact makes the old fact no longer applicable \
              (e.g., \"Saturday brunch at local cafe\" superseded by \"Saturday sunrise yoga at beach\")\n\n\
            A fact is NOT superseded when:\n\
            - Both can be true simultaneously (e.g., \"works with Alice\" and \"works with Bob\")\n\
            - They describe different attributes (e.g., \"likes coffee\" and \"lives in NYC\")\n\
            - The old fact is a permanent trait unaffected by the change\n\n\
            Return a JSON array of the numbers of superseded facts. Empty array [] if none.\n\
            Output ONLY the JSON array.",
            existing = existing_facts, new_fact = new_statement
        );

        let request = crate::llm_client::LlmRequest {
            system_prompt:
                "You detect factual contradictions. Output only a JSON array of numbers."
                    .to_string(),
            user_prompt: prompt,
            temperature: 0.0,
            max_tokens: 50,
            json_mode: true,
        };

        match tokio::time::timeout(std::time::Duration::from_secs(10), llm.complete(request)).await
        {
            Ok(Ok(response)) => {
                // Parse the JSON array of conflicting fact indices
                if let Ok(indices) = serde_json::from_str::<Vec<usize>>(&response.content) {
                    if !indices.is_empty() {
                        let mut inference = self.inference.write().await;
                        let graph = inference.graph_mut();
                        for idx in indices {
                            if idx > 0 && idx <= conflict_candidates.len() {
                                let (eid, old_stmt) = &conflict_candidates[idx - 1]; // 1-indexed
                                if let Some(e) = graph.edges.get_mut(*eid) {
                                    e.valid_until = Some(timestamp);
                                    graph.dirty_edges.insert(*eid);
                                }
                                tracing::info!(
                                    "write_fact_to_graph CONFLICT (LLM) invalidated eid={} old='{}' new='{}'",
                                    eid,
                                    old_stmt.get(..60).unwrap_or(old_stmt),
                                    new_statement.get(..60).unwrap_or(new_statement)
                                );
                            }
                        }
                    }
                }
            },
            Ok(Err(e)) => {
                tracing::debug!("Conflict detection LLM call failed: {}", e);
            },
            Err(_) => {
                tracing::debug!("Conflict detection LLM call timed out");
            },
        }
    }
}
