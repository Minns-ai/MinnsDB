// crates/agent-db-graph/src/integration/pipeline/conflict_detection.rs
//
// LLM-based fact conflict detection and resolution.
//
// For each candidate existing fact, the LLM returns one of four verdicts:
//   SUPERSEDES — new fact replaces old (state evolved, life event).
//   CONTRADICTS — new and old disagree but neither is clearly newer.
//   REAFFIRMS — new fact restates old (paraphrase / different wording).
//   NONE — facts are independent.
//
// Each verdict drives a different graph mutation. SUPERSEDES seals the
// old edge with `valid_until`; CONTRADICTS flags both edges `disputed`
// (kept active so callers can surface the conflict); REAFFIRMS leaves
// both edges alone; NONE is a no-op.

use super::contradiction::ContradictionReason;
use super::*;
use serde::Deserialize;

/// One verdict per existing candidate. Format matches the JSON the LLM
/// produces; unknown verdict strings deserialise to `Verdict::None` so a
/// hallucinated label can't corrupt the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub(crate) enum Verdict {
    Supersedes,
    Contradicts,
    Reaffirms,
    #[serde(other)]
    None,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LlmVerdict {
    /// 1-indexed position in the candidate list shown to the LLM.
    pub(crate) index: usize,
    pub(crate) verdict: Verdict,
}

/// Parse the LLM's verdict array. Returns an empty Vec if the payload is
/// malformed — never propagates a parse error into the pipeline.
pub(crate) fn parse_verdicts(content: &str) -> Vec<LlmVerdict> {
    let trimmed = content.trim();
    // Accept either a bare array or `{"verdicts": [...]}` shape.
    if let Ok(arr) = serde_json::from_str::<Vec<LlmVerdict>>(trimmed) {
        return arr;
    }
    #[derive(Deserialize)]
    struct Wrapper {
        verdicts: Vec<LlmVerdict>,
    }
    serde_json::from_str::<Wrapper>(trimmed)
        .map(|w| w.verdicts)
        .unwrap_or_default()
}

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
            "Given a NEW fact and EXISTING facts about the same person, classify each existing \
            fact's relationship to the new fact with exactly one verdict.\n\n\
            EXISTING FACTS:\n{existing}\n\n\
            NEW FACT: {new_fact}\n\n\
            Verdicts:\n\
            - SUPERSEDES: the new fact replaces the existing one because state legitimately \
              changed (new location, life event, ended routine, opposite preference at a later \
              time). The existing fact was true then but is not true now.\n\
            - CONTRADICTS: the new fact and existing fact disagree without a clear temporal \
              ordering. Both are claimed but only one can be right. Examples: conflicting \
              biographical facts, mutually exclusive values from different sources.\n\
            - REAFFIRMS: the new fact restates the existing fact in different words. Same \
              meaning, no change.\n\
            - NONE: the facts are independent or compatible.\n\n\
            Examples:\n\
            - existing \"lives in London\", new \"moved to NYC\" → SUPERSEDES\n\
            - existing \"works with Alice\", new \"works with Bob\" → NONE\n\
            - existing \"likes coffee\", new \"enjoys coffee\" → REAFFIRMS\n\
            - existing \"born in Paris\", new \"born in Lyon\" → CONTRADICTS\n\n\
            Return a JSON array of objects with the candidate's 1-based index and verdict. \
            Omit candidates whose verdict is NONE.\n\
            Example: [{{\"index\": 1, \"verdict\": \"SUPERSEDES\"}}, {{\"index\": 3, \"verdict\": \"REAFFIRMS\"}}]\n\
            Output ONLY the JSON array.",
            existing = existing_facts, new_fact = new_statement
        );

        let request = crate::llm_client::LlmRequest {
            system_prompt:
                "You classify factual relationships. Output only a JSON array of verdicts."
                    .to_string(),
            user_prompt: prompt,
            temperature: 0.0,
            max_tokens: 150,
            json_mode: true,
        };

        match tokio::time::timeout(std::time::Duration::from_secs(10), llm.complete(request)).await
        {
            Ok(Ok(response)) => {
                let verdicts = parse_verdicts(&response.content);
                if verdicts.is_empty() {
                    return;
                }
                let mut inference = self.inference.write().await;
                let graph = inference.graph_mut();
                for v in verdicts {
                    if v.index == 0 || v.index > conflict_candidates.len() {
                        continue;
                    }
                    let (eid, old_stmt) = &conflict_candidates[v.index - 1]; // 1-indexed
                    match v.verdict {
                        Verdict::Supersedes => {
                            if let Some(e) = graph.edges.get_mut(*eid) {
                                e.valid_until = Some(timestamp);
                                graph.dirty_edges.insert(*eid);
                            }
                            tracing::info!(
                                "write_fact_to_graph SUPERSEDES (LLM) sealed eid={} old='{}' new='{}'",
                                eid,
                                old_stmt.get(..60).unwrap_or(old_stmt),
                                new_statement.get(..60).unwrap_or(new_statement)
                            );
                        },
                        Verdict::Contradicts => {
                            // Mark the existing edge disputed but leave it
                            // active. The new fact will be written by the
                            // normal path and also flagged via the metadata
                            // hand-off below.
                            if let Some(e) = graph.edges.get_mut(*eid) {
                                e.properties
                                    .insert("disputed".to_string(), serde_json::Value::Bool(true));
                                e.properties.insert(
                                    "disputed_reason".to_string(),
                                    serde_json::Value::String(
                                        ContradictionReason::LogicalConflict.tag().to_string(),
                                    ),
                                );
                                graph.dirty_edges.insert(*eid);
                            }
                            tracing::warn!(
                                "write_fact_to_graph CONTRADICTS (LLM) flagged disputed eid={} old='{}' new='{}'",
                                eid,
                                old_stmt.get(..60).unwrap_or(old_stmt),
                                new_statement.get(..60).unwrap_or(new_statement)
                            );
                        },
                        Verdict::Reaffirms => {
                            // No structural change; bump observation_count
                            // and refresh `last_confirmed_at` so downstream
                            // freshness signals reflect the repeated mention.
                            if let Some(e) = graph.edges.get_mut(*eid) {
                                e.observation_count = e.observation_count.saturating_add(1);
                                e.properties.insert(
                                    "last_confirmed_at".to_string(),
                                    serde_json::Value::Number(timestamp.into()),
                                );
                                graph.dirty_edges.insert(*eid);
                            }
                            tracing::debug!(
                                "write_fact_to_graph REAFFIRMS (LLM) eid={} old='{}' new='{}'",
                                eid,
                                old_stmt.get(..60).unwrap_or(old_stmt),
                                new_statement.get(..60).unwrap_or(new_statement)
                            );
                        },
                        Verdict::None => {},
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_verdicts_bare_array() {
        let raw =
            r#"[{"index": 1, "verdict": "SUPERSEDES"}, {"index": 3, "verdict": "CONTRADICTS"}]"#;
        let v = parse_verdicts(raw);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].index, 1);
        assert_eq!(v[0].verdict, Verdict::Supersedes);
        assert_eq!(v[1].index, 3);
        assert_eq!(v[1].verdict, Verdict::Contradicts);
    }

    #[test]
    fn parse_verdicts_wrapped() {
        let raw = r#"{"verdicts": [{"index": 2, "verdict": "REAFFIRMS"}]}"#;
        let v = parse_verdicts(raw);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].verdict, Verdict::Reaffirms);
    }

    #[test]
    fn parse_verdicts_unknown_label_becomes_none() {
        // LLM hallucinates an unrecognised verdict — must not corrupt
        // downstream graph mutations.
        let raw = r#"[{"index": 1, "verdict": "DEFENESTRATES"}]"#;
        let v = parse_verdicts(raw);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].verdict, Verdict::None);
    }

    #[test]
    fn parse_verdicts_malformed_returns_empty() {
        assert!(parse_verdicts("not json").is_empty());
        assert!(parse_verdicts("").is_empty());
    }
}
