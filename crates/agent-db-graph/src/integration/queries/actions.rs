// crates/agent-db-graph/src/integration/queries/actions.rs
//
// Policy guide: next-action suggestions ranked by centrality, PPR, and world model.

use super::*;

impl GraphEngine {
    /// Get policy guide suggestions for what action to take next
    ///
    /// **Returns:** Action suggestions ranked by success probability and centrality
    pub async fn get_next_action_suggestions(
        &self,
        context_hash: ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        let mut suggestions = self.traversal.get_next_step_suggestions(
            graph,
            context_hash,
            last_action_node,
            limit * 2,
        )?;

        // Calculate centrality scores for ranking
        let centrality_scores = self.centrality.all_centralities(graph)?;

        // Re-rank suggestions using combined score: success_probability * centrality
        for suggestion in &mut suggestions {
            let combined_score = centrality_scores.combined_score(suggestion.action_node_id);

            // Blend success probability (60%) with centrality importance (40%)
            let original_prob = suggestion.success_probability;
            suggestion.success_probability = (original_prob * 0.6) + (combined_score * 0.4);
        }

        // Re-sort by updated success probability
        suggestions.sort_by(|a, b| {
            b.success_probability
                .partial_cmp(&a.success_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // PPR-based proximity boost: nodes closer to last action get a small bonus
        if let Some(source_node) = last_action_node {
            if let Ok(ppr_scores) = self.random_walker.personalized_pagerank(graph, source_node) {
                // Rank-based normalization: sort PPR scores for the candidate set,
                // map to [0.0, 1.0] rank percentiles
                let n = suggestions.len();
                if n > 0 {
                    let mut indexed_scores: Vec<(usize, f64)> = suggestions
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let score = ppr_scores.get(&s.action_node_id).copied().unwrap_or(0.0);
                            (i, score)
                        })
                        .collect();
                    indexed_scores.sort_by(|a, b| a.1.total_cmp(&b.1));

                    let mut rank_scores = vec![0.0f64; n];
                    for (rank, &(idx, _)) in indexed_scores.iter().enumerate() {
                        rank_scores[idx] = rank as f64 / (n.max(2) - 1) as f64;
                    }

                    for (i, suggestion) in suggestions.iter_mut().enumerate() {
                        suggestion.success_probability =
                            suggestion.success_probability * 0.9 + rank_scores[i] as f32 * 0.1;
                    }

                    suggestions.sort_by(|a, b| {
                        b.success_probability
                            .partial_cmp(&a.success_probability)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // World model reranking (ScoringAndReranking or Full mode only)
        if matches!(
            self.config.effective_world_model_mode(),
            WorldModelMode::ScoringAndReranking | WorldModelMode::Full
        ) {
            if let Some(ref wm) = self.world_model {
                let wm_guard = wm.read().await;
                if wm_guard.energy_stats().is_warmed_up {
                    let policy = agent_db_world_model::PolicyFeatures {
                        goal_count: 1,
                        top_goal_priority: 0.8,
                        resource_cpu_percent: 0.0,
                        resource_memory_bytes: 0,
                        context_fingerprint: context_hash,
                    };
                    let strategy = world_model::extract_strategy_features(None);
                    let report = wm_guard.score_strategy(&policy, &strategy);
                    let wm_score = (-report.total_energy).clamp(0.0, 1.0);

                    for suggestion in &mut suggestions {
                        // Blend: 80% existing score + 20% world model compatibility
                        suggestion.success_probability =
                            suggestion.success_probability * 0.8 + wm_score * 0.2;
                    }

                    // Re-sort after blending
                    suggestions.sort_by(|a, b| {
                        b.success_probability
                            .partial_cmp(&a.success_probability)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // Return top-k after centrality ranking
        Ok(suggestions.into_iter().take(limit).collect())
    }
}
