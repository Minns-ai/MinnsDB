//! Policy guide queries: next-step suggestions, successful continuations,
//! dead-end detection, and context-aware action discovery.

use super::iterators::{DfsIter, DijkstraIter};
use super::types::ActionSuggestion;
use super::GraphTraversal;
use crate::structures::{EdgeWeight, Graph, NodeId, NodeType};
use crate::GraphResult;
use std::cmp::Ordering;
use std::collections::HashSet;

impl GraphTraversal {
    /// Get next step suggestions based on current context and last action
    pub fn get_next_step_suggestions(
        &self,
        graph: &Graph,
        current_context_hash: agent_db_core::types::ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();

        if let Some(action_node_id) = last_action_node {
            let continuations = self.get_successful_continuations(graph, action_node_id)?;
            suggestions.extend(continuations);
        }

        let context_actions = self.get_actions_for_context(graph, current_context_hash)?;
        suggestions.extend(context_actions);

        let dead_ends = self.get_dead_ends(graph, current_context_hash)?;
        suggestions.retain(|s| !dead_ends.contains(&s.action_name));

        suggestions.sort_by(|a, b| {
            b.success_probability
                .partial_cmp(&a.success_probability)
                .unwrap_or(Ordering::Equal)
        });

        let mut seen = HashSet::new();
        suggestions.retain(|s| seen.insert(s.action_name.clone()));

        Ok(suggestions.into_iter().take(limit).collect())
    }

    /// Find actions that have successfully followed a given action.
    ///
    /// Uses `DijkstraIter` to explore up to 3 hops outward in cost order,
    /// discovering not just immediate successors but multi-step action chains
    /// that tend to follow from `from_action_node`. Closer (lower-cost)
    /// continuations rank higher.
    pub fn get_successful_continuations(
        &self,
        graph: &Graph,
        from_action_node: NodeId,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();
        let mut seen_events = HashSet::new();

        // Explore outward in cost-ascending order, skipping the start node.
        // Cap at 50 nodes explored to bound work on dense graphs.
        for (node_id, cost) in DijkstraIter::new(graph, from_action_node).skip(1).take(50) {
            if let Some(node) = graph.get_node(node_id) {
                if let NodeType::Event { ref event_type, .. } = node.node_type {
                    // Deduplicate by event type name
                    if !seen_events.insert(event_type.clone()) {
                        continue;
                    }

                    // For direct neighbors, use the empirical edge weight;
                    // for multi-hop discoveries, derive probability from cost.
                    let (success_probability, evidence_count) = if let Some(w) =
                        self.get_edge_weight_between(graph, from_action_node, node_id)
                    {
                        let count =
                            self.count_pattern_occurrences(graph, from_action_node, node_id)?;
                        (w.min(1.0), count)
                    } else {
                        // Multi-hop: convert cost back to probability (cost ≈ 1 - p)
                        ((1.0 - cost).clamp(0.01, 1.0), 0)
                    };

                    let hops = if cost < 0.001 {
                        0
                    } else {
                        (cost / 0.3).ceil() as u32
                    };
                    suggestions.push(ActionSuggestion {
                        action_name: event_type.clone(),
                        action_node_id: node_id,
                        success_probability,
                        evidence_count,
                        reasoning: if hops <= 1 {
                            format!(
                                "Direct successor ({} times, {:.0}% success)",
                                evidence_count,
                                success_probability * 100.0
                            )
                        } else {
                            format!(
                                "Reachable via ~{} hops (cost {:.2}), estimated {:.0}% success",
                                hops,
                                cost,
                                success_probability * 100.0
                            )
                        },
                    });
                }
            }
        }

        Ok(suggestions)
    }

    /// Get actions that work well in a specific context.
    ///
    /// Uses `DijkstraIter` from each context node to discover Event nodes
    /// in cost-ascending order — semantically closer actions surface first.
    /// Explores up to 30 nodes per context to find relevant actions beyond
    /// immediate neighbors.
    fn get_actions_for_context(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();
        let mut seen_events = HashSet::new();
        let context_nodes = self.find_context_nodes(graph, context_hash)?;

        for context_node_id in context_nodes {
            // Dijkstra outward from context, skip self, cap exploration.
            for (node_id, cost) in DijkstraIter::new(graph, context_node_id).skip(1).take(30) {
                if let Some(node) = graph.get_node(node_id) {
                    if let NodeType::Event { ref event_type, .. } = node.node_type {
                        if !seen_events.insert(event_type.clone()) {
                            continue;
                        }

                        let success_probability = self
                            .get_edge_weight_between(graph, context_node_id, node_id)
                            .unwrap_or_else(|| (1.0 - cost).clamp(0.01, 1.0));

                        let evidence_count =
                            self.count_pattern_occurrences(graph, context_node_id, node_id)?;

                        suggestions.push(ActionSuggestion {
                            action_name: event_type.clone(),
                            action_node_id: node_id,
                            success_probability,
                            evidence_count,
                            reasoning: format!(
                                "Works well in this context (cost {:.2}, {} observations, {:.0}% success)",
                                cost, evidence_count, success_probability * 100.0
                            ),
                        });
                    }
                }
            }
        }

        Ok(suggestions)
    }

    /// Get actions that are known to fail in this context (dead ends).
    ///
    /// Uses `DfsIter` to explore up to 3 hops deep from each context node,
    /// catching not just immediately-adjacent failures but also actions that
    /// lead *through* intermediate nodes to dead-end outcomes. This finds
    /// "slow death" paths that a 1-hop check would miss.
    pub fn get_dead_ends(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<String>> {
        let mut dead_ends = HashSet::new();
        let context_nodes = self.find_context_nodes(graph, context_hash)?;

        for context_node_id in context_nodes {
            // DFS from context, skip the context node itself, explore 3 hops.
            for (node_id, _depth) in DfsIter::new(graph, context_node_id, 3).skip(1) {
                if let Some(node) = graph.get_node(node_id) {
                    if let NodeType::Event { ref event_type, .. } = node.node_type {
                        // Check all incoming edges to this node for failure signal.
                        let predecessors = graph.get_incoming_neighbors(node_id);
                        for pred_id in predecessors {
                            if let Some(edge) = graph.get_edge_between(pred_id, node_id) {
                                let success_count = edge.get_success_count();
                                let failure_count = edge.get_failure_count();
                                let total = success_count + failure_count;

                                let is_dead_end = if total >= 3 {
                                    let failure_rate = failure_count as f32 / total as f32;
                                    failure_rate > 0.7
                                } else {
                                    edge.get_success_rate().is_some_and(|sr| sr < 0.2)
                                };

                                if is_dead_end {
                                    dead_ends.insert(event_type.clone());
                                    break; // One bad incoming edge is enough
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(dead_ends.into_iter().collect())
    }

    // Helper methods

    pub(crate) fn find_context_nodes(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<NodeId>> {
        let mut matching_nodes = Vec::new();

        if let Some(context_node) = graph.get_context_node(context_hash) {
            matching_nodes.push(context_node.id);
        }

        Ok(matching_nodes)
    }

    pub(crate) fn get_edge_weight_between(
        &self,
        graph: &Graph,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Option<EdgeWeight> {
        if let Some(edge) = graph.get_edge_between(from_node, to_node) {
            if let Some(success_rate) = edge.get_success_rate() {
                let total_observations = edge.get_success_count() + edge.get_failure_count();

                if total_observations >= 3 {
                    Some(success_rate)
                } else {
                    let prior_weight = edge.weight;
                    let evidence_weight = total_observations as f32;
                    let prior_strength = 2.0;

                    let blended = (success_rate * evidence_weight + prior_weight * prior_strength)
                        / (evidence_weight + prior_strength);
                    Some(blended.clamp(0.0, 1.0))
                }
            } else {
                Some(edge.weight)
            }
        } else {
            None
        }
    }

    pub(crate) fn count_pattern_occurrences(
        &self,
        graph: &Graph,
        from_node: NodeId,
        to_node: NodeId,
    ) -> GraphResult<u32> {
        if let Some(edge) = graph.get_edge_between(from_node, to_node) {
            Ok(edge.observation_count)
        } else {
            Ok(0)
        }
    }
}
