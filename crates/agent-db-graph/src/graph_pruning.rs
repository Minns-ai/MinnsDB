//! Streaming graph pruner with bounded resource usage.
//!
//! Scans NodeHeaders (lightweight), identifies merge candidates via an
//! inverted-neighbor index, and deletes dead nodes — all within a fixed
//! read budget so the pruner never dominates I/O.
//!
//! Performance constraints:
//! - No O(N^2) comparisons; candidate generation via inverted-neighbor index
//! - Every scan is capped; partial progress is returned when limits hit
//! - NodeHeader (~40 bytes) avoids full deserialization during scoring

use std::collections::HashMap;

use agent_db_core::types::Timestamp;

use crate::graph_store::{EvictionTier, GraphStore, NodeHeader};
use crate::redb_graph_store::RedbGraphStore;
use crate::structures::{Graph, NodeId};

/// Configuration for the streaming graph pruner.
#[derive(Debug, Clone)]
pub struct GraphPruningConfig {
    /// Maximum NodeHeaders scanned per pruning pass.
    pub max_nodes_scanned_per_pass: usize,
    /// Maximum redb reads before stopping (protects I/O budget).
    pub max_redb_reads_per_pass: usize,
    /// Maximum merge candidates retained per pass (min-heap size).
    pub max_merge_candidates_per_pass: usize,
    /// Maximum neighbor edges sampled per candidate for adjacency loading.
    pub max_neighbors_sampled: usize,
    /// Minimum shared neighbors required before considering a merge.
    pub min_shared_neighbors: usize,
    /// Maximum merge partners considered per node.
    pub max_partners_per_node: usize,
    /// Jaccard similarity threshold for merging.
    pub merge_similarity_threshold: f32,
    /// Maximum merges executed per pass.
    pub max_merges_per_pass: usize,
    /// Nodes scoring below this are deleted outright.
    pub dead_threshold: f32,
    /// Maximum edges loaded when cold-loading a node for merge.
    pub max_edges_loaded_per_cold_node: usize,
}

impl Default for GraphPruningConfig {
    fn default() -> Self {
        Self {
            max_nodes_scanned_per_pass: 50_000,
            max_redb_reads_per_pass: 200_000,
            max_merge_candidates_per_pass: 2_000,
            max_neighbors_sampled: 64,
            min_shared_neighbors: 2,
            max_partners_per_node: 20,
            merge_similarity_threshold: 0.6,
            max_merges_per_pass: 200,
            dead_threshold: 0.05,
            max_edges_loaded_per_cold_node: 256,
        }
    }
}

/// Tracks remaining I/O budget. Every redb read decrements; when exhausted
/// the pruner returns partial progress.
#[derive(Debug)]
pub struct ReadBudget {
    remaining: usize,
}

impl ReadBudget {
    pub fn new(limit: usize) -> Self {
        Self { remaining: limit }
    }

    /// Consume one read. Returns `false` if budget exhausted.
    pub fn consume(&mut self, n: usize) -> bool {
        if self.remaining >= n {
            self.remaining -= n;
            true
        } else {
            self.remaining = 0;
            false
        }
    }

    pub fn exhausted(&self) -> bool {
        self.remaining == 0
    }

    pub fn remaining(&self) -> usize {
        self.remaining
    }
}

/// Result of a single pruning pass.
#[derive(Debug, Clone, Default)]
pub struct PruneResult {
    pub nodes_merged: usize,
    pub nodes_deleted: usize,
    pub total_headers_scanned: usize,
    pub budget_remaining: usize,
    pub stopped_early: bool,
}

/// Scored candidate for merge/delete consideration.
#[derive(Debug)]
struct ScoredCandidate {
    node_id: NodeId,
    score: f32,
    header: NodeHeader,
}

impl PartialEq for ScoredCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for ScoredCandidate {}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: highest score at top (max-heap)
        // We want to keep highest-scored and eject lowest-scored
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Streaming graph pruner.
pub struct GraphPruner {
    config: GraphPruningConfig,
}

impl GraphPruner {
    pub fn new(config: GraphPruningConfig) -> Self {
        Self { config }
    }

    /// Run a full pruning pass: score → merge → delete.
    ///
    /// Operates on both the hot in-memory `Graph` and the cold `RedbGraphStore`.
    /// Returns partial progress if the read budget is exhausted.
    pub fn prune_full_graph(
        &self,
        graph: &mut Graph,
        store: &mut RedbGraphStore,
        now: Timestamp,
    ) -> Result<PruneResult, crate::error::GraphError> {
        let mut budget = ReadBudget::new(self.config.max_redb_reads_per_pass);
        let mut result = PruneResult::default();

        // Phase 0: Stream NodeHeaders and score them
        let headers = store
            .scan_headers(self.config.max_nodes_scanned_per_pass)
            .map_err(|e| {
                crate::error::GraphError::OperationError(format!("scan_headers: {}", e))
            })?;

        budget.consume(headers.len());
        result.total_headers_scanned = headers.len();

        // Separate dead nodes and merge candidates
        let mut dead_nodes: Vec<NodeHeader> = Vec::new();
        let mut candidates: Vec<ScoredCandidate> = Vec::new();

        for header in headers {
            // Skip Protected tier — never prune
            if header.tier == EvictionTier::Protected {
                continue;
            }

            let score = header.score(now);

            if score < self.config.dead_threshold {
                dead_nodes.push(header);
            } else {
                candidates.push(ScoredCandidate {
                    node_id: header.node_id,
                    score,
                    header,
                });
            }
        }

        // Sort candidates by score ascending (lowest first = most likely to merge)
        candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.max_merge_candidates_per_pass);

        // Phase A: Merge via inverted neighbor index
        if !budget.exhausted() && !candidates.is_empty() {
            let merged = self.phase_merge(graph, store, &candidates, &mut budget)?;
            result.nodes_merged = merged;
        }

        if budget.exhausted() {
            result.stopped_early = true;
            result.budget_remaining = 0;
            return Ok(result);
        }

        // Phase B: Delete dead nodes
        let mut deleted = 0;
        for header in &dead_nodes {
            if budget.exhausted() {
                result.stopped_early = true;
                break;
            }

            // Delete from RedbGraphStore
            let _ = store.delete_header(header.goal_bucket, header.node_id);
            let _ = store.delete_node(header.goal_bucket, header.node_id);
            budget.consume(2);

            // If in RAM, remove from hot graph
            graph.remove_node(header.node_id);

            deleted += 1;
        }
        result.nodes_deleted = deleted;
        result.budget_remaining = budget.remaining();

        Ok(result)
    }

    /// Phase A: Build inverted-neighbor index and merge similar nodes.
    fn phase_merge(
        &self,
        graph: &mut Graph,
        store: &mut RedbGraphStore,
        candidates: &[ScoredCandidate],
        budget: &mut ReadBudget,
    ) -> Result<usize, crate::error::GraphError> {
        // Step 1: Load adjacency for each candidate (capped)
        let mut candidate_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut candidate_discriminants: HashMap<NodeId, u8> = HashMap::new();

        for cand in candidates {
            if budget.exhausted() {
                break;
            }

            let bucket = cand.header.goal_bucket;
            candidate_discriminants.insert(cand.node_id, cand.header.node_type_discriminant);

            // Try RAM first
            let neighbors: Vec<NodeId> = {
                let ram_neighbors = graph.get_neighbors(cand.node_id);
                if !ram_neighbors.is_empty() {
                    ram_neighbors
                        .into_iter()
                        .take(self.config.max_neighbors_sampled)
                        .collect()
                } else {
                    // Fall back to cold store
                    budget.consume(1);
                    store
                        .get_neighbors(bucket, cand.node_id)
                        .unwrap_or_default()
                        .into_iter()
                        .take(self.config.max_neighbors_sampled)
                        .collect()
                }
            };

            candidate_neighbors.insert(cand.node_id, neighbors);
        }

        // Step 2: Build inverted map: neighbor_id → Vec<candidate_id>
        let mut inverted: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (cand_id, neighbors) in &candidate_neighbors {
            for &neighbor in neighbors {
                inverted.entry(neighbor).or_default().push(*cand_id);
            }
        }

        // Step 3: Generate merge pairs
        // For each neighbor that has multiple candidates pointing to it,
        // those candidates share that neighbor → potential merge
        let mut merge_pairs: Vec<(NodeId, NodeId, f32)> = Vec::new();
        let mut seen_pairs: std::collections::HashSet<(NodeId, NodeId)> =
            std::collections::HashSet::new();

        for (cand_id, neighbors) in &candidate_neighbors {
            if merge_pairs.len() >= self.config.max_merges_per_pass * 2 {
                break;
            }

            // Find partners via inverted index
            let mut partner_shared: HashMap<NodeId, usize> = HashMap::new();
            for &neighbor in neighbors {
                if let Some(cands) = inverted.get(&neighbor) {
                    for &other in cands {
                        if other != *cand_id {
                            *partner_shared.entry(other).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Filter: same type, enough shared neighbors
            let cand_disc = candidate_discriminants.get(cand_id).copied().unwrap_or(255);
            let mut partners: Vec<(NodeId, usize)> = partner_shared
                .into_iter()
                .filter(|(other, shared)| {
                    *shared >= self.config.min_shared_neighbors
                        && candidate_discriminants.get(other).copied() == Some(cand_disc)
                })
                .collect();

            // Sort by shared count descending, cap
            partners.sort_by(|a, b| b.1.cmp(&a.1));
            partners.truncate(self.config.max_partners_per_node);

            for (partner_id, shared_count) in partners {
                let pair = if *cand_id < partner_id {
                    (*cand_id, partner_id)
                } else {
                    (partner_id, *cand_id)
                };

                if seen_pairs.contains(&pair) {
                    continue;
                }

                // Jaccard similarity
                let cand_neigh = &candidate_neighbors[cand_id];
                let partner_neigh = match candidate_neighbors.get(&partner_id) {
                    Some(n) => n,
                    None => continue,
                };

                let union_size = {
                    let mut all: std::collections::HashSet<NodeId> =
                        cand_neigh.iter().copied().collect();
                    all.extend(partner_neigh.iter().copied());
                    all.len()
                };

                let jaccard = if union_size > 0 {
                    shared_count as f32 / union_size as f32
                } else {
                    0.0
                };

                if jaccard >= self.config.merge_similarity_threshold {
                    seen_pairs.insert(pair);
                    merge_pairs.push((pair.0, pair.1, jaccard));
                }
            }
        }

        // Sort by similarity descending (merge most similar first)
        merge_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        merge_pairs.truncate(self.config.max_merges_per_pass);

        // Step 4: Execute merges
        let mut merged_count = 0;
        let mut already_merged: std::collections::HashSet<NodeId> =
            std::collections::HashSet::new();

        for (node_a, node_b, _similarity) in &merge_pairs {
            if budget.exhausted() {
                break;
            }

            if already_merged.contains(node_a) || already_merged.contains(node_b) {
                continue;
            }

            // Pick survivor as the one with higher score (or lower ID as tiebreaker)
            let (survivor, absorbed) = {
                let score_a = candidates
                    .iter()
                    .find(|c| c.node_id == *node_a)
                    .map(|c| c.score)
                    .unwrap_or(0.0);
                let score_b = candidates
                    .iter()
                    .find(|c| c.node_id == *node_b)
                    .map(|c| c.score)
                    .unwrap_or(0.0);

                if score_a >= score_b {
                    (*node_a, *node_b)
                } else {
                    (*node_b, *node_a)
                }
            };

            // Merge in RAM graph
            match graph.merge_nodes(survivor, absorbed) {
                Ok(_survivor_node) => {
                    // Write updated survivor + header to store
                    if let Some(s_node) = graph.get_node(survivor) {
                        let bucket = s_node.node_type.goal_bucket();
                        let _ = store.add_node(bucket, s_node.clone());
                        let _ = store.store_header(bucket, NodeHeader::from_node(s_node, bucket));
                        budget.consume(2);
                    }

                    // Delete absorbed from store
                    let absorbed_header = candidates
                        .iter()
                        .find(|c| c.node_id == absorbed)
                        .map(|c| &c.header);
                    if let Some(ah) = absorbed_header {
                        let _ = store.delete_node(ah.goal_bucket, absorbed);
                        let _ = store.delete_header(ah.goal_bucket, absorbed);
                        budget.consume(2);
                    }

                    already_merged.insert(absorbed);
                    merged_count += 1;
                },
                Err(_) => {
                    // Nodes might not both be in RAM — skip
                    continue;
                },
            }
        }

        Ok(merged_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{GraphNode, NodeType};
    use agent_db_storage::{RedbBackend, RedbConfig};
    use std::sync::Arc;

    #[test]
    fn test_read_budget() {
        let mut budget = ReadBudget::new(10);
        assert!(!budget.exhausted());
        assert!(budget.consume(5));
        assert_eq!(budget.remaining(), 5);
        assert!(budget.consume(5));
        assert!(budget.exhausted());
        assert!(!budget.consume(1));
    }

    #[test]
    fn test_pruning_config_defaults() {
        let config = GraphPruningConfig::default();
        assert_eq!(config.max_nodes_scanned_per_pass, 50_000);
        assert_eq!(config.max_redb_reads_per_pass, 200_000);
        assert_eq!(config.max_merge_candidates_per_pass, 2_000);
        assert_eq!(config.max_merges_per_pass, 200);
    }

    #[test]
    fn test_prune_result_default() {
        let result = PruneResult::default();
        assert_eq!(result.nodes_merged, 0);
        assert_eq!(result.nodes_deleted, 0);
        assert!(!result.stopped_early);
    }

    #[test]
    fn test_pruner_empty_graph() {
        let config = GraphPruningConfig::default();
        let pruner = GraphPruner::new(config);

        let mut graph = Graph::new();
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            RedbBackend::open(RedbConfig {
                data_path: dir.path().join("test.redb"),
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            })
            .unwrap(),
        );
        let mut store = RedbGraphStore::new(backend, 4);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;

        let result = pruner
            .prune_full_graph(&mut graph, &mut store, now)
            .unwrap();
        assert_eq!(result.nodes_merged, 0);
        assert_eq!(result.nodes_deleted, 0);
        assert_eq!(result.total_headers_scanned, 0);
        assert!(!result.stopped_early);
    }

    #[test]
    fn test_pruner_respects_protected_tier() {
        let config = GraphPruningConfig {
            dead_threshold: 999.0, // Everything would be "dead" except Protected
            ..Default::default()
        };
        let pruner = GraphPruner::new(config);

        let mut graph = Graph::new();
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            RedbBackend::open(RedbConfig {
                data_path: dir.path().join("test.redb"),
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            })
            .unwrap(),
        );
        let mut store = RedbGraphStore::new(backend, 4);

        // Add a Protected node (Agent)
        let node = GraphNode::new(NodeType::Agent {
            agent_id: 1,
            agent_type: "test".to_string(),
            capabilities: vec![],
        });
        let node_id = graph.add_node(node.clone());
        let bucket = graph.get_node(node_id).unwrap().node_type.goal_bucket();
        let stored_node = graph.get_node(node_id).unwrap().clone();
        store.add_node(bucket, stored_node.clone()).unwrap();
        store
            .store_header(bucket, NodeHeader::from_node(&stored_node, bucket))
            .unwrap();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;

        let result = pruner
            .prune_full_graph(&mut graph, &mut store, now)
            .unwrap();

        // Protected node should NOT be deleted
        assert_eq!(result.nodes_deleted, 0);
        assert!(graph.get_node(node_id).is_some());
    }

    #[test]
    fn test_pruner_deletes_dead_ephemeral_nodes() {
        let config = GraphPruningConfig {
            dead_threshold: 1000.0, // All Ephemeral nodes will score below this
            ..Default::default()
        };
        let pruner = GraphPruner::new(config);

        let mut graph = Graph::new();
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            RedbBackend::open(RedbConfig {
                data_path: dir.path().join("test.redb"),
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            })
            .unwrap(),
        );
        let mut store = RedbGraphStore::new(backend, 4);

        // Add an Ephemeral node (Result) with old timestamp
        let mut node = GraphNode::new(NodeType::Result {
            result_key: "old_result".to_string(),
            result_type: "test".to_string(),
            summary: "old".to_string(),
        });
        node.created_at = 1000; // Very old
        node.updated_at = 1000;

        let node_id = graph.add_node(node);
        let stored = graph.get_node(node_id).unwrap().clone();
        let bucket = stored.node_type.goal_bucket();
        store.add_node(bucket, stored.clone()).unwrap();
        store
            .store_header(bucket, NodeHeader::from_node(&stored, bucket))
            .unwrap();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;

        let result = pruner
            .prune_full_graph(&mut graph, &mut store, now)
            .unwrap();

        assert_eq!(result.nodes_deleted, 1);
        assert!(graph.get_node(node_id).is_none());
    }

    #[test]
    fn test_pruner_stops_early_on_budget() {
        let config = GraphPruningConfig {
            max_redb_reads_per_pass: 1, // Tiny budget
            dead_threshold: 1000.0,
            ..Default::default()
        };
        let pruner = GraphPruner::new(config);

        let mut graph = Graph::new();
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            RedbBackend::open(RedbConfig {
                data_path: dir.path().join("test.redb"),
                cache_size_bytes: 64 * 1024 * 1024,
                repair_on_open: false,
            })
            .unwrap(),
        );
        let mut store = RedbGraphStore::new(backend, 4);

        // Add several ephemeral nodes
        for i in 0..5 {
            let mut node = GraphNode::new(NodeType::Result {
                result_key: format!("r{}", i),
                result_type: "test".to_string(),
                summary: format!("result {}", i),
            });
            node.created_at = 1000;
            node.updated_at = 1000;
            let nid = graph.add_node(node);
            let stored = graph.get_node(nid).unwrap().clone();
            let bucket = stored.node_type.goal_bucket();
            store.add_node(bucket, stored.clone()).unwrap();
            store
                .store_header(bucket, NodeHeader::from_node(&stored, bucket))
                .unwrap();
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;

        let result = pruner
            .prune_full_graph(&mut graph, &mut store, now)
            .unwrap();
        assert!(result.stopped_early);
    }
}
