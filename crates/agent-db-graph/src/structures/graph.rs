// crates/agent-db-graph/src/structures/graph.rs
//
// Graph struct definition, GraphStats, Default impl, edge_text_for_bm25,
// constructors, and internal helpers (update_stats, flatten_json_to_text).

use crate::intern::Interner;
use crate::slot_vec::SlotVec;
use agent_db_core::types::{AgentId, ContextHash, EventId, Timestamp};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::adj_list::AdjList;
use super::edge::{EdgeType, GraphEdge};
use super::node::GraphNode;
use super::types::{EdgeId, NodeId};
use super::vector_index::NodeVectorIndex;

/// Graph structure with optimized storage and indexing.
///
/// Core storage uses `SlotVec` for O(1) node/edge access by monotonic ID.
#[derive(Debug)]
pub struct Graph {
    /// All nodes in the graph (O(1) access by NodeId via dense Vec)
    pub(crate) nodes: SlotVec<GraphNode>,

    /// All edges in the graph (O(1) access by EdgeId via dense Vec)
    pub(crate) edges: SlotVec<GraphEdge>,

    /// Adjacency list for fast traversal (outgoing edges, indexed by NodeId)
    pub(crate) adjacency_out: SlotVec<AdjList>,

    /// Reverse adjacency list (incoming edges, indexed by NodeId)
    pub(crate) adjacency_in: SlotVec<AdjList>,

    /// Index by node type discriminant (u8) for O(1) type-filtered queries.
    /// Key is `NodeType::discriminant()` (0–10), not a heap-allocated string.
    pub(crate) type_index: FxHashMap<u8, HashSet<NodeId>>,

    /// Temporal index for efficient time-range queries.
    /// Maps `created_at` timestamp to the nodes created at that instant.
    pub(crate) temporal_index: BTreeMap<Timestamp, SmallVec<[NodeId; 4]>>,

    /// Monotonically increasing generation counter. Incremented on every
    /// structural mutation (add/remove node or edge, merge). Used by the
    /// query cache to detect staleness without manual invalidation.
    pub(crate) generation: u64,

    /// Spatial index for context nodes
    pub(crate) context_index: FxHashMap<ContextHash, NodeId>,

    /// Agent index for quick agent lookup
    pub(crate) agent_index: FxHashMap<AgentId, NodeId>,

    /// Event index for event-node mapping
    pub(crate) event_index: FxHashMap<EventId, NodeId>,

    /// Goal index for goal-node mapping
    pub(crate) goal_index: FxHashMap<u64, NodeId>,

    /// Episode index for episode-node mapping
    pub(crate) episode_index: FxHashMap<u64, NodeId>,

    /// Memory index for memory-node mapping
    pub(crate) memory_index: FxHashMap<u64, NodeId>,

    /// Strategy index for strategy-node mapping
    pub(crate) strategy_index: FxHashMap<u64, NodeId>,

    /// Tool index for tool-node mapping (interned keys)
    pub(crate) tool_index: HashMap<Arc<str>, NodeId>,

    /// Result index for result-node mapping (interned keys)
    pub(crate) result_index: HashMap<Arc<str>, NodeId>,

    /// Claim index for claim-node mapping
    pub(crate) claim_index: FxHashMap<u64, NodeId>,

    /// Concept index for concept-node mapping (interned keys)
    pub(crate) concept_index: HashMap<Arc<str>, NodeId>,

    /// BM25 full-text search index
    pub(crate) bm25_index: crate::indexing::Bm25Index,

    /// Node vector index for semantic similarity search on graph nodes
    pub(crate) node_vector_index: NodeVectorIndex,

    /// Edge/triplet vector index for semantic similarity search on graph edges.
    /// Stores embeddings of "subject predicate object" triplet text, keyed by EdgeId.
    /// Enables triplet scoring: query vs full triplet context.
    pub(crate) edge_vector_index: NodeVectorIndex,

    /// String interner for deduplicating repeated string values
    pub(crate) interner: Interner,

    /// Statistics
    pub(crate) stats: GraphStats,

    /// Maximum number of nodes allowed (enforced at add_node)
    pub(crate) max_graph_size: usize,

    // ── Delta tracking for incremental persistence ──
    /// Nodes added or modified since last persist
    pub(crate) dirty_nodes: HashSet<NodeId>,

    /// Edges added or modified since last persist
    pub(crate) dirty_edges: HashSet<EdgeId>,

    /// Nodes deleted since last persist (need disk cleanup)
    pub(crate) deleted_nodes: HashSet<NodeId>,

    /// Edges deleted since last persist (need disk cleanup)
    pub(crate) deleted_edges: HashSet<EdgeId>,

    /// Whether adjacency metadata blob needs re-persisting
    pub(crate) adjacency_dirty: bool,

    /// Running sum of all node degrees for O(1) avg_degree computation.
    pub(crate) total_degree: u64,

    /// Broadcast channel for subscription deltas. None = no subscribers, negligible overhead.
    pub(crate) delta_tx:
        Option<tokio::sync::broadcast::Sender<crate::subscription::delta::DeltaBatch>>,
}

/// Graph statistics for monitoring and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f32,
    pub max_degree: u32,
    pub component_count: usize,
    pub largest_component_size: usize,
    pub clustering_coefficient: f32,
    pub last_updated: Timestamp,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract searchable text from a graph edge for BM25 indexing.
///
/// Converts edge labels like "preference:food" into "preference food" and
/// includes edge property values (category, details) so that queries for
/// edge-related terms can find the connected nodes.
/// Build BM25-indexable text for an edge. Uses a single String buffer
/// instead of Vec<String> + join to avoid intermediate allocations.
pub(crate) fn edge_text_for_bm25(edge: &GraphEdge) -> String {
    let mut text = String::with_capacity(64);
    match &edge.edge_type {
        EdgeType::Association {
            association_type, ..
        } => {
            // "preference:food" → "preference food"
            for (i, ch) in association_type.chars().enumerate() {
                if ch == ':' {
                    text.push(' ');
                } else {
                    text.push(ch);
                }
                // Safety: won't exceed reasonable edge type lengths
                if i > 256 {
                    break;
                }
            }
            if let Some(details) = edge.properties.get("details") {
                if let Some(cat) = details.get("category").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(cat);
                }
            }
            if let Some(cat) = edge.properties.get("category").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(cat);
            }
        },
        EdgeType::About {
            predicate: Some(p), ..
        } => {
            text.push_str(p);
        },
        _ => {},
    }
    text
}

impl Graph {
    /// Create a new empty graph with default max size (1,000,000 nodes)
    pub fn new() -> Self {
        Self::with_max_size(1_000_000)
    }

    /// Create a new empty graph with a specific max node capacity
    pub fn with_max_size(max_graph_size: usize) -> Self {
        Self {
            nodes: SlotVec::new(),
            edges: SlotVec::new(),
            adjacency_out: SlotVec::new(),
            adjacency_in: SlotVec::new(),
            type_index: FxHashMap::default(),
            temporal_index: BTreeMap::new(),
            generation: 0,
            context_index: FxHashMap::default(),
            agent_index: FxHashMap::default(),
            event_index: FxHashMap::default(),
            goal_index: FxHashMap::default(),
            episode_index: FxHashMap::default(),
            memory_index: FxHashMap::default(),
            strategy_index: FxHashMap::default(),
            tool_index: HashMap::new(),
            result_index: HashMap::new(),
            claim_index: FxHashMap::default(),
            concept_index: HashMap::new(),
            bm25_index: crate::indexing::Bm25Index::new(),
            node_vector_index: NodeVectorIndex::new(),
            edge_vector_index: NodeVectorIndex::new(),
            interner: Interner::new(),
            stats: GraphStats::default(),
            max_graph_size,
            dirty_nodes: HashSet::new(),
            dirty_edges: HashSet::new(),
            deleted_nodes: HashSet::new(),
            deleted_edges: HashSet::new(),
            adjacency_dirty: false,
            total_degree: 0,
            delta_tx: None,
        }
    }

    /// Enable the subscription delta broadcast channel.
    ///
    /// If already enabled, reuses the existing sender and returns a new receiver.
    /// Prior receivers remain valid. Channel capacity is 4096 batches.
    pub fn enable_subscriptions(
        &mut self,
    ) -> tokio::sync::broadcast::Receiver<crate::subscription::delta::DeltaBatch> {
        if let Some(tx) = &self.delta_tx {
            tx.subscribe()
        } else {
            let (tx, rx) = tokio::sync::broadcast::channel(4096);
            self.delta_tx = Some(tx);
            rx
        }
    }

    /// Update graph statistics (O(1) — uses incremental total_degree).
    pub(crate) fn update_stats(&mut self) {
        self.stats.node_count = self.nodes.len();
        self.stats.edge_count = self.edges.len();

        if !self.nodes.is_empty() {
            self.stats.avg_degree = self.total_degree as f32 / self.nodes.len() as f32;
        } else {
            self.stats.avg_degree = 0.0;
        }

        self.stats.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
    }

    /// Get graph statistics.
    ///
    /// Note: `max_degree` is computed lazily on access (O(N) scan) rather than
    /// being maintained on every mutation. Call `refresh_max_degree()` first
    /// if you need an up-to-date value.
    pub fn stats(&self) -> &GraphStats {
        &self.stats
    }

    /// Recompute `max_degree` by scanning all nodes. O(N).
    /// Call this before reading `stats().max_degree` if accuracy is needed.
    pub fn refresh_max_degree(&mut self) {
        self.stats.max_degree = self.nodes.values().map(|n| n.degree).max().unwrap_or(0);
    }

    /// Flatten a JSON value into searchable text by recursively extracting string values.
    pub(crate) fn flatten_json_to_text(value: &serde_json::Value) -> String {
        let mut parts = Vec::new();
        Self::collect_json_strings(value, &mut parts, 3);
        parts.join(" ")
    }

    fn collect_json_strings(value: &serde_json::Value, parts: &mut Vec<String>, depth: u8) {
        if depth == 0 {
            return;
        }
        match value {
            serde_json::Value::String(s) if !s.is_empty() => {
                parts.push(s.clone());
            },
            serde_json::Value::Number(n) => {
                parts.push(n.to_string());
            },
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    parts.push(k.clone());
                    Self::collect_json_strings(v, parts, depth - 1);
                }
            },
            serde_json::Value::Array(arr) => {
                for v in arr.iter().take(20) {
                    Self::collect_json_strings(v, parts, depth - 1);
                }
            },
            _ => {},
        }
    }
}
