//! Random Walk infrastructure — foundation for PersonalizedPageRank, Node2Vec.
//!
//! Provides configurable random walks over the graph with:
//! - Walk length and restart probability (for PPR-style walks)
//! - Edge-weight-biased neighbor selection
//! - Multiple walks per node for statistical stability
//! - Deterministic mode via explicit seed

use crate::structures::{Graph, NodeId};
use crate::GraphResult;
use std::collections::HashMap;

/// Configuration for random walks.
#[derive(Debug, Clone)]
pub struct RandomWalkConfig {
    /// Maximum number of steps per walk.
    pub walk_length: usize,

    /// Probability of teleporting back to the start node at each step (0.0–1.0).
    /// Set to 0.0 for pure random walks, ~0.15 for PersonalizedPageRank.
    pub restart_probability: f64,

    /// Number of independent walks to perform per source node.
    pub num_walks: usize,

    /// Whether to weight neighbor selection by edge weight.
    /// If false, all outgoing neighbors are equally likely.
    pub weighted: bool,

    /// Optional seed for reproducible walks.
    pub seed: Option<u64>,
}

impl Default for RandomWalkConfig {
    fn default() -> Self {
        Self {
            walk_length: 80,
            restart_probability: 0.0,
            num_walks: 10,
            weighted: true,
            seed: None,
        }
    }
}

/// A single random walk path.
#[derive(Debug, Clone)]
pub struct WalkPath {
    /// Ordered sequence of visited node IDs (includes start node).
    pub nodes: Vec<NodeId>,

    /// Whether the walk terminated early due to a dead-end (no outgoing edges).
    pub terminated_early: bool,
}

/// Aggregate statistics from multiple random walks.
#[derive(Debug, Clone)]
pub struct RandomWalkResult {
    /// All walk paths generated.
    pub walks: Vec<WalkPath>,

    /// Visit frequency: node_id → number of times visited across all walks.
    pub visit_counts: HashMap<NodeId, usize>,

    /// Total steps taken across all walks.
    pub total_steps: usize,
}

/// Simple, fast PRNG (xoshiro256**). Avoids pulling in `rand` as a dependency.
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to seed xoshiro256**
        let mut state = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Return a float in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Pick a random index in 0..len using rejection sampling.
    fn index(&mut self, len: usize) -> usize {
        (self.next_u64() as usize) % len
    }
}

/// Random walk engine.
pub struct RandomWalker {
    config: RandomWalkConfig,
}

impl RandomWalker {
    /// Create with default config.
    pub fn new() -> Self {
        Self {
            config: RandomWalkConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: RandomWalkConfig) -> Self {
        Self { config }
    }

    /// Perform random walks starting from a single source node.
    pub fn walk(&self, graph: &Graph, source: NodeId) -> GraphResult<RandomWalkResult> {
        if graph.get_node(source).is_none() {
            return Err(crate::GraphError::NodeNotFound(format!(
                "source node {} not found",
                source
            )));
        }

        let seed = self.config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        });

        let mut rng = Rng::new(seed);
        let mut walks = Vec::with_capacity(self.config.num_walks);
        let mut visit_counts: HashMap<NodeId, usize> = HashMap::new();
        let mut total_steps = 0;

        for _ in 0..self.config.num_walks {
            let path = self.single_walk(graph, source, &mut rng);
            for &node in &path.nodes {
                *visit_counts.entry(node).or_insert(0) += 1;
            }
            total_steps += path.nodes.len().saturating_sub(1);
            walks.push(path);
        }

        Ok(RandomWalkResult {
            walks,
            visit_counts,
            total_steps,
        })
    }

    /// Perform random walks from every node in the graph.
    /// Returns walks grouped by source node.
    pub fn walk_all(
        &self,
        graph: &Graph,
    ) -> GraphResult<HashMap<NodeId, RandomWalkResult>> {
        let nodes = graph.get_all_node_ids();
        let mut results = HashMap::with_capacity(nodes.len());

        for &node in &nodes {
            results.insert(node, self.walk(graph, node)?);
        }

        Ok(results)
    }

    /// Compute PersonalizedPageRank scores from a source node.
    ///
    /// Uses random walks with restart to estimate the stationary distribution
    /// of a random surfer who teleports back to `source` with probability
    /// `restart_probability` at each step.
    ///
    /// Returns node_id → PPR score (normalized to sum to 1.0).
    pub fn personalized_pagerank(
        &self,
        graph: &Graph,
        source: NodeId,
    ) -> GraphResult<HashMap<NodeId, f64>> {
        let result = self.walk(graph, source)?;

        let total_visits: usize = result.visit_counts.values().sum();
        if total_visits == 0 {
            return Ok(HashMap::new());
        }

        let scores: HashMap<NodeId, f64> = result
            .visit_counts
            .iter()
            .map(|(&node, &count)| (node, count as f64 / total_visits as f64))
            .collect();

        Ok(scores)
    }

    /// Perform a single random walk.
    fn single_walk(&self, graph: &Graph, start: NodeId, rng: &mut Rng) -> WalkPath {
        let mut path = Vec::with_capacity(self.config.walk_length + 1);
        path.push(start);

        let mut current = start;
        let mut terminated_early = false;

        for _ in 0..self.config.walk_length {
            // Restart check
            if self.config.restart_probability > 0.0
                && rng.next_f64() < self.config.restart_probability
            {
                current = start;
                path.push(current);
                continue;
            }

            // Get outgoing edges
            let edges = graph.get_edges_from(current);
            if edges.is_empty() {
                terminated_early = true;
                break;
            }

            // Select next node
            let next = if self.config.weighted {
                self.weighted_select(&edges, rng)
            } else {
                let idx = rng.index(edges.len());
                edges[idx].target
            };

            current = next;
            path.push(current);
        }

        WalkPath {
            nodes: path,
            terminated_early,
        }
    }

    /// Select a neighbor proportional to edge weight.
    fn weighted_select(
        &self,
        edges: &[&crate::structures::GraphEdge],
        rng: &mut Rng,
    ) -> NodeId {
        let total_weight: f64 = edges.iter().map(|e| e.weight.max(0.0) as f64).sum();

        if total_weight <= 0.0 {
            // Fallback to uniform if all weights are zero/negative
            let idx = rng.index(edges.len());
            return edges[idx].target;
        }

        let threshold = rng.next_f64() * total_weight;
        let mut cumulative = 0.0;

        for edge in edges {
            cumulative += edge.weight.max(0.0) as f64;
            if cumulative >= threshold {
                return edge.target;
            }
        }

        // Fallback (shouldn't happen with valid weights)
        edges.last().unwrap().target
    }
}

impl Default for RandomWalker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn make_node(event_id: u128) -> GraphNode {
        GraphNode::new(NodeType::Event {
            event_id,
            event_type: "test".into(),
            significance: 0.5,
        })
    }

    fn make_edge(source: NodeId, target: NodeId, weight: f32) -> GraphEdge {
        GraphEdge::new(
            source,
            target,
            EdgeType::Association {
                association_type: "test".into(),
                evidence_count: 1,
                statistical_significance: weight,
            },
            weight,
        )
    }

    #[test]
    fn test_walk_linear_chain() {
        // A -> B -> C (no other options, walk must follow chain)
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        let c = graph.add_node(make_node(3)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, c, 1.0));

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 10,
            num_walks: 1,
            seed: Some(42),
            ..Default::default()
        });

        let result = walker.walk(&graph, a).unwrap();
        assert_eq!(result.walks.len(), 1);

        let path = &result.walks[0];
        // Must start at a, go to b, then c, then stop (dead end)
        assert_eq!(path.nodes[0], a);
        assert_eq!(path.nodes[1], b);
        assert_eq!(path.nodes[2], c);
        assert!(path.terminated_early);
    }

    #[test]
    fn test_walk_cycle() {
        // A -> B -> A (cycle), walk should not terminate early
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, a, 1.0));

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 10,
            num_walks: 1,
            seed: Some(42),
            ..Default::default()
        });

        let result = walker.walk(&graph, a).unwrap();
        let path = &result.walks[0];

        assert_eq!(path.nodes.len(), 11); // start + 10 steps
        assert!(!path.terminated_early);
    }

    #[test]
    fn test_walk_isolated_node() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 10,
            num_walks: 3,
            seed: Some(42),
            ..Default::default()
        });

        let result = walker.walk(&graph, a).unwrap();
        assert_eq!(result.walks.len(), 3);

        for path in &result.walks {
            assert_eq!(path.nodes, vec![a]);
            assert!(path.terminated_early);
        }
    }

    #[test]
    fn test_walk_source_not_found() {
        let graph = Graph::new();
        let walker = RandomWalker::new();
        let result = walker.walk(&graph, 999);
        assert!(result.is_err());
    }

    #[test]
    fn test_visit_counts() {
        // A -> B -> C -> B (B should be visited most)
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        let c = graph.add_node(make_node(3)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, c, 1.0));
        graph.add_edge(make_edge(c, b, 1.0)); // cycle on b-c

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 100,
            num_walks: 10,
            seed: Some(42),
            ..Default::default()
        });

        let result = walker.walk(&graph, a).unwrap();

        // B and C should be heavily visited due to the cycle
        let b_visits = result.visit_counts.get(&b).copied().unwrap_or(0);
        let c_visits = result.visit_counts.get(&c).copied().unwrap_or(0);
        assert!(b_visits > 10, "B should be frequently visited: {}", b_visits);
        assert!(c_visits > 10, "C should be frequently visited: {}", c_visits);
    }

    #[test]
    fn test_weighted_selection_bias() {
        // A -> B (weight 0.01), A -> C (weight 10.0)
        // With many walks, C should be visited far more than B
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        let c = graph.add_node(make_node(3)).unwrap();

        graph.add_edge(make_edge(a, b, 0.01));
        graph.add_edge(make_edge(a, c, 10.0));

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 1, // single step
            num_walks: 1000,
            weighted: true,
            seed: Some(42),
            ..Default::default()
        });

        let result = walker.walk(&graph, a).unwrap();
        let b_visits = result.visit_counts.get(&b).copied().unwrap_or(0);
        let c_visits = result.visit_counts.get(&c).copied().unwrap_or(0);

        // C should dominate due to much higher weight
        assert!(
            c_visits > b_visits * 5,
            "C ({}) should be visited much more than B ({})",
            c_visits,
            b_visits
        );
    }

    #[test]
    fn test_restart_probability() {
        // A -> B -> C (chain), high restart = keep coming back to A
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        let c = graph.add_node(make_node(3)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, c, 1.0));
        // No edges from C, so without restart walks die at C

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 100,
            num_walks: 50,
            restart_probability: 0.5, // 50% chance of restart each step
            seed: Some(42),
            ..Default::default()
        });

        let result = walker.walk(&graph, a).unwrap();
        let a_visits = result.visit_counts.get(&a).copied().unwrap_or(0);

        // A should be visited many times due to restarts
        assert!(
            a_visits > 100,
            "A should be visited frequently with restarts: {}",
            a_visits
        );
    }

    #[test]
    fn test_personalized_pagerank() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(b, a, 1.0));

        let walker = RandomWalker::with_config(RandomWalkConfig {
            walk_length: 100,
            num_walks: 100,
            restart_probability: 0.15,
            seed: Some(42),
            ..Default::default()
        });

        let ppr = walker.personalized_pagerank(&graph, a).unwrap();

        // Both nodes should have positive scores
        assert!(ppr.get(&a).copied().unwrap_or(0.0) > 0.0);
        assert!(ppr.get(&b).copied().unwrap_or(0.0) > 0.0);

        // Scores should sum to ~1.0
        let total: f64 = ppr.values().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "PPR scores should sum to 1.0, got {}",
            total
        );

        // Source node should have higher PPR due to restart bias
        let a_score = ppr[&a];
        let b_score = ppr[&b];
        assert!(
            a_score > b_score,
            "Source should have higher PPR: a={}, b={}",
            a_score,
            b_score
        );
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node(1)).unwrap();
        let b = graph.add_node(make_node(2)).unwrap();
        let c = graph.add_node(make_node(3)).unwrap();

        graph.add_edge(make_edge(a, b, 1.0));
        graph.add_edge(make_edge(a, c, 1.0));
        graph.add_edge(make_edge(b, a, 1.0));
        graph.add_edge(make_edge(c, a, 1.0));

        let config = RandomWalkConfig {
            walk_length: 20,
            num_walks: 5,
            seed: Some(12345),
            ..Default::default()
        };

        let walker1 = RandomWalker::with_config(config.clone());
        let walker2 = RandomWalker::with_config(config);

        let result1 = walker1.walk(&graph, a).unwrap();
        let result2 = walker2.walk(&graph, a).unwrap();

        // Same seed → same walks
        for (w1, w2) in result1.walks.iter().zip(result2.walks.iter()) {
            assert_eq!(w1.nodes, w2.nodes, "Walks should be identical with same seed");
        }
    }
}
