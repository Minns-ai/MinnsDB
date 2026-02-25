//! Query template registry for mapping intent + entities to GraphQuery.

use super::entity::ResolvedEntity;
use super::intent::{PathAlgorithm, QueryIntent, RankingMetric};
use crate::structures::{Depth, Direction};
use crate::traversal::{EdgeFilterExpr, GraphQuery, Instruction, NodeFilterExpr, TraversalRequest};

/// A slot type that a template requires.
#[derive(Debug, Clone, PartialEq)]
pub enum SlotType {
    SourceNode,
    TargetNode,
    NodeTypeFilter,
    EdgeTypeFilter,
    DepthLimit,
}

/// Parameters extracted from the question for template instantiation.
#[derive(Debug, Clone)]
pub struct TemplateParams {
    pub depth: Option<u32>,
    pub limit: Option<usize>,
    pub direction: Direction,
    pub edge_type: Option<String>,
    pub node_type: Option<String>,
    /// Natural language node filters (Enhancement 9).
    pub node_filters: Vec<NodeFilterExpr>,
    /// Natural language edge filters (Enhancement 9).
    pub edge_filters: Vec<EdgeFilterExpr>,
}

impl Default for TemplateParams {
    fn default() -> Self {
        Self {
            depth: None,
            limit: None,
            direction: Direction::Both,
            edge_type: None,
            node_type: None,
            node_filters: Vec::new(),
            edge_filters: Vec::new(),
        }
    }
}

/// A query template that maps intent + entities to a GraphQuery.
pub struct QueryTemplate {
    pub name: &'static str,
    pub required_slots: Vec<SlotType>,
    pub matches: fn(&QueryIntent) -> bool,
    pub build: fn(&[ResolvedEntity], &TemplateParams) -> Option<GraphQuery>,
}

/// Registry of built-in query templates.
pub struct TemplateRegistry {
    templates: Vec<QueryTemplate>,
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self {
            templates: builtin_templates(),
        }
    }
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find the best matching template for the given intent and resolved entities.
    pub fn match_template(
        &self,
        intent: &QueryIntent,
        resolved: &[ResolvedEntity],
        question: &str,
    ) -> Option<(&QueryTemplate, TemplateParams)> {
        let params = extract_params(question, intent);

        for template in &self.templates {
            if !(template.matches)(intent) {
                continue;
            }

            // Check required slots are satisfied
            let slots_ok = template.required_slots.iter().all(|slot| match slot {
                SlotType::SourceNode => !resolved.is_empty(),
                SlotType::TargetNode => resolved.len() >= 2,
                SlotType::NodeTypeFilter => params.node_type.is_some(),
                SlotType::EdgeTypeFilter => params.edge_type.is_some(),
                SlotType::DepthLimit => params.depth.is_some(),
            });

            if slots_ok {
                return Some((template, params));
            }
        }

        None
    }
}

/// Extract query parameters from the question text.
fn extract_params(question: &str, intent: &QueryIntent) -> TemplateParams {
    let q = question.to_lowercase();
    let mut params = TemplateParams::default();

    // Extract depth/distance from phrases like "within 3 hops", "depth 5"
    for window in q.split_whitespace().collect::<Vec<_>>().windows(2) {
        if matches!(window[0], "within" | "depth" | "radius" | "distance") {
            if let Ok(n) = window[1]
                .trim_matches(|c: char| !c.is_ascii_digit())
                .parse::<u32>()
            {
                params.depth = Some(n.min(10));
            }
        }
        if window[1] == "hops" || window[1] == "hop" {
            if let Ok(n) = window[0].parse::<u32>() {
                params.depth = Some(n.min(10));
            }
        }
    }

    // Multi-hop detection (Enhancement 2)
    if params.depth.is_none() {
        if let Some(hops) = super::intent::detect_multi_hop(&q) {
            params.depth = Some(hops);
        }
    }

    // Extract limit from "top N", "first N"
    for window in q.split_whitespace().collect::<Vec<_>>().windows(2) {
        if matches!(window[0], "top" | "first" | "limit") {
            if let Ok(n) = window[1]
                .trim_matches(|c: char| !c.is_ascii_digit())
                .parse::<usize>()
            {
                params.limit = Some(n.min(100));
            }
        }
    }

    // Direction from intent
    match intent {
        QueryIntent::FindNeighbors {
            direction,
            edge_hint,
        } => {
            params.direction = *direction;
            params.edge_type = edge_hint.clone();
        },
        QueryIntent::TemporalChain { .. } => {
            params.direction = Direction::Out;
            if params.edge_type.is_none() {
                params.edge_type = Some("Temporal".to_string());
            }
        },
        QueryIntent::FilteredTraversal {
            node_type_filter,
            edge_type_filter,
        } => {
            params.node_type = node_type_filter.clone();
            params.edge_type = edge_type_filter.clone();
        },
        _ => {},
    }

    // Extract natural language filters (Enhancement 9)
    extract_nl_filters(&q, &mut params);

    params
}

/// Extract natural language filters from the question text.
fn extract_nl_filters(q: &str, params: &mut TemplateParams) {
    // Weight threshold: "weight > 0.8", "weight above 0.5"
    if let Some(w) = extract_weight_threshold(q) {
        params
            .edge_filters
            .push(EdgeFilterExpr::MinWeight(ordered_float::OrderedFloat(w)));
    }

    // Edge type filter: "causal edges", "temporal relationships"
    if let Some(et) = extract_edge_type_from_nl(q) {
        if params.edge_type.is_none() {
            params.edge_type = Some(et.clone());
        }
        params.edge_filters.push(EdgeFilterExpr::ByType(et));
    }

    // Minimum degree: "degree > 5", "at least 3 connections"
    if let Some(d) = extract_min_degree(q) {
        params.node_filters.push(NodeFilterExpr::MinDegree(d));
    }
}

/// Parse "weight > 0.8" or "weight above 0.5" patterns.
fn extract_weight_threshold(q: &str) -> Option<f32> {
    let words: Vec<&str> = q.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        if words[i] == "weight"
            && (words[i + 1] == ">" || words[i + 1] == "above" || words[i + 1] == "greater")
        {
            let val_str =
                if words[i + 1] == "greater" && i + 3 < words.len() && words[i + 2] == "than" {
                    words[i + 3]
                } else {
                    words[i + 2]
                };
            if let Ok(v) = val_str.parse::<f32>() {
                if (0.0..=1.0).contains(&v) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Parse "causal edges", "temporal relationships" patterns.
fn extract_edge_type_from_nl(q: &str) -> Option<String> {
    if super::intent::contains_any(q, &["causal edge", "causal relationship", "causality"]) {
        Some("Causality".to_string())
    } else if super::intent::contains_any(q, &["temporal edge", "temporal relationship"]) {
        Some("Temporal".to_string())
    } else if super::intent::contains_any(q, &["association edge", "association"]) {
        Some("Association".to_string())
    } else {
        None
    }
}

/// Parse "degree > 5" or "at least 3 connections" patterns.
fn extract_min_degree(q: &str) -> Option<u32> {
    let words: Vec<&str> = q.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        if words[i] == "degree" && (words[i + 1] == ">" || words[i + 1] == "above") {
            if let Ok(v) = words[i + 2].parse::<u32>() {
                return Some(v.min(1000));
            }
        }
        if words[i] == "at" && i + 2 < words.len() && words[i + 1] == "least" {
            if let Ok(v) = words[i + 2]
                .trim_matches(|c: char| !c.is_ascii_digit())
                .parse::<u32>()
            {
                if i + 3 < words.len()
                    && super::intent::contains_any(
                        words[i + 3],
                        &["connection", "link", "edge", "degree"],
                    )
                {
                    return Some(v.min(1000));
                }
            }
        }
    }
    None
}

/// Build the set of built-in query templates.
fn builtin_templates() -> Vec<QueryTemplate> {
    vec![
        // Shortest path (2 entities required)
        QueryTemplate {
            name: "shortest_path",
            required_slots: vec![SlotType::SourceNode, SlotType::TargetNode],
            matches: |intent| {
                matches!(
                    intent,
                    QueryIntent::FindPath {
                        algorithm: PathAlgorithm::Shortest
                    }
                )
            },
            build: |resolved, _params| {
                let start = resolved.first()?.node_id;
                let end = resolved.get(1)?.node_id;
                Some(GraphQuery::ShortestPath { start, end })
            },
        },
        // K-shortest paths
        QueryTemplate {
            name: "k_shortest_paths",
            required_slots: vec![SlotType::SourceNode, SlotType::TargetNode],
            matches: |intent| {
                matches!(
                    intent,
                    QueryIntent::FindPath {
                        algorithm: PathAlgorithm::KShortest(_)
                    }
                )
            },
            build: |resolved, _params| {
                let start = resolved.first()?.node_id;
                let end = resolved.get(1)?.node_id;
                // K is encoded in the intent
                Some(GraphQuery::KShortestPaths { start, end, k: 3 })
            },
        },
        // Bidirectional path
        QueryTemplate {
            name: "bidirectional_path",
            required_slots: vec![SlotType::SourceNode, SlotType::TargetNode],
            matches: |intent| {
                matches!(
                    intent,
                    QueryIntent::FindPath {
                        algorithm: PathAlgorithm::Bidirectional
                    }
                )
            },
            build: |resolved, _params| {
                let start = resolved.first()?.node_id;
                let end = resolved.get(1)?.node_id;
                Some(GraphQuery::BidirectionalPath { start, end })
            },
        },
        // Neighbors within distance
        QueryTemplate {
            name: "neighbors",
            required_slots: vec![SlotType::SourceNode],
            matches: |intent| matches!(intent, QueryIntent::FindNeighbors { .. }),
            build: |resolved, params| {
                let start = resolved.first()?.node_id;
                let max_distance = params.depth.unwrap_or(1);
                Some(GraphQuery::NeighborsWithinDistance {
                    start,
                    max_distance,
                })
            },
        },
        // Filtered traversal (e.g., "What tools did X use?")
        QueryTemplate {
            name: "filtered_traversal",
            required_slots: vec![SlotType::SourceNode],
            matches: |intent| matches!(intent, QueryIntent::FilteredTraversal { .. }),
            build: |resolved, params| {
                let start = resolved.first()?.node_id;
                let mut node_filters = Vec::new();
                if let Some(ref nt) = params.node_type {
                    node_filters.push(NodeFilterExpr::ByType(nt.clone()));
                }
                let mut edge_filters = Vec::new();
                if let Some(ref et) = params.edge_type {
                    edge_filters.push(EdgeFilterExpr::ByType(et.clone()));
                }
                Some(GraphQuery::RecursiveTraversal(TraversalRequest {
                    start,
                    direction: params.direction,
                    depth: Depth::Range(1, params.depth.unwrap_or(3)),
                    instruction: Instruction::Collect,
                    node_filters,
                    edge_filters,
                    max_nodes_visited: Some(1000),
                    max_edges_traversed: Some(5000),
                    time_window: None,
                }))
            },
        },
        // Subgraph extraction
        QueryTemplate {
            name: "subgraph",
            required_slots: vec![SlotType::SourceNode],
            matches: |intent| matches!(intent, QueryIntent::Subgraph { .. }),
            build: |resolved, params| {
                let center = resolved.first()?.node_id;
                let radius = params.depth.unwrap_or(2);
                Some(GraphQuery::Subgraph {
                    center,
                    radius,
                    node_types: None,
                })
            },
        },
        // Temporal chain (after / before)
        QueryTemplate {
            name: "temporal_chain",
            required_slots: vec![SlotType::SourceNode],
            matches: |intent| matches!(intent, QueryIntent::TemporalChain { .. }),
            build: |resolved, params| {
                let start = resolved.first()?.node_id;
                let mut edge_filters = Vec::new();
                if let Some(ref et) = params.edge_type {
                    edge_filters.push(EdgeFilterExpr::ByType(et.clone()));
                }
                Some(GraphQuery::RecursiveTraversal(TraversalRequest {
                    start,
                    direction: params.direction,
                    depth: Depth::Range(1, params.depth.unwrap_or(5)),
                    instruction: Instruction::Collect,
                    node_filters: Vec::new(),
                    edge_filters,
                    max_nodes_visited: Some(500),
                    max_edges_traversed: Some(2000),
                    time_window: None,
                }))
            },
        },
        // PageRank ranking
        QueryTemplate {
            name: "pagerank",
            required_slots: vec![],
            matches: |intent| {
                matches!(
                    intent,
                    QueryIntent::Ranking {
                        metric: RankingMetric::PageRank
                    }
                )
            },
            build: |_resolved, _params| {
                Some(GraphQuery::PageRank {
                    iterations: 20,
                    damping_factor: 0.85,
                })
            },
        },
        // Centrality / degree ranking
        QueryTemplate {
            name: "centrality_ranking",
            required_slots: vec![],
            matches: |intent| {
                matches!(
                    intent,
                    QueryIntent::Ranking {
                        metric: RankingMetric::Centrality | RankingMetric::Degree
                    }
                )
            },
            build: |_resolved, _params| {
                // Use PageRank as the best available ranking
                Some(GraphQuery::PageRank {
                    iterations: 20,
                    damping_factor: 0.85,
                })
            },
        },
        // Similarity search
        QueryTemplate {
            name: "similarity_search",
            required_slots: vec![SlotType::SourceNode],
            matches: |intent| matches!(intent, QueryIntent::SimilaritySearch),
            build: |resolved, params| {
                let start = resolved.first()?.node_id;
                let k = params.limit.unwrap_or(10);
                Some(GraphQuery::NearestByCost { start, k })
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlq::entity::{EntityHint, EntityMention, ResolvedEntity};

    fn make_entity(name: &str, node_id: u64) -> ResolvedEntity {
        ResolvedEntity {
            mention: EntityMention {
                text: name.to_string(),
                span: (0, name.len()),
                hint: EntityHint::Unknown,
                confidence: 1.0,
            },
            node_id,
            node_type: "Concept".to_string(),
            confidence: 1.0,
        }
    }

    #[test]
    fn test_template_match_shortest_path() {
        let registry = TemplateRegistry::new();
        let entities = vec![make_entity("Alice", 1), make_entity("Bob", 2)];
        let intent = QueryIntent::FindPath {
            algorithm: PathAlgorithm::Shortest,
        };
        let result = registry.match_template(&intent, &entities, "shortest path from Alice to Bob");
        assert!(result.is_some());
        let (template, _params) = result.unwrap();
        assert_eq!(template.name, "shortest_path");
    }

    #[test]
    fn test_template_match_neighbors() {
        let registry = TemplateRegistry::new();
        let entities = vec![make_entity("Alice", 1)];
        let intent = QueryIntent::FindNeighbors {
            direction: Direction::Both,
            edge_hint: None,
        };
        let result = registry.match_template(&intent, &entities, "Who does Alice connect to?");
        assert!(result.is_some());
        let (template, _params) = result.unwrap();
        assert_eq!(template.name, "neighbors");
    }

    #[test]
    fn test_template_match_pagerank() {
        let registry = TemplateRegistry::new();
        let intent = QueryIntent::Ranking {
            metric: RankingMetric::PageRank,
        };
        let result = registry.match_template(&intent, &[], "Most important nodes");
        assert!(result.is_some());
        let (template, _params) = result.unwrap();
        assert_eq!(template.name, "pagerank");
    }

    #[test]
    fn test_extract_params_depth() {
        let params = extract_params("Neighbors within 3 hops", &QueryIntent::Unknown);
        assert_eq!(params.depth, Some(3));
    }

    #[test]
    fn test_extract_params_limit() {
        let params = extract_params("Top 5 important nodes", &QueryIntent::Unknown);
        assert_eq!(params.limit, Some(5));
    }

    // Enhancement 2: Multi-hop detection via template params
    #[test]
    fn test_extract_params_multi_hop() {
        let params = extract_params("2-hop neighbors of Alice", &QueryIntent::Unknown);
        assert_eq!(params.depth, Some(2));
    }

    #[test]
    fn test_extract_params_friends_of_friends() {
        let params = extract_params("friends of friends of Alice", &QueryIntent::Unknown);
        assert_eq!(params.depth, Some(2));
    }

    // Enhancement 9: NL filter extraction
    #[test]
    fn test_extract_weight_threshold() {
        assert_eq!(
            extract_weight_threshold("edges with weight > 0.8"),
            Some(0.8)
        );
    }

    #[test]
    fn test_extract_edge_type_nl() {
        assert_eq!(
            extract_edge_type_from_nl("show causal edges"),
            Some("Causality".to_string())
        );
    }

    #[test]
    fn test_extract_min_degree() {
        assert_eq!(extract_min_degree("nodes with degree > 5"), Some(5));
    }

    #[test]
    fn test_extract_params_with_nl_filters() {
        let params = extract_params("neighbors with weight > 0.5", &QueryIntent::Unknown);
        assert!(!params.edge_filters.is_empty());
    }
}
