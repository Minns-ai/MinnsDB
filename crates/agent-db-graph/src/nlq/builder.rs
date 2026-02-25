//! Query builder: constructs a GraphQuery from template + entities + params.

use super::entity::ResolvedEntity;
use super::intent::{PathAlgorithm, QueryIntent};
use super::template::{QueryTemplate, TemplateParams};
use crate::structures::Depth;
use crate::traversal::{GraphQuery, Instruction, TraversalRequest};
use crate::{GraphError, GraphResult};

/// Build a GraphQuery from a matched template, resolved entities, and params.
pub fn build_query(
    template: &QueryTemplate,
    resolved: &[ResolvedEntity],
    params: &TemplateParams,
    intent: &QueryIntent,
) -> GraphResult<GraphQuery> {
    // Apply intent-specific K value for KShortestPaths
    let mut query = (template.build)(resolved, params).ok_or_else(|| {
        GraphError::InvalidQuery("Template build failed: insufficient entities".to_string())
    })?;

    // Patch K value from intent if this is a K-shortest query
    if let (
        QueryIntent::FindPath {
            algorithm: PathAlgorithm::KShortest(k),
        },
        GraphQuery::KShortestPaths { k: ref mut qk, .. },
    ) = (intent, &mut query)
    {
        *qk = *k;
    }

    // Enhancement 9: If NL filters are present and query is a simple variant,
    // upgrade to RecursiveTraversal with filters attached.
    if (!params.node_filters.is_empty() || !params.edge_filters.is_empty())
        && !matches!(query, GraphQuery::RecursiveTraversal(_))
    {
        if let GraphQuery::NeighborsWithinDistance {
            start,
            max_distance,
        } = query
        {
            query = GraphQuery::RecursiveTraversal(TraversalRequest {
                start,
                direction: params.direction,
                depth: Depth::Range(1, max_distance),
                instruction: Instruction::Collect,
                node_filters: params.node_filters.clone(),
                edge_filters: params.edge_filters.clone(),
                max_nodes_visited: Some(1000),
                max_edges_traversed: Some(5000),
                time_window: None,
            });
        }
    }

    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlq::entity::{EntityHint, EntityMention, ResolvedEntity};
    use crate::nlq::template::TemplateRegistry;
    use crate::structures::Direction;

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
    fn test_build_shortest_path() {
        let registry = TemplateRegistry::new();
        let entities = vec![make_entity("A", 1), make_entity("B", 2)];
        let intent = QueryIntent::FindPath {
            algorithm: PathAlgorithm::Shortest,
        };
        let (template, params) = registry
            .match_template(&intent, &entities, "shortest path from A to B")
            .unwrap();
        let query = build_query(template, &entities, &params, &intent).unwrap();
        assert!(matches!(
            query,
            GraphQuery::ShortestPath { start: 1, end: 2 }
        ));
    }

    #[test]
    fn test_build_neighbors() {
        let registry = TemplateRegistry::new();
        let entities = vec![make_entity("Alice", 5)];
        let intent = QueryIntent::FindNeighbors {
            direction: Direction::Both,
            edge_hint: None,
        };
        let (template, params) = registry
            .match_template(&intent, &entities, "Who does Alice know?")
            .unwrap();
        let query = build_query(template, &entities, &params, &intent).unwrap();
        assert!(matches!(
            query,
            GraphQuery::NeighborsWithinDistance {
                start: 5,
                max_distance: 1
            }
        ));
    }
}
