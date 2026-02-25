//! Query executor — thin wrapper around GraphTraversal::execute_query.

use crate::structures::Graph;
use crate::traversal::{GraphQuery, GraphTraversal, QueryResult};
use crate::GraphResult;

/// Execute a validated query against the graph.
pub fn execute(
    query: &GraphQuery,
    graph: &Graph,
    traversal: &mut GraphTraversal,
) -> GraphResult<QueryResult> {
    traversal.execute_query(graph, query.clone())
}
