//! MinnsQL query language: parsing, planning, and execution.

pub mod ast;
pub mod executor;
pub mod lexer;
pub mod parser;
pub mod planner;
pub mod token;
pub mod types;

pub use types::{QueryError, QueryOutput, QueryStats, Value};

/// Parse and execute a MinnsQL query against a Graph.
pub fn execute_query(
    input: &str,
    graph: &crate::Graph,
    ontology: &crate::ontology::OntologyRegistry,
) -> Result<QueryOutput, QueryError> {
    let ast = parser::Parser::parse(input)?;
    let plan = planner::plan(ast)?;
    executor::Executor::execute(graph, ontology, plan)
}
