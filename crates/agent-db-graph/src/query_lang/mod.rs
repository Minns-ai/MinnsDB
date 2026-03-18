//! MinnsQL query language: parsing, planning, and execution.

pub mod ast;
pub mod executor;
pub mod lexer;
pub mod parser;
pub mod planner;
pub mod table_executor;
pub mod token;
pub mod types;

pub use ast::Statement;
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

/// Parse and execute a MinnsQL table query against the TableCatalog.
pub fn execute_table_query(
    input: &str,
    catalog: &agent_db_tables::catalog::TableCatalog,
    group_id: u64,
) -> Result<QueryOutput, QueryError> {
    let ast = parser::Parser::parse(input)?;
    let plan = planner::plan(ast)?;
    table_executor::execute_table_query(catalog, &plan, group_id)
}
