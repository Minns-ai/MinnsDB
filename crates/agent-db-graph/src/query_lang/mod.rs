//! MinnsQL query language: parsing, planning, and execution.

pub mod ast;
pub mod executor;
pub mod lexer;
pub mod optimizer;
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

/// Parse and execute a MinnsQL table query with optimizer.
pub fn execute_table_query_optimized(
    input: &str,
    catalog: &agent_db_tables::catalog::TableCatalog,
    group_id: u64,
) -> Result<QueryOutput, QueryError> {
    let ast = parser::Parser::parse(input)?;
    let mut plan = planner::plan(ast)?;

    // Collect catalog stats and run optimizer
    let catalog_stats = catalog.collect_stats();
    let table_stats = CatalogStatsAdapter {
        stats: &catalog_stats,
        catalog,
        group_id,
    };
    let _trace = optimizer::optimize(&mut plan, &table_stats);

    table_executor::execute_table_query(catalog, &plan, group_id)
}

/// Adapter to bridge agent_db_tables::stats::CatalogStats to the optimizer's OptimizerStats trait.
pub struct CatalogStatsAdapter<'a> {
    pub stats: &'a agent_db_tables::stats::CatalogStats,
    pub catalog: &'a agent_db_tables::catalog::TableCatalog,
    pub group_id: u64,
}

impl<'a> optimizer::OptimizerStats for CatalogStatsAdapter<'a> {
    fn estimated_row_count(&self, table_name: &str) -> usize {
        self.stats.estimated_row_count(table_name, self.group_id)
    }

    fn has_index(&self, table_name: &str, columns: &[String]) -> bool {
        if let Some(table) = self.catalog.get_table(table_name) {
            table
                .index_columns()
                .iter()
                .any(|idx_cols| idx_cols == columns)
        } else {
            false
        }
    }

    fn index_cardinality(&self, table_name: &str, columns: &[String]) -> usize {
        if let Some(ts) = self.stats.get(table_name) {
            if let Some(table) = self.catalog.get_table(table_name) {
                let all_indexes = table.index_columns();
                for (i, idx_cols) in all_indexes.iter().enumerate() {
                    if idx_cols == columns {
                        return ts.index_cardinalities.get(i).copied().unwrap_or(0);
                    }
                }
            }
        }
        0
    }
}
