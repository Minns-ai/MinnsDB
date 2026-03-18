//! Multi-table manager.

use rustc_hash::FxHashMap;

use crate::error::TableError;
use crate::schema::{ColumnDef, Constraint, TableSchema};
use crate::table::Table;
use crate::types::{TableId, Timestamp};

use agent_db_core::types::current_timestamp;

pub struct TableCatalog {
    tables: FxHashMap<TableId, Table>,
    name_index: FxHashMap<String, TableId>,
    next_table_id: TableId,
}

impl TableCatalog {
    pub fn new() -> Self {
        TableCatalog {
            tables: FxHashMap::default(),
            name_index: FxHashMap::default(),
            next_table_id: 1,
        }
    }

    pub fn create_table(
        &mut self,
        name: String,
        columns: Vec<ColumnDef>,
        constraints: Vec<Constraint>,
    ) -> Result<TableId, TableError> {
        if self.name_index.contains_key(&name) {
            return Err(TableError::TableAlreadyExists(name));
        }

        let table_id = self.next_table_id;
        let schema = TableSchema {
            table_id,
            name: name.clone(),
            columns,
            constraints,
            created_at: current_timestamp(),
            schema_version: 1,
        };

        let table = Table::new(schema)?;
        self.tables.insert(table_id, table);
        self.name_index.insert(name, table_id);
        self.next_table_id += 1;

        Ok(table_id)
    }

    pub fn drop_table(&mut self, name: &str) -> Result<TableId, TableError> {
        let table_id = self
            .name_index
            .remove(name)
            .ok_or_else(|| TableError::TableNotFound(name.into()))?;
        self.tables.remove(&table_id);
        Ok(table_id)
    }

    pub fn get_table(&self, name: &str) -> Option<&Table> {
        let id = self.name_index.get(name)?;
        self.tables.get(id)
    }

    pub fn get_table_mut(&mut self, name: &str) -> Option<&mut Table> {
        let id = self.name_index.get(name)?;
        self.tables.get_mut(id)
    }

    pub fn get_table_by_id(&self, id: TableId) -> Option<&Table> {
        self.tables.get(&id)
    }

    pub fn get_table_by_id_mut(&mut self, id: TableId) -> Option<&mut Table> {
        self.tables.get_mut(&id)
    }

    pub fn list_tables(&self) -> Vec<&TableSchema> {
        self.tables.values().map(|t| &t.schema).collect()
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub fn next_table_id(&self) -> TableId {
        self.next_table_id
    }

    pub fn set_next_table_id(&mut self, id: TableId) {
        self.next_table_id = id;
    }

    /// Handle graph node merge across all tables.
    pub fn on_node_merged(&mut self, survivor: u64, absorbed: u64) {
        for table in self.tables.values_mut() {
            table.on_node_merged(survivor, absorbed);
        }
    }

    /// Compact all tables.
    pub fn compact_all(&mut self, now: Timestamp) -> usize {
        let mut total = 0;
        for table in self.tables.values_mut() {
            total += crate::compaction::compact_table(table, now).versions_removed;
        }
        total
    }

    /// Insert a pre-built table (used during persistence load).
    pub fn insert_table(&mut self, table: Table) {
        let name = table.schema.name.clone();
        let id = table.schema.table_id;
        self.tables.insert(id, table);
        self.name_index.insert(name, id);
    }

    /// Iterate all tables.
    pub fn tables(&self) -> impl Iterator<Item = &Table> {
        self.tables.values()
    }

    pub fn tables_mut(&mut self) -> impl Iterator<Item = &mut Table> {
        self.tables.values_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ColumnDef;
    use crate::types::ColumnType;

    fn col(name: &str, ct: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type: ct,
            nullable: true,
            default_value: None,
        }
    }

    #[test]
    fn test_create_and_list() {
        let mut cat = TableCatalog::new();
        let id = cat
            .create_table(
                "users".into(),
                vec![
                    col("id", ColumnType::Int64),
                    col("name", ColumnType::String),
                ],
                vec![],
            )
            .unwrap();
        assert_eq!(id, 1);
        assert_eq!(cat.table_count(), 1);
        assert!(cat.get_table("users").is_some());
    }

    #[test]
    fn test_duplicate_name() {
        let mut cat = TableCatalog::new();
        cat.create_table("t".into(), vec![col("a", ColumnType::Int64)], vec![])
            .unwrap();
        let r = cat.create_table("t".into(), vec![col("b", ColumnType::Int64)], vec![]);
        assert!(matches!(r, Err(TableError::TableAlreadyExists(_))));
    }

    #[test]
    fn test_drop() {
        let mut cat = TableCatalog::new();
        cat.create_table("t".into(), vec![col("a", ColumnType::Int64)], vec![])
            .unwrap();
        cat.drop_table("t").unwrap();
        assert_eq!(cat.table_count(), 0);
        assert!(cat.get_table("t").is_none());
    }

    #[test]
    fn test_drop_not_found() {
        let mut cat = TableCatalog::new();
        assert!(matches!(
            cat.drop_table("nope"),
            Err(TableError::TableNotFound(_))
        ));
    }
}
