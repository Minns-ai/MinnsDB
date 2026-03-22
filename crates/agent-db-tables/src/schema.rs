use serde::{Deserialize, Serialize};

use crate::error::TableError;
use crate::types::{CellValue, ColumnType, TableId, Timestamp, ROW_HEADER_SIZE};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_id: TableId,
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<Constraint>,
    pub created_at: Timestamp,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: ColumnType,
    pub nullable: bool,
    pub default_value: Option<CellValue>,
    /// Column auto-increments on insert when the value is NULL or omitted.
    #[serde(default)]
    pub autoincrement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    PrimaryKey(Vec<String>),
    Unique(Vec<String>),
    NotNull(String),
    /// Column references a graph node.
    ForeignKeyGraph(String),
    /// Non-unique secondary index (persisted for rebuild on restart).
    Index(Vec<String>),
}

impl TableSchema {
    /// Validate the schema is well-formed.
    pub fn validate(&self) -> Result<(), TableError> {
        if self.name.is_empty() {
            return Err(TableError::SchemaInvalid("table name is empty".into()));
        }
        if self.columns.is_empty() {
            return Err(TableError::SchemaInvalid(
                "table must have at least one column".into(),
            ));
        }

        // Check for duplicate column names
        let mut seen = std::collections::HashSet::new();
        for col in &self.columns {
            if col.name.is_empty() {
                return Err(TableError::SchemaInvalid("column name is empty".into()));
            }
            if !seen.insert(&col.name) {
                return Err(TableError::SchemaInvalid(format!(
                    "duplicate column name: '{}'",
                    col.name
                )));
            }
        }

        // Validate constraints reference real columns
        for constraint in &self.constraints {
            let cols = match constraint {
                Constraint::PrimaryKey(cols)
                | Constraint::Unique(cols)
                | Constraint::Index(cols) => cols,
                Constraint::NotNull(col) | Constraint::ForeignKeyGraph(col) => {
                    // Single column check
                    if !self.columns.iter().any(|c| &c.name == col) {
                        return Err(TableError::SchemaInvalid(format!(
                            "constraint references unknown column: '{col}'"
                        )));
                    }
                    continue;
                },
            };
            for col_name in cols {
                if !self.columns.iter().any(|c| &c.name == col_name) {
                    return Err(TableError::SchemaInvalid(format!(
                        "constraint references unknown column: '{col_name}'"
                    )));
                }
            }
        }

        // ForeignKeyGraph must be NodeRef type
        for constraint in &self.constraints {
            if let Constraint::ForeignKeyGraph(col_name) = constraint {
                let col = self.columns.iter().find(|c| &c.name == col_name).unwrap();
                if col.col_type != ColumnType::NodeRef {
                    return Err(TableError::SchemaInvalid(format!(
                        "ForeignKeyGraph constraint on '{}' requires NodeRef type, got {:?}",
                        col_name, col.col_type
                    )));
                }
            }
        }

        Ok(())
    }

    /// Find a column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Get primary key column indices, if a PK constraint exists.
    pub fn pk_columns(&self) -> Option<Vec<usize>> {
        for c in &self.constraints {
            if let Constraint::PrimaryKey(cols) = c {
                let indices: Vec<usize> = cols
                    .iter()
                    .filter_map(|name| self.column_index(name))
                    .collect();
                if indices.len() == cols.len() {
                    return Some(indices);
                }
            }
        }
        None
    }

    /// Get NodeRef column indices (for building node_ref_index).
    pub fn node_ref_columns(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.col_type == ColumnType::NodeRef)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Precomputed layout for fast row encoding/decoding.
/// Calculated once from schema, reused for every row operation.
pub struct RowLayout {
    /// For each user column: offset or var-len index.
    pub column_offsets: Vec<ColumnOffset>,
    /// Total size of the fixed section (after header): null bitmap + fixed-width columns.
    pub fixed_section_size: usize,
    /// Number of variable-length columns.
    pub var_len_count: usize,
    /// Size of null bitmap in bytes.
    pub null_bitmap_bytes: usize,
    /// True if all columns are fixed-width (enables memcpy fast path).
    pub all_fixed: bool,
    /// Total minimum row size (header + fixed section + var-len offset table).
    pub min_row_size: usize,
}

#[derive(Debug, Clone)]
pub enum ColumnOffset {
    /// Fixed-width: byte offset from start of fixed-values area (after null bitmap).
    Fixed { offset: usize, size: usize },
    /// Variable-length: index into the var-len offset table.
    VarLen { index: usize },
}

impl RowLayout {
    /// Compute layout from schema. Called once at table creation.
    pub fn from_schema(schema: &TableSchema) -> Self {
        let null_bitmap_bytes = schema.columns.len().div_ceil(8);
        let mut fixed_offset = 0usize;
        let mut var_idx = 0usize;
        let mut column_offsets = Vec::with_capacity(schema.columns.len());

        for col in &schema.columns {
            if let Some(size) = col.col_type.fixed_size() {
                column_offsets.push(ColumnOffset::Fixed {
                    offset: fixed_offset,
                    size,
                });
                fixed_offset += size;
            } else {
                column_offsets.push(ColumnOffset::VarLen { index: var_idx });
                var_idx += 1;
            }
        }

        let fixed_section_size = null_bitmap_bytes + fixed_offset;
        let var_len_count = var_idx;
        let all_fixed = var_len_count == 0;
        // var-len offset table: 4 bytes (u16 offset + u16 len) per var-len column
        let var_len_table_size = var_len_count * 4;
        let min_row_size = ROW_HEADER_SIZE + fixed_section_size + var_len_table_size;

        RowLayout {
            column_offsets,
            fixed_section_size,
            var_len_count,
            null_bitmap_bytes,
            all_fixed,
            min_row_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema(columns: Vec<ColumnDef>, constraints: Vec<Constraint>) -> TableSchema {
        TableSchema {
            table_id: 1,
            name: "test".into(),
            columns,
            constraints,
            created_at: 0,
            schema_version: 1,
        }
    }

    fn col(name: &str, col_type: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type,
            nullable: true,
            default_value: None,
            autoincrement: false,
        }
    }

    #[test]
    fn test_schema_validation_empty_name() {
        let mut s = make_schema(vec![col("a", ColumnType::Int64)], vec![]);
        s.name = String::new();
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_schema_validation_no_columns() {
        let s = make_schema(vec![], vec![]);
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_schema_validation_duplicate_columns() {
        let s = make_schema(
            vec![col("a", ColumnType::Int64), col("a", ColumnType::String)],
            vec![],
        );
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_schema_validation_ok() {
        let s = make_schema(
            vec![
                col("id", ColumnType::Int64),
                col("name", ColumnType::String),
            ],
            vec![Constraint::PrimaryKey(vec!["id".into()])],
        );
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_schema_constraint_unknown_column() {
        let s = make_schema(
            vec![col("id", ColumnType::Int64)],
            vec![Constraint::PrimaryKey(vec!["missing".into()])],
        );
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_foreign_key_graph_wrong_type() {
        let s = make_schema(
            vec![col("ref", ColumnType::String)],
            vec![Constraint::ForeignKeyGraph("ref".into())],
        );
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_layout_all_fixed() {
        let s = make_schema(
            vec![
                col("a", ColumnType::Int64),
                col("b", ColumnType::Bool),
                col("c", ColumnType::Float64),
            ],
            vec![],
        );
        let layout = RowLayout::from_schema(&s);
        assert!(layout.all_fixed);
        assert_eq!(layout.var_len_count, 0);
        // null bitmap: ceil(3/8) = 1 byte
        assert_eq!(layout.null_bitmap_bytes, 1);
        // fixed values: 8 + 1 + 8 = 17
        // fixed_section_size = 1 (bitmap) + 17 = 18
        assert_eq!(layout.fixed_section_size, 18);
        // min_row_size = 52 (header) + 18 = 70
        assert_eq!(layout.min_row_size, ROW_HEADER_SIZE + 18);
    }

    #[test]
    fn test_layout_mixed() {
        let s = make_schema(
            vec![
                col("id", ColumnType::Int64),
                col("name", ColumnType::String),
                col("data", ColumnType::Json),
                col("active", ColumnType::Bool),
            ],
            vec![],
        );
        let layout = RowLayout::from_schema(&s);
        assert!(!layout.all_fixed);
        assert_eq!(layout.var_len_count, 2);
        // null bitmap: ceil(4/8) = 1
        assert_eq!(layout.null_bitmap_bytes, 1);
        // fixed values: 8 (Int64) + 1 (Bool) = 9
        // fixed_section_size = 1 + 9 = 10
        assert_eq!(layout.fixed_section_size, 10);
        // var-len offset table: 2 * 4 = 8
        // min_row_size = 52 + 10 + 8 = 70
        assert_eq!(layout.min_row_size, ROW_HEADER_SIZE + 10 + 8);
    }

    #[test]
    fn test_pk_columns() {
        let s = make_schema(
            vec![col("a", ColumnType::Int64), col("b", ColumnType::String)],
            vec![Constraint::PrimaryKey(vec!["a".into()])],
        );
        assert_eq!(s.pk_columns(), Some(vec![0]));
    }

    #[test]
    fn test_node_ref_columns() {
        let s = make_schema(
            vec![
                col("id", ColumnType::Int64),
                col("ref1", ColumnType::NodeRef),
                col("name", ColumnType::String),
                col("ref2", ColumnType::NodeRef),
            ],
            vec![],
        );
        assert_eq!(s.node_ref_columns(), vec![1, 3]);
    }
}
