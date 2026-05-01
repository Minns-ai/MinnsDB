use agent_db_core::types::RowId;

use crate::types::ColumnType;

#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("table not found: {0}")]
    TableNotFound(String),
    #[error("table already exists: {0}")]
    TableAlreadyExists(String),
    #[error("invalid schema: {0}")]
    SchemaInvalid(String),
    #[error("row not found: {0}")]
    RowNotFound(RowId),
    #[error("row already deleted: {0}")]
    RowAlreadyDeleted(RowId),
    #[error("column count mismatch: expected {expected}, got {got}")]
    ColumnCountMismatch { expected: usize, got: usize },
    #[error("type mismatch for column '{column}': expected {expected:?}, got {got}")]
    TypeMismatch {
        column: String,
        expected: ColumnType,
        got: String,
    },
    #[error("null constraint violation: column '{0}' cannot be null")]
    NullConstraintViolation(String),
    #[error("unique constraint violation on columns: {0:?}")]
    UniqueConstraintViolation(Vec<String>),
    #[error("primary key violation")]
    PrimaryKeyViolation,
    #[error("import error: {0}")]
    ImportError(String),
    #[error("persistence error: {0}")]
    PersistenceError(String),
}
