use serde::{Deserialize, Serialize};

pub use agent_db_core::types::{GroupId, RowId, RowVersionId, TableId, Timestamp};

/// 8KB page size.
pub const PAGE_SIZE: usize = 8192;

/// Maximum usable payload per page after header (32B) and one slot entry (4B).
pub const MAX_ROW_PAYLOAD: usize = PAGE_SIZE - PAGE_HEADER_SIZE - SLOT_SIZE;

/// Page header size in bytes.
pub const PAGE_HEADER_SIZE: usize = 32;

/// Slot directory entry size in bytes.
pub const SLOT_SIZE: usize = 4;

/// Row header size in bytes.
/// version_id(8) + row_id(8) + group_id(8) + valid_from(8) + valid_until(8) + created_at(8) + flags(2) + _reserved(2)
pub const ROW_HEADER_SIZE: usize = 52;

/// Row flags: COMMITTED bit.
pub const ROW_FLAG_COMMITTED: u16 = 0x0001;

/// Physical pointer to a row version within a table's page store.
/// 6 bytes total. This is a **physical address**, not a stable identity.
/// Stable across in-page compaction (slot indices preserved), but not
/// across full page rewrites. Use (RowId, RowVersionId) for logical identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowPointer {
    pub page_id: u32,
    pub slot_idx: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    String,
    Int64,
    Float64,
    Bool,
    Timestamp,
    Json,
    NodeRef,
}

impl ColumnType {
    /// Returns the fixed-width byte size if this is a fixed-width type, else None.
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            ColumnType::Int64 => Some(8),
            ColumnType::Float64 => Some(8),
            ColumnType::Bool => Some(1),
            ColumnType::Timestamp => Some(8),
            ColumnType::NodeRef => Some(8),
            ColumnType::String | ColumnType::Json => None,
        }
    }

    /// True if this is a variable-length type.
    pub fn is_variable_length(&self) -> bool {
        self.fixed_size().is_none()
    }
}

/// Cell value at the API boundary. Used for insert/update requests
/// and query results. NOT stored directly -- rows are encoded into pages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CellValue {
    String(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Timestamp(u64),
    Json(serde_json::Value),
    NodeRef(u64),
    Null,
}

impl CellValue {
    pub fn matches_type(&self, col_type: &ColumnType) -> bool {
        match (self, col_type) {
            (CellValue::Null, _) => true,
            (CellValue::String(_), ColumnType::String) => true,
            (CellValue::Int64(_), ColumnType::Int64) => true,
            (CellValue::Float64(_), ColumnType::Float64) => true,
            (CellValue::Bool(_), ColumnType::Bool) => true,
            (CellValue::Timestamp(_), ColumnType::Timestamp) => true,
            (CellValue::Json(_), ColumnType::Json) => true,
            (CellValue::NodeRef(_), ColumnType::NodeRef) => true,
            // Allow Int64 where Timestamp expected and vice versa (both u64/i64)
            (CellValue::Int64(_), ColumnType::Timestamp) => true,
            (CellValue::Timestamp(_), ColumnType::Int64) => true,
            _ => false,
        }
    }

    pub fn as_node_ref(&self) -> Option<u64> {
        match self {
            CellValue::NodeRef(id) => Some(*id),
            _ => None,
        }
    }

    pub fn parse_from_str(s: &str, col_type: &ColumnType) -> Result<Self, std::string::String> {
        if s.is_empty() || s.eq_ignore_ascii_case("null") {
            return Ok(CellValue::Null);
        }
        match col_type {
            ColumnType::String => Ok(CellValue::String(s.to_string())),
            ColumnType::Int64 => s
                .parse::<i64>()
                .map(CellValue::Int64)
                .map_err(|e| e.to_string()),
            ColumnType::Float64 => s
                .parse::<f64>()
                .map(CellValue::Float64)
                .map_err(|e| e.to_string()),
            ColumnType::Bool => s
                .parse::<bool>()
                .map(CellValue::Bool)
                .map_err(|e| e.to_string()),
            ColumnType::Timestamp => s
                .parse::<u64>()
                .map(CellValue::Timestamp)
                .map_err(|e| e.to_string()),
            ColumnType::Json => serde_json::from_str(s)
                .map(CellValue::Json)
                .map_err(|e| e.to_string()),
            ColumnType::NodeRef => s
                .parse::<u64>()
                .map(CellValue::NodeRef)
                .map_err(|e| e.to_string()),
        }
    }

    /// Type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            CellValue::String(_) => "String",
            CellValue::Int64(_) => "Int64",
            CellValue::Float64(_) => "Float64",
            CellValue::Bool(_) => "Bool",
            CellValue::Timestamp(_) => "Timestamp",
            CellValue::Json(_) => "Json",
            CellValue::NodeRef(_) => "NodeRef",
            CellValue::Null => "Null",
        }
    }
}

/// Sortable key for BTree indexes and constraint enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexKey {
    Null,
    Bool(bool),
    Int64(i64),
    /// f64 stored as bits for Eq/Hash/Ord (total ordering via to_bits).
    Float64(u64),
    Timestamp(u64),
    String(String),
    NodeRef(u64),
    Composite(Vec<IndexKey>),
}

impl PartialOrd for IndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use IndexKey::*;
        // Define a discriminant for ordering across types
        fn disc(k: &IndexKey) -> u8 {
            match k {
                Null => 0,
                Bool(_) => 1,
                Int64(_) => 2,
                Float64(_) => 3,
                Timestamp(_) => 4,
                String(_) => 5,
                NodeRef(_) => 6,
                Composite(_) => 7,
            }
        }
        let d = disc(self).cmp(&disc(other));
        if d != Ordering::Equal {
            return d;
        }
        match (self, other) {
            (Null, Null) => Ordering::Equal,
            (Bool(a), Bool(b)) => a.cmp(b),
            (Int64(a), Int64(b)) => a.cmp(b),
            (Float64(a), Float64(b)) => a.cmp(b),
            (Timestamp(a), Timestamp(b)) => a.cmp(b),
            (String(a), String(b)) => a.cmp(b),
            (NodeRef(a), NodeRef(b)) => a.cmp(b),
            (Composite(a), Composite(b)) => a.cmp(b),
            _ => unreachable!(),
        }
    }
}

impl IndexKey {
    /// Build an IndexKey from a CellValue.
    pub fn from_cell(value: &CellValue) -> Self {
        match value {
            CellValue::Null => IndexKey::Null,
            CellValue::Bool(b) => IndexKey::Bool(*b),
            CellValue::Int64(i) => IndexKey::Int64(*i),
            CellValue::Float64(f) => IndexKey::Float64(f.to_bits()),
            CellValue::Timestamp(t) => IndexKey::Timestamp(*t),
            CellValue::String(s) => IndexKey::String(s.clone()),
            CellValue::NodeRef(n) => IndexKey::NodeRef(*n),
            CellValue::Json(_) => {
                // JSON is not indexable; use Null as fallback
                IndexKey::Null
            },
        }
    }
}
