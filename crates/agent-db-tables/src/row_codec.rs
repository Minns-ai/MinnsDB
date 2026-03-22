//! Row encoding/decoding: CellValue <-> compact binary row bytes within pages.
//! O(1) column access — read any column without parsing the whole row.
//!
//! Row layout (within page):
//!   [RowHeader 52B] [NullBitmap] [FixedCols] [VarLenOffsets] [VarLenData]
//!
//! RowHeader: version_id(8) + row_id(8) + group_id(8) + valid_from(8) +
//!            valid_until(8) + created_at(8) + flags(2) + _reserved(2)

use crate::schema::{ColumnOffset, RowLayout};
use crate::types::*;

/// Decoded row returned by queries.
#[derive(Debug, Clone)]
pub struct DecodedRow {
    pub version_id: RowVersionId,
    pub row_id: RowId,
    pub group_id: GroupId,
    pub valid_from: Timestamp,
    pub valid_until: Option<Timestamp>,
    pub created_at: Timestamp,
    pub flags: u16,
    pub values: Vec<CellValue>,
}

/// Row header fields at known offsets.
#[derive(Debug, Clone, Copy)]
pub struct RowHeader {
    pub version_id: RowVersionId,
    pub row_id: RowId,
    pub group_id: GroupId,
    pub valid_from: Timestamp,
    pub valid_until: Option<Timestamp>,
    pub created_at: Timestamp,
    pub flags: u16,
}

// Header field byte offsets
const OFF_VERSION_ID: usize = 0;
const OFF_ROW_ID: usize = 8;
const OFF_GROUP_ID: usize = 16;
const OFF_VALID_FROM: usize = 24;
const OFF_VALID_UNTIL: usize = 32;
const OFF_CREATED_AT: usize = 40;
const OFF_FLAGS: usize = 48;
// OFF_RESERVED = 50, 2 bytes, always 0

/// Encode a row into bytes for page insertion.
#[allow(clippy::too_many_arguments)]
pub fn encode_row(
    layout: &RowLayout,
    version_id: RowVersionId,
    row_id: RowId,
    group_id: GroupId,
    valid_from: Timestamp,
    valid_until: Option<Timestamp>,
    created_at: Timestamp,
    flags: u16,
    values: &[CellValue],
    schema_columns: &[crate::schema::ColumnDef],
) -> Vec<u8> {
    // Pre-calculate var-len data to know total size
    let mut var_data_parts: Vec<Vec<u8>> = Vec::with_capacity(layout.var_len_count);
    for (i, col) in schema_columns.iter().enumerate() {
        if col.col_type.is_variable_length() {
            let data = match &values[i] {
                CellValue::String(s) => s.as_bytes().to_vec(),
                CellValue::Json(v) => rmp_serde::to_vec(v).unwrap_or_default(),
                CellValue::Null => vec![],
                _ => vec![],
            };
            var_data_parts.push(data);
        }
    }

    let var_data_total: usize = var_data_parts.iter().map(|d| d.len()).sum();
    let total_size = layout.min_row_size + var_data_total;
    let mut buf = vec![0u8; total_size];

    // Write header
    buf[OFF_VERSION_ID..OFF_VERSION_ID + 8].copy_from_slice(&version_id.to_le_bytes());
    buf[OFF_ROW_ID..OFF_ROW_ID + 8].copy_from_slice(&row_id.to_le_bytes());
    buf[OFF_GROUP_ID..OFF_GROUP_ID + 8].copy_from_slice(&group_id.to_le_bytes());
    buf[OFF_VALID_FROM..OFF_VALID_FROM + 8].copy_from_slice(&valid_from.to_le_bytes());
    let vu = valid_until.unwrap_or(0);
    buf[OFF_VALID_UNTIL..OFF_VALID_UNTIL + 8].copy_from_slice(&vu.to_le_bytes());
    buf[OFF_CREATED_AT..OFF_CREATED_AT + 8].copy_from_slice(&created_at.to_le_bytes());
    buf[OFF_FLAGS..OFF_FLAGS + 2].copy_from_slice(&flags.to_le_bytes());
    // reserved bytes stay 0

    // Null bitmap starts after header
    let bitmap_start = ROW_HEADER_SIZE;
    for (i, val) in values.iter().enumerate() {
        if matches!(val, CellValue::Null) {
            let byte_idx = bitmap_start + i / 8;
            buf[byte_idx] |= 1 << (i % 8);
        }
    }

    // Fixed-width values start after null bitmap
    let fixed_start = ROW_HEADER_SIZE + layout.null_bitmap_bytes;
    for (i, col) in schema_columns.iter().enumerate() {
        if let ColumnOffset::Fixed { offset, size } = &layout.column_offsets[i] {
            let dst = fixed_start + offset;
            match (&values[i], &col.col_type) {
                (CellValue::Int64(v), _) => {
                    buf[dst..dst + 8].copy_from_slice(&v.to_le_bytes());
                },
                (CellValue::Float64(v), _) => {
                    buf[dst..dst + 8].copy_from_slice(&v.to_le_bytes());
                },
                (CellValue::Bool(v), _) => {
                    buf[dst] = if *v { 1 } else { 0 };
                },
                (CellValue::Timestamp(v), _) | (CellValue::NodeRef(v), _) => {
                    buf[dst..dst + 8].copy_from_slice(&v.to_le_bytes());
                },
                (CellValue::Null, _) => {
                    // Zero-fill (already zero from vec init)
                    for b in &mut buf[dst..dst + size] {
                        *b = 0;
                    }
                },
                _ => {}, // type mismatch caught at validation layer
            }
        }
    }

    // Variable-length offset table + data
    let var_table_start = ROW_HEADER_SIZE + layout.fixed_section_size;
    let var_data_start = var_table_start + layout.var_len_count * 4;
    let mut var_data_offset: u16 = 0;
    let mut var_idx = 0;

    for (i, _col) in schema_columns.iter().enumerate() {
        if let ColumnOffset::VarLen { index } = &layout.column_offsets[i] {
            let data = &var_data_parts[*index];
            let len = data.len() as u16;
            let table_pos = var_table_start + var_idx * 4;
            buf[table_pos..table_pos + 2].copy_from_slice(&var_data_offset.to_le_bytes());
            buf[table_pos + 2..table_pos + 4].copy_from_slice(&len.to_le_bytes());

            if !data.is_empty() {
                let dst = var_data_start + var_data_offset as usize;
                buf[dst..dst + data.len()].copy_from_slice(data);
            }

            var_data_offset += len;
            var_idx += 1;
        }
    }

    buf
}

/// Read just the row header. First 52 bytes.
///
/// # Panics
/// Panics if `row_bytes.len() < ROW_HEADER_SIZE`. Callers must validate row
/// bytes before calling (page.read_row always returns correctly-sized data
/// for non-corrupted pages).
pub fn read_header(row_bytes: &[u8]) -> RowHeader {
    assert!(
        row_bytes.len() >= ROW_HEADER_SIZE,
        "row_bytes too short for header: {} < {}",
        row_bytes.len(),
        ROW_HEADER_SIZE,
    );
    let version_id = u64::from_le_bytes(
        row_bytes[OFF_VERSION_ID..OFF_VERSION_ID + 8]
            .try_into()
            .unwrap(),
    );
    let row_id = u64::from_le_bytes(row_bytes[OFF_ROW_ID..OFF_ROW_ID + 8].try_into().unwrap());
    let group_id = u64::from_le_bytes(
        row_bytes[OFF_GROUP_ID..OFF_GROUP_ID + 8]
            .try_into()
            .unwrap(),
    );
    let valid_from = u64::from_le_bytes(
        row_bytes[OFF_VALID_FROM..OFF_VALID_FROM + 8]
            .try_into()
            .unwrap(),
    );
    let raw_vu = u64::from_le_bytes(
        row_bytes[OFF_VALID_UNTIL..OFF_VALID_UNTIL + 8]
            .try_into()
            .unwrap(),
    );
    let valid_until = if raw_vu == 0 { None } else { Some(raw_vu) };
    let created_at = u64::from_le_bytes(
        row_bytes[OFF_CREATED_AT..OFF_CREATED_AT + 8]
            .try_into()
            .unwrap(),
    );
    let flags = u16::from_le_bytes(row_bytes[OFF_FLAGS..OFF_FLAGS + 2].try_into().unwrap());

    RowHeader {
        version_id,
        row_id,
        group_id,
        valid_from,
        valid_until,
        created_at,
        flags,
    }
}

/// Read valid_until directly from row bytes. Offset 32, 8 bytes.
pub fn read_valid_until(row_bytes: &[u8]) -> Option<Timestamp> {
    let raw = u64::from_le_bytes(
        row_bytes[OFF_VALID_UNTIL..OFF_VALID_UNTIL + 8]
            .try_into()
            .unwrap(),
    );
    if raw == 0 {
        None
    } else {
        Some(raw)
    }
}

/// Write valid_until in place within row bytes.
pub fn write_valid_until(row_bytes: &mut [u8], valid_until: Timestamp) {
    row_bytes[OFF_VALID_UNTIL..OFF_VALID_UNTIL + 8].copy_from_slice(&valid_until.to_le_bytes());
}

/// Read flags from row bytes.
pub fn read_flags(row_bytes: &[u8]) -> u16 {
    u16::from_le_bytes(row_bytes[OFF_FLAGS..OFF_FLAGS + 2].try_into().unwrap())
}

/// Set the COMMITTED bit in flags. Single 2-byte in-place write.
pub fn set_committed(row_bytes: &mut [u8]) {
    let flags = read_flags(row_bytes) | ROW_FLAG_COMMITTED;
    row_bytes[OFF_FLAGS..OFF_FLAGS + 2].copy_from_slice(&flags.to_le_bytes());
}

/// Check if row version is committed.
pub fn is_committed(row_bytes: &[u8]) -> bool {
    read_flags(row_bytes) & ROW_FLAG_COMMITTED != 0
}

/// Check null bitmap for a column.
pub fn is_null(row_bytes: &[u8], col_idx: usize) -> bool {
    let bitmap_start = ROW_HEADER_SIZE;
    let byte_idx = bitmap_start + col_idx / 8;
    let bit = col_idx % 8;
    (row_bytes[byte_idx] >> bit) & 1 == 1
}

/// Read a single column value from encoded row bytes WITHOUT decoding the full row.
/// O(1) — uses precomputed layout offsets.
pub fn read_column(
    layout: &RowLayout,
    row_bytes: &[u8],
    col_idx: usize,
    col_type: &ColumnType,
) -> CellValue {
    if is_null(row_bytes, col_idx) {
        return CellValue::Null;
    }

    let fixed_start = ROW_HEADER_SIZE + layout.null_bitmap_bytes;

    match &layout.column_offsets[col_idx] {
        ColumnOffset::Fixed { offset, .. } => {
            let pos = fixed_start + offset;
            match col_type {
                ColumnType::Int64 => CellValue::Int64(i64::from_le_bytes(
                    row_bytes[pos..pos + 8].try_into().unwrap(),
                )),
                ColumnType::Float64 => CellValue::Float64(f64::from_le_bytes(
                    row_bytes[pos..pos + 8].try_into().unwrap(),
                )),
                ColumnType::Bool => CellValue::Bool(row_bytes[pos] != 0),
                ColumnType::Timestamp => CellValue::Timestamp(u64::from_le_bytes(
                    row_bytes[pos..pos + 8].try_into().unwrap(),
                )),
                ColumnType::NodeRef => CellValue::NodeRef(u64::from_le_bytes(
                    row_bytes[pos..pos + 8].try_into().unwrap(),
                )),
                _ => CellValue::Null, // shouldn't happen for fixed types
            }
        },
        ColumnOffset::VarLen { index } => {
            let var_table_start = ROW_HEADER_SIZE + layout.fixed_section_size;
            let table_pos = var_table_start + index * 4;
            let data_offset =
                u16::from_le_bytes(row_bytes[table_pos..table_pos + 2].try_into().unwrap())
                    as usize;
            let data_len =
                u16::from_le_bytes(row_bytes[table_pos + 2..table_pos + 4].try_into().unwrap())
                    as usize;

            if data_len == 0 {
                return match col_type {
                    ColumnType::String => CellValue::String(String::new()),
                    ColumnType::Json => CellValue::Json(serde_json::Value::Null),
                    _ => CellValue::Null,
                };
            }

            let var_data_start = var_table_start + layout.var_len_count * 4;
            let start = var_data_start + data_offset;
            let end = start + data_len;
            let data = &row_bytes[start..end];

            match col_type {
                ColumnType::String => {
                    CellValue::String(std::str::from_utf8(data).unwrap_or_default().to_string())
                },
                ColumnType::Json => match rmp_serde::from_slice(data) {
                    Ok(v) => CellValue::Json(v),
                    Err(_) => CellValue::Json(serde_json::Value::Null),
                },
                _ => CellValue::Null,
            }
        },
    }
}

/// Decode a full row from page bytes back to CellValues.
pub fn decode_row(
    layout: &RowLayout,
    row_bytes: &[u8],
    schema_columns: &[crate::schema::ColumnDef],
) -> DecodedRow {
    let hdr = read_header(row_bytes);
    let mut values = Vec::with_capacity(schema_columns.len());
    for (i, col) in schema_columns.iter().enumerate() {
        values.push(read_column(layout, row_bytes, i, &col.col_type));
    }
    DecodedRow {
        version_id: hdr.version_id,
        row_id: hdr.row_id,
        group_id: hdr.group_id,
        valid_from: hdr.valid_from,
        valid_until: hdr.valid_until,
        created_at: hdr.created_at,
        flags: hdr.flags,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColumnDef, RowLayout, TableSchema};

    fn col(name: &str, col_type: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type,
            nullable: true,
            default_value: None,
            autoincrement: false,
        }
    }

    fn test_schema() -> TableSchema {
        TableSchema {
            table_id: 1,
            name: "test".into(),
            columns: vec![
                col("id", ColumnType::Int64),
                col("name", ColumnType::String),
                col("score", ColumnType::Float64),
                col("active", ColumnType::Bool),
                col("data", ColumnType::Json),
                col("ref", ColumnType::NodeRef),
            ],
            constraints: vec![],
            created_at: 0,
            schema_version: 1,
        }
    }

    #[test]
    fn test_roundtrip_encode_decode() {
        let schema = test_schema();
        let layout = RowLayout::from_schema(&schema);
        let values = vec![
            CellValue::Int64(42),
            CellValue::String("hello world".into()),
            CellValue::Float64(3.125),
            CellValue::Bool(true),
            CellValue::Json(serde_json::json!({"key": "value"})),
            CellValue::NodeRef(999),
        ];

        let bytes = encode_row(
            &layout,
            1,
            100,
            5,
            1000,
            None,
            1000,
            0,
            &values,
            &schema.columns,
        );
        let decoded = decode_row(&layout, &bytes, &schema.columns);

        assert_eq!(decoded.version_id, 1);
        assert_eq!(decoded.row_id, 100);
        assert_eq!(decoded.group_id, 5);
        assert_eq!(decoded.valid_from, 1000);
        assert_eq!(decoded.valid_until, None);
        assert_eq!(decoded.created_at, 1000);
        assert_eq!(decoded.flags, 0);

        // Check values
        match &decoded.values[0] {
            CellValue::Int64(v) => assert_eq!(*v, 42),
            other => panic!("expected Int64, got {:?}", other),
        }
        match &decoded.values[1] {
            CellValue::String(v) => assert_eq!(v, "hello world"),
            other => panic!("expected String, got {:?}", other),
        }
        match &decoded.values[2] {
            CellValue::Float64(v) => assert!((v - 3.125).abs() < 1e-10),
            other => panic!("expected Float64, got {:?}", other),
        }
        match &decoded.values[3] {
            CellValue::Bool(v) => assert!(*v),
            other => panic!("expected Bool, got {:?}", other),
        }
        match &decoded.values[4] {
            CellValue::Json(v) => assert_eq!(v["key"], "value"),
            other => panic!("expected Json, got {:?}", other),
        }
        match &decoded.values[5] {
            CellValue::NodeRef(v) => assert_eq!(*v, 999),
            other => panic!("expected NodeRef, got {:?}", other),
        }
    }

    #[test]
    fn test_null_handling() {
        let schema = test_schema();
        let layout = RowLayout::from_schema(&schema);
        let values = vec![
            CellValue::Int64(1),
            CellValue::Null,
            CellValue::Null,
            CellValue::Bool(false),
            CellValue::Null,
            CellValue::Null,
        ];

        let bytes = encode_row(&layout, 1, 1, 0, 0, None, 0, 0, &values, &schema.columns);
        let decoded = decode_row(&layout, &bytes, &schema.columns);

        assert!(matches!(decoded.values[1], CellValue::Null));
        assert!(matches!(decoded.values[2], CellValue::Null));
        assert!(matches!(decoded.values[4], CellValue::Null));
        assert!(matches!(decoded.values[5], CellValue::Null));
    }

    #[test]
    fn test_o1_column_access() {
        let schema = test_schema();
        let layout = RowLayout::from_schema(&schema);
        let values = vec![
            CellValue::Int64(42),
            CellValue::String("test".into()),
            CellValue::Float64(2.725),
            CellValue::Bool(true),
            CellValue::Json(serde_json::json!(null)),
            CellValue::NodeRef(7),
        ];

        let bytes = encode_row(&layout, 1, 1, 0, 0, None, 0, 0, &values, &schema.columns);

        // Read individual columns without full decode
        match read_column(&layout, &bytes, 0, &ColumnType::Int64) {
            CellValue::Int64(v) => assert_eq!(v, 42),
            other => panic!("expected Int64, got {:?}", other),
        }
        match read_column(&layout, &bytes, 1, &ColumnType::String) {
            CellValue::String(v) => assert_eq!(v, "test"),
            other => panic!("expected String, got {:?}", other),
        }
        match read_column(&layout, &bytes, 5, &ColumnType::NodeRef) {
            CellValue::NodeRef(v) => assert_eq!(v, 7),
            other => panic!("expected NodeRef, got {:?}", other),
        }
    }

    #[test]
    fn test_valid_until_in_place() {
        let schema = test_schema();
        let layout = RowLayout::from_schema(&schema);
        let values = vec![
            CellValue::Int64(1),
            CellValue::String("x".into()),
            CellValue::Float64(0.0),
            CellValue::Bool(false),
            CellValue::Json(serde_json::json!(null)),
            CellValue::NodeRef(0),
        ];

        let mut bytes = encode_row(
            &layout,
            1,
            1,
            0,
            100,
            None,
            100,
            0,
            &values,
            &schema.columns,
        );
        assert_eq!(read_valid_until(&bytes), None);

        write_valid_until(&mut bytes, 200);
        assert_eq!(read_valid_until(&bytes), Some(200));

        let hdr = read_header(&bytes);
        assert_eq!(hdr.valid_until, Some(200));
        assert_eq!(hdr.valid_from, 100); // unchanged
    }

    #[test]
    fn test_committed_flag() {
        let schema = test_schema();
        let layout = RowLayout::from_schema(&schema);
        let values = vec![
            CellValue::Int64(1),
            CellValue::String("x".into()),
            CellValue::Float64(0.0),
            CellValue::Bool(false),
            CellValue::Json(serde_json::json!(null)),
            CellValue::NodeRef(0),
        ];

        let mut bytes = encode_row(&layout, 1, 1, 0, 0, None, 0, 0, &values, &schema.columns);
        assert!(!is_committed(&bytes));
        assert_eq!(read_flags(&bytes), 0);

        set_committed(&mut bytes);
        assert!(is_committed(&bytes));
        assert_eq!(read_flags(&bytes), ROW_FLAG_COMMITTED);
    }

    #[test]
    fn test_all_fixed_schema() {
        let schema = TableSchema {
            table_id: 1,
            name: "fixed_only".into(),
            columns: vec![
                col("a", ColumnType::Int64),
                col("b", ColumnType::Bool),
                col("c", ColumnType::Timestamp),
            ],
            constraints: vec![],
            created_at: 0,
            schema_version: 1,
        };
        let layout = RowLayout::from_schema(&schema);
        assert!(layout.all_fixed);

        let values = vec![
            CellValue::Int64(-100),
            CellValue::Bool(true),
            CellValue::Timestamp(999_999),
        ];

        let bytes = encode_row(
            &layout,
            5,
            10,
            1,
            500,
            Some(600),
            500,
            0,
            &values,
            &schema.columns,
        );
        let decoded = decode_row(&layout, &bytes, &schema.columns);

        assert_eq!(decoded.version_id, 5);
        assert_eq!(decoded.row_id, 10);
        assert_eq!(decoded.group_id, 1);
        assert_eq!(decoded.valid_from, 500);
        assert_eq!(decoded.valid_until, Some(600));
        match &decoded.values[0] {
            CellValue::Int64(v) => assert_eq!(*v, -100),
            other => panic!("expected Int64, got {:?}", other),
        }
        match &decoded.values[2] {
            CellValue::Timestamp(v) => assert_eq!(*v, 999_999),
            other => panic!("expected Timestamp, got {:?}", other),
        }
    }
}
