//! CSV/JSON/NDJSON bulk import.

use rustc_hash::FxHashMap;

use crate::error::TableError;
use crate::table::Table;
use crate::types::{CellValue, GroupId};

pub struct ImportConfig {
    pub batch_size: usize,
    pub skip_errors: bool,
    pub column_mapping: Option<FxHashMap<String, String>>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        ImportConfig {
            batch_size: 10_000,
            skip_errors: false,
            column_mapping: None,
        }
    }
}

pub struct ImportResult {
    pub rows_imported: u64,
    pub rows_skipped: u64,
    pub errors: Vec<(u64, String)>,
}

/// Import CSV data into a table.
pub fn import_csv<R: std::io::Read>(
    table: &mut Table,
    group_id: GroupId,
    reader: R,
    config: &ImportConfig,
) -> Result<ImportResult, TableError> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(reader);

    let headers: Vec<String> = csv_reader
        .headers()
        .map_err(|e| TableError::ImportError(e.to_string()))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    // Map CSV columns to table columns
    let col_map = build_column_map(&headers, table, config)?;

    let mut result = ImportResult {
        rows_imported: 0,
        rows_skipped: 0,
        errors: Vec::new(),
    };
    let mut batch = Vec::new();
    let mut line_num = 1u64;

    for record in csv_reader.records() {
        line_num += 1;
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                if config.skip_errors {
                    result.errors.push((line_num, e.to_string()));
                    result.rows_skipped += 1;
                    continue;
                }
                return Err(TableError::ImportError(format!("line {}: {}", line_num, e)));
            },
        };

        match parse_csv_row(&record, &col_map, table) {
            Ok(values) => batch.push(values),
            Err(e) => {
                if config.skip_errors {
                    result.errors.push((line_num, e.to_string()));
                    result.rows_skipped += 1;
                    continue;
                }
                return Err(e);
            },
        }

        if batch.len() >= config.batch_size {
            let count = batch.len() as u64;
            table.insert_batch(group_id, std::mem::take(&mut batch))?;
            result.rows_imported += count;
        }
    }

    if !batch.is_empty() {
        let count = batch.len() as u64;
        table.insert_batch(group_id, batch)?;
        result.rows_imported += count;
    }

    Ok(result)
}

/// Import NDJSON (newline-delimited JSON) into a table.
pub fn import_ndjson<R: std::io::BufRead>(
    table: &mut Table,
    group_id: GroupId,
    reader: R,
    config: &ImportConfig,
) -> Result<ImportResult, TableError> {
    let mut result = ImportResult {
        rows_imported: 0,
        rows_skipped: 0,
        errors: Vec::new(),
    };
    let mut batch = Vec::new();
    let mut line_num = 0u64;

    for line in reader.lines() {
        line_num += 1;
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                if config.skip_errors {
                    result.errors.push((line_num, e.to_string()));
                    result.rows_skipped += 1;
                    continue;
                }
                return Err(TableError::ImportError(format!("line {}: {}", line_num, e)));
            },
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let obj: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                if config.skip_errors {
                    result.errors.push((line_num, e.to_string()));
                    result.rows_skipped += 1;
                    continue;
                }
                return Err(TableError::ImportError(format!("line {}: {}", line_num, e)));
            },
        };

        match json_obj_to_row(&obj, table, config) {
            Ok(values) => batch.push(values),
            Err(e) => {
                if config.skip_errors {
                    result.errors.push((line_num, e.to_string()));
                    result.rows_skipped += 1;
                    continue;
                }
                return Err(e);
            },
        }

        if batch.len() >= config.batch_size {
            let count = batch.len() as u64;
            table.insert_batch(group_id, std::mem::take(&mut batch))?;
            result.rows_imported += count;
        }
    }

    if !batch.is_empty() {
        let count = batch.len() as u64;
        table.insert_batch(group_id, batch)?;
        result.rows_imported += count;
    }

    Ok(result)
}

/// Import a JSON array of objects into a table.
pub fn import_json_array(
    table: &mut Table,
    group_id: GroupId,
    json: &serde_json::Value,
    config: &ImportConfig,
) -> Result<ImportResult, TableError> {
    let arr = json
        .as_array()
        .ok_or_else(|| TableError::ImportError("expected JSON array".into()))?;

    let mut result = ImportResult {
        rows_imported: 0,
        rows_skipped: 0,
        errors: Vec::new(),
    };
    let mut batch = Vec::new();

    for (i, obj) in arr.iter().enumerate() {
        match json_obj_to_row(obj, table, config) {
            Ok(values) => batch.push(values),
            Err(e) => {
                if config.skip_errors {
                    result.errors.push((i as u64, e.to_string()));
                    result.rows_skipped += 1;
                    continue;
                }
                return Err(e);
            },
        }

        if batch.len() >= config.batch_size {
            let count = batch.len() as u64;
            table.insert_batch(group_id, std::mem::take(&mut batch))?;
            result.rows_imported += count;
        }
    }

    if !batch.is_empty() {
        let count = batch.len() as u64;
        table.insert_batch(group_id, batch)?;
        result.rows_imported += count;
    }

    Ok(result)
}

// -- Internal helpers --

struct ColumnMapping {
    /// Index into table schema columns.
    table_col_idx: usize,
    /// Index into source (CSV) columns.
    source_col_idx: usize,
}

fn build_column_map(
    headers: &[String],
    table: &Table,
    config: &ImportConfig,
) -> Result<Vec<ColumnMapping>, TableError> {
    let mut mappings = Vec::new();

    for (table_idx, col_def) in table.schema.columns.iter().enumerate() {
        let source_name = if let Some(ref mapping) = config.column_mapping {
            mapping
                .get(&col_def.name)
                .cloned()
                .unwrap_or_else(|| col_def.name.clone())
        } else {
            col_def.name.clone()
        };

        if let Some(source_idx) = headers.iter().position(|h| h == &source_name) {
            mappings.push(ColumnMapping {
                table_col_idx: table_idx,
                source_col_idx: source_idx,
            });
        } else if !col_def.nullable && col_def.default_value.is_none() {
            return Err(TableError::ImportError(format!(
                "required column '{}' not found in CSV headers",
                col_def.name
            )));
        }
    }

    Ok(mappings)
}

fn parse_csv_row(
    record: &csv::StringRecord,
    col_map: &[ColumnMapping],
    table: &Table,
) -> Result<Vec<CellValue>, TableError> {
    let mut values = vec![CellValue::Null; table.schema.columns.len()];

    for mapping in col_map {
        let raw = record.get(mapping.source_col_idx).unwrap_or("");
        let col_def = &table.schema.columns[mapping.table_col_idx];
        let val = CellValue::parse_from_str(raw, &col_def.col_type)
            .map_err(|e| TableError::ImportError(format!("column '{}': {}", col_def.name, e)))?;
        values[mapping.table_col_idx] = val;
    }

    // Apply defaults for missing columns
    for (i, col_def) in table.schema.columns.iter().enumerate() {
        if matches!(values[i], CellValue::Null) {
            if let Some(ref default) = col_def.default_value {
                values[i] = default.clone();
            }
        }
    }

    Ok(values)
}

fn json_obj_to_row(
    obj: &serde_json::Value,
    table: &Table,
    config: &ImportConfig,
) -> Result<Vec<CellValue>, TableError> {
    let map = obj
        .as_object()
        .ok_or_else(|| TableError::ImportError("expected JSON object".into()))?;

    let mut values = vec![CellValue::Null; table.schema.columns.len()];

    for (i, col_def) in table.schema.columns.iter().enumerate() {
        let source_name = if let Some(ref mapping) = config.column_mapping {
            mapping
                .get(&col_def.name)
                .cloned()
                .unwrap_or_else(|| col_def.name.clone())
        } else {
            col_def.name.clone()
        };

        if let Some(json_val) = map.get(&source_name) {
            values[i] = json_value_to_cell(json_val, &col_def.col_type)?;
        } else if let Some(ref default) = col_def.default_value {
            values[i] = default.clone();
        }
    }

    Ok(values)
}

fn json_value_to_cell(
    val: &serde_json::Value,
    col_type: &crate::types::ColumnType,
) -> Result<CellValue, TableError> {
    use crate::types::ColumnType;

    if val.is_null() {
        return Ok(CellValue::Null);
    }

    match col_type {
        ColumnType::String => match val {
            serde_json::Value::String(s) => Ok(CellValue::String(s.clone())),
            other => Ok(CellValue::String(other.to_string())),
        },
        ColumnType::Int64 => {
            if let Some(n) = val.as_i64() {
                Ok(CellValue::Int64(n))
            } else if let Some(s) = val.as_str() {
                s.parse::<i64>()
                    .map(CellValue::Int64)
                    .map_err(|e| TableError::ImportError(e.to_string()))
            } else {
                Err(TableError::ImportError(format!(
                    "cannot convert {:?} to Int64",
                    val
                )))
            }
        },
        ColumnType::Float64 => {
            if let Some(n) = val.as_f64() {
                Ok(CellValue::Float64(n))
            } else if let Some(s) = val.as_str() {
                s.parse::<f64>()
                    .map(CellValue::Float64)
                    .map_err(|e| TableError::ImportError(e.to_string()))
            } else {
                Err(TableError::ImportError(format!(
                    "cannot convert {:?} to Float64",
                    val
                )))
            }
        },
        ColumnType::Bool => {
            if let Some(b) = val.as_bool() {
                Ok(CellValue::Bool(b))
            } else {
                Err(TableError::ImportError(format!(
                    "cannot convert {:?} to Bool",
                    val
                )))
            }
        },
        ColumnType::Timestamp => {
            if let Some(n) = val.as_u64() {
                Ok(CellValue::Timestamp(n))
            } else if let Some(n) = val.as_i64() {
                Ok(CellValue::Timestamp(n as u64))
            } else {
                Err(TableError::ImportError(format!(
                    "cannot convert {:?} to Timestamp",
                    val
                )))
            }
        },
        ColumnType::Json => Ok(CellValue::Json(val.clone())),
        ColumnType::NodeRef => {
            if let Some(n) = val.as_u64() {
                Ok(CellValue::NodeRef(n))
            } else if let Some(n) = val.as_i64() {
                Ok(CellValue::NodeRef(n as u64))
            } else {
                Err(TableError::ImportError(format!(
                    "cannot convert {:?} to NodeRef",
                    val
                )))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColumnDef, Constraint, TableSchema};
    use crate::types::ColumnType;

    fn col(name: &str, ct: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type: ct,
            nullable: true,
            default_value: None,
        }
    }

    fn test_table() -> Table {
        Table::new(TableSchema {
            table_id: 1,
            name: "items".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                    default_value: None,
                },
                col("name", ColumnType::String),
                col("price", ColumnType::Float64),
            ],
            constraints: vec![Constraint::PrimaryKey(vec!["id".into()])],
            created_at: 0,
            schema_version: 1,
        })
        .unwrap()
    }

    #[test]
    fn test_import_csv() {
        let mut table = test_table();
        let csv_data = "id,name,price\n1,Apple,1.50\n2,Banana,0.75\n3,Cherry,3.00\n";
        let config = ImportConfig::default();
        let result = import_csv(&mut table, 0, csv_data.as_bytes(), &config).unwrap();
        assert_eq!(result.rows_imported, 3);
        assert_eq!(result.rows_skipped, 0);
        assert_eq!(table.active_row_count(), 3);
    }

    #[test]
    fn test_import_csv_skip_errors() {
        let mut table = test_table();
        let csv_data = "id,name,price\n1,Apple,1.50\nnot_int,Bad,0.0\n3,Cherry,3.00\n";
        let config = ImportConfig {
            skip_errors: true,
            ..Default::default()
        };
        let result = import_csv(&mut table, 0, csv_data.as_bytes(), &config).unwrap();
        assert_eq!(result.rows_imported, 2);
        assert_eq!(result.rows_skipped, 1);
    }

    #[test]
    fn test_import_ndjson() {
        let mut table = test_table();
        let ndjson = r#"{"id":1,"name":"Apple","price":1.50}
{"id":2,"name":"Banana","price":0.75}
"#;
        let config = ImportConfig::default();
        let result = import_ndjson(
            &mut table,
            0,
            std::io::BufReader::new(ndjson.as_bytes()),
            &config,
        )
        .unwrap();
        assert_eq!(result.rows_imported, 2);
        assert_eq!(table.active_row_count(), 2);
    }

    #[test]
    fn test_import_json_array() {
        let mut table = test_table();
        let json = serde_json::json!([
            {"id": 1, "name": "Apple", "price": 1.50},
            {"id": 2, "name": "Banana", "price": 0.75},
        ]);
        let config = ImportConfig::default();
        let result = import_json_array(&mut table, 0, &json, &config).unwrap();
        assert_eq!(result.rows_imported, 2);
        assert_eq!(table.active_row_count(), 2);
    }
}
