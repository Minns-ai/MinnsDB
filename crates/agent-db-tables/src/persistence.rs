//! ReDB persistence for temporal tables.
//!
//! Rows are stored individually keyed by (table_id, version_id).
//!
//! ReDB key formats:
//! - table_schemas:  [table_id: 8B BE]                     -> msgpack(TableSchema)
//! - table_rows:     [table_id: 8B BE][version_id: 8B BE]  -> raw row bytes
//! - table_meta:     [table_id: 8B BE]                     -> msgpack(TableMeta)

use agent_db_storage::{BatchOperation, RedbBackend};
use serde::{Deserialize, Serialize};

use crate::catalog::TableCatalog;
use crate::error::TableError;
use crate::table::Table;
use crate::types::*;

const TABLE_SCHEMAS: &str = "table_schemas";
const TABLE_ROWS: &str = "table_rows";
const TABLE_META: &str = "table_meta";

#[derive(Debug, Serialize, Deserialize)]
pub struct TableMeta {
    pub next_version_id: RowVersionId,
    pub next_row_id: RowId,
    pub next_table_id: TableId,
    pub generation: u64,
}

/// Persist a single table (all rows + schema + meta) atomically.
pub fn persist_table(backend: &RedbBackend, table: &Table) -> Result<(), TableError> {
    let table_id = table.schema.table_id;
    let schema_key = table_id.to_be_bytes().to_vec();

    let mut ops = Vec::new();

    // Schema
    let schema_bytes = rmp_serde::to_vec(&table.schema)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;
    ops.push(BatchOperation::Put {
        table_name: TABLE_SCHEMAS.into(),
        key: schema_key.clone(),
        value: schema_bytes,
    });

    // All rows
    for (vid, row_bytes) in table.iter_rows() {
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&table_id.to_be_bytes());
        key.extend_from_slice(&vid.to_be_bytes());
        ops.push(BatchOperation::Put {
            table_name: TABLE_ROWS.into(),
            key,
            value: row_bytes.to_vec(),
        });
    }

    // Meta
    let meta = TableMeta {
        next_version_id: table.next_version_id(),
        next_row_id: table.next_row_id(),
        next_table_id: 0,
        generation: table.generation(),
    };
    let meta_bytes =
        rmp_serde::to_vec(&meta).map_err(|e| TableError::PersistenceError(e.to_string()))?;
    ops.push(BatchOperation::Put {
        table_name: TABLE_META.into(),
        key: schema_key,
        value: meta_bytes,
    });

    backend
        .write_batch(ops)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;

    Ok(())
}

/// Persist all tables in the catalog.
pub fn persist_catalog(
    backend: &RedbBackend,
    catalog: &mut TableCatalog,
) -> Result<(), TableError> {
    // Persist catalog-level meta (next_table_id)
    let catalog_meta = TableMeta {
        next_version_id: 0,
        next_row_id: 0,
        next_table_id: catalog.next_table_id(),
        generation: 0,
    };
    let meta_key = 0u64.to_be_bytes();
    let meta_bytes = rmp_serde::to_vec(&catalog_meta)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;
    backend
        .put_raw(TABLE_META, &meta_key[..], &meta_bytes)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;

    for table in catalog.tables_mut() {
        persist_table(backend, table)?;
        table.clear_dirty();
    }
    Ok(())
}

/// Load the full catalog from ReDB.
pub fn load_catalog(backend: &RedbBackend) -> Result<TableCatalog, TableError> {
    let mut catalog = TableCatalog::new();

    // Load catalog-level meta
    let meta_key = 0u64.to_be_bytes();
    if let Some(meta_bytes) = backend
        .get_raw(TABLE_META, &meta_key[..])
        .map_err(|e| TableError::PersistenceError(e.to_string()))?
    {
        let meta: TableMeta = rmp_serde::from_slice(&meta_bytes)
            .map_err(|e| TableError::PersistenceError(e.to_string()))?;
        catalog.set_next_table_id(meta.next_table_id);
    }

    // Load all schemas
    let schema_entries = backend
        .scan_all_raw(TABLE_SCHEMAS)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;

    for (key_bytes, schema_bytes) in &schema_entries {
        if key_bytes.len() != 8 {
            continue;
        }
        let table_id = u64::from_be_bytes(key_bytes[..8].try_into().unwrap());

        let schema: crate::schema::TableSchema = rmp_serde::from_slice(schema_bytes)
            .map_err(|e| TableError::PersistenceError(e.to_string()))?;

        // Load rows for this table
        let prefix = table_id.to_be_bytes();
        let row_entries = backend
            .scan_prefix_raw(TABLE_ROWS, &prefix[..])
            .map_err(|e| TableError::PersistenceError(e.to_string()))?;

        let mut persisted_rows = Vec::new();
        for (key, row_data) in row_entries {
            if key.len() == 16 {
                let vid = u64::from_be_bytes(key[8..16].try_into().unwrap());
                persisted_rows.push((vid, row_data));
            }
        }

        // Load meta for this table
        let meta_key = table_id.to_be_bytes();
        let (next_version_id, next_row_id, generation) = if let Some(meta_bytes) = backend
            .get_raw(TABLE_META, &meta_key[..])
            .map_err(|e| TableError::PersistenceError(e.to_string()))?
        {
            let meta: TableMeta = rmp_serde::from_slice(&meta_bytes)
                .map_err(|e| TableError::PersistenceError(e.to_string()))?;
            (meta.next_version_id, meta.next_row_id, meta.generation)
        } else {
            (1, 1, 0)
        };

        let table = Table::from_persisted(
            schema,
            persisted_rows,
            next_version_id,
            next_row_id,
            generation,
        )?;
        catalog.insert_table(table);
    }

    // Ensure next_table_id is at least max(loaded_table_ids) + 1 to prevent
    // collisions if the catalog-level meta was stale due to a crash.
    let max_loaded_id = catalog
        .tables()
        .map(|t| t.schema.table_id)
        .max()
        .unwrap_or(0);
    if catalog.next_table_id() <= max_loaded_id {
        catalog.set_next_table_id(max_loaded_id + 1);
    }

    Ok(catalog)
}
