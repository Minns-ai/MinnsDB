//! ReDB persistence for temporal tables.
//!
//! Pages are stored as raw 8KB blobs — no per-field serialization overhead.
//!
//! ReDB key formats:
//! - table_schemas:  [table_id: 8B BE]                 -> msgpack(TableSchema)
//! - table_pages:    [table_id: 8B BE][page_id: 4B BE] -> raw [u8; 8192]
//! - table_meta:     [table_id: 8B BE]                 -> msgpack(TableMeta)

use agent_db_storage::RedbBackend;
use serde::{Deserialize, Serialize};

use crate::catalog::TableCatalog;
use crate::error::TableError;
use crate::page::Page;
use crate::table::Table;
use crate::types::*;

const TABLE_SCHEMAS: &str = "table_schemas";
const TABLE_PAGES: &str = "table_pages";
const TABLE_META: &str = "table_meta";

#[derive(Debug, Serialize, Deserialize)]
pub struct TableMeta {
    pub next_version_id: RowVersionId,
    pub next_row_id: RowId,
    pub next_table_id: TableId,
    pub generation: u64,
}

/// Persist a single table (dirty pages + schema + meta).
/// Call `update_checksums()` on the table's store before this if checksums are needed.
pub fn persist_table(backend: &RedbBackend, table: &Table) -> Result<(), TableError> {
    let table_id = table.schema.table_id;

    // Write schema
    let schema_key = table_id.to_be_bytes();
    let schema_bytes = rmp_serde::to_vec(&table.schema)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;
    backend
        .put_raw(TABLE_SCHEMAS, &schema_key[..], &schema_bytes)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;

    // Write dirty pages as raw 8KB blobs
    for &page_id in table.store().dirty_pages() {
        if let Some(page) = table.store().get_page(page_id) {
            let mut key = Vec::with_capacity(12);
            key.extend_from_slice(&table_id.to_be_bytes());
            key.extend_from_slice(&page_id.to_be_bytes());
            backend
                .put_raw(TABLE_PAGES, key.as_slice(), page.as_bytes_readonly())
                .map_err(|e| TableError::PersistenceError(e.to_string()))?;
        }
    }

    // Write meta
    let meta = TableMeta {
        next_version_id: table.next_version_id(),
        next_row_id: table.next_row_id(),
        next_table_id: 0, // not used per-table; set at catalog level
        generation: table.generation(),
    };
    let meta_bytes =
        rmp_serde::to_vec(&meta).map_err(|e| TableError::PersistenceError(e.to_string()))?;
    backend
        .put_raw(TABLE_META, &schema_key[..], &meta_bytes)
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
    let meta_key = 0u64.to_be_bytes(); // special key for catalog-level meta
    let meta_bytes = rmp_serde::to_vec(&catalog_meta)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;
    backend
        .put_raw(TABLE_META, &meta_key[..], &meta_bytes)
        .map_err(|e| TableError::PersistenceError(e.to_string()))?;

    for table in catalog.tables_mut() {
        table.store_mut().update_checksums();
        persist_table(backend, table)?;
        table.store_mut().clear_dirty();
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

        // Load pages for this table
        let prefix = table_id.to_be_bytes();
        let page_entries = backend
            .scan_prefix_raw(TABLE_PAGES, &prefix[..])
            .map_err(|e| TableError::PersistenceError(e.to_string()))?;

        let mut pages = Vec::new();
        for (_key, page_data) in page_entries {
            if page_data.len() == PAGE_SIZE {
                let mut arr = Box::new([0u8; PAGE_SIZE]);
                arr.copy_from_slice(&page_data);
                pages.push(Page::from_bytes(arr));
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

        let table = Table::from_persisted(schema, pages, next_version_id, next_row_id, generation)?;
        catalog.insert_table(table);
    }

    Ok(catalog)
}
