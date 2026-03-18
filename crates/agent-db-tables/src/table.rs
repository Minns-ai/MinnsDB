//! Table engine: page store + indexes + mutations + queries.

use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::error::TableError;
use crate::page::Page;
use crate::page_store::PageStore;
use crate::row_codec::{self, DecodedRow};
use crate::schema::{Constraint, RowLayout, TableSchema};
use crate::types::*;

use agent_db_core::types::current_timestamp;

pub struct Table {
    pub schema: TableSchema,
    pub layout: RowLayout,

    // -- Storage --
    store: PageStore,

    // -- Indexes (all derived — rebuilt from pages on load) --
    /// (group_id, RowId) -> RowPointer of active version. None = deleted.
    pk_index: FxHashMap<(GroupId, RowId), Option<RowPointer>>,
    /// (group_id, RowId) -> all RowPointers, append-ordered by RowVersionId.
    history_index: FxHashMap<(GroupId, RowId), Vec<RowPointer>>,
    /// RowVersionId -> RowPointer. Direct version lookup.
    version_index: FxHashMap<RowVersionId, RowPointer>,
    /// valid_from -> RowPointers. For temporal range queries.
    temporal_index: BTreeMap<Timestamp, SmallVec<[RowPointer; 4]>>,
    /// Graph NodeId -> set of (group_id, RowId) referencing it.
    node_ref_index: FxHashMap<u64, FxHashSet<(GroupId, RowId)>>,
    /// Unique constraint enforcement (active rows only, scoped by group_id).
    unique_indexes: Vec<UniqueIndex>,

    // -- State --
    next_version_id: RowVersionId,
    next_row_id: RowId,
    generation: u64,
    active_row_count: usize,

    /// Column indices for NodeRef columns (cached from schema).
    node_ref_col_indices: Vec<usize>,
}

struct UniqueIndex {
    columns: Vec<usize>,
    /// (group_id, IndexKey) -> RowId for active rows.
    entries: FxHashMap<(GroupId, IndexKey), RowId>,
}

impl Table {
    pub fn new(schema: TableSchema) -> Result<Self, TableError> {
        schema.validate()?;
        let layout = RowLayout::from_schema(&schema);
        let node_ref_col_indices = schema.node_ref_columns();

        // Build unique indexes from constraints
        let mut unique_indexes = Vec::new();
        for constraint in &schema.constraints {
            match constraint {
                Constraint::Unique(cols) | Constraint::PrimaryKey(cols) => {
                    let indices: Vec<usize> = cols
                        .iter()
                        .filter_map(|name| schema.column_index(name))
                        .collect();
                    if indices.len() == cols.len() {
                        unique_indexes.push(UniqueIndex {
                            columns: indices,
                            entries: FxHashMap::default(),
                        });
                    }
                },
                _ => {},
            }
        }

        Ok(Table {
            schema,
            layout,
            store: PageStore::new(),
            pk_index: FxHashMap::default(),
            history_index: FxHashMap::default(),
            version_index: FxHashMap::default(),
            temporal_index: BTreeMap::new(),
            node_ref_index: FxHashMap::default(),
            unique_indexes,
            next_version_id: 1,
            next_row_id: 1,
            generation: 0,
            active_row_count: 0,
            node_ref_col_indices,
        })
    }

    /// Restore from persisted pages. Rebuilds all indexes by scanning page data.
    pub fn from_persisted(
        schema: TableSchema,
        pages: Vec<Page>,
        next_version_id: RowVersionId,
        next_row_id: RowId,
        generation: u64,
    ) -> Result<Self, TableError> {
        let mut table = Self::new(schema)?;
        table.store = PageStore::from_pages(pages);
        table.next_version_id = next_version_id;
        table.next_row_id = next_row_id;
        table.generation = generation;

        // Rebuild indexes from page contents
        table.rebuild_indexes();
        Ok(table)
    }

    /// Rebuild all indexes from page data. Used on load and recovery.
    fn rebuild_indexes(&mut self) {
        self.pk_index.clear();
        self.history_index.clear();
        self.version_index.clear();
        self.temporal_index.clear();
        self.node_ref_index.clear();
        for idx in &mut self.unique_indexes {
            idx.entries.clear();
        }
        self.active_row_count = 0;

        let entries: Vec<(RowPointer, Vec<u8>)> = self
            .store
            .iter_live()
            .map(|(ptr, data)| (ptr, data.to_vec()))
            .collect();

        for (ptr, data) in &entries {
            let hdr = row_codec::read_header(data);

            // Skip uncommitted rows during recovery
            if !row_codec::is_committed(data) {
                continue;
            }

            let key = (hdr.group_id, hdr.row_id);
            self.version_index.insert(hdr.version_id, *ptr);
            self.history_index.entry(key).or_default().push(*ptr);
            self.temporal_index
                .entry(hdr.valid_from)
                .or_default()
                .push(*ptr);

            if hdr.valid_until.is_none() {
                // Active version
                self.pk_index.insert(key, Some(*ptr));
                self.active_row_count += 1;

                // Unique index entries
                let values = self.decode_values(data);
                self.insert_unique_entries(hdr.group_id, hdr.row_id, &values);

                // NodeRef index
                self.insert_node_ref_entries(hdr.group_id, hdr.row_id, &values);
            }
        }

        // Sort history entries by version_id
        for entries in self.history_index.values_mut() {
            entries.sort_by_key(|ptr| {
                self.store
                    .read(*ptr)
                    .map(|d| row_codec::read_header(d).version_id)
                    .unwrap_or(0)
            });
        }
    }

    // -- Mutations --

    /// Insert one row.
    pub fn insert(
        &mut self,
        group_id: GroupId,
        values: Vec<CellValue>,
    ) -> Result<(RowId, RowVersionId), TableError> {
        self.validate_values(&values)?;

        let row_id = self.next_row_id;
        let vid = self.next_version_id;
        let key = (group_id, row_id);

        // Check unique constraints
        self.check_unique_constraints(group_id, &values, None)?;

        let now = current_timestamp();
        let bytes = row_codec::encode_row(
            &self.layout,
            vid,
            row_id,
            group_id,
            now,
            None,
            now,
            0, // uncommitted
            &values,
            &self.schema.columns,
        );

        // Check row size
        if bytes.len() > MAX_ROW_PAYLOAD {
            return Err(TableError::RowTooLarge {
                size: bytes.len(),
                max: MAX_ROW_PAYLOAD,
            });
        }

        let ptr = self.store.insert(&bytes);

        // Set committed
        let row_bytes = self.store.read_mut(ptr).unwrap();
        row_codec::set_committed(row_bytes);

        // Update indexes
        self.pk_index.insert(key, Some(ptr));
        self.history_index.entry(key).or_default().push(ptr);
        self.version_index.insert(vid, ptr);
        self.temporal_index.entry(now).or_default().push(ptr);
        self.insert_unique_entries(group_id, row_id, &values);
        self.insert_node_ref_entries(group_id, row_id, &values);

        self.next_row_id += 1;
        self.next_version_id += 1;
        self.generation += 1;
        self.active_row_count += 1;

        Ok((row_id, vid))
    }

    /// Update: append new version, then close old version in place.
    pub fn update(
        &mut self,
        group_id: GroupId,
        row_id: RowId,
        values: Vec<CellValue>,
    ) -> Result<(RowVersionId, RowVersionId), TableError> {
        let key = (group_id, row_id);
        let old_ptr = self.active_ptr(key)?;
        self.validate_values(&values)?;
        self.check_unique_constraints(group_id, &values, Some(row_id))?;

        let now = current_timestamp();
        let new_vid = self.next_version_id;

        // Step 1: Append new version (uncommitted)
        let new_bytes = row_codec::encode_row(
            &self.layout,
            new_vid,
            row_id,
            group_id,
            now,
            None,
            now,
            0,
            &values,
            &self.schema.columns,
        );

        if new_bytes.len() > MAX_ROW_PAYLOAD {
            return Err(TableError::RowTooLarge {
                size: new_bytes.len(),
                max: MAX_ROW_PAYLOAD,
            });
        }

        let new_ptr = self.store.insert(&new_bytes);

        // Step 2: Close old version
        let old_bytes = self.store.read_mut(old_ptr).unwrap();
        row_codec::write_valid_until(old_bytes, now);
        let old_vid = row_codec::read_header(old_bytes).version_id;

        // Step 3: Update indexes
        // Remove old active entries
        let old_values = {
            let old_data = self.store.read(old_ptr).unwrap();
            self.decode_values(old_data)
        };
        self.remove_unique_entries(group_id, &old_values);
        self.remove_node_ref_entries(group_id, row_id, &old_values);

        self.pk_index.insert(key, Some(new_ptr));
        self.history_index.entry(key).or_default().push(new_ptr);
        self.version_index.insert(new_vid, new_ptr);
        self.temporal_index.entry(now).or_default().push(new_ptr);
        self.insert_unique_entries(group_id, row_id, &values);
        self.insert_node_ref_entries(group_id, row_id, &values);

        // Step 4: Set committed flag on new version
        let committed_bytes = self.store.read_mut(new_ptr).unwrap();
        row_codec::set_committed(committed_bytes);

        // Step 5: Bump generation
        self.next_version_id += 1;
        self.generation += 1;

        Ok((old_vid, new_vid))
    }

    /// Delete: close version in place.
    pub fn delete(&mut self, group_id: GroupId, row_id: RowId) -> Result<RowVersionId, TableError> {
        let key = (group_id, row_id);
        let ptr = self.active_ptr(key)?;
        let now = current_timestamp();

        let bytes = self.store.read_mut(ptr).unwrap();
        row_codec::write_valid_until(bytes, now);
        let vid = row_codec::read_header(bytes).version_id;

        // Remove active entries
        let values = {
            let data = self.store.read(ptr).unwrap();
            self.decode_values(data)
        };
        self.remove_unique_entries(group_id, &values);
        self.remove_node_ref_entries(group_id, row_id, &values);

        self.pk_index.insert(key, None);
        self.active_row_count -= 1;
        self.generation += 1;

        Ok(vid)
    }

    /// Batch insert for bulk import.
    pub fn insert_batch(
        &mut self,
        group_id: GroupId,
        rows: Vec<Vec<CellValue>>,
    ) -> Result<Vec<(RowId, RowVersionId)>, TableError> {
        // Validate all rows first
        for (i, values) in rows.iter().enumerate() {
            self.validate_values(values)
                .map_err(|e| TableError::ImportError(format!("row {}: {}", i, e)))?;
        }

        let mut results = Vec::with_capacity(rows.len());
        for values in rows {
            results.push(self.insert(group_id, values)?);
        }
        Ok(results)
    }

    // -- Queries --

    /// Scan active rows for a group.
    pub fn scan_active(&self, group_id: GroupId) -> Vec<DecodedRow> {
        let mut results = Vec::new();
        for (&(gid, _rid), opt_ptr) in &self.pk_index {
            if gid != group_id {
                continue;
            }
            if let Some(ptr) = opt_ptr {
                if let Some(data) = self.store.read(*ptr) {
                    results.push(row_codec::decode_row(
                        &self.layout,
                        data,
                        &self.schema.columns,
                    ));
                }
            }
        }
        results
    }

    /// Point-in-time query: rows active at a specific timestamp for a group.
    pub fn scan_as_of(&self, group_id: GroupId, timestamp: Timestamp) -> Vec<DecodedRow> {
        let mut results = Vec::new();
        for (&(gid, _rid), ptrs) in &self.history_index {
            if gid != group_id {
                continue;
            }
            for ptr in ptrs.iter().rev() {
                if let Some(bytes) = self.store.read(*ptr) {
                    let hdr = row_codec::read_header(bytes);
                    if hdr.valid_from <= timestamp && hdr.valid_until.is_none_or(|u| u > timestamp)
                    {
                        results.push(row_codec::decode_row(
                            &self.layout,
                            bytes,
                            &self.schema.columns,
                        ));
                        break;
                    }
                }
            }
        }
        results
    }

    /// Full history (WHEN ALL) for a group.
    pub fn scan_all(&self, group_id: GroupId) -> Vec<DecodedRow> {
        let mut results = Vec::new();
        for (&(gid, _rid), ptrs) in &self.history_index {
            if gid != group_id {
                continue;
            }
            for ptr in ptrs {
                if let Some(data) = self.store.read(*ptr) {
                    results.push(row_codec::decode_row(
                        &self.layout,
                        data,
                        &self.schema.columns,
                    ));
                }
            }
        }
        results
    }

    /// Temporal range for a group.
    /// Returns all row versions that overlap with [start, end]:
    /// valid_from <= end AND (valid_until IS NULL OR valid_until >= start)
    pub fn scan_range(
        &self,
        group_id: GroupId,
        start: Timestamp,
        end: Timestamp,
    ) -> Vec<DecodedRow> {
        let mut results = Vec::new();
        // Must check ALL versions, not just those with valid_from in range,
        // because a row with valid_from < start may still be active during the range.
        for (&(gid, _rid), ptrs) in &self.history_index {
            if gid != group_id {
                continue;
            }
            for ptr in ptrs {
                if let Some(data) = self.store.read(*ptr) {
                    let hdr = row_codec::read_header(data);
                    // Overlap check: valid_from <= end AND (valid_until is None OR valid_until >= start)
                    if hdr.valid_from <= end && hdr.valid_until.is_none_or(|vu| vu >= start) {
                        results.push(row_codec::decode_row(
                            &self.layout,
                            data,
                            &self.schema.columns,
                        ));
                    }
                }
            }
        }
        results
    }

    /// Rows referencing a graph node (within a group).
    pub fn rows_by_node(&self, group_id: GroupId, node_id: u64) -> Vec<DecodedRow> {
        let mut results = Vec::new();
        if let Some(row_keys) = self.node_ref_index.get(&node_id) {
            for &(gid, rid) in row_keys {
                if gid != group_id {
                    continue;
                }
                if let Some(decoded) = self.get_active(group_id, rid) {
                    results.push(decoded);
                }
            }
        }
        results
    }

    /// Single row by RowId (active version).
    pub fn get_active(&self, group_id: GroupId, row_id: RowId) -> Option<DecodedRow> {
        let key = (group_id, row_id);
        let ptr = (*self.pk_index.get(&key)?)?;
        let data = self.store.read(ptr)?;
        Some(row_codec::decode_row(
            &self.layout,
            data,
            &self.schema.columns,
        ))
    }

    /// Read a single column from a row without decoding the full row.
    pub fn read_column(&self, ptr: RowPointer, col_idx: usize) -> Option<CellValue> {
        if col_idx >= self.schema.columns.len() {
            return None;
        }
        let bytes = self.store.read(ptr)?;
        Some(row_codec::read_column(
            &self.layout,
            bytes,
            col_idx,
            &self.schema.columns[col_idx].col_type,
        ))
    }

    /// Handle node merge: update all NodeRef columns pointing to absorbed -> survivor.
    pub fn on_node_merged(
        &mut self,
        survivor: u64,
        absorbed: u64,
    ) -> Vec<(RowVersionId, RowVersionId)> {
        let affected: Vec<(GroupId, RowId)> = self
            .node_ref_index
            .get(&absorbed)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let mut version_pairs = Vec::new();
        for (gid, rid) in affected {
            if let Some(decoded) = self.get_active(gid, rid) {
                let mut new_values = decoded.values;
                let mut changed = false;
                for &col_idx in &self.node_ref_col_indices {
                    if let CellValue::NodeRef(nid) = &new_values[col_idx] {
                        if *nid == absorbed {
                            new_values[col_idx] = CellValue::NodeRef(survivor);
                            changed = true;
                        }
                    }
                }
                if changed {
                    if let Ok(pair) = self.update(gid, rid, new_values) {
                        version_pairs.push(pair);
                    }
                }
            }
        }
        version_pairs
    }

    // -- Stats --

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn active_row_count(&self) -> usize {
        self.active_row_count
    }

    pub fn total_version_count(&self) -> usize {
        self.store.live_row_count()
    }

    pub fn page_count(&self) -> usize {
        self.store.page_count()
    }

    pub fn next_version_id(&self) -> RowVersionId {
        self.next_version_id
    }

    pub fn next_row_id(&self) -> RowId {
        self.next_row_id
    }

    pub fn store(&self) -> &PageStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut PageStore {
        &mut self.store
    }

    /// Remove stale entries from history_index, version_index, and temporal_index
    /// that point to dead (compacted) rows. Call after compaction.
    pub fn clean_stale_index_entries(&mut self) {
        // Clean history_index: remove pointers to dead rows
        for ptrs in self.history_index.values_mut() {
            ptrs.retain(|ptr| self.store.read(*ptr).is_some());
        }
        // Remove empty entries
        self.history_index.retain(|_, ptrs| !ptrs.is_empty());

        // Clean version_index
        self.version_index
            .retain(|_, ptr| self.store.read(*ptr).is_some());

        // Clean temporal_index
        for ptrs in self.temporal_index.values_mut() {
            ptrs.retain(|ptr| self.store.read(*ptr).is_some());
        }
        self.temporal_index.retain(|_, ptrs| !ptrs.is_empty());
    }

    // -- Internal helpers --

    fn active_ptr(&self, key: (GroupId, RowId)) -> Result<RowPointer, TableError> {
        match self.pk_index.get(&key) {
            Some(Some(ptr)) => Ok(*ptr),
            Some(None) => Err(TableError::RowAlreadyDeleted(key.1)),
            None => Err(TableError::RowNotFound(key.1)),
        }
    }

    fn validate_values(&self, values: &[CellValue]) -> Result<(), TableError> {
        if values.len() != self.schema.columns.len() {
            return Err(TableError::ColumnCountMismatch {
                expected: self.schema.columns.len(),
                got: values.len(),
            });
        }
        for (i, (val, col)) in values.iter().zip(&self.schema.columns).enumerate() {
            if !val.matches_type(&col.col_type) {
                return Err(TableError::TypeMismatch {
                    column: col.name.clone(),
                    expected: col.col_type.clone(),
                    got: val.type_name().to_string(),
                });
            }
            if matches!(val, CellValue::Null) && !col.nullable {
                return Err(TableError::NullConstraintViolation(col.name.clone()));
            }
            let _ = i; // suppress unused
        }
        Ok(())
    }

    fn check_unique_constraints(
        &self,
        group_id: GroupId,
        values: &[CellValue],
        exclude_row_id: Option<RowId>,
    ) -> Result<(), TableError> {
        for idx in &self.unique_indexes {
            // SQL semantics: NULLs are not equal, so skip uniqueness check
            // if any column in the key is NULL.
            let has_null = idx
                .columns
                .iter()
                .any(|&i| matches!(values[i], CellValue::Null));
            if has_null {
                continue;
            }
            let key = self.build_index_key(&idx.columns, values);
            if let Some(&existing_rid) = idx.entries.get(&(group_id, key)) {
                if exclude_row_id != Some(existing_rid) {
                    let col_names: Vec<String> = idx
                        .columns
                        .iter()
                        .map(|&i| self.schema.columns[i].name.clone())
                        .collect();
                    return Err(TableError::UniqueConstraintViolation(col_names));
                }
            }
        }
        Ok(())
    }

    fn build_index_key(&self, columns: &[usize], values: &[CellValue]) -> IndexKey {
        if columns.len() == 1 {
            IndexKey::from_cell(&values[columns[0]])
        } else {
            IndexKey::Composite(
                columns
                    .iter()
                    .map(|&i| IndexKey::from_cell(&values[i]))
                    .collect(),
            )
        }
    }

    fn insert_unique_entries(&mut self, group_id: GroupId, row_id: RowId, values: &[CellValue]) {
        for idx in &mut self.unique_indexes {
            let key = if idx.columns.len() == 1 {
                IndexKey::from_cell(&values[idx.columns[0]])
            } else {
                IndexKey::Composite(
                    idx.columns
                        .iter()
                        .map(|&i| IndexKey::from_cell(&values[i]))
                        .collect(),
                )
            };
            idx.entries.insert((group_id, key), row_id);
        }
    }

    fn remove_unique_entries(&mut self, group_id: GroupId, values: &[CellValue]) {
        for idx in &mut self.unique_indexes {
            let key = if idx.columns.len() == 1 {
                IndexKey::from_cell(&values[idx.columns[0]])
            } else {
                IndexKey::Composite(
                    idx.columns
                        .iter()
                        .map(|&i| IndexKey::from_cell(&values[i]))
                        .collect(),
                )
            };
            idx.entries.remove(&(group_id, key));
        }
    }

    fn insert_node_ref_entries(&mut self, group_id: GroupId, row_id: RowId, values: &[CellValue]) {
        for &col_idx in &self.node_ref_col_indices {
            if let Some(node_id) = values[col_idx].as_node_ref() {
                self.node_ref_index
                    .entry(node_id)
                    .or_default()
                    .insert((group_id, row_id));
            }
        }
    }

    fn remove_node_ref_entries(&mut self, group_id: GroupId, row_id: RowId, values: &[CellValue]) {
        for &col_idx in &self.node_ref_col_indices {
            if let Some(node_id) = values[col_idx].as_node_ref() {
                if let Some(set) = self.node_ref_index.get_mut(&node_id) {
                    set.remove(&(group_id, row_id));
                    if set.is_empty() {
                        self.node_ref_index.remove(&node_id);
                    }
                }
            }
        }
    }

    fn decode_values(&self, row_bytes: &[u8]) -> Vec<CellValue> {
        let mut values = Vec::with_capacity(self.schema.columns.len());
        for (i, col) in self.schema.columns.iter().enumerate() {
            values.push(row_codec::read_column(
                &self.layout,
                row_bytes,
                i,
                &col.col_type,
            ));
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColumnDef, Constraint};

    fn col(name: &str, col_type: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type,
            nullable: true,
            default_value: None,
        }
    }

    fn col_nn(name: &str, col_type: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type,
            nullable: false,
            default_value: None,
        }
    }

    fn test_schema() -> TableSchema {
        TableSchema {
            table_id: 1,
            name: "orders".into(),
            columns: vec![
                col_nn("id", ColumnType::Int64),
                col("customer", ColumnType::String),
                col("amount", ColumnType::Float64),
                col("node", ColumnType::NodeRef),
            ],
            constraints: vec![Constraint::PrimaryKey(vec!["id".into()])],
            created_at: 0,
            schema_version: 1,
        }
    }

    #[test]
    fn test_insert_and_get() {
        let mut table = Table::new(test_schema()).unwrap();
        let (rid, vid) = table
            .insert(
                0,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Alice".into()),
                    CellValue::Float64(99.99),
                    CellValue::NodeRef(100),
                ],
            )
            .unwrap();

        assert_eq!(rid, 1);
        assert_eq!(vid, 1);
        assert_eq!(table.active_row_count(), 1);
        assert_eq!(table.generation(), 1);

        let row = table.get_active(0, rid).unwrap();
        assert_eq!(row.row_id, 1);
        match &row.values[1] {
            CellValue::String(s) => assert_eq!(s, "Alice"),
            other => panic!("expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_update() {
        let mut table = Table::new(test_schema()).unwrap();
        let (rid, _) = table
            .insert(
                0,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Alice".into()),
                    CellValue::Float64(50.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        let (old_vid, new_vid) = table
            .update(
                0,
                rid,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Alice Updated".into()),
                    CellValue::Float64(75.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        assert_eq!(old_vid, 1);
        assert_eq!(new_vid, 2);
        assert_eq!(table.active_row_count(), 1);
        assert_eq!(table.generation(), 2);

        let row = table.get_active(0, rid).unwrap();
        match &row.values[1] {
            CellValue::String(s) => assert_eq!(s, "Alice Updated"),
            other => panic!("expected String, got {:?}", other),
        }

        // History should have 2 versions
        let all = table.scan_all(0);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_delete() {
        let mut table = Table::new(test_schema()).unwrap();
        let (rid, _) = table
            .insert(
                0,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Bob".into()),
                    CellValue::Float64(10.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        let vid = table.delete(0, rid).unwrap();
        assert_eq!(vid, 1);
        assert_eq!(table.active_row_count(), 0);
        assert!(table.get_active(0, rid).is_none());

        // Historical versions still queryable
        let all = table.scan_all(0);
        assert_eq!(all.len(), 1);
        assert!(all[0].valid_until.is_some());
    }

    #[test]
    fn test_unique_constraint() {
        let mut table = Table::new(test_schema()).unwrap();
        table
            .insert(
                0,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Alice".into()),
                    CellValue::Float64(50.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        // Same PK should fail
        let result = table.insert(
            0,
            vec![
                CellValue::Int64(1),
                CellValue::String("Bob".into()),
                CellValue::Float64(60.0),
                CellValue::Null,
            ],
        );
        assert!(matches!(
            result,
            Err(TableError::UniqueConstraintViolation(_))
        ));
    }

    #[test]
    fn test_unique_constraint_different_groups() {
        let mut table = Table::new(test_schema()).unwrap();
        // Same PK value in different groups should be fine
        table
            .insert(
                1,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Alice".into()),
                    CellValue::Float64(50.0),
                    CellValue::Null,
                ],
            )
            .unwrap();
        table
            .insert(
                2,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Bob".into()),
                    CellValue::Float64(60.0),
                    CellValue::Null,
                ],
            )
            .unwrap();
        assert_eq!(table.active_row_count(), 2);
    }

    #[test]
    fn test_node_ref_index() {
        let mut table = Table::new(test_schema()).unwrap();
        table
            .insert(
                0,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Order 1".into()),
                    CellValue::Float64(100.0),
                    CellValue::NodeRef(42),
                ],
            )
            .unwrap();
        table
            .insert(
                0,
                vec![
                    CellValue::Int64(2),
                    CellValue::String("Order 2".into()),
                    CellValue::Float64(200.0),
                    CellValue::NodeRef(42),
                ],
            )
            .unwrap();

        let by_node = table.rows_by_node(0, 42);
        assert_eq!(by_node.len(), 2);

        let by_node_other = table.rows_by_node(0, 99);
        assert_eq!(by_node_other.len(), 0);
    }

    #[test]
    fn test_scan_active() {
        let mut table = Table::new(test_schema()).unwrap();
        for i in 1..=5 {
            table
                .insert(
                    0,
                    vec![
                        CellValue::Int64(i),
                        CellValue::String(format!("row{}", i)),
                        CellValue::Float64(i as f64),
                        CellValue::Null,
                    ],
                )
                .unwrap();
        }

        let active = table.scan_active(0);
        assert_eq!(active.len(), 5);
    }

    #[test]
    fn test_type_mismatch() {
        let mut table = Table::new(test_schema()).unwrap();
        let result = table.insert(
            0,
            vec![
                CellValue::String("not an int".into()),
                CellValue::String("Alice".into()),
                CellValue::Float64(50.0),
                CellValue::Null,
            ],
        );
        assert!(matches!(result, Err(TableError::TypeMismatch { .. })));
    }

    #[test]
    fn test_column_count_mismatch() {
        let mut table = Table::new(test_schema()).unwrap();
        let result = table.insert(0, vec![CellValue::Int64(1)]);
        assert!(matches!(
            result,
            Err(TableError::ColumnCountMismatch { .. })
        ));
    }

    #[test]
    fn test_null_constraint() {
        let mut table = Table::new(test_schema()).unwrap();
        let result = table.insert(
            0,
            vec![
                CellValue::Null, // id is NOT NULL
                CellValue::String("Alice".into()),
                CellValue::Float64(50.0),
                CellValue::Null,
            ],
        );
        assert!(matches!(
            result,
            Err(TableError::NullConstraintViolation(_))
        ));
    }

    #[test]
    fn test_batch_insert() {
        let mut table = Table::new(test_schema()).unwrap();
        let rows = (1..=10)
            .map(|i| {
                vec![
                    CellValue::Int64(i),
                    CellValue::String(format!("item{}", i)),
                    CellValue::Float64(i as f64 * 10.0),
                    CellValue::Null,
                ]
            })
            .collect();

        let results = table.insert_batch(0, rows).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(table.active_row_count(), 10);
    }

    #[test]
    fn test_group_isolation() {
        let mut table = Table::new(test_schema()).unwrap();
        table
            .insert(
                1,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("Group1".into()),
                    CellValue::Float64(10.0),
                    CellValue::Null,
                ],
            )
            .unwrap();
        table
            .insert(
                2,
                vec![
                    CellValue::Int64(2),
                    CellValue::String("Group2".into()),
                    CellValue::Float64(20.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        assert_eq!(table.scan_active(1).len(), 1);
        assert_eq!(table.scan_active(2).len(), 1);
        assert!(table.get_active(1, 2).is_none()); // row 2 belongs to group 2
    }

    #[test]
    fn test_temporal_as_of() {
        let mut table = Table::new(test_schema()).unwrap();

        let (rid, _) = table
            .insert(
                0,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("v1".into()),
                    CellValue::Float64(1.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        let after_insert = current_timestamp();
        // Small delay to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(1));

        table
            .update(
                0,
                rid,
                vec![
                    CellValue::Int64(1),
                    CellValue::String("v2".into()),
                    CellValue::Float64(2.0),
                    CellValue::Null,
                ],
            )
            .unwrap();

        // Current should be v2
        let current = table.get_active(0, rid).unwrap();
        match &current.values[1] {
            CellValue::String(s) => assert_eq!(s, "v2"),
            other => panic!("expected String, got {:?}", other),
        }

        // AS OF after_insert should be v1
        let historical = table.scan_as_of(0, after_insert);
        assert_eq!(historical.len(), 1);
        match &historical[0].values[1] {
            CellValue::String(s) => assert_eq!(s, "v1"),
            other => panic!("expected String, got {:?}", other),
        }
    }
}
