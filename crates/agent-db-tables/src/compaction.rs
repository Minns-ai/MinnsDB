//! Page compaction: reclaim space from dead rows (old closed versions past retention).

use crate::row_codec;
use crate::table::Table;
use crate::types::Timestamp;

pub struct CompactionConfig {
    /// Retention window in nanoseconds. Default: 7 days.
    pub retention_nanos: u64,
    /// Minimum versions to retain per row. Default: 1.
    pub min_versions_per_row: usize,
    /// Compact page when dead space exceeds this fraction. Default: 0.3.
    pub dead_space_threshold: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        CompactionConfig {
            retention_nanos: 7 * 24 * 60 * 60 * 1_000_000_000, // 7 days
            min_versions_per_row: 1,
            dead_space_threshold: 0.3,
        }
    }
}

pub struct CompactionResult {
    pub versions_removed: usize,
    pub pages_compacted: usize,
    pub bytes_reclaimed: usize,
}

/// Compact a table: identify expired versions, mark dead, compact pages.
pub fn compact_table(table: &mut Table, now: Timestamp) -> CompactionResult {
    compact_table_with_config(table, now, &CompactionConfig::default())
}

/// Compact with custom config.
pub fn compact_table_with_config(
    table: &mut Table,
    now: Timestamp,
    config: &CompactionConfig,
) -> CompactionResult {
    let cutoff = now.saturating_sub(config.retention_nanos);
    let mut versions_removed = 0usize;

    // Phase 1: identify expired versions and mark dead
    // Collect version pointers to check
    let all_entries: Vec<(crate::types::RowPointer, Vec<u8>)> = table
        .store()
        .iter_live()
        .map(|(ptr, data)| (ptr, data.to_vec()))
        .collect();

    // Group by (group_id, row_id) to enforce min_versions_per_row
    let mut by_row: rustc_hash::FxHashMap<
        (u64, u64),
        Vec<(crate::types::RowPointer, row_codec::RowHeader)>,
    > = rustc_hash::FxHashMap::default();

    for (ptr, data) in &all_entries {
        let hdr = row_codec::read_header(data);
        by_row
            .entry((hdr.group_id, hdr.row_id))
            .or_default()
            .push((*ptr, hdr));
    }

    let mut to_mark_dead = Vec::new();

    for ((_gid, _rid), mut versions) in by_row {
        // Sort by version_id ascending
        versions.sort_by_key(|(_, hdr)| hdr.version_id);

        let total = versions.len();
        let mut remaining = total;

        for (ptr, hdr) in &versions {
            // Only consider closed versions
            if let Some(valid_until) = hdr.valid_until {
                if valid_until <= cutoff && remaining > config.min_versions_per_row {
                    to_mark_dead.push(*ptr);
                    remaining -= 1;
                }
            }
        }
    }

    for ptr in &to_mark_dead {
        table.store_mut().mark_dead(*ptr);
        versions_removed += 1;
    }

    // Phase 2: compact pages with high dead space
    let mut pages_compacted = 0usize;
    let page_count = table.store().page_count();
    for page_id in 0..page_count as u32 {
        if let Some(page) = table.store().get_page(page_id) {
            let dead = page.dead_space();
            let total = crate::types::PAGE_SIZE - crate::types::PAGE_HEADER_SIZE;
            let ratio = dead as f64 / total as f64;
            if ratio > config.dead_space_threshold {
                table.store_mut().compact_page(page_id);
                pages_compacted += 1;
            }
        }
    }

    // Phase 3: clean stale index entries pointing to dead rows
    if versions_removed > 0 {
        table.clean_stale_index_entries();
    }

    CompactionResult {
        versions_removed,
        pages_compacted,
        bytes_reclaimed: 0, // approximate; exact tracking would need pre/post free space diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColumnDef, Constraint, TableSchema};
    use crate::types::{CellValue, ColumnType};

    fn col(name: &str, ct: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.into(),
            col_type: ct,
            nullable: true,
            default_value: None,
            autoincrement: false,
        }
    }

    fn test_table() -> Table {
        let schema = TableSchema {
            table_id: 1,
            name: "test".into(),
            columns: vec![
                ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,
                    default_value: None,
                    autoincrement: false,
                },
                col("val", ColumnType::String),
            ],
            constraints: vec![Constraint::PrimaryKey(vec!["id".into()])],
            created_at: 0,
            schema_version: 1,
        };
        Table::new(schema).unwrap()
    }

    #[test]
    fn test_compact_removes_old_versions() {
        let mut table = test_table();

        // Insert and update a row several times
        let (rid, _) = table
            .insert(0, vec![CellValue::Int64(1), CellValue::String("v1".into())])
            .unwrap();
        table
            .update(
                0,
                rid,
                vec![CellValue::Int64(1), CellValue::String("v2".into())],
            )
            .unwrap();
        table
            .update(
                0,
                rid,
                vec![CellValue::Int64(1), CellValue::String("v3".into())],
            )
            .unwrap();

        assert_eq!(table.total_version_count(), 3);

        // Compact with zero retention (everything expired)
        let config = CompactionConfig {
            retention_nanos: 0,
            min_versions_per_row: 1,
            dead_space_threshold: 0.0,
        };
        let far_future = u64::MAX;
        let result = compact_table_with_config(&mut table, far_future, &config);

        // Should remove 2 old versions, keep 1 (min_versions + active)
        assert_eq!(result.versions_removed, 2);
    }

    #[test]
    fn test_compact_respects_retention() {
        let mut table = test_table();

        let (rid, _) = table
            .insert(0, vec![CellValue::Int64(1), CellValue::String("v1".into())])
            .unwrap();
        table
            .update(
                0,
                rid,
                vec![CellValue::Int64(1), CellValue::String("v2".into())],
            )
            .unwrap();

        // Compact with very long retention
        let config = CompactionConfig {
            retention_nanos: u64::MAX / 2,
            min_versions_per_row: 1,
            dead_space_threshold: 0.0,
        };
        let now = agent_db_core::types::current_timestamp();
        let result = compact_table_with_config(&mut table, now, &config);

        // Nothing should be removed (retention not expired)
        assert_eq!(result.versions_removed, 0);
    }
}
