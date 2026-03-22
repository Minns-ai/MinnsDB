//! Table statistics for query optimizer.

use crate::types::GroupId;
use rustc_hash::FxHashMap;

/// Per-table statistics maintained incrementally.
#[derive(Debug, Clone, Default)]
pub struct TableStats {
    /// Active row count per group.
    pub per_group_counts: FxHashMap<GroupId, usize>,
    /// Number of distinct keys per index (by index ordinal in unique_indexes vec).
    pub index_cardinalities: Vec<usize>,
}

impl TableStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_insert(&mut self, group_id: GroupId, num_indexes: usize) {
        *self.per_group_counts.entry(group_id).or_insert(0) += 1;
        // Grow cardinalities vec if needed (new index added after stats init)
        while self.index_cardinalities.len() < num_indexes {
            self.index_cardinalities.push(0);
        }
    }

    pub fn on_delete(&mut self, group_id: GroupId) {
        if let Some(c) = self.per_group_counts.get_mut(&group_id) {
            *c = c.saturating_sub(1);
        }
    }

    pub fn active_count_for_group(&self, group_id: GroupId) -> usize {
        self.per_group_counts.get(&group_id).copied().unwrap_or(0)
    }

    pub fn total_active_count(&self) -> usize {
        self.per_group_counts.values().sum()
    }

    pub fn set_index_cardinality(&mut self, index_ordinal: usize, cardinality: usize) {
        if index_ordinal >= self.index_cardinalities.len() {
            self.index_cardinalities.resize(index_ordinal + 1, 0);
        }
        self.index_cardinalities[index_ordinal] = cardinality;
    }
}

/// Aggregated statistics across all tables in the catalog.
#[derive(Debug, Clone, Default)]
pub struct CatalogStats {
    /// Table name -> TableStats
    pub table_stats: FxHashMap<String, TableStats>,
}

impl CatalogStats {
    pub fn get(&self, table_name: &str) -> Option<&TableStats> {
        self.table_stats.get(table_name)
    }

    pub fn estimated_row_count(&self, table_name: &str, group_id: GroupId) -> usize {
        self.table_stats
            .get(table_name)
            .map(|s| s.active_count_for_group(group_id))
            .unwrap_or(0)
    }
}
