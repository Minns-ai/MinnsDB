//! Full-rerun output diffing using structural RowId identity.
//!
//! When a subscription uses FullRerun strategy (or is forced to rerun),
//! this module compares old cached output against new execution results
//! using the same structural RowId model as incremental mode.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::query_lang::planner::SlotIdx;
use crate::query_lang::types::{QueryOutput, Value};

use super::incremental::{BoundEntityId, RowId};

/// Binding info extracted from executor for a single result row.
/// Used to construct structural RowIds from full-rerun output.
pub type BindingRow = SmallVec<[(SlotIdx, BoundEntityId); 4]>;

/// Cached subscription output keyed by structural RowId.
#[derive(Debug, Clone, Default)]
pub struct CachedOutput {
    pub columns: Vec<String>,
    pub rows: FxHashMap<RowId, Vec<Value>>,
}

/// Result of diffing old vs new output.
pub struct DiffResult {
    pub inserts: Vec<(RowId, Vec<Value>)>,
    pub deletes: Vec<RowId>,
}

/// Diff old cached output against new execution results.
///
/// Both old and new use the same structural RowId model:
/// RowId is built from the executor's binding rows (entity IDs per slot),
/// ensuring consistent identity between incremental and full-rerun modes.
pub fn diff_outputs(
    old: &CachedOutput,
    new_rows: &FxHashMap<RowId, Vec<Value>>,
) -> DiffResult {
    let mut inserts = Vec::new();
    let mut deletes = Vec::new();

    // Keys in old not in new → Delete
    for old_key in old.rows.keys() {
        if !new_rows.contains_key(old_key) {
            deletes.push(old_key.clone());
        }
    }

    // Keys in new not in old → Insert
    // Keys in both with different values → Delete + Insert
    for (new_key, new_vals) in new_rows {
        match old.rows.get(new_key) {
            None => {
                inserts.push((new_key.clone(), new_vals.clone()));
            }
            Some(old_vals) => {
                if old_vals != new_vals {
                    deletes.push(new_key.clone());
                    inserts.push((new_key.clone(), new_vals.clone()));
                }
            }
        }
    }

    DiffResult { inserts, deletes }
}

/// Build a CachedOutput from query results and their binding rows.
pub fn build_cached_output(
    output: &QueryOutput,
    binding_rows: &[BindingRow],
) -> CachedOutput {
    let mut rows = FxHashMap::default();

    for (i, values) in output.rows.iter().enumerate() {
        if let Some(bindings) = binding_rows.get(i) {
            let row_id = RowId::new(bindings.clone());
            rows.insert(row_id, values.clone());
        }
    }

    CachedOutput {
        columns: output.columns.clone(),
        rows,
    }
}
