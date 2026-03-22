//! Rule-based query optimizer.
//!
//! Transforms an `ExecutionPlan` in-place between planning and execution.
//! Each rule is a separate, testable function applied in a fixed order.

use super::ast::{CompOp, Literal};
use super::planner::{ExecutionPlan, JoinCondition, PlanStep, RBoolExpr, RExpr};

/// Statistics interface the optimizer queries for catalog-dependent rules.
pub trait OptimizerStats {
    /// Estimated row count for a table (for a specific group).
    fn estimated_row_count(&self, table_name: &str) -> usize;
    /// Whether an index exists on the given columns for a table.
    fn has_index(&self, table_name: &str, columns: &[String]) -> bool;
    /// Cardinality estimate for an index.
    fn index_cardinality(&self, table_name: &str, columns: &[String]) -> usize;
}

/// No-op stats implementation for when no catalog is available.
pub struct NoStats;

impl OptimizerStats for NoStats {
    fn estimated_row_count(&self, _: &str) -> usize {
        0
    }
    fn has_index(&self, _: &str, _: &[String]) -> bool {
        false
    }
    fn index_cardinality(&self, _: &str, _: &[String]) -> usize {
        0
    }
}

/// Trace of optimizations applied (for debugging/testing).
#[derive(Debug, Clone, Default)]
pub struct OptimizeTrace {
    pub rules_applied: Vec<String>,
}

/// Run the full optimization pipeline on a plan.
pub fn optimize(plan: &mut ExecutionPlan, stats: &dyn OptimizerStats) -> OptimizeTrace {
    let mut trace = OptimizeTrace::default();

    if constant_fold(plan) {
        trace.rules_applied.push("constant_fold".into());
    }
    if eliminate_redundant_filters(plan) {
        trace
            .rules_applied
            .push("eliminate_redundant_filters".into());
    }
    if predicate_pushdown(plan) {
        trace.rules_applied.push("predicate_pushdown".into());
    }
    if scan_to_index_rewrite(plan, stats) {
        trace.rules_applied.push("scan_to_index_rewrite".into());
    }
    if join_reorder(plan, stats) {
        trace.rules_applied.push("join_reorder".into());
    }
    if hash_to_index_join(plan, stats) {
        trace.rules_applied.push("hash_to_index_join".into());
    }
    if limit_pushdown(plan) {
        trace.rules_applied.push("limit_pushdown".into());
    }

    trace
}

// ---------------------------------------------------------------------------
// Rule 1: Constant Folding
// ---------------------------------------------------------------------------

/// Evaluate literal-vs-literal comparisons at plan time.
/// Returns true if any changes were made.
pub fn constant_fold(plan: &mut ExecutionPlan) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < plan.steps.len() {
        if let PlanStep::Filter(ref expr) = plan.steps[i] {
            match try_fold_bool(expr) {
                FoldResult::AlwaysTrue => {
                    plan.steps.remove(i);
                    changed = true;
                    continue; // don't increment i
                },
                FoldResult::AlwaysFalse => {
                    // Replace with a filter that's always false — executor will produce no rows.
                    // We keep it as a single false filter.
                    plan.steps[i] = PlanStep::Filter(RBoolExpr::Comparison(
                        RExpr::Literal(Literal::Bool(false)),
                        CompOp::Eq,
                        RExpr::Literal(Literal::Bool(true)),
                    ));
                    changed = true;
                },
                FoldResult::Simplified(new_expr) => {
                    plan.steps[i] = PlanStep::Filter(new_expr);
                    changed = true;
                },
                FoldResult::Unchanged => {},
            }
        }
        i += 1;
    }
    changed
}

enum FoldResult {
    AlwaysTrue,
    AlwaysFalse,
    Simplified(RBoolExpr),
    Unchanged,
}

fn try_fold_bool(expr: &RBoolExpr) -> FoldResult {
    match expr {
        RBoolExpr::Comparison(RExpr::Literal(l), op, RExpr::Literal(r)) => {
            if eval_literal_cmp(l, op, r) {
                FoldResult::AlwaysTrue
            } else {
                FoldResult::AlwaysFalse
            }
        },
        RBoolExpr::And(a, b) => {
            let fa = try_fold_bool(a);
            let fb = try_fold_bool(b);
            match (&fa, &fb) {
                (FoldResult::AlwaysFalse, _) | (_, FoldResult::AlwaysFalse) => {
                    FoldResult::AlwaysFalse
                },
                (FoldResult::AlwaysTrue, FoldResult::AlwaysTrue) => FoldResult::AlwaysTrue,
                (FoldResult::AlwaysTrue, _) => match fb {
                    FoldResult::Simplified(e) => FoldResult::Simplified(e),
                    FoldResult::Unchanged => FoldResult::Simplified((**b).clone()),
                    _ => FoldResult::Unchanged,
                },
                (_, FoldResult::AlwaysTrue) => match fa {
                    FoldResult::Simplified(e) => FoldResult::Simplified(e),
                    FoldResult::Unchanged => FoldResult::Simplified((**a).clone()),
                    _ => FoldResult::Unchanged,
                },
                _ => FoldResult::Unchanged,
            }
        },
        RBoolExpr::Or(a, b) => {
            let fa = try_fold_bool(a);
            let fb = try_fold_bool(b);
            match (&fa, &fb) {
                (FoldResult::AlwaysTrue, _) | (_, FoldResult::AlwaysTrue) => FoldResult::AlwaysTrue,
                (FoldResult::AlwaysFalse, FoldResult::AlwaysFalse) => FoldResult::AlwaysFalse,
                (FoldResult::AlwaysFalse, _) => match fb {
                    FoldResult::Simplified(e) => FoldResult::Simplified(e),
                    FoldResult::Unchanged => FoldResult::Simplified((**b).clone()),
                    _ => FoldResult::Unchanged,
                },
                (_, FoldResult::AlwaysFalse) => match fa {
                    FoldResult::Simplified(e) => FoldResult::Simplified(e),
                    FoldResult::Unchanged => FoldResult::Simplified((**a).clone()),
                    _ => FoldResult::Unchanged,
                },
                _ => FoldResult::Unchanged,
            }
        },
        RBoolExpr::Not(inner) => match try_fold_bool(inner) {
            FoldResult::AlwaysTrue => FoldResult::AlwaysFalse,
            FoldResult::AlwaysFalse => FoldResult::AlwaysTrue,
            _ => FoldResult::Unchanged,
        },
        RBoolExpr::Paren(inner) => try_fold_bool(inner),
        _ => FoldResult::Unchanged,
    }
}

fn eval_literal_cmp(l: &Literal, op: &CompOp, r: &Literal) -> bool {
    match (l, r) {
        (Literal::Int(a), Literal::Int(b)) => match op {
            CompOp::Eq => a == b,
            CompOp::Neq => a != b,
            CompOp::Lt => a < b,
            CompOp::Gt => a > b,
            CompOp::Lte => a <= b,
            CompOp::Gte => a >= b,
            _ => false,
        },
        (Literal::Float(a), Literal::Float(b)) => match op {
            CompOp::Eq => a.to_bits() == b.to_bits(),
            CompOp::Neq => a.to_bits() != b.to_bits(),
            CompOp::Lt => a < b,
            CompOp::Gt => a > b,
            CompOp::Lte => a <= b,
            CompOp::Gte => a >= b,
            _ => false,
        },
        (Literal::String(a), Literal::String(b)) => match op {
            CompOp::Eq => a == b,
            CompOp::Neq => a != b,
            CompOp::Lt => a < b,
            CompOp::Gt => a > b,
            CompOp::Lte => a <= b,
            CompOp::Gte => a >= b,
            CompOp::Contains => a.contains(b.as_str()),
            CompOp::StartsWith => a.starts_with(b.as_str()),
        },
        (Literal::Bool(a), Literal::Bool(b)) => match op {
            CompOp::Eq => a == b,
            CompOp::Neq => a != b,
            _ => false,
        },
        // NULL comparisons are unknown in SQL semantics — don't fold them.
        (Literal::Null, _) | (_, Literal::Null) => false,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Rule 2: Eliminate Redundant Filters
// ---------------------------------------------------------------------------

/// Deduplicate AND conjuncts and remove tautologies (1=1).
pub fn eliminate_redundant_filters(plan: &mut ExecutionPlan) -> bool {
    let mut changed = false;

    // Collect all Filter step indices
    let filter_indices: Vec<usize> = plan
        .steps
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s, PlanStep::Filter(_)))
        .map(|(i, _)| i)
        .collect();

    if filter_indices.len() <= 1 {
        return false;
    }

    // Collect filter expressions (by index) and deduplicate
    let mut seen: std::collections::HashSet<RBoolExpr> = std::collections::HashSet::new();
    let mut to_remove: Vec<usize> = Vec::new();

    for &idx in &filter_indices {
        if let PlanStep::Filter(ref expr) = plan.steps[idx] {
            if !seen.insert(expr.clone()) {
                to_remove.push(idx);
                changed = true;
            }
        }
    }

    // Remove duplicates in reverse order to preserve indices
    for idx in to_remove.into_iter().rev() {
        plan.steps.remove(idx);
    }

    changed
}

// ---------------------------------------------------------------------------
// Rule 3: Predicate Pushdown
// ---------------------------------------------------------------------------

/// Move single-table filters to just after the corresponding scan step.
/// Uses slot indices to identify which scan a filter belongs to — each
/// ScanTable/IndexScan binds a specific slot, and Property(slot, col)
/// references that slot.
pub fn predicate_pushdown(plan: &mut ExecutionPlan) -> bool {
    let mut changed = false;

    // Build a map: slot index → position of the scan step that binds it.
    // ScanTable/IndexScan steps bind slot 0 for the first table, etc.
    let mut slot_to_scan_pos: std::collections::HashMap<u8, usize> =
        std::collections::HashMap::new();
    let mut next_slot: u8 = 0;
    for (pos, step) in plan.steps.iter().enumerate() {
        match step {
            PlanStep::ScanTable { .. } | PlanStep::IndexScan { .. } => {
                slot_to_scan_pos.insert(next_slot, pos);
                next_slot = next_slot.saturating_add(1);
            },
            PlanStep::JoinTable { .. } | PlanStep::IndexJoin { .. } => {
                slot_to_scan_pos.insert(next_slot, pos);
                next_slot = next_slot.saturating_add(1);
            },
            _ => {},
        }
    }

    let mut i = 0;
    while i < plan.steps.len() {
        if let PlanStep::Filter(ref expr) = plan.steps[i] {
            if let Some(slot) = single_slot_filter(expr) {
                if let Some(&scan_pos) = slot_to_scan_pos.get(&slot) {
                    let target = scan_pos + 1;
                    if i > target {
                        let filter = plan.steps.remove(i);
                        plan.steps.insert(target, filter);
                        changed = true;
                        // Rebuild slot_to_scan_pos since positions shifted
                        slot_to_scan_pos.values_mut().for_each(|p| {
                            if *p >= target && *p < i {
                                *p += 1;
                            }
                        });
                        continue;
                    }
                }
            }
        }
        i += 1;
    }

    changed
}

/// Check if a filter expression references only a single slot.
/// Returns the slot index if so.
fn single_slot_filter(expr: &RBoolExpr) -> Option<u8> {
    let mut slots = std::collections::HashSet::new();
    collect_slots_bool(expr, &mut slots);
    if slots.len() == 1 {
        slots.into_iter().next()
    } else {
        None
    }
}

fn collect_slots_bool(expr: &RBoolExpr, slots: &mut std::collections::HashSet<u8>) {
    match expr {
        RBoolExpr::Comparison(l, _, r) => {
            collect_slots_expr(l, slots);
            collect_slots_expr(r, slots);
        },
        RBoolExpr::IsNull(e) | RBoolExpr::IsNotNull(e) => {
            collect_slots_expr(e, slots);
        },
        RBoolExpr::In(e, vals) | RBoolExpr::NotIn(e, vals) => {
            collect_slots_expr(e, slots);
            for v in vals {
                collect_slots_expr(v, slots);
            }
        },
        RBoolExpr::Between(e, lo, hi) => {
            collect_slots_expr(e, slots);
            collect_slots_expr(lo, slots);
            collect_slots_expr(hi, slots);
        },
        RBoolExpr::Like(e, _) => {
            collect_slots_expr(e, slots);
        },
        RBoolExpr::And(a, b) | RBoolExpr::Or(a, b) => {
            collect_slots_bool(a, slots);
            collect_slots_bool(b, slots);
        },
        RBoolExpr::Not(inner) | RBoolExpr::Paren(inner) => {
            collect_slots_bool(inner, slots);
        },
        RBoolExpr::FuncPredicate(_, args) => {
            for a in args {
                collect_slots_expr(a, slots);
            }
        },
    }
}

fn collect_slots_expr(expr: &RExpr, slots: &mut std::collections::HashSet<u8>) {
    match expr {
        RExpr::Property(slot, _) | RExpr::Var(slot) => {
            slots.insert(*slot);
        },
        RExpr::FuncCall(_, args) => {
            for a in args {
                collect_slots_expr(a, slots);
            }
        },
        RExpr::Literal(_) | RExpr::Star => {},
    }
}

// ---------------------------------------------------------------------------
// Rule 4: Scan-to-Index Rewrite
// ---------------------------------------------------------------------------

/// Replace ScanTable + Filter(col=val) with IndexScan when an index exists on col.
pub fn scan_to_index_rewrite(plan: &mut ExecutionPlan, stats: &dyn OptimizerStats) -> bool {
    let mut changed = false;

    // Look for pattern: ScanTable at position i, Filter at position i+1
    // where the filter is col = literal and the table has an index on col.
    let mut i = 0;
    while i + 1 < plan.steps.len() {
        let (is_scan, table_name, alias, scan_limit) = match &plan.steps[i] {
            PlanStep::ScanTable {
                table_name,
                alias,
                scan_limit,
            } => (true, table_name.clone(), alias.clone(), *scan_limit),
            _ => (false, String::new(), None, None),
        };

        if !is_scan {
            i += 1;
            continue;
        }

        // Check if next step is a compatible Filter
        if let PlanStep::Filter(ref expr) = plan.steps[i + 1] {
            if let Some((columns, values)) = extract_equality_predicates(expr) {
                let col_names: Vec<String> = columns.iter().map(|c| c.to_string()).collect();
                if stats.has_index(&table_name, &col_names) {
                    // Replace ScanTable + Filter with IndexScan
                    plan.steps.remove(i + 1); // remove filter
                    plan.steps[i] = PlanStep::IndexScan {
                        table_name,
                        alias,
                        index_columns: col_names,
                        key_values: values,
                        is_point: true,
                        scan_limit,
                    };
                    changed = true;
                    continue;
                }
            }
        }

        i += 1;
    }

    changed
}

/// Extract equality predicates (col = literal) from a filter expression.
/// Returns (column_names, literal_values) if the filter is a conjunction of equalities.
fn extract_equality_predicates(expr: &RBoolExpr) -> Option<(Vec<String>, Vec<Literal>)> {
    match expr {
        RBoolExpr::Comparison(RExpr::Property(_, col), CompOp::Eq, RExpr::Literal(lit)) => {
            Some((vec![col.clone()], vec![lit.clone()]))
        },
        RBoolExpr::Comparison(RExpr::Literal(lit), CompOp::Eq, RExpr::Property(_, col)) => {
            Some((vec![col.clone()], vec![lit.clone()]))
        },
        RBoolExpr::And(a, b) => {
            let (mut cols_a, mut vals_a) = extract_equality_predicates(a)?;
            let (cols_b, vals_b) = extract_equality_predicates(b)?;
            cols_a.extend(cols_b);
            vals_a.extend(vals_b);
            Some((cols_a, vals_a))
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Rule 5: Join Reorder
// ---------------------------------------------------------------------------

/// Reorder joins so that smallest estimated result tables are scanned first.
pub fn join_reorder(plan: &mut ExecutionPlan, stats: &dyn OptimizerStats) -> bool {
    // Find consecutive JoinTable steps and reorder by estimated size.
    let join_start = plan
        .steps
        .iter()
        .position(|s| matches!(s, PlanStep::JoinTable { .. }));
    let join_start = match join_start {
        Some(idx) => idx,
        None => return false,
    };

    let join_end = plan.steps[join_start..]
        .iter()
        .position(|s| !matches!(s, PlanStep::JoinTable { .. }))
        .map(|p| join_start + p)
        .unwrap_or(plan.steps.len());

    let join_count = join_end - join_start;
    if join_count <= 1 {
        return false;
    }

    // Collect (index, estimated_size) for sorting
    let mut indexed: Vec<(usize, usize)> = (join_start..join_end)
        .map(|i| {
            let size = if let PlanStep::JoinTable { table_name, .. } = &plan.steps[i] {
                stats.estimated_row_count(table_name)
            } else {
                usize::MAX
            };
            (i, size)
        })
        .collect();

    // Check if already sorted
    let already_sorted = indexed.windows(2).all(|w| w[0].1 <= w[1].1);
    if already_sorted {
        return false;
    }

    // Sort by estimated row count (ascending)
    indexed.sort_by_key(|&(_, size)| size);

    // Extract steps in sorted order, then put them back
    let sorted_steps: Vec<PlanStep> = indexed
        .iter()
        .map(|&(orig_idx, _)| plan.steps[orig_idx].clone())
        .collect();

    for (offset, step) in sorted_steps.into_iter().enumerate() {
        plan.steps[join_start + offset] = step;
    }

    true
}

// ---------------------------------------------------------------------------
// Rule 6: Hash-to-Index Join
// ---------------------------------------------------------------------------

/// Replace JoinTable with IndexJoin when the right side has a matching index.
pub fn hash_to_index_join(plan: &mut ExecutionPlan, stats: &dyn OptimizerStats) -> bool {
    let mut changed = false;

    for i in 0..plan.steps.len() {
        let replacement = match &plan.steps[i] {
            PlanStep::JoinTable {
                table_name,
                alias,
                join_type,
                on:
                    JoinCondition::TableToTable {
                        left_table,
                        left_col,
                        right_col,
                    },
            } => {
                if stats.has_index(table_name, std::slice::from_ref(right_col)) {
                    Some(PlanStep::IndexJoin {
                        table_name: table_name.clone(),
                        alias: alias.clone(),
                        join_type: join_type.clone(),
                        index_column: right_col.clone(),
                        left_table: left_table.clone(),
                        left_col: left_col.clone(),
                    })
                } else {
                    None
                }
            },
            _ => None,
        };

        if let Some(new_step) = replacement {
            plan.steps[i] = new_step;
            changed = true;
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Rule 7: Limit Pushdown
// ---------------------------------------------------------------------------

/// Push LIMIT down to ScanTable/IndexScan when safe.
/// Safe when there are no joins, aggregations, or ORDER BY that need all rows.
pub fn limit_pushdown(plan: &mut ExecutionPlan) -> bool {
    let limit = match plan.limit {
        Some(l) => l as usize,
        None => return false,
    };

    // Not safe if there are joins, aggregations, or ordering
    let has_joins = plan
        .steps
        .iter()
        .any(|s| matches!(s, PlanStep::JoinTable { .. } | PlanStep::IndexJoin { .. }));
    if has_joins || !plan.aggregations.is_empty() || !plan.ordering.is_empty() {
        return false;
    }

    let mut changed = false;
    for step in &mut plan.steps {
        match step {
            PlanStep::ScanTable { scan_limit, .. } => {
                if scan_limit.is_none() {
                    *scan_limit = Some(limit);
                    changed = true;
                }
            },
            PlanStep::IndexScan { scan_limit, .. } => {
                if scan_limit.is_none() {
                    *scan_limit = Some(limit);
                    changed = true;
                }
            },
            _ => {},
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::planner::{
        ExecutionPlan, JoinCondition, JoinType, OrderSpec, TemporalViewport,
    };
    use super::*;

    fn empty_plan(steps: Vec<PlanStep>) -> ExecutionPlan {
        ExecutionPlan {
            steps,
            projections: vec![],
            aggregations: vec![],
            group_by_keys: vec![],
            having: None,
            ordering: vec![],
            limit: None,
            temporal_viewport: TemporalViewport::ActiveOnly,
            transaction_cutoff: None,
            var_count: 0,
            slot_to_table: vec![],
        }
    }

    #[test]
    fn test_constant_fold_true() {
        let mut plan = empty_plan(vec![PlanStep::Filter(RBoolExpr::Comparison(
            RExpr::Literal(Literal::Int(1)),
            CompOp::Eq,
            RExpr::Literal(Literal::Int(1)),
        ))]);
        let changed = constant_fold(&mut plan);
        assert!(changed);
        // Tautology removed entirely
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_constant_fold_false() {
        let mut plan = empty_plan(vec![PlanStep::Filter(RBoolExpr::Comparison(
            RExpr::Literal(Literal::Int(1)),
            CompOp::Eq,
            RExpr::Literal(Literal::Int(2)),
        ))]);
        let changed = constant_fold(&mut plan);
        assert!(changed);
        // Contradiction stays as an always-false filter
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn test_eliminate_redundant_filters() {
        let filter = RBoolExpr::Comparison(
            RExpr::Property(0, "name".into()),
            CompOp::Eq,
            RExpr::Literal(Literal::String("Alice".into())),
        );
        let mut plan = empty_plan(vec![
            PlanStep::Filter(filter.clone()),
            PlanStep::Filter(filter.clone()),
        ]);
        let changed = eliminate_redundant_filters(&mut plan);
        assert!(changed);
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn test_scan_to_index_rewrite() {
        struct MockStats;
        impl OptimizerStats for MockStats {
            fn estimated_row_count(&self, _: &str) -> usize {
                1000
            }
            fn has_index(&self, table: &str, cols: &[String]) -> bool {
                table == "users" && cols == ["email"]
            }
            fn index_cardinality(&self, _: &str, _: &[String]) -> usize {
                1000
            }
        }

        let mut plan = empty_plan(vec![
            PlanStep::ScanTable {
                table_name: "users".into(),
                alias: None,
                scan_limit: None,
            },
            PlanStep::Filter(RBoolExpr::Comparison(
                RExpr::Property(0, "email".into()),
                CompOp::Eq,
                RExpr::Literal(Literal::String("alice@example.com".into())),
            )),
        ]);
        let changed = scan_to_index_rewrite(&mut plan, &MockStats);
        assert!(changed);
        assert_eq!(plan.steps.len(), 1);
        assert!(matches!(plan.steps[0], PlanStep::IndexScan { .. }));
    }

    #[test]
    fn test_limit_pushdown() {
        let mut plan = empty_plan(vec![PlanStep::ScanTable {
            table_name: "users".into(),
            alias: None,
            scan_limit: None,
        }]);
        plan.limit = Some(10);
        let changed = limit_pushdown(&mut plan);
        assert!(changed);
        if let PlanStep::ScanTable { scan_limit, .. } = &plan.steps[0] {
            assert_eq!(*scan_limit, Some(10));
        } else {
            panic!("expected ScanTable");
        }
    }

    #[test]
    fn test_limit_pushdown_blocked_by_order() {
        let mut plan = empty_plan(vec![PlanStep::ScanTable {
            table_name: "users".into(),
            alias: None,
            scan_limit: None,
        }]);
        plan.limit = Some(10);
        plan.ordering = vec![OrderSpec {
            column_alias: "name".into(),
            descending: false,
        }];
        let changed = limit_pushdown(&mut plan);
        assert!(!changed);
    }

    #[test]
    fn test_hash_to_index_join() {
        struct MockStats;
        impl OptimizerStats for MockStats {
            fn estimated_row_count(&self, _: &str) -> usize {
                1000
            }
            fn has_index(&self, table: &str, cols: &[String]) -> bool {
                table == "orders" && cols == ["user_id"]
            }
            fn index_cardinality(&self, _: &str, _: &[String]) -> usize {
                100
            }
        }

        let mut plan = empty_plan(vec![
            PlanStep::ScanTable {
                table_name: "users".into(),
                alias: None,
                scan_limit: None,
            },
            PlanStep::JoinTable {
                table_name: "orders".into(),
                alias: None,
                join_type: JoinType::Inner,
                on: JoinCondition::TableToTable {
                    left_table: "users".into(),
                    left_col: "id".into(),
                    right_col: "user_id".into(),
                },
            },
        ]);
        let changed = hash_to_index_join(&mut plan, &MockStats);
        assert!(changed);
        assert!(matches!(plan.steps[1], PlanStep::IndexJoin { .. }));
    }

    #[test]
    fn test_full_optimize_pipeline() {
        let mut plan = empty_plan(vec![
            PlanStep::ScanTable {
                table_name: "t".into(),
                alias: None,
                scan_limit: None,
            },
            PlanStep::Filter(RBoolExpr::Comparison(
                RExpr::Literal(Literal::Int(1)),
                CompOp::Eq,
                RExpr::Literal(Literal::Int(1)),
            )),
        ]);
        let trace = optimize(&mut plan, &NoStats);
        assert!(trace.rules_applied.contains(&"constant_fold".to_string()));
        // The tautology filter should be removed
        assert_eq!(plan.steps.len(), 1);
    }
}
