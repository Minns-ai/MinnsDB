//! Pure walks over typed structured-memory projections.
//!
//! A walker takes a built projection (`StateMachine`, eventually `Tree`,
//! `Ledger`, `PreferenceList`) plus an NLQ frame + intent, and returns the
//! answer as a typed result. No I/O, no async, no graph access — the
//! projection layer already did that work.
//!
//! Splitting walk-from-projection is what lets the planner pick a cheap path
//! (e.g. active-edge fetch for Current) without touching the same code that
//! handles First / Last / Nth / Historical.

use crate::nlq::llm_hint::{IntentHint, TemporalFrame};
use crate::structured_memory::StateMachine;

/// Result of walking a state-machine projection.
///
/// Carries the answer + minimal provenance (which transition, what timestamp)
/// so the synthesis layer can include "since/until" context in its phrasing
/// without re-reading the projection.
#[derive(Debug, Clone)]
pub struct StateMachineAnswer {
    /// The target value the walk landed on (e.g. "luna", "max"). Empty when
    /// the state machine has no matching entry (e.g. asked for the 5th pet
    /// when only 2 exist).
    pub value: String,
    /// `valid_from` of the transition this value came from. `None` for
    /// `current_state` lookups (the field isn't carried on the SM root).
    pub since: Option<u64>,
    /// Whether the answer was inferred from history (`true`) or read straight
    /// off `current_state` (`false`).
    pub from_history: bool,
    /// For `Historical`, the full ordered list of values (oldest → newest).
    /// Empty otherwise.
    pub all_values: Vec<String>,
}

impl StateMachineAnswer {
    fn empty() -> Self {
        Self {
            value: String::new(),
            since: None,
            from_history: false,
            all_values: Vec::new(),
        }
    }
}

/// Default cap on `Historical` results. Surfaces the most-recent N
/// transitions to the synthesis layer. Rule 45 (limits) — every projection
/// walker that can grow output must expose a bound; if you need more,
/// call the lower-level form.
pub const DEFAULT_HISTORICAL_LIMIT: usize = 50;

/// Walk a state-machine projection according to `frame` (and `intent` where
/// it disambiguates).
///
/// `history_limit` caps the `Historical` / `Comparative` paths so a runaway
/// state machine (entity with many thousands of transitions) can't blow up
/// the synthesis prompt or the allocation. For `First` / `Last` / `Current`
/// the limit is unused — those return at most one value.
///
/// Routing:
///
/// - `Current` is unusual here — the cheap path is `project_entity_state` in
///   the planner. If we *do* land in this walker with Current, we return
///   `current_state` directly.
/// - `First`  → `history[0].to`.
/// - `Last`   → `history.last().to`, or `current_state` if non-empty
///   (same answer; cheaper to read).
/// - `Historical` → the last `history_limit` `.to`s, chronological order.
/// - `Comparative` → first vs last for now; a future iteration can return a
///   richer diff structure.
/// - `Timeless` → not handled here (the planner should not route Timeless
///   through a state-machine walk).
pub fn walk_state_machine(
    sm: &StateMachine,
    frame: &TemporalFrame,
    _intent: &IntentHint,
    history_limit: usize,
) -> StateMachineAnswer {
    match frame {
        TemporalFrame::Current => StateMachineAnswer {
            value: sm.current_state.clone(),
            since: None,
            from_history: false,
            all_values: Vec::new(),
        },

        TemporalFrame::First => match sm.history.first() {
            Some(t) => StateMachineAnswer {
                value: t.to.clone(),
                since: Some(t.timestamp),
                from_history: true,
                all_values: Vec::new(),
            },
            None => StateMachineAnswer::empty(),
        },

        TemporalFrame::Last => {
            // current_state and history.last() should agree when an active
            // edge exists. Prefer current_state because it's also the answer
            // when the latest history step is the active one. Fall back to
            // history.last() if the state has ended without replacement
            // (current_state is empty in that case).
            if !sm.current_state.is_empty() {
                let since = sm.history.last().map(|t| t.timestamp);
                return StateMachineAnswer {
                    value: sm.current_state.clone(),
                    since,
                    from_history: false,
                    all_values: Vec::new(),
                };
            }
            match sm.history.last() {
                Some(t) => StateMachineAnswer {
                    value: t.to.clone(),
                    since: Some(t.timestamp),
                    from_history: true,
                    all_values: Vec::new(),
                },
                None => StateMachineAnswer::empty(),
            }
        },

        TemporalFrame::Historical | TemporalFrame::Comparative => {
            // Bounded: take the last `history_limit` transitions. For an
            // entity with millions of edges this would otherwise allocate
            // an enormous vec. The synthesis prompt is also bounded so
            // anything beyond ~50 entries is dropped downstream regardless.
            let total = sm.history.len();
            let take_n = history_limit.min(total);
            let start = total.saturating_sub(take_n);
            let all: Vec<String> = sm.history[start..total]
                .iter()
                .map(|t| t.to.clone())
                .collect();
            let value = all.last().cloned().unwrap_or_default();
            let since = sm.history.last().map(|t| t.timestamp);
            StateMachineAnswer {
                value,
                since,
                from_history: true,
                all_values: all,
            }
        },

        // Timeless and any future variants the planner didn't intend to
        // route here. Return empty so the caller can fall back without
        // crashing on a state-machine answer that doesn't apply.
        TemporalFrame::Timeless => StateMachineAnswer::empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_memory::{MemoryProvenance, StateTransition};

    fn sm_two_pets() -> StateMachine {
        StateMachine {
            entity: "user".to_string(),
            current_state: "max".to_string(),
            history: vec![
                StateTransition {
                    from: String::new(),
                    to: "luna".to_string(),
                    timestamp: 100,
                    trigger: "adopted".to_string(),
                },
                StateTransition {
                    from: "luna".to_string(),
                    to: "max".to_string(),
                    timestamp: 200,
                    trigger: "adopted".to_string(),
                },
            ],
            provenance: MemoryProvenance::EpisodePipeline,
        }
    }

    #[test]
    fn first_returns_oldest_target() {
        let sm = sm_two_pets();
        let a = walk_state_machine(
            &sm,
            &TemporalFrame::First,
            &IntentHint::Knowledge,
            DEFAULT_HISTORICAL_LIMIT,
        );
        assert_eq!(a.value, "luna");
        assert_eq!(a.since, Some(100));
        assert!(a.from_history);
    }

    #[test]
    fn last_prefers_current_state_when_active() {
        let sm = sm_two_pets();
        let a = walk_state_machine(
            &sm,
            &TemporalFrame::Last,
            &IntentHint::Knowledge,
            DEFAULT_HISTORICAL_LIMIT,
        );
        assert_eq!(a.value, "max");
        assert!(!a.from_history);
    }

    #[test]
    fn last_falls_back_to_history_when_state_ended() {
        let mut sm = sm_two_pets();
        sm.current_state = String::new();
        let a = walk_state_machine(
            &sm,
            &TemporalFrame::Last,
            &IntentHint::Knowledge,
            DEFAULT_HISTORICAL_LIMIT,
        );
        assert_eq!(a.value, "max");
        assert!(a.from_history);
    }

    #[test]
    fn historical_returns_all_values() {
        let sm = sm_two_pets();
        let a = walk_state_machine(
            &sm,
            &TemporalFrame::Historical,
            &IntentHint::Knowledge,
            DEFAULT_HISTORICAL_LIMIT,
        );
        assert_eq!(a.all_values, vec!["luna", "max"]);
    }

    #[test]
    fn current_reads_current_state() {
        let sm = sm_two_pets();
        let a = walk_state_machine(
            &sm,
            &TemporalFrame::Current,
            &IntentHint::Knowledge,
            DEFAULT_HISTORICAL_LIMIT,
        );
        assert_eq!(a.value, "max");
        assert!(!a.from_history);
    }

    #[test]
    fn empty_history_yields_empty_answer_for_first() {
        let sm = StateMachine {
            entity: "u".to_string(),
            current_state: String::new(),
            history: vec![],
            provenance: MemoryProvenance::EpisodePipeline,
        };
        let a = walk_state_machine(
            &sm,
            &TemporalFrame::First,
            &IntentHint::Knowledge,
            DEFAULT_HISTORICAL_LIMIT,
        );
        assert_eq!(a.value, "");
        assert!(!a.from_history);
    }
}
