//! Question optimiser.
//!
//! Maps a classified question (`LlmHintResponse`) onto the cheapest execution
//! shape that can answer it. Returns a pure-data `NlqPlan`; execution lives
//! in `integration/queries/nlq.rs` which dispatches on the variant.
//!
//! This is the single place where routing decisions are made. Tracing emits
//! `nlq.plan` per call so the chosen branch is grep-able from logs.
//!
//! Design rules:
//! - **Pure**: no I/O, no async, no graph access. `plan(hint) -> NlqPlan`.
//! - **Cost-aware**: `Current`-frame state-machine queries get an
//!   `ActiveEdgeFetch` plan that skips trajectory construction. `First`,
//!   `Last`, `Nth`, `Historical`, `Comparative` get `StateMachineWalk` which
//!   pays for the full projection.
//! - **Conservative**: when subject + predicate aren't available from the
//!   classifier, fall back to `UnifiedRetrieval`. Don't guess.

use crate::nlq::llm_hint::{IntentHint, LlmHintResponse, StructureHint, TemporalFrame};

/// The execution plan chosen for an NLQ question.
///
/// One variant per cost shape. The integration layer matches on this enum
/// once and runs the corresponding code path; no plan variant requires
/// re-classifying or re-prompting.
#[derive(Debug, Clone)]
pub enum NlqPlan {
    /// Walk a state-machine projection built from graph edges.
    /// Used for `First / Last / Nth / Historical / Comparative` on a
    /// state-like predicate.
    StateMachineWalk {
        subject: String,
        predicate: String,
        frame: TemporalFrame,
        intent: IntentHint,
    },

    /// Fast path: filter to active edges only. No trajectory build.
    /// Used for `Current` frame on a state-like predicate.
    ActiveEdgeFetch {
        subject: String,
        predicate: String,
        intent: IntentHint,
    },

    // Phase 2 — when Tree / Ledger / PreferenceList projections land:
    // TreeWalk            { root: String, intent: IntentHint },
    // LedgerLookup        { pair: (String, String), intent: IntentHint },
    // PreferenceListWalk  { subject: String, category: String, intent: IntentHint },
    /// Fall through to today's BM25 + vector + claims + entity retrieval +
    /// LLM synthesis pipeline. Used for `GenericGraph` hints and for any
    /// case the structured paths above couldn't satisfy (e.g. missing
    /// subject/predicate extraction from the classifier).
    UnifiedRetrieval { reason: &'static str },
}

impl NlqPlan {
    /// Short identifier for the `nlq.plan` tracing line.
    pub fn variant_name(&self) -> &'static str {
        match self {
            NlqPlan::StateMachineWalk { .. } => "StateMachineWalk",
            NlqPlan::ActiveEdgeFetch { .. } => "ActiveEdgeFetch",
            NlqPlan::UnifiedRetrieval { .. } => "UnifiedRetrieval",
        }
    }
}

/// Pure routing. Reads a classified question, returns the plan.
///
/// No graph access, no async. The integration layer is the only thing that
/// touches a graph or an LLM; this function is fully testable from a
/// `LlmHintResponse` literal.
pub fn plan(hint: &LlmHintResponse) -> NlqPlan {
    match &hint.structure_hint {
        StructureHint::StateMachine => plan_state_machine(hint),

        // Phase 2:
        // StructureHint::Tree            => plan_tree(hint),
        // StructureHint::Ledger          => plan_ledger(hint),
        // StructureHint::PreferenceList  => plan_preference_list(hint),
        _ => NlqPlan::UnifiedRetrieval {
            reason: "structure_hint not yet routed to a projection",
        },
    }
}

/// Plan a StateMachine-hinted question. Falls back to UnifiedRetrieval when
/// the classifier didn't extract a subject + predicate (we can't project
/// without knowing what slice of the graph to walk).
fn plan_state_machine(hint: &LlmHintResponse) -> NlqPlan {
    let Some(subject) = hint.subject.clone() else {
        return NlqPlan::UnifiedRetrieval {
            reason: "state_machine hint without subject",
        };
    };
    let Some(predicate) = hint.predicate.clone() else {
        return NlqPlan::UnifiedRetrieval {
            reason: "state_machine hint without predicate",
        };
    };

    match hint.temporal_frame {
        TemporalFrame::Current => NlqPlan::ActiveEdgeFetch {
            subject,
            predicate,
            intent: hint.intent_hint.clone(),
        },

        TemporalFrame::First
        | TemporalFrame::Last
        | TemporalFrame::Historical
        | TemporalFrame::Comparative => NlqPlan::StateMachineWalk {
            subject,
            predicate,
            frame: hint.temporal_frame.clone(),
            intent: hint.intent_hint.clone(),
        },

        // Timeless facts (identity, family) don't need a temporal walk —
        // an active-edge fetch is the right shape: at any moment in time
        // the answer is the active edge.
        TemporalFrame::Timeless => NlqPlan::ActiveEdgeFetch {
            subject,
            predicate,
            intent: hint.intent_hint.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlq::llm_hint::{IntentHint, LlmHintResponse, StructureHint, TemporalFrame};

    fn hint(
        structure: StructureHint,
        frame: TemporalFrame,
        subject: Option<&str>,
        predicate: Option<&str>,
    ) -> LlmHintResponse {
        LlmHintResponse {
            structure_hint: structure,
            intent_hint: IntentHint::Knowledge,
            temporal_frame: frame,
            subject: subject.map(|s| s.to_string()),
            predicate: predicate.map(|s| s.to_string()),
        }
    }

    #[test]
    fn first_on_state_machine_goes_to_walk() {
        let p = plan(&hint(
            StructureHint::StateMachine,
            TemporalFrame::First,
            Some("user"),
            Some("adopted"),
        ));
        assert!(matches!(p, NlqPlan::StateMachineWalk { .. }));
    }

    #[test]
    fn current_on_state_machine_goes_to_active_edge_fetch() {
        let p = plan(&hint(
            StructureHint::StateMachine,
            TemporalFrame::Current,
            Some("user"),
            Some("lives_in"),
        ));
        assert!(matches!(p, NlqPlan::ActiveEdgeFetch { .. }));
    }

    #[test]
    fn missing_subject_falls_back_to_unified() {
        let p = plan(&hint(
            StructureHint::StateMachine,
            TemporalFrame::First,
            None,
            Some("adopted"),
        ));
        assert!(matches!(p, NlqPlan::UnifiedRetrieval { .. }));
    }

    #[test]
    fn missing_predicate_falls_back_to_unified() {
        let p = plan(&hint(
            StructureHint::StateMachine,
            TemporalFrame::First,
            Some("user"),
            None,
        ));
        assert!(matches!(p, NlqPlan::UnifiedRetrieval { .. }));
    }

    #[test]
    fn generic_graph_always_falls_back_to_unified() {
        let p = plan(&hint(
            StructureHint::GenericGraph,
            TemporalFrame::Current,
            Some("user"),
            Some("anything"),
        ));
        assert!(matches!(p, NlqPlan::UnifiedRetrieval { .. }));
    }
}
