// crates/agent-db-graph/src/integration/queries/mod.rs
//
// Query execution, NLQ pipeline, search, retrieval, and temporal filtering
// for the GraphEngine.

// Inherit all imports from the integration parent module
use super::*;

mod accessors;
mod actions;
mod drift;
mod filters;
mod nlq;
mod retrieval;
mod search;
mod synthesis;

#[cfg(test)]
mod tests;

// Re-export filter functions for use within the queries sub-modules
pub(super) use filters::{
    apply_epoch_filter, apply_state_anchor_filter, apply_temporal_validity_filter,
};
pub(super) use synthesis::{
    build_dynamic_projection, extract_entity_names_from_question, synthesize_answer,
};
