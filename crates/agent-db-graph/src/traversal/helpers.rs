//! Helper functions shared across traversal modules.

use crate::structures::EdgeType;

/// Get a string name for an edge type (for constraint matching).
pub(crate) fn edge_type_name(et: &EdgeType) -> String {
    match et {
        EdgeType::Causality { .. } => "Causality".to_string(),
        EdgeType::Temporal { .. } => "Temporal".to_string(),
        EdgeType::Contextual { .. } => "Contextual".to_string(),
        EdgeType::Interaction { .. } => "Interaction".to_string(),
        EdgeType::GoalRelation { .. } => "GoalRelation".to_string(),
        EdgeType::Association { .. } => "Association".to_string(),
        EdgeType::Communication { .. } => "Communication".to_string(),
        EdgeType::DerivedFrom { .. } => "DerivedFrom".to_string(),
        EdgeType::SupportedBy { .. } => "SupportedBy".to_string(),
        EdgeType::About { .. } => "About".to_string(),
        EdgeType::CodeStructure { .. } => "CodeStructure".to_string(),
    }
}
