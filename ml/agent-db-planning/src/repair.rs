//! Plan repair logic — revise strategies and actions when prediction errors spike.
//!
//! The repair module bridges bottom-up prediction errors from the critic to
//! the generator's revision interface.

use agent_db_world_model::{CriticReport, MismatchLayer, PredictionErrorReport};

use crate::types::{
    ActionGenerationRequest, GeneratedActionPlan, GeneratedStep, GeneratedStrategyPlan,
    PlanningConfig, PlanningError, StrategyGenerationRequest,
};
use crate::{ActionGenerator, StrategyGenerator};

/// Determines whether a prediction error should trigger repair.
pub fn should_repair(error: &PredictionErrorReport, config: &PlanningConfig) -> bool {
    if !config.repair_enabled {
        return false;
    }
    error.total_z > config.repair_z
}

/// Determines the repair scope based on the mismatch layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairScope {
    /// Only repair the current action.
    ActionRepair,
    /// Revise the entire strategy.
    StrategyRevision,
    /// Mismatch is at the policy layer (goal may need revision).
    /// No automated repair — the caller should log and continue.
    PolicyObservation,
}

/// Determine the repair scope from a prediction error.
pub fn determine_repair_scope(
    error: &PredictionErrorReport,
    consecutive_action_repairs: u32,
) -> RepairScope {
    match error.mismatch_layer {
        MismatchLayer::Event => {
            if consecutive_action_repairs >= 3 {
                // Too many action repairs → escalate to strategy revision
                RepairScope::StrategyRevision
            } else {
                RepairScope::ActionRepair
            }
        },
        MismatchLayer::Memory => RepairScope::ActionRepair,
        MismatchLayer::Strategy => RepairScope::StrategyRevision,
        MismatchLayer::Policy => RepairScope::PolicyObservation,
        MismatchLayer::None => RepairScope::ActionRepair,
    }
}

/// Convert a PredictionErrorReport into a CriticReport for passing to the
/// generator's revision interface.
pub fn error_to_diagnostics(error: &PredictionErrorReport) -> CriticReport {
    CriticReport {
        total_energy: error.event_energy + error.memory_energy + error.strategy_energy,
        policy_strategy_energy: error.strategy_energy,
        strategy_memory_energy: error.memory_energy,
        memory_event_energy: error.event_energy,
        novelty_z: error.total_z,
        is_novel: error.total_z > 2.0,
        mismatch_layer: error.mismatch_layer,
        confidence: 1.0, // errors are observed, not estimated
        support_count: 0,
    }
}

/// Attempt to repair an action using the action generator.
pub async fn repair_action(
    action_generator: &dyn ActionGenerator,
    current_step: &GeneratedStep,
    error: &PredictionErrorReport,
    request: &ActionGenerationRequest,
) -> Result<Vec<GeneratedActionPlan>, PlanningError> {
    action_generator.repair(current_step, error, request).await
}

/// Attempt to revise a strategy using the strategy generator.
pub async fn revise_strategy(
    strategy_generator: &dyn StrategyGenerator,
    strategy: &GeneratedStrategyPlan,
    error: &PredictionErrorReport,
    request: &StrategyGenerationRequest,
) -> Result<Vec<GeneratedStrategyPlan>, PlanningError> {
    let diagnostics = error_to_diagnostics(error);
    strategy_generator
        .revise(strategy, &diagnostics, request)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PlanningConfig;

    fn make_error(total_z: f32, layer: MismatchLayer) -> PredictionErrorReport {
        PredictionErrorReport {
            event_energy: 1.0,
            memory_energy: 0.5,
            strategy_energy: 0.3,
            event_z: if layer == MismatchLayer::Event {
                total_z
            } else {
                0.5
            },
            memory_z: if layer == MismatchLayer::Memory {
                total_z
            } else {
                0.5
            },
            strategy_z: if layer == MismatchLayer::Strategy {
                total_z
            } else {
                0.5
            },
            total_z,
            mismatch_layer: layer,
        }
    }

    #[test]
    fn test_should_repair_when_enabled_and_above_threshold() {
        let config = PlanningConfig {
            repair_enabled: true,
            repair_z: 2.0,
            ..PlanningConfig::default()
        };
        assert!(should_repair(
            &make_error(2.5, MismatchLayer::Event),
            &config
        ));
        assert!(!should_repair(
            &make_error(1.5, MismatchLayer::Event),
            &config
        ));
    }

    #[test]
    fn test_should_not_repair_when_disabled() {
        let config = PlanningConfig {
            repair_enabled: false,
            repair_z: 2.0,
            ..PlanningConfig::default()
        };
        assert!(!should_repair(
            &make_error(5.0, MismatchLayer::Event),
            &config
        ));
    }

    #[test]
    fn test_repair_scope_event_layer() {
        let error = make_error(3.0, MismatchLayer::Event);
        assert_eq!(determine_repair_scope(&error, 0), RepairScope::ActionRepair);
        assert_eq!(determine_repair_scope(&error, 2), RepairScope::ActionRepair);
        assert_eq!(
            determine_repair_scope(&error, 3),
            RepairScope::StrategyRevision,
            "3 consecutive action repairs should escalate to strategy revision"
        );
    }

    #[test]
    fn test_repair_scope_strategy_layer() {
        let error = make_error(3.0, MismatchLayer::Strategy);
        assert_eq!(
            determine_repair_scope(&error, 0),
            RepairScope::StrategyRevision
        );
    }

    #[test]
    fn test_repair_scope_policy_layer() {
        let error = make_error(3.0, MismatchLayer::Policy);
        assert_eq!(
            determine_repair_scope(&error, 0),
            RepairScope::PolicyObservation
        );
    }

    #[test]
    fn test_error_to_diagnostics() {
        let error = make_error(2.5, MismatchLayer::Strategy);
        let diag = error_to_diagnostics(&error);
        assert_eq!(diag.mismatch_layer, MismatchLayer::Strategy);
        assert_eq!(diag.novelty_z, 2.5);
        assert!(diag.is_novel);
    }
}
