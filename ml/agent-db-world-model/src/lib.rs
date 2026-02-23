//! Energy-based predictive coding world model (Critic) for EventGraphDB.
//!
//! This crate implements a **Generator + Critic** architecture's Critic component:
//! an energy-based model (EBM) that scores configurations of (policy, strategy,
//! memory, event) tuples and computes bottom-up prediction errors.
//!
//! # Architecture
//!
//! The world model is **bidirectional**:
//!
//! - **Top-down** (planning/evaluation): Higher layers predict lower layers.
//!   Used to score and rank generated strategy/action candidates.
//!
//! - **Bottom-up** (validation/learning): Reality pushes upward via prediction
//!   errors. Used for novelty detection, repair triggers, and training.
//!
//! # Core Trait
//!
//! [`WorldModelCritic`] is the frozen boundary trait that the graph engine depends
//! on. It provides scoring, prediction error computation, and training methods.

pub mod ebm;
pub mod encoders;
pub mod energy;
pub mod persistence;
pub mod scoring;
pub mod training;
pub mod types;

// Re-export primary types at crate root for ergonomic imports.
pub use types::{
    CriticReport, EnergyStats, EventFeatures, LayerStats, MemoryFeatures, MismatchLayer,
    PolicyFeatures, PredictionErrorReport, StrategyFeatures, TrainingTuple, WorldModelConfig,
};

// Re-export the concrete implementation.
pub use ebm::EbmWorldModel;

/// The world model critic scores configurations and computes prediction errors.
///
/// This is the primary integration trait — GraphEngine depends on this.
/// All methods are synchronous because the world model is CPU-only and fast
/// (< 100 μs per score, < 10 ms per training batch).
pub trait WorldModelCritic: Send + Sync {
    /// Score a full (policy, strategy, memory, event) configuration.
    ///
    /// Used for top-down evaluation of generated candidates. Returns a
    /// [`CriticReport`] with total energy, per-layer energies, novelty z-score,
    /// mismatch attribution, and confidence.
    fn score(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
        memory: &MemoryFeatures,
        event: &EventFeatures,
    ) -> CriticReport;

    /// Score a strategy candidate against a policy (no event/memory needed).
    ///
    /// Used when ranking strategy candidates before execution. The memory and
    /// event energies are set to zero (only policy→strategy compatibility is
    /// evaluated).
    fn score_strategy(&self, policy: &PolicyFeatures, strategy: &StrategyFeatures) -> CriticReport;

    /// Compute bottom-up prediction error for an observed event.
    ///
    /// Used during execution for validation, repair triggers, and learning.
    /// Returns per-layer z-scores indicating how surprising each layer is.
    fn prediction_error(
        &self,
        observed_event: &EventFeatures,
        memory_context: &MemoryFeatures,
        active_strategy: &StrategyFeatures,
        active_policy: &PolicyFeatures,
    ) -> PredictionErrorReport;

    /// Is the model trained enough to produce reliable scores?
    ///
    /// Returns `false` until `warmup_episodes` training examples have been
    /// processed. When not warmed up, scores should be treated as low-confidence.
    fn is_warmed_up(&self) -> bool;

    /// Submit a training tuple (called by learner after episode completion).
    fn submit_training(&mut self, tuple: TrainingTuple);

    /// Run a training step (batch of pending tuples). Returns avg loss.
    ///
    /// Processes up to `training_batch_size` pending tuples using contrastive
    /// learning with margin-based loss.
    fn train_step(&mut self) -> f32;

    /// Get current energy statistics for monitoring.
    fn energy_stats(&self) -> EnergyStats;
}

/// Persistence: serialize/deserialize world model state.
///
/// Uses the same envelope pattern as `agent_db_storage::serialize_versioned`.
pub trait WorldModelStateStore: Send + Sync {
    /// Serialize the current model state to bytes.
    fn save(&self, model: &dyn WorldModelCritic) -> Result<Vec<u8>, String>;

    /// Deserialize a model from bytes.
    fn load(&self, bytes: &[u8]) -> Result<Box<dyn WorldModelCritic>, String>;
}
