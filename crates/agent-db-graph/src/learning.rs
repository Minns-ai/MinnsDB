//! Learning statistics store for persistent transition and motif tracking
//!
//! This module provides the LearningStatsStore trait and its redb-backed implementation.
//! It tracks transition statistics (Markov/MDP layer) and motif statistics (contrastive distiller).

use agent_db_core::types::Timestamp;
use agent_db_storage::{RedbBackend, StorageResult};
use serde::{Serialize, Deserialize};
use std::sync::Arc;

/// Transition statistics for state-action-state transitions
///
/// Used for Markov/MDP-based learning and strategy refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionStats {
    /// Total number of times this transition was observed
    pub count: u64,

    /// Number of successful outcomes
    pub success_count: u64,

    /// Number of failed outcomes
    pub failure_count: u64,

    /// Bayesian posterior alpha parameter (successes + prior)
    pub posterior_alpha: f32,

    /// Bayesian posterior beta parameter (failures + prior)
    pub posterior_beta: f32,

    /// Last time this transition was updated
    pub last_updated: Timestamp,
}

/// Motif statistics for contrastive learning
///
/// Tracks success/failure rates and uplift for behavior patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifStats {
    /// Number of successful episodes with this motif
    pub success_count: u32,

    /// Number of failed episodes with this motif
    pub failure_count: u32,

    /// Lift relative to baseline (>1.0 means better than average)
    pub lift: f32,

    /// Uplift (additive improvement over baseline)
    pub uplift: f32,

    /// Last time this motif was updated
    pub last_updated: Timestamp,
}

/// Learning statistics store trait
///
/// Provides persistence for transition statistics and motif patterns
pub trait LearningStatsStore: Send + Sync {
    // ========== Transition Stats (Markov/MDP layer) ==========

    /// Store transition statistics
    fn put_transition(
        &mut self,
        goal_bucket_id: u64,
        state: String,
        action: String,
        next_state: String,
        stats: TransitionStats,
    ) -> StorageResult<()>;

    /// Get transition statistics for a specific state-action-state triple
    fn get_transition(
        &self,
        goal_bucket_id: u64,
        state: String,
        action: String,
        next_state: String,
    ) -> StorageResult<Option<TransitionStats>>;

    /// Get all transitions from a given state
    ///
    /// Returns: (action, next_state, stats) tuples
    fn get_transitions_from_state(
        &self,
        goal_bucket_id: u64,
        state: String,
    ) -> StorageResult<Vec<(String, String, TransitionStats)>>;

    // ========== Motif Stats (Contrastive distiller) ==========

    /// Store motif statistics
    fn put_motif(
        &mut self,
        goal_bucket_id: u64,
        motif_id: String,
        stats: MotifStats,
    ) -> StorageResult<()>;

    /// Get motif statistics
    fn get_motif(
        &self,
        goal_bucket_id: u64,
        motif_id: String,
    ) -> StorageResult<Option<MotifStats>>;

    /// Get all motifs for a goal bucket
    ///
    /// Returns: (motif_id, stats) tuples
    fn get_motifs(
        &self,
        goal_bucket_id: u64,
    ) -> StorageResult<Vec<(String, MotifStats)>>;
}

/// Redb-backed learning statistics store
///
/// Uses the following redb tables:
/// - transition_stats: (goal_bucket_id, state, action, next_state) → TransitionStats
/// - motif_stats: (goal_bucket_id, motif_id) → MotifStats
pub struct RedbLearningStatsStore {
    backend: Arc<RedbBackend>,
}

impl RedbLearningStatsStore {
    /// Create a new redb learning stats store
    pub fn new(backend: Arc<RedbBackend>) -> Self {
        Self { backend }
    }

    /// Encode transition key: (goal_bucket_id, state, action, next_state) → bytes
    fn encode_transition_key(
        goal_bucket_id: u64,
        state: &str,
        action: &str,
        next_state: &str,
    ) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(&goal_bucket_id.to_be_bytes());
        key.push(0xFF); // separator
        key.extend_from_slice(state.as_bytes());
        key.push(0xFF); // separator
        key.extend_from_slice(action.as_bytes());
        key.push(0xFF); // separator
        key.extend_from_slice(next_state.as_bytes());
        key
    }

    /// Encode transition prefix key for scanning: (goal_bucket_id, state) → bytes
    fn encode_transition_prefix(goal_bucket_id: u64, state: &str) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(&goal_bucket_id.to_be_bytes());
        key.push(0xFF); // separator
        key.extend_from_slice(state.as_bytes());
        key.push(0xFF); // separator
        key
    }

    /// Decode transition key components from a full key
    fn decode_transition_key(key: &[u8]) -> StorageResult<(u64, String, String, String)> {
        let parts: Vec<&[u8]> = key.split(|&b| b == 0xFF).collect();
        if parts.len() != 4 {
            return Err(agent_db_storage::StorageError::DatabaseError(
                format!("Invalid transition key format: expected 4 parts, got {}", parts.len())
            ));
        }

        let goal_bucket_id = u64::from_be_bytes(
            parts[0].try_into().map_err(|_| {
                agent_db_storage::StorageError::DatabaseError("Invalid goal_bucket_id".to_string())
            })?
        );
        let state = String::from_utf8_lossy(parts[1]).to_string();
        let action = String::from_utf8_lossy(parts[2]).to_string();
        let next_state = String::from_utf8_lossy(parts[3]).to_string();

        Ok((goal_bucket_id, state, action, next_state))
    }

    /// Encode motif key: (goal_bucket_id, motif_id) → bytes
    fn encode_motif_key(goal_bucket_id: u64, motif_id: &str) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(&goal_bucket_id.to_be_bytes());
        key.push(0xFF); // separator
        key.extend_from_slice(motif_id.as_bytes());
        key
    }

    /// Encode motif prefix key for scanning: goal_bucket_id → bytes
    fn encode_motif_prefix(goal_bucket_id: u64) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(&goal_bucket_id.to_be_bytes());
        key.push(0xFF); // separator
        key
    }

    /// Decode motif key components
    fn decode_motif_key(key: &[u8]) -> StorageResult<(u64, String)> {
        let parts: Vec<&[u8]> = key.split(|&b| b == 0xFF).collect();
        if parts.len() != 2 {
            return Err(agent_db_storage::StorageError::DatabaseError(
                format!("Invalid motif key format: expected 2 parts, got {}", parts.len())
            ));
        }

        let goal_bucket_id = u64::from_be_bytes(
            parts[0].try_into().map_err(|_| {
                agent_db_storage::StorageError::DatabaseError("Invalid goal_bucket_id".to_string())
            })?
        );
        let motif_id = String::from_utf8_lossy(parts[1]).to_string();

        Ok((goal_bucket_id, motif_id))
    }
}

impl LearningStatsStore for RedbLearningStatsStore {
    fn put_transition(
        &mut self,
        goal_bucket_id: u64,
        state: String,
        action: String,
        next_state: String,
        stats: TransitionStats,
    ) -> StorageResult<()> {
        let key = Self::encode_transition_key(goal_bucket_id, &state, &action, &next_state);
        self.backend.put("transition_stats", key, &stats)
    }

    fn get_transition(
        &self,
        goal_bucket_id: u64,
        state: String,
        action: String,
        next_state: String,
    ) -> StorageResult<Option<TransitionStats>> {
        let key = Self::encode_transition_key(goal_bucket_id, &state, &action, &next_state);
        self.backend.get("transition_stats", key)
    }

    fn get_transitions_from_state(
        &self,
        goal_bucket_id: u64,
        state: String,
    ) -> StorageResult<Vec<(String, String, TransitionStats)>> {
        let prefix = Self::encode_transition_prefix(goal_bucket_id, &state);
        let results: Vec<(Vec<u8>, TransitionStats)> =
            self.backend.scan_prefix("transition_stats", prefix)?;

        let mut transitions = Vec::new();
        for (key_bytes, stats) in results {
            // Decode key to extract action and next_state
            let (_bucket, _state, action, next_state) = Self::decode_transition_key(&key_bytes)?;
            transitions.push((action, next_state, stats));
        }

        Ok(transitions)
    }

    fn put_motif(
        &mut self,
        goal_bucket_id: u64,
        motif_id: String,
        stats: MotifStats,
    ) -> StorageResult<()> {
        let key = Self::encode_motif_key(goal_bucket_id, &motif_id);
        self.backend.put("motif_stats", key, &stats)
    }

    fn get_motif(
        &self,
        goal_bucket_id: u64,
        motif_id: String,
    ) -> StorageResult<Option<MotifStats>> {
        let key = Self::encode_motif_key(goal_bucket_id, &motif_id);
        self.backend.get("motif_stats", key)
    }

    fn get_motifs(
        &self,
        goal_bucket_id: u64,
    ) -> StorageResult<Vec<(String, MotifStats)>> {
        let prefix = Self::encode_motif_prefix(goal_bucket_id);
        let results: Vec<(Vec<u8>, MotifStats)> =
            self.backend.scan_prefix("motif_stats", prefix)?;

        let mut motifs = Vec::new();
        for (key_bytes, stats) in results {
            // Decode key to extract motif_id
            let (_bucket, motif_id) = Self::decode_motif_key(&key_bytes)?;
            motifs.push((motif_id, stats));
        }

        Ok(motifs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_storage::RedbConfig;
    use tempfile::TempDir;

    fn create_test_backend() -> (Arc<RedbBackend>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.db"),
            cache_size_bytes: 1024 * 1024,
            repair_on_open: false,
        };
        let backend = Arc::new(RedbBackend::open(config).unwrap());
        (backend, temp_dir)
    }

    fn create_test_transition_stats() -> TransitionStats {
        TransitionStats {
            count: 10,
            success_count: 8,
            failure_count: 2,
            posterior_alpha: 9.0,
            posterior_beta: 3.0,
            last_updated: 1000,
        }
    }

    fn create_test_motif_stats() -> MotifStats {
        MotifStats {
            success_count: 15,
            failure_count: 5,
            lift: 1.5,
            uplift: 0.25,
            last_updated: 2000,
        }
    }

    #[test]
    fn test_put_and_get_transition() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbLearningStatsStore::new(backend);

        let stats = create_test_transition_stats();
        store.put_transition(
            1,
            "state_a".to_string(),
            "action_1".to_string(),
            "state_b".to_string(),
            stats.clone(),
        ).unwrap();

        let retrieved = store.get_transition(
            1,
            "state_a".to_string(),
            "action_1".to_string(),
            "state_b".to_string(),
        ).unwrap();

        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.count, 10);
        assert_eq!(retrieved.success_count, 8);
    }

    #[test]
    fn test_get_transitions_from_state() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbLearningStatsStore::new(backend);

        // Store multiple transitions from the same state
        let stats1 = create_test_transition_stats();
        store.put_transition(
            1,
            "state_a".to_string(),
            "action_1".to_string(),
            "state_b".to_string(),
            stats1,
        ).unwrap();

        let mut stats2 = create_test_transition_stats();
        stats2.count = 5;
        store.put_transition(
            1,
            "state_a".to_string(),
            "action_2".to_string(),
            "state_c".to_string(),
            stats2,
        ).unwrap();

        // Retrieve all transitions from state_a
        let transitions = store.get_transitions_from_state(1, "state_a".to_string()).unwrap();
        assert_eq!(transitions.len(), 2);
    }

    #[test]
    fn test_put_and_get_motif() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbLearningStatsStore::new(backend);

        let stats = create_test_motif_stats();
        store.put_motif(1, "motif_1".to_string(), stats.clone()).unwrap();

        let retrieved = store.get_motif(1, "motif_1".to_string()).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.success_count, 15);
        assert_eq!(retrieved.lift, 1.5);
    }

    #[test]
    fn test_get_motifs() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbLearningStatsStore::new(backend);

        // Store multiple motifs
        let stats1 = create_test_motif_stats();
        store.put_motif(1, "motif_1".to_string(), stats1).unwrap();

        let mut stats2 = create_test_motif_stats();
        stats2.lift = 2.0;
        store.put_motif(1, "motif_2".to_string(), stats2).unwrap();

        // Retrieve all motifs for bucket 1
        let motifs = store.get_motifs(1).unwrap();
        assert_eq!(motifs.len(), 2);
    }
}
