//! Decision trace store for auditable retrieved → used → outcome tracking
//!
//! This module provides the DecisionTraceStore trait and its redb-backed implementation.
//! It enables auditing of which memories/strategies were retrieved, which were actually used,
//! and what outcomes resulted from their application.

use agent_db_core::types::{AgentId, SessionId, Timestamp, current_timestamp};
use agent_db_storage::{RedbBackend, StorageResult};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::memory::MemoryId;
use crate::strategies::StrategyId;

/// Outcome signal for a decision trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSignal {
    /// Whether the outcome was successful
    pub success: bool,

    /// Additional metadata about the outcome
    pub metadata: HashMap<String, String>,
}

/// Decision trace record
///
/// Tracks the full lifecycle of a retrieval: what was retrieved, what was used, and the outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    /// Unique query identifier
    pub query_id: String,

    /// Agent making the decision
    pub agent_id: AgentId,

    /// Session identifier
    pub session_id: SessionId,

    /// Memories retrieved for this decision
    pub retrieved_memory_ids: Vec<MemoryId>,

    /// Strategies retrieved for this decision
    pub retrieved_strategy_ids: Vec<StrategyId>,

    /// Memories actually used (subset of retrieved)
    pub used_memory_ids: Vec<MemoryId>,

    /// Strategies actually used (subset of retrieved)
    pub used_strategy_ids: Vec<StrategyId>,

    /// Outcome of this decision (None if still pending)
    pub outcome: Option<OutcomeSignal>,

    /// Policy version used for this decision
    pub policy_version: String,

    /// When this trace started
    pub started_at: Timestamp,

    /// When this trace was closed (None if still open)
    pub closed_at: Option<Timestamp>,
}

/// Decision trace store trait
///
/// Provides persistence for auditable decision tracking
pub trait DecisionTraceStore: Send + Sync {
    /// Start a decision trace (when agent retrieves memories/strategies)
    fn start(
        &mut self,
        query_id: String,
        agent_id: AgentId,
        session_id: SessionId,
        retrieved_memory_ids: Vec<MemoryId>,
        retrieved_strategy_ids: Vec<StrategyId>,
    ) -> StorageResult<()>;

    /// Mark a memory as used (agent actually applied it)
    fn mark_memory_used(&mut self, query_id: String, memory_id: MemoryId) -> StorageResult<()>;

    /// Mark a strategy as used (agent actually applied it)
    fn mark_strategy_used(&mut self, query_id: String, strategy_id: StrategyId) -> StorageResult<()>;

    /// Close the trace with an outcome signal
    fn close(&mut self, query_id: String, outcome: OutcomeSignal) -> StorageResult<()>;

    /// Get the full trace (for audits, debugging, learning updates)
    fn get(&self, query_id: String) -> StorageResult<Option<DecisionTrace>>;

    /// List recent traces for an agent (for debugging)
    fn list_recent(&self, agent_id: AgentId, limit: usize) -> StorageResult<Vec<DecisionTrace>>;
}

/// Redb-backed decision trace store
///
/// Uses the following redb tables:
/// - decision_trace: query_id → DecisionTrace  (main storage)
/// - outcome_signals: (agent_id, timestamp, query_id) → empty  (agent index for range queries)
pub struct RedbDecisionTraceStore {
    backend: Arc<RedbBackend>,
}

impl RedbDecisionTraceStore {
    /// Create a new redb decision trace store
    pub fn new(backend: Arc<RedbBackend>) -> Self {
        Self { backend }
    }

    /// Encode query ID as key
    fn encode_query_key(query_id: &str) -> Vec<u8> {
        query_id.as_bytes().to_vec()
    }

    /// Encode agent index key: (agent_id, timestamp, query_id) → bytes
    fn encode_agent_index_key(agent_id: AgentId, timestamp: Timestamp, query_id: &str) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(&agent_id.to_be_bytes());
        key.extend_from_slice(&timestamp.to_be_bytes());
        key.push(0xFF); // separator
        key.extend_from_slice(query_id.as_bytes());
        key
    }

    /// Decode agent index key to extract query_id
    fn decode_agent_index_key(key: &[u8]) -> StorageResult<(AgentId, Timestamp, String)> {
        if key.len() < 17 {
            return Err(agent_db_storage::StorageError::DatabaseError(
                "Agent index key too short".to_string()
            ));
        }

        let agent_id = u64::from_be_bytes(key[0..8].try_into().unwrap());
        let timestamp = u64::from_be_bytes(key[8..16].try_into().unwrap());

        // Find the separator
        let separator_idx = key.iter().position(|&b| b == 0xFF).ok_or_else(|| {
            agent_db_storage::StorageError::DatabaseError("No separator in agent index key".to_string())
        })?;

        let query_id = String::from_utf8_lossy(&key[separator_idx + 1..]).to_string();

        Ok((agent_id, timestamp, query_id))
    }

    /// Update agent index for a trace
    fn update_agent_index(&self, trace: &DecisionTrace) -> StorageResult<()> {
        let timestamp = trace.closed_at.unwrap_or(trace.started_at);
        let key = Self::encode_agent_index_key(trace.agent_id, timestamp, &trace.query_id);
        self.backend.put("outcome_signals", key, &())?;
        Ok(())
    }
}

impl DecisionTraceStore for RedbDecisionTraceStore {
    fn start(
        &mut self,
        query_id: String,
        agent_id: AgentId,
        session_id: SessionId,
        retrieved_memory_ids: Vec<MemoryId>,
        retrieved_strategy_ids: Vec<StrategyId>,
    ) -> StorageResult<()> {
        let trace = DecisionTrace {
            query_id: query_id.clone(),
            agent_id,
            session_id,
            retrieved_memory_ids,
            retrieved_strategy_ids,
            used_memory_ids: Vec::new(),
            used_strategy_ids: Vec::new(),
            outcome: None,
            policy_version: "v1".to_string(), // TODO: Make configurable
            started_at: current_timestamp(),
            closed_at: None,
        };

        // Store trace
        let key = Self::encode_query_key(&query_id);
        self.backend.put("decision_trace", key, &trace)?;

        // Update agent index
        self.update_agent_index(&trace)?;

        Ok(())
    }

    fn mark_memory_used(&mut self, query_id: String, memory_id: MemoryId) -> StorageResult<()> {
        // Load existing trace
        let key = Self::encode_query_key(&query_id);
        let mut trace: DecisionTrace = self.backend.get("decision_trace", key.clone())?
            .ok_or_else(|| agent_db_storage::StorageError::DatabaseError(
                format!("Decision trace not found: {}", query_id)
            ))?;

        // Add memory to used list if not already there
        if !trace.used_memory_ids.contains(&memory_id) {
            trace.used_memory_ids.push(memory_id);
        }

        // Store updated trace
        self.backend.put("decision_trace", key, &trace)?;

        Ok(())
    }

    fn mark_strategy_used(&mut self, query_id: String, strategy_id: StrategyId) -> StorageResult<()> {
        // Load existing trace
        let key = Self::encode_query_key(&query_id);
        let mut trace: DecisionTrace = self.backend.get("decision_trace", key.clone())?
            .ok_or_else(|| agent_db_storage::StorageError::DatabaseError(
                format!("Decision trace not found: {}", query_id)
            ))?;

        // Add strategy to used list if not already there
        if !trace.used_strategy_ids.contains(&strategy_id) {
            trace.used_strategy_ids.push(strategy_id);
        }

        // Store updated trace
        self.backend.put("decision_trace", key, &trace)?;

        Ok(())
    }

    fn close(&mut self, query_id: String, outcome: OutcomeSignal) -> StorageResult<()> {
        // Load existing trace
        let key = Self::encode_query_key(&query_id);
        let mut trace: DecisionTrace = self.backend.get("decision_trace", key.clone())?
            .ok_or_else(|| agent_db_storage::StorageError::DatabaseError(
                format!("Decision trace not found: {}", query_id)
            ))?;

        // Close trace
        trace.outcome = Some(outcome);
        trace.closed_at = Some(current_timestamp());

        // Store updated trace
        self.backend.put("decision_trace", key, &trace)?;

        // Update agent index with closed_at timestamp
        self.update_agent_index(&trace)?;

        Ok(())
    }

    fn get(&self, query_id: String) -> StorageResult<Option<DecisionTrace>> {
        let key = Self::encode_query_key(&query_id);
        self.backend.get("decision_trace", key)
    }

    fn list_recent(&self, agent_id: AgentId, limit: usize) -> StorageResult<Vec<DecisionTrace>> {
        // Use agent_id as prefix to get all traces for this agent
        let agent_prefix = agent_id.to_be_bytes().to_vec();

        // Get all entries from agent index
        let results: Vec<(Vec<u8>, ())> =
            self.backend.scan_prefix("outcome_signals", agent_prefix)?;

        // Decode query_ids from index keys
        let mut query_ids = Vec::new();
        for (key_bytes, _) in results {
            let (_, _, query_id) = Self::decode_agent_index_key(&key_bytes)?;
            query_ids.push(query_id);
        }

        // Load full traces
        let mut traces = Vec::new();
        for query_id in query_ids {
            if let Some(trace) = self.get(query_id)? {
                traces.push(trace);
            }

            // Stop if we've reached the limit
            if traces.len() >= limit {
                break;
            }
        }

        // Sort by timestamp (descending - most recent first)
        traces.sort_by(|a, b| {
            let a_ts = a.closed_at.unwrap_or(a.started_at);
            let b_ts = b.closed_at.unwrap_or(b.started_at);
            b_ts.cmp(&a_ts)
        });

        // Truncate to limit
        traces.truncate(limit);

        Ok(traces)
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

    #[test]
    fn test_start_and_get_trace() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbDecisionTraceStore::new(backend);

        store.start(
            "query_1".to_string(),
            100,
            1,
            vec![1, 2, 3],
            vec![10, 20],
        ).unwrap();

        let trace = store.get("query_1".to_string()).unwrap();
        assert!(trace.is_some());
        let trace = trace.unwrap();
        assert_eq!(trace.agent_id, 100);
        assert_eq!(trace.retrieved_memory_ids.len(), 3);
    }

    #[test]
    fn test_mark_memory_used() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbDecisionTraceStore::new(backend);

        store.start(
            "query_1".to_string(),
            100,
            1,
            vec![1, 2, 3],
            vec![],
        ).unwrap();

        store.mark_memory_used("query_1".to_string(), 2).unwrap();

        let trace = store.get("query_1".to_string()).unwrap().unwrap();
        assert_eq!(trace.used_memory_ids, vec![2]);
    }

    #[test]
    fn test_close_trace() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbDecisionTraceStore::new(backend);

        store.start(
            "query_1".to_string(),
            100,
            1,
            vec![1, 2],
            vec![10],
        ).unwrap();

        let outcome = OutcomeSignal {
            success: true,
            metadata: HashMap::new(),
        };
        store.close("query_1".to_string(), outcome).unwrap();

        let trace = store.get("query_1".to_string()).unwrap().unwrap();
        assert!(trace.outcome.is_some());
        assert!(trace.closed_at.is_some());
        assert!(trace.outcome.unwrap().success);
    }

    #[test]
    fn test_list_recent() {
        let (backend, _temp) = create_test_backend();
        let mut store = RedbDecisionTraceStore::new(backend);

        // Create multiple traces
        store.start("query_1".to_string(), 100, 1, vec![1], vec![]).unwrap();
        store.start("query_2".to_string(), 100, 1, vec![2], vec![]).unwrap();
        store.start("query_3".to_string(), 100, 1, vec![3], vec![]).unwrap();

        // List recent traces
        let traces = store.list_recent(100, 10).unwrap();
        assert_eq!(traces.len(), 3);
    }
}
