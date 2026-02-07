//! Episode Catalog - Persistent join spine for events → episodes → memories/strategies
//!
//! This module provides the EpisodeCatalog trait and its redb-backed implementation.
//! The catalog is the foundational data structure that enables persistent joins between
//! events, episodes, and learning artifacts (memories/strategies).

use agent_db_core::types::{AgentId, ContextHash, EventId, SessionId, Timestamp};
use agent_db_storage::{RedbBackend, StorageError, StorageResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::episodes::{EpisodeId, EpisodeOutcome};

/// Episode record stored in the catalog
///
/// This is a persistent summary of an episode, optimized for joins and queries.
/// It contains just enough information to enable retrieval without loading full events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Unique episode identifier
    pub id: EpisodeId,

    /// Episode version (incremented on late corrections)
    pub version: u32,

    /// Agent that experienced this episode
    pub agent_id: AgentId,

    /// Session identifier
    pub session_id: SessionId,

    /// Episode start timestamp
    pub start_timestamp: Timestamp,

    /// Episode end timestamp (None if still active)
    pub end_timestamp: Option<Timestamp>,

    /// Overall outcome of the episode
    pub outcome: Option<EpisodeOutcome>,

    /// Significance score (0.0 to 1.0)
    pub significance: f32,

    /// Context signature for this episode
    pub context_hash: ContextHash,

    /// Goal bucket ID for semantic sharding
    pub goal_bucket_id: u64,

    /// Behavior signature for pattern matching
    pub behavior_signature: String,

    /// All event IDs in this episode
    pub event_ids: Vec<EventId>,
}

/// Episode catalog trait - the join spine for persistent learning
///
/// This trait provides the contract for storing and retrieving episode metadata.
/// It enables efficient joins between events, episodes, and learning artifacts.
pub trait EpisodeCatalog: Send + Sync {
    /// Store an episode record (upsert by episode_id + version)
    fn put_episode(
        &mut self,
        episode_id: EpisodeId,
        version: u32,
        record: EpisodeRecord,
    ) -> StorageResult<()>;

    /// Get episode by ID (latest version or specific version)
    fn get_episode(
        &self,
        episode_id: EpisodeId,
        version: Option<u32>,
    ) -> StorageResult<Option<EpisodeRecord>>;

    /// Get episode that contains a specific event (for join queries)
    fn get_episode_by_event(&self, event_id: EventId) -> StorageResult<Option<EpisodeRecord>>;

    /// List recent episodes for an agent (for debugging + API)
    fn list_recent(
        &self,
        agent_id: AgentId,
        time_range: (Timestamp, Timestamp),
    ) -> StorageResult<Vec<EpisodeRecord>>;
}

/// Redb-backed episode catalog implementation
///
/// Uses the following tables:
/// - episode_catalog: (episode_id, version) → EpisodeRecord
/// - episode_by_event: event_id → episode_id  (reverse index)
/// - episode_by_agent: (agent_id, timestamp) → episode_id  (range queries)
pub struct RedbEpisodeCatalog {
    backend: Arc<RedbBackend>,
}

impl RedbEpisodeCatalog {
    /// Create a new redb episode catalog
    pub fn new(backend: Arc<RedbBackend>) -> Self {
        Self { backend }
    }

    /// Encode episode key: (episode_id, version) → bytes
    fn encode_episode_key(episode_id: EpisodeId, version: u32) -> Vec<u8> {
        let mut key = Vec::with_capacity(12);
        key.extend_from_slice(&episode_id.to_be_bytes());
        key.extend_from_slice(&version.to_be_bytes());
        key
    }

    /// Decode episode key: bytes → (episode_id, version)
    fn decode_episode_key(key: &[u8]) -> StorageResult<(EpisodeId, u32)> {
        if key.len() != 12 {
            return Err(StorageError::DatabaseError(format!(
                "Invalid episode key length: {}",
                key.len()
            )));
        }

        let episode_id = u64::from_be_bytes(key[0..8].try_into().unwrap());
        let version = u32::from_be_bytes(key[8..12].try_into().unwrap());
        Ok((episode_id, version))
    }

    /// Encode agent key for range queries: (agent_id, timestamp) → bytes
    fn encode_agent_key(agent_id: AgentId, timestamp: Timestamp) -> Vec<u8> {
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&agent_id.to_be_bytes());
        key.extend_from_slice(&timestamp.to_be_bytes());
        key
    }
}

impl EpisodeCatalog for RedbEpisodeCatalog {
    fn put_episode(
        &mut self,
        episode_id: EpisodeId,
        version: u32,
        record: EpisodeRecord,
    ) -> StorageResult<()> {
        // Store main record
        let key = Self::encode_episode_key(episode_id, version);
        self.backend.put("episode_catalog", key, &record)?;

        // Update reverse index: event_id → episode_id
        for &event_id in &record.event_ids {
            self.backend.put(
                "partition_map", // Reusing partition_map as event→episode index
                event_id.to_be_bytes(),
                &episode_id,
            )?;
        }

        // Update agent index for range queries: (agent_id, timestamp) → episode_id
        if let Some(end_ts) = record.end_timestamp {
            let agent_key = Self::encode_agent_key(record.agent_id, end_ts);
            self.backend.put(
                "memory_records", // Reusing memory_records for agent→episode index
                agent_key,
                &episode_id,
            )?;
        }

        Ok(())
    }

    fn get_episode(
        &self,
        episode_id: EpisodeId,
        version: Option<u32>,
    ) -> StorageResult<Option<EpisodeRecord>> {
        match version {
            Some(v) => {
                // Get specific version
                let key = Self::encode_episode_key(episode_id, v);
                self.backend.get("episode_catalog", key)
            },
            None => {
                // Get latest version by scanning all versions of this episode
                let prefix = episode_id.to_be_bytes().to_vec();
                let results: Vec<(Vec<u8>, EpisodeRecord)> =
                    self.backend.scan_prefix("episode_catalog", prefix)?;

                if results.is_empty() {
                    return Ok(None);
                }

                // Find the highest version
                let mut latest: Option<(u32, EpisodeRecord)> = None;
                for (key_bytes, record) in results {
                    let (_, ver) = Self::decode_episode_key(&key_bytes)?;
                    match latest {
                        None => latest = Some((ver, record)),
                        Some((max_ver, _)) if ver > max_ver => latest = Some((ver, record)),
                        _ => {},
                    }
                }

                Ok(latest.map(|(_, record)| record))
            },
        }
    }

    fn get_episode_by_event(&self, event_id: EventId) -> StorageResult<Option<EpisodeRecord>> {
        // Look up episode_id from reverse index
        let episode_id: Option<EpisodeId> = self
            .backend
            .get("partition_map", event_id.to_be_bytes())?;

        match episode_id {
            Some(ep_id) => self.get_episode(ep_id, None),
            None => Ok(None),
        }
    }

    fn list_recent(
        &self,
        agent_id: AgentId,
        time_range: (Timestamp, Timestamp),
    ) -> StorageResult<Vec<EpisodeRecord>> {
        // Use agent_id as prefix to get all episodes for this agent
        let agent_prefix = agent_id.to_be_bytes().to_vec();

        // Get all episode IDs for this agent
        let results: Vec<(Vec<u8>, EpisodeId)> =
            self.backend.scan_prefix("memory_records", agent_prefix)?;

        let mut episodes = Vec::new();
        for (key_bytes, episode_id) in results {
            // Decode the key to get timestamp
            if key_bytes.len() < 16 {
                continue;
            }
            let timestamp = u64::from_be_bytes(key_bytes[8..16].try_into().unwrap());

            // Filter by time range
            if timestamp < time_range.0 || timestamp > time_range.1 {
                continue;
            }

            // Load full episode record
            if let Some(record) = self.get_episode(episode_id, None)? {
                episodes.push(record);
            }
        }

        // Sort by timestamp (descending - most recent first)
        episodes.sort_by(|a, b| {
            let a_ts = a.end_timestamp.unwrap_or(a.start_timestamp);
            let b_ts = b.end_timestamp.unwrap_or(b.start_timestamp);
            b_ts.cmp(&a_ts)
        });

        Ok(episodes)
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

    fn create_test_record(id: EpisodeId, agent_id: AgentId) -> EpisodeRecord {
        EpisodeRecord {
            id,
            version: 1,
            agent_id,
            session_id: 1,
            start_timestamp: 1000,
            end_timestamp: Some(2000),
            outcome: Some(EpisodeOutcome::Success),
            significance: 0.8,
            context_hash: 12345,
            goal_bucket_id: 1,
            behavior_signature: "test_pattern".to_string(),
            event_ids: vec![1, 2, 3],
        }
    }

    #[test]
    fn test_put_and_get_episode() {
        let (backend, _temp) = create_test_backend();
        let mut catalog = RedbEpisodeCatalog::new(backend);

        let record = create_test_record(1, 100);
        catalog.put_episode(1, 1, record.clone()).unwrap();

        let retrieved = catalog.get_episode(1, Some(1)).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, 1);
        assert_eq!(retrieved.agent_id, 100);
    }

    #[test]
    fn test_get_latest_version() {
        let (backend, _temp) = create_test_backend();
        let mut catalog = RedbEpisodeCatalog::new(backend);

        // Store version 1
        let mut record_v1 = create_test_record(1, 100);
        record_v1.significance = 0.5;
        catalog.put_episode(1, 1, record_v1).unwrap();

        // Store version 2
        let mut record_v2 = create_test_record(1, 100);
        record_v2.version = 2;
        record_v2.significance = 0.9;
        catalog.put_episode(1, 2, record_v2).unwrap();

        // Get latest should return version 2
        let latest = catalog.get_episode(1, None).unwrap().unwrap();
        assert_eq!(latest.version, 2);
        assert_eq!(latest.significance, 0.9);
    }

    #[test]
    fn test_get_episode_by_event() {
        let (backend, _temp) = create_test_backend();
        let mut catalog = RedbEpisodeCatalog::new(backend);

        let record = create_test_record(1, 100);
        catalog.put_episode(1, 1, record.clone()).unwrap();

        // Look up by event ID
        let episode = catalog.get_episode_by_event(2).unwrap();
        assert!(episode.is_some());
        assert_eq!(episode.unwrap().id, 1);
    }

    #[test]
    fn test_list_recent() {
        let (backend, _temp) = create_test_backend();
        let mut catalog = RedbEpisodeCatalog::new(backend);

        // Store multiple episodes for same agent
        let mut record1 = create_test_record(1, 100);
        record1.start_timestamp = 1000;
        record1.end_timestamp = Some(2000);
        catalog.put_episode(1, 1, record1).unwrap();

        let mut record2 = create_test_record(2, 100);
        record2.start_timestamp = 3000;
        record2.end_timestamp = Some(4000);
        catalog.put_episode(2, 1, record2).unwrap();

        // List recent episodes
        let episodes = catalog.list_recent(100, (1000, 5000)).unwrap();
        assert_eq!(episodes.len(), 2);
        // Should be sorted by timestamp descending
        assert_eq!(episodes[0].id, 2);
        assert_eq!(episodes[1].id, 1);
    }
}
