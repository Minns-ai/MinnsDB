//! Persistent storage for NER features using redb

use agent_db_core::types::EventId;
use agent_db_events::ExtractedFeatures;
use anyhow::{anyhow, Result};
use redb::{Database, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info, warn};

const NER_FEATURES_TABLE: TableDefinition<u128, &[u8]> = TableDefinition::new("ner_features");
const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Versioned wrapper for stored features to enable future migrations
#[derive(Debug, Serialize, Deserialize)]
struct VersionedFeatures {
    version: u32,
    features: ExtractedFeatures,
}

/// Storage for NER extracted features
pub struct NerFeatureStore {
    db: Database,
}

impl NerFeatureStore {
    /// Create a new NER feature store at the given path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Opening NER feature store at: {:?}", path.as_ref());

        let db = Database::create(path)?;

        // Initialize table
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(NER_FEATURES_TABLE)?;
        }
        write_txn.commit()?;

        info!("NER feature store initialized successfully");

        Ok(Self { db })
    }

    /// Store extracted features for an event
    pub fn store(&self, features: &ExtractedFeatures) -> Result<()> {
        debug!(
            "Storing NER features for event {} ({} entities)",
            features.event_id,
            features.entity_spans.len()
        );

        let versioned = VersionedFeatures {
            version: CURRENT_SCHEMA_VERSION,
            features: features.clone(),
        };

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(NER_FEATURES_TABLE)?;
            let serialized = bincode::serialize(&versioned)?;
            table.insert(features.event_id, serialized.as_slice())?;
        }
        write_txn.commit()?;

        debug!(
            "Successfully stored NER features for event {}",
            features.event_id
        );

        Ok(())
    }

    /// Get extracted features for an event
    pub fn get(&self, event_id: EventId) -> Result<Option<ExtractedFeatures>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NER_FEATURES_TABLE)?;

        if let Some(value) = table.get(event_id)? {
            let bytes = value.value();

            // Try to deserialize as versioned first, fall back to legacy format
            let features = match bincode::deserialize::<VersionedFeatures>(bytes) {
                Ok(versioned) => {
                    if versioned.version != CURRENT_SCHEMA_VERSION {
                        warn!(
                            "Loading features with schema version {} (current: {})",
                            versioned.version, CURRENT_SCHEMA_VERSION
                        );
                        // Future: Handle migrations here if needed
                    }
                    versioned.features
                },
                Err(_) => {
                    // Try legacy format without version wrapper
                    warn!(
                        "Loading legacy unversioned NER features for event {}",
                        event_id
                    );
                    bincode::deserialize::<ExtractedFeatures>(bytes)?
                },
            };

            debug!(
                "Retrieved NER features for event {} ({} entities)",
                event_id,
                features.entity_spans.len()
            );
            Ok(Some(features))
        } else {
            debug!("No NER features found for event {}", event_id);
            Ok(None)
        }
    }

    /// Check if features exist for an event (for idempotency)
    pub fn exists(&self, event_id: EventId) -> Result<bool> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NER_FEATURES_TABLE)?;
        Ok(table.get(event_id)?.is_some())
    }

    /// Delete features for an event
    pub fn delete(&self, event_id: EventId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(NER_FEATURES_TABLE)?;
            table.remove(event_id)?;
        }
        write_txn.commit()?;

        debug!("Deleted NER features for event {}", event_id);
        Ok(())
    }

    /// Get count of stored features
    pub fn count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NER_FEATURES_TABLE)?;
        let count = table.len()?;
        count.try_into().map_err(|e| {
            anyhow!(
                "Feature count {} exceeds usize::MAX on this platform: {}",
                count,
                e
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_events::EntitySpan;
    use tempfile::tempdir;

    fn create_test_features(event_id: EventId) -> ExtractedFeatures {
        let spans = vec![EntitySpan::new(
            "PERSON".to_string(),
            0,
            4,
            0.95,
            "John".to_string(),
        )];

        ExtractedFeatures::new(
            event_id,
            spans,
            "test-model".to_string(),
            "John works here",
            None,
        )
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");

        let store = NerFeatureStore::new(&db_path).unwrap();
        let features = create_test_features(1);

        // Store
        store.store(&features).unwrap();

        // Retrieve
        let retrieved = store.get(1).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.event_id, 1);
        assert_eq!(retrieved.entity_spans.len(), 1);
    }

    #[test]
    fn test_exists() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");

        let store = NerFeatureStore::new(&db_path).unwrap();
        let features = create_test_features(1);

        assert!(!store.exists(1).unwrap());

        store.store(&features).unwrap();

        assert!(store.exists(1).unwrap());
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");

        let store = NerFeatureStore::new(&db_path).unwrap();
        let features = create_test_features(1);

        store.store(&features).unwrap();
        assert!(store.exists(1).unwrap());

        store.delete(1).unwrap();
        assert!(!store.exists(1).unwrap());
    }

    #[test]
    fn test_count() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");

        let store = NerFeatureStore::new(&db_path).unwrap();

        assert_eq!(store.count().unwrap(), 0);

        store.store(&create_test_features(1)).unwrap();
        store.store(&create_test_features(2)).unwrap();

        assert_eq!(store.count().unwrap(), 2);
    }
}
