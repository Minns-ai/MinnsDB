//! Schema version tracking for the redb database.
//!
//! Stamps fresh databases with the current schema version on first open,
//! and rejects incompatible databases on subsequent opens.

use crate::{RedbBackend, StorageError, StorageResult};
use serde::{Deserialize, Serialize};

/// Current schema major version.
/// Bump this when a breaking change is made (table layout, key encoding, etc.).
pub const SCHEMA_MAJOR: u16 = 1;

/// Current schema minor version.
/// Bump this for backward-compatible additions (new tables, new optional fields).
pub const SCHEMA_MINOR: u16 = 0;

/// Key used to store the schema version in the `schema_versions` table.
const SCHEMA_KEY: &[u8] = b"__schema__";

/// Persisted schema version record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
    /// Timestamp (seconds since epoch) when this version was written.
    pub written_at: u64,
    /// Software version string that wrote this schema (e.g. "0.1.0").
    pub software_version: String,
}

/// Errors specific to schema version checks.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Database requires migration: on-disk major={on_disk}, current={current}")]
    NeedsMigration { on_disk: u16, current: u16 },

    #[error("Database was created by a newer version: on-disk major={on_disk}, current={current}")]
    TooNew { on_disk: u16, current: u16 },

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
}

fn current_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_software_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Stamp a fresh database with the current schema version.
pub fn stamp_schema_version(backend: &RedbBackend) -> StorageResult<()> {
    let version = SchemaVersion {
        major: SCHEMA_MAJOR,
        minor: SCHEMA_MINOR,
        written_at: current_epoch_secs(),
        software_version: current_software_version(),
    };
    backend.put(
        crate::redb_backend::table_names::SCHEMA_VERSIONS,
        SCHEMA_KEY,
        &version,
    )
}

/// Check the schema version stored in the database.
///
/// - Fresh database (no version stamped) → stamp with current version, return Ok.
/// - Same major, same or older minor → update stamp, return Ok.
/// - Older major → return `SchemaError::NeedsMigration`.
/// - Newer major → return `SchemaError::TooNew`.
pub fn check_schema_version(backend: &RedbBackend) -> Result<(), SchemaError> {
    let existing: Option<SchemaVersion> = backend.get(
        crate::redb_backend::table_names::SCHEMA_VERSIONS,
        SCHEMA_KEY,
    )?;

    match existing {
        None => {
            // Fresh database — stamp it
            stamp_schema_version(backend)?;
            tracing::info!(
                "Fresh database stamped with schema v{}.{}",
                SCHEMA_MAJOR,
                SCHEMA_MINOR
            );
            Ok(())
        },
        Some(on_disk) => {
            if on_disk.major == SCHEMA_MAJOR {
                // Compatible — update stamp if minor has advanced
                if on_disk.minor < SCHEMA_MINOR {
                    tracing::info!(
                        "Upgrading schema stamp: v{}.{} → v{}.{}",
                        on_disk.major,
                        on_disk.minor,
                        SCHEMA_MAJOR,
                        SCHEMA_MINOR
                    );
                    stamp_schema_version(backend)?;
                }
                Ok(())
            } else if on_disk.major < SCHEMA_MAJOR {
                Err(SchemaError::NeedsMigration {
                    on_disk: on_disk.major,
                    current: SCHEMA_MAJOR,
                })
            } else {
                Err(SchemaError::TooNew {
                    on_disk: on_disk.major,
                    current: SCHEMA_MAJOR,
                })
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RedbConfig;
    use tempfile::TempDir;

    #[test]
    fn test_fresh_database_gets_stamped() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Should stamp without error
        check_schema_version(&backend).unwrap();

        // Should be readable
        let version: SchemaVersion = backend
            .get(
                crate::redb_backend::table_names::SCHEMA_VERSIONS,
                SCHEMA_KEY,
            )
            .unwrap()
            .unwrap();
        assert_eq!(version.major, SCHEMA_MAJOR);
        assert_eq!(version.minor, SCHEMA_MINOR);
    }

    #[test]
    fn test_same_version_ok() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Stamp once
        check_schema_version(&backend).unwrap();
        // Check again — should be fine
        check_schema_version(&backend).unwrap();
    }

    #[test]
    fn test_newer_major_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let config = RedbConfig {
            data_path: temp_dir.path().join("test.redb"),
            ..Default::default()
        };
        let backend = RedbBackend::open(config).unwrap();

        // Write a future major version
        let future = SchemaVersion {
            major: SCHEMA_MAJOR + 1,
            minor: 0,
            written_at: 0,
            software_version: "future".to_string(),
        };
        backend
            .put(
                crate::redb_backend::table_names::SCHEMA_VERSIONS,
                SCHEMA_KEY,
                &future,
            )
            .unwrap();

        match check_schema_version(&backend) {
            Err(SchemaError::TooNew { .. }) => {}, // expected
            other => panic!("Expected TooNew, got {:?}", other),
        }
    }
}
