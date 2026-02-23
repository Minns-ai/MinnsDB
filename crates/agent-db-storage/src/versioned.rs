//! Envelope-based version tagging for all serialized data.
//!
//! Format: `[0x00 magic][version: u8][msgpack payload...]`
//!
//! `0x00` is never a valid first byte for msgpack map/array/string/bin,
//! so legacy (unversioned) data is always distinguishable from versioned data.
//! Version 0 = legacy (no envelope), version 1 = first versioned format.

use crate::StorageError;
use serde::{de::DeserializeOwned, Serialize};

/// Current data envelope version written by this software.
/// - v1: compact (array) msgpack via `rmp_serde::to_vec` (broken for `serde_json::Value` fields)
/// - v2: named (map) msgpack via `rmp_serde::to_vec_named` (supports `deserialize_any`)
pub const CURRENT_DATA_VERSION: u8 = 2;

/// Magic byte that prefixes all versioned payloads.
/// 0x00 is never a valid first byte for msgpack map, array, string, or bin
/// (those start with 0x80+ for fixmap, 0x90+ for fixarray, etc.).
pub const VERSION_MAGIC: u8 = 0x00;

/// Wrap a raw payload with a version envelope.
///
/// Output: `[0x00][version][payload...]`
pub fn wrap_versioned(version: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + payload.len());
    out.push(VERSION_MAGIC);
    out.push(version);
    out.extend_from_slice(payload);
    out
}

/// Unwrap a potentially-versioned byte slice.
///
/// Returns `(version, payload)`:
/// - If `data` starts with `VERSION_MAGIC` → `(data[1], &data[2..])`
/// - Otherwise → `(0, data)` (legacy unversioned data)
pub fn unwrap_versioned(data: &[u8]) -> (u8, &[u8]) {
    if data.len() >= 2 && data[0] == VERSION_MAGIC {
        (data[1], &data[2..])
    } else {
        (0, data)
    }
}

/// Serialize `T` → versioned msgpack bytes (named/map format).
///
/// Uses `rmp_serde::to_vec_named` so struct fields are keyed by name,
/// which allows `deserialize_any` (required by `serde_json::Value`).
pub fn serialize_versioned<T: Serialize>(value: &T) -> Result<Vec<u8>, StorageError> {
    let payload =
        rmp_serde::to_vec_named(value).map_err(|e| StorageError::Serialization(e.to_string()))?;
    Ok(wrap_versioned(CURRENT_DATA_VERSION, &payload))
}

/// Deserialize versioned-or-legacy bytes → `T`.
///
/// Transparently handles both legacy (no envelope, version 0) and
/// versioned (envelope present) data, providing backward compatibility.
pub fn deserialize_versioned<T: DeserializeOwned>(data: &[u8]) -> Result<T, StorageError> {
    let (_version, payload) = unwrap_versioned(data);
    // `rmp_serde::from_slice` auto-detects both compact (v0/v1) and named (v2) format.
    // v0/v1 compact data still deserializes for types without `deserialize_any` fields.
    // v2 named data additionally supports `serde_json::Value` and similar types.
    rmp_serde::from_slice(payload).map_err(|e| StorageError::Deserialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestData {
        id: u64,
        name: String,
    }

    #[test]
    fn test_roundtrip_versioned() {
        let data = TestData {
            id: 42,
            name: "hello".to_string(),
        };
        let bytes = serialize_versioned(&data).unwrap();
        assert_eq!(bytes[0], VERSION_MAGIC);
        assert_eq!(bytes[1], CURRENT_DATA_VERSION);

        let restored: TestData = deserialize_versioned(&bytes).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_legacy_transparent_read() {
        let data = TestData {
            id: 7,
            name: "legacy".to_string(),
        };
        // Simulate legacy data: raw msgpack with no envelope
        let legacy_bytes = rmp_serde::to_vec(&data).unwrap();
        assert_ne!(legacy_bytes[0], VERSION_MAGIC); // sanity check

        let restored: TestData = deserialize_versioned(&legacy_bytes).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_unwrap_versioned_envelope() {
        let payload = b"hello";
        let wrapped = wrap_versioned(3, payload);
        let (version, unwrapped) = unwrap_versioned(&wrapped);
        assert_eq!(version, 3);
        assert_eq!(unwrapped, payload);
    }

    #[test]
    fn test_unwrap_versioned_legacy() {
        let data = b"\x82\xa2id\x01\xa4name\xa3foo"; // some msgpack-ish bytes
        let (version, payload) = unwrap_versioned(data);
        assert_eq!(version, 0);
        assert_eq!(payload, data);
    }

    #[test]
    fn test_empty_data() {
        let (version, payload) = unwrap_versioned(&[]);
        assert_eq!(version, 0);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_single_byte() {
        let (version, payload) = unwrap_versioned(&[0x00]);
        // Only 1 byte, not enough for envelope (need >= 2)
        assert_eq!(version, 0);
        assert_eq!(payload, &[0x00]);
    }

    #[test]
    fn test_roundtrip_json_value_fields() {
        use serde_json::json;
        use std::collections::HashMap;

        // Struct with serde_json::Value — the exact pattern that was broken with compact format
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct WithJsonValue {
            id: u64,
            data: HashMap<String, serde_json::Value>,
        }

        let val = WithJsonValue {
            id: 1,
            data: HashMap::from([
                ("key".into(), json!({"nested": true})),
                ("num".into(), json!(42)),
                ("arr".into(), json!([1, "two", null])),
            ]),
        };
        let bytes = serialize_versioned(&val).unwrap();
        assert_eq!(bytes[0], VERSION_MAGIC);
        assert_eq!(bytes[1], CURRENT_DATA_VERSION);

        let restored: WithJsonValue = deserialize_versioned(&bytes).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn test_v1_compact_data_still_reads() {
        // Simulate v1 data written with to_vec (compact format)
        let data = TestData {
            id: 42,
            name: "v1".into(),
        };
        let compact_payload = rmp_serde::to_vec(&data).unwrap();
        let v1_bytes = wrap_versioned(1, &compact_payload);

        let restored: TestData = deserialize_versioned(&v1_bytes).unwrap();
        assert_eq!(data, restored);
    }
}
