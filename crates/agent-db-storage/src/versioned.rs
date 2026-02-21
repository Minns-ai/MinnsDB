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
pub const CURRENT_DATA_VERSION: u8 = 1;

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

/// Serialize `T` → versioned msgpack bytes.
///
/// Equivalent to `wrap_versioned(CURRENT_DATA_VERSION, &rmp_serde::to_vec(value)?)`.
pub fn serialize_versioned<T: Serialize>(value: &T) -> Result<Vec<u8>, StorageError> {
    let payload =
        rmp_serde::to_vec(value).map_err(|e| StorageError::Serialization(e.to_string()))?;
    Ok(wrap_versioned(CURRENT_DATA_VERSION, &payload))
}

/// Deserialize versioned-or-legacy bytes → `T`.
///
/// Transparently handles both legacy (no envelope, version 0) and
/// versioned (envelope present) data, providing backward compatibility.
pub fn deserialize_versioned<T: DeserializeOwned>(data: &[u8]) -> Result<T, StorageError> {
    let (_version, payload) = unwrap_versioned(data);
    // For now, all versions (0 and 1) use the same msgpack format.
    // Future versions can branch here:
    //   match version { 0 | 1 => ..., 2 => migrate(...), _ => error }
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
}
