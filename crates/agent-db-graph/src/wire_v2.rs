//! Binary wire format v2 for streaming export/import.
//!
//! All integers are big-endian. See the format specification in the plan doc.
//!
//! ## Layout
//! ```text
//! Header (21 bytes):
//!   magic[4]               = b"EGDB"
//!   version[1]             = 0x02
//!   header_record_count[8] = 0 (advisory, always 0 in streaming mode)
//!   flags_reserved[8]      = 0 (must be 0 for v2 writes, ignored by v2 readers)
//!
//! Record (repeated):
//!   tag[1]         — record type
//!   key_len[4]     — u32 BE
//!   key[key_len]
//!   value_len[4]   — u32 BE
//!   value[value_len]
//!
//! Footer:
//!   tag[1]               = 0xFF
//!   total_record_count[8] = u64 BE
//!   checksum[32]         = SHA-256 of all bytes before footer tag
//! ```

use std::fmt;
use std::io::{self, Read, Write};

// ========== Constants ==========

pub const MAGIC: &[u8; 4] = b"EGDB";
pub const FORMAT_VERSION: u8 = 0x02;
pub const HEADER_LEN: usize = 21;
pub const FOOTER_TAG: u8 = 0xFF;

pub const MAX_KEY_LEN: u32 = 64 * 1024; // 64 KB
pub const MAX_VALUE_LEN: u32 = 256 * 1024 * 1024; // 256 MB

// Record type tags
pub const TAG_MEMORY: u8 = 0x01;
pub const TAG_STRATEGY: u8 = 0x02;
pub const TAG_GRAPH_NODE: u8 = 0x03;
pub const TAG_GRAPH_EDGE: u8 = 0x04;
pub const TAG_GRAPH_META: u8 = 0x05;
pub const TAG_TRANSITION_MODEL: u8 = 0x06;
pub const TAG_EPISODE_DETECTOR: u8 = 0x07;
pub const TAG_ID_ALLOCATOR: u8 = 0x08;

// ========== Error ==========

#[derive(Debug)]
pub enum WireError {
    Io(io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u8),
    UnknownTag(u8),
    KeyTooLarge(u32),
    ValueTooLarge(u32),
    ChecksumMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    RecordCountMismatch {
        expected: u64,
        actual: u64,
    },
    DuplicateSingleton(u8),
    UnexpectedEof,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::Io(e) => write!(f, "IO error: {}", e),
            WireError::BadMagic(m) => write!(f, "Bad magic: {:?}", m),
            WireError::UnsupportedVersion(v) => write!(f, "Unsupported format version: {}", v),
            WireError::UnknownTag(t) => write!(f, "Unknown record tag: 0x{:02X}", t),
            WireError::KeyTooLarge(len) => write!(f, "Key too large: {} bytes", len),
            WireError::ValueTooLarge(len) => write!(f, "Value too large: {} bytes", len),
            WireError::ChecksumMismatch { .. } => write!(f, "Checksum mismatch"),
            WireError::RecordCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Record count mismatch: expected {}, got {}",
                    expected, actual
                )
            },
            WireError::DuplicateSingleton(t) => write!(f, "Duplicate singleton tag: 0x{:02X}", t),
            WireError::UnexpectedEof => write!(f, "Unexpected end of stream"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<io::Error> for WireError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            WireError::UnexpectedEof
        } else {
            WireError::Io(e)
        }
    }
}

// ========== Encoding ==========

/// Write the 21-byte header.
pub fn write_header<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&[FORMAT_VERSION])?;
    w.write_all(&0u64.to_be_bytes())?; // header_record_count (advisory)
    w.write_all(&0u64.to_be_bytes())?; // flags_reserved
    Ok(())
}

/// Write one record: tag + key_len + key + value_len + value.
pub fn write_record<W: Write>(w: &mut W, tag: u8, key: &[u8], value: &[u8]) -> io::Result<()> {
    debug_assert!(tag != FOOTER_TAG, "Use write_footer for footer tag");
    w.write_all(&[tag])?;
    w.write_all(&(key.len() as u32).to_be_bytes())?;
    w.write_all(key)?;
    w.write_all(&(value.len() as u32).to_be_bytes())?;
    w.write_all(value)?;
    Ok(())
}

/// Write the footer: tag(0xFF) + record_count(u64 BE) + checksum(32 bytes).
pub fn write_footer<W: Write>(w: &mut W, record_count: u64, checksum: &[u8; 32]) -> io::Result<()> {
    w.write_all(&[FOOTER_TAG])?;
    w.write_all(&record_count.to_be_bytes())?;
    w.write_all(checksum)?;
    Ok(())
}

// ========== Decoding ==========

/// Read and validate the 21-byte header.
pub fn read_header<R: Read>(r: &mut R) -> Result<(), WireError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(WireError::BadMagic(magic));
    }

    let mut version = [0u8; 1];
    r.read_exact(&mut version)?;
    if version[0] != FORMAT_VERSION {
        return Err(WireError::UnsupportedVersion(version[0]));
    }

    // Skip advisory record count (8 bytes) and reserved flags (8 bytes)
    let mut skip = [0u8; 16];
    r.read_exact(&mut skip)?;

    Ok(())
}

/// Read one record tag byte. Returns the tag value.
/// If the tag is `FOOTER_TAG` (0xFF), the caller should use `read_footer` next.
pub fn read_record_tag<R: Read>(r: &mut R) -> Result<u8, WireError> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    let t = tag[0];

    // Validate known tags
    if t != FOOTER_TAG {
        match t {
            TAG_MEMORY | TAG_STRATEGY | TAG_GRAPH_NODE | TAG_GRAPH_EDGE | TAG_GRAPH_META
            | TAG_TRANSITION_MODEL | TAG_EPISODE_DETECTOR | TAG_ID_ALLOCATOR => {},
            _ => return Err(WireError::UnknownTag(t)),
        }
    }

    Ok(t)
}

/// Read the record body (key_len + key + value_len + value) after the tag.
/// Returns `(key, value)`.
pub fn read_record_body<R: Read>(r: &mut R) -> Result<(Vec<u8>, Vec<u8>), WireError> {
    // key_len
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let key_len = u32::from_be_bytes(len_buf);
    if key_len > MAX_KEY_LEN {
        return Err(WireError::KeyTooLarge(key_len));
    }

    // key
    let mut key = vec![0u8; key_len as usize];
    r.read_exact(&mut key)?;

    // value_len
    r.read_exact(&mut len_buf)?;
    let value_len = u32::from_be_bytes(len_buf);
    if value_len > MAX_VALUE_LEN {
        return Err(WireError::ValueTooLarge(value_len));
    }

    // value
    let mut value = vec![0u8; value_len as usize];
    r.read_exact(&mut value)?;

    Ok((key, value))
}

/// Read the footer body after the 0xFF tag has been consumed.
/// Returns `(record_count, checksum)`.
pub fn read_footer<R: Read>(r: &mut R) -> Result<(u64, [u8; 32]), WireError> {
    let mut count_buf = [0u8; 8];
    r.read_exact(&mut count_buf)?;
    let record_count = u64::from_be_bytes(count_buf);

    let mut checksum = [0u8; 32];
    r.read_exact(&mut checksum)?;

    Ok((record_count, checksum))
}

/// Returns `true` if the tag is a singleton type (only one allowed per export).
pub fn is_singleton_tag(tag: u8) -> bool {
    matches!(
        tag,
        TAG_GRAPH_META | TAG_TRANSITION_MODEL | TAG_EPISODE_DETECTOR | TAG_ID_ALLOCATOR
    )
}

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_LEN);
        assert_eq!(&buf[0..4], MAGIC);
        assert_eq!(buf[4], FORMAT_VERSION);

        let mut cursor = io::Cursor::new(&buf);
        read_header(&mut cursor).unwrap();
        assert_eq!(cursor.position() as usize, HEADER_LEN);
    }

    #[test]
    fn test_record_roundtrip_various_sizes() {
        for (key, value) in [
            (b"".as_slice(), b"".as_slice()),
            (b"k".as_slice(), b"v".as_slice()),
            (&[0xAB; 1024], &[0xCD; 8192]),
        ] {
            let mut buf = Vec::new();
            write_record(&mut buf, TAG_MEMORY, key, value).unwrap();

            let mut cursor = io::Cursor::new(&buf);
            let tag = read_record_tag(&mut cursor).unwrap();
            assert_eq!(tag, TAG_MEMORY);
            let (k, v) = read_record_body(&mut cursor).unwrap();
            assert_eq!(k, key);
            assert_eq!(v, value);
        }
    }

    #[test]
    fn test_footer_roundtrip() {
        let checksum = [0x42u8; 32];
        let mut buf = Vec::new();
        write_footer(&mut buf, 12345, &checksum).unwrap();

        // footer = 1 (tag) + 8 (count) + 32 (checksum) = 41 bytes
        assert_eq!(buf.len(), 41);
        assert_eq!(buf[0], FOOTER_TAG);

        let mut cursor = io::Cursor::new(&buf);
        let tag = read_record_tag(&mut cursor).unwrap();
        assert_eq!(tag, FOOTER_TAG);
        let (count, cs) = read_footer(&mut cursor).unwrap();
        assert_eq!(count, 12345);
        assert_eq!(cs, checksum);
    }

    #[test]
    fn test_reject_key_too_large() {
        let key_len = MAX_KEY_LEN + 1;
        let mut buf = Vec::new();
        buf.push(TAG_MEMORY);
        buf.extend_from_slice(&key_len.to_be_bytes());
        // Don't need actual key bytes — read_record_body should reject before reading

        let mut cursor = io::Cursor::new(&buf);
        let tag = read_record_tag(&mut cursor).unwrap();
        assert_eq!(tag, TAG_MEMORY);
        match read_record_body(&mut cursor) {
            Err(WireError::KeyTooLarge(len)) => assert_eq!(len, key_len),
            other => panic!("Expected KeyTooLarge, got {:?}", other),
        }
    }

    #[test]
    fn test_reject_value_too_large() {
        let val_len = MAX_VALUE_LEN + 1;
        let mut buf = Vec::new();
        buf.push(TAG_MEMORY);
        buf.extend_from_slice(&0u32.to_be_bytes()); // key_len = 0
        buf.extend_from_slice(&val_len.to_be_bytes());

        let mut cursor = io::Cursor::new(&buf);
        let _ = read_record_tag(&mut cursor).unwrap();
        match read_record_body(&mut cursor) {
            Err(WireError::ValueTooLarge(len)) => assert_eq!(len, val_len),
            other => panic!("Expected ValueTooLarge, got {:?}", other),
        }
    }

    #[test]
    fn test_reject_unknown_tag() {
        let buf = [0xFE]; // unknown tag
        let mut cursor = io::Cursor::new(&buf);
        match read_record_tag(&mut cursor) {
            Err(WireError::UnknownTag(0xFE)) => {},
            other => panic!("Expected UnknownTag(0xFE), got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_header() {
        // Only 3 bytes of magic
        let buf = b"EGD";
        let mut cursor = io::Cursor::new(buf.as_slice());
        match read_header(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_key_len() {
        let mut buf = Vec::new();
        buf.push(TAG_MEMORY);
        buf.extend_from_slice(&[0x00, 0x00]); // only 2 of 4 bytes for key_len

        let mut cursor = io::Cursor::new(&buf);
        let _ = read_record_tag(&mut cursor).unwrap();
        match read_record_body(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_key() {
        let mut buf = Vec::new();
        buf.push(TAG_MEMORY);
        buf.extend_from_slice(&4u32.to_be_bytes()); // key_len = 4
        buf.extend_from_slice(&[0xAA, 0xBB]); // only 2 of 4 key bytes

        let mut cursor = io::Cursor::new(&buf);
        let _ = read_record_tag(&mut cursor).unwrap();
        match read_record_body(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_value_len() {
        let mut buf = Vec::new();
        buf.push(TAG_STRATEGY);
        buf.extend_from_slice(&0u32.to_be_bytes()); // key_len = 0
        buf.extend_from_slice(&[0x00]); // only 1 of 4 bytes for value_len

        let mut cursor = io::Cursor::new(&buf);
        let _ = read_record_tag(&mut cursor).unwrap();
        match read_record_body(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_value() {
        let mut buf = Vec::new();
        buf.push(TAG_GRAPH_NODE);
        buf.extend_from_slice(&0u32.to_be_bytes()); // key_len = 0
        buf.extend_from_slice(&10u32.to_be_bytes()); // value_len = 10
        buf.extend_from_slice(&[0xCC; 5]); // only 5 of 10 value bytes

        let mut cursor = io::Cursor::new(&buf);
        let _ = read_record_tag(&mut cursor).unwrap();
        match read_record_body(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_footer_count() {
        let mut buf = Vec::new();
        buf.push(FOOTER_TAG);
        buf.extend_from_slice(&[0x00; 4]); // only 4 of 8 bytes for count

        let mut cursor = io::Cursor::new(&buf);
        let tag = read_record_tag(&mut cursor).unwrap();
        assert_eq!(tag, FOOTER_TAG);
        match read_footer(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_truncation_mid_footer_checksum() {
        let mut buf = Vec::new();
        buf.push(FOOTER_TAG);
        buf.extend_from_slice(&100u64.to_be_bytes()); // full count
        buf.extend_from_slice(&[0xAA; 16]); // only 16 of 32 bytes for checksum

        let mut cursor = io::Cursor::new(&buf);
        let tag = read_record_tag(&mut cursor).unwrap();
        assert_eq!(tag, FOOTER_TAG);
        match read_footer(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_stream() {
        let buf: &[u8] = &[];
        let mut cursor = io::Cursor::new(buf);
        match read_header(&mut cursor) {
            Err(WireError::UnexpectedEof) => {},
            other => panic!("Expected UnexpectedEof, got {:?}", other),
        }
    }

    #[test]
    fn test_bad_magic() {
        let buf = b"XYZW\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = io::Cursor::new(buf.as_slice());
        match read_header(&mut cursor) {
            Err(WireError::BadMagic(m)) => assert_eq!(&m, b"XYZW"),
            other => panic!("Expected BadMagic, got {:?}", other),
        }
    }

    #[test]
    fn test_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(0x99); // unsupported version
        buf.extend_from_slice(&[0u8; 16]);

        let mut cursor = io::Cursor::new(&buf);
        match read_header(&mut cursor) {
            Err(WireError::UnsupportedVersion(0x99)) => {},
            other => panic!("Expected UnsupportedVersion(0x99), got {:?}", other),
        }
    }

    #[test]
    fn test_all_record_tags() {
        let tags = [
            TAG_MEMORY,
            TAG_STRATEGY,
            TAG_GRAPH_NODE,
            TAG_GRAPH_EDGE,
            TAG_GRAPH_META,
            TAG_TRANSITION_MODEL,
            TAG_EPISODE_DETECTOR,
            TAG_ID_ALLOCATOR,
        ];
        for &tag in &tags {
            let mut buf = Vec::new();
            write_record(&mut buf, tag, b"k", b"v").unwrap();
            let mut cursor = io::Cursor::new(&buf);
            let read_tag = read_record_tag(&mut cursor).unwrap();
            assert_eq!(read_tag, tag);
            let (k, v) = read_record_body(&mut cursor).unwrap();
            assert_eq!(k, b"k");
            assert_eq!(v, b"v");
        }
    }

    #[test]
    fn test_is_singleton_tag() {
        assert!(!is_singleton_tag(TAG_MEMORY));
        assert!(!is_singleton_tag(TAG_STRATEGY));
        assert!(!is_singleton_tag(TAG_GRAPH_NODE));
        assert!(!is_singleton_tag(TAG_GRAPH_EDGE));
        assert!(is_singleton_tag(TAG_GRAPH_META));
        assert!(is_singleton_tag(TAG_TRANSITION_MODEL));
        assert!(is_singleton_tag(TAG_EPISODE_DETECTOR));
        assert!(is_singleton_tag(TAG_ID_ALLOCATOR));
    }
}
