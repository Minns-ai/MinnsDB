//! EventTime: sequence-disambiguated temporal ordering.
//!
//! Combines a nanosecond timestamp with a monotonic sequence counter
//! to provide total ordering even when events arrive at the same
//! nanosecond.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::{AtomicU32, Ordering};

/// Global atomic sequence counter for disambiguating same-nanosecond events.
static GLOBAL_SEQ: AtomicU32 = AtomicU32::new(0);

/// A totally-ordered timestamp that combines wall-clock time with a
/// sequence number for disambiguation.
///
/// Ordering: first by `nanos`, then by `seq`. Two events at the same
/// nanosecond are ordered by their sequence number.
///
/// Size: 12 bytes (u64 + u32), fits in a register pair.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash)]
pub struct EventTime {
    /// Nanoseconds since Unix epoch (same as current Timestamp)
    pub nanos: u64,
    /// Monotonic sequence counter within the same process
    pub seq: u32,
}

impl EventTime {
    pub const MIN: EventTime = EventTime { nanos: 0, seq: 0 };
    pub const MAX: EventTime = EventTime {
        nanos: u64::MAX,
        seq: u32::MAX,
    };

    /// Create a new EventTime with the current wall clock and a fresh sequence number.
    pub fn now() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let seq = GLOBAL_SEQ.fetch_add(1, Ordering::Relaxed);
        Self { nanos, seq }
    }

    /// Create from a raw nanosecond timestamp (for backward compatibility).
    /// Assigns sequence number 0.
    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos, seq: 0 }
    }

    /// Create with explicit components.
    pub fn new(nanos: u64, seq: u32) -> Self {
        Self { nanos, seq }
    }

    /// Extract the nanosecond component (for backward-compatible APIs).
    pub fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Lower bound for a given nanosecond: `(nanos, 0)`.
    /// Use with ranges to capture all events at a timestamp: `start(t)..start(t+1)`.
    pub fn start(nanos: u64) -> Self {
        Self { nanos, seq: 0 }
    }

    /// Upper bound for a given nanosecond: `(nanos, u32::MAX)`.
    /// Includes all events at this timestamp in range queries.
    pub fn end(nanos: u64) -> Self {
        Self {
            nanos,
            seq: u32::MAX,
        }
    }

    /// Next EventTime in total order. Increments seq, rolling over to next nanosecond.
    /// Saturates at MAX.
    pub fn next(self) -> Self {
        if self.seq < u32::MAX {
            Self {
                nanos: self.nanos,
                seq: self.seq + 1,
            }
        } else if self.nanos < u64::MAX {
            Self {
                nanos: self.nanos + 1,
                seq: 0,
            }
        } else {
            self // saturate at MAX
        }
    }

    /// Previous EventTime in total order. Decrements seq, rolling over to previous nanosecond.
    /// Saturates at MIN.
    pub fn previous(self) -> Self {
        if self.seq > 0 {
            Self {
                nanos: self.nanos,
                seq: self.seq - 1,
            }
        } else if self.nanos > 0 {
            Self {
                nanos: self.nanos - 1,
                seq: u32::MAX,
            }
        } else {
            self // saturate at MIN
        }
    }
}

impl PartialEq for EventTime {
    fn eq(&self, other: &Self) -> bool {
        self.nanos == other.nanos && self.seq == other.seq
    }
}
impl Eq for EventTime {}

impl PartialOrd for EventTime {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}
impl Ord for EventTime {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.nanos.cmp(&other.nanos).then(self.seq.cmp(&other.seq))
    }
}

/// Implicit conversion from legacy u64 Timestamp.
impl From<u64> for EventTime {
    fn from(nanos: u64) -> Self {
        Self::from_nanos(nanos)
    }
}

/// Implicit conversion to legacy u64 Timestamp (loses seq).
impl From<EventTime> for u64 {
    fn from(et: EventTime) -> u64 {
        et.nanos
    }
}

impl std::fmt::Display for EventTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.nanos, self.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordering_by_nanos() {
        let a = EventTime::new(100, 0);
        let b = EventTime::new(200, 0);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_ordering_by_seq() {
        let a = EventTime::new(100, 0);
        let b = EventTime::new(100, 1);
        assert!(a < b);
    }

    #[test]
    fn test_equality() {
        let a = EventTime::new(100, 5);
        let b = EventTime::new(100, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_inequality() {
        let a = EventTime::new(100, 5);
        let b = EventTime::new(100, 6);
        assert_ne!(a, b);
    }

    #[test]
    fn test_min_max() {
        assert!(EventTime::MIN < EventTime::MAX);
        assert!(EventTime::MIN <= EventTime::new(0, 0));
    }

    #[test]
    fn test_from_nanos() {
        let et = EventTime::from_nanos(42);
        assert_eq!(et.nanos, 42);
        assert_eq!(et.seq, 0);
    }

    #[test]
    fn test_from_u64() {
        let et: EventTime = 42u64.into();
        assert_eq!(et.nanos, 42);
        assert_eq!(et.seq, 0);
    }

    #[test]
    fn test_into_u64() {
        let et = EventTime::new(42, 7);
        let nanos: u64 = et.into();
        assert_eq!(nanos, 42);
    }

    #[test]
    fn test_now_sequence_disambiguation() {
        let a = EventTime::now();
        let b = EventTime::now();
        // Even if same nanosecond, seq should differ
        assert!(a.seq != b.seq || a.nanos != b.nanos);
        // b should be >= a
        assert!(b >= a);
    }

    #[test]
    fn test_serde_roundtrip() {
        let et = EventTime::new(123456789, 42);
        let json = serde_json::to_string(&et).unwrap();
        let et2: EventTime = serde_json::from_str(&json).unwrap();
        assert_eq!(et, et2);
    }

    #[test]
    fn test_display() {
        let et = EventTime::new(100, 5);
        assert_eq!(format!("{}", et), "100:5");
    }

    #[test]
    fn test_as_nanos() {
        let et = EventTime::new(999, 7);
        assert_eq!(et.as_nanos(), 999);
    }

    #[test]
    fn test_start_end_boundaries() {
        let s = EventTime::start(100);
        let e = EventTime::end(100);
        assert_eq!(s, EventTime::new(100, 0));
        assert_eq!(e, EventTime::new(100, u32::MAX));
        assert!(s < e);
        // All events at nanos=100 fall within [start(100), end(100)]
        assert!(EventTime::new(100, 42) >= s);
        assert!(EventTime::new(100, 42) <= e);
    }

    #[test]
    fn test_next() {
        // Normal case: increment seq
        let a = EventTime::new(100, 5);
        assert_eq!(a.next(), EventTime::new(100, 6));

        // Seq overflow: roll to next nanosecond
        let b = EventTime::new(100, u32::MAX);
        assert_eq!(b.next(), EventTime::new(101, 0));

        // Saturate at MAX
        assert_eq!(EventTime::MAX.next(), EventTime::MAX);
    }

    #[test]
    fn test_previous() {
        // Normal case: decrement seq
        let a = EventTime::new(100, 5);
        assert_eq!(a.previous(), EventTime::new(100, 4));

        // Seq underflow: roll to previous nanosecond
        let b = EventTime::new(100, 0);
        assert_eq!(b.previous(), EventTime::new(99, u32::MAX));

        // Saturate at MIN
        assert_eq!(EventTime::MIN.previous(), EventTime::MIN);
    }

    #[test]
    fn test_next_previous_roundtrip() {
        let t = EventTime::new(500, 10);
        assert_eq!(t.next().previous(), t);
        assert_eq!(t.previous().next(), t);
    }
}
