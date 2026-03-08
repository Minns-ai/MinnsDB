//! TCell: time-indexed property container.
//!
//! Uses a 4-tier storage strategy:
//!
//! - Empty: no values (zero allocation)
//! - One(EventTime, Value): single value (inline, common case)
//! - Small(SortedVec): 2-64 values (sorted, binary search, cache-friendly)
//! - Large(BTreeMap): 65+ values (log(n) lookup)

use agent_db_core::event_time::EventTime;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const SMALL_THRESHOLD: usize = 64;

/// A time-indexed cell storing the history of a single property value.
///
/// Generic over the value type `V`. Common instantiations:
/// - `TCell<f32>` for confidence, weight, significance
/// - `TCell<u32>` for observation_count, frequency
/// - `TCell<serde_json::Value>` for untyped properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TCell<V> {
    /// No values recorded.
    Empty,
    /// Exactly one value at one time. Most common case for properties
    /// that rarely change.
    One(EventTime, V),
    /// 2-64 values, stored sorted by time. Binary search for lookups.
    Small(Vec<(EventTime, V)>),
    /// 65+ values, BTreeMap for efficient range queries.
    Large(BTreeMap<EventTime, V>),
}

impl<V> Default for TCell<V> {
    fn default() -> Self {
        TCell::Empty
    }
}

impl<V: Clone> TCell<V> {
    pub fn new() -> Self {
        TCell::Empty
    }

    /// Set the value at a specific time. Maintains sorted order.
    pub fn set(&mut self, time: EventTime, value: V) {
        *self = match std::mem::replace(self, TCell::Empty) {
            TCell::Empty => TCell::One(time, value),
            TCell::One(t, v) => {
                if t == time {
                    TCell::One(time, value)
                } else {
                    let mut vec = vec![(t, v), (time, value)];
                    vec.sort_by_key(|(t, _)| *t);
                    TCell::Small(vec)
                }
            },
            TCell::Small(mut vec) => {
                let pos = vec.binary_search_by_key(&time, |(t, _)| *t);
                match pos {
                    Ok(i) => {
                        vec[i].1 = value;
                    },
                    Err(i) => {
                        vec.insert(i, (time, value));
                    },
                }
                if vec.len() > SMALL_THRESHOLD {
                    TCell::Large(vec.into_iter().collect())
                } else {
                    TCell::Small(vec)
                }
            },
            TCell::Large(mut map) => {
                map.insert(time, value);
                TCell::Large(map)
            },
        };
    }

    /// Get the value at a specific time (exact match).
    pub fn at(&self, time: &EventTime) -> Option<&V> {
        match self {
            TCell::Empty => None,
            TCell::One(t, v) => {
                if t == time {
                    Some(v)
                } else {
                    None
                }
            },
            TCell::Small(vec) => vec
                .binary_search_by_key(time, |(t, _)| *t)
                .ok()
                .map(|i| &vec[i].1),
            TCell::Large(map) => map.get(time),
        }
    }

    /// Get the latest value at or before a given time (inclusive).
    ///
    /// Returns the entry with the largest time `t` where `t <= time`.
    pub fn last_at_or_before(&self, time: EventTime) -> Option<(&EventTime, &V)> {
        match self {
            TCell::Empty => None,
            TCell::One(t, v) => {
                if *t <= time {
                    Some((t, v))
                } else {
                    None
                }
            },
            TCell::Small(vec) => {
                let pos = vec.partition_point(|(t, _)| *t <= time);
                if pos > 0 {
                    Some((&vec[pos - 1].0, &vec[pos - 1].1))
                } else {
                    None
                }
            },
            TCell::Large(map) => map.range(..=time).next_back(),
        }
    }

    /// Get the latest value regardless of time.
    pub fn latest(&self) -> Option<(&EventTime, &V)> {
        match self {
            TCell::Empty => None,
            TCell::One(t, v) => Some((t, v)),
            TCell::Small(vec) => vec.last().map(|(t, v)| (t, v)),
            TCell::Large(map) => map.iter().next_back(),
        }
    }

    /// Get the latest value only (convenience for "current value" queries).
    pub fn latest_value(&self) -> Option<&V> {
        self.latest().map(|(_, v)| v)
    }

    /// Iterate over all (time, value) pairs in chronological order.
    pub fn iter(&self) -> TCellIter<'_, V> {
        match self {
            TCell::Empty => TCellIter::Empty,
            TCell::One(t, v) => TCellIter::One(Some((t, v))),
            TCell::Small(vec) => TCellIter::Small(vec.iter()),
            TCell::Large(map) => TCellIter::Large(map.iter()),
        }
    }

    /// Iterate over values in a time window [start, end] (inclusive).
    /// Returns a lazy iterator — no allocation.
    pub fn iter_window(&self, start: EventTime, end: EventTime) -> TCellWindowIter<'_, V> {
        match self {
            TCell::Empty => TCellWindowIter::Empty,
            TCell::One(t, v) => {
                if *t >= start && *t <= end {
                    TCellWindowIter::One(Some((t, v)))
                } else {
                    TCellWindowIter::Empty
                }
            },
            TCell::Small(vec) => {
                let lo = vec.partition_point(|(t, _)| *t < start);
                let hi = vec.partition_point(|(t, _)| *t <= end);
                TCellWindowIter::Small(vec[lo..hi].iter())
            },
            TCell::Large(map) => TCellWindowIter::Large(map.range(start..=end)),
        }
    }

    /// Number of recorded values.
    pub fn len(&self) -> usize {
        match self {
            TCell::Empty => 0,
            TCell::One(..) => 1,
            TCell::Small(vec) => vec.len(),
            TCell::Large(map) => map.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries older than `cutoff`, keeping only entries where `time >= cutoff`.
    pub fn compact(&mut self, cutoff: EventTime) {
        *self = match std::mem::replace(self, TCell::Empty) {
            TCell::Empty => TCell::Empty,
            TCell::One(t, v) => {
                if t >= cutoff {
                    TCell::One(t, v)
                } else {
                    TCell::Empty
                }
            },
            TCell::Small(mut vec) => {
                vec.retain(|(t, _)| *t >= cutoff);
                match vec.len() {
                    0 => TCell::Empty,
                    1 => {
                        let (t, v) = vec.into_iter().next().unwrap();
                        TCell::One(t, v)
                    },
                    _ => TCell::Small(vec),
                }
            },
            TCell::Large(mut map) => {
                // O(log n) split instead of O(n) rebuild
                let kept = map.split_off(&cutoff);
                drop(map); // discard entries before cutoff
                match kept.len() {
                    0 => TCell::Empty,
                    1 => {
                        let (t, v) = kept.into_iter().next().unwrap();
                        TCell::One(t, v)
                    },
                    n if n <= SMALL_THRESHOLD => TCell::Small(kept.into_iter().collect()),
                    _ => TCell::Large(kept),
                }
            },
        };
    }
}

// Keep backward compat: last_before is an alias for last_at_or_before
impl<V: Clone> TCell<V> {
    #[doc(hidden)]
    #[inline]
    pub fn last_before(&self, time: EventTime) -> Option<(&EventTime, &V)> {
        self.last_at_or_before(time)
    }
}

/// Iterator over TCell entries (forward and backward).
pub enum TCellIter<'a, V> {
    Empty,
    One(Option<(&'a EventTime, &'a V)>),
    Small(std::slice::Iter<'a, (EventTime, V)>),
    Large(std::collections::btree_map::Iter<'a, EventTime, V>),
}

impl<'a, V> Iterator for TCellIter<'a, V> {
    type Item = (&'a EventTime, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TCellIter::Empty => None,
            TCellIter::One(item) => item.take(),
            TCellIter::Small(it) => it.next().map(|(t, v)| (t, v)),
            TCellIter::Large(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            TCellIter::Empty => (0, Some(0)),
            TCellIter::One(Some(_)) => (1, Some(1)),
            TCellIter::One(None) => (0, Some(0)),
            TCellIter::Small(it) => it.size_hint(),
            TCellIter::Large(it) => it.size_hint(),
        }
    }
}

impl<'a, V> DoubleEndedIterator for TCellIter<'a, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            TCellIter::Empty => None,
            TCellIter::One(item) => item.take(),
            TCellIter::Small(it) => it.next_back().map(|(t, v)| (t, v)),
            TCellIter::Large(it) => it.next_back(),
        }
    }
}

impl<'a, V> ExactSizeIterator for TCellIter<'a, V> {}

/// Windowed iterator over TCell entries in [start, end].
pub enum TCellWindowIter<'a, V> {
    Empty,
    One(Option<(&'a EventTime, &'a V)>),
    Small(std::slice::Iter<'a, (EventTime, V)>),
    Large(std::collections::btree_map::Range<'a, EventTime, V>),
}

impl<'a, V> Iterator for TCellWindowIter<'a, V> {
    type Item = (&'a EventTime, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TCellWindowIter::Empty => None,
            TCellWindowIter::One(item) => item.take(),
            TCellWindowIter::Small(it) => it.next().map(|(t, v)| (t, v)),
            TCellWindowIter::Large(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            TCellWindowIter::Empty => (0, Some(0)),
            TCellWindowIter::One(Some(_)) => (1, Some(1)),
            TCellWindowIter::One(None) => (0, Some(0)),
            TCellWindowIter::Small(it) => it.size_hint(),
            TCellWindowIter::Large(it) => it.size_hint(),
        }
    }
}

impl<'a, V> DoubleEndedIterator for TCellWindowIter<'a, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            TCellWindowIter::Empty => None,
            TCellWindowIter::One(item) => item.take(),
            TCellWindowIter::Small(it) => it.next_back().map(|(t, v)| (t, v)),
            TCellWindowIter::Large(it) => it.next_back(),
        }
    }
}

impl<'a, V> ExactSizeIterator for TCellWindowIter<'a, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let cell: TCell<f32> = TCell::new();
        assert!(cell.is_empty());
        assert_eq!(cell.len(), 0);
        assert!(cell.latest().is_none());
        assert!(cell.latest_value().is_none());
    }

    #[test]
    fn test_one() {
        let mut cell = TCell::new();
        let t = EventTime::new(100, 0);
        cell.set(t, 0.5f32);

        assert_eq!(cell.len(), 1);
        assert!(!cell.is_empty());
        assert_eq!(cell.at(&t), Some(&0.5));
        assert_eq!(cell.latest().unwrap().1, &0.5);
        assert_eq!(cell.latest_value(), Some(&0.5));
    }

    #[test]
    fn test_two_values() {
        let mut cell = TCell::new();
        let t1 = EventTime::new(100, 0);
        let t2 = EventTime::new(200, 0);
        cell.set(t1, 0.5f32);
        cell.set(t2, 0.8f32);

        assert_eq!(cell.len(), 2);
        assert_eq!(cell.at(&t1), Some(&0.5));
        assert_eq!(cell.at(&t2), Some(&0.8));
        assert_eq!(cell.latest().unwrap().1, &0.8);
    }

    #[test]
    fn test_overwrite_same_time() {
        let mut cell = TCell::new();
        let t = EventTime::new(100, 0);
        cell.set(t, 0.5f32);
        cell.set(t, 0.9f32);

        assert_eq!(cell.len(), 1);
        assert_eq!(cell.at(&t), Some(&0.9));
    }

    #[test]
    fn test_last_at_or_before() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);

        // Between two values
        let result = cell.last_at_or_before(EventTime::new(250, 0));
        assert_eq!(result.unwrap().1, &2.0);

        // Before any value
        let result = cell.last_at_or_before(EventTime::new(50, 0));
        assert!(result.is_none());

        // Exact match (inclusive)
        let result = cell.last_at_or_before(EventTime::new(300, 0));
        assert_eq!(result.unwrap().1, &3.0);
    }

    #[test]
    fn test_last_before_alias() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        // last_before should work identically to last_at_or_before
        assert_eq!(cell.last_before(EventTime::new(200, 0)).unwrap().1, &2.0);
    }

    #[test]
    fn test_iter() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(300, 0), 3.0f32);
        cell.set(EventTime::new(100, 0), 1.0);
        cell.set(EventTime::new(200, 0), 2.0);

        let values: Vec<f32> = cell.iter().map(|(_, v)| *v).collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_iter_double_ended() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);

        // Iterate backwards
        let values: Vec<f32> = cell.iter().rev().map(|(_, v)| *v).collect();
        assert_eq!(values, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_iter_exact_size() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);

        assert_eq!(cell.iter().len(), 3);
    }

    #[test]
    fn test_iter_window() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);
        cell.set(EventTime::new(400, 0), 4.0);

        // Lazy iterator, no Vec allocation
        let values: Vec<f32> = cell
            .iter_window(EventTime::new(150, 0), EventTime::new(350, 0))
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(values, vec![2.0, 3.0]);
    }

    #[test]
    fn test_iter_window_reverse() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);

        let last = cell
            .iter_window(EventTime::new(100, 0), EventTime::new(300, 0))
            .next_back();
        assert_eq!(last.unwrap().1, &3.0);
    }

    #[test]
    fn test_promotion_to_large() {
        let mut cell = TCell::new();
        for i in 0..=SMALL_THRESHOLD as u64 {
            cell.set(EventTime::new(i * 100, 0), i as f32);
        }
        assert_eq!(cell.len(), SMALL_THRESHOLD + 1);
        assert!(matches!(cell, TCell::Large(_)));
    }

    #[test]
    fn test_compact() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);

        cell.compact(EventTime::new(200, 0));
        assert_eq!(cell.len(), 2);
        assert!(cell.at(&EventTime::new(100, 0)).is_none());
        assert_eq!(cell.at(&EventTime::new(200, 0)), Some(&2.0));
        assert_eq!(cell.at(&EventTime::new(300, 0)), Some(&3.0));
    }

    #[test]
    fn test_compact_large_uses_split_off() {
        // Build a Large TCell, then compact
        let mut cell = TCell::new();
        for i in 0..=SMALL_THRESHOLD as u64 + 10 {
            cell.set(EventTime::new(i * 100, 0), i as f32);
        }
        assert!(matches!(cell, TCell::Large(_)));
        let cutoff = EventTime::new(SMALL_THRESHOLD as u64 * 100, 0);
        cell.compact(cutoff);
        // Should have 11 entries remaining (SMALL_THRESHOLD..=SMALL_THRESHOLD+10)
        assert_eq!(cell.len(), 11);
    }

    #[test]
    fn test_compact_to_empty() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.compact(EventTime::new(200, 0));
        assert!(cell.is_empty());
    }

    #[test]
    fn test_compact_to_one() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.compact(EventTime::new(200, 0));
        assert_eq!(cell.len(), 1);
        assert!(matches!(cell, TCell::One(..)));
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut cell = TCell::new();
        cell.set(EventTime::new(100, 0), 1.0f32);
        cell.set(EventTime::new(200, 0), 2.0);
        cell.set(EventTime::new(300, 0), 3.0);

        let json = serde_json::to_string(&cell).unwrap();
        let cell2: TCell<f32> = serde_json::from_str(&json).unwrap();

        assert_eq!(cell2.len(), 3);
        assert_eq!(cell2.at(&EventTime::new(200, 0)), Some(&2.0));
    }

    #[test]
    fn test_serde_empty_roundtrip() {
        let cell: TCell<f32> = TCell::new();
        let json = serde_json::to_string(&cell).unwrap();
        let cell2: TCell<f32> = serde_json::from_str(&json).unwrap();
        assert!(cell2.is_empty());
    }
}
