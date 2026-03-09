//! Dense slot array for O(1) access by monotonic ID.
//!
//! Nodes are stored in a dense Vec indexed by `(id - 1)` (since IDs start at 1).
//! Deleted slots are represented as `None`.

use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Dense Vec with O(1) access by monotonic ID.
///
/// - `get(id)` is O(1) array index: `self.slots[(id - 1) as usize]`
/// - `insert()` appends with a fresh monotonic ID
/// - `remove(id)` sets slot to None
/// - Iteration skips empty slots
#[derive(Debug, Clone)]
pub struct SlotVec<T> {
    slots: Vec<Option<T>>,
    len: usize,
    next_id: u64,
}

impl<T> SlotVec<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
            next_id: 1,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            len: 0,
            next_id: 1,
        }
    }

    /// Insert a value, returning its assigned ID (monotonic, starts at 1).
    pub fn insert(&mut self, value: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let index = (id - 1) as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index, || None);
            self.slots.push(Some(value));
        } else {
            // Overwriting an existing slot — adjust len if it was empty
            if self.slots[index].is_none() {
                self.len += 1;
            }
            self.slots[index] = Some(value);
            return id;
        }
        self.len += 1;
        id
    }

    /// Insert a value at a specific pre-assigned ID (for deserialization/restore).
    pub fn insert_at(&mut self, id: u64, value: T) {
        assert!(id >= 1, "SlotVec IDs start at 1");
        let index = (id - 1) as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index, || None);
            self.slots.push(Some(value));
            self.len += 1;
        } else {
            if self.slots[index].is_none() {
                self.len += 1;
            }
            self.slots[index] = Some(value);
        }
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }

    /// O(1) immutable access by ID. Returns None if empty or out of range.
    #[inline]
    pub fn get(&self, id: u64) -> Option<&T> {
        if id == 0 {
            return None;
        }
        self.slots.get((id - 1) as usize)?.as_ref()
    }

    /// O(1) mutable access by ID.
    #[inline]
    pub fn get_mut(&mut self, id: u64) -> Option<&mut T> {
        if id == 0 {
            return None;
        }
        self.slots.get_mut((id - 1) as usize)?.as_mut()
    }

    /// Remove by ID. Returns the removed value.
    pub fn remove(&mut self, id: u64) -> Option<T> {
        if id == 0 {
            return None;
        }
        let slot = self.slots.get_mut((id - 1) as usize)?;
        let value = slot.take()?;
        self.len -= 1;
        Some(value)
    }

    /// Check if an ID is occupied.
    #[inline]
    pub fn contains_key(&self, id: u64) -> bool {
        self.get(id).is_some()
    }

    /// Number of occupied slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Next ID that will be assigned by `insert()`.
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Ensure a slot exists at `id`, inserting a default value if empty.
    /// Returns a mutable reference to the value at that slot.
    pub fn ensure_at(&mut self, id: u64, default: T) -> &mut T {
        assert!(id >= 1, "SlotVec IDs start at 1");
        let index = (id - 1) as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index, || None);
            self.slots.push(Some(default));
            self.len += 1;
            if id >= self.next_id {
                self.next_id = id + 1;
            }
        } else if self.slots[index].is_none() {
            self.slots[index] = Some(default);
            self.len += 1;
            if id >= self.next_id {
                self.next_id = id + 1;
            }
        }
        self.slots[index].as_mut().unwrap()
    }

    /// Clear all slots, resetting to empty state but preserving next_id.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.len = 0;
    }

    /// Iterator over (id, &value) for occupied slots.
    pub fn iter(&self) -> SlotVecIter<'_, T> {
        SlotVecIter {
            slots: &self.slots,
            index: 0,
        }
    }

    /// Mutable iterator over (id, &mut value) for occupied slots.
    pub fn iter_mut(&mut self) -> SlotVecIterMut<'_, T> {
        SlotVecIterMut {
            slots: self.slots.iter_mut().enumerate(),
        }
    }

    /// Iterator over all occupied values.
    pub fn values(&self) -> SlotVecValues<'_, T> {
        SlotVecValues {
            slots: &self.slots,
            index: 0,
        }
    }

    /// Mutable iterator over all occupied values.
    pub fn values_mut(&mut self) -> SlotVecValuesMut<'_, T> {
        SlotVecValuesMut {
            slots: self.slots.iter_mut(),
        }
    }

    /// Iterator over all occupied IDs.
    pub fn keys(&self) -> SlotVecKeys<'_, T> {
        SlotVecKeys {
            slots: &self.slots,
            index: 0,
        }
    }
}

impl<T> Default for SlotVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Iterators ────────────────────────────────────────────────────────────

pub struct SlotVecIter<'a, T> {
    slots: &'a [Option<T>],
    index: usize,
}

impl<'a, T> Iterator for SlotVecIter<'a, T> {
    type Item = (u64, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.slots.len() {
            let idx = self.index;
            self.index += 1;
            if let Some(value) = &self.slots[idx] {
                return Some(((idx as u64) + 1, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len().saturating_sub(self.index)))
    }
}

pub struct SlotVecIterMut<'a, T> {
    slots: std::iter::Enumerate<std::slice::IterMut<'a, Option<T>>>,
}

impl<'a, T> Iterator for SlotVecIterMut<'a, T> {
    type Item = (u64, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        for (idx, slot) in self.slots.by_ref() {
            if let Some(value) = slot {
                return Some(((idx as u64) + 1, value));
            }
        }
        None
    }
}

pub struct SlotVecValues<'a, T> {
    slots: &'a [Option<T>],
    index: usize,
}

impl<'a, T> Iterator for SlotVecValues<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.slots.len() {
            let idx = self.index;
            self.index += 1;
            if let Some(value) = &self.slots[idx] {
                return Some(value);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len().saturating_sub(self.index)))
    }
}

pub struct SlotVecValuesMut<'a, T> {
    slots: std::slice::IterMut<'a, Option<T>>,
}

impl<'a, T> Iterator for SlotVecValuesMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.slots.by_ref().flatten().next()
    }
}

pub struct SlotVecKeys<'a, T> {
    slots: &'a [Option<T>],
    index: usize,
}

impl<'a, T> Iterator for SlotVecKeys<'a, T> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.slots.len() {
            let idx = self.index;
            self.index += 1;
            if self.slots[idx].is_some() {
                return Some((idx as u64) + 1);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len().saturating_sub(self.index)))
    }
}

// ── Serde: serialize only occupied slots as (next_id, [(id, value), ...]) ──

impl<T: Serialize> Serialize for SlotVec<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&self.next_id)?;
        let entries: Vec<(u64, &T)> = self.iter().collect();
        tup.serialize_element(&entries)?;
        tup.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for SlotVec<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SlotVecVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for SlotVecVisitor<T> {
            type Value = SlotVec<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a SlotVec as (next_id, [(id, value), ...])")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let next_id: u64 = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let entries: Vec<(u64, T)> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;

                let mut sv = SlotVec::new();
                sv.next_id = next_id;
                for (id, value) in entries {
                    sv.insert_at(id, value);
                }
                // Restore next_id after insert_at may have advanced it
                if next_id > sv.next_id {
                    sv.next_id = next_id;
                }
                Ok(sv)
            }
        }

        deserializer.deserialize_tuple(2, SlotVecVisitor(std::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut sv = SlotVec::new();
        let id1 = sv.insert(10);
        let id2 = sv.insert(20);
        let id3 = sv.insert(30);

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        assert_eq!(sv.get(1), Some(&10));
        assert_eq!(sv.get(2), Some(&20));
        assert_eq!(sv.get(3), Some(&30));
        assert_eq!(sv.get(4), None);
        assert_eq!(sv.get(0), None);
        assert_eq!(sv.len(), 3);
    }

    #[test]
    fn test_insert_at() {
        let mut sv = SlotVec::new();
        sv.insert_at(5, "five");
        sv.insert_at(3, "three");
        sv.insert_at(1, "one");

        assert_eq!(sv.get(1), Some(&"one"));
        assert_eq!(sv.get(2), None);
        assert_eq!(sv.get(3), Some(&"three"));
        assert_eq!(sv.get(4), None);
        assert_eq!(sv.get(5), Some(&"five"));
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.next_id(), 6);
    }

    #[test]
    fn test_insert_at_overwrite() {
        let mut sv = SlotVec::new();
        sv.insert_at(1, 100);
        assert_eq!(sv.len(), 1);
        sv.insert_at(1, 200);
        assert_eq!(sv.get(1), Some(&200));
        assert_eq!(sv.len(), 1); // should not double-count
    }

    #[test]
    fn test_remove() {
        let mut sv = SlotVec::new();
        sv.insert(10);
        sv.insert(20);
        sv.insert(30);

        assert_eq!(sv.remove(2), Some(20));
        assert_eq!(sv.get(2), None);
        assert_eq!(sv.len(), 2);
        assert!(sv.contains_key(1));
        assert!(!sv.contains_key(2));
        assert!(sv.contains_key(3));

        // Remove non-existent
        assert_eq!(sv.remove(2), None);
        assert_eq!(sv.remove(99), None);
        assert_eq!(sv.remove(0), None);
    }

    #[test]
    fn test_get_mut() {
        let mut sv = SlotVec::new();
        sv.insert(10);
        if let Some(val) = sv.get_mut(1) {
            *val = 42;
        }
        assert_eq!(sv.get(1), Some(&42));
    }

    #[test]
    fn test_iter() {
        let mut sv = SlotVec::new();
        sv.insert("a");
        sv.insert("b");
        sv.insert("c");
        sv.remove(2);

        let items: Vec<(u64, &&str)> = sv.iter().collect();
        assert_eq!(items, vec![(1, &"a"), (3, &"c")]);
    }

    #[test]
    fn test_iter_mut() {
        let mut sv = SlotVec::new();
        sv.insert(1);
        sv.insert(2);
        sv.insert(3);

        for (_id, val) in sv.iter_mut() {
            *val *= 10;
        }

        assert_eq!(sv.get(1), Some(&10));
        assert_eq!(sv.get(2), Some(&20));
        assert_eq!(sv.get(3), Some(&30));
    }

    #[test]
    fn test_values() {
        let mut sv = SlotVec::new();
        sv.insert(10);
        sv.insert(20);
        sv.insert(30);
        sv.remove(2);

        let vals: Vec<&i32> = sv.values().collect();
        assert_eq!(vals, vec![&10, &30]);
    }

    #[test]
    fn test_values_mut() {
        let mut sv = SlotVec::new();
        sv.insert(1);
        sv.insert(2);
        sv.insert(3);
        sv.remove(2);

        for val in sv.values_mut() {
            *val *= 100;
        }

        assert_eq!(sv.get(1), Some(&100));
        assert_eq!(sv.get(2), None);
        assert_eq!(sv.get(3), Some(&300));
    }

    #[test]
    fn test_keys() {
        let mut sv = SlotVec::new();
        sv.insert("x");
        sv.insert("y");
        sv.insert("z");
        sv.remove(2);

        let keys: Vec<u64> = sv.keys().collect();
        assert_eq!(keys, vec![1, 3]);
    }

    #[test]
    fn test_clear() {
        let mut sv = SlotVec::new();
        sv.insert(1);
        sv.insert(2);
        let next = sv.next_id();
        sv.clear();
        assert_eq!(sv.len(), 0);
        assert!(sv.is_empty());
        assert_eq!(sv.get(1), None);
        // next_id is preserved across clear
        assert_eq!(sv.next_id(), next);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut sv = SlotVec::new();
        sv.insert(10u64);
        sv.insert(20u64);
        sv.insert(30u64);
        sv.remove(2);

        let json = serde_json::to_string(&sv).unwrap();
        let sv2: SlotVec<u64> = serde_json::from_str(&json).unwrap();

        assert_eq!(sv2.len(), 2);
        assert_eq!(sv2.get(1), Some(&10));
        assert_eq!(sv2.get(2), None);
        assert_eq!(sv2.get(3), Some(&30));
        assert_eq!(sv2.next_id(), sv.next_id());
    }

    #[test]
    fn test_default() {
        let sv: SlotVec<i32> = SlotVec::default();
        assert_eq!(sv.len(), 0);
        assert!(sv.is_empty());
        assert_eq!(sv.next_id(), 1);
    }

    #[test]
    fn test_with_capacity() {
        let sv: SlotVec<i32> = SlotVec::with_capacity(100);
        assert_eq!(sv.len(), 0);
        assert_eq!(sv.next_id(), 1);
    }

    #[test]
    fn test_contains_key() {
        let mut sv = SlotVec::new();
        sv.insert(42);
        assert!(sv.contains_key(1));
        assert!(!sv.contains_key(2));
        assert!(!sv.contains_key(0));
    }

    #[test]
    fn test_sparse_insert_at() {
        let mut sv = SlotVec::new();
        sv.insert_at(1000, "far");
        assert_eq!(sv.len(), 1);
        assert_eq!(sv.get(1000), Some(&"far"));
        assert_eq!(sv.get(999), None);
        assert_eq!(sv.next_id(), 1001);
    }

    #[test]
    fn test_ensure_at_updates_next_id() {
        let mut sv: SlotVec<i32> = SlotVec::new();
        sv.insert(10); // id=1
        sv.insert(20); // id=2
                       // ensure_at with id=5 (beyond current next_id=3)
        sv.ensure_at(5, 50);
        assert_eq!(sv.next_id(), 6);
        assert_eq!(sv.get(5), Some(&50));
        // insert() should now use id=6, not collide with 5
        let id = sv.insert(60);
        assert_eq!(id, 6);
        assert_eq!(sv.len(), 4);
    }

    #[test]
    fn test_insert_overwrite_occupied_counts_correctly() {
        let mut sv = SlotVec::new();
        sv.insert_at(1, 10);
        sv.insert_at(2, 20);
        assert_eq!(sv.len(), 2);
        // Overwrite via insert_at
        sv.insert_at(1, 100);
        assert_eq!(sv.len(), 2); // unchanged
        assert_eq!(sv.get(1), Some(&100));
    }
}
