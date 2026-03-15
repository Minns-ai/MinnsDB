// crates/agent-db-graph/src/structures/adj_list.rs
//
// Adaptive adjacency list with Empty/One/Small/Large variants.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::BTreeSet;

use super::types::EdgeId;

// ============================================================================
// Adaptive Adjacency List
// ============================================================================

/// Threshold at which adjacency list is promoted from Small to Large.
const ADJ_LARGE_THRESHOLD: usize = 1024;

/// Adaptive adjacency list that selects the optimal representation
/// based on edge count.
///
/// - `Empty`: zero edges, no allocation
/// - `One(EdgeId)`: single edge, 8 bytes inline
/// - `Small(SmallVec<[EdgeId; 8]>)`: 2–1024 edges, inline up to 8
/// - `Large(BTreeSet<EdgeId>)`: 1025+ edges, O(log n) operations
#[derive(Debug, Clone, Default)]
pub enum AdjList {
    #[default]
    Empty,
    One(EdgeId),
    Small(SmallVec<[EdgeId; 8]>),
    Large(BTreeSet<EdgeId>),
}

impl AdjList {
    pub fn new() -> Self {
        AdjList::Empty
    }

    pub fn push(&mut self, edge_id: EdgeId) {
        *self = match std::mem::take(self) {
            AdjList::Empty => AdjList::One(edge_id),
            AdjList::One(existing) => {
                let mut sv = SmallVec::new();
                sv.push(existing);
                sv.push(edge_id);
                AdjList::Small(sv)
            },
            AdjList::Small(mut sv) => {
                sv.push(edge_id);
                if sv.len() > ADJ_LARGE_THRESHOLD {
                    AdjList::Large(sv.into_iter().collect())
                } else {
                    AdjList::Small(sv)
                }
            },
            AdjList::Large(mut set) => {
                set.insert(edge_id);
                AdjList::Large(set)
            },
        };
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&EdgeId) -> bool,
    {
        match self {
            AdjList::Empty => {},
            AdjList::One(eid) => {
                if !f(eid) {
                    *self = AdjList::Empty;
                }
            },
            AdjList::Small(sv) => {
                sv.retain(|eid| f(eid));
                match sv.len() {
                    0 => *self = AdjList::Empty,
                    1 => {
                        let eid = sv[0];
                        *self = AdjList::One(eid);
                    },
                    _ => {},
                }
            },
            AdjList::Large(set) => {
                set.retain(|eid| f(eid));
            },
        }
    }

    pub fn iter(&self) -> AdjListIter<'_> {
        match self {
            AdjList::Empty => AdjListIter::Empty,
            AdjList::One(eid) => AdjListIter::One(std::iter::once(eid)),
            AdjList::Small(sv) => AdjListIter::Small(sv.iter()),
            AdjList::Large(set) => AdjListIter::Large(set.iter()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            AdjList::Empty => 0,
            AdjList::One(_) => 1,
            AdjList::Small(sv) => sv.len(),
            AdjList::Large(set) => set.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, AdjList::Empty)
    }

    pub fn contains(&self, edge_id: &EdgeId) -> bool {
        match self {
            AdjList::Empty => false,
            AdjList::One(eid) => eid == edge_id,
            AdjList::Small(sv) => sv.contains(edge_id),
            AdjList::Large(set) => set.contains(edge_id),
        }
    }

    /// Build an AdjList from a Vec, picking the optimal variant.
    fn from_vec(ids: Vec<EdgeId>) -> Self {
        match ids.len() {
            0 => AdjList::Empty,
            1 => AdjList::One(ids[0]),
            n if n > ADJ_LARGE_THRESHOLD => AdjList::Large(ids.into_iter().collect()),
            _ => AdjList::Small(SmallVec::from_vec(ids)),
        }
    }
}

/// Custom Serialize: writes AdjList as a flat sequence of EdgeIds.
/// This is backward-compatible with SmallVec<[EdgeId; 8]> serialization.
impl Serialize for AdjList {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let len = self.len();
        let mut seq = serializer.serialize_seq(Some(len))?;
        for eid in self.iter() {
            seq.serialize_element(eid)?;
        }
        seq.end()
    }
}

/// Custom Deserialize: reads a sequence of EdgeIds and picks the optimal variant.
/// Backward-compatible with SmallVec<[EdgeId; 8]> deserialization.
impl<'de> Deserialize<'de> for AdjList {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let ids: Vec<EdgeId> = Vec::deserialize(deserializer)?;
        Ok(AdjList::from_vec(ids))
    }
}

/// Iterator over AdjList entries.
pub enum AdjListIter<'a> {
    Empty,
    One(std::iter::Once<&'a EdgeId>),
    Small(std::slice::Iter<'a, EdgeId>),
    Large(std::collections::btree_set::Iter<'a, EdgeId>),
}

impl<'a> Iterator for AdjListIter<'a> {
    type Item = &'a EdgeId;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AdjListIter::Empty => None,
            AdjListIter::One(it) => it.next(),
            AdjListIter::Small(it) => it.next(),
            AdjListIter::Large(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            AdjListIter::Empty => (0, Some(0)),
            AdjListIter::One(it) => it.size_hint(),
            AdjListIter::Small(it) => it.size_hint(),
            AdjListIter::Large(it) => it.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for AdjListIter<'a> {}

impl<'a> IntoIterator for &'a AdjList {
    type Item = &'a EdgeId;
    type IntoIter = AdjListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
