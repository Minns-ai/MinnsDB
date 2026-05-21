//! Value types shared across the public API: points, payloads, filter values.
//!
//! These types exist so callers never have to depend on a specific backend's
//! representation. Conversions to and from the backend's wire types live next
//! to the backend implementation (see [`crate::qdrant`]).

use std::collections::BTreeMap;

/// A point destined for or retrieved from a vector store.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// The point's identifier. MinnsDB's internal `NodeId` / `MemoryId` /
    /// `ClaimId` types all map cleanly onto `u128`; backends encode this as
    /// whatever native id format they support (Qdrant uses a UUID).
    pub id: u128,
    /// The embedding itself. Length must match the collection's configured
    /// dimension, enforced at the backend boundary.
    pub vector: Vec<f32>,
    /// Structured metadata used for filter-during-search. Stored alongside
    /// the vector in the backend. Empty for ids that need no filtering.
    pub payload: Payload,
}

impl Point {
    /// Construct a point with the given id, vector, and payload.
    pub fn new(id: u128, vector: Vec<f32>, payload: Payload) -> Self {
        Self {
            id,
            vector,
            payload,
        }
    }
}

/// A point retrieved by [`VectorStore::search`](crate::VectorStore::search),
/// scored against the query vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPoint {
    /// The point id.
    pub id: u128,
    /// The similarity score under the collection's distance metric. Higher is
    /// more similar for cosine and dot product; lower is more similar for
    /// euclidean. Callers should interpret in the context of the collection's
    /// configured distance.
    pub score: f32,
    /// The payload, if `with_payload` was requested. Implementations always
    /// return a payload, possibly [`Payload::EMPTY`], so callers do not have
    /// to handle a missing-payload case.
    pub payload: Payload,
}

/// Strongly typed metadata attached to a point.
///
/// Backed by [`BTreeMap`] so iteration order is stable across runs, which
/// matters for log output and snapshot-style tests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Payload {
    fields: BTreeMap<String, FilterValue>,
}

impl Payload {
    /// An empty payload. Equivalent to [`Payload::default()`] but usable in
    /// `const` contexts and reads slightly clearer at call sites.
    pub const EMPTY: Self = Self {
        fields: BTreeMap::new(),
    };

    /// Construct an empty payload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a field, consuming and returning self. Designed for fluent
    /// construction:
    ///
    /// ```
    /// # use minns_vectors::Payload;
    /// let p = Payload::new()
    ///     .with("agent_id", 42u64)
    ///     .with("tier", "episodic");
    /// assert_eq!(p.len(), 2);
    /// ```
    pub fn with(mut self, key: impl Into<String>, value: impl Into<FilterValue>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Insert a field by mutable reference. Equivalent to [`Self::with`] when
    /// fluent construction is not convenient.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<FilterValue>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Look up a field by name.
    pub fn get(&self, key: &str) -> Option<&FilterValue> {
        self.fields.get(key)
    }

    /// Iterate fields in deterministic (sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &FilterValue)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of fields in the payload.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the payload has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// A scalar value that can appear in a [`Payload`] field or as the right-hand
/// side of a filter condition.
///
/// Kept deliberately small: just the primitive shapes the existing call sites
/// need (u64 for ids, i64 for signed counters, bool for flags, String for
/// enum-like tags). Adding a new variant is a real schema decision and a
/// breaking change for backend impls, which is the right cost to impose.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    /// Unsigned 64-bit integer. Used for ids, counters, bucket keys.
    U64(u64),
    /// Signed 64-bit integer. Used for timestamps, signed counters.
    I64(i64),
    /// Boolean flag.
    Bool(bool),
    /// String tag. Use for enum-like values (`"episodic"`, `"archived"`).
    Str(String),
}

impl From<u64> for FilterValue {
    fn from(v: u64) -> Self {
        Self::U64(v)
    }
}

impl From<i64> for FilterValue {
    fn from(v: i64) -> Self {
        Self::I64(v)
    }
}

impl From<bool> for FilterValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<&str> for FilterValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
}

impl From<String> for FilterValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_builds_with_fluent_api() {
        let p = Payload::new()
            .with("agent_id", 42u64)
            .with("tier", "episodic")
            .with("archived", false);

        assert_eq!(p.len(), 3);
        assert_eq!(p.get("agent_id"), Some(&FilterValue::U64(42)));
        assert_eq!(p.get("tier"), Some(&FilterValue::Str("episodic".into())));
        assert_eq!(p.get("archived"), Some(&FilterValue::Bool(false)));
        assert_eq!(p.get("missing"), None);
    }

    #[test]
    fn payload_iter_is_sorted() {
        let p = Payload::new()
            .with("zeta", 1u64)
            .with("alpha", 2u64)
            .with("mu", 3u64);

        let keys: Vec<&str> = p.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn payload_empty_is_empty() {
        let p = Payload::EMPTY;
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert!(p.iter().next().is_none());
    }

    #[test]
    fn payload_insert_overwrites() {
        let mut p = Payload::new();
        p.insert("k", 1u64);
        p.insert("k", 2u64);
        assert_eq!(p.get("k"), Some(&FilterValue::U64(2)));
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn filter_value_conversions() {
        assert_eq!(FilterValue::from(5u64), FilterValue::U64(5));
        assert_eq!(FilterValue::from(-1i64), FilterValue::I64(-1));
        assert_eq!(FilterValue::from(true), FilterValue::Bool(true));
        assert_eq!(FilterValue::from("x"), FilterValue::Str("x".into()));
        assert_eq!(
            FilterValue::from(String::from("x")),
            FilterValue::Str("x".into())
        );
    }

    #[test]
    fn point_constructor_round_trips_fields() {
        let payload = Payload::new().with("agent_id", 7u64);
        let p = Point::new(123, vec![0.1, 0.2, 0.3], payload.clone());
        assert_eq!(p.id, 123);
        assert_eq!(p.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(p.payload, payload);
    }
}
