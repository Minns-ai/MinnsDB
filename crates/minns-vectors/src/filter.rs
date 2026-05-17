//! Conjunctive filter applied during search.
//!
//! The grammar is deliberately small: every `must_eq` condition must match
//! and every `must_neq` condition must not. Disjunctions, ranges, and nested
//! groups are omitted on purpose. Add a variant when a real call site needs
//! it. A small surface makes the backend translation obvious and forces every
//! impl to support the same set.

use crate::point::FilterValue;

/// An AND-of-conditions filter. Build incrementally with [`Self::eq`] and
/// [`Self::neq`].
///
/// ```
/// # use minns_vectors::{Filter, FilterValue};
/// let f = Filter::new()
///     .eq("agent_id", 7u64)
///     .neq("consolidation_status", "archived");
/// assert_eq!(f.eq_conditions(),  &[("agent_id".to_string(), FilterValue::U64(7))][..]);
/// assert_eq!(f.neq_conditions(), &[("consolidation_status".to_string(), FilterValue::Str("archived".into()))][..]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Filter {
    must_eq: Vec<(String, FilterValue)>,
    must_neq: Vec<(String, FilterValue)>,
}

impl Filter {
    /// Construct an empty filter (matches every point).
    pub fn new() -> Self {
        Self::default()
    }

    /// Require `field == value`.
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<FilterValue>) -> Self {
        self.must_eq.push((field.into(), value.into()));
        self
    }

    /// Require `field != value`.
    pub fn neq(mut self, field: impl Into<String>, value: impl Into<FilterValue>) -> Self {
        self.must_neq.push((field.into(), value.into()));
        self
    }

    /// Whether the filter has no conditions (and therefore matches every point).
    pub fn is_empty(&self) -> bool {
        self.must_eq.is_empty() && self.must_neq.is_empty()
    }

    /// Equality conditions, in insertion order.
    pub fn eq_conditions(&self) -> &[(String, FilterValue)] {
        &self.must_eq
    }

    /// Inequality conditions, in insertion order.
    pub fn neq_conditions(&self) -> &[(String, FilterValue)] {
        &self.must_neq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_is_empty() {
        assert!(Filter::new().is_empty());
    }

    #[test]
    fn eq_adds_must_condition() {
        let f = Filter::new().eq("agent_id", 5u64).eq("tier", "semantic");
        assert!(!f.is_empty());
        assert_eq!(
            f.eq_conditions(),
            &[
                ("agent_id".to_string(), FilterValue::U64(5)),
                ("tier".to_string(), FilterValue::Str("semantic".into())),
            ]
        );
        assert!(f.neq_conditions().is_empty());
    }

    #[test]
    fn neq_adds_must_not_condition() {
        let f = Filter::new().neq("status", "archived");
        assert_eq!(
            f.neq_conditions(),
            &[("status".to_string(), FilterValue::Str("archived".into()))]
        );
        assert!(f.eq_conditions().is_empty());
    }

    #[test]
    fn eq_and_neq_compose() {
        let f = Filter::new()
            .eq("agent_id", 1u64)
            .neq("status", "archived")
            .eq("tier", "episodic");
        assert_eq!(f.eq_conditions().len(), 2);
        assert_eq!(f.neq_conditions().len(), 1);
    }

    #[test]
    fn ordering_is_preserved() {
        // Ensures translation to backend filters can rely on insertion order
        // for stable tests and deterministic logs.
        let f = Filter::new().eq("c", 1u64).eq("a", 2u64).eq("b", 3u64);
        let keys: Vec<&str> = f.eq_conditions().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["c", "a", "b"]);
    }
}
