//! Search queries.
//!
//! A [`Query`] is built with [`Query::builder`] and consumed by
//! [`VectorStore::search`](crate::VectorStore::search). The builder takes the
//! query vector by value (a `search` returns owned [`ScoredPoint`]s, so
//! cloning the vector at construction time is consistent with the rest of the
//! API and avoids a lifetime parameter that would propagate everywhere).

use crate::filter::Filter;

/// A search request.
///
/// Defaults: `top_k = 10`, no minimum score, no filter.
#[derive(Debug, Clone)]
pub struct Query {
    vector: Vec<f32>,
    top_k: usize,
    min_score: Option<f32>,
    filter: Option<Filter>,
}

impl Query {
    /// Start building a query against `vector`. Default `top_k` is `10`.
    pub fn builder(vector: Vec<f32>) -> QueryBuilder {
        QueryBuilder::new(vector)
    }

    /// The query vector.
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Number of results to return.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Minimum score threshold, if any.
    pub fn min_score(&self) -> Option<f32> {
        self.min_score
    }

    /// Filter to apply during search, if any.
    pub fn filter(&self) -> Option<&Filter> {
        self.filter.as_ref()
    }
}

/// Builder for [`Query`].
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    vector: Vec<f32>,
    top_k: usize,
    min_score: Option<f32>,
    filter: Option<Filter>,
}

impl QueryBuilder {
    /// Construct a builder against `vector`. Prefer [`Query::builder`] at call
    /// sites.
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            top_k: 10,
            min_score: None,
            filter: None,
        }
    }

    /// Number of results to return. Must be `>= 1`. Setting `0` will cause the
    /// backend to reject the query with
    /// [`VectorError::InvalidQuery`](crate::VectorError::InvalidQuery).
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Minimum score threshold. Points scoring below this are dropped before
    /// returning.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Filter to apply during search. Replaces any previous filter.
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Finalize the query.
    pub fn build(self) -> Query {
        Query {
            vector: self.vector,
            top_k: self.top_k,
            min_score: self.min_score,
            filter: self.filter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_top_k_ten_no_filter_no_threshold() {
        let q = Query::builder(vec![1.0, 2.0, 3.0]).build();
        assert_eq!(q.vector(), &[1.0, 2.0, 3.0]);
        assert_eq!(q.top_k(), 10);
        assert!(q.min_score().is_none());
        assert!(q.filter().is_none());
    }

    #[test]
    fn options_set_through_builder() {
        let q = Query::builder(vec![0.1])
            .top_k(25)
            .min_score(0.7)
            .filter(Filter::new().eq("agent_id", 42u64))
            .build();

        assert_eq!(q.top_k(), 25);
        assert_eq!(q.min_score(), Some(0.7));
        let f = q.filter().expect("filter set");
        assert_eq!(f.eq_conditions().len(), 1);
    }

    #[test]
    fn last_filter_wins() {
        let q = Query::builder(vec![0.0])
            .filter(Filter::new().eq("a", 1u64))
            .filter(Filter::new().eq("b", 2u64))
            .build();

        let f = q.filter().expect("filter set");
        assert_eq!(f.eq_conditions().len(), 1);
        assert_eq!(f.eq_conditions()[0].0, "b");
    }
}
