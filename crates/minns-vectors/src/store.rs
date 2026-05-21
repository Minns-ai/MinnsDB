//! The backend-agnostic [`VectorStore`] trait.

use async_trait::async_trait;

use crate::error::VectorResult;
use crate::point::{Point, ScoredPoint};
use crate::query::Query;

/// An abstract vector index over points keyed by [`u128`] ids with strongly
/// typed payload fields.
///
/// Implementations own persistence and search. Callers depend on this trait
/// rather than any concrete backend so the underlying engine can be swapped
/// out without touching call sites.
///
/// All methods are `&self`: implementations are expected to handle internal
/// synchronization. Callers may freely share an implementation across threads
/// via `Arc<dyn VectorStore>`.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Upsert a batch of points. Points whose ids already exist are replaced
    /// in full (vector and payload). Empty batches are a no-op.
    async fn upsert(&self, points: Vec<Point>) -> VectorResult<()>;

    /// Fetch points by id. Missing ids yield `None` at the corresponding
    /// position so the result is positionally aligned with the input.
    ///
    /// Implementations are expected to include both the vector and the payload
    /// in the returned [`Point`], so callers can use this to recover an
    /// original embedding (e.g. for centroid math) without keeping a separate
    /// copy.
    async fn fetch(&self, ids: &[u128]) -> VectorResult<Vec<Option<Point>>>;

    /// Delete points by id. Returns the number actually removed (ids that did
    /// not exist are not counted). Empty input is a no-op that returns `0`.
    async fn delete(&self, ids: &[u128]) -> VectorResult<usize>;

    /// Top-k search by vector with optional filter and score threshold.
    ///
    /// The returned [`ScoredPoint`]s always include their payload. Results are
    /// ordered by score (highest first under cosine / dot, lowest first under
    /// euclidean; the order matches whatever distance the collection was
    /// configured with).
    async fn search(&self, query: &Query) -> VectorResult<Vec<ScoredPoint>>;

    /// Approximate point count in the collection. May be expensive on some
    /// backends (Qdrant performs a full segment scan when an exact count is
    /// requested); intended for diagnostics and admin paths, not the hot
    /// retrieval loop.
    async fn count(&self) -> VectorResult<usize>;
}
