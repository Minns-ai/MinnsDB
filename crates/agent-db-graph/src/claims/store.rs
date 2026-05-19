//! Persistent storage for claims using redb.
//!
//! The [`ClaimStore`] keeps three things in sync:
//!
//! 1. a redb table holding the full [`DerivedClaim`] payload (text, metadata,
//!    status, counters),
//! 2. an in-memory BM25 index over claim text for keyword search, and
//! 3. a backend-agnostic [`VectorStore`] (Qdrant in production) holding the
//!    embedding vectors, accessed via [`Vectors::claims`](crate::vectors::Vectors).
//!
//! BM25 is rebuilt from disk on construction. Vector membership is *not*
//! reloaded — vectors live in the external store and persist independently of
//! the ClaimStore lifecycle.

use super::types::*;
use crate::indexing::Bm25Index;
use anyhow::Result;
use minns_vectors::{Payload, Point, Query, VectorStore};
use parking_lot::RwLock;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

const CLAIMS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");
const CLAIM_COUNTER: TableDefinition<&str, u64> = TableDefinition::new("claim_counter");

/// Persistent store for derived claims.
///
/// State:
/// - **redb** owns the claim payloads on disk.
/// - **BM25** owns the keyword index in memory.
/// - **VectorStore** (Qdrant) owns the embeddings out of process.
///
/// All vector-touching methods are `async` so the caller pays the network
/// round-trip cost explicitly; BM25 and redb operations remain synchronous.
pub struct ClaimStore {
    db: Database,
    bm25_index: RwLock<Bm25Index>,
    vectors: Arc<dyn VectorStore>,
}

impl ClaimStore {
    /// Open the claim store at `path`, attaching it to `vectors` (the
    /// `claims` collection of [`Vectors`](crate::vectors::Vectors)).
    ///
    /// On open, all active claims are rebuilt into the in-memory BM25 index.
    /// Vector entries are *not* reloaded — they live in `vectors` already.
    pub fn new<P: AsRef<Path>>(path: P, vectors: Arc<dyn VectorStore>) -> Result<Self> {
        info!("Opening claim store at: {:?}", path.as_ref());

        let db = Database::create(path)?;

        // Initialize tables
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(CLAIMS_TABLE)?;
            let _ = write_txn.open_table(CLAIM_COUNTER)?;
        }
        write_txn.commit()?;

        // Rebuild BM25 from disk
        let mut bm25 = Bm25Index::new();
        {
            let read_txn = db.begin_read()?;
            let table = read_txn.open_table(CLAIMS_TABLE)?;
            let mut loaded = 0usize;
            for item in table.iter()? {
                let (id_guard, value) = item?;
                let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
                if claim.status == ClaimStatus::Active {
                    bm25.index_document(id_guard.value(), &claim.claim_text);
                    loaded += 1;
                }
            }
            if loaded > 0 {
                info!("Loaded {} active claims into BM25 index", loaded);
            }
        }

        info!("Claim store initialized successfully");

        Ok(Self {
            db,
            bm25_index: RwLock::new(bm25),
            vectors,
        })
    }

    /// Persist a claim. Updates BM25 immediately. Vector upload is a
    /// separate step (see [`ClaimStore::update_embedding`]).
    pub fn store(&self, claim: &DerivedClaim) -> Result<()> {
        debug!(
            "Storing claim {} ({} evidence spans)",
            claim.id,
            claim.supporting_evidence.len()
        );

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let serialized = rmp_serde::to_vec(claim)?;
            table.insert(claim.id, serialized.as_slice())?;
        }
        write_txn.commit()?;

        if claim.status == ClaimStatus::Active {
            self.bm25_index
                .write()
                .index_document(claim.id, &claim.claim_text);
        }

        debug!("Successfully stored claim {}", claim.id);

        Ok(())
    }

    /// Get a claim by ID.
    pub fn get(&self, claim_id: ClaimId) -> Result<Option<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        if let Some(value) = table.get(claim_id)? {
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            Ok(Some(claim))
        } else {
            Ok(None)
        }
    }

    /// Allocate the next claim ID.
    pub fn next_id(&self) -> Result<ClaimId> {
        let write_txn = self.db.begin_write()?;
        let id = {
            let mut table = write_txn.open_table(CLAIM_COUNTER)?;
            let current = table.get("counter")?.map(|v| v.value()).unwrap_or(0);
            let next = current + 1;
            table.insert("counter", next)?;
            next
        };
        write_txn.commit()?;
        Ok(id)
    }

    /// Increment the support counter for a claim. Vector index untouched.
    pub fn add_support(&self, claim_id: ClaimId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.support_count += 1;
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Update a claim's status. When transitioning away from `Active` the
    /// claim is removed from both the BM25 index and the vector store.
    ///
    /// **Cost:** one redb write txn (sync). When the transition is
    /// Active → inactive, one additional BM25 mutation (sync, in-memory) and
    /// one Qdrant `delete` RPC (async). Other transitions are sync redb only.
    ///
    /// **Atomicity:** redb commits before the Qdrant call. If the future is
    /// dropped or Qdrant fails after the redb commit, the claim is marked
    /// inactive in redb but its vector may still sit in the Qdrant
    /// collection. The vector becomes unreachable through the public API
    /// (search results are joined back to redb where status filters block
    /// inactive rows), but it is not reclaimed automatically — it leaks
    /// until a future `delete` or operator-driven cleanup runs. Qdrant
    /// errors are logged, not propagated.
    pub async fn update_status(&self, claim_id: ClaimId, status: ClaimStatus) -> Result<()> {
        let became_inactive = self.write_status_inner(claim_id, status, None)?;
        if became_inactive {
            self.bm25_index.write().remove_document(claim_id);
            if let Err(e) = self.vectors.delete(&[claim_id as u128]).await {
                tracing::warn!("Vector delete failed for claim {}: {}", claim_id, e);
            }
        }
        Ok(())
    }

    /// Mark a claim as superseded with a `valid_until` timestamp. The claim
    /// is removed from BM25 and the vector store. Same cost and atomicity
    /// behaviour as [`ClaimStore::update_status`].
    pub async fn supersede(&self, claim_id: ClaimId, valid_until: u64) -> Result<()> {
        let became_inactive =
            self.write_status_inner(claim_id, ClaimStatus::Superseded, Some(valid_until))?;
        if became_inactive {
            self.bm25_index.write().remove_document(claim_id);
            if let Err(e) = self.vectors.delete(&[claim_id as u128]).await {
                tracing::warn!("Vector delete failed for claim {}: {}", claim_id, e);
            }
        }
        Ok(())
    }

    /// Helper: update status (and optionally valid_until) in redb, returning
    /// `true` when the claim transitioned from `Active` to a non-active state.
    fn write_status_inner(
        &self,
        claim_id: ClaimId,
        status: ClaimStatus,
        valid_until: Option<u64>,
    ) -> Result<bool> {
        let write_txn = self.db.begin_write()?;
        let became_inactive = {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                Some(rmp_serde::from_slice(value.value())?)
            } else {
                None
            };
            if let Some(mut claim) = maybe_claim {
                let was_active = claim.status == ClaimStatus::Active;
                claim.status = status;
                if let Some(vu) = valid_until {
                    claim.valid_until = Some(vu);
                }
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                was_active && status != ClaimStatus::Active
            } else {
                false
            }
        };
        write_txn.commit()?;
        Ok(became_inactive)
    }

    /// Merge metadata entries into a claim's existing metadata.
    pub fn update_metadata(
        &self,
        claim_id: ClaimId,
        new_metadata: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                for (k, v) in new_metadata {
                    claim.metadata.insert(k.clone(), v.clone());
                }
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Mark a claim as accessed.
    pub fn mark_accessed(&self, claim_id: ClaimId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.mark_accessed();
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Record an outcome (success/failure) for a claim.
    pub fn record_outcome(&self, claim_id: ClaimId, success: bool) -> Result<Option<DerivedClaim>> {
        let write_txn = self.db.begin_write()?;
        let result = {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.record_outcome(success);
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                Some(claim)
            } else {
                None
            }
        };
        write_txn.commit()?;
        Ok(result)
    }

    /// Delete a claim from all stores.
    ///
    /// **Cost:** one redb write txn (sync), one BM25 mutation (sync,
    /// in-memory), one Qdrant `delete` RPC (async).
    ///
    /// **Atomicity:** redb commits before the Qdrant call. A dropped future
    /// or Qdrant failure leaves the row removed from redb and the vector
    /// orphaned in Qdrant (logged, not propagated). The orphan is benign —
    /// no claim row references it, so it cannot surface in search joins —
    /// but it is not reclaimed automatically.
    pub async fn delete(&self, claim_id: ClaimId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            table.remove(claim_id)?;
        }
        write_txn.commit()?;

        self.bm25_index.write().remove_document(claim_id);
        if let Err(e) = self.vectors.delete(&[claim_id as u128]).await {
            tracing::warn!("Vector delete failed for claim {}: {}", claim_id, e);
        }

        debug!("Deleted claim {}", claim_id);
        Ok(())
    }

    /// Total claim count in redb.
    pub fn count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;
        Ok(table.len()?.try_into().unwrap_or(0))
    }

    /// Get all active claims, up to `limit`.
    pub fn get_all_active(&self, limit: usize) -> Result<Vec<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        let mut claims = Vec::new();
        let iter = table.iter()?;

        for item in iter {
            if claims.len() >= limit {
                break;
            }
            let (_id, value) = item?;
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            if claim.status == ClaimStatus::Active {
                claims.push(claim);
            }
        }

        Ok(claims)
    }

    /// Find similar claims via the vector store. Returns
    /// `(claim_id, similarity)` pairs.
    ///
    /// **Cost:** one Qdrant `search` RPC. Bounded by `top_k`; an empty
    /// `query_embedding` or `top_k == 0` returns immediately without a
    /// network call.
    ///
    /// The score interpretation matches the underlying collection's
    /// configured distance (cosine by default — higher is closer).
    pub async fn find_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(ClaimId, f32)>> {
        if query_embedding.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let mut builder = Query::builder(query_embedding.to_vec()).top_k(top_k);
        if min_similarity > 0.0 {
            builder = builder.min_score(min_similarity);
        }
        let query = builder.build();

        let scored = self
            .vectors
            .search(&query)
            .await
            .map_err(|e| anyhow::anyhow!("claim vector search failed: {e}"))?;

        let results: Vec<(ClaimId, f32)> =
            scored.into_iter().map(|p| (p.id as u64, p.score)).collect();

        debug!(
            "find_similar: {} results (top_k={}, min_similarity={})",
            results.len(),
            top_k,
            min_similarity
        );

        Ok(results)
    }

    /// Fetch the stored embedding for a single claim, if any.
    ///
    /// **Cost:** one Qdrant `fetch` RPC. Used by the avoidance-claim flow
    /// to inherit a source claim's vector without paying for a new
    /// embedding model call.
    pub async fn fetch_embedding(&self, claim_id: ClaimId) -> Result<Option<Vec<f32>>> {
        let points = self
            .vectors
            .fetch(&[claim_id as u128])
            .await
            .map_err(|e| anyhow::anyhow!("claim vector fetch failed: {e}"))?;
        Ok(points.into_iter().next().flatten().map(|p| p.vector))
    }

    /// Persist `embedding` against `claim_id` (redb + vector store) and
    /// mark `has_embedding = true`. No-op if the claim does not exist or
    /// the embedding is empty.
    ///
    /// **Cost:** one redb write txn (sync) plus one Qdrant `upsert` RPC
    /// (async, single point). For batches, prefer
    /// [`ClaimStore::mark_has_embedding`] + [`ClaimStore::upsert_vectors`]
    /// which collapses N RPCs into one.
    ///
    /// **Atomicity:** redb commits the `has_embedding = true` flag before
    /// the Qdrant upsert. If the future is dropped or Qdrant rejects the
    /// upsert, redb claims the vector exists but Qdrant does not have it.
    /// The Qdrant error *is* propagated (unlike the delete path), so the
    /// caller observes the inconsistency. Callers may want to retry the
    /// upsert or reset `has_embedding`.
    pub async fn update_embedding(&self, claim_id: ClaimId, embedding: Vec<f32>) -> Result<()> {
        if embedding.is_empty() {
            return Ok(());
        }

        // Read-modify-write the claim record. Capture whether the claim is
        // active so we can decide on the vector upsert below.
        let should_upsert = {
            let write_txn = self.db.begin_write()?;
            let active = {
                let mut table = write_txn.open_table(CLAIMS_TABLE)?;
                let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                    Some(rmp_serde::from_slice(value.value())?)
                } else {
                    None
                };
                if let Some(mut claim) = maybe_claim {
                    claim.has_embedding = true;
                    let active = claim.status == ClaimStatus::Active;
                    let serialized = rmp_serde::to_vec(&claim)?;
                    table.insert(claim_id, serialized.as_slice())?;
                    active
                } else {
                    false
                }
            };
            write_txn.commit()?;
            active
        };

        if should_upsert {
            let point = Point::new(claim_id as u128, embedding, Payload::EMPTY);
            self.vectors
                .upsert(vec![point])
                .await
                .map_err(|e| anyhow::anyhow!("claim vector upsert failed: {e}"))?;
            debug!("Upserted embedding for claim {}", claim_id);
        }

        Ok(())
    }

    /// Flip `has_embedding` to true for `claim_id` and persist. Returns
    /// `true` when the claim exists and is active (so callers should follow
    /// up with a vector upsert), and `false` otherwise.
    ///
    /// Designed for batch flows: pair with [`ClaimStore::upsert_vectors`] so
    /// a batch of N claims costs N tiny redb writes and one vector RPC, not
    /// N vector RPCs.
    pub fn mark_has_embedding(&self, claim_id: ClaimId) -> Result<bool> {
        let write_txn = self.db.begin_write()?;
        let active = {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let maybe_claim: Option<DerivedClaim> = if let Some(value) = table.get(claim_id)? {
                Some(rmp_serde::from_slice(value.value())?)
            } else {
                None
            };
            if let Some(mut claim) = maybe_claim {
                let active = claim.status == ClaimStatus::Active;
                if active {
                    claim.has_embedding = true;
                    let serialized = rmp_serde::to_vec(&claim)?;
                    table.insert(claim_id, serialized.as_slice())?;
                }
                active
            } else {
                false
            }
        };
        write_txn.commit()?;
        Ok(active)
    }

    /// Upsert a batch of points into the claims vector collection.
    ///
    /// **Cost:** one Qdrant `upsert` RPC regardless of batch size; empty
    /// batches are a no-op. Pair with [`ClaimStore::mark_has_embedding`]
    /// to keep redb and the vector store consistent.
    pub async fn upsert_vectors(&self, points: Vec<Point>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        self.vectors
            .upsert(points)
            .await
            .map_err(|e| anyhow::anyhow!("claim vector batch upsert failed: {e}"))?;
        Ok(())
    }

    /// Get all claims that still need an embedding generated.
    pub fn get_claims_needing_embeddings(&self, limit: usize) -> Result<Vec<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        let mut claims = Vec::new();
        for item in table.iter()? {
            if claims.len() >= limit {
                break;
            }
            let (_id, value) = item?;
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            if claim.status == ClaimStatus::Active && !claim.has_embedding {
                claims.push(claim);
            }
        }

        Ok(claims)
    }

    /// Access the BM25 index (for hybrid search).
    pub fn bm25_index(&self) -> &RwLock<Bm25Index> {
        &self.bm25_index
    }

    /// Find active Avoidance claims, returning up to `limit` sorted by recency.
    pub fn find_avoidance_claims(&self, limit: usize) -> Result<Vec<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;
        let mut claims = Vec::new();
        for item in table.iter()? {
            let (_, value) = item?;
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            if claim.status == ClaimStatus::Active && claim.claim_type == ClaimType::Avoidance {
                claims.push(claim);
            }
        }
        claims.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        claims.truncate(limit);
        Ok(claims)
    }

    /// Delete all claims in non-active terminal states from the persistent
    /// store. Returns the number of claims purged.
    pub fn purge_inactive_claims(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        let mut to_purge: Vec<ClaimId> = Vec::new();
        for item in table.iter()? {
            let (id_guard, value) = item?;
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            match claim.status {
                ClaimStatus::Dormant | ClaimStatus::Rejected | ClaimStatus::Superseded => {
                    to_purge.push(id_guard.value());
                },
                _ => {},
            }
        }
        drop(table);
        drop(read_txn);

        let count = to_purge.len();
        if count > 0 {
            let write_txn = self.db.begin_write()?;
            {
                let mut table = write_txn.open_table(CLAIMS_TABLE)?;
                for cid in &to_purge {
                    table.remove(*cid)?;
                }
            }
            write_txn.commit()?;
            info!("Purged {} inactive claims from disk", count);
        }

        Ok(count)
    }

    /// Expire all active claims whose `expires_at` is in the past. Returns
    /// the number of claims moved to `Dormant`.
    pub async fn expire_stale_claims(&self) -> Result<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        let mut to_dormant: Vec<ClaimId> = Vec::new();
        for item in table.iter()? {
            let (id_guard, value) = item?;
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            if claim.status == ClaimStatus::Active {
                if let Some(exp) = claim.expires_at {
                    if now > exp {
                        to_dormant.push(id_guard.value());
                    }
                }
            }
        }
        drop(table);
        drop(read_txn);

        let count = to_dormant.len();
        for cid in to_dormant {
            self.update_status(cid, ClaimStatus::Dormant).await?;
        }

        if count > 0 {
            info!("Expired {} stale claims to Dormant", count);
        }

        Ok(count)
    }
}
