//! Persistent storage for claims using redb
//!
//! The `ClaimStore` persists claims to redb and maintains an in-memory vector
//! index for fast similarity search.  On construction it loads all active
//! embeddings from disk so that `find_similar` never touches redb.

use super::types::*;
use crate::indexing::Bm25Index;
use anyhow::Result;
use crossbeam::queue::SegQueue;
use parking_lot::RwLock;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use tracing::{debug, info};

const CLAIMS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");
const CLAIM_COUNTER: TableDefinition<&str, u64> = TableDefinition::new("claim_counter");

// ── In-memory vector index ──────────────────────────────────────────────────

/// Threshold: when the index has >= this many entries, HNSW kicks in.
const HNSW_THRESHOLD: usize = 1000;

/// Entry in the in-memory vector index.
/// Embeddings are pre-normalized to unit length so cosine similarity = dot product.
struct VectorEntry {
    id: ClaimId,
    embedding: Vec<f32>,
}

/// Point wrapper for instant-distance HNSW.
/// Stores a normalized embedding; distance = 1 - dot_product (cosine distance).
#[derive(Clone)]
struct HnswPoint(Vec<f32>);

impl instant_distance::Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - dot_product (both are L2-normalized)
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        1.0 - dot
    }
}

/// Flat vector index kept in RAM with optional HNSW acceleration.
///
/// Below `HNSW_THRESHOLD` entries the flat `entries` vec is the source of truth
/// and search is a brute-force dot-product scan.  Above the threshold the HNSW
/// graph owns the embeddings and `entries` is cleared to avoid doubling memory.
struct VectorIndex {
    /// Flat entries — used for brute-force search when below HNSW threshold.
    /// Cleared (memory freed) once HNSW is built so embeddings aren't stored twice.
    entries: Vec<VectorEntry>,
    /// HNSW graph; owns embeddings + claim IDs when built.
    /// `None` when below threshold.
    hnsw: Option<instant_distance::HnswMap<HnswPoint, ClaimId>>,
    /// Set to true whenever entries change, requiring HNSW rebuild.
    hnsw_dirty: bool,
}

impl VectorIndex {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            hnsw: None,
            hnsw_dirty: false,
        }
    }

    /// Number of indexed embeddings.
    fn len(&self) -> usize {
        if self.hnsw.is_some() && self.entries.is_empty() {
            // HNSW is the source of truth — count its values
            self.hnsw.as_ref().map(|h| h.values.len()).unwrap_or(0)
        } else {
            self.entries.len()
        }
    }

    /// Pre-normalize and insert (or update) an embedding.
    fn upsert(&mut self, id: ClaimId, mut embedding: Vec<f32>) {
        if embedding.is_empty() {
            return;
        }
        // L2-normalize so cosine = dot product
        let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in embedding.iter_mut() {
                *v /= mag;
            }
        }
        // Update in-place if already present, else push
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.embedding = embedding;
        } else {
            self.entries.push(VectorEntry { id, embedding });
        }
        self.hnsw_dirty = true;
    }

    /// Remove an entry by id.
    fn remove(&mut self, id: ClaimId) {
        self.entries.retain(|e| e.id != id);
        self.hnsw_dirty = true;
    }

    /// Borrow the raw entries for metric-generic scans.
    fn entries(&self) -> &[VectorEntry] {
        &self.entries
    }

    /// Rebuild HNSW graph from current entries, then free the flat vec.
    ///
    /// After this call `entries` is empty and `hnsw` owns the data.
    /// This is called inside `apply_pending` (write-locked) so searches
    /// never see the intermediate state.
    fn rebuild_hnsw(&mut self) {
        if self.entries.len() < HNSW_THRESHOLD {
            self.hnsw = None;
            self.hnsw_dirty = false;
            return;
        }

        let points: Vec<HnswPoint> = self
            .entries
            .iter()
            .map(|e| HnswPoint(e.embedding.clone()))
            .collect();
        let values: Vec<ClaimId> = self.entries.iter().map(|e| e.id).collect();

        let hnsw_map = instant_distance::Builder::default()
            .ef_construction(100)
            .ef_search(50)
            .build(points, values);

        self.hnsw = Some(hnsw_map);
        // Free the flat entries — HNSW now owns all embedding data
        self.entries = Vec::new();
        self.hnsw_dirty = false;
    }

    /// Materialise the flat entries from the HNSW graph.
    ///
    /// Called when the HNSW graph is dirty (mutations happened) and needs
    /// rebuilding, or when we need the flat entries for a non-cosine scan.
    fn materialise_entries_from_hnsw(&mut self) {
        if let Some(ref hnsw) = self.hnsw {
            if self.entries.is_empty() {
                self.entries = hnsw
                    .iter()
                    .map(|(_, point)| {
                        // We need the claim id too — stored in values at same index
                        // iter yields (PointId, &HnswPoint)
                        VectorEntry {
                            id: 0, // placeholder, filled below
                            embedding: point.0.clone(),
                        }
                    })
                    .collect();
                // Fix up IDs from values vec
                for (i, entry) in self.entries.iter_mut().enumerate() {
                    entry.id = hnsw.values[i];
                }
            }
        }
    }

    /// Read-only search.  HNSW must already be up-to-date (via `apply_pending`).
    /// `query` must already be L2-normalized.
    fn find_similar(
        &self,
        query_normalized: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<(ClaimId, f32)> {
        // Use HNSW path if available
        if let Some(ref hnsw) = self.hnsw {
            let query_point = HnswPoint(query_normalized.to_vec());
            let mut search = instant_distance::Search::default();
            let results: Vec<(ClaimId, f32)> = hnsw
                .search(&query_point, &mut search)
                .take(top_k)
                .filter_map(|item| {
                    let similarity = 1.0 - item.distance;
                    if similarity >= min_similarity {
                        Some((*item.value, similarity))
                    } else {
                        None
                    }
                })
                .collect();
            return results;
        }

        // Brute-force path (below HNSW_THRESHOLD)
        let mut results: Vec<(ClaimId, f32)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                if entry.embedding.len() != query_normalized.len() {
                    return None;
                }
                let dot: f32 = query_normalized
                    .iter()
                    .zip(entry.embedding.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if dot >= min_similarity {
                    Some((entry.id, dot))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(top_k);
        results
    }
}

// ── Pending index update buffer ──────────────────────────────────────────────

/// A deferred index update, pushed to a lock-free queue and batch-applied later.
///
/// Only stores the claim ID — embedding/text are re-read from redb at
/// apply time to avoid cloning large embedding vectors into the queue.
enum PendingIndexUpdate {
    Upsert { claim_id: ClaimId },
    Remove { claim_id: ClaimId },
}

/// Lock-free queue of pending index updates.
struct PendingBuffer {
    queue: SegQueue<PendingIndexUpdate>,
}

impl PendingBuffer {
    fn new() -> Self {
        Self {
            queue: SegQueue::new(),
        }
    }

    fn push(&self, update: PendingIndexUpdate) {
        self.queue.push(update);
    }

    /// Returns true if the queue is non-empty (best-effort, no synchronization).
    fn has_pending(&self) -> bool {
        !self.queue.is_empty()
    }
}

// ── ClaimStore ──────────────────────────────────────────────────────────────

/// Persistent store for derived claims backed by redb with an in-memory
/// vector index for O(n·d) dot-product similarity search (no disk I/O).
pub struct ClaimStore {
    db: Database,
    /// In-memory vector index (pre-normalized embeddings, dot-product scan).
    index: RwLock<VectorIndex>,
    /// In-memory BM25 index for keyword search over claim texts.
    bm25_index: RwLock<Bm25Index>,
    /// Lock-free queue of pending index updates, batch-applied before search.
    pending: PendingBuffer,
}

impl ClaimStore {
    /// Create a new claim store at the given path.
    ///
    /// All active claims with embeddings are loaded into the in-memory vector
    /// index so that `find_similar` is fast from the first call.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Opening claim store at: {:?}", path.as_ref());

        let db = Database::create(path)?;

        // Initialize tables
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(CLAIMS_TABLE)?;
            let _ = write_txn.open_table(CLAIM_COUNTER)?;
        }
        write_txn.commit()?;

        // Load existing embeddings and text into in-memory indexes
        let mut index = VectorIndex::new();
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
                    if !claim.embedding.is_empty() {
                        index.upsert(id_guard.value(), claim.embedding);
                        loaded += 1;
                    }
                }
            }
            if loaded > 0 {
                info!(
                    "Loaded {} claim embeddings into in-memory vector index",
                    loaded
                );
            }
        }

        // Build HNSW eagerly if we loaded enough entries.
        // Without this, read-only workloads would brute-force scan forever
        // because apply_pending() (which normally triggers rebuild) returns
        // early when the pending queue is empty.
        if index.hnsw_dirty && index.entries.len() >= HNSW_THRESHOLD {
            index.rebuild_hnsw();
            info!("Built HNSW index at startup ({} entries)", index.len());
        }

        info!("Claim store initialized successfully");

        Ok(Self {
            db,
            index: RwLock::new(index),
            bm25_index: RwLock::new(bm25),
            pending: PendingBuffer::new(),
        })
    }

    /// Store a claim (persists to redb and updates in-memory index).
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

        // Queue index update (lock-free push, batch-applied before search)
        if claim.status == ClaimStatus::Active {
            self.pending
                .push(PendingIndexUpdate::Upsert { claim_id: claim.id });
        }

        debug!("Successfully stored claim {}", claim.id);

        Ok(())
    }

    /// Get a claim by ID
    pub fn get(&self, claim_id: ClaimId) -> Result<Option<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        if let Some(value) = table.get(claim_id)? {
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            debug!("Retrieved claim {}", claim_id);
            Ok(Some(claim))
        } else {
            debug!("No claim found for ID {}", claim_id);
            Ok(None)
        }
    }

    /// Get next claim ID
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

    /// Add support to an existing claim (increment support count)
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
                debug!(
                    "Added support to claim {} (now: {})",
                    claim_id, claim.support_count
                );
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Update claim status
    pub fn update_status(&self, claim_id: ClaimId, status: ClaimStatus) -> Result<()> {
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
                claim.status = status;
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                debug!("Updated claim {} status to {:?}", claim_id, status);

                // Queue index removal if no longer active
                if status != ClaimStatus::Active {
                    self.pending.push(PendingIndexUpdate::Remove { claim_id });
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Supersede a claim: set status to Superseded and record valid_until timestamp.
    ///
    /// This unifies claim temporal tracking with graph edge supersession.
    /// The claim is removed from search indexes (BM25 + vector).
    pub fn supersede(&self, claim_id: ClaimId, valid_until: u64) -> Result<()> {
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
                claim.status = ClaimStatus::Superseded;
                claim.valid_until = Some(valid_until);
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                debug!(
                    "Superseded claim {} with valid_until={}",
                    claim_id, valid_until
                );
                self.pending.push(PendingIndexUpdate::Remove { claim_id });
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Merge metadata entries into a claim's existing metadata.
    ///
    /// Used to attach state anchors after claim extraction when graph state
    /// is available.
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

    /// Mark claim as accessed
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
    ///
    /// Returns the updated claim, or `None` if the claim was not found.
    /// Follows the same read-modify-write-commit pattern as `add_support`.
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
                debug!(
                    "Recorded {} outcome for claim {} (pos={}, neg={})",
                    if success { "positive" } else { "negative" },
                    claim_id,
                    claim.positive_outcomes(),
                    claim.negative_outcomes()
                );
                Some(claim)
            } else {
                None
            }
        };
        write_txn.commit()?;
        Ok(result)
    }

    /// Delete a claim
    pub fn delete(&self, claim_id: ClaimId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            table.remove(claim_id)?;
        }
        write_txn.commit()?;

        // Queue index removal (lock-free push)
        self.pending.push(PendingIndexUpdate::Remove { claim_id });

        debug!("Deleted claim {}", claim_id);
        Ok(())
    }

    /// Get total count of claims
    pub fn count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;
        Ok(table.len()?.try_into().unwrap_or(0))
    }

    /// Get all active claims (for testing/debugging)
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

    /// Drain the pending update queue and batch-apply all mutations to the
    /// vector index and BM25 index.  Called automatically before search.
    ///
    /// Claim data is re-read from redb so that `PendingIndexUpdate` only
    /// carries a lightweight `ClaimId` (no cloned embeddings / text).
    /// HNSW is rebuilt here (under the write lock) so that `find_similar`
    /// only ever needs a **read lock**.
    pub fn apply_pending(&self) {
        if !self.pending.has_pending() {
            return;
        }

        // Drain IDs from the lock-free queue
        let mut upsert_ids = Vec::new();
        let mut remove_ids = Vec::new();
        while let Some(update) = self.pending.queue.pop() {
            match update {
                PendingIndexUpdate::Upsert { claim_id } => upsert_ids.push(claim_id),
                PendingIndexUpdate::Remove { claim_id } => remove_ids.push(claim_id),
            }
        }

        if upsert_ids.is_empty() && remove_ids.is_empty() {
            return;
        }

        // One read-txn to fetch all claim data needed for the upserts
        let mut upsert_data: Vec<(ClaimId, Vec<f32>, String)> =
            Vec::with_capacity(upsert_ids.len());
        if !upsert_ids.is_empty() {
            if let Ok(read_txn) = self.db.begin_read() {
                if let Ok(table) = read_txn.open_table(CLAIMS_TABLE) {
                    for cid in &upsert_ids {
                        if let Ok(Some(value)) = table.get(*cid) {
                            if let Ok(claim) = rmp_serde::from_slice::<DerivedClaim>(value.value())
                            {
                                if claim.status == ClaimStatus::Active {
                                    upsert_data.push((claim.id, claim.embedding, claim.claim_text));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Single write-lock acquisition for vector index
        {
            let mut idx = self.index.write();
            // Materialise flat entries from HNSW if needed for mutation
            if idx.hnsw.is_some()
                && idx.entries.is_empty()
                && (!upsert_data.is_empty() || !remove_ids.is_empty())
            {
                idx.materialise_entries_from_hnsw();
            }

            for &cid in &remove_ids {
                idx.remove(cid);
            }
            for (cid, embedding, _) in &upsert_data {
                if !embedding.is_empty() {
                    idx.upsert(*cid, embedding.clone());
                }
            }

            // Rebuild HNSW if dirty (frees flat entries after build)
            if idx.hnsw_dirty && idx.entries.len() >= HNSW_THRESHOLD {
                idx.rebuild_hnsw();
            } else if idx.hnsw_dirty && idx.entries.len() < HNSW_THRESHOLD {
                // Dropped below threshold — discard stale HNSW
                idx.hnsw = None;
                idx.hnsw_dirty = false;
            }
        }

        // Single write-lock acquisition for BM25 index
        {
            let mut bm25 = self.bm25_index.write();
            for &cid in &remove_ids {
                bm25.remove_document(cid);
            }
            for (cid, _, text) in &upsert_data {
                bm25.index_document(*cid, text);
            }
        }

        debug!(
            "Applied {} pending index updates ({} upserts, {} removes)",
            upsert_data.len() + remove_ids.len(),
            upsert_data.len(),
            remove_ids.len()
        );
    }

    /// Number of pending (not yet applied) index updates.
    pub fn pending_count(&self) -> usize {
        self.pending.queue.len()
    }

    /// Find similar claims using the in-memory vector index.
    ///
    /// This is a dot-product scan over pre-normalized embeddings — no disk I/O,
    /// no deserialization.  Sub-millisecond for up to ~50k claims.
    ///
    /// Returns (claim_id, similarity_score) pairs sorted by descending similarity.
    pub fn find_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(ClaimId, f32)>> {
        // Apply pending updates for consistency
        self.apply_pending();

        // Normalize query to unit length so dot product = cosine similarity
        let mut query_norm = query_embedding.to_vec();
        let mag: f32 = query_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in query_norm.iter_mut() {
                *v /= mag;
            }
        }

        let results = self
            .index
            .read()
            .find_similar(&query_norm, top_k, min_similarity);

        debug!(
            "Found {} similar claims (top_k={}, min_similarity={}) via in-memory index",
            results.len(),
            top_k,
            min_similarity
        );

        Ok(results)
    }

    /// Find similar claims using the specified distance metric.
    ///
    /// Works like `find_similar()` but allows choosing between Cosine (default),
    /// Euclidean, or Manhattan distance metrics.
    pub fn find_similar_with_metric(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
        metric: super::embeddings::DistanceMetric,
    ) -> Result<Vec<(ClaimId, f32)>> {
        use super::embeddings::{DistanceMetric, VectorSimilarity};

        // For Cosine, delegate to the optimized pre-normalized path
        // (find_similar calls apply_pending internally)
        if metric == DistanceMetric::Cosine {
            return self.find_similar(query_embedding, top_k, min_similarity);
        }

        // Apply pending updates for consistency
        self.apply_pending();

        // For other metrics, do a full scan using VectorSimilarity::compute()
        let idx = self.index.read();

        // When HNSW is active, entries is empty — scan HNSW points instead
        let mut results: Vec<(ClaimId, f32)> = if idx.entries().is_empty() {
            if let Some(ref hnsw) = idx.hnsw {
                hnsw.iter()
                    .enumerate()
                    .filter_map(|(i, (_, point))| {
                        if point.0.len() != query_embedding.len() {
                            return None;
                        }
                        let sim = VectorSimilarity::compute(query_embedding, &point.0, metric);
                        if sim >= min_similarity {
                            Some((hnsw.values[i], sim))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            idx.entries()
                .iter()
                .filter_map(|entry| {
                    if entry.embedding.len() != query_embedding.len() {
                        return None;
                    }
                    let sim = VectorSimilarity::compute(query_embedding, &entry.embedding, metric);
                    if sim >= min_similarity {
                        Some((entry.id, sim))
                    } else {
                        None
                    }
                })
                .collect()
        };

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(top_k);

        debug!(
            "Found {} similar claims (metric={:?}, top_k={}, min_similarity={}) via in-memory index",
            results.len(),
            metric,
            top_k,
            min_similarity
        );

        Ok(results)
    }

    /// Update claim embedding (persists to redb and updates in-memory index).
    pub fn update_embedding(&self, claim_id: ClaimId, embedding: Vec<f32>) -> Result<()> {
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
                claim.embedding = embedding.clone();
                let serialized = rmp_serde::to_vec(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                debug!(
                    "Updated embedding for claim {} ({} dimensions)",
                    claim_id,
                    claim.embedding.len()
                );

                // Queue index update (lock-free push)
                if claim.status == ClaimStatus::Active {
                    self.pending.push(PendingIndexUpdate::Upsert { claim_id });
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get all claims that need embeddings (have empty embedding vectors)
    pub fn get_claims_needing_embeddings(&self, limit: usize) -> Result<Vec<DerivedClaim>> {
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

            if claim.status == ClaimStatus::Active && claim.embedding.is_empty() {
                claims.push(claim);
            }
        }

        Ok(claims)
    }

    /// Access the BM25 index (for hybrid search).
    pub fn bm25_index(&self) -> &RwLock<Bm25Index> {
        &self.bm25_index
    }

    /// Number of embeddings in the in-memory vector index.
    /// Applies pending updates first for consistency.
    pub fn vector_index_size(&self) -> usize {
        self.apply_pending();
        self.index.read().len()
    }

    /// Find active Avoidance claims, returning up to `limit` sorted by recency.
    ///
    /// Simple table scan filtered by `ClaimType::Avoidance` — acceptable because
    /// avoidance claims are rare (only generated after 2+ failures).
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
        // Sort by recency (most recent first)
        claims.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        claims.truncate(limit);
        Ok(claims)
    }

    /// Enforce a cap on the in-memory vector index.
    ///
    /// When the index exceeds `max_size`, the oldest claims (by `created_at`)
    /// are moved to `Dormant` status, removing them from the search index.
    /// Returns the number of claims evicted.
    pub fn enforce_vector_index_cap(&self, max_size: usize) -> Result<usize> {
        let current_size = self.vector_index_size();
        if current_size <= max_size {
            return Ok(0);
        }

        // Collect all active claim IDs with their created_at timestamps
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        let mut active_claims: Vec<(ClaimId, u64)> = Vec::new(); // (id, created_at)
        for item in table.iter()? {
            let (id_guard, value) = item?;
            let claim: DerivedClaim = rmp_serde::from_slice(value.value())?;
            if claim.status == ClaimStatus::Active && !claim.embedding.is_empty() {
                active_claims.push((id_guard.value(), claim.created_at));
            }
        }
        drop(table);
        drop(read_txn);

        // Sort by created_at ascending (oldest first)
        active_claims.sort_by_key(|&(_, created_at)| created_at);

        // Evict oldest until we're at cap
        let to_evict = active_claims.len().saturating_sub(max_size);
        let mut evicted = 0usize;
        for &(claim_id, _) in active_claims.iter().take(to_evict) {
            self.update_status(claim_id, ClaimStatus::Dormant)?;
            evicted += 1;
        }

        if evicted > 0 {
            info!(
                "Vector index cap enforced: evicted {} claims (was {}, cap {})",
                evicted, current_size, max_size
            );
        }

        Ok(evicted)
    }

    /// Delete all claims in non-active terminal states (Dormant, Rejected, Superseded)
    /// from the persistent store to reclaim disk space.
    /// Returns the number of claims purged.
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

    /// Expire all active claims whose `expires_at` is in the past.
    ///
    /// Returns the number of claims that were moved to `Dormant`.
    pub fn expire_stale_claims(&self) -> Result<usize> {
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
            self.update_status(cid, ClaimStatus::Dormant)?;
        }

        if count > 0 {
            info!("Expired {} stale claims to Dormant", count);
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_claim(id: ClaimId, text: &str) -> DerivedClaim {
        let evidence = vec![EvidenceSpan::new(0, 4, "test")];
        DerivedClaim::new(
            id,
            text.to_string(),
            evidence,
            0.9,
            vec![0.1, 0.2, 0.3],
            123,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");

        let store = ClaimStore::new(&db_path).unwrap();
        let claim = create_test_claim(1, "Test claim");

        store.store(&claim).unwrap();

        let retrieved = store.get(1).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, 1);
        assert_eq!(retrieved.claim_text, "Test claim");
    }

    #[test]
    fn test_next_id() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");

        let store = ClaimStore::new(&db_path).unwrap();

        let id1 = store.next_id().unwrap();
        let id2 = store.next_id().unwrap();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_add_support() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");

        let store = ClaimStore::new(&db_path).unwrap();
        let claim = create_test_claim(1, "Test");

        store.store(&claim).unwrap();
        assert_eq!(store.get(1).unwrap().unwrap().support_count, 1);

        store.add_support(1).unwrap();
        assert_eq!(store.get(1).unwrap().unwrap().support_count, 2);
    }

    #[test]
    fn test_update_status() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");

        let store = ClaimStore::new(&db_path).unwrap();
        let claim = create_test_claim(1, "Test");

        store.store(&claim).unwrap();
        assert_eq!(store.get(1).unwrap().unwrap().status, ClaimStatus::Active);

        store.update_status(1, ClaimStatus::Dormant).unwrap();
        assert_eq!(store.get(1).unwrap().unwrap().status, ClaimStatus::Dormant);
    }

    #[test]
    fn test_count() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");

        let store = ClaimStore::new(&db_path).unwrap();

        assert_eq!(store.count().unwrap(), 0);

        store.store(&create_test_claim(1, "Claim 1")).unwrap();
        store.store(&create_test_claim(2, "Claim 2")).unwrap();

        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn test_in_memory_vector_index_find_similar() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");
        let store = ClaimStore::new(&db_path).unwrap();

        // Store claims with known embeddings
        let mut claim1 = create_test_claim(1, "Claim about Google");
        claim1.embedding = vec![1.0, 0.0, 0.0]; // unit vector along x
        store.store(&claim1).unwrap();

        let mut claim2 = create_test_claim(2, "Claim about Apple");
        claim2.embedding = vec![0.9, 0.1, 0.0]; // very similar to x
        store.store(&claim2).unwrap();

        let mut claim3 = create_test_claim(3, "Unrelated claim");
        claim3.embedding = vec![0.0, 0.0, 1.0]; // orthogonal
        store.store(&claim3).unwrap();

        // Query with vector similar to claim1
        let query = vec![1.0, 0.0, 0.0];
        let results = store.find_similar(&query, 3, 0.5).unwrap();

        // Should return claim1 and claim2 (both similar), not claim3
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // claim1 most similar
        assert!(results[0].1 > 0.99); // ~1.0 similarity
        assert_eq!(results[1].0, 2); // claim2 second
        assert!(results[1].1 > 0.9); // high similarity

        // Verify index size
        assert_eq!(store.vector_index_size(), 3);
    }

    #[test]
    fn test_vector_index_survives_restart() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");

        // Store claims in first instance
        {
            let store = ClaimStore::new(&db_path).unwrap();
            let mut claim = create_test_claim(1, "Persistent claim");
            claim.embedding = vec![1.0, 0.0, 0.0];
            store.store(&claim).unwrap();
            assert_eq!(store.vector_index_size(), 1);
        }

        // Re-open — index should be rebuilt from redb
        {
            let store = ClaimStore::new(&db_path).unwrap();
            assert_eq!(store.vector_index_size(), 1);

            let results = store.find_similar(&[1.0, 0.0, 0.0], 5, 0.5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, 1);
        }
    }

    #[test]
    fn test_vector_index_update_embedding() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");
        let store = ClaimStore::new(&db_path).unwrap();

        // Store claim without embedding
        let claim = create_test_claim(1, "No embedding yet");
        // Override embedding to empty
        let mut claim = claim;
        claim.embedding = vec![];
        store.store(&claim).unwrap();
        assert_eq!(store.vector_index_size(), 0); // not indexed

        // Update with real embedding
        store.update_embedding(1, vec![0.5, 0.5, 0.0]).unwrap();
        assert_eq!(store.vector_index_size(), 1); // now indexed

        let results = store.find_similar(&[1.0, 0.0, 0.0], 5, 0.0).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_vector_index_delete_removes_entry() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");
        let store = ClaimStore::new(&db_path).unwrap();

        let mut claim = create_test_claim(1, "Will be deleted");
        claim.embedding = vec![1.0, 0.0, 0.0];
        store.store(&claim).unwrap();
        assert_eq!(store.vector_index_size(), 1);

        store.delete(1).unwrap();
        assert_eq!(store.vector_index_size(), 0);
    }

    #[test]
    fn test_vector_index_dormant_removes_entry() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("claims.redb");
        let store = ClaimStore::new(&db_path).unwrap();

        let mut claim = create_test_claim(1, "Will go dormant");
        claim.embedding = vec![1.0, 0.0, 0.0];
        store.store(&claim).unwrap();
        assert_eq!(store.vector_index_size(), 1);

        store.update_status(1, ClaimStatus::Dormant).unwrap();
        assert_eq!(store.vector_index_size(), 0);
    }

    // ── P3: Multi-metric tests ─────────────────────────────────────────────

    #[test]
    fn test_find_similar_with_metric_cosine_backward_compat() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let mut c = create_test_claim(1, "metric test");
        c.embedding = vec![1.0, 0.0, 0.0];
        store.store(&c).unwrap();

        let r1 = store.find_similar(&[1.0, 0.0, 0.0], 5, 0.5).unwrap();
        let r2 = store
            .find_similar_with_metric(
                &[1.0, 0.0, 0.0],
                5,
                0.5,
                super::super::embeddings::DistanceMetric::Cosine,
            )
            .unwrap();
        // Both should return the same claim with the same score
        assert_eq!(r1.len(), r2.len());
        assert_eq!(r1[0].0, r2[0].0);
        assert!((r1[0].1 - r2[0].1).abs() < 0.001);
    }

    #[test]
    fn test_find_similar_euclidean_metric() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let mut c1 = create_test_claim(1, "close");
        c1.embedding = vec![1.0, 0.0, 0.0];
        store.store(&c1).unwrap();

        let mut c2 = create_test_claim(2, "far");
        c2.embedding = vec![0.0, 0.0, 1.0];
        store.store(&c2).unwrap();

        let results = store
            .find_similar_with_metric(
                &[1.0, 0.0, 0.0],
                5,
                0.0,
                super::super::embeddings::DistanceMetric::Euclidean,
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        // c1 should be first (closer to query)
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 > results[1].1);
    }

    // ── P5: Pending buffer tests ─────────────────────────────────────────

    #[test]
    fn test_pending_buffer_store_and_apply() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        // Store 5 claims — they go to pending queue
        for i in 1..=5 {
            let mut c = create_test_claim(i, &format!("Pending claim {}", i));
            c.embedding = vec![1.0, 0.0, 0.0];
            store.store(&c).unwrap();
        }

        // Pending count should be 5 (not yet applied)
        assert!(store.pending_count() >= 5);

        // After apply_pending, index should have 5 entries
        store.apply_pending();
        assert_eq!(store.vector_index_size(), 5);
        assert_eq!(store.pending_count(), 0);
    }

    #[test]
    fn test_search_auto_applies_pending() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let mut c = create_test_claim(1, "Auto apply test");
        c.embedding = vec![1.0, 0.0, 0.0];
        store.store(&c).unwrap();

        // Search should auto-apply pending updates and find the claim
        let results = store.find_similar(&[1.0, 0.0, 0.0], 5, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    // ── P3: record_outcome tests ──────────────────────────────────────────

    #[test]
    fn test_record_outcome_persists() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let claim = create_test_claim(1, "Outcome test");
        store.store(&claim).unwrap();

        // Record outcomes
        let updated = store.record_outcome(1, true).unwrap();
        assert!(updated.is_some());
        let updated = updated.unwrap();
        assert_eq!(updated.positive_outcomes(), 1);
        assert_eq!(updated.negative_outcomes(), 0);

        store.record_outcome(1, false).unwrap();
        store.record_outcome(1, true).unwrap();

        // Verify persisted via fresh read
        let claim = store.get(1).unwrap().unwrap();
        assert_eq!(claim.positive_outcomes(), 2);
        assert_eq!(claim.negative_outcomes(), 1);
        assert_eq!(claim.total_outcomes(), 3);
    }

    #[test]
    fn test_record_outcome_nonexistent_returns_none() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let result = store.record_outcome(999, true).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_bm25_index_populated_via_pending() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        let mut c = create_test_claim(1, "The quick brown fox jumps");
        c.embedding = vec![1.0, 0.0, 0.0];
        store.store(&c).unwrap();

        // Apply pending to populate BM25
        store.apply_pending();

        // BM25 should find the claim
        let bm25 = store.bm25_index().read();
        let results = bm25.search("quick fox", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_find_avoidance_claims() {
        let dir = tempdir().unwrap();
        let store = ClaimStore::new(dir.path().join("claims.redb")).unwrap();

        // Create 3 claims: 1 Avoidance, 1 Fact, 1 Avoidance
        let mut c1 = create_test_claim(1, "Avoid relying on: flaky API");
        c1.claim_type = ClaimType::Avoidance;
        c1.created_at = 1000;
        store.store(&c1).unwrap();

        let mut c2 = create_test_claim(2, "The API uses REST");
        c2.claim_type = ClaimType::Fact;
        store.store(&c2).unwrap();

        let mut c3 = create_test_claim(3, "Avoid relying on: cached state");
        c3.claim_type = ClaimType::Avoidance;
        c3.created_at = 2000; // more recent
        store.store(&c3).unwrap();

        let avoidance = store.find_avoidance_claims(10).unwrap();
        assert_eq!(avoidance.len(), 2, "Should find 2 avoidance claims");
        // Most recent first
        assert_eq!(avoidance[0].id, 3, "Most recent avoidance claim first");
        assert_eq!(avoidance[1].id, 1);

        // Test limit
        let limited = store.find_avoidance_claims(1).unwrap();
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].id, 3);
    }
}
