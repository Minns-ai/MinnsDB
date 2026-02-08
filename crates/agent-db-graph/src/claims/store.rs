//! Persistent storage for claims using redb
//!
//! The `ClaimStore` persists claims to redb and maintains an in-memory vector
//! index for fast similarity search.  On construction it loads all active
//! embeddings from disk so that `find_similar` never touches redb.

use super::types::*;
use anyhow::Result;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::path::Path;
use std::sync::RwLock;
use tracing::{debug, info};

const CLAIMS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");
const CLAIM_COUNTER: TableDefinition<&str, u64> = TableDefinition::new("claim_counter");

// ── In-memory vector index ──────────────────────────────────────────────────

/// Entry in the in-memory vector index.
/// Embeddings are pre-normalized to unit length so cosine similarity = dot product.
struct VectorEntry {
    id: ClaimId,
    embedding: Vec<f32>,
}

/// Flat vector index kept in RAM.  All embeddings are L2-normalized on insert
/// so `find_similar` only needs a dot-product scan (no sqrt / magnitude).
struct VectorIndex {
    entries: Vec<VectorEntry>,
}

impl VectorIndex {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
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
    }

    /// Remove an entry by id.
    fn remove(&mut self, id: ClaimId) {
        self.entries.retain(|e| e.id != id);
    }

    /// Dot-product scan.  `query` must already be L2-normalized.
    fn find_similar(
        &self,
        query_normalized: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<(ClaimId, f32)> {
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

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

// ── ClaimStore ──────────────────────────────────────────────────────────────

/// Persistent store for derived claims backed by redb with an in-memory
/// vector index for O(n·d) dot-product similarity search (no disk I/O).
pub struct ClaimStore {
    db: Database,
    /// In-memory vector index (pre-normalized embeddings, dot-product scan).
    index: RwLock<VectorIndex>,
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

        // Load existing embeddings into in-memory index
        let mut index = VectorIndex::new();
        {
            let read_txn = db.begin_read()?;
            let table = read_txn.open_table(CLAIMS_TABLE)?;
            let mut loaded = 0usize;
            for item in table.iter()? {
                let (id_guard, value) = item?;
                let claim: DerivedClaim = bincode::deserialize(value.value())?;
                if claim.status == ClaimStatus::Active && !claim.embedding.is_empty() {
                    index.upsert(id_guard.value(), claim.embedding);
                    loaded += 1;
                }
            }
            if loaded > 0 {
                info!(
                    "Loaded {} claim embeddings into in-memory vector index",
                    loaded
                );
            }
        }

        info!("Claim store initialized successfully");

        Ok(Self {
            db,
            index: RwLock::new(index),
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
            let serialized = bincode::serialize(claim)?;
            table.insert(claim.id, serialized.as_slice())?;
        }
        write_txn.commit()?;

        // Update in-memory vector index
        if claim.status == ClaimStatus::Active && !claim.embedding.is_empty() {
            if let Ok(mut idx) = self.index.write() {
                idx.upsert(claim.id, claim.embedding.clone());
            }
        }

        debug!("Successfully stored claim {}", claim.id);

        Ok(())
    }

    /// Get a claim by ID
    pub fn get(&self, claim_id: ClaimId) -> Result<Option<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        if let Some(value) = table.get(claim_id)? {
            let claim: DerivedClaim = bincode::deserialize(value.value())?;
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
                let claim: DerivedClaim = bincode::deserialize(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.support_count += 1;
                let serialized = bincode::serialize(&claim)?;
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
                let claim: DerivedClaim = bincode::deserialize(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.status = status;
                let serialized = bincode::serialize(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                debug!("Updated claim {} status to {:?}", claim_id, status);

                // Update vector index: remove if no longer active
                if status != ClaimStatus::Active {
                    if let Ok(mut idx) = self.index.write() {
                        idx.remove(claim_id);
                    }
                }
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
                let claim: DerivedClaim = bincode::deserialize(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.mark_accessed();
                let serialized = bincode::serialize(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Delete a claim
    pub fn delete(&self, claim_id: ClaimId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            table.remove(claim_id)?;
        }
        write_txn.commit()?;

        // Remove from in-memory index
        if let Ok(mut idx) = self.index.write() {
            idx.remove(claim_id);
        }

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
            let claim: DerivedClaim = bincode::deserialize(value.value())?;

            if claim.status == ClaimStatus::Active {
                claims.push(claim);
            }
        }

        Ok(claims)
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
            .map_err(|e| anyhow::anyhow!("Vector index read lock poisoned: {}", e))?
            .find_similar(&query_norm, top_k, min_similarity);

        debug!(
            "Found {} similar claims (top_k={}, min_similarity={}) via in-memory index",
            results.len(),
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
                let claim: DerivedClaim = bincode::deserialize(value.value())?;
                Some(claim)
            } else {
                None
            };

            if let Some(mut claim) = maybe_claim {
                claim.embedding = embedding.clone();
                let serialized = bincode::serialize(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                debug!(
                    "Updated embedding for claim {} ({} dimensions)",
                    claim_id,
                    claim.embedding.len()
                );

                // Update in-memory vector index
                if claim.status == ClaimStatus::Active {
                    if let Ok(mut idx) = self.index.write() {
                        idx.upsert(claim_id, embedding);
                    }
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
            let claim: DerivedClaim = bincode::deserialize(value.value())?;

            if claim.status == ClaimStatus::Active && claim.embedding.is_empty() {
                claims.push(claim);
            }
        }

        Ok(claims)
    }

    /// Number of embeddings in the in-memory vector index.
    pub fn vector_index_size(&self) -> usize {
        self.index.read().map(|idx| idx.entries.len()).unwrap_or(0)
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
}
