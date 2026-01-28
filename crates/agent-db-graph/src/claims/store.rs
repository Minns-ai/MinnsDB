//! Persistent storage for claims using redb

use super::types::*;
use anyhow::Result;
use redb::{Database, TableDefinition, ReadableTableMetadata, ReadableTable};
use std::path::Path;
use tracing::{debug, info};

const CLAIMS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");
const CLAIM_COUNTER: TableDefinition<&str, u64> = TableDefinition::new("claim_counter");

/// Persistent store for derived claims
pub struct ClaimStore {
    db: Database,
}

impl ClaimStore {
    /// Create a new claim store at the given path
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

        info!("Claim store initialized successfully");

        Ok(Self { db })
    }

    /// Store a claim
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
                debug!("Added support to claim {} (now: {})", claim_id, claim.support_count);
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

    /// Find similar claims using vector similarity search
    ///
    /// Returns (claim_id, similarity_score) pairs sorted by descending similarity
    pub fn find_similar(&self, query_embedding: &[f32], top_k: usize, min_similarity: f32) -> Result<Vec<(ClaimId, f32)>> {
        use crate::claims::embeddings::VectorSimilarity;

        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        let mut similarities = Vec::new();
        let iter = table.iter()?;

        // Brute-force search through all claims
        // TODO: For large datasets, consider using HNSW or other ANN index
        for item in iter {
            let (id_guard, value) = item?;
            let id = id_guard.value();
            let claim: DerivedClaim = bincode::deserialize(value.value())?;

            // Only search active claims with embeddings
            if claim.status == ClaimStatus::Active && !claim.embedding.is_empty() {
                let similarity = VectorSimilarity::cosine_similarity(query_embedding, &claim.embedding);

                if similarity >= min_similarity {
                    similarities.push((id, similarity));
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        similarities.truncate(top_k);

        debug!("Found {} similar claims (top_k={}, min_similarity={})", similarities.len(), top_k, min_similarity);

        Ok(similarities)
    }

    /// Update claim embedding
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
                claim.embedding = embedding;
                let serialized = bincode::serialize(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
                debug!("Updated embedding for claim {} ({} dimensions)", claim_id, claim.embedding.len());
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
}
