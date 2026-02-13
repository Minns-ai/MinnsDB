//! Async embedding generation queue for claims

use super::embeddings::{EmbeddingClient, EmbeddingRequest};
use super::store::ClaimStore;
use super::types::DerivedClaim;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Embedding generation job
struct EmbeddingJob {
    claim: DerivedClaim,
    result_tx: Option<tokio::sync::oneshot::Sender<Result<()>>>,
}

/// Default bounded channel capacity for the embedding queue
const DEFAULT_EMBEDDING_QUEUE_CAPACITY: usize = 1_000;

/// Async queue for embedding generation
pub struct EmbeddingQueue {
    sender: mpsc::Sender<EmbeddingJob>,
}

impl EmbeddingQueue {
    /// Create a new embedding queue with worker pool
    pub fn new(
        embedding_client: Arc<dyn EmbeddingClient>,
        claim_store: Arc<ClaimStore>,
        workers: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<EmbeddingJob>(DEFAULT_EMBEDDING_QUEUE_CAPACITY);
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        info!(
            "Starting embedding generation queue with {} workers",
            workers
        );

        // Spawn worker pool
        for worker_id in 0..workers {
            let rx = rx.clone();
            let embedding_client = embedding_client.clone();
            let claim_store = claim_store.clone();

            tokio::spawn(async move {
                info!("Embedding worker {} started", worker_id);

                loop {
                    // Lock and receive job
                    let job = {
                        let mut rx_guard = rx.lock().await;
                        rx_guard.recv().await
                    };

                    match job {
                        Some(job) => {
                            let result =
                                Self::process_job(job.claim, &*embedding_client, &claim_store)
                                    .await;

                            // Send result if channel exists
                            if let Some(tx) = job.result_tx {
                                let _ = tx.send(result);
                            } else if let Err(e) = &result {
                                error!(
                                    "Worker {} failed to process embedding job: {}",
                                    worker_id, e
                                );
                            }
                        },
                        None => {
                            info!("Embedding worker {} channel closed, stopping", worker_id);
                            break;
                        },
                    }
                }

                info!("Embedding worker {} stopped", worker_id);
            });
        }

        Self { sender: tx }
    }

    /// Process a single embedding generation job
    async fn process_job(
        claim: DerivedClaim,
        embedding_client: &dyn EmbeddingClient,
        claim_store: &ClaimStore,
    ) -> Result<()> {
        debug!("Generating embedding for claim {}", claim.id);

        // Generate embedding for claim text
        let request = EmbeddingRequest {
            text: claim.claim_text.clone(),
            context: None, // Could add evidence context here if needed
        };

        let response = embedding_client.embed(request).await?;

        // Update claim with embedding
        claim_store.update_embedding(claim.id, response.embedding)?;

        info!(
            "Generated embedding for claim {} ({} dimensions, {} tokens)",
            claim.id,
            embedding_client.dimensions(),
            response.tokens_used
        );

        Ok(())
    }

    /// Submit a claim for embedding generation (blocks if queue is full)
    pub async fn generate_embedding(&self, claim: DerivedClaim) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender
            .send(EmbeddingJob {
                claim,
                result_tx: Some(tx),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send embedding job: {}", e))?;

        rx.await
            .map_err(|e| anyhow::anyhow!("Failed to receive embedding result: {}", e))?
    }

    /// Submit embedding generation without waiting for result (drops if queue full)
    pub fn generate_embedding_async(&self, claim: DerivedClaim) {
        if let Err(e) = self.sender.try_send(EmbeddingJob {
            claim,
            result_tx: None,
        }) {
            warn!("Embedding queue full or closed, dropping job: {}", e);
        }
    }

    /// Process all claims that need embeddings
    pub async fn process_pending_embeddings(
        &self,
        claim_store: &ClaimStore,
        batch_size: usize,
    ) -> Result<usize> {
        debug!("Processing pending embeddings (batch_size={})", batch_size);

        let claims = claim_store.get_claims_needing_embeddings(batch_size)?;
        let count = claims.len();

        if count == 0 {
            debug!("No claims need embeddings");
            return Ok(0);
        }

        info!("Found {} claims needing embeddings", count);

        // Submit all claims for embedding generation
        for claim in claims {
            self.generate_embedding_async(claim);
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::embeddings::MockEmbeddingClient;
    use crate::claims::types::EvidenceSpan;
    use crate::ClaimId;
    use tempfile::tempdir;

    fn create_test_claim(id: ClaimId, text: &str) -> DerivedClaim {
        let evidence = vec![EvidenceSpan::new(0, 4, "test")];
        DerivedClaim::new(
            id,
            text.to_string(),
            evidence,
            0.9,
            vec![], // Empty embedding
            123,
            None,
            None,
            None,
            None,
        )
    }

    #[tokio::test]
    async fn test_embedding_queue() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");

        let client = Arc::new(MockEmbeddingClient::new(384));
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());

        let queue = EmbeddingQueue::new(client, store.clone(), 1);

        // Create and store a claim without embedding
        let claim = create_test_claim(1, "Test claim");
        store.store(&claim).unwrap();

        // Verify it has no embedding
        let retrieved = store.get(1).unwrap().unwrap();
        assert!(retrieved.embedding.is_empty());

        // Generate embedding
        queue.generate_embedding(claim).await.unwrap();

        // Verify embedding was added
        let retrieved = store.get(1).unwrap().unwrap();
        assert!(!retrieved.embedding.is_empty());
        assert_eq!(retrieved.embedding.len(), 384);
    }

    #[tokio::test]
    async fn test_process_pending_embeddings() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");

        let client = Arc::new(MockEmbeddingClient::new(384));
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());

        let queue = EmbeddingQueue::new(client, store.clone(), 2);

        // Create multiple claims without embeddings
        for i in 1..=5 {
            let claim = create_test_claim(i, &format!("Claim {}", i));
            store.store(&claim).unwrap();
        }

        // Process pending embeddings
        let count = queue.process_pending_embeddings(&store, 10).await.unwrap();
        assert_eq!(count, 5);

        // Wait a bit for async processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify all claims now have embeddings
        let claims_needing = store.get_claims_needing_embeddings(10).unwrap();
        assert_eq!(claims_needing.len(), 0);
    }
}
