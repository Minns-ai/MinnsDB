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

    /// Submit embedding generation without waiting for result.
    ///
    /// Retries up to 3 times with a short yield between attempts if the
    /// queue is full, to avoid silently losing embedding jobs under
    /// transient backpressure.
    pub fn generate_embedding_async(&self, claim: DerivedClaim) {
        let sender = self.sender.clone();
        let claim_id = claim.id;
        tokio::spawn(async move {
            let mut job = Some(EmbeddingJob {
                claim,
                result_tx: None,
            });
            for attempt in 0..3u32 {
                match sender.try_send(job.take().unwrap()) {
                    Ok(()) => return,
                    Err(mpsc::error::TrySendError::Full(returned)) => {
                        job = Some(returned);
                        warn!(
                            "Embedding queue full (attempt {}/3) for claim {}, retrying after yield",
                            attempt + 1,
                            claim_id
                        );
                        tokio::task::yield_now().await;
                    },
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        error!(
                            "Embedding queue closed, permanently dropping claim {}",
                            claim_id
                        );
                        return;
                    },
                }
            }
            // Final attempt: await with a timeout so we don't block forever
            if let Some(final_job) = job {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    sender.send(final_job),
                ).await {
                    Ok(Ok(())) => {},
                    Ok(Err(_)) => error!("Embedding queue closed, permanently dropping claim {}", claim_id),
                    Err(_) => error!(
                        "Embedding queue full for >5s, dropping claim {} (consider increasing queue capacity)",
                        claim_id
                    ),
                }
            }
        });
    }

    /// Process all claims that need embeddings using batch API.
    ///
    /// Collects up to `batch_size` claims and sends them in a single
    /// `embed_batch()` call for much higher throughput.
    pub async fn process_pending_embeddings(
        &self,
        claim_store: &ClaimStore,
        embedding_client: &dyn super::embeddings::EmbeddingClient,
        batch_size: usize,
    ) -> Result<usize> {
        debug!("Processing pending embeddings (batch_size={})", batch_size);

        let claims = claim_store.get_claims_needing_embeddings(batch_size)?;
        let count = claims.len();

        if count == 0 {
            debug!("No claims need embeddings");
            return Ok(0);
        }

        info!("Found {} claims needing embeddings, sending batch", count);

        // Build batch request
        let requests: Vec<super::embeddings::EmbeddingRequest> = claims
            .iter()
            .map(|c| super::embeddings::EmbeddingRequest {
                text: c.claim_text.clone(),
                context: None,
            })
            .collect();

        // Single batch call
        let responses = embedding_client.embed_batch(requests).await?;

        // Zip responses with claims and update store
        let mut updated = 0usize;
        for (claim, response) in claims.iter().zip(responses.into_iter()) {
            if let Err(e) = claim_store.update_embedding(claim.id, response.embedding) {
                error!("Failed to update embedding for claim {}: {}", claim.id, e);
            } else {
                updated += 1;
            }
        }

        info!(
            "Batch embedding complete: {}/{} claims updated",
            updated, count
        );
        Ok(updated)
    }

    /// Process pending embeddings using the old sequential approach (one-at-a-time via queue workers).
    pub async fn process_pending_embeddings_queued(
        &self,
        claim_store: &ClaimStore,
        batch_size: usize,
    ) -> Result<usize> {
        let claims = claim_store.get_claims_needing_embeddings(batch_size)?;
        let count = claims.len();
        if count == 0 {
            return Ok(0);
        }
        for claim in claims {
            self.generate_embedding_async(claim);
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::embeddings::openai_client_from_env;
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
    #[ignore = "requires LLM_API_KEY — run with: cargo test --ignored"]
    async fn test_embedding_queue() {
        let client = Arc::new(openai_client_from_env().expect("LLM_API_KEY required"));
        let dims = client.dimensions();

        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());

        let queue = EmbeddingQueue::new(client, store.clone(), 1);

        // Create and store a claim without embedding
        let claim = create_test_claim(1, "Test claim about artificial intelligence");
        store.store(&claim).unwrap();

        // Verify it has no embedding
        let retrieved = store.get(1).unwrap().unwrap();
        assert!(retrieved.embedding.is_empty());

        // Generate embedding
        queue.generate_embedding(claim).await.unwrap();

        // Verify embedding was added
        let retrieved = store.get(1).unwrap().unwrap();
        assert!(!retrieved.embedding.is_empty());
        assert_eq!(retrieved.embedding.len(), dims);
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY — run with: cargo test --ignored"]
    async fn test_process_pending_embeddings_batch() {
        let client = Arc::new(openai_client_from_env().expect("LLM_API_KEY required"));

        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());

        let queue = EmbeddingQueue::new(client.clone(), store.clone(), 2);

        // Create multiple claims without embeddings
        for i in 1..=5 {
            let claim = create_test_claim(i, &format!("Claim number {} about different topics", i));
            store.store(&claim).unwrap();
        }

        // Process pending embeddings via batch API
        let count = queue
            .process_pending_embeddings(&store, &*client, 10)
            .await
            .unwrap();
        assert_eq!(count, 5);

        // Verify all claims now have embeddings
        let claims_needing = store.get_claims_needing_embeddings(10).unwrap();
        assert_eq!(claims_needing.len(), 0);
    }

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY — run with: cargo test --ignored"]
    async fn test_batch_embed_20_claims() {
        let client = Arc::new(openai_client_from_env().expect("LLM_API_KEY required"));
        let dims = client.dimensions();

        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());

        let queue = EmbeddingQueue::new(client.clone(), store.clone(), 1);

        // Store 20 claims without embeddings
        for i in 1..=20 {
            let claim =
                create_test_claim(i, &format!("Batch claim number {} about topic {}", i, i));
            store.store(&claim).unwrap();
        }

        let count = queue
            .process_pending_embeddings(&store, &*client, 100)
            .await
            .unwrap();
        assert_eq!(count, 20);

        // All 20 should have embeddings now
        let remaining = store.get_claims_needing_embeddings(100).unwrap();
        assert_eq!(remaining.len(), 0);

        // Verify embeddings are correct dimension
        for i in 1..=20u64 {
            let claim = store.get(i).unwrap().unwrap();
            assert_eq!(claim.embedding.len(), dims);
        }
    }
}
