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

        // Generate embedding using per-type text with structured context
        let (text, context) = claim.embedding_text();
        let request = EmbeddingRequest { text, context };

        let response = embedding_client.embed(request).await?;

        // Update claim with embedding
        claim_store
            .update_embedding(claim.id, response.embedding)
            .await?;

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
            .map(|c| {
                let (text, context) = c.embedding_text();
                super::embeddings::EmbeddingRequest { text, context }
            })
            .collect();

        // Single batch embed RPC
        let responses = embedding_client.embed_batch(requests).await?;

        // Build a batch of vector points and the corresponding redb updates.
        // We split the work so the redb write happens for all claims (cheap,
        // sync), then the vector upsert is a single RPC for the whole batch.
        let mut points = Vec::with_capacity(responses.len());
        let mut bm25_updates: Vec<(u64, String)> = Vec::with_capacity(responses.len());
        let mut updated = 0usize;
        for (claim, response) in claims.iter().zip(responses.into_iter()) {
            if response.embedding.is_empty() {
                continue;
            }
            match claim_store.mark_has_embedding(claim.id) {
                Ok(true) => {
                    points.push(minns_vectors::Point::new(
                        claim.id as u128,
                        response.embedding,
                        minns_vectors::Payload::EMPTY,
                    ));
                    bm25_updates.push((claim.id, claim.claim_text.clone()));
                    updated += 1;
                },
                Ok(false) => {
                    debug!("Claim {} no longer active; skipping embedding", claim.id);
                },
                Err(e) => {
                    error!("Failed to mark embedding for claim {}: {}", claim.id, e);
                },
            }
        }

        if !points.is_empty() {
            if let Err(e) = claim_store.upsert_vectors(points).await {
                error!("Batch vector upsert failed: {}", e);
                return Err(e);
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

// Tests live in `tests/claims_integration.rs` and exercise the queue end-to-end
// against a real Qdrant container and (when `LLM_API_KEY` is set) a real
// embedding client.
