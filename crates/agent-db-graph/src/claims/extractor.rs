//! Claim extraction pipeline with validation and geometric checks

use super::embeddings::{EmbeddingClient, EmbeddingRequest, VectorSimilarity};
use super::llm_client::{LlmClient, LlmExtractionRequest};
use super::store::ClaimStore;
use super::types::*;
use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Claim extraction job
struct ClaimExtractionJob {
    request: ClaimExtractionRequest,
    result_tx: tokio::sync::oneshot::Sender<Result<ClaimExtractionResult>>,
}

/// Async queue for claim extraction
pub struct ClaimExtractionQueue {
    sender: mpsc::UnboundedSender<ClaimExtractionJob>,
}

impl ClaimExtractionQueue {
    /// Create a new extraction queue with worker pool
    pub fn new(
        llm_client: Arc<dyn LlmClient>,
        embedding_client: Arc<dyn EmbeddingClient>,
        claim_store: Arc<ClaimStore>,
        workers: usize,
        config: ClaimExtractionConfig,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        info!("Starting claim extraction queue with {} workers", workers);

        // Spawn worker pool
        for worker_id in 0..workers {
            let rx = rx.clone();
            let llm_client = llm_client.clone();
            let embedding_client = embedding_client.clone();
            let claim_store = claim_store.clone();
            let config = config.clone();

            tokio::spawn(async move {
                info!("Claim extraction worker {} started", worker_id);

                loop {
                    // Lock and receive job
                    let job = {
                        let mut rx_guard = rx.lock().await;
                        rx_guard.recv().await
                    };

                    match job {
                        Some(job) => {
                            let result = Self::process_job(
                                job,
                                &*llm_client,
                                &*embedding_client,
                                &*claim_store,
                                &config,
                            )
                            .await;

                            // Errors are logged but don't crash the worker
                            if let Err(e) = &result {
                                error!("Worker {} failed to process job: {}", worker_id, e);
                            }
                        },
                        None => {
                            info!(
                                "Claim extraction worker {} channel closed, stopping",
                                worker_id
                            );
                            break;
                        },
                    }
                }

                info!("Claim extraction worker {} stopped", worker_id);
            });
        }

        Self { sender: tx }
    }

    /// Process a single claim extraction job
    async fn process_job(
        job: ClaimExtractionJob,
        llm_client: &dyn LlmClient,
        embedding_client: &dyn EmbeddingClient,
        claim_store: &ClaimStore,
        config: &ClaimExtractionConfig,
    ) -> Result<()> {
        let request = job.request;
        let start_time = std::time::Instant::now();

        debug!("Processing claim extraction for event {}", request.event_id);

        let result = async {
            // Step 1: Extract entities from NER features for the context
            let context_sentence_entities = if let Some(ner) = &request.ner_features {
                ner.sentence_entities.clone().unwrap_or_else(|| {
                    // Fallback to splitting if sentence_entities is missing but ner_features exists
                    agent_db_events::SentenceEntities::split_into_sentences(&request.canonical_text)
                })
            } else {
                // Free Tier fallback: Split into sentences even without NER
                agent_db_events::SentenceEntities::split_into_sentences(&request.canonical_text)
            };

            let all_context_entities: Vec<String> = if let Some(ner) = &request.ner_features {
                ner.entity_spans
                    .iter()
                    .map(|span| span.text.clone())
                    .collect()
            } else {
                Vec::new()
            };

            // Step 2: Call LLM for claim extraction
            let llm_request = LlmExtractionRequest {
                text: request.canonical_text.clone(),
                entities: all_context_entities.clone(),
                max_claims: config.max_claims_per_input,
            };

            let llm_response = llm_client.extract_claims(llm_request).await?;

            // Step 3: Validate and process claims using the NER Minimal Algorithm
            let mut accepted_claims = Vec::new();
            let mut rejected_claims = Vec::new();

            for llm_claim in llm_response.claims {
                // 3.1 Run NER on the claim text
                // Since we don't have a direct NER extractor here, we'll use literal matches
                // from the already extracted context entities as a high-confidence proxy.
                let mut e_claim = HashSet::new();
                for entity in &all_context_entities {
                    if llm_claim.claim_text.contains(entity) {
                        e_claim.insert(entity.clone());
                    }
                }

                // 3.2 Select candidate evidence sentences
                let mut candidate_sentences = Vec::new();
                for (i, sent) in context_sentence_entities.iter().enumerate() {
                    let mut e_sent = HashSet::new();
                    for ent in &sent.entities {
                        e_sent.insert(ent.text.clone());
                    }

                    // keep sentences where E_claim ∩ E_sent[i] is non-empty
                    let overlap_entities: HashSet<_> = e_claim.intersection(&e_sent).collect();
                    if !overlap_entities.is_empty() {
                        candidate_sentences.push((i, sent.clone(), 1.0)); // Initial score 1.0 for entity overlap
                    }
                }

                // fallback to top-k sentences by semantic similarity if none found
                if candidate_sentences.is_empty() && !context_sentence_entities.is_empty() {
                    // Generate embedding for claim
                    let claim_embedding = embedding_client
                        .embed(EmbeddingRequest {
                            text: llm_claim.claim_text.clone(),
                            context: None,
                        })
                        .await?
                        .embedding;

                    let mut scored_sentences = Vec::new();
                    for (i, sent) in context_sentence_entities.iter().enumerate() {
                        let sent_embedding = embedding_client
                            .embed(EmbeddingRequest {
                                text: sent.text.clone(),
                                context: None,
                            })
                            .await?
                            .embedding;

                        let sim =
                            VectorSimilarity::cosine_similarity(&claim_embedding, &sent_embedding);
                        scored_sentences.push((i, sent.clone(), sim));
                    }

                    scored_sentences
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    candidate_sentences = scored_sentences.into_iter().take(3).collect();
                }

                if candidate_sentences.is_empty() {
                    rejected_claims.push(RejectedClaim {
                        claim_text: llm_claim.claim_text,
                        rejection_reason: RejectionReason::NoEvidence,
                    });
                    continue;
                }

                // 3.3 Compute geometric score over candidate sentences and take the best one
                let claim_embedding = embedding_client
                    .embed(EmbeddingRequest {
                        text: llm_claim.claim_text.clone(),
                        context: None,
                    })
                    .await?
                    .embedding;

                let mut best_sentence = None;
                let mut max_support_score = -1.0;
                let mut best_diagnostics = None;

                for (sent_idx, sent, _initial_sim) in candidate_sentences {
                    let sent_embedding = embedding_client
                        .embed(EmbeddingRequest {
                            text: sent.text.clone(),
                            context: None,
                        })
                        .await?
                        .embedding;

                    let similarity =
                        VectorSimilarity::cosine_similarity(&claim_embedding, &sent_embedding);
                    let _distance = 1.0 - similarity;

                    // overlap = |E_claim ∩ E_evidence| / max(1, |E_claim|)
                    let mut e_evidence = HashSet::new();
                    for ent in &sent.entities {
                        e_evidence.insert(ent.text.clone());
                    }
                    let intersection_count = e_claim.intersection(&e_evidence).count() as f32;
                    let overlap = intersection_count / (e_claim.len().max(1) as f32);

                    // exact_entity_presence: Check if entities in claim are literally in sentence
                    let mut missing_entities = Vec::new();
                    for ent in &e_claim {
                        if !sent.text.contains(ent) {
                            missing_entities.push(ent.clone());
                        }
                    }
                    let exact_entity_presence =
                        1.0 - (missing_entities.len() as f32 / e_claim.len().max(1) as f32);

                    // support = w1 * (1 - distance) + w2 * overlap + w3 * exact_entity_presence
                    // Weights: similarity (0.5), NER overlap (0.3), Literal match (0.2)
                    let support_score = if e_claim.is_empty() {
                        // Fallback: If no entities are found (Free Tier or abstract claim),
                        // use 100% semantic similarity.
                        similarity
                    } else {
                        (0.5 * similarity) + (0.3 * overlap) + (0.2 * exact_entity_presence)
                    };

                    if support_score > max_support_score {
                        max_support_score = support_score;
                        best_sentence = Some((sent, sent_idx));
                        best_diagnostics = Some((similarity, overlap, missing_entities));
                    }
                }

                // 3.4 Check support threshold
                if max_support_score < config.min_confidence {
                    rejected_claims.push(RejectedClaim {
                        claim_text: llm_claim.claim_text,
                        rejection_reason: RejectionReason::BelowConfidenceThreshold,
                    });
                    continue;
                }

                let (best_sent, best_sent_idx) = best_sentence.unwrap();
                let (similarity, overlap, missing_entities) = best_diagnostics.unwrap();

                // Create EvidenceSpan from the best sentence
                let evidence = vec![EvidenceSpan::new(
                    best_sent.start_offset,
                    best_sent.start_offset + best_sent.text.len(),
                    &best_sent.text,
                )];

                // Create claim
                let claim_id = claim_store.next_id()?;
                let mut claim = DerivedClaim::new(
                    claim_id,
                    llm_claim.claim_text,
                    evidence,
                    max_support_score,
                    claim_embedding,
                    request.event_id,
                    request.episode_id,
                    request.thread_id.clone(),
                    request.user_id.clone(),
                    request.workspace_id.clone(),
                );

                // Add diagnostics to metadata
                claim
                    .metadata
                    .insert("similarity".to_string(), similarity.to_string());
                claim
                    .metadata
                    .insert("overlap".to_string(), overlap.to_string());
                claim.metadata.insert(
                    "found_entities".to_string(),
                    e_claim.iter().cloned().collect::<Vec<_>>().join(", "),
                );
                claim
                    .metadata
                    .insert("missing_entities".to_string(), missing_entities.join(", "));
                claim
                    .metadata
                    .insert("sentence_index".to_string(), best_sent_idx.to_string());

                // Store claim
                claim_store.store(&claim)?;
                accepted_claims.push(claim);
            }

            let extraction_time_ms = start_time.elapsed().as_millis() as u64;

            info!(
                "Claim extraction complete for event {}: {} accepted, {} rejected, {} tokens, {}ms",
                request.event_id,
                accepted_claims.len(),
                rejected_claims.len(),
                llm_response.tokens_used,
                extraction_time_ms
            );

            Ok(ClaimExtractionResult {
                accepted_claims,
                rejected_claims,
                tokens_used: llm_response.tokens_used,
                extraction_time_ms,
            })
        }
        .await;

        // Send result back
        let _ = job.result_tx.send(result);

        Ok(())
    }

    /// Submit a claim extraction request
    pub async fn extract(&self, request: ClaimExtractionRequest) -> Result<ClaimExtractionResult> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender
            .send(ClaimExtractionJob {
                request,
                result_tx: tx,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send claim extraction job: {}", e))?;

        rx.await
            .map_err(|e| anyhow::anyhow!("Failed to receive claim extraction result: {}", e))?
    }

    /// Submit extraction without waiting for result
    pub fn extract_async(
        &self,
        request: ClaimExtractionRequest,
    ) -> tokio::sync::oneshot::Receiver<Result<ClaimExtractionResult>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        if let Err(e) = self.sender.send(ClaimExtractionJob {
            request,
            result_tx: tx,
        }) {
            warn!("Failed to send async claim extraction job: {}", e);
        }

        rx
    }
}

/// Configuration for claim extraction
#[derive(Debug, Clone)]
pub struct ClaimExtractionConfig {
    /// Maximum claims per input
    pub max_claims_per_input: usize,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Minimum evidence length (characters)
    pub min_evidence_length: usize,
}

impl Default for ClaimExtractionConfig {
    fn default() -> Self {
        Self {
            max_claims_per_input: 10,
            min_confidence: 0.7,
            min_evidence_length: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::llm_client::MockClient;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_queue_creation() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");

        let client = Arc::new(MockClient::new());
        let embedding_client = Arc::new(crate::claims::embeddings::MockEmbeddingClient::new(384));
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());
        let config = ClaimExtractionConfig::default();

        let _queue = ClaimExtractionQueue::new(client, embedding_client, store, 1, config);
    }
}
