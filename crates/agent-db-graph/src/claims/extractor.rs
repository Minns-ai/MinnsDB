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

/// Default bounded channel capacity for the claim extraction queue
const DEFAULT_EXTRACTION_QUEUE_CAPACITY: usize = 500;

/// Async queue for claim extraction
pub struct ClaimExtractionQueue {
    sender: mpsc::Sender<ClaimExtractionJob>,
}

impl ClaimExtractionQueue {
    /// Create a new extraction queue with worker pool
    pub fn new(
        llm_client: Arc<dyn LlmClient>,
        embedding_client: Arc<dyn EmbeddingClient>,
        claim_store: Arc<ClaimStore>,
        workers: usize,
        config: ClaimExtractionConfig,
        unified_llm: Arc<dyn crate::llm_client::LlmClient>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(DEFAULT_EXTRACTION_QUEUE_CAPACITY);
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        info!("Starting claim extraction queue with {} workers", workers);

        // Spawn worker pool
        for worker_id in 0..workers {
            let rx = rx.clone();
            let llm_client = llm_client.clone();
            let embedding_client = embedding_client.clone();
            let claim_store = claim_store.clone();
            let config = config.clone();
            let unified_llm = unified_llm.clone();

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
                                &claim_store,
                                &config,
                                &*unified_llm,
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
        unified_llm: &dyn crate::llm_client::LlmClient,
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

            // Build labeled entities for LLM (text + NER label)
            let labeled_entities: Vec<super::llm_client::LabeledEntity> =
                if let Some(ner) = &request.ner_features {
                    ner.entity_spans
                        .iter()
                        .map(|span| super::llm_client::LabeledEntity {
                            text: span.text.clone(),
                            label: span.label.clone(),
                            confidence: span.confidence,
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            // Bare text list for backward-compat NER overlap checks below
            let all_context_entities: Vec<String> = labeled_entities
                .iter()
                .map(|e| e.text.clone())
                .collect();

            // Step 2: Call LLM for claim extraction
            let llm_request = LlmExtractionRequest {
                text: request.canonical_text.clone(),
                entities: labeled_entities.clone(),
                max_claims: config.max_claims_per_input,
                source_role: request.source_role,
                custom_instructions: config.custom_instructions.clone(),
                extraction_includes: config.extraction_includes.clone(),
                extraction_excludes: config.extraction_excludes.clone(),
                few_shot_examples: config.few_shot_examples.clone(),
                rolling_summary: request.rolling_summary.clone(),
            };

            let llm_response = llm_client.extract_claims(llm_request).await?;

            // Step 3: Validate and process claims using the NER Minimal Algorithm
            let mut accepted_claims = Vec::new();
            let mut rejected_claims = Vec::new();

            for llm_claim in llm_response.claims {
                // 3.1 Run NER on the claim text
                // Since we don't have a direct NER extractor here, we'll use literal matches
                // from the already extracted context entities as a high-confidence proxy.
                // Case-insensitive to match entity attachment logic (line ~386).
                let claim_text_lower = llm_claim.claim_text.to_lowercase();
                let mut e_claim = HashSet::new();
                for entity in &all_context_entities {
                    if claim_text_lower.contains(&entity.to_lowercase()) {
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

                    // keep sentences where any claim entity fuzzy-matches a sentence entity
                    let has_overlap = e_claim.iter().any(|ce| e_sent.iter().any(|se| entities_match_fuzzy(ce, se)));
                    if has_overlap {
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
                        .sort_by(|a, b| b.2.total_cmp(&a.2));
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
                let mut max_support_score = f32::NEG_INFINITY;
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
                    // Normalize both sides for case-insensitive comparison
                    let e_claim_norm: HashSet<String> =
                        e_claim.iter().map(|e| normalize_entity(e)).collect();
                    let e_evidence_norm: HashSet<String> = sent
                        .entities
                        .iter()
                        .map(|ent| normalize_entity(&ent.text))
                        .collect();
                    let overlap = jaccard_similarity(&e_claim_norm, &e_evidence_norm);

                    // exact_entity_presence: Check if entities in claim are literally in sentence
                    let sent_text_lower = sent.text.to_lowercase();
                    let mut missing_entities = Vec::new();
                    for ent in &e_claim {
                        if !sent_text_lower.contains(&ent.to_lowercase()) {
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

                // ── Dedup / Contradiction check (LLM-based) ────────────────
                let claim_text_for_dedup = llm_claim.claim_text.clone();
                if config.enable_dedup && !claim_embedding.is_empty() {
                    use crate::maintenance::{check_claim_dedup, ClaimDedupDecision};

                    let decision = check_claim_dedup(
                        claim_store,
                        &claim_text_for_dedup,
                        &claim_embedding,
                        &config.maintenance_config,
                        unified_llm,
                    ).await;

                    match decision {
                        ClaimDedupDecision::Duplicate {
                            existing_id,
                            similarity: dedup_sim,
                        } => {
                            info!(
                                "Claim dedup: merging into existing claim {} (sim={:.3}): \"{}\"",
                                existing_id, dedup_sim, claim_text_for_dedup
                            );
                            let _ = claim_store.add_support(existing_id);
                            rejected_claims.push(RejectedClaim {
                                claim_text: claim_text_for_dedup,
                                rejection_reason: RejectionReason::DuplicateMerged,
                            });
                            continue;
                        },
                        ClaimDedupDecision::Contradiction {
                            existing_id,
                            similarity: contra_sim,
                        } => {
                            info!(
                                "Claim contradiction: superseding claim {} (sim={:.3}): old=\"...\" new=\"{}\"",
                                existing_id, contra_sim, claim_text_for_dedup
                            );
                            // Mark old claim as Superseded with temporal validity end
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            let _ = claim_store.supersede(existing_id, now);

                            // Fall through to store the new (superseding) claim below,
                            // with metadata linking back to the old one.
                            // We'll add the metadata after creating the claim object.
                        },
                        ClaimDedupDecision::NewClaim => {
                            // Normal path — nothing to do
                        },
                    }
                }

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

                // ── Set claim type, subject entity, and category from LLM output ────
                claim.claim_type =
                    crate::claims::types::ClaimType::from_str_loose(&llm_claim.claim_type);
                claim.subject_entity = llm_claim
                    .subject_entity
                    .as_deref()
                    .map(normalize_entity);
                claim.predicate = llm_claim
                    .predicate
                    .as_deref()
                    .map(|p| p.to_lowercase().trim().to_string());
                claim.object_entity = llm_claim
                    .object_entity
                    .as_deref()
                    .map(normalize_entity);
                claim.category = llm_claim.category.clone();
                if let Some(ref cat) = claim.category {
                    claim.metadata.insert("_category".to_string(), cat.clone());
                }
                claim.temporal_type =
                    crate::claims::types::TemporalType::from_str_loose(&llm_claim.temporal_type);

                // ── Attach NER-labeled entities found in the claim text (with roles) ────
                {
                    use crate::claims::types::EntityRole;

                    let claim_lower = claim.claim_text.to_lowercase();
                    let mut seen = HashSet::new();

                    // Pre-compute normalized subject/object for role matching
                    let norm_subject = claim.subject_entity.as_deref().unwrap_or("");
                    let norm_object = claim.object_entity.as_deref().unwrap_or("");

                    for labeled in &labeled_entities {
                        // Match entity text case-insensitively in the claim
                        if claim_lower.contains(&labeled.text.to_lowercase()) {
                            let norm = normalize_entity(&labeled.text);
                            if seen.insert(norm.clone()) {
                                // Determine role: exact match then fuzzy match
                                let role = if !norm_subject.is_empty()
                                    && (norm == norm_subject
                                        || entities_match_fuzzy(&norm, norm_subject))
                                {
                                    EntityRole::Subject
                                } else if !norm_object.is_empty()
                                    && (norm == norm_object
                                        || entities_match_fuzzy(&norm, norm_object))
                                {
                                    EntityRole::Object
                                } else {
                                    EntityRole::Mentioned
                                };

                                claim.entities.push(ClaimEntity {
                                    text: labeled.text.clone(),
                                    label: labeled.label.clone(),
                                    normalized: norm,
                                    confidence: labeled.confidence,
                                    role,
                                });
                            }
                        }
                    }
                }

                // Add diagnostics to metadata
                claim
                    .metadata
                    .insert("similarity".to_string(), similarity.to_string());
                claim
                    .metadata
                    .insert("overlap".to_string(), overlap.to_string());
                claim.metadata.insert(
                    "found_entities".to_string(),
                    claim
                        .entities
                        .iter()
                        .map(|e| format!("{}:{}", e.text, e.label))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
                claim
                    .metadata
                    .insert("missing_entities".to_string(), missing_entities.join(", "));
                claim
                    .metadata
                    .insert("sentence_index".to_string(), best_sent_idx.to_string());
                claim
                    .metadata
                    .insert("claim_type".to_string(), claim.claim_type.to_string());
                if let Some(ref ent) = claim.subject_entity {
                    claim
                        .metadata
                        .insert("subject_entity".to_string(), ent.clone());
                }
                if let Some(ref pred) = claim.predicate {
                    claim
                        .metadata
                        .insert("predicate".to_string(), pred.clone());
                }
                if let Some(ref obj) = claim.object_entity {
                    claim
                        .metadata
                        .insert("object_entity".to_string(), obj.clone());
                }

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

    /// Submit a claim extraction request (blocks if queue is full)
    pub async fn extract(&self, request: ClaimExtractionRequest) -> Result<ClaimExtractionResult> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender
            .send(ClaimExtractionJob {
                request,
                result_tx: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send claim extraction job: {}", e))?;

        rx.await
            .map_err(|e| anyhow::anyhow!("Failed to receive claim extraction result: {}", e))?
    }

    /// Submit extraction without waiting for result (drops if queue full)
    pub fn extract_async(
        &self,
        request: ClaimExtractionRequest,
    ) -> tokio::sync::oneshot::Receiver<Result<ClaimExtractionResult>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        if let Err(e) = self.sender.try_send(ClaimExtractionJob {
            request,
            result_tx: tx,
        }) {
            warn!("Claim extraction queue full or closed, dropping job: {}", e);
        }

        rx
    }
}

// ── Fuzzy entity matching ─────────────────────────────────────────────────────

/// Jaccard similarity between two string sets: |A∩B| / |A∪B|.
/// Returns 0.0 when both sets are empty.
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.iter().filter(|x| b.contains(*x)).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// Fuzzy entity matching: returns true if normalized forms have
/// Jaro-Winkler similarity >= 0.85.
fn entities_match_fuzzy(a: &str, b: &str) -> bool {
    let na = normalize_entity(a);
    let nb = normalize_entity(b);
    if na == nb {
        return true;
    }
    strsim::jaro_winkler(&na, &nb) >= 0.85
}

// ── Entity resolution ────────────────────────────────────────────────────────

/// Normalize an entity string for deduplication.
///
/// * Lowercases
/// * Trims whitespace
/// * Collapses internal whitespace
/// * Strips common determiners ("the", "a", "an")
/// * Strips trailing punctuation
///
/// "The  Adidas  Shoes." → "adidas shoes"
/// "OPENAI API" → "openai api"
pub fn normalize_entity(raw: &str) -> String {
    let lower = raw.to_lowercase();
    // Collapse whitespace
    let words: Vec<&str> = lower.split_whitespace().collect();
    // Strip determiners from the front
    let start = words
        .iter()
        .position(|w| !matches!(*w, "the" | "a" | "an"))
        .unwrap_or(0);
    let trimmed: Vec<&str> = words[start..].to_vec();
    let mut result = trimmed.join(" ");
    // Strip trailing punctuation
    while result.ends_with(|c: char| c.is_ascii_punctuation()) {
        result.pop();
    }
    result
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
    /// Enable dedup / contradiction detection before storing claims
    pub enable_dedup: bool,
    /// Maintenance config for dedup thresholds (shared with maintenance loop)
    pub maintenance_config: crate::maintenance::MaintenanceConfig,
    /// Custom natural-language instructions for extraction.
    pub custom_instructions: Option<String>,
    /// Types of facts to specifically look for.
    pub extraction_includes: Vec<String>,
    /// Types of facts to specifically ignore.
    pub extraction_excludes: Vec<String>,
    /// Few-shot examples to include in extraction prompts.
    pub few_shot_examples: Vec<crate::claims::llm_client::FewShotExample>,
}

impl Default for ClaimExtractionConfig {
    fn default() -> Self {
        Self {
            max_claims_per_input: 10,
            min_confidence: 0.7,
            min_evidence_length: 10,
            enable_dedup: true,
            maintenance_config: crate::maintenance::MaintenanceConfig::default(),
            custom_instructions: None,
            extraction_includes: vec![],
            extraction_excludes: vec![],
            few_shot_examples: crate::claims::llm_client::default_few_shot_examples(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::llm_client::MockClient;
    use tempfile::tempdir;

    #[tokio::test]
    #[ignore = "requires LLM_API_KEY — run with: cargo test --ignored"]
    async fn test_queue_creation() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("claims.redb");

        let embedding_client = Arc::new(
            crate::claims::embeddings::openai_client_from_env().expect("LLM_API_KEY required"),
        );
        let client = Arc::new(MockClient::new());
        let store = Arc::new(ClaimStore::new(&store_path).unwrap());
        let config = ClaimExtractionConfig::default();

        let unified_llm: Arc<dyn crate::llm_client::LlmClient> =
            Arc::new(crate::llm_client::OpenAiLlmClient::new(
                std::env::var("LLM_API_KEY").expect("LLM_API_KEY required"),
                "gpt-4o-mini".to_string(),
            ));
        let _queue =
            ClaimExtractionQueue::new(client, embedding_client, store, 1, config, unified_llm);
    }

    #[test]
    fn test_normalize_entity_basic() {
        assert_eq!(normalize_entity("Adidas"), "adidas");
        assert_eq!(normalize_entity("  OPENAI  API  "), "openai api");
        assert_eq!(normalize_entity("The Adidas Shoes."), "adidas shoes");
        assert_eq!(normalize_entity("a Big Company!"), "big company");
        assert_eq!(normalize_entity("An Example"), "example");
    }

    #[test]
    fn test_normalize_entity_preserves_meaningful_content() {
        assert_eq!(normalize_entity("dark mode"), "dark mode");
        assert_eq!(normalize_entity("John"), "john");
        assert_eq!(normalize_entity("REST API v3.2"), "rest api v3.2");
    }

    #[test]
    fn test_normalize_entity_empty() {
        assert_eq!(normalize_entity(""), "");
        assert_eq!(normalize_entity("   "), "");
    }

    #[test]
    fn test_entities_match_fuzzy_close_names() {
        // "John" vs "Jon" — normalized: "john" vs "jon"
        assert!(entities_match_fuzzy("John", "Jon"));
        // Identical after normalization
        assert!(entities_match_fuzzy("OpenAI", "openai"));
        // Very different strings should not match
        assert!(!entities_match_fuzzy("Apple", "Microsoft"));
    }

    #[test]
    fn test_jaccard_similarity_basic() {
        let a: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["b", "c"].iter().map(|s| s.to_string()).collect();
        let sim = jaccard_similarity(&a, &b);
        // intersection = {b} = 1, union = {a,b,c} = 3 → 1/3
        assert!((sim - 1.0 / 3.0).abs() < 0.001);

        // Identical sets → 1.0
        let sim2 = jaccard_similarity(&a, &a);
        assert!((sim2 - 1.0).abs() < 0.001);

        // Disjoint sets → 0.0
        let c: HashSet<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &c)).abs() < 0.001);

        // Both empty → 0.0
        let empty: HashSet<String> = HashSet::new();
        assert!((jaccard_similarity(&empty, &empty)).abs() < 0.001);
    }
}
