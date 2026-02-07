//! Claim types for semantic memory

use agent_db_core::types::EventId;
use agent_db_events::ExtractedFeatures;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use crate::episodes::EpisodeId;

pub type ClaimId = u64;
pub type ThreadId = String;

/// Status of a claim in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimStatus {
    /// Active and available for retrieval
    Active,
    /// Aged out or low relevance, not in active indexes
    Dormant,
    /// Contradicted by other evidence, linked but downranked
    Disputed,
    /// Failed validation, kept for audit only
    Rejected,
}

/// A derived claim extracted from context
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedClaim {
    /// Unique claim ID
    pub id: ClaimId,

    /// Atomic claim text (single fact/point)
    pub claim_text: String,

    /// Supporting evidence spans (required, non-empty)
    pub supporting_evidence: Vec<EvidenceSpan>,

    /// Confidence score [0.0, 1.0]
    pub confidence: f32,

    /// Claim embedding for semantic search
    pub embedding: Vec<f32>,

    /// Source event this was derived from
    #[serde_as(as = "DisplayFromStr")]
    pub source_event_id: EventId,

    /// Episode ID if part of an episode
    pub episode_id: Option<EpisodeId>,

    /// Thread ID for grouping (e.g., conversation thread)
    pub thread_id: Option<ThreadId>,

    /// User/workspace for scoping
    pub user_id: Option<String>,
    pub workspace_id: Option<String>,

    /// Creation timestamp
    pub created_at: u64,

    /// Last accessed timestamp
    pub last_accessed: u64,

    /// Access count
    pub access_count: u32,

    /// Current status
    pub status: ClaimStatus,

    /// Support count (how many times this claim was reinforced)
    pub support_count: u32,

    /// Diagnostic metadata (optional)
    pub metadata: HashMap<String, String>,
}

/// Evidence span within source text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSpan {
    /// Start offset (UTF-8 bytes)
    pub start_offset: usize,

    /// End offset (UTF-8 bytes, exclusive)
    pub end_offset: usize,

    /// Text snippet for convenience
    pub text_snippet: String,

    /// Hash of snippet for integrity check
    pub snippet_hash: u64,
}

impl EvidenceSpan {
    /// Create a new evidence span with validation
    pub fn new(start: usize, end: usize, text: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);

        Self {
            start_offset: start,
            end_offset: end,
            text_snippet: text.to_string(),
            snippet_hash: hasher.finish(),
        }
    }

    /// Validate span against source text
    pub fn validate(&self, source_text: &str) -> bool {
        // Check bounds
        if self.end_offset > source_text.len() {
            return false;
        }

        if self.start_offset >= self.end_offset {
            return false;
        }

        // Extract text
        let extracted = &source_text[self.start_offset..self.end_offset];

        // Check snippet matches
        if extracted != self.text_snippet {
            return false;
        }

        // Verify hash
        let mut hasher = DefaultHasher::new();
        extracted.hash(&mut hasher);

        hasher.finish() == self.snippet_hash
    }
}

/// Request structure for claim extraction
#[derive(Debug, Clone)]
pub struct ClaimExtractionRequest {
    pub event_id: EventId,
    pub canonical_text: String,
    pub ner_features: Option<ExtractedFeatures>,
    pub context_embedding: Option<Vec<f32>>,
    pub episode_id: Option<EpisodeId>,
    pub thread_id: Option<ThreadId>,
    pub user_id: Option<String>,
    pub workspace_id: Option<String>,
}

/// Result of claim extraction
#[derive(Debug, Clone)]
pub struct ClaimExtractionResult {
    pub accepted_claims: Vec<DerivedClaim>,
    pub rejected_claims: Vec<RejectedClaim>,
    pub tokens_used: u64,
    pub extraction_time_ms: u64,
}

/// Rejected claim with reason
#[derive(Debug, Clone)]
pub struct RejectedClaim {
    pub claim_text: String,
    pub rejection_reason: RejectionReason,
}

/// Reason a claim was rejected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// No supporting evidence provided
    NoEvidence,
    /// Evidence spans are invalid or don't match source text
    InvalidSpans,
    /// Failed geometric distance sanity checks
    GeometricFailure,
    /// Duplicate claim, merged with existing
    DuplicateMerged,
    /// Below minimum confidence threshold
    BelowConfidenceThreshold,
}

impl DerivedClaim {
    /// Create a new claim with current timestamp
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ClaimId,
        claim_text: String,
        supporting_evidence: Vec<EvidenceSpan>,
        confidence: f32,
        embedding: Vec<f32>,
        source_event_id: EventId,
        episode_id: Option<EpisodeId>,
        thread_id: Option<ThreadId>,
        user_id: Option<String>,
        workspace_id: Option<String>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id,
            claim_text,
            supporting_evidence,
            confidence,
            embedding,
            source_event_id,
            episode_id,
            thread_id,
            user_id,
            workspace_id,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            status: ClaimStatus::Active,
            support_count: 1,
            metadata: HashMap::new(),
        }
    }

    /// Validate all evidence spans against source text
    pub fn validate_evidence(&self, source_text: &str) -> bool {
        if self.supporting_evidence.is_empty() {
            return false;
        }

        for span in &self.supporting_evidence {
            if !span.validate(source_text) {
                return false;
            }
        }

        true
    }

    /// Update last accessed timestamp and increment access count
    pub fn mark_accessed(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.last_accessed = now;
        self.access_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_span_validation() {
        let text = "John works at Google in California.";

        let span = EvidenceSpan::new(0, 4, "John");
        assert!(span.validate(text));

        let span2 = EvidenceSpan::new(14, 20, "Google");
        assert!(span2.validate(text));
    }

    #[test]
    fn test_evidence_span_invalid_text() {
        let text = "John works at Google";

        let span = EvidenceSpan::new(0, 4, "Jane"); // Wrong text
        assert!(!span.validate(text));
    }

    #[test]
    fn test_evidence_span_out_of_bounds() {
        let text = "Short";

        let span = EvidenceSpan::new(0, 100, "Invalid");
        assert!(!span.validate(text));
    }

    #[test]
    fn test_claim_requires_evidence() {
        let source_text = "Test text";
        let claim = DerivedClaim::new(
            1,
            "Test claim".to_string(),
            vec![], // No evidence
            0.9,
            vec![0.1, 0.2, 0.3],
            123,
            None,
            None,
            None,
            None,
        );

        assert!(!claim.validate_evidence(source_text));
    }

    #[test]
    fn test_claim_with_valid_evidence() {
        let source_text = "John works at Google";
        let evidence = vec![EvidenceSpan::new(0, 4, "John")];

        let claim = DerivedClaim::new(
            1,
            "Person named John exists".to_string(),
            evidence,
            0.9,
            vec![0.1, 0.2, 0.3],
            123,
            None,
            None,
            None,
            None,
        );

        assert!(claim.validate_evidence(source_text));
    }
}
