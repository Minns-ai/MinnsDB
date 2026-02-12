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

/// An entity attached to a claim, carrying the NER label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimEntity {
    /// Entity text as it appears in the source (e.g. "John", "Google")
    pub text: String,
    /// NER label (e.g. "PERSON", "ORG", "LOC", "DATE", "PRODUCT", "EVENT")
    pub label: String,
    /// Normalized form for dedup (lowercased, trimmed, determiners stripped)
    pub normalized: String,
}

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
    /// Replaced by a newer claim that contradicts this one (temporal recency wins)
    Superseded,
}

/// Type of knowledge a claim represents.
///
/// Each type has a different default half-life (temporal decay rate) and
/// different contradiction-resolution rules:
///
/// | Type        | Half-life | Notes |
/// |-------------|-----------|-------|
/// | Preference  | 30 days   | Personal taste, can flip |
/// | Fact        | 365 days  | Objective, stable until corrected |
/// | Belief      | 14 days   | Uncertain, needs validation |
/// | Intention   | 3 days    | Time-bound, action-oriented |
/// | Capability  | 180 days  | System/agent feature, binary |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClaimType {
    /// Personal taste or preference — "I like X", "I prefer Y"
    Preference,
    /// Objective statement — "X costs $10", "The API uses REST"
    #[default]
    Fact,
    /// Uncertain opinion — "I think X will work", "Maybe we should Y"
    Belief,
    /// Desired future action — "I want to do X", "I plan to Y"
    Intention,
    /// System/agent ability — "The system supports X", "Agent can Y"
    Capability,
}

impl ClaimType {
    /// Default half-life for this claim type in seconds.
    ///
    /// After `half_life` seconds the temporal weight drops to 0.5.
    pub fn half_life_secs(&self) -> f64 {
        match self {
            ClaimType::Intention => 3.0 * 86_400.0,    // 3 days
            ClaimType::Belief => 14.0 * 86_400.0,      // 14 days
            ClaimType::Preference => 30.0 * 86_400.0,  // 30 days
            ClaimType::Capability => 180.0 * 86_400.0, // 180 days
            ClaimType::Fact => 365.0 * 86_400.0,       // 365 days
        }
    }

    /// Attempt to classify from the LLM-provided string.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "preference" => ClaimType::Preference,
            "fact" => ClaimType::Fact,
            "belief" => ClaimType::Belief,
            "intention" | "intent" => ClaimType::Intention,
            "capability" => ClaimType::Capability,
            _ => ClaimType::Fact, // default
        }
    }
}

impl std::fmt::Display for ClaimType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimType::Preference => write!(f, "Preference"),
            ClaimType::Fact => write!(f, "Fact"),
            ClaimType::Belief => write!(f, "Belief"),
            ClaimType::Intention => write!(f, "Intention"),
            ClaimType::Capability => write!(f, "Capability"),
        }
    }
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

    // ── New fields (P0 upgrade) ──────────────────────────────────────────
    /// Semantic type of this claim (Preference, Fact, Belief, …)
    #[serde(default)]
    pub claim_type: ClaimType,

    /// Normalized primary entity this claim is about (lowercased, trimmed).
    /// e.g. "adidas", "openai api", "dark mode".
    #[serde(default)]
    pub subject_entity: Option<String>,

    /// Optional explicit expiry timestamp (epoch seconds).
    /// Claims past this time get status Dormant during maintenance.
    #[serde(default)]
    pub expires_at: Option<u64>,

    /// If this claim was superseded, the ID of the replacing claim.
    #[serde(default)]
    pub superseded_by: Option<ClaimId>,

    /// NER entities found in this claim, with labels.
    /// e.g. [("John", "PERSON"), ("Google", "ORG")]
    #[serde(default)]
    pub entities: Vec<ClaimEntity>,
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
            claim_type: ClaimType::Fact,
            subject_entity: None,
            expires_at: None,
            superseded_by: None,
            entities: Vec::new(),
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

    /// Temporal weight ∈ (0, 1] based on claim age and type-specific half-life.
    ///
    /// Uses exponential decay: `w(t) = 2^(−age / half_life)`.
    ///
    /// If `expires_at` is set and in the past, returns 0.0 immediately.
    pub fn temporal_weight(&self) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Expired?
        if let Some(exp) = self.expires_at {
            if now > exp {
                return 0.0;
            }
        }

        let age_secs = now.saturating_sub(self.created_at) as f64;
        let half_life = self.claim_type.half_life_secs();

        // 2^(-age/half_life)
        let weight = (2.0_f64).powf(-age_secs / half_life);
        weight as f32
    }

    /// Compute a retrieval score that blends similarity with temporal freshness.
    ///
    /// `score = similarity * (alpha + (1 - alpha) * temporal_weight)`
    ///
    /// With alpha = 0.6 a perfect-similarity claim that is ancient still gets 0.6,
    /// while a fresh one gets 1.0.
    pub fn retrieval_score(&self, similarity: f32) -> f32 {
        const ALPHA: f32 = 0.6;
        let tw = self.temporal_weight();
        similarity * (ALPHA + (1.0 - ALPHA) * tw)
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

    #[test]
    fn test_claim_type_from_str_loose() {
        assert_eq!(
            ClaimType::from_str_loose("preference"),
            ClaimType::Preference
        );
        assert_eq!(ClaimType::from_str_loose("Fact"), ClaimType::Fact);
        assert_eq!(ClaimType::from_str_loose("BELIEF"), ClaimType::Belief);
        assert_eq!(ClaimType::from_str_loose("intention"), ClaimType::Intention);
        assert_eq!(ClaimType::from_str_loose("intent"), ClaimType::Intention);
        assert_eq!(
            ClaimType::from_str_loose("capability"),
            ClaimType::Capability
        );
        assert_eq!(ClaimType::from_str_loose("unknown"), ClaimType::Fact); // default
    }

    #[test]
    fn test_claim_type_half_lives_ordered() {
        // Intention < Belief < Preference < Capability < Fact
        assert!(ClaimType::Intention.half_life_secs() < ClaimType::Belief.half_life_secs());
        assert!(ClaimType::Belief.half_life_secs() < ClaimType::Preference.half_life_secs());
        assert!(ClaimType::Preference.half_life_secs() < ClaimType::Capability.half_life_secs());
        assert!(ClaimType::Capability.half_life_secs() < ClaimType::Fact.half_life_secs());
    }

    #[test]
    fn test_temporal_weight_fresh_claim() {
        let claim = DerivedClaim::new(
            1,
            "Fresh claim".to_string(),
            vec![EvidenceSpan::new(0, 5, "Fresh")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        // Just created — weight should be very close to 1.0
        assert!(claim.temporal_weight() > 0.99);
    }

    #[test]
    fn test_temporal_weight_expired() {
        let mut claim = DerivedClaim::new(
            1,
            "Expired claim".to_string(),
            vec![EvidenceSpan::new(0, 7, "Expired")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        // Set expires_at to 1 second ago
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        claim.expires_at = Some(now - 1);
        assert_eq!(claim.temporal_weight(), 0.0);
    }

    #[test]
    fn test_retrieval_score_fresh() {
        let claim = DerivedClaim::new(
            1,
            "Test".to_string(),
            vec![EvidenceSpan::new(0, 4, "Test")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        // Fresh claim with perfect similarity → score ≈ 1.0
        let score = claim.retrieval_score(1.0);
        assert!(score > 0.99);
    }

    #[test]
    fn test_new_claim_has_default_type() {
        let claim = DerivedClaim::new(
            1,
            "Default".to_string(),
            vec![EvidenceSpan::new(0, 7, "Default")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        assert_eq!(claim.claim_type, ClaimType::Fact);
        assert!(claim.subject_entity.is_none());
        assert!(claim.expires_at.is_none());
        assert!(claim.superseded_by.is_none());
    }
}
