//! Claim types for semantic memory

use agent_db_core::types::EventId;
use agent_db_events::ExtractedFeatures;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use crate::episodes::EpisodeId;

/// Metadata key for positive outcome count on a claim.
pub const META_POSITIVE_OUTCOMES: &str = "_positive_outcomes";
/// Metadata key for negative outcome count on a claim.
pub const META_NEGATIVE_OUTCOMES: &str = "_negative_outcomes";
/// Metadata key for exponential moving average Q-value on a claim.
pub const META_Q_VALUE: &str = "_q_value";

/// Minimum number of outcomes before Q-value scoring kicks in.
/// Below this threshold, Bayesian `(pos+1)/(total+2)` is used (stable, prior-dominated).
/// At and above, the EMA Q-value takes over (responsive to distribution shift).
pub const Q_KICK_IN: u32 = 5;

/// Learning rate for EMA Q-value updates: `Q = Q + α(r − Q)`.
/// 0.3 means each new outcome contributes 30% and prior history 70%.
pub const Q_ALPHA: f32 = 0.3;

pub type ClaimId = u64;
pub type ThreadId = String;

/// Role of an entity within a claim's Subject-Predicate-Object triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntityRole {
    /// The primary entity performing the action or being described.
    Subject,
    /// The target entity of the predicate.
    Object,
    /// Mentioned but not the subject or object of the claim's core relationship.
    #[default]
    Mentioned,
}

impl std::fmt::Display for EntityRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityRole::Subject => write!(f, "subject"),
            EntityRole::Object => write!(f, "object"),
            EntityRole::Mentioned => write!(f, "mentioned"),
        }
    }
}

/// An entity attached to a claim, carrying the NER label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimEntity {
    /// Entity text as it appears in the source (e.g. "John", "Google")
    pub text: String,
    /// NER label (e.g. "PERSON", "ORG", "LOC", "DATE", "PRODUCT", "EVENT")
    pub label: String,
    /// Normalized form for dedup (lowercased, trimmed, determiners stripped)
    pub normalized: String,
    /// NER confidence score [0.0, 1.0], default 1.0 for backward compat
    #[serde(default = "default_entity_confidence")]
    pub confidence: f32,
    /// Role of this entity in the claim's SPO triple
    #[serde(default)]
    pub role: EntityRole,
}

fn default_entity_confidence() -> f32 {
    1.0
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

/// Temporal stability classification for a claim.
///
/// Determines how the claim's decay curve behaves:
///
/// | Type       | Decay Behavior |
/// |------------|----------------|
/// | Static     | No decay — stable facts ("Paris is the capital of France") |
/// | Dynamic    | Normal type-based decay — can be superseded ("Alice lives in NYC") |
/// | Atemporal  | No decay — mathematical/logical truths ("2+2=4") |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TemporalType {
    /// Stable real-world fact that should never decay (e.g., "Paris is the capital of France").
    Static,
    /// Fact that can change over time and should decay normally (e.g., "Alice lives in NYC").
    #[default]
    Dynamic,
    /// Mathematical, logical, or definitional truth — permanent (e.g., "2+2=4").
    Atemporal,
}

impl TemporalType {
    /// Attempt to classify from the LLM-provided string.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "static" | "permanent" | "stable" => TemporalType::Static,
            "dynamic" | "changing" | "temporal" => TemporalType::Dynamic,
            "atemporal" | "timeless" | "mathematical" | "logical" => TemporalType::Atemporal,
            _ => TemporalType::Dynamic, // default
        }
    }

    /// Whether this temporal type should undergo decay at all.
    pub fn should_decay(&self) -> bool {
        matches!(self, TemporalType::Dynamic)
    }
}

impl std::fmt::Display for TemporalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalType::Static => write!(f, "static"),
            TemporalType::Dynamic => write!(f, "dynamic"),
            TemporalType::Atemporal => write!(f, "atemporal"),
        }
    }
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
    /// Negative lesson — "Don't do X", learned from repeated failures
    Avoidance,
    /// Architectural decision or code pattern — "We use the repository pattern"
    CodePattern,
    /// API contract or interface specification — "The endpoint accepts JSON"
    ApiContract,
    /// Bug fix or known issue — "Fixed null pointer in login flow"
    BugFix,
}

impl ClaimType {
    /// Default half-life for this claim type in seconds.
    ///
    /// After `half_life` seconds the temporal weight drops to 0.5.
    pub fn half_life_secs(&self) -> f64 {
        match self {
            ClaimType::Intention => 3.0 * 86_400.0,     // 3 days
            ClaimType::Belief => 14.0 * 86_400.0,       // 14 days
            ClaimType::Preference => 30.0 * 86_400.0,   // 30 days
            ClaimType::BugFix => 90.0 * 86_400.0,       // 90 days
            ClaimType::Capability => 180.0 * 86_400.0,  // 180 days
            ClaimType::ApiContract => 180.0 * 86_400.0, // 180 days
            ClaimType::Fact => 365.0 * 86_400.0,        // 365 days
            ClaimType::CodePattern => 365.0 * 86_400.0, // 365 days
            ClaimType::Avoidance => 14.0 * 86_400.0,    // 14 days (same as Belief)
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
            "avoidance" | "avoid" | "negative" | "constraint" => ClaimType::Avoidance,
            "code_pattern" | "pattern" | "architecture" => ClaimType::CodePattern,
            "api" | "api_contract" | "contract" | "interface_spec" => ClaimType::ApiContract,
            "bug" | "bugfix" | "bug_fix" | "fix" => ClaimType::BugFix,
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
            ClaimType::Avoidance => write!(f, "Avoidance"),
            ClaimType::CodePattern => write!(f, "CodePattern"),
            ClaimType::ApiContract => write!(f, "ApiContract"),
            ClaimType::BugFix => write!(f, "BugFix"),
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

    /// Verb/relationship linking subject to object (e.g. "works at", "prefers").
    #[serde(default)]
    pub predicate: Option<String>,

    /// Target entity of the predicate (e.g. "google", "dark mode").
    #[serde(default)]
    pub object_entity: Option<String>,

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

    /// Auto-tagged category (e.g., "personal", "preferences", "work").
    #[serde(default)]
    pub category: Option<String>,

    /// Temporal stability classification (Static/Dynamic/Atemporal).
    /// Determines whether this claim undergoes temporal decay.
    #[serde(default)]
    pub temporal_type: TemporalType,
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

/// The role/source of the content being extracted from.
///
/// Different roles get different extraction prompts to prevent cross-role
/// pollution (e.g., extracting agent instructions as user preferences).
/// Inspired by prior work's USER_MEMORY_EXTRACTION_PROMPT vs AGENT_MEMORY_EXTRACTION_PROMPT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceRole {
    /// Content from a human user — extract personal facts, preferences, intentions.
    #[default]
    User,
    /// Content from an AI assistant — extract capabilities, tool usage patterns, system facts.
    Assistant,
    /// Content from system/context — extract environmental facts, constraints, configurations.
    System,
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
    /// Role of the content source for role-aware extraction prompts.
    #[allow(dead_code)]
    pub source_role: SourceRole,
    /// Rolling conversation summary for cross-message context (optional).
    pub rolling_summary: Option<String>,
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
            predicate: None,
            object_entity: None,
            expires_at: None,
            superseded_by: None,
            entities: Vec::new(),
            category: None,
            temporal_type: TemporalType::Dynamic,
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
    /// If `temporal_type` is Static or Atemporal, returns 1.0 (no decay).
    /// If `expires_at` is set and in the past, returns 0.0 immediately.
    pub fn temporal_weight(&self) -> f32 {
        // Static and Atemporal claims never decay
        if !self.temporal_type.should_decay() {
            // Still respect explicit expiry
            if let Some(exp) = self.expires_at {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now > exp {
                    return 0.0;
                }
            }
            return 1.0;
        }

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

    /// Compute a retrieval score that blends similarity with temporal freshness
    /// and outcome history.
    ///
    /// `base = similarity * (alpha + (1 - alpha) * temporal_weight)`
    /// `multiplier = 0.6 + 0.8 * outcome_score`  (maps [0..1] → [0.6..1.4])
    /// `score = base * multiplier`
    ///
    /// With alpha = 0.6 a perfect-similarity claim that is ancient still gets 0.6,
    /// while a fresh one gets 1.0. Claims with no outcomes get multiplier = 1.0
    /// (neutral). All-positive → 1.4 (40% boost). All-negative → 0.6 (40% penalty).
    pub fn retrieval_score(&self, similarity: f32) -> f32 {
        const ALPHA: f32 = 0.6;
        let tw = self.temporal_weight();
        let base = similarity * (ALPHA + (1.0 - ALPHA) * tw);
        let multiplier = 0.6 + 0.8 * self.outcome_score();
        base * multiplier
    }

    /// Number of positive outcomes recorded for this claim.
    pub fn positive_outcomes(&self) -> u32 {
        self.metadata
            .get(META_POSITIVE_OUTCOMES)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// Number of negative outcomes recorded for this claim.
    pub fn negative_outcomes(&self) -> u32 {
        self.metadata
            .get(META_NEGATIVE_OUTCOMES)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// Total outcomes (positive + negative).
    pub fn total_outcomes(&self) -> u32 {
        self.positive_outcomes() + self.negative_outcomes()
    }

    /// Record an outcome (success or failure) for this claim.
    ///
    /// Updates both the lifetime counters (for auditability) and the EMA
    /// Q-value (for responsive scoring once enough evidence accumulates).
    pub fn record_outcome(&mut self, success: bool) {
        // Update lifetime counters (lossless)
        let key = if success {
            META_POSITIVE_OUTCOMES
        } else {
            META_NEGATIVE_OUTCOMES
        };
        let current: u32 = self
            .metadata
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        self.metadata
            .insert(key.to_string(), (current + 1).to_string());

        // Update EMA Q-value: Q = Q + α(r − Q)
        let r = if success { 1.0_f32 } else { 0.0 };
        let q_old = self.q_value();
        let q_new = q_old + Q_ALPHA * (r - q_old);
        self.metadata
            .insert(META_Q_VALUE.to_string(), format!("{:.6}", q_new));
    }

    /// Current EMA Q-value (exponential moving average of outcomes).
    /// Starts at 0.5 (neutral prior), updated on each `record_outcome`.
    pub fn q_value(&self) -> f32 {
        self.metadata
            .get(META_Q_VALUE)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5)
    }

    /// Piecewise outcome score ∈ [0.0, 1.0].
    ///
    /// **Below `Q_KICK_IN` outcomes:** Bayesian `(positive + 1) / (total + 2)`.
    /// Stable, prior-dominated, resistant to noise from small samples.
    ///
    /// **At or above `Q_KICK_IN` outcomes:** EMA Q-value.
    /// Responsive to distribution shift — recent outcomes weighted exponentially
    /// higher than old ones via learning rate `Q_ALPHA`.
    ///
    /// The Q-value accumulates from outcome #1 but is not *read* until
    /// the threshold, giving the EMA time to warm up.
    pub fn outcome_score(&self) -> f32 {
        if self.total_outcomes() < Q_KICK_IN {
            // Phase 1: Conservative Bayesian
            let pos = self.positive_outcomes() as f32;
            let total = self.total_outcomes() as f32;
            (pos + 1.0) / (total + 2.0)
        } else {
            // Phase 2: Responsive Q-value
            self.q_value()
        }
    }

    /// Build embedding text with a type prefix and optional structured context.
    ///
    /// Returns `(text, context)` — the existing `OpenAiEmbeddingClient` already
    /// prepends context to text when present, so no client changes needed.
    pub fn embedding_text(&self) -> (String, Option<String>) {
        let type_prefix = match self.claim_type {
            ClaimType::Preference => "Preference",
            ClaimType::Fact => "Fact",
            ClaimType::Belief => "Belief",
            ClaimType::Intention => "Intention",
            ClaimType::Capability => "Capability",
            ClaimType::Avoidance => "Avoidance",
            ClaimType::CodePattern => "CodePattern",
            ClaimType::ApiContract => "ApiContract",
            ClaimType::BugFix => "BugFix",
        };
        let text = format!("{}: {}", type_prefix, self.claim_text);

        let mut parts: Vec<String> = Vec::new();
        if let Some(ref s) = self.subject_entity {
            parts.push(format!("Subject: {}", s));
        }
        if let Some(ref p) = self.predicate {
            parts.push(format!("Predicate: {}", p));
        }
        if let Some(ref o) = self.object_entity {
            parts.push(format!("Object: {}", o));
        }
        if let Some(ref c) = self.category {
            parts.push(format!("Category: {}", c));
        }

        let labeled: Vec<String> = self
            .entities
            .iter()
            .map(|e| format!("{} ({})", e.text, e.label))
            .collect();
        if !labeled.is_empty() {
            parts.push(format!("Entities: {}", labeled.join(", ")));
        }

        let context = if parts.is_empty() {
            None
        } else {
            Some(parts.join(". "))
        };
        (text, context)
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

    #[test]
    fn test_derived_claim_category_default_none() {
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
        assert!(claim.category.is_none());
    }

    // ── P0: Outcome tracking tests ───────────────────────────────────────

    #[test]
    fn test_outcome_counters_default_zero() {
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
        assert_eq!(claim.positive_outcomes(), 0);
        assert_eq!(claim.negative_outcomes(), 0);
        assert_eq!(claim.total_outcomes(), 0);
    }

    #[test]
    fn test_record_outcome_increments() {
        let mut claim = DerivedClaim::new(
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
        claim.record_outcome(true);
        claim.record_outcome(true);
        claim.record_outcome(false);
        assert_eq!(claim.positive_outcomes(), 2);
        assert_eq!(claim.negative_outcomes(), 1);
        assert_eq!(claim.total_outcomes(), 3);
    }

    #[test]
    fn test_outcome_score_bayesian_phase() {
        let mut claim = DerivedClaim::new(
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
        // No outcomes → Bayesian (0+1)/(0+2) = 0.5
        assert!((claim.outcome_score() - 0.5).abs() < 0.001);

        // 3 positive → still in Bayesian phase (< Q_KICK_IN=5)
        // (3+1)/(3+2) = 0.8
        claim.record_outcome(true);
        claim.record_outcome(true);
        claim.record_outcome(true);
        assert!((claim.outcome_score() - 0.8).abs() < 0.001);

        // Reset for negative test
        let mut claim2 = DerivedClaim::new(
            2,
            "Test2".to_string(),
            vec![EvidenceSpan::new(0, 5, "Test2")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        // 3 negative → still Bayesian: (0+1)/(3+2) = 0.2
        claim2.record_outcome(false);
        claim2.record_outcome(false);
        claim2.record_outcome(false);
        assert!((claim2.outcome_score() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_outcome_score_q_phase_kicks_in() {
        let mut claim = DerivedClaim::new(
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
        // Record Q_KICK_IN (5) positive outcomes to cross into Q phase
        for _ in 0..5 {
            claim.record_outcome(true);
        }
        assert_eq!(claim.total_outcomes(), 5);
        // Now in Q phase — Q has been accumulating via EMA
        let q = claim.outcome_score();
        // After 5 consecutive successes, Q should be well above 0.5
        assert!(q > 0.7, "Q after 5 successes should be >0.7, got {}", q);
    }

    #[test]
    fn test_q_adapts_faster_than_bayesian_on_shift() {
        // Simulate distribution shift: 10 successes then 5 failures
        let mut claim = DerivedClaim::new(
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
        for _ in 0..10 {
            claim.record_outcome(true);
        }
        let score_before_shift = claim.outcome_score();

        // Now 5 failures (environment changed)
        for _ in 0..5 {
            claim.record_outcome(false);
        }
        let score_after_shift = claim.outcome_score();

        // Q should drop substantially — faster than Bayesian would
        // Bayesian would give: (10+1)/(15+2) = 0.647
        // Q (EMA) should be much lower since recent outcomes are all failures
        assert!(
            score_after_shift < 0.5,
            "Q should drop below 0.5 after 5 consecutive failures following shift, got {}",
            score_after_shift
        );
        assert!(
            score_after_shift < score_before_shift - 0.3,
            "Q should drop substantially: before={}, after={}",
            score_before_shift,
            score_after_shift
        );
    }

    #[test]
    fn test_q_value_stored_in_metadata() {
        let mut claim = DerivedClaim::new(
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
        // Initial Q = 0.5 (neutral default)
        assert!((claim.q_value() - 0.5).abs() < 0.001);

        // After one success: Q = 0.5 + 0.3*(1.0 - 0.5) = 0.65
        claim.record_outcome(true);
        assert!((claim.q_value() - 0.65).abs() < 0.001);

        // After one failure: Q = 0.65 + 0.3*(0.0 - 0.65) = 0.455
        claim.record_outcome(false);
        assert!((claim.q_value() - 0.455).abs() < 0.001);
    }

    // ── P1: Outcome-aware retrieval scoring tests ────────────────────────

    #[test]
    fn test_retrieval_score_neutral_no_outcomes() {
        // Fresh claim with no outcomes should behave identically to old formula
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
        // outcome_score = 0.5, multiplier = 0.6 + 0.8*0.5 = 1.0
        let score = claim.retrieval_score(1.0);
        // Fresh claim: tw ≈ 1.0, base ≈ 1.0, multiplier = 1.0
        assert!(score > 0.99, "neutral score should be ~1.0, got {}", score);
    }

    #[test]
    fn test_retrieval_score_positive_boosts() {
        let mut claim = DerivedClaim::new(
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
        let neutral_score = claim.retrieval_score(1.0);

        // Add positive outcomes
        for _ in 0..10 {
            claim.record_outcome(true);
        }
        let boosted_score = claim.retrieval_score(1.0);
        assert!(
            boosted_score > neutral_score,
            "positive outcomes should boost score: {} > {}",
            boosted_score,
            neutral_score
        );
    }

    #[test]
    fn test_retrieval_score_negative_penalizes() {
        let mut claim = DerivedClaim::new(
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
        let neutral_score = claim.retrieval_score(1.0);

        // Add negative outcomes
        for _ in 0..10 {
            claim.record_outcome(false);
        }
        let penalized_score = claim.retrieval_score(1.0);
        assert!(
            penalized_score < neutral_score,
            "negative outcomes should penalize score: {} < {}",
            penalized_score,
            neutral_score
        );
    }

    // ── P2: ClaimType::Avoidance tests ───────────────────────────────────

    #[test]
    fn test_avoidance_half_life() {
        assert_eq!(
            ClaimType::Avoidance.half_life_secs(),
            ClaimType::Belief.half_life_secs(),
            "Avoidance should have same half-life as Belief (14 days)"
        );
    }

    #[test]
    fn test_avoidance_from_str_loose() {
        assert_eq!(ClaimType::from_str_loose("avoidance"), ClaimType::Avoidance);
        assert_eq!(ClaimType::from_str_loose("avoid"), ClaimType::Avoidance);
        assert_eq!(ClaimType::from_str_loose("negative"), ClaimType::Avoidance);
        assert_eq!(
            ClaimType::from_str_loose("constraint"),
            ClaimType::Avoidance
        );
    }

    #[test]
    fn test_avoidance_display() {
        assert_eq!(ClaimType::Avoidance.to_string(), "Avoidance");
    }

    // ── TemporalType tests ───────────────────────────────────────────────

    #[test]
    fn test_temporal_type_default_is_dynamic() {
        assert_eq!(TemporalType::default(), TemporalType::Dynamic);
    }

    #[test]
    fn test_temporal_type_from_str_loose() {
        assert_eq!(TemporalType::from_str_loose("static"), TemporalType::Static);
        assert_eq!(
            TemporalType::from_str_loose("permanent"),
            TemporalType::Static
        );
        assert_eq!(TemporalType::from_str_loose("stable"), TemporalType::Static);
        assert_eq!(
            TemporalType::from_str_loose("dynamic"),
            TemporalType::Dynamic
        );
        assert_eq!(
            TemporalType::from_str_loose("changing"),
            TemporalType::Dynamic
        );
        assert_eq!(
            TemporalType::from_str_loose("atemporal"),
            TemporalType::Atemporal
        );
        assert_eq!(
            TemporalType::from_str_loose("timeless"),
            TemporalType::Atemporal
        );
        assert_eq!(
            TemporalType::from_str_loose("mathematical"),
            TemporalType::Atemporal
        );
        assert_eq!(
            TemporalType::from_str_loose("unknown"),
            TemporalType::Dynamic
        );
    }

    #[test]
    fn test_temporal_type_should_decay() {
        assert!(!TemporalType::Static.should_decay());
        assert!(TemporalType::Dynamic.should_decay());
        assert!(!TemporalType::Atemporal.should_decay());
    }

    #[test]
    fn test_temporal_type_display() {
        assert_eq!(TemporalType::Static.to_string(), "static");
        assert_eq!(TemporalType::Dynamic.to_string(), "dynamic");
        assert_eq!(TemporalType::Atemporal.to_string(), "atemporal");
    }

    #[test]
    fn test_static_claim_no_decay() {
        let mut claim = DerivedClaim::new(
            1,
            "Paris is the capital of France".to_string(),
            vec![EvidenceSpan::new(0, 5, "Paris")],
            0.95,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        claim.temporal_type = TemporalType::Static;
        // Even though created_at is "now", a Static claim should always return 1.0
        assert!((claim.temporal_weight() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_atemporal_claim_no_decay() {
        let mut claim = DerivedClaim::new(
            1,
            "2+2=4".to_string(),
            vec![EvidenceSpan::new(0, 5, "2+2=4")],
            0.99,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        claim.temporal_type = TemporalType::Atemporal;
        assert!((claim.temporal_weight() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_static_claim_still_respects_expiry() {
        let mut claim = DerivedClaim::new(
            1,
            "Static but expired".to_string(),
            vec![EvidenceSpan::new(0, 6, "Static")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        claim.temporal_type = TemporalType::Static;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        claim.expires_at = Some(now - 1);
        assert_eq!(claim.temporal_weight(), 0.0);
    }

    #[test]
    fn test_new_claim_has_dynamic_temporal_type() {
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
        assert_eq!(claim.temporal_type, TemporalType::Dynamic);
    }

    // ── EntityRole tests ─────────────────────────────────────────────────

    #[test]
    fn test_entity_role_default_is_mentioned() {
        assert_eq!(EntityRole::default(), EntityRole::Mentioned);
    }

    #[test]
    fn test_entity_role_display() {
        assert_eq!(EntityRole::Subject.to_string(), "subject");
        assert_eq!(EntityRole::Object.to_string(), "object");
        assert_eq!(EntityRole::Mentioned.to_string(), "mentioned");
    }

    #[test]
    fn test_claim_entity_backward_compat_deserialization() {
        // Old format without confidence/role should deserialize cleanly
        let json = r#"{"text":"John","label":"PERSON","normalized":"john"}"#;
        let entity: ClaimEntity = serde_json::from_str(json).unwrap();
        assert_eq!(entity.text, "John");
        assert_eq!(entity.label, "PERSON");
        assert_eq!(entity.normalized, "john");
        assert_eq!(entity.confidence, 1.0); // default
        assert_eq!(entity.role, EntityRole::Mentioned); // default
    }

    #[test]
    fn test_claim_entity_full_deserialization() {
        let json = r#"{"text":"Google","label":"ORG","normalized":"google","confidence":0.92,"role":"Object"}"#;
        let entity: ClaimEntity = serde_json::from_str(json).unwrap();
        assert_eq!(entity.text, "Google");
        assert_eq!(entity.confidence, 0.92);
        assert_eq!(entity.role, EntityRole::Object);
    }

    #[test]
    fn test_derived_claim_spo_defaults() {
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
        assert!(claim.predicate.is_none());
        assert!(claim.object_entity.is_none());
    }

    #[test]
    fn test_derived_claim_spo_backward_compat_deserialization() {
        // Simulate old serialized DerivedClaim without predicate/object_entity
        let mut claim = DerivedClaim::new(
            1,
            "User works at Google".to_string(),
            vec![EvidenceSpan::new(0, 4, "User")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        claim.subject_entity = Some("user".to_string());
        // Serialize and deserialize
        let serialized = serde_json::to_string(&claim).unwrap();
        let deserialized: DerivedClaim = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.subject_entity.as_deref(), Some("user"));
        assert!(deserialized.predicate.is_none());
        assert!(deserialized.object_entity.is_none());
    }

    #[test]
    fn test_embedding_text_fact_with_spo() {
        let mut claim = DerivedClaim::new(
            1,
            "Alice works at Google".to_string(),
            vec![EvidenceSpan::new(0, 5, "Alice")],
            0.9,
            vec![],
            100,
            None,
            None,
            None,
            None,
        );
        claim.claim_type = ClaimType::Fact;
        claim.subject_entity = Some("alice".to_string());
        claim.predicate = Some("works at".to_string());
        claim.object_entity = Some("google".to_string());

        let (text, context) = claim.embedding_text();
        assert_eq!(text, "Fact: Alice works at Google");
        let ctx = context.unwrap();
        assert!(ctx.contains("Subject: alice"));
        assert!(ctx.contains("Predicate: works at"));
        assert!(ctx.contains("Object: google"));
    }

    #[test]
    fn test_embedding_text_preference_with_entities() {
        let mut claim = DerivedClaim::new(
            2,
            "I prefer Rust over Python".to_string(),
            vec![EvidenceSpan::new(0, 1, "I")],
            0.85,
            vec![],
            200,
            None,
            None,
            None,
            None,
        );
        claim.claim_type = ClaimType::Preference;
        claim.entities = vec![
            ClaimEntity {
                text: "Rust".to_string(),
                label: "PRODUCT".to_string(),
                normalized: "rust".to_string(),
                confidence: 1.0,
                role: EntityRole::default(),
            },
            ClaimEntity {
                text: "Python".to_string(),
                label: "PRODUCT".to_string(),
                normalized: "python".to_string(),
                confidence: 1.0,
                role: EntityRole::default(),
            },
        ];

        let (text, context) = claim.embedding_text();
        assert_eq!(text, "Preference: I prefer Rust over Python");
        let ctx = context.unwrap();
        assert!(ctx.contains("Entities: Rust (PRODUCT), Python (PRODUCT)"));
    }

    #[test]
    fn test_embedding_text_no_metadata() {
        let claim = DerivedClaim::new(
            3,
            "Something happened".to_string(),
            vec![EvidenceSpan::new(0, 9, "Something")],
            0.5,
            vec![],
            300,
            None,
            None,
            None,
            None,
        );

        let (text, context) = claim.embedding_text();
        assert_eq!(text, "Fact: Something happened");
        assert!(context.is_none());
    }
}
