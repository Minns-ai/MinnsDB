//! Core data types for conversation ingestion.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Conversation input types (matching benchmark JSON schema)
// ---------------------------------------------------------------------------

/// Top-level ingestion request containing one or more conversation sessions.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConversationIngest {
    pub case_id: Option<String>,
    pub sessions: Vec<ConversationSession>,
    /// Optional queries for benchmark evaluation.
    #[serde(default)]
    pub queries: Vec<BenchmarkQuery>,
}

/// A single conversation session with ordered messages.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConversationSession {
    pub session_id: String,
    #[serde(default)]
    pub topic: Option<String>,
    pub messages: Vec<ConversationMessage>,
    /// Benchmark metadata (ignored during ingestion).
    #[serde(default)]
    pub contains_fact: Option<bool>,
    #[serde(default)]
    pub fact_id: Option<String>,
    #[serde(default)]
    pub fact_quote: Option<String>,
    /// Benchmark answer checkpoints (per-session cumulative settlements).
    #[serde(default)]
    pub answers: Vec<String>,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

/// A benchmark query with reference answer.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkQuery {
    pub question: String,
    pub reference_answer: Option<ReferenceAnswer>,
}

/// Reference answer for benchmark scoring.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReferenceAnswer {
    pub text: String,
}

// ---------------------------------------------------------------------------
// Message classification
// ---------------------------------------------------------------------------

/// Category of a classified message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageCategory {
    Transaction,
    StateChange,
    Relationship,
    Preference,
    Chitchat,
}

/// Result of classification: category + confidence score.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub category: MessageCategory,
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Split mode for transactions
// ---------------------------------------------------------------------------

/// How a transaction amount is divided among beneficiaries.
#[derive(Debug, Clone, PartialEq)]
pub enum SplitMode {
    /// Split equally among all beneficiaries.
    Equal,
    /// Each beneficiary gets a percentage of the total.
    Percentage(Vec<(String, f64)>),
    /// Each beneficiary gets an explicit amount.
    ExplicitShares(Vec<(String, f64)>),
    /// Entire amount goes to a single beneficiary.
    SoleBeneficiary,
    /// Could not determine split mode.
    Unknown,
}

/// Kind of financial transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionKind {
    Payment,
    Reimbursement,
    Tip,
    Transfer,
    DebtStatement,
}

/// Scope of participants in a transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParticipantsScope {
    /// Only the explicitly named beneficiaries.
    Explicit,
    /// All participants known so far in the case.
    EveryoneKnown,
}

// ---------------------------------------------------------------------------
// Parsed data structs (one per category)
// ---------------------------------------------------------------------------

/// Parsed transaction data.
#[derive(Debug, Clone)]
pub struct TransactionData {
    pub payer: String,
    pub beneficiaries: Vec<String>,
    pub amount: f64,
    pub currency: String,
    pub description: String,
    pub split_mode: SplitMode,
    pub kind: TransactionKind,
    pub participants_scope: ParticipantsScope,
}

/// Parsed state change data.
#[derive(Debug, Clone)]
pub struct StateChangeData {
    pub entity: String,
    pub attribute: String,
    pub new_value: String,
    pub old_value: Option<String>,
}

/// Parsed relationship data.
#[derive(Debug, Clone)]
pub struct RelationshipData {
    pub subject: String,
    pub object: String,
    pub relation_type: String,
}

/// Parsed preference data.
#[derive(Debug, Clone)]
pub struct PreferenceData {
    pub entity: String,
    pub item: String,
    pub category: String,
    pub sentiment: f32,
}

/// Parsed payload — one variant per category.
#[derive(Debug, Clone)]
pub enum ParsedPayload {
    Transaction(TransactionData),
    StateChange(StateChangeData),
    Relationship(RelationshipData),
    Preference(PreferenceData),
    Chitchat(String),
}

/// Envelope wrapping a parsed message with metadata.
#[derive(Debug, Clone)]
pub struct ParsedMessage {
    pub category: MessageCategory,
    pub original_content: String,
    pub session_id: String,
    pub message_index: usize,
    pub confidence: f32,
    pub parsed: ParsedPayload,
}

// ---------------------------------------------------------------------------
// Context structs (threaded through classifier/parsers/bridge)
// ---------------------------------------------------------------------------

/// Ingestion options controlling behavior.
#[derive(Debug, Clone)]
pub struct IngestOptions {
    /// Whether to process assistant messages for facts (default false).
    pub include_assistant_facts: bool,
    /// Whether to generate claims from parsed messages (default false).
    pub enable_claim_generation: bool,
    /// Minimum parse confidence to generate claims (default 0.8).
    pub claim_confidence_threshold: f32,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            include_assistant_facts: false,
            enable_claim_generation: false,
            claim_confidence_threshold: 0.8,
        }
    }
}

/// Immutable context for a single message being processed.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub case_id: String,
    pub session_id: String,
    pub message_index: usize,
    pub speaker_entity: Option<String>,
    pub ingest_options: IngestOptions,
}

/// Name registry mapping person names to stable u64 IDs.
/// IDs are assigned sequentially starting from 1.
#[derive(Debug, Clone, Default)]
pub struct NameRegistry {
    name_to_id: HashMap<String, u64>,
    id_to_name: HashMap<u64, String>,
    next_id: u64,
}

impl NameRegistry {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
            next_id: 1,
        }
    }

    /// Get or create a stable ID for a name.
    pub fn get_or_create(&mut self, name: &str) -> u64 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.name_to_id.insert(name.to_string(), id);
        self.id_to_name.insert(id, name.to_string());
        id
    }

    /// Look up a name by ID.
    pub fn name_for_id(&self, id: u64) -> Option<&str> {
        self.id_to_name.get(&id).map(|s| s.as_str())
    }

    /// Look up an ID by name.
    pub fn id_for_name(&self, name: &str) -> Option<u64> {
        self.name_to_id.get(name).copied()
    }

    /// All known names.
    pub fn all_names(&self) -> Vec<&str> {
        self.name_to_id.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered names.
    pub fn len(&self) -> usize {
        self.name_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.name_to_id.is_empty()
    }
}

/// Mutable shared state accumulated across messages.
#[derive(Debug, Clone)]
pub struct ConversationState {
    /// All participant names seen so far in this case.
    pub known_participants: HashSet<String>,
    /// Default currency (inferred from first monetary mention).
    pub default_currency: String,
    /// Name → ID registry for this case.
    pub name_registry: NameRegistry,
    /// Idempotency tracker: (case_id, session_id, message_index).
    pub processed_messages: HashSet<(String, String, usize)>,
}

impl ConversationState {
    pub fn new() -> Self {
        Self {
            known_participants: HashSet::new(),
            default_currency: "USD".to_string(),
            name_registry: NameRegistry::new(),
            processed_messages: HashSet::new(),
        }
    }
}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Ingestion result
// ---------------------------------------------------------------------------

/// Summary of ingestion results.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IngestResult {
    pub case_id: String,
    pub messages_processed: usize,
    pub transactions_found: usize,
    pub state_changes_found: usize,
    pub relationships_found: usize,
    pub preferences_found: usize,
    pub chitchat_skipped: usize,
    /// Captured (fact_id, fact_quote) pairs from sessions.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub facts_captured: Vec<(String, String)>,
}

/// Per-session ingest result with isolated store.
#[derive(Debug)]
pub struct SessionIngestResult {
    pub session_id: String,
    pub store: crate::structured_memory::StructuredMemoryStore,
    pub name_registry: NameRegistry,
    pub result: IngestResult,
}
