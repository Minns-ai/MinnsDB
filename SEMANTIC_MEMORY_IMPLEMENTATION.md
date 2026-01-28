# Semantic Memory Implementation Plan

## Overview
Integration of claim-based semantic memory into EventGraphDB with control-plane managed cost controls.

## Key Design Principles
1. **Control plane manages costs**: API accepts `enable_semantic` flag per request
2. **Async processing**: Claims extracted off critical path
3. **Dual storage**: redb (canonical) + graph (explainability)
4. **CPU-only**: No GPU dependencies for deployment simplicity
5. **Progressive enhancement**: Works without ANN, adds it when volume demands

---

## Phase 1: Foundation - Types & Event Extensions

### 1.1 Extend Event struct
**File**: `crates/agent-db-events/src/core.rs`

```rust
pub struct Event {
    // ... existing fields ...

    /// Size of context in bytes (for promotion threshold)
    #[serde(default)]
    pub context_size_bytes: usize,

    /// Pointer to segment storage for large contexts
    /// Format: "segment://{bucket}/{key}" or null for inline
    #[serde(default)]
    pub segment_pointer: Option<String>,
}
```

### 1.2 Add Context EventType
**File**: `crates/agent-db-events/src/core.rs`

```rust
pub enum EventType {
    // ... existing variants ...

    /// Explicit context event for semantic distillation
    Context {
        /// Raw text content (or pointer via segment_pointer)
        text: String,
        /// Type: "conversation", "document", "transcript", etc.
        context_type: String,
        /// Optional: language hint for NER
        language: Option<String>,
    },
}
```

### 1.3 Create NER types module
**File**: `crates/agent-db-events/src/ner.rs`

```rust
use serde::{Deserialize, Serialize};
use agent_db_core::types::EventId;

/// Extracted NER features for an event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFeatures {
    /// Source event ID
    pub event_id: EventId,

    /// Entity spans with labels and offsets
    pub entity_spans: Vec<EntitySpan>,

    /// Fingerprint for idempotency (hash of source text + model version)
    pub feature_fingerprint: u64,

    /// Extraction timestamp
    pub extracted_at: u64,

    /// Model/version used
    pub ner_model: String,
}

/// Single entity span with UTF-8 byte offsets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySpan {
    /// Entity label: PERSON, ORG, LOC, DATE, etc.
    pub label: String,

    /// Start offset (UTF-8 bytes)
    pub start_offset: usize,

    /// End offset (UTF-8 bytes, exclusive)
    pub end_offset: usize,

    /// Confidence score [0.0, 1.0]
    pub confidence: f32,

    /// Extracted text (for convenience)
    pub text: String,
}

impl ExtractedFeatures {
    /// Compute fingerprint for deduplication
    pub fn compute_fingerprint(text: &str, model: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        model.hash(&mut hasher);
        hasher.finish()
    }
}
```

**Update**: `crates/agent-db-events/src/lib.rs`
```rust
pub mod ner;
pub use ner::{EntitySpan, ExtractedFeatures};
```

### 1.4 Create claim types module
**File**: `crates/agent-db-graph/src/claims/types.rs`

```rust
use serde::{Deserialize, Serialize};
use agent_db_core::types::{AgentId, EventId, SessionId};
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
    pub fn new(start: usize, end: usize, text: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

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
        if self.end_offset > source_text.len() {
            return false;
        }

        let extracted = &source_text[self.start_offset..self.end_offset];

        // Check snippet matches
        if extracted != self.text_snippet {
            return false;
        }

        // Verify hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    NoEvidence,
    InvalidSpans,
    GeometricFailure,
    DuplicateMerged,
    BelowConfidenceThreshold,
}
```

**Create**: `crates/agent-db-graph/src/claims/mod.rs`
```rust
pub mod types;

pub use types::{
    ClaimId, ClaimStatus, DerivedClaim, EvidenceSpan, ClaimExtractionRequest,
    ClaimExtractionResult, RejectedClaim, RejectionReason, ThreadId,
};
```

**Update**: `crates/agent-db-graph/src/lib.rs`
```rust
pub mod claims;
pub use claims::{ClaimId, ClaimStatus, DerivedClaim, EvidenceSpan};
```

---

## Phase 2: NER Pipeline

### 2.1 Create agent-db-ner crate
**Command**:
```bash
cargo new --lib crates/agent-db-ner
```

**File**: `crates/agent-db-ner/Cargo.toml`
```toml
[package]
name = "agent-db-ner"
version = "0.1.0"
edition = "2021"

[dependencies]
# Local deps
agent-db-core = { path = "../agent-db-core" }
agent-db-events = { path = "../agent-db-events" }

# NER (external service)
reqwest = { version = "0.11", features = ["json"] }

# Async
tokio = { version = "1.28", features = ["full"] }
async-trait = "0.1"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
```

### 2.2 Implement NER extractor
**File**: `crates/agent-db-ner/src/extractor.rs`

```rust
use agent_db_events::EntitySpan;
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait NerExtractor: Send + Sync {
    /// Extract entities from text
    async fn extract(&self, text: &str) -> Result<Vec<EntitySpan>>;

    /// Get model name/version
    fn model_name(&self) -> &str;
}

/// External NER service configuration
pub struct NerServiceConfig {
    pub base_url: String,
    pub request_timeout_ms: u64,
    pub model: Option<String>,
}

/// External NER service implementation
pub struct NerServiceExtractor {
    client: Client,
    config: NerServiceConfig,
    model_name: String,
}

#[async_trait]
impl NerExtractor for NerServiceExtractor {
    async fn extract(&self, text: &str) -> Result<Vec<EntitySpan>> {
        let request = serde_json::json!({
            "text": text,
            "model": self.config.model,
        });

        let response = self.client
            .post(&self.config.base_url)
            .json(&request)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        // Map response entities -> EntitySpan (assumes byte offsets)
        let spans = response["entities"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|entity| {
                let label = entity["label"].as_str()?.to_string();
                let start = entity["start_offset"].as_u64()? as usize;
                let end = entity["end_offset"].as_u64()? as usize;
                let confidence = entity["confidence"].as_f64()? as f32;
                let text = entity["text"].as_str()?.to_string();
                Some(EntitySpan::new(label, start, end, confidence, text))
            })
            .collect();

        Ok(spans)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}
```

### 2.3 Create async extraction queue
**File**: `crates/agent-db-ner/src/queue.rs`

```rust
use agent_db_core::types::EventId;
use agent_db_events::ExtractedFeatures;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error};
use crate::extractor::NerExtractor;

pub struct NerExtractionQueue {
    sender: mpsc::UnboundedSender<NerJob>,
}

struct NerJob {
    event_id: EventId,
    text: String,
    result_tx: tokio::sync::oneshot::Sender<Result<ExtractedFeatures>>,
}

impl NerExtractionQueue {
    pub fn new(extractor: Arc<dyn NerExtractor>, workers: usize) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn worker pool
        for i in 0..workers {
            let mut rx = rx.clone();
            let extractor = extractor.clone();

            tokio::spawn(async move {
                info!("NER worker {} started", i);
                while let Some(job) = rx.recv().await {
                    let result = Self::process_job(job, &*extractor).await;
                    // Result sent via oneshot channel in process_job
                }
                info!("NER worker {} stopped", i);
            });
        }

        Self { sender: tx }
    }

    async fn process_job(
        job: NerJob,
        extractor: &dyn NerExtractor,
    ) -> Result<()> {
        let result = async {
            let spans = extractor.extract(&job.text).await?;

            let fingerprint = ExtractedFeatures::compute_fingerprint(
                &job.text,
                extractor.model_name(),
            );

            Ok(ExtractedFeatures {
                event_id: job.event_id,
                entity_spans: spans,
                feature_fingerprint: fingerprint,
                extracted_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
                ner_model: extractor.model_name().to_string(),
            })
        }.await;

        let _ = job.result_tx.send(result);
        Ok(())
    }

    pub async fn extract(
        &self,
        event_id: EventId,
        text: String,
    ) -> Result<ExtractedFeatures> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender.send(NerJob {
            event_id,
            text,
            result_tx: tx,
        })?;

        rx.await?
    }
}
```

### 2.4 Add redb storage for NER features
**File**: `crates/agent-db-ner/src/storage.rs`

```rust
use agent_db_core::types::EventId;
use agent_db_events::ExtractedFeatures;
use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

const NER_FEATURES_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("ner_features");

pub struct NerFeatureStore {
    db: Database,
}

impl NerFeatureStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = Database::create(path)?;

        // Initialize table
        let write_txn = db.begin_write()?;
        let _ = write_txn.open_table(NER_FEATURES_TABLE)?;
        write_txn.commit()?;

        Ok(Self { db })
    }

    pub fn store(&self, features: &ExtractedFeatures) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(NER_FEATURES_TABLE)?;
            let serialized = bincode::serialize(features)?;
            table.insert(features.event_id, serialized.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get(&self, event_id: EventId) -> Result<Option<ExtractedFeatures>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NER_FEATURES_TABLE)?;

        if let Some(value) = table.get(event_id)? {
            let features: ExtractedFeatures = bincode::deserialize(value.value())?;
            Ok(Some(features))
        } else {
            Ok(None)
        }
    }
}
```

### 2.5 Module entry point
**File**: `crates/agent-db-ner/src/lib.rs`

```rust
pub mod extractor;
pub mod queue;
pub mod storage;

pub use extractor::{NerExtractor, NerServiceConfig, NerServiceExtractor};
pub use queue::NerExtractionQueue;
pub use storage::NerFeatureStore;
```

**Update root Cargo.toml**:
```toml
[workspace]
members = [
    # ... existing
    "crates/agent-db-ner",
]
```

---

## Phase 3: Claim Extraction (LLM Integration)

### 3.1 Create LLM client abstraction
**File**: `crates/agent-db-graph/src/claims/llm_client.rs`

```rust
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Extract claims from text + NER features
    async fn extract_claims(&self, request: LlmExtractionRequest) -> Result<LlmExtractionResponse>;
}

#[derive(Debug, Clone)]
pub struct LlmExtractionRequest {
    pub text: String,
    pub entities: Vec<String>, // Simplified entity mentions
    pub max_claims: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionResponse {
    pub claims: Vec<LlmClaim>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmClaim {
    pub claim_text: String,
    pub evidence_spans: Vec<LlmEvidenceSpan>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmEvidenceSpan {
    pub start_offset: usize,
    pub end_offset: usize,
}

/// OpenAI client implementation
pub struct OpenAiClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn extract_claims(&self, request: LlmExtractionRequest) -> Result<LlmExtractionResponse> {
        let system_prompt = format!(
            "You are a claim extractor. Extract atomic factual claims from the text. \
             For each claim, provide supporting evidence spans as byte offsets. \
             Maximum {} claims. Format: JSON with claims array.",
            request.max_claims
        );

        let user_prompt = format!(
            "Text:\n{}\n\nEntities found: {}\n\nExtract claims with evidence:",
            request.text,
            request.entities.join(", ")
        );

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }))
            .send()
            .await?;

        let json: serde_json::Value = response.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

        let extraction: LlmExtractionResponse = serde_json::from_str(content)?;
        Ok(extraction)
    }
}
```

### 3.2 Create claim store (redb)
**File**: `crates/agent-db-graph/src/claims/store.rs`

```rust
use super::types::*;
use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

const CLAIMS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");
const CLAIM_COUNTER: TableDefinition<&str, u64> = TableDefinition::new("claim_counter");

pub struct ClaimStore {
    db: Database,
}

impl ClaimStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = Database::create(path)?;

        let write_txn = db.begin_write()?;
        let _ = write_txn.open_table(CLAIMS_TABLE)?;
        let _ = write_txn.open_table(CLAIM_COUNTER)?;
        write_txn.commit()?;

        Ok(Self { db })
    }

    pub fn store(&self, claim: &DerivedClaim) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            let serialized = bincode::serialize(claim)?;
            table.insert(claim.id, serialized.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get(&self, claim_id: ClaimId) -> Result<Option<DerivedClaim>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CLAIMS_TABLE)?;

        if let Some(value) = table.get(claim_id)? {
            let claim: DerivedClaim = bincode::deserialize(value.value())?;
            Ok(Some(claim))
        } else {
            Ok(None)
        }
    }

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

    pub fn add_support(&self, claim_id: ClaimId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            if let Some(value) = table.get(claim_id)? {
                let mut claim: DerivedClaim = bincode::deserialize(value.value())?;
                claim.support_count += 1;
                let serialized = bincode::serialize(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn update_status(&self, claim_id: ClaimId, status: ClaimStatus) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CLAIMS_TABLE)?;
            if let Some(value) = table.get(claim_id)? {
                let mut claim: DerivedClaim = bincode::deserialize(value.value())?;
                claim.status = status;
                let serialized = bincode::serialize(&claim)?;
                table.insert(claim_id, serialized.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }
}
```

---

## Phase 4: Embeddings & ANN (Detailed in separate section)

## Phase 5: API Integration

### 5.1 Update ProcessEventRequest
**File**: `server/src/main.rs`

```rust
#[derive(Debug, Serialize, Deserialize)]
struct ProcessEventRequest {
    event: Event,

    /// Enable semantic claim extraction
    #[serde(default)]
    enable_semantic: bool,
}
```

### 5.2 Add claim search endpoint
```rust
#[derive(Debug, Deserialize)]
struct ClaimSearchRequest {
    query_text: String,
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Debug, Serialize)]
struct ClaimResponse {
    id: ClaimId,
    claim_text: String,
    confidence: f32,
    evidence: Vec<EvidenceSpanResponse>,
    source_event_id: EventId,
    created_at: u64,
    support_count: u32,
}

async fn search_claims(
    State(state): State<AppState>,
    Json(request): Json<ClaimSearchRequest>,
) -> Result<Json<Vec<ClaimResponse>>, ApiError> {
    // Implementation in Phase 4
    todo!()
}
```

---

## Configuration

**Add to GraphEngineConfig**:
```rust
pub struct GraphEngineConfig {
    // ... existing ...

    // Semantic Memory
    pub enable_semantic_memory: bool,
    pub ner_workers: usize,
    pub ner_service_url: String,
    pub ner_request_timeout_ms: u64,
    pub ner_model: Option<String>,
    pub max_claims_per_input: usize,
    pub llm_api_key: Option<String>,
    pub llm_model: String, // "gpt-4o-mini"
    pub geometric_threshold: f32,
    pub merge_similarity_threshold: f32,
}
```

---

## Testing Strategy

1. **Unit tests**: Each module (NER, claim extraction, storage)
2. **Integration tests**: Full pipeline from event → claims
3. **API tests**: REST endpoints with semantic flag
4. **Performance tests**: Throughput with/without semantic processing

---

## Deployment Considerations

1. **Feature flag**: `enable_semantic` controls entire pipeline
2. **Async processing**: Claims don't block event ingestion
3. **Resource isolation**: NER workers in separate tokio tasks
4. **Graceful degradation**: System works if LLM/NER unavailable
5. **Monitoring**: Metrics for extraction success/failure rates

---

## Next Steps

Start with Phase 1.1: Extend Event struct with new fields.
