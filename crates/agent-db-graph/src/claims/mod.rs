//! Semantic memory claim extraction and management

pub mod embedding_queue;
pub mod embeddings;
pub mod extractor;
pub mod hybrid_search;
pub mod llm_client;
pub mod store;
pub mod types;

pub use embedding_queue::EmbeddingQueue;
pub use embeddings::{
    AnthropicEmbeddingClient, DistanceMetric, Embedding, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse, MockEmbeddingClient, OpenAiEmbeddingClient, VectorSimilarity,
};
pub use extractor::{ClaimExtractionConfig, ClaimExtractionQueue};
pub use hybrid_search::{HybridClaimSearch, HybridSearchConfig};
pub use llm_client::LabeledEntity;
pub use llm_client::{
    AnthropicClient, LlmClient, LlmExtractionRequest, LlmExtractionResponse, MockClient,
    OpenAiClient,
};
pub use store::ClaimStore;
pub use types::{
    ClaimEntity, ClaimExtractionRequest, ClaimExtractionResult, ClaimId, ClaimStatus, ClaimType,
    DerivedClaim, EvidenceSpan, RejectedClaim, RejectionReason, ThreadId, META_NEGATIVE_OUTCOMES,
    META_POSITIVE_OUTCOMES, META_Q_VALUE, Q_ALPHA, Q_KICK_IN,
};
