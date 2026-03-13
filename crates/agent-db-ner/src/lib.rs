//! Named Entity Recognition (NER) for MinnsDB
//!
//! This crate provides NER capabilities via an external service.
//! Features:
//! - Async extraction queue for non-blocking processing
//! - Persistent storage using redb
//! - UTF-8 byte offset tracking
//! - Idempotency via fingerprinting

pub mod extractor;
pub mod queue;
pub mod storage;

pub use extractor::{NerExtractor, NerServiceConfig, NerServiceExtractor};
pub use queue::NerExtractionQueue;
pub use storage::NerFeatureStore;

// Re-export for convenience
pub use agent_db_events::{EntitySpan, ExtractedFeatures};
