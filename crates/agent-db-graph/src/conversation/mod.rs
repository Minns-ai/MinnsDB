//! Conversation ingestion layer.
//!
//! Converts raw `{role, content}` messages into structured events and memory.
//! Generic NL→structured-data adapter — not tied to any specific benchmark.
//!
//! # Modules
//!
//! - [`types`] — Core data types for conversation ingestion
//! - [`classifier`] — Rule-based message classifier
//! - [`parsers`] — Category-specific parsers (transaction, state, relationship, preference)
//! - [`bridge`] — Direct structured memory population (bypasses event pipeline)
//! - [`numeric_reasoning`] — Generic numeric operations over structured memory
//! - [`nlq_ext`] — NLQ query extensions for conversation-aware queries

pub mod answer_composer;
pub mod bridge;
pub mod classifier;
pub mod llm_classifier;
pub mod nlq_ext;
pub mod numeric_reasoning;
pub mod parsers;
pub mod types;

// Re-export key types
pub use answer_composer::{gather_memory_context, MemoryContextEntry, MemorySummary, StrategySummary};
pub use bridge::{ingest, ingest_incremental, ingest_per_session, ingest_to_events, ingest_with_llm, ingest_with_llm_incremental};
pub use nlq_ext::{classify_conversation_query, execute_conversation_query, ConversationQueryType};
pub use types::{
    ConversationIngest, ConversationMessage, ConversationSession, ConversationState, IngestOptions,
    IngestResult, NameRegistry, SessionIngestResult,
};
