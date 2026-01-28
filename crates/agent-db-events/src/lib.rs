//! Event system for the Agentic Database
//!
//! This crate provides comprehensive event handling including:
//! - Event structures and types
//! - Event validation and causality checking
//! - Event buffering for high-throughput ingestion
//! - Context processing and fingerprinting
//! - Named Entity Recognition (NER) types
//! - Serialization support

pub mod buffer;
pub mod causality;
pub mod context;
pub mod core;
pub mod ner;
pub mod validation;

// Re-export commonly used items
pub use buffer::EventBuffer;
pub use context::ContextMatcher;
pub use core::*;
pub use ner::{EntitySpan, ExtractedFeatures, SentenceEntities};
pub use validation::{BasicEventValidator, ContextualEventValidator};
