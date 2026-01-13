//! Event system for the Agentic Database
//! 
//! This crate provides comprehensive event handling including:
//! - Event structures and types
//! - Event validation and causality checking
//! - Event buffering for high-throughput ingestion
//! - Context processing and fingerprinting
//! - Serialization support

pub mod core;
pub mod validation;
pub mod buffer;
pub mod context;
pub mod causality;

// Re-export commonly used items
pub use core::*;
pub use validation::{BasicEventValidator, ContextualEventValidator};
pub use buffer::EventBuffer;
pub use context::ContextMatcher;