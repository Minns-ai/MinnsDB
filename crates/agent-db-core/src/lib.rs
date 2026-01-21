//! Core types, traits, and utilities for the Agentic Database
//!
//! This crate provides the foundational abstractions used throughout
//! the Agentic Database system, including:
//!
//! - Core data types (EventId, AgentId, Timestamp, etc.)
//! - Database trait definitions
//! - Comprehensive error handling
//! - Configuration structures
//! - Common utilities

pub mod config;
pub mod error;
pub mod traits;
pub mod types;
pub mod utils;

// Re-export commonly used items
pub use config::DatabaseConfig;
pub use error::{DatabaseError, DatabaseResult};
pub use traits::*;
pub use types::*;
