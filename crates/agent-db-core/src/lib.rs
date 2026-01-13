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

pub mod types;
pub mod traits;
pub mod error;
pub mod config;
pub mod utils;

// Re-export commonly used items
pub use types::*;
pub use traits::*;
pub use error::{DatabaseError, DatabaseResult};
pub use config::DatabaseConfig;