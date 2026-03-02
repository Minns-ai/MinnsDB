//! Per-language tree-sitter extraction (feature-gated)

#[cfg(feature = "lang-rust")]
pub mod rust;
