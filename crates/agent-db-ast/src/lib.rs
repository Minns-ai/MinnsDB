//! AST parsing for code intelligence
//!
//! This crate provides tree-sitter based AST parsing for extracting
//! structured code entities and relationships from source code.
//! It is fully graph-agnostic — no dependencies on ConceptType/EdgeType.

pub mod diff;
pub mod entities;
pub mod error;
pub mod languages;
pub mod parser;

#[cfg(test)]
mod tests;

pub use diff::{
    parse_unified_diff, DiffFile, DiffFileStatus, DiffHunk, DiffLine, DiffLineKind, ParsedDiff,
};
pub use entities::{CodeEntity, CodeEntityKind, CodeRelationKind, CodeRelationship, Parameter};
pub use error::{AstError, AstResult};
pub use parser::{AstParser, DiagnosticKind, ParseDiagnostic, ParseResult};
