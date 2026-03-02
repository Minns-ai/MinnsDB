//! Error types for AST parsing

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AstError {
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Parse error in {file}: {message}")]
    ParseError { file: String, message: String },

    #[error("Tree-sitter error: {0}")]
    TreeSitter(String),

    #[error("Diff parse error: {0}")]
    DiffParseError(String),
}

pub type AstResult<T> = Result<T, AstError>;
