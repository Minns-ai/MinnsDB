//! AST parser with language dispatch

use crate::entities::{CodeEntity, CodeRelationship};
use crate::error::{AstError, AstResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub file_path: String,
    pub language: String,
    pub entities: Vec<CodeEntity>,
    pub relationships: Vec<CodeRelationship>,
    pub errors: Vec<ParseDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseDiagnostic {
    pub kind: DiagnosticKind,
    pub message: String,
    pub byte_range: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticKind {
    ParserError,
    UnsupportedNode,
    PartialExtraction,
}

/// Stateless AST parser — creates tree-sitter parsers on demand.
pub struct AstParser;

impl AstParser {
    pub fn new() -> Self {
        AstParser
    }

    /// Returns list of supported language identifiers.
    #[allow(clippy::vec_init_then_push)]
    pub fn supported_languages(&self) -> Vec<&str> {
        let mut langs = Vec::new();
        #[cfg(feature = "lang-rust")]
        langs.push("rust");
        #[cfg(feature = "lang-python")]
        langs.push("python");
        #[cfg(feature = "lang-javascript")]
        langs.push("javascript");
        #[cfg(feature = "lang-typescript")]
        langs.push("typescript");
        #[cfg(feature = "lang-go")]
        langs.push("go");
        langs
    }

    /// Detect language from file extension. Returns None for unsupported extensions.
    pub fn detect_language(file_path: &str) -> Option<String> {
        let ext = file_path.rsplit('.').next()?;
        match ext {
            "rs" => Some("rust".to_string()),
            "py" => Some("python".to_string()),
            "js" | "jsx" => Some("javascript".to_string()),
            "ts" | "tsx" => Some("typescript".to_string()),
            "go" => Some("go".to_string()),
            _ => None,
        }
    }

    /// Parse source code and extract entities + relationships.
    pub fn parse(&self, source: &str, language: &str, file_path: &str) -> AstResult<ParseResult> {
        match language {
            #[cfg(feature = "lang-rust")]
            "rust" => crate::languages::rust::parse_rust(source, file_path),
            _ => Err(AstError::UnsupportedLanguage(language.to_string())),
        }
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new()
    }
}
