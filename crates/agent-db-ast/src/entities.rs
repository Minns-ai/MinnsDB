//! Code entity and relationship types

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeEntityKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Module,
    Variable,
    Constant,
    TypeAlias,
    Import,
    Field,
    EnumVariant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntity {
    pub name: String,
    /// Dedup key. Format varies by language:
    ///
    /// - Rust: `crate::module::Type::method` (uses `::`)
    /// - Python: `package.module.Class.method` (uses `.`)
    /// - TS/JS: `module/path.Class.method`
    ///
    /// Always lowercase-normalized for dedup.
    pub qualified_name: String,
    pub kind: CodeEntityKind,
    pub file_path: String,
    pub language: String,
    pub byte_range: (usize, usize),
    /// 1-indexed line range
    pub line_range: (usize, usize),
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    pub visibility: Option<String>,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
    /// Parent's qualified_name
    pub parent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default_value: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeRelationKind {
    /// Module/class contains function/field (safe)
    Contains,
    /// File imports symbol (safe)
    Imports,
    /// Field belongs to struct/class (safe)
    FieldOf,
    /// Function returns type, if explicit (safe)
    Returns,
    /// Parameter is of type, if explicit (safe)
    ParameterType,
    /// Function calls function (heuristic)
    Calls,
    /// Struct implements trait (heuristic)
    Implements,
    /// Class extends class (heuristic)
    Inherits,
    /// Method overrides parent (heuristic)
    Overrides,
    /// References a type/variable (heuristic)
    Uses,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelationship {
    /// Source qualified name
    pub source: String,
    /// Target qualified name
    pub target: String,
    pub kind: CodeRelationKind,
    pub file_path: String,
    /// 1.0 for safe relationships, <1.0 for heuristic
    pub confidence: f32,
}
