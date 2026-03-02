//! Rust language extraction using tree-sitter

use crate::entities::*;
use crate::error::{AstError, AstResult};
use crate::parser::{ParseDiagnostic, ParseResult};

pub fn parse_rust(source: &str, file_path: &str) -> AstResult<ParseResult> {
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_rust::LANGUAGE;
    parser
        .set_language(&language.into())
        .map_err(|e| AstError::TreeSitter(format!("Failed to set Rust language: {}", e)))?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| AstError::ParseError {
            file: file_path.to_string(),
            message: "tree-sitter returned no tree".to_string(),
        })?;

    let mut extractor = RustExtractor {
        source: source.as_bytes(),
        file_path: file_path.to_string(),
        entities: Vec::new(),
        relationships: Vec::new(),
        errors: Vec::new(),
    };

    extractor.walk_node(tree.root_node(), None);

    Ok(ParseResult {
        file_path: file_path.to_string(),
        language: "rust".to_string(),
        entities: extractor.entities,
        relationships: extractor.relationships,
        errors: extractor.errors,
    })
}

struct RustExtractor<'a> {
    source: &'a [u8],
    file_path: String,
    entities: Vec<CodeEntity>,
    relationships: Vec<CodeRelationship>,
    errors: Vec<ParseDiagnostic>,
}

impl<'a> RustExtractor<'a> {
    fn node_text(&self, node: tree_sitter::Node) -> String {
        node.utf8_text(self.source).unwrap_or("").to_string()
    }

    fn line_range(&self, node: tree_sitter::Node) -> (usize, usize) {
        // tree-sitter is 0-indexed, we want 1-indexed
        (node.start_position().row + 1, node.end_position().row + 1)
    }

    fn byte_range(&self, node: tree_sitter::Node) -> (usize, usize) {
        (node.start_byte(), node.end_byte())
    }

    fn build_qualified_name(&self, name: &str, parent: Option<&str>) -> String {
        match parent {
            Some(p) => format!("{}::{}", p, name).to_lowercase(),
            None => name.to_lowercase(),
        }
    }

    fn extract_doc_comment(&self, node: tree_sitter::Node) -> Option<String> {
        // Look for preceding line_comment or block_comment nodes that start with /// or //!
        let mut doc_lines = Vec::new();
        let mut sibling = node.prev_sibling();
        while let Some(s) = sibling {
            if s.kind() == "line_comment" {
                let text = self.node_text(s);
                if text.starts_with("///") || text.starts_with("//!") {
                    doc_lines.push(
                        text.trim_start_matches("///")
                            .trim_start_matches("//!")
                            .trim()
                            .to_string(),
                    );
                } else {
                    break;
                }
            } else if s.kind() == "block_comment" {
                let text = self.node_text(s);
                if text.starts_with("/**") {
                    doc_lines.push(
                        text.trim_start_matches("/**")
                            .trim_end_matches("*/")
                            .trim()
                            .to_string(),
                    );
                }
                break;
            } else {
                break;
            }
            sibling = s.prev_sibling();
        }
        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }

    fn extract_visibility(&self, node: tree_sitter::Node) -> Option<String> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "visibility_modifier" {
                    return Some(self.node_text(child));
                }
            }
        }
        None
    }

    fn find_child_by_kind<'b>(
        &self,
        node: tree_sitter::Node<'b>,
        kind: &str,
    ) -> Option<tree_sitter::Node<'b>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == kind {
                    return Some(child);
                }
            }
        }
        None
    }

    fn extract_parameters(&self, params_node: tree_sitter::Node) -> Vec<Parameter> {
        let mut params = Vec::new();
        for i in 0..params_node.child_count() {
            if let Some(child) = params_node.child(i) {
                match child.kind() {
                    "parameter" => {
                        let name = self
                            .find_child_by_kind(child, "identifier")
                            .map(|n| self.node_text(n))
                            .filter(|n| !n.is_empty())
                            .unwrap_or_else(|| "_".to_string());
                        let type_ann = child.child_by_field_name("type").map(|n| self.node_text(n));
                        params.push(Parameter {
                            name,
                            type_annotation: type_ann,
                            default_value: None,
                        });
                    },
                    "self_parameter" => {
                        params.push(Parameter {
                            name: self.node_text(child),
                            type_annotation: None,
                            default_value: None,
                        });
                    },
                    _ => {},
                }
            }
        }
        params
    }

    fn extract_function_signature(&self, node: tree_sitter::Node) -> Option<String> {
        // Use tree-sitter's structural children to find the block node (body),
        // then extract text from start of function to start of body.
        // This avoids naive text search for `{` which can break on complex signatures.
        let body_node = self.find_child_by_kind(node, "block");
        if let Some(body) = body_node {
            let sig_end = body.start_byte();
            let sig_start = node.start_byte();
            if sig_end > sig_start {
                let sig_bytes = &self.source[sig_start..sig_end];
                if let Ok(sig_text) = std::str::from_utf8(sig_bytes) {
                    return Some(sig_text.trim().to_string());
                }
            }
        }
        // Fallback: no block body (e.g., trait method declaration)
        Some(self.node_text(node).trim().to_string())
    }

    fn walk_node(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        match node.kind() {
            "function_item" => self.extract_function(node, parent_qn),
            "struct_item" => self.extract_struct(node, parent_qn),
            "enum_item" => self.extract_enum(node, parent_qn),
            "impl_item" => self.extract_impl(node, parent_qn),
            "trait_item" => self.extract_trait(node, parent_qn),
            "use_declaration" => self.extract_use(node, parent_qn),
            "mod_item" => self.extract_mod(node, parent_qn),
            "const_item" => self.extract_const(node, parent_qn),
            "type_item" => self.extract_type_alias(node, parent_qn),
            "static_item" => self.extract_const(node, parent_qn), // treat static like const
            _ => {
                // Recurse into children
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        self.walk_node(child, parent_qn);
                    }
                }
            },
        }
    }

    fn extract_function(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);
        let signature = self.extract_function_signature(node);

        let params = node
            .child_by_field_name("parameters")
            .map(|n| self.extract_parameters(n))
            .unwrap_or_default();

        let return_type = node
            .child_by_field_name("return_type")
            .map(|n| self.node_text(n));

        let kind = if parent_qn.is_some() {
            CodeEntityKind::Method
        } else {
            CodeEntityKind::Function
        };

        // Collect parameter type annotations before moving params into entity
        let param_types: Vec<(String, String)> = params
            .iter()
            .filter_map(|p| {
                p.type_annotation
                    .as_ref()
                    .map(|t| (qualified_name.clone(), t.to_lowercase()))
            })
            .collect();

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature,
            doc_comment,
            visibility,
            parameters: params,
            return_type: return_type.clone(),
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Contains relationship from parent
        if let Some(parent) = parent_qn {
            self.relationships.push(CodeRelationship {
                source: parent.to_string(),
                target: qualified_name.clone(),
                kind: CodeRelationKind::Contains,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }

        // Returns relationship
        if let Some(ref ret_type) = return_type {
            let target_qn = ret_type.to_lowercase();
            self.relationships.push(CodeRelationship {
                source: qualified_name.clone(),
                target: target_qn,
                kind: CodeRelationKind::Returns,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }

        // ParameterType relationships (collected before params was moved)
        for (source, target) in param_types {
            self.relationships.push(CodeRelationship {
                source,
                target,
                kind: CodeRelationKind::ParameterType,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }
    }

    fn extract_struct(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::Struct,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: Some(format!("struct {}", name)),
            doc_comment,
            visibility,
            parameters: Vec::new(),
            return_type: None,
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Contains relationship from parent
        if let Some(parent) = parent_qn {
            self.relationships.push(CodeRelationship {
                source: parent.to_string(),
                target: qualified_name.clone(),
                kind: CodeRelationKind::Contains,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }

        // Extract fields from field_declaration_list
        if let Some(body) = self.find_child_by_kind(node, "field_declaration_list") {
            for i in 0..body.child_count() {
                if let Some(field) = body.child(i) {
                    if field.kind() == "field_declaration" {
                        self.extract_field(field, &qualified_name);
                    }
                }
            }
        }
    }

    fn extract_field(&mut self, node: tree_sitter::Node, parent_qn: &str) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, Some(parent_qn));
        let visibility = self.extract_visibility(node);
        let type_ann = node.child_by_field_name("type").map(|n| self.node_text(n));

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::Field,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: type_ann.as_ref().map(|t| format!("{}: {}", name, t)),
            doc_comment: self.extract_doc_comment(node),
            visibility,
            parameters: Vec::new(),
            return_type: type_ann,
            parent: Some(parent_qn.to_string()),
        };

        self.entities.push(entity);

        // FieldOf relationship
        self.relationships.push(CodeRelationship {
            source: qualified_name,
            target: parent_qn.to_string(),
            kind: CodeRelationKind::FieldOf,
            file_path: self.file_path.clone(),
            confidence: 1.0,
        });
    }

    fn extract_enum(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::Enum,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: Some(format!("enum {}", name)),
            doc_comment,
            visibility,
            parameters: Vec::new(),
            return_type: None,
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Extract variants from enum_variant_list
        if let Some(body) = self.find_child_by_kind(node, "enum_variant_list") {
            for i in 0..body.child_count() {
                if let Some(variant) = body.child(i) {
                    if variant.kind() == "enum_variant" {
                        self.extract_enum_variant(variant, &qualified_name);
                    }
                }
            }
        }
    }

    fn extract_enum_variant(&mut self, node: tree_sitter::Node, parent_qn: &str) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, Some(parent_qn));

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::EnumVariant,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: Some(self.node_text(node)),
            doc_comment: self.extract_doc_comment(node),
            visibility: None,
            parameters: Vec::new(),
            return_type: None,
            parent: Some(parent_qn.to_string()),
        };

        self.entities.push(entity);

        // Contains relationship from parent enum
        self.relationships.push(CodeRelationship {
            source: parent_qn.to_string(),
            target: qualified_name,
            kind: CodeRelationKind::Contains,
            file_path: self.file_path.clone(),
            confidence: 1.0,
        });
    }

    fn extract_impl(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        // Get the type being implemented
        let type_name = match node.child_by_field_name("type") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let impl_qn = self.build_qualified_name(&type_name, parent_qn);

        // Check if this is a trait impl (impl Trait for Type)
        if let Some(trait_node) = node.child_by_field_name("trait") {
            let trait_name = self.node_text(trait_node);
            let trait_qn = trait_name.to_lowercase();
            self.relationships.push(CodeRelationship {
                source: impl_qn.clone(),
                target: trait_qn,
                kind: CodeRelationKind::Implements,
                file_path: self.file_path.clone(),
                confidence: 0.9, // heuristic — qualified name of trait might not be fully resolved
            });
        }

        // Extract methods from the impl body
        if let Some(body) = self.find_child_by_kind(node, "declaration_list") {
            for i in 0..body.child_count() {
                if let Some(child) = body.child(i) {
                    if child.kind() == "function_item" {
                        self.extract_function(child, Some(&impl_qn));
                    } else if child.kind() == "const_item" {
                        self.extract_const(child, Some(&impl_qn));
                    } else if child.kind() == "type_item" {
                        self.extract_type_alias(child, Some(&impl_qn));
                    }
                }
            }
        }
    }

    fn extract_trait(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::Interface,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: Some(format!("trait {}", name)),
            doc_comment,
            visibility,
            parameters: Vec::new(),
            return_type: None,
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Extract trait methods from declaration_list
        if let Some(body) = self.find_child_by_kind(node, "declaration_list") {
            for i in 0..body.child_count() {
                if let Some(child) = body.child(i) {
                    if child.kind() == "function_item" || child.kind() == "function_signature_item"
                    {
                        self.extract_function(child, Some(&qualified_name));
                    }
                }
            }
        }
    }

    fn extract_use(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let use_text = self.node_text(node);
        let path_text = use_text
            .strip_prefix("use ")
            .unwrap_or(&use_text)
            .trim_end_matches(';')
            .trim();

        // Handle grouped imports: `use foo::{A, B, C};`
        // Split into individual import names
        let (base_path, import_names) = if let Some(brace_start) = path_text.find('{') {
            let base = path_text[..brace_start].trim_end_matches("::");
            let group = path_text[brace_start..]
                .trim_matches(|c: char| c == '{' || c == '}')
                .trim();
            let names: Vec<&str> = group
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty() && *s != "*")
                .collect();
            (Some(base), names)
        } else {
            // Single import: `use foo::bar;`
            let name = path_text.rsplit("::").next().unwrap_or(path_text).trim();
            if name.is_empty() || name == "*" {
                return;
            }
            (None, vec![name])
        };

        let visibility = self.extract_visibility(node);
        let byte_range = self.byte_range(node);
        let line_range = self.line_range(node);
        let source_module = parent_qn.unwrap_or(&self.file_path).to_string();

        for import_name in &import_names {
            let qualified_name = self.build_qualified_name(import_name, parent_qn);

            let entity = CodeEntity {
                name: import_name.to_string(),
                qualified_name: qualified_name.clone(),
                kind: CodeEntityKind::Import,
                file_path: self.file_path.clone(),
                language: "rust".to_string(),
                byte_range,
                line_range,
                signature: Some(use_text.trim().to_string()),
                doc_comment: None,
                visibility: visibility.clone(),
                parameters: Vec::new(),
                return_type: None,
                parent: parent_qn.map(|s| s.to_string()),
            };

            self.entities.push(entity);

            // Imports relationship
            let target_qn = if let Some(base) = base_path {
                format!("{}::{}", base, import_name).to_lowercase()
            } else {
                path_text.to_lowercase()
            };
            self.relationships.push(CodeRelationship {
                source: source_module.clone(),
                target: target_qn,
                kind: CodeRelationKind::Imports,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }
    }

    fn extract_mod(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::Module,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: Some(format!("mod {}", name)),
            doc_comment,
            visibility,
            parameters: Vec::new(),
            return_type: None,
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Contains relationship from parent
        if let Some(parent) = parent_qn {
            self.relationships.push(CodeRelationship {
                source: parent.to_string(),
                target: qualified_name.clone(),
                kind: CodeRelationKind::Contains,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }

        // Walk into module body if it has one (inline mod)
        if let Some(body) = self.find_child_by_kind(node, "declaration_list") {
            for i in 0..body.child_count() {
                if let Some(child) = body.child(i) {
                    self.walk_node(child, Some(&qualified_name));
                }
            }
        }
    }

    fn extract_const(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);
        let type_ann = node.child_by_field_name("type").map(|n| self.node_text(n));

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::Constant,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: type_ann.as_ref().map(|t| format!("const {}: {}", name, t)),
            doc_comment,
            visibility,
            parameters: Vec::new(),
            return_type: type_ann,
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Contains relationship from parent
        if let Some(parent) = parent_qn {
            self.relationships.push(CodeRelationship {
                source: parent.to_string(),
                target: qualified_name,
                kind: CodeRelationKind::Contains,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }
    }

    fn extract_type_alias(&mut self, node: tree_sitter::Node, parent_qn: Option<&str>) {
        let name = match node.child_by_field_name("name") {
            Some(n) => self.node_text(n),
            None => return,
        };

        let qualified_name = self.build_qualified_name(&name, parent_qn);
        let doc_comment = self.extract_doc_comment(node);
        let visibility = self.extract_visibility(node);
        let type_ann = node.child_by_field_name("type").map(|n| self.node_text(n));

        let entity = CodeEntity {
            name: name.clone(),
            qualified_name: qualified_name.clone(),
            kind: CodeEntityKind::TypeAlias,
            file_path: self.file_path.clone(),
            language: "rust".to_string(),
            byte_range: self.byte_range(node),
            line_range: self.line_range(node),
            signature: type_ann.as_ref().map(|t| format!("type {} = {}", name, t)),
            doc_comment,
            visibility,
            parameters: Vec::new(),
            return_type: type_ann,
            parent: parent_qn.map(|s| s.to_string()),
        };

        self.entities.push(entity);

        // Contains relationship from parent
        if let Some(parent) = parent_qn {
            self.relationships.push(CodeRelationship {
                source: parent.to_string(),
                target: qualified_name,
                kind: CodeRelationKind::Contains,
                file_path: self.file_path.clone(),
                confidence: 1.0,
            });
        }
    }
}
