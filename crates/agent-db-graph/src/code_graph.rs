//! AST-to-Graph bridge and code search
//!
//! Ingests `ParseResult` from `agent-db-ast` into the graph as Concept nodes
//! with CodeStructure edges. Also provides structural code search.

#[cfg(feature = "code-intelligence")]
use agent_db_ast::{CodeEntityKind, CodeRelationKind, ParseResult};

#[cfg(feature = "code-intelligence")]
use crate::structures::{ConceptType, EdgeType, Graph, GraphEdge, GraphNode, NodeId, NodeType};
#[cfg(not(feature = "code-intelligence"))]
use crate::structures::{Graph, NodeType};
#[cfg(feature = "code-intelligence")]
use crate::GraphResult;
#[cfg(feature = "code-intelligence")]
use serde_json::json;

/// Map a CodeEntityKind (from agent-db-ast) to a ConceptType (in this crate).
#[cfg(feature = "code-intelligence")]
pub fn entity_kind_to_concept_type(kind: CodeEntityKind) -> ConceptType {
    match kind {
        CodeEntityKind::Function | CodeEntityKind::Method => ConceptType::Function,
        CodeEntityKind::Class | CodeEntityKind::Struct => ConceptType::Class,
        CodeEntityKind::Enum => ConceptType::Enum,
        CodeEntityKind::Interface => ConceptType::Interface,
        CodeEntityKind::Module => ConceptType::Module,
        CodeEntityKind::Variable | CodeEntityKind::Constant => ConceptType::Variable,
        CodeEntityKind::TypeAlias => ConceptType::TypeAlias,
        CodeEntityKind::Import => ConceptType::Module, // imports reference modules
        CodeEntityKind::Field => ConceptType::Variable,
        CodeEntityKind::EnumVariant => ConceptType::Enum,
    }
}

/// Map a CodeRelationKind to a string for EdgeType::CodeStructure.
#[cfg(feature = "code-intelligence")]
fn relation_kind_to_string(kind: CodeRelationKind) -> String {
    match kind {
        CodeRelationKind::Contains => "contains".to_string(),
        CodeRelationKind::Imports => "imports".to_string(),
        CodeRelationKind::FieldOf => "field_of".to_string(),
        CodeRelationKind::Returns => "returns".to_string(),
        CodeRelationKind::ParameterType => "parameter_type".to_string(),
        CodeRelationKind::Calls => "calls".to_string(),
        CodeRelationKind::Implements => "implements".to_string(),
        CodeRelationKind::Inherits => "inherits".to_string(),
        CodeRelationKind::Overrides => "overrides".to_string(),
        CodeRelationKind::Uses => "uses".to_string(),
    }
}

/// Ingest a ParseResult into the graph.
///
/// - Nodes are deduped by `qualified_name` (used as `concept_name` in the concept index).
///   Existing nodes get their properties updated in place.
/// - Old `CodeStructure` edges from the same `file_path` are removed before new ones are added,
///   so re-indexing a file replaces stale relationships.
/// - Entities that existed in a previous parse but are absent in the new one get removed
///   (only for entities from the same file).
#[cfg(feature = "code-intelligence")]
pub fn ingest_parse_result(
    graph: &mut Graph,
    parse_result: &ParseResult,
    source_event_node_id: Option<NodeId>,
) -> GraphResult<Vec<NodeId>> {
    use std::collections::{HashMap, HashSet};

    let mut created_nodes = Vec::with_capacity(parse_result.entities.len());
    let mut qn_to_node_id: HashMap<String, NodeId> =
        HashMap::with_capacity(parse_result.entities.len());

    // Collect the set of qualified_names in this parse (for stale-node detection)
    let current_qnames: HashSet<&str> = parse_result
        .entities
        .iter()
        .filter(|e| e.kind != CodeEntityKind::Import)
        .map(|e| e.qualified_name.as_str())
        .collect();

    // Pass 1: Create/update Concept nodes for each entity (skip Import entities)
    for entity in &parse_result.entities {
        if entity.kind == CodeEntityKind::Import {
            continue; // Imports → edges only, no standalone concept node
        }

        let concept_type = entity_kind_to_concept_type(entity.kind);

        // Lookup by qualified_name — this is the concept_index key
        let existing = graph.find_concept_node(&entity.qualified_name);
        let node_id = if let Some(existing_id) = existing {
            // Update properties on existing node
            if let Some(node) = graph.get_node_mut(existing_id) {
                node.properties
                    .insert("file_path".to_string(), json!(entity.file_path));
                node.properties
                    .insert("line_range".to_string(), json!(entity.line_range));
                node.properties
                    .insert("language".to_string(), json!(entity.language));
                node.properties
                    .insert("content_type".to_string(), json!("code"));
                node.properties
                    .insert("kind".to_string(), json!(format!("{:?}", entity.kind)));
                // Store short name for display
                node.properties
                    .insert("display_name".to_string(), json!(entity.name));
                if let Some(ref sig) = entity.signature {
                    node.properties.insert("signature".to_string(), json!(sig));
                }
                if let Some(ref doc) = entity.doc_comment {
                    node.properties
                        .insert("doc_comment".to_string(), json!(doc));
                }
                if let Some(ref vis) = entity.visibility {
                    node.properties.insert("visibility".to_string(), json!(vis));
                }
                if let Some(ref ret) = entity.return_type {
                    node.properties
                        .insert("return_type".to_string(), json!(ret));
                }
                node.touch();
            }
            existing_id
        } else {
            // Create new concept node — use qualified_name as concept_name so the
            // concept_index correctly deduplicates by qualified_name on future ingests.
            let mut node = GraphNode::new(NodeType::Concept {
                concept_name: entity.qualified_name.clone(),
                concept_type,
                confidence: 1.0,
            });

            node.properties
                .insert("qualified_name".to_string(), json!(entity.qualified_name));
            node.properties
                .insert("display_name".to_string(), json!(entity.name));
            node.properties
                .insert("file_path".to_string(), json!(entity.file_path));
            node.properties
                .insert("line_range".to_string(), json!(entity.line_range));
            node.properties
                .insert("language".to_string(), json!(entity.language));
            node.properties
                .insert("content_type".to_string(), json!("code"));
            node.properties
                .insert("kind".to_string(), json!(format!("{:?}", entity.kind)));
            if let Some(ref sig) = entity.signature {
                node.properties.insert("signature".to_string(), json!(sig));
            }
            if let Some(ref doc) = entity.doc_comment {
                node.properties
                    .insert("doc_comment".to_string(), json!(doc));
            }
            if let Some(ref vis) = entity.visibility {
                node.properties.insert("visibility".to_string(), json!(vis));
            }
            if let Some(ref ret) = entity.return_type {
                node.properties
                    .insert("return_type".to_string(), json!(ret));
            }

            let nid = graph.add_node(node)?;
            created_nodes.push(nid);
            nid
        };

        qn_to_node_id.insert(entity.qualified_name.clone(), node_id);
    }

    // Pass 2: Remove stale CodeStructure edges from the same file, then add new ones.
    //
    // Collect all existing CodeStructure edge IDs whose file_path matches this parse,
    // so we can remove them before adding the fresh set.
    let file_path = &parse_result.file_path;
    let all_node_ids: Vec<NodeId> = qn_to_node_id.values().copied().collect();
    let mut stale_edge_ids: Vec<crate::structures::EdgeId> = Vec::new();
    for &nid in &all_node_ids {
        for edge in graph.get_edges_from(nid) {
            if let EdgeType::CodeStructure {
                file_path: ref efp, ..
            } = edge.edge_type
            {
                if efp == file_path {
                    stale_edge_ids.push(edge.id);
                }
            }
        }
    }
    for edge_id in stale_edge_ids {
        graph.remove_edge(edge_id);
    }

    // Now add fresh CodeStructure edges
    for rel in &parse_result.relationships {
        let source_id = qn_to_node_id.get(&rel.source).copied();
        let target_id = qn_to_node_id.get(&rel.target).copied();

        if let (Some(src), Some(tgt)) = (source_id, target_id) {
            let edge = GraphEdge::new(
                src,
                tgt,
                EdgeType::CodeStructure {
                    relation_kind: relation_kind_to_string(rel.kind),
                    file_path: rel.file_path.clone(),
                    confidence: rel.confidence,
                },
                rel.confidence,
            );
            graph.add_edge(edge);
        }
    }

    // Pass 3: Remove stale concept nodes — entities from this file that no longer exist
    // in the current parse. Only remove nodes whose file_path matches and whose
    // qualified_name is absent from the new parse.
    let mut stale_node_ids: Vec<NodeId> = Vec::new();
    for node in graph.nodes() {
        if let NodeType::Concept { .. } = &node.node_type {
            let is_code =
                node.properties.get("content_type").and_then(|v| v.as_str()) == Some("code");
            let same_file =
                node.properties.get("file_path").and_then(|v| v.as_str()) == Some(file_path);
            if is_code && same_file {
                let qn = node
                    .properties
                    .get("qualified_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if !current_qnames.contains(qn) {
                    stale_node_ids.push(node.id);
                }
            }
        }
    }
    for nid in stale_node_ids {
        graph.remove_node(nid);
    }

    // Pass 4: Link new concepts to source event via About edges
    if let Some(event_node_id) = source_event_node_id {
        for &node_id in &created_nodes {
            let edge = GraphEdge::new(
                event_node_id,
                node_id,
                EdgeType::About {
                    relevance_score: 1.0,
                    mention_count: 1,
                    entity_role: crate::claims::types::EntityRole::default(),
                    predicate: Some("defines".to_string()),
                },
                0.9,
            );
            graph.add_edge(edge);
        }
    }

    Ok(created_nodes)
}

/// Search code entities in the graph by filtering Concept nodes.
///
/// This is the non-feature-gated version that works on existing graph state.
pub fn search_code_entities_in_graph(
    graph: &Graph,
    name_pattern: Option<&str>,
    kind: Option<&str>,
    language: Option<&str>,
    file_pattern: Option<&str>,
    limit: usize,
) -> Vec<CodeEntityMatch> {
    let mut results = Vec::new();

    for node in graph.nodes() {
        if let NodeType::Concept {
            concept_name,
            concept_type,
            ..
        } = &node.node_type
        {
            // Only include code-derived concepts
            let is_code =
                node.properties.get("content_type").and_then(|v| v.as_str()) == Some("code");
            if !is_code {
                continue;
            }

            // Filter by kind
            if let Some(kind_filter) = kind {
                let node_kind = format!("{:?}", concept_type).to_lowercase();
                if !node_kind.contains(&kind_filter.to_lowercase()) {
                    continue;
                }
            }

            // Filter by name pattern (match against display_name or concept_name)
            if let Some(name_pat) = name_pattern {
                let pat = name_pat.to_lowercase();
                let display = node
                    .properties
                    .get("display_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(concept_name);
                if !display.to_lowercase().contains(&pat)
                    && !concept_name.to_lowercase().contains(&pat)
                {
                    continue;
                }
            }

            // Filter by language
            if let Some(lang_filter) = language {
                if let Some(lang) = node.properties.get("language").and_then(|v| v.as_str()) {
                    if !lang.eq_ignore_ascii_case(lang_filter) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Filter by file pattern
            if let Some(file_pat) = file_pattern {
                if let Some(fp) = node.properties.get("file_path").and_then(|v| v.as_str()) {
                    let pat = file_pat.to_lowercase();
                    if !fp.to_lowercase().contains(&pat) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            results.push(CodeEntityMatch {
                name: node
                    .properties
                    .get("display_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(concept_name)
                    .to_string(),
                qualified_name: node
                    .properties
                    .get("qualified_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(concept_name)
                    .to_string(),
                kind: format!("{:?}", concept_type),
                file_path: node
                    .properties
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                language: node
                    .properties
                    .get("language")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                line_range: node.properties.get("line_range").and_then(|v| {
                    let arr = v.as_array()?;
                    Some((
                        arr.first()?.as_u64()? as usize,
                        arr.get(1)?.as_u64()? as usize,
                    ))
                }),
                signature: node
                    .properties
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                doc_comment: node
                    .properties
                    .get("doc_comment")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                visibility: node
                    .properties
                    .get("visibility")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
            });

            if results.len() >= limit {
                break;
            }
        }
    }

    results
}

/// Build rich embedding text for a code entity node.
///
/// Combines kind, name, signature, doc comment, and file path into a
/// natural-language-ish string that will match semantic search queries like
/// "how does the AST parser work?" or "authentication handler".
pub fn build_code_embedding_text(node: &crate::structures::GraphNode) -> String {
    let kind = node
        .properties
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("code");
    let name = node
        .properties
        .get("display_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let qualified = node
        .properties
        .get("qualified_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file = node
        .properties
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let lang = node
        .properties
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let sig = node.properties.get("signature").and_then(|v| v.as_str());
    let doc = node.properties.get("doc_comment").and_then(|v| v.as_str());

    let mut text = format!("{} {} {}", lang, kind.to_lowercase(), name);
    if qualified != name && !qualified.is_empty() {
        text.push_str(&format!(" ({})", qualified));
    }
    if let Some(sig) = sig {
        text.push_str(&format!(": {}", sig));
    }
    if let Some(doc) = doc {
        // Truncate very long doc comments
        let doc_trimmed = if doc.len() > 500 { &doc[..500] } else { doc };
        text.push_str(&format!(". {}", doc_trimmed));
    }
    if !file.is_empty() {
        text.push_str(&format!(" in {}", file));
    }
    text
}

/// Result of a code entity search.
#[derive(Debug, Clone)]
pub struct CodeEntityMatch {
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub file_path: String,
    pub language: String,
    pub line_range: Option<(usize, usize)>,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
    pub visibility: Option<String>,
}
