//! Bulk knowledge graph import endpoint.
//!
//! POST /api/graph/import — Import nodes and edges directly into the graph.
//!
//! Supports all node types (Agent, Event, Context, Concept, Goal, Episode,
//! Memory, Strategy, Tool, Result, Claim) and all edge types (Causality,
//! Temporal, Contextual, Interaction, GoalRelation, Association,
//! Communication, DerivedFrom, SupportedBy, CodeStructure, About).
//!
//! Nodes are deduplicated by name for Concept nodes. Other node types
//! are always created fresh (they have internal IDs).

use crate::errors::ApiError;
use crate::state::AppState;
use agent_db_graph::structures::{
    ConceptType, EdgeType, GoalStatus, GraphEdge, GraphNode, NodeType,
};
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct GraphImportRequest {
    #[serde(default)]
    pub nodes: Vec<ImportNode>,
    #[serde(default)]
    pub edges: Vec<ImportEdge>,
    #[serde(default)]
    pub group_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ImportNode {
    /// Local reference name used by edges in this batch to refer to this node.
    pub name: String,

    /// Node type. See below for recognized values and their required fields.
    #[serde(default = "default_concept")]
    pub r#type: String,

    /// Type-specific and arbitrary properties. Which keys matter depends on `type`:
    ///
    /// **Concept** (default): `concept_type` (string), `confidence` (f64)
    /// **Agent**: `agent_id` (u64), `agent_type` (string), `capabilities` (string[])
    /// **Event**: `event_id` (u64), `event_type` (string), `significance` (f64)
    /// **Context**: `context_hash` (u64), `context_type` (string), `frequency` (u64)
    /// **Goal**: `goal_id` (u64), `description` (string), `priority` (f64), `status` (string)
    /// **Episode**: `episode_id` (u64), `agent_id` (u64), `session_id` (u64), `outcome` (string)
    /// **Memory**: `memory_id` (u64), `agent_id` (u64), `session_id` (u64)
    /// **Strategy**: `strategy_id` (u64), `agent_id` (u64)
    /// **Tool**: `tool_type` (string)
    /// **Result**: `result_type` (string), `summary` (string)
    /// **Claim**: `claim_id` (u64), `claim_text` (string), `confidence` (f64), `source_event_id` (u64)
    ///
    /// Unrecognized keys are stored as generic node properties.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
pub struct ImportEdge {
    /// Source node name (references `name` in the nodes array, or an existing Concept node).
    pub source: String,
    /// Target node name.
    pub target: String,

    /// Edge type. Recognized values:
    /// `association` (default), `causality`, `temporal`, `contextual`,
    /// `interaction`, `goal_relation`, `communication`, `derived_from`,
    /// `supported_by`, `code_structure`, `about`.
    ///
    /// For `association`, the `label` field (or `association_type`) is the relationship name.
    #[serde(default = "default_association")]
    pub r#type: String,

    /// Relationship label (used as association_type for Association edges,
    /// relation_kind for CodeStructure, predicate for About, etc.)
    #[serde(default)]
    pub label: Option<String>,

    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub valid_from: Option<u64>,
    #[serde(default)]
    pub valid_until: Option<u64>,

    /// Type-specific properties. Unrecognized keys stored as edge properties.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

fn default_concept() -> String {
    "concept".to_string()
}
fn default_association() -> String {
    "association".to_string()
}
fn default_confidence() -> f32 {
    0.9
}
fn default_weight() -> f32 {
    0.8
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct GraphImportResponse {
    pub nodes_created: usize,
    pub nodes_reused: usize,
    pub edges_created: usize,
    pub errors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub async fn import_graph(
    State(state): State<AppState>,
    Json(request): Json<GraphImportRequest>,
) -> Result<Json<GraphImportResponse>, ApiError> {
    if request.nodes.len() > 100_000 {
        return Err(ApiError::BadRequest("Too many nodes (max 100,000)".into()));
    }
    if request.edges.len() > 500_000 {
        return Err(ApiError::BadRequest("Too many edges (max 500,000)".into()));
    }

    info!(
        "Graph import: {} nodes, {} edges",
        request.nodes.len(),
        request.edges.len()
    );

    let group_id = request.group_id.unwrap_or_default();
    let mut name_to_id: HashMap<String, u64> = HashMap::with_capacity(request.nodes.len());
    let mut nodes_created = 0usize;
    let mut nodes_reused = 0usize;
    let mut edges_created = 0usize;
    let mut errors = Vec::new();

    // Phase 1: Nodes (single lock — HashMap inserts are O(1), fast even for 100k)
    {
        let mut inference = state.engine.inference().write().await;
        let graph = inference.graph_mut();

        for n in &request.nodes {
            let is_concept = matches!(
                n.r#type.to_lowercase().as_str(),
                "concept" | "person" | "organization" | "org" | "company"
                    | "location" | "place" | "city" | "country"
                    | "product" | "brand" | "named_entity" | "entity" | ""
            );

            if is_concept {
                let name_lower = n.name.to_lowercase();
                if let Some(existing) = graph
                    .get_concept_node(&name_lower)
                    .or_else(|| graph.get_concept_node(&n.name))
                {
                    name_to_id.insert(n.name.clone(), existing.id);
                    nodes_reused += 1;
                    continue;
                }
            }

            let node_type = build_node_type(&n.name, &n.r#type, &n.properties);
            let mut node = GraphNode::new(node_type);
            node.group_id = group_id.clone();
            for (k, v) in &n.properties {
                if !is_type_specific_key(&n.r#type, k) {
                    node.properties.insert(k.clone(), v.clone());
                }
            }

            match graph.add_node(node) {
                Ok(id) => {
                    name_to_id.insert(n.name.clone(), id);
                    nodes_created += 1;
                },
                Err(e) => errors.push(format!("Node '{}': {}", n.name, e)),
            }
        }
    }
    // Lock released — reads can proceed while we prepare edge batches

    // Phase 2: Edges in batches of 10k (release lock between batches)
    const EDGE_BATCH: usize = 10_000;
    for chunk in request.edges.chunks(EDGE_BATCH) {
        let mut inference = state.engine.inference().write().await;
        let graph = inference.graph_mut();

        for e in chunk {
            let source_id = match resolve_node(&e.source, &name_to_id, graph) {
                Some(id) => id,
                None => {
                    errors.push(format!("Edge source '{}' not found", e.source));
                    continue;
                },
            };
            let target_id = match resolve_node(&e.target, &name_to_id, graph) {
                Some(id) => id,
                None => {
                    errors.push(format!("Edge target '{}' not found", e.target));
                    continue;
                },
            };

            let label = e.label.clone().unwrap_or_else(|| {
                if e.r#type.to_lowercase() == "association" {
                    "related_to".to_string()
                } else {
                    e.r#type.clone()
                }
            });

            let edge_type = build_edge_type(&e.r#type, &label, e.confidence, &e.properties);
            let mut edge = GraphEdge::new(source_id, target_id, edge_type, e.weight);
            edge.confidence = e.confidence;
            edge.group_id = group_id.clone();
            edge.valid_from = e.valid_from;
            edge.valid_until = e.valid_until;
            for (k, v) in &e.properties {
                if !is_edge_type_specific_key(&e.r#type, k) {
                    edge.properties.insert(k.clone(), v.clone());
                }
            }

            match graph.add_edge(edge) {
                Some(_) => edges_created += 1,
                None => errors.push(format!(
                    "Edge '{}' → '{}': source or target node not in graph",
                    e.source, e.target
                )),
            }
        }

        drop(inference);
        tokio::task::yield_now().await;
    }

    info!(
        "Graph import done: {} created, {} reused, {} edges, {} errors",
        nodes_created,
        nodes_reused,
        edges_created,
        errors.len()
    );

    Ok(Json(GraphImportResponse {
        nodes_created,
        nodes_reused,
        edges_created,
        errors,
    }))
}

// ---------------------------------------------------------------------------
// Node type builders
// ---------------------------------------------------------------------------

fn build_node_type(
    name: &str,
    type_str: &str,
    props: &HashMap<String, serde_json::Value>,
) -> NodeType {
    match type_str.to_lowercase().as_str() {
        "agent" => NodeType::Agent {
            agent_id: prop_u64(props, "agent_id"),
            agent_type: prop_str(props, "agent_type"),
            capabilities: prop_str_vec(props, "capabilities"),
        },
        "event" => NodeType::Event {
            event_id: prop_u64(props, "event_id") as u128,
            event_type: prop_str_or(props, "event_type", "generic"),
            significance: prop_f32_or(props, "significance", 0.5),
        },
        "context" => NodeType::Context {
            context_hash: prop_u64(props, "context_hash"),
            context_type: prop_str_or(props, "context_type", "generic"),
            frequency: prop_u64(props, "frequency") as u32,
        },
        "goal" => NodeType::Goal {
            goal_id: prop_u64(props, "goal_id"),
            description: prop_str_or(props, "description", name),
            priority: prop_f32_or(props, "priority", 0.5),
            status: match prop_str(props, "status").to_lowercase().as_str() {
                "completed" => GoalStatus::Completed,
                "failed" => GoalStatus::Failed,
                "paused" => GoalStatus::Paused,
                _ => GoalStatus::Active,
            },
        },
        "episode" => NodeType::Episode {
            episode_id: prop_u64(props, "episode_id"),
            agent_id: prop_u64(props, "agent_id"),
            session_id: prop_u64(props, "session_id"),
            outcome: prop_str_or(props, "outcome", "unknown"),
        },
        "memory" => NodeType::Memory {
            memory_id: prop_u64(props, "memory_id"),
            agent_id: prop_u64(props, "agent_id"),
            session_id: prop_u64(props, "session_id"),
        },
        "strategy" => NodeType::Strategy {
            strategy_id: prop_u64(props, "strategy_id"),
            agent_id: prop_u64(props, "agent_id"),
            name: name.to_string(),
        },
        "tool" => NodeType::Tool {
            tool_name: name.to_string(),
            tool_type: prop_str_or(props, "tool_type", "generic"),
        },
        "result" => NodeType::Result {
            result_key: name.to_string(),
            result_type: prop_str_or(props, "result_type", "generic"),
            summary: prop_str_or(props, "summary", ""),
        },
        "claim" => NodeType::Claim {
            claim_id: prop_u64(props, "claim_id"),
            claim_text: prop_str_or(props, "claim_text", name),
            confidence: prop_f32_or(props, "confidence", 0.9),
            source_event_id: prop_u64(props, "source_event_id") as u128,
        },
        // Default: Concept node with mapped concept_type.
        _ => NodeType::Concept {
            concept_name: name.to_lowercase(),
            concept_type: map_concept_type(type_str),
            confidence: prop_f32_or(props, "confidence", 0.9),
        },
    }
}

fn build_edge_type(
    type_str: &str,
    label: &str,
    confidence: f32,
    props: &HashMap<String, serde_json::Value>,
) -> EdgeType {
    match type_str.to_lowercase().as_str() {
        "causality" | "causal" => EdgeType::Causality {
            strength: confidence,
            lag_ms: prop_u64(props, "lag_ms"),
        },
        "temporal" => EdgeType::Temporal {
            average_interval_ms: prop_u64(props, "average_interval_ms"),
            sequence_confidence: confidence,
        },
        "contextual" => EdgeType::Contextual {
            similarity: confidence,
            co_occurrence_rate: prop_f32_or(props, "co_occurrence_rate", 0.5),
        },
        "interaction" => EdgeType::Interaction {
            interaction_type: agent_db_graph::InteractionType::Communication,
            frequency: prop_u64(props, "frequency") as u32,
            success_rate: prop_f32_or(props, "success_rate", 0.5),
        },
        "goal_relation" => EdgeType::GoalRelation {
            relation_type: agent_db_graph::GoalRelationType::Dependency,
            dependency_strength: confidence,
        },
        "communication" => EdgeType::Communication {
            bandwidth: prop_f32_or(props, "bandwidth", 1.0),
            reliability: confidence,
            protocol: prop_str_or(props, "protocol", "direct"),
        },
        "derived_from" => EdgeType::DerivedFrom {
            extraction_confidence: confidence,
            extraction_timestamp: prop_u64(props, "extraction_timestamp"),
        },
        "supported_by" => EdgeType::SupportedBy {
            evidence_strength: confidence,
            span_offset: (0, 0),
        },
        "code_structure" | "code" => EdgeType::CodeStructure {
            relation_kind: label.to_string(),
            file_path: prop_str_or(props, "file_path", ""),
            confidence,
        },
        "about" => EdgeType::About {
            relevance_score: confidence,
            mention_count: prop_u64(props, "mention_count") as u32,
            entity_role: Default::default(),
            predicate: Some(label.to_string()),
        },
        // Default: Association with label as association_type.
        _ => EdgeType::Association {
            association_type: label.to_string(),
            evidence_count: 1,
            statistical_significance: confidence,
        },
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_node(
    name: &str,
    batch_map: &HashMap<String, u64>,
    graph: &agent_db_graph::Graph,
) -> Option<u64> {
    if let Some(&id) = batch_map.get(name) {
        return Some(id);
    }
    if let Some(n) = graph.get_concept_node(&name.to_lowercase()) {
        return Some(n.id);
    }
    if let Some(n) = graph.get_concept_node(name) {
        return Some(n.id);
    }
    None
}

fn map_concept_type(type_str: &str) -> ConceptType {
    match type_str.to_lowercase().as_str() {
        "person" => ConceptType::Person,
        "organization" | "org" | "company" => ConceptType::Organization,
        "location" | "place" | "city" | "country" => ConceptType::Location,
        "product" | "brand" => ConceptType::Product,
        "date" | "time" | "datetime" => ConceptType::DateTime,
        "function" | "method" => ConceptType::Function,
        "class" | "struct" => ConceptType::Class,
        "module" | "package" | "crate" => ConceptType::Module,
        "variable" | "constant" => ConceptType::Variable,
        "interface" | "trait" => ConceptType::Interface,
        "enum" => ConceptType::Enum,
        "pattern" | "behavior" => ConceptType::BehaviorPattern,
        _ => ConceptType::NamedEntity,
    }
}

fn prop_str(props: &HashMap<String, serde_json::Value>, key: &str) -> String {
    props
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn prop_str_or(props: &HashMap<String, serde_json::Value>, key: &str, default: &str) -> String {
    props
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

fn prop_u64(props: &HashMap<String, serde_json::Value>, key: &str) -> u64 {
    props.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

fn prop_f32_or(props: &HashMap<String, serde_json::Value>, key: &str, default: f32) -> f32 {
    props
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(default)
}

fn prop_str_vec(props: &HashMap<String, serde_json::Value>, key: &str) -> Vec<String> {
    props
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn is_type_specific_key(node_type: &str, key: &str) -> bool {
    match node_type.to_lowercase().as_str() {
        "agent" => matches!(key, "agent_id" | "agent_type" | "capabilities"),
        "event" => matches!(key, "event_id" | "event_type" | "significance"),
        "context" => matches!(key, "context_hash" | "context_type" | "frequency"),
        "goal" => matches!(key, "goal_id" | "description" | "priority" | "status"),
        "episode" => matches!(key, "episode_id" | "agent_id" | "session_id" | "outcome"),
        "memory" => matches!(key, "memory_id" | "agent_id" | "session_id"),
        "strategy" => matches!(key, "strategy_id" | "agent_id"),
        "tool" => matches!(key, "tool_type"),
        "result" => matches!(key, "result_type" | "summary"),
        "claim" => matches!(
            key,
            "claim_id" | "claim_text" | "confidence" | "source_event_id"
        ),
        _ => matches!(key, "concept_type" | "confidence"),
    }
}

fn is_edge_type_specific_key(edge_type: &str, key: &str) -> bool {
    match edge_type.to_lowercase().as_str() {
        "causality" | "causal" => matches!(key, "lag_ms"),
        "temporal" => matches!(key, "average_interval_ms"),
        "contextual" => matches!(key, "co_occurrence_rate"),
        "interaction" => matches!(key, "frequency" | "success_rate"),
        "communication" => matches!(key, "bandwidth" | "protocol"),
        "derived_from" => matches!(key, "extraction_timestamp"),
        "code_structure" | "code" => matches!(key, "file_path"),
        "about" => matches!(key, "mention_count"),
        _ => false,
    }
}
