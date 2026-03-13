//! Agent registry handlers — register agents and list agents in a group.
//!
//! ## Graph Structure
//!
//! ```text
//! [Agent Node]  ──agent:works_on──→  [Concept Node: repo:{repository}]
//!   (agent_id, agent_type, capabilities)
//! ```
//!
//! Agents and repos are scoped by `group_id` (stored as node property).
//! Multiple agents can work on the same repo. One agent can work on multiple repos.

use crate::errors::ApiError;
use crate::state::AppState;
use agent_db_graph::{ConceptType, EdgeType, GraphEdge, GraphNode, NodeType};
use axum::extract::{Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ============================================================================
// Request / Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct RegisterAgentRequest {
    pub agent_id: String,
    pub group_id: String,
    /// Repository this agent works on (e.g. "frontend", "backend")
    pub repository: Option<String>,
    /// What this agent does (e.g. ["code", "test", "review"])
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct RegisterAgentResponse {
    pub agent_node_id: u64,
    pub repo_node_id: Option<u64>,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct ListAgentsQuery {
    pub group_id: String,
}

#[derive(Debug, Serialize)]
pub struct AgentInfo {
    pub node_id: u64,
    pub agent_id: String,
    pub group_id: String,
    pub repositories: Vec<String>,
    pub capabilities: Vec<String>,
    pub last_seen: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ListAgentsResponse {
    pub agents: Vec<AgentInfo>,
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/agents/register — Register or update an agent.
///
/// Creates an Agent node (or updates existing) and links it to a repo Concept node.
pub async fn register_agent(
    State(state): State<AppState>,
    Json(req): Json<RegisterAgentRequest>,
) -> Result<Json<RegisterAgentResponse>, ApiError> {
    let engine = Arc::clone(&state.engine);
    let capabilities = req.capabilities.unwrap_or_default();

    let (agent_nid, repo_nid) = {
        let mut inf = engine.inference().write().await;
        let graph = inf.graph_mut();

        // Find or create Agent node
        let agent_nid = find_or_create_agent(graph, &req.agent_id, &req.group_id, &capabilities)
            .map_err(|e| ApiError::Internal(format!("Failed to create agent node: {e}")))?;

        // Update last_seen timestamp and capabilities
        if let Some(node) = graph.get_node_mut(agent_nid) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            node.properties.insert("last_seen".to_string(), json!(now));
            node.properties
                .insert("agent_name".to_string(), json!(req.agent_id));
            if !capabilities.is_empty() {
                node.properties
                    .insert("capabilities_list".to_string(), json!(capabilities));
            }
            node.touch();
        }

        // If repository specified, create repo concept + edge
        let repo_nid = if let Some(ref repo_name) = req.repository {
            let repo_concept_name = format!("repo:{}", repo_name);

            // Find or create repo Concept node
            let repo_nid = if let Some(nid) = graph.find_concept_node(&repo_concept_name) {
                if let Some(node) = graph.get_node_mut(nid) {
                    node.properties
                        .insert("group_id".to_string(), json!(req.group_id));
                    node.touch();
                }
                nid
            } else {
                let mut node = GraphNode::new(NodeType::Concept {
                    concept_name: repo_concept_name.clone(),
                    concept_type: ConceptType::Module,
                    confidence: 1.0,
                });
                node.properties
                    .insert("group_id".to_string(), json!(req.group_id));
                node.properties
                    .insert("repository".to_string(), json!(repo_name));
                node.properties
                    .insert("content_type".to_string(), json!("repository"));
                graph.add_node(node).map_err(|e| ApiError::Internal(format!("Failed to create repo node: {e}")))?
            };

            // Check if edge already exists
            let edges = graph.get_edges_from(agent_nid);
            let already_linked = edges.iter().any(|e: &&GraphEdge| {
                e.is_valid()
                    && e.target == repo_nid
                    && matches!(
                        &e.edge_type,
                        EdgeType::Association {
                            association_type, ..
                        } if association_type == "agent:works_on"
                    )
            });

            if !already_linked {
                let edge = GraphEdge::new(
                    agent_nid,
                    repo_nid,
                    EdgeType::Association {
                        association_type: "agent:works_on".to_string(),
                        evidence_count: 1,
                        statistical_significance: 1.0,
                    },
                    1.0,
                );
                graph.add_edge(edge);
            }

            Some(repo_nid)
        } else {
            None
        };

        (agent_nid, repo_nid)
    };

    // Embed agent node with descriptive text
    let embed_text = format!(
        "agent {} in group {} working on {} with capabilities: {}",
        req.agent_id,
        req.group_id,
        req.repository.as_deref().unwrap_or("unspecified"),
        capabilities.join(", "),
    );
    engine.embed_nodes_async(vec![(agent_nid, embed_text)]);

    Ok(Json(RegisterAgentResponse {
        agent_node_id: agent_nid,
        repo_node_id: repo_nid,
        status: "registered".to_string(),
    }))
}

/// GET /api/agents?group_id=... — List all agents in a group.
///
/// Walks Agent nodes filtered by group_id, resolves their repo links.
pub async fn list_agents(
    State(state): State<AppState>,
    Query(query): Query<ListAgentsQuery>,
) -> Result<Json<ListAgentsResponse>, ApiError> {
    let engine = Arc::clone(&state.engine);
    let inf = engine.inference().read().await;
    let graph = inf.graph();

    let mut agents = Vec::new();

    for node in graph.nodes() {
        if let NodeType::Agent {
            ref capabilities, ..
        } = node.node_type
        {
            let group = node
                .properties
                .get("group_id")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("");
            if group != query.group_id {
                continue;
            }

            let agent_name = node
                .properties
                .get("agent_name")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            // Resolve repositories from agent:works_on edges
            let mut repositories = Vec::new();
            for edge in graph.get_edges_from(node.id) {
                if !edge.is_valid() {
                    continue;
                }
                if let EdgeType::Association {
                    ref association_type,
                    ..
                } = edge.edge_type
                {
                    if association_type == "agent:works_on" {
                        if let Some(target) = graph.get_node(edge.target) {
                            if let Some(repo) = target
                                .properties
                                .get("repository")
                                .and_then(|v: &serde_json::Value| v.as_str())
                            {
                                repositories.push(repo.to_string());
                            }
                        }
                    }
                }
            }

            let last_seen = node
                .properties
                .get("last_seen")
                .and_then(|v: &serde_json::Value| v.as_u64());

            agents.push(AgentInfo {
                node_id: node.id,
                agent_id: agent_name,
                group_id: query.group_id.clone(),
                repositories,
                capabilities: capabilities.clone(),
                last_seen,
            });
        }
    }

    Ok(Json(ListAgentsResponse { agents }))
}

// ============================================================================
// Helpers
// ============================================================================

/// Deterministic hash of a string agent name to a numeric AgentId.
fn agent_name_to_id(name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

/// Find an existing Agent node by name + group_id, or create a new one.
fn find_or_create_agent(
    graph: &mut agent_db_graph::Graph,
    agent_name: &str,
    group_id: &str,
    capabilities: &[String],
) -> Result<u64, agent_db_graph::GraphError> {
    // Search for existing agent node by agent_name property
    for node in graph.nodes() {
        if let NodeType::Agent { .. } = node.node_type {
            let name = node
                .properties
                .get("agent_name")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("");
            let node_group = node
                .properties
                .get("group_id")
                .and_then(|v: &serde_json::Value| v.as_str())
                .unwrap_or("");
            if name == agent_name && node_group == group_id {
                return Ok(node.id);
            }
        }
    }

    // Create new Agent node
    let numeric_id = agent_name_to_id(agent_name);
    let mut node = GraphNode::new(NodeType::Agent {
        agent_id: numeric_id,
        agent_type: "claude-code".to_string(),
        capabilities: capabilities.to_vec(),
    });
    node.properties
        .insert("group_id".to_string(), json!(group_id));
    node.properties
        .insert("agent_name".to_string(), json!(agent_name));
    graph.add_node(node)
}
