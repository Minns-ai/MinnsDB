//! Workflow handlers — decompose workflow definitions into graph nodes and edges.
//!
//! ## Endpoints
//!
//! - `POST /api/workflows` — Create workflow: accepts a JSON definition, decomposes
//!   into Concept nodes (one per step + one root) and Association edges
//!   (`workflow:member_of`, `workflow:depends_on`).
//!
//! - `GET /api/workflows` — List workflows: finds all workflow root nodes.
//!
//! - `GET /api/workflows/:id` — Get workflow: reconstructs the full workflow
//!   definition from graph traversal.
//!
//! - `PUT /api/workflows/:id` — Update workflow: diffs old vs new, supersedes
//!   changed/removed edges (sets `valid_until`), creates new edges/nodes.
//!
//! - `DELETE /api/workflows/:id` — Delete workflow: supersedes all edges.
//!
//! - `POST /api/workflows/:id/steps/:step_id/transition` — Transition step state
//!   via structured memory state machines.
//!
//! ## Graph Structure
//!
//! ```text
//! [Workflow Root Node]  ←──member_of──  [Step Node: analyze_diff]
//!   (Concept/Strategy)                    (Concept/Strategy)
//!                       ←──member_of──  [Step Node: write_tests]
//!                                         │
//!                                         └──depends_on──→ [analyze_diff]
//! ```
//!
//! All nodes carry `group_id` for multi-tenant isolation.
//! All edges carry `valid_from`/`valid_until` for temporal tracking.

use crate::errors::ApiError;
use crate::state::AppState;
use crate::write_lanes::{WriteError, WriteJob};
use agent_db_graph::{ConceptType, EdgeType, GraphEdge, GraphNode, NodeType};
use axum::extract::{Path, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

// ============================================================================
// Request / Response Types
// ============================================================================

/// POST /api/workflows — create a new workflow.
#[derive(Debug, Deserialize)]
pub struct CreateWorkflowRequest {
    /// Workflow name (used as concept_name for the root node).
    pub name: String,
    /// Natural language intent that generated this workflow.
    #[serde(default)]
    pub intent: Option<String>,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Workflow steps.
    pub steps: Vec<WorkflowStepDef>,
    /// Multi-tenant partition key.
    #[serde(default)]
    pub group_id: String,
    /// Arbitrary metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A single step in the workflow definition.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkflowStepDef {
    /// Unique step identifier within this workflow.
    pub id: String,
    /// Role/agent type: research, code, test, review, etc.
    pub role: String,
    /// Task description.
    pub task: String,
    /// IDs of steps this step depends on (must complete first).
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Expected input data keys (from dependency outputs).
    #[serde(default)]
    pub inputs: Vec<String>,
    /// Output data keys this step produces.
    #[serde(default)]
    pub outputs: Vec<String>,
    /// Arbitrary step metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// PUT /api/workflows/:id — update an existing workflow.
#[derive(Debug, Deserialize)]
pub struct UpdateWorkflowRequest {
    /// Updated workflow name (optional).
    #[serde(default)]
    pub name: Option<String>,
    /// Updated intent.
    #[serde(default)]
    pub intent: Option<String>,
    /// Updated description.
    #[serde(default)]
    pub description: Option<String>,
    /// Full replacement step list.
    pub steps: Vec<WorkflowStepDef>,
}

/// POST /api/workflows/:id/steps/:step_id/transition
#[derive(Debug, Deserialize)]
pub struct StepTransitionRequest {
    /// Target state: ready, running, completed, failed.
    pub state: String,
    /// Optional result/output data (for completed steps).
    #[serde(default)]
    pub result: Option<String>,
}

/// GET /api/workflows query params.
#[derive(Debug, Deserialize)]
pub struct ListWorkflowsQuery {
    #[serde(default)]
    pub group_id: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

/// Response for workflow creation.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateWorkflowResponse {
    pub success: bool,
    pub workflow_id: u64,
    pub workflow_name: String,
    pub nodes_created: usize,
    pub edges_created: usize,
    pub step_node_ids: HashMap<String, u64>,
}

/// Response for workflow retrieval.
#[derive(Debug, Serialize)]
pub struct WorkflowResponse {
    pub workflow_id: u64,
    pub name: String,
    pub intent: Option<String>,
    pub description: Option<String>,
    pub group_id: String,
    pub created_at: String,
    pub steps: Vec<WorkflowStepResponse>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A step in the workflow response.
#[derive(Debug, Serialize)]
pub struct WorkflowStepResponse {
    pub node_id: u64,
    pub id: String,
    pub role: String,
    pub task: String,
    pub depends_on: Vec<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub state: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Response for workflow update.
#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateWorkflowResponse {
    pub success: bool,
    pub workflow_id: u64,
    pub nodes_created: usize,
    pub nodes_superseded: usize,
    pub edges_created: usize,
    pub edges_superseded: usize,
    pub step_node_ids: HashMap<String, u64>,
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/workflows — create a new workflow from a definition.
pub async fn create_workflow(
    State(state): State<AppState>,
    Json(request): Json<CreateWorkflowRequest>,
) -> Result<Json<CreateWorkflowResponse>, ApiError> {
    info!("Create workflow: name={}", request.name);

    // Validate step IDs are unique
    let mut seen_ids = std::collections::HashSet::new();
    for step in &request.steps {
        if !seen_ids.insert(step.id.clone()) {
            return Err(ApiError::BadRequest(format!(
                "Duplicate step ID: {}",
                step.id
            )));
        }
    }

    // Validate depends_on references
    for step in &request.steps {
        for dep in &step.depends_on {
            if !seen_ids.contains(dep) {
                return Err(ApiError::BadRequest(format!(
                    "Step '{}' depends on unknown step '{}'",
                    step.id, dep
                )));
            }
        }
    }

    let workflow_name = request.name.clone();
    let routing_key = hash_string_key(&workflow_name);

    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let now = timestamp_now_u64();

                    // ── Phase 1: Create graph nodes and edges ──
                    let mut inference = engine.inference().write().await;
                    let graph = inference.graph_mut();

                    // Create root workflow node
                    let mut root_node = GraphNode::new(NodeType::Concept {
                        concept_name: format!("workflow:{}", request.name),
                        concept_type: ConceptType::Strategy,
                        confidence: 1.0,
                    });
                    root_node.group_id = request.group_id.clone();
                    root_node
                        .properties
                        .insert("workflow_type".into(), json!("vibe_graph"));
                    root_node
                        .properties
                        .insert("workflow_name".into(), json!(request.name));
                    if let Some(ref intent) = request.intent {
                        root_node.properties.insert("intent".into(), json!(intent));
                    }
                    if let Some(ref desc) = request.description {
                        root_node
                            .properties
                            .insert("description".into(), json!(desc));
                    }
                    if !request.metadata.is_empty() {
                        root_node
                            .properties
                            .insert("metadata".into(), json!(request.metadata));
                    }
                    root_node
                        .properties
                        .insert("step_count".into(), json!(request.steps.len()));

                    let root_id = graph.add_node(root_node).map_err(|e| e.to_string())?;

                    // Create step nodes and edges
                    let mut step_node_ids: HashMap<String, u64> = HashMap::new();
                    let mut nodes_created = 1usize; // root
                    let mut edges_created = 0usize;

                    for step in &request.steps {
                        let step_concept_name = format!("wf:{}:step:{}", request.name, step.id);
                        let mut step_node = GraphNode::new(NodeType::Concept {
                            concept_name: step_concept_name,
                            concept_type: ConceptType::Strategy,
                            confidence: 1.0,
                        });
                        step_node.group_id = request.group_id.clone();
                        step_node
                            .properties
                            .insert("step_id".into(), json!(step.id));
                        step_node.properties.insert("role".into(), json!(step.role));
                        step_node.properties.insert("task".into(), json!(step.task));
                        step_node
                            .properties
                            .insert("workflow_name".into(), json!(request.name));
                        if !step.inputs.is_empty() {
                            step_node
                                .properties
                                .insert("inputs".into(), json!(step.inputs));
                        }
                        if !step.outputs.is_empty() {
                            step_node
                                .properties
                                .insert("outputs".into(), json!(step.outputs));
                        }
                        if !step.metadata.is_empty() {
                            step_node
                                .properties
                                .insert("metadata".into(), json!(step.metadata));
                        }

                        let step_nid = graph.add_node(step_node).map_err(|e| e.to_string())?;
                        step_node_ids.insert(step.id.clone(), step_nid);
                        nodes_created += 1;

                        // Create workflow:member_of edge (step → root)
                        let mut member_edge = GraphEdge::new(
                            step_nid,
                            root_id,
                            EdgeType::Association {
                                association_type: "workflow:member_of".to_string(),
                                evidence_count: 1,
                                statistical_significance: 1.0,
                            },
                            1.0,
                        );
                        member_edge.valid_from = Some(now);
                        member_edge.confidence = 1.0;
                        member_edge.group_id = request.group_id.clone();
                        member_edge
                            .properties
                            .insert("step_id".into(), json!(step.id));

                        if graph.add_edge(member_edge).is_some() {
                            edges_created += 1;
                        }
                    }

                    // Create workflow:depends_on edges
                    for step in &request.steps {
                        let from_nid = step_node_ids[&step.id];
                        for dep_id in &step.depends_on {
                            let to_nid = step_node_ids[dep_id];

                            // Data keys that flow through this dependency
                            let dep_step = request.steps.iter().find(|s| &s.id == dep_id);
                            let data_keys: Vec<&str> = dep_step
                                .map(|d| {
                                    d.outputs
                                        .iter()
                                        .filter(|o| step.inputs.contains(o))
                                        .map(|s| s.as_str())
                                        .collect()
                                })
                                .unwrap_or_default();

                            let mut dep_edge = GraphEdge::new(
                                from_nid,
                                to_nid,
                                EdgeType::Association {
                                    association_type: "workflow:depends_on".to_string(),
                                    evidence_count: 1,
                                    statistical_significance: 1.0,
                                },
                                1.0,
                            );
                            dep_edge.valid_from = Some(now);
                            dep_edge.confidence = 1.0;
                            dep_edge.group_id = request.group_id.clone();
                            if !data_keys.is_empty() {
                                dep_edge
                                    .properties
                                    .insert("data_keys".into(), json!(data_keys));
                            }

                            if graph.add_edge(dep_edge).is_some() {
                                edges_created += 1;
                            }
                        }
                    }

                    // ── Phase 2: Create initial state edges for each step ──
                    // Each step gets a `workflow:step_state` edge pointing to a
                    // "pending" concept node — same pattern as pipeline state tracking.
                    let pending_nid = ensure_concept_node(graph, "pending");
                    for step in &request.steps {
                        let step_nid = step_node_ids[&step.id];
                        let mut state_edge = GraphEdge::new(
                            step_nid,
                            pending_nid,
                            EdgeType::Association {
                                association_type: "workflow:step_state".to_string(),
                                evidence_count: 1,
                                statistical_significance: 1.0,
                            },
                            1.0,
                        );
                        state_edge.valid_from = Some(now);
                        state_edge.confidence = 1.0;
                        state_edge.group_id = request.group_id.clone();
                        state_edge
                            .properties
                            .insert("step_id".into(), json!(step.id));
                        if graph.add_edge(state_edge).is_some() {
                            edges_created += 1;
                        }
                    }

                    drop(inference);

                    // ── Phase 3: Embed nodes asynchronously ──
                    // Build rich text for each node so semantic search finds them.
                    let mut nodes_to_embed = Vec::new();

                    // Root node: "workflow: <name>. <intent>. <description>"
                    let mut root_text = format!("workflow: {}", request.name);
                    if let Some(ref intent) = request.intent {
                        root_text.push_str(&format!(". {}", intent));
                    }
                    if let Some(ref desc) = request.description {
                        root_text.push_str(&format!(". {}", desc));
                    }
                    nodes_to_embed.push((root_id, root_text));

                    // Step nodes: "workflow step: <role> — <task>"
                    for step in &request.steps {
                        let step_nid = step_node_ids[&step.id];
                        let step_text = format!("workflow step: {} — {}", step.role, step.task);
                        nodes_to_embed.push((step_nid, step_text));
                    }

                    engine.embed_nodes_async(nodes_to_embed);

                    let resp = CreateWorkflowResponse {
                        success: true,
                        workflow_id: root_id,
                        workflow_name: request.name.clone(),
                        nodes_created,
                        edges_created,
                        step_node_ids,
                    };

                    serde_json::to_value(resp).map_err(|e| e.to_string())
                })
            }),
            result_tx: tx,
        })
        .await?;

    let resp: CreateWorkflowResponse =
        serde_json::from_value(result).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(Json(resp))
}

/// GET /api/workflows — list all workflows.
pub async fn list_workflows(
    State(state): State<AppState>,
    Query(query): Query<ListWorkflowsQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    let inference = state.engine.inference().read().await;
    let graph = inference.graph();

    // Find all workflow root nodes by iterating concept nodes
    let mut workflows = Vec::new();
    for node in graph.nodes() {
        // Check if this is a workflow root node
        let is_workflow = node
            .properties
            .get("workflow_type")
            .and_then(|v| v.as_str())
            == Some("vibe_graph");
        if !is_workflow {
            continue;
        }

        // Filter by group_id if specified
        if let Some(ref gid) = query.group_id {
            if &node.group_id != gid {
                continue;
            }
        }

        // Check edges are still valid (not deleted)
        let has_valid_edges = graph.get_edges_to(node.id).iter().any(|e| e.is_valid());

        let wf_name = node
            .properties
            .get("workflow_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let step_count = node
            .properties
            .get("step_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let intent = node.properties.get("intent").and_then(|v| v.as_str());

        workflows.push(json!({
            "workflow_id": node.id,
            "name": wf_name,
            "intent": intent,
            "step_count": step_count,
            "group_id": node.group_id,
            "created_at": format_timestamp(node.created_at),
            "active": has_valid_edges,
        }));

        if workflows.len() >= query.limit {
            break;
        }
    }

    Ok(Json(json!({
        "workflows": workflows,
        "count": workflows.len(),
    })))
}

/// GET /api/workflows/:id — get workflow by root node ID.
pub async fn get_workflow(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<WorkflowResponse>, ApiError> {
    let _permit = state
        .read_gate
        .acquire()
        .await
        .map_err(ApiError::ServiceUnavailable)?;

    let inference = state.engine.inference().read().await;
    let graph = inference.graph();

    // Find root node
    let root_node = graph
        .get_node(id)
        .ok_or_else(|| ApiError::NotFound(format!("Workflow node {} not found", id)))?;

    // Verify it's a workflow node
    if root_node
        .properties
        .get("workflow_type")
        .and_then(|v| v.as_str())
        != Some("vibe_graph")
    {
        return Err(ApiError::BadRequest(format!(
            "Node {} is not a workflow root",
            id
        )));
    }

    let wf_name = root_node
        .properties
        .get("workflow_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Find step nodes via incoming member_of edges
    let incoming_edges = graph.get_edges_to(id);
    let step_entries: Vec<(u64, String)> = incoming_edges
        .iter()
        .filter_map(|edge| {
            if !edge.is_valid() {
                return None;
            }
            if let EdgeType::Association {
                ref association_type,
                ..
            } = edge.edge_type
            {
                if association_type == "workflow:member_of" {
                    let step_id = edge
                        .properties
                        .get("step_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    return Some((edge.source, step_id));
                }
            }
            None
        })
        .collect();

    // Collect step info from graph
    struct StepInfo {
        node_id: u64,
        step_id: String,
        role: String,
        task: String,
        depends_on: Vec<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        metadata: HashMap<String, serde_json::Value>,
    }

    let mut step_infos = Vec::new();
    for (step_nid, step_id) in &step_entries {
        let step_node = match graph.get_node(*step_nid) {
            Some(n) => n,
            None => continue,
        };

        let role = step_node
            .properties
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let task = step_node
            .properties
            .get("task")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let inputs: Vec<String> = step_node
            .properties
            .get("inputs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let outputs: Vec<String> = step_node
            .properties
            .get("outputs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let meta: HashMap<String, serde_json::Value> = step_node
            .properties
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Find depends_on via outgoing workflow:depends_on edges
        let mut depends_on = Vec::new();
        for edge in graph.get_edges_from(*step_nid) {
            if !edge.is_valid() {
                continue;
            }
            if let EdgeType::Association {
                ref association_type,
                ..
            } = edge.edge_type
            {
                if association_type == "workflow:depends_on" {
                    if let Some(dep_entry) =
                        step_entries.iter().find(|(nid, _)| *nid == edge.target)
                    {
                        depends_on.push(dep_entry.1.clone());
                    }
                }
            }
        }

        step_infos.push(StepInfo {
            node_id: *step_nid,
            step_id: step_id.clone(),
            role,
            task,
            depends_on,
            inputs,
            outputs,
            metadata: meta,
        });
    }

    let group_id = root_node.group_id.clone();
    let created_at = format_timestamp(root_node.created_at);
    let intent = root_node
        .properties
        .get("intent")
        .and_then(|v| v.as_str())
        .map(String::from);
    let description = root_node
        .properties
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);
    let metadata: HashMap<String, serde_json::Value> = root_node
        .properties
        .get("metadata")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    // Read step states from graph edges (workflow:step_state)
    let steps: Vec<WorkflowStepResponse> = step_infos
        .into_iter()
        .map(|si| {
            let current_state = resolve_step_state(graph, si.node_id);

            WorkflowStepResponse {
                node_id: si.node_id,
                id: si.step_id,
                role: si.role,
                task: si.task,
                depends_on: si.depends_on,
                inputs: si.inputs,
                outputs: si.outputs,
                state: current_state,
                metadata: si.metadata,
            }
        })
        .collect();

    drop(inference);

    Ok(Json(WorkflowResponse {
        workflow_id: id,
        name: wf_name,
        intent,
        description,
        group_id,
        created_at,
        steps,
        metadata,
    }))
}

/// PUT /api/workflows/:id — update a workflow definition.
pub async fn update_workflow(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Json(request): Json<UpdateWorkflowRequest>,
) -> Result<Json<UpdateWorkflowResponse>, ApiError> {
    info!("Update workflow: id={}", id);

    // Validate step IDs
    let mut seen_ids = std::collections::HashSet::new();
    for step in &request.steps {
        if !seen_ids.insert(step.id.clone()) {
            return Err(ApiError::BadRequest(format!(
                "Duplicate step ID: {}",
                step.id
            )));
        }
    }
    for step in &request.steps {
        for dep in &step.depends_on {
            if !seen_ids.contains(dep) {
                return Err(ApiError::BadRequest(format!(
                    "Step '{}' depends on unknown step '{}'",
                    step.id, dep
                )));
            }
        }
    }

    let routing_key = id;

    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let now = timestamp_now_u64();
                    let mut inference = engine.inference().write().await;
                    let graph = inference.graph_mut();

                    // Verify root exists
                    let root_node = graph
                        .get_node(id)
                        .ok_or_else(|| format!("Workflow node {} not found", id))?;

                    let wf_name = root_node
                        .properties
                        .get("workflow_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let group_id = root_node.group_id.clone();

                    // Collect existing step node IDs from incoming member_of edges
                    let mut existing_steps: HashMap<String, u64> = HashMap::new();
                    for edge in graph.get_edges_to(id) {
                        if !edge.is_valid() {
                            continue;
                        }
                        if let EdgeType::Association {
                            ref association_type,
                            ..
                        } = edge.edge_type
                        {
                            if association_type == "workflow:member_of" {
                                let step_id = edge
                                    .properties
                                    .get("step_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if !step_id.is_empty() {
                                    existing_steps.insert(step_id, edge.source);
                                }
                            }
                        }
                    }

                    let new_step_ids: std::collections::HashSet<String> =
                        request.steps.iter().map(|s| s.id.clone()).collect();

                    let mut nodes_created = 0usize;
                    let mut nodes_superseded = 0usize;
                    let mut edges_created = 0usize;
                    let mut edges_superseded = 0usize;
                    let mut step_node_ids: HashMap<String, u64> = HashMap::new();

                    // ── Supersede removed steps ──
                    for (step_id, step_nid) in &existing_steps {
                        if !new_step_ids.contains(step_id) {
                            edges_superseded += supersede_node_edges(graph, *step_nid, now);
                            nodes_superseded += 1;
                        }
                    }

                    // ── Create or update steps ──
                    for step in &request.steps {
                        if let Some(&existing_nid) = existing_steps.get(&step.id) {
                            // Step exists — update properties
                            if let Some(node) = graph.get_node_mut(existing_nid) {
                                node.properties.insert("role".into(), json!(step.role));
                                node.properties.insert("task".into(), json!(step.task));
                                if !step.inputs.is_empty() {
                                    node.properties.insert("inputs".into(), json!(step.inputs));
                                }
                                if !step.outputs.is_empty() {
                                    node.properties
                                        .insert("outputs".into(), json!(step.outputs));
                                }
                                node.touch();
                            }
                            step_node_ids.insert(step.id.clone(), existing_nid);

                            // Supersede old depends_on edges
                            let dep_eids: Vec<u64> = graph
                                .get_edges_from(existing_nid)
                                .iter()
                                .filter(|e| {
                                    e.is_valid()
                                        && matches!(
                                            &e.edge_type,
                                            EdgeType::Association {
                                                association_type,
                                                ..
                                            } if association_type == "workflow:depends_on"
                                        )
                                })
                                .map(|e| e.id)
                                .collect();
                            for eid in dep_eids {
                                graph.invalidate_edge(eid, "workflow_update");
                                edges_superseded += 1;
                            }
                        } else {
                            // New step — create node
                            let step_concept_name = format!("wf:{}:step:{}", wf_name, step.id);
                            let mut step_node = GraphNode::new(NodeType::Concept {
                                concept_name: step_concept_name,
                                concept_type: ConceptType::Strategy,
                                confidence: 1.0,
                            });
                            step_node.group_id = group_id.clone();
                            step_node
                                .properties
                                .insert("step_id".into(), json!(step.id));
                            step_node.properties.insert("role".into(), json!(step.role));
                            step_node.properties.insert("task".into(), json!(step.task));
                            step_node
                                .properties
                                .insert("workflow_name".into(), json!(wf_name));
                            if !step.inputs.is_empty() {
                                step_node
                                    .properties
                                    .insert("inputs".into(), json!(step.inputs));
                            }
                            if !step.outputs.is_empty() {
                                step_node
                                    .properties
                                    .insert("outputs".into(), json!(step.outputs));
                            }

                            let step_nid = graph.add_node(step_node).map_err(|e| e.to_string())?;
                            step_node_ids.insert(step.id.clone(), step_nid);
                            nodes_created += 1;

                            // Create member_of edge
                            let mut member_edge = GraphEdge::new(
                                step_nid,
                                id,
                                EdgeType::Association {
                                    association_type: "workflow:member_of".to_string(),
                                    evidence_count: 1,
                                    statistical_significance: 1.0,
                                },
                                1.0,
                            );
                            member_edge.valid_from = Some(now);
                            member_edge.confidence = 1.0;
                            member_edge.group_id = group_id.clone();
                            member_edge
                                .properties
                                .insert("step_id".into(), json!(step.id));
                            if graph.add_edge(member_edge).is_some() {
                                edges_created += 1;
                            }
                        }
                    }

                    // Merge step_node_ids with existing
                    for (k, v) in &existing_steps {
                        step_node_ids.entry(k.clone()).or_insert(*v);
                    }

                    // ── Recreate depends_on edges ──
                    for step in &request.steps {
                        let from_nid = match step_node_ids.get(&step.id) {
                            Some(&nid) => nid,
                            None => continue,
                        };
                        for dep_id in &step.depends_on {
                            let to_nid = match step_node_ids.get(dep_id) {
                                Some(&nid) => nid,
                                None => continue,
                            };

                            let dep_step = request.steps.iter().find(|s| &s.id == dep_id);
                            let data_keys: Vec<&str> = dep_step
                                .map(|d| {
                                    d.outputs
                                        .iter()
                                        .filter(|o| step.inputs.contains(o))
                                        .map(|s| s.as_str())
                                        .collect()
                                })
                                .unwrap_or_default();

                            let mut dep_edge = GraphEdge::new(
                                from_nid,
                                to_nid,
                                EdgeType::Association {
                                    association_type: "workflow:depends_on".to_string(),
                                    evidence_count: 1,
                                    statistical_significance: 1.0,
                                },
                                1.0,
                            );
                            dep_edge.valid_from = Some(now);
                            dep_edge.confidence = 1.0;
                            dep_edge.group_id = group_id.clone();
                            if !data_keys.is_empty() {
                                dep_edge
                                    .properties
                                    .insert("data_keys".into(), json!(data_keys));
                            }

                            if graph.add_edge(dep_edge).is_some() {
                                edges_created += 1;
                            }
                        }
                    }

                    // Update root node
                    if let Some(root) = graph.get_node_mut(id) {
                        if let Some(ref name) = request.name {
                            root.properties.insert("workflow_name".into(), json!(name));
                        }
                        if let Some(ref intent) = request.intent {
                            root.properties.insert("intent".into(), json!(intent));
                        }
                        if let Some(ref desc) = request.description {
                            root.properties.insert("description".into(), json!(desc));
                        }
                        root.properties
                            .insert("step_count".into(), json!(request.steps.len()));
                        root.touch();
                    }

                    // Create initial state edges for new steps
                    let pending_nid = ensure_concept_node(graph, "pending");
                    for step in &request.steps {
                        if !existing_steps.contains_key(&step.id) {
                            let step_nid = step_node_ids[&step.id];
                            let mut state_edge = GraphEdge::new(
                                step_nid,
                                pending_nid,
                                EdgeType::Association {
                                    association_type: "workflow:step_state".to_string(),
                                    evidence_count: 1,
                                    statistical_significance: 1.0,
                                },
                                1.0,
                            );
                            state_edge.valid_from = Some(now);
                            state_edge.confidence = 1.0;
                            state_edge.group_id = group_id.clone();
                            state_edge
                                .properties
                                .insert("step_id".into(), json!(step.id));
                            if graph.add_edge(state_edge).is_some() {
                                edges_created += 1;
                            }
                        }
                    }

                    drop(inference);

                    let resp = UpdateWorkflowResponse {
                        success: true,
                        workflow_id: id,
                        nodes_created,
                        nodes_superseded,
                        edges_created,
                        edges_superseded,
                        step_node_ids,
                    };

                    serde_json::to_value(resp).map_err(|e| e.to_string())
                })
            }),
            result_tx: tx,
        })
        .await?;

    let resp: UpdateWorkflowResponse =
        serde_json::from_value(result).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(Json(resp))
}

/// DELETE /api/workflows/:id — supersede all workflow edges (soft delete).
pub async fn delete_workflow(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!("Delete workflow: id={}", id);

    let routing_key = id;

    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let now = timestamp_now_u64();
                    let mut inference = engine.inference().write().await;
                    let graph = inference.graph_mut();

                    // Verify root exists
                    if graph.get_node(id).is_none() {
                        return Err(format!("Workflow node {} not found", id));
                    }

                    // Find all step nodes via incoming member_of edges
                    let step_nids: Vec<u64> = graph
                        .get_edges_to(id)
                        .iter()
                        .filter_map(|edge| {
                            if !edge.is_valid() {
                                return None;
                            }
                            if let EdgeType::Association {
                                ref association_type,
                                ..
                            } = edge.edge_type
                            {
                                if association_type == "workflow:member_of" {
                                    return Some(edge.source);
                                }
                            }
                            None
                        })
                        .collect();

                    let mut edges_superseded = 0usize;

                    // Supersede all edges for each step
                    for step_nid in &step_nids {
                        edges_superseded += supersede_node_edges(graph, *step_nid, now);
                    }
                    // Supersede root edges
                    edges_superseded += supersede_node_edges(graph, id, now);

                    drop(inference);

                    // State edges are already superseded as part of
                    // supersede_node_edges — no separate cleanup needed.

                    Ok(json!({
                        "success": true,
                        "workflow_id": id,
                        "edges_superseded": edges_superseded,
                    }))
                })
            }),
            result_tx: tx,
        })
        .await?;

    Ok(Json(result))
}

/// POST /api/workflows/:id/steps/:step_id/transition — transition step state.
///
/// Uses the same pattern as the pipeline: supersede the old state edge
/// (`valid_until = now`) and create a new edge to the new state concept node.
/// This makes step state visible to NLQ graph projections.
pub async fn workflow_step_transition(
    State(state): State<AppState>,
    Path((id, step_id)): Path<(u64, String)>,
    Json(request): Json<StepTransitionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!(
        "Workflow step transition: wf={}, step={}, state={}",
        id, step_id, request.state
    );

    let routing_key = id;

    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let now = timestamp_now_u64();
                    let mut inference = engine.inference().write().await;
                    let graph = inference.graph_mut();

                    // Verify root exists
                    let root_node = graph
                        .get_node(id)
                        .ok_or_else(|| format!("Workflow node {} not found", id))?;
                    let group_id = root_node.group_id.clone();

                    // Find the step node via incoming member_of edges
                    let step_nid = graph
                        .get_edges_to(id)
                        .iter()
                        .find_map(|edge| {
                            if !edge.is_valid() {
                                return None;
                            }
                            if let EdgeType::Association {
                                ref association_type,
                                ..
                            } = edge.edge_type
                            {
                                if association_type == "workflow:member_of" {
                                    let sid =
                                        edge.properties.get("step_id").and_then(|v| v.as_str());
                                    if sid == Some(&step_id) {
                                        return Some(edge.source);
                                    }
                                }
                            }
                            None
                        })
                        .ok_or_else(|| {
                            format!("Step '{}' not found in workflow {}", step_id, id)
                        })?;

                    // Supersede the current active state edge
                    let old_state_eids: Vec<u64> = graph
                        .get_edges_from(step_nid)
                        .iter()
                        .filter(|e| {
                            e.is_valid()
                                && matches!(
                                    &e.edge_type,
                                    EdgeType::Association {
                                        association_type,
                                        ..
                                    } if association_type == "workflow:step_state"
                                )
                        })
                        .map(|e| e.id)
                        .collect();

                    for eid in old_state_eids {
                        graph.invalidate_edge(eid, "step_transition");
                    }

                    // Create concept node for the new state value
                    let new_state_nid = ensure_concept_node(graph, &request.state);

                    // Create new state edge
                    let mut state_edge = GraphEdge::new(
                        step_nid,
                        new_state_nid,
                        EdgeType::Association {
                            association_type: "workflow:step_state".to_string(),
                            evidence_count: 1,
                            statistical_significance: 1.0,
                        },
                        1.0,
                    );
                    state_edge.valid_from = Some(now);
                    state_edge.confidence = 1.0;
                    state_edge.group_id = group_id;
                    state_edge
                        .properties
                        .insert("step_id".into(), json!(step_id.clone()));

                    // Store result text on the edge if provided
                    if let Some(ref result_text) = request.result {
                        state_edge
                            .properties
                            .insert("result".into(), json!(result_text));
                    }

                    graph.add_edge(state_edge);

                    drop(inference);

                    Ok(json!({
                        "success": true,
                        "workflow_id": id,
                        "step_id": step_id,
                        "new_state": request.state,
                    }))
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::BadRequest(e)),
        Err(e) => Err(ApiError::from(e)),
    }
}

// ============================================================================
// Workflow Feedback
// ============================================================================

/// Request to attach feedback to a workflow.
#[derive(Debug, Deserialize)]
pub struct WorkflowFeedbackRequest {
    /// Free-form feedback text (what worked, what didn't, lessons learned).
    pub feedback: String,
    /// Overall outcome: "success", "partial", "failure".
    #[serde(default)]
    pub outcome: Option<String>,
}

/// POST /api/workflows/:id/feedback — attach feedback to a completed workflow.
///
/// Creates a `workflow:feedback` edge from the workflow root to a new concept
/// node containing the feedback text. The feedback is embedded so NLQ can find
/// it when searching for past outcomes and lessons learned.
pub async fn workflow_feedback(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Json(request): Json<WorkflowFeedbackRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!(
        "Workflow feedback: wf={}, outcome={:?}",
        id, request.outcome
    );

    let routing_key = id;

    let result = state
        .write_lanes
        .submit_and_await(routing_key, |tx| WriteJob::GenericWrite {
            operation: Box::new(move |engine: Arc<agent_db_graph::GraphEngine>| {
                Box::pin(async move {
                    let now = timestamp_now_u64();
                    let mut inference = engine.inference().write().await;
                    let graph = inference.graph_mut();

                    // Verify root exists and is a workflow
                    let root_node = graph
                        .get_node(id)
                        .ok_or_else(|| format!("Workflow node {} not found", id))?;

                    if root_node
                        .properties
                        .get("workflow_type")
                        .and_then(|v| v.as_str())
                        != Some("vibe_graph")
                    {
                        return Err(format!("Node {} is not a workflow root", id));
                    }

                    let group_id = root_node.group_id.clone();
                    let wf_name = root_node
                        .properties
                        .get("workflow_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    // Create a concept node for the feedback
                    let feedback_concept_name = format!("wf:{}:feedback:{}", wf_name, now);
                    let mut feedback_node = GraphNode::new(NodeType::Concept {
                        concept_name: feedback_concept_name,
                        concept_type: ConceptType::NamedEntity,
                        confidence: 1.0,
                    });
                    feedback_node.group_id = group_id.clone();
                    feedback_node
                        .properties
                        .insert("feedback".into(), json!(request.feedback));
                    feedback_node
                        .properties
                        .insert("workflow_name".into(), json!(wf_name));
                    if let Some(ref outcome) = request.outcome {
                        feedback_node
                            .properties
                            .insert("outcome".into(), json!(outcome));
                    }

                    let feedback_nid = graph.add_node(feedback_node).map_err(|e| e.to_string())?;

                    // Create workflow:feedback edge from root → feedback node
                    let mut feedback_edge = GraphEdge::new(
                        id,
                        feedback_nid,
                        EdgeType::Association {
                            association_type: "workflow:feedback".to_string(),
                            evidence_count: 1,
                            statistical_significance: 1.0,
                        },
                        1.0,
                    );
                    feedback_edge.valid_from = Some(now);
                    feedback_edge.confidence = 1.0;
                    feedback_edge.group_id = group_id;
                    if let Some(ref outcome) = request.outcome {
                        feedback_edge
                            .properties
                            .insert("outcome".into(), json!(outcome));
                    }
                    graph.add_edge(feedback_edge);

                    drop(inference);

                    // Embed the feedback node with rich text for semantic search
                    let embed_text = format!(
                        "workflow feedback for {}: {}{}",
                        wf_name,
                        request.feedback,
                        request
                            .outcome
                            .as_ref()
                            .map(|o| format!(". outcome: {}", o))
                            .unwrap_or_default()
                    );
                    engine.embed_nodes_async(vec![(feedback_nid, embed_text)]);

                    Ok(json!({
                        "success": true,
                        "workflow_id": id,
                        "feedback_node_id": feedback_nid,
                    }))
                })
            }),
            result_tx: tx,
        })
        .await;

    match result {
        Ok(v) => Ok(Json(v)),
        Err(WriteError::OperationFailed(e)) => Err(ApiError::BadRequest(e)),
        Err(e) => Err(ApiError::from(e)),
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Supersede all valid edges for a node (both incoming and outgoing).
/// Returns the number of edges superseded.
fn supersede_node_edges(graph: &mut agent_db_graph::Graph, node_id: u64, _timestamp: u64) -> usize {
    let mut count = 0;

    // Collect edge IDs to supersede (outgoing + incoming)
    let out_eids: Vec<u64> = graph
        .get_edges_from(node_id)
        .iter()
        .filter(|e| e.is_valid())
        .map(|e| e.id)
        .collect();
    let in_eids: Vec<u64> = graph
        .get_edges_to(node_id)
        .iter()
        .filter(|e| e.is_valid())
        .map(|e| e.id)
        .collect();

    for eid in out_eids.into_iter().chain(in_eids) {
        graph.invalidate_edge(eid, "workflow_deleted");
        count += 1;
    }

    count
}

/// Ensure a concept node exists for a state value (e.g., "pending", "running", "completed").
/// Returns the node ID, creating if needed. Same pattern as `ensure_concept` in pipeline.rs.
fn ensure_concept_node(graph: &mut agent_db_graph::Graph, name: &str) -> u64 {
    // Check if concept already exists
    if let Some(nid) = graph.find_concept_node(name) {
        return nid;
    }
    // Create new concept node
    let node = GraphNode::new(NodeType::Concept {
        concept_name: name.to_string(),
        concept_type: ConceptType::NamedEntity,
        confidence: 1.0,
    });
    match graph.add_node(node) {
        Ok(nid) => nid,
        Err(e) => {
            tracing::error!("Failed to create concept node '{name}': {e}");
            0
        }
    }
}

/// Resolve the current step state by walking `workflow:step_state` edges.
/// Returns the concept name of the target node of the active (valid_until = None) edge
/// with the highest valid_from — same temporal resolution as graph_projection.
fn resolve_step_state(graph: &agent_db_graph::Graph, step_nid: u64) -> Option<String> {
    let mut best: Option<(u64, u64)> = None; // (valid_from, target_nid)

    for edge in graph.get_edges_from(step_nid) {
        if !edge.is_valid() {
            continue;
        }
        if let EdgeType::Association {
            ref association_type,
            ..
        } = edge.edge_type
        {
            if association_type == "workflow:step_state" {
                let vf = edge.valid_from.unwrap_or(0);
                if best.is_none() || vf > best.unwrap().0 {
                    best = Some((vf, edge.target));
                }
            }
        }
    }

    best.and_then(|(_, target_nid)| {
        graph.get_node(target_nid).and_then(|node| {
            if let NodeType::Concept {
                ref concept_name, ..
            } = node.node_type
            {
                Some(concept_name.clone())
            } else {
                None
            }
        })
    })
}

fn timestamp_now_u64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn format_timestamp(ts: u64) -> String {
    let secs = (ts / 1_000_000_000) as i64;
    chrono::DateTime::from_timestamp(secs, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| ts.to_string())
}

fn hash_string_key(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}
