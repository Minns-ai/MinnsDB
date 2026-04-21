// Request and Response Models for MinnsDB REST API

use agent_db_core::types::{AgentId, AgentType, ContextHash, EventId, SessionId};
use agent_db_events::core::EventContext;
use agent_db_events::Event;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr, PickFirst};

// ============================================================================
// Request Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessEventRequest {
    pub event: Event,
    /// Enable semantic memory processing (NER + claim extraction + embeddings)
    #[serde(default)]
    pub enable_semantic: bool,
    /// Partition key for multi-tenant isolation.
    /// All nodes/edges created from this event are tagged with this group_id.
    #[serde(default)]
    pub group_id: String,
}

/// Simplified event request for easy integration
/// Only requires the absolute minimum fields
#[serde_as]
#[derive(Debug, Serialize, Deserialize)]
pub struct SimpleEventRequest {
    /// Agent identifier
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub agent_id: AgentId,
    /// Agent type (e.g., "chatbot", "assistant")
    pub agent_type: AgentType,
    /// Session identifier
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub session_id: SessionId,
    /// Action name or event type
    pub action: String,
    /// Event data/payload (can be any JSON)
    pub data: serde_json::Value,
    /// Optional: outcome status (defaults to Success)
    #[serde(default)]
    pub success: Option<bool>,
    /// Optional: enable semantic processing
    #[serde(default)]
    pub enable_semantic: bool,
    /// Partition key for multi-tenant isolation.
    /// All nodes/edges created from this event are tagged with this group_id.
    #[serde(default)]
    pub group_id: String,
}

#[derive(Debug, Deserialize)]
pub struct PaginationQuery {
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct ActionSuggestionsQuery {
    pub context_hash: ContextHash,
    #[serde(default)]
    pub last_action_node: Option<u64>,
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct GraphQuery {
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
    #[serde(default)]
    pub session_id: Option<SessionId>,
    #[serde(default)]
    pub agent_type: Option<AgentType>,
}

#[derive(Debug, Deserialize)]
pub struct GraphContextQuery {
    pub context_hash: ContextHash,
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
    #[serde(default)]
    pub session_id: Option<SessionId>,
    #[serde(default)]
    pub agent_type: Option<AgentType>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
pub struct ContextMemoriesRequest {
    pub context: EventContext,
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
    #[serde(default)]
    #[serde_as(as = "Option<PickFirst<(_, DisplayFromStr)>>")]
    pub min_similarity: Option<f32>,
    #[serde(default)]
    #[serde_as(as = "Option<PickFirst<(_, DisplayFromStr)>>")]
    pub agent_id: Option<AgentId>,
    #[serde(default)]
    #[serde_as(as = "Option<PickFirst<(_, DisplayFromStr)>>")]
    pub session_id: Option<SessionId>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
pub struct StrategySimilarityRequest {
    #[serde(default)]
    #[serde_as(as = "Vec<PickFirst<(_, DisplayFromStr)>>")]
    pub goal_ids: Vec<u64>,
    #[serde(default)]
    pub tool_names: Vec<String>,
    #[serde(default)]
    pub result_types: Vec<String>,
    #[serde(default)]
    #[serde_as(as = "Option<PickFirst<(_, DisplayFromStr)>>")]
    pub context_hash: Option<ContextHash>,
    #[serde(default)]
    #[serde_as(as = "Option<PickFirst<(_, DisplayFromStr)>>")]
    pub agent_id: Option<AgentId>,
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
    #[serde(default)]
    #[serde_as(as = "Option<PickFirst<(_, DisplayFromStr)>>")]
    pub min_score: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimSearchRequest {
    pub query_text: String,
    #[serde(
        default = "default_top_k",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub top_k: usize,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
}

#[serde_as]
#[derive(Debug, Deserialize)]
pub struct ClaimListQuery {
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
    #[serde(default)]
    #[serde_as(as = "Option<DisplayFromStr>")]
    pub event_id: Option<u128>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Search query text
    pub query: String,
    /// Search mode: keyword, semantic, or hybrid
    #[serde(default)]
    pub mode: agent_db_graph::indexing::SearchMode,
    /// Maximum number of results
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
    /// Fusion strategy for hybrid search
    #[serde(default)]
    pub fusion_strategy: Option<agent_db_graph::indexing::FusionStrategy>,
}

// ============================================================================
// Response Types
// ============================================================================

#[serde_as]
#[derive(Debug, Serialize)]
pub struct ProcessEventResponse {
    pub success: bool,
    #[serde_as(as = "DisplayFromStr")]
    pub event_id: EventId,
    pub nodes_created: usize,
    pub patterns_detected: usize,
    pub processing_time_ms: u64,
}

#[serde_as]
#[derive(Debug, Serialize)]
pub struct MemoryResponse {
    pub id: u64,
    pub agent_id: AgentId,
    pub session_id: SessionId,
    // ========== LLM-Retrievable Fields ==========
    pub summary: String,
    pub takeaway: String,
    pub causal_note: String,
    pub tier: String,
    pub consolidation_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_id: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub consolidated_from: Vec<u64>,
    // ========== Core Fields ==========
    pub strength: f32,
    pub relevance_score: f32,
    pub access_count: u32,
    #[serde_as(as = "DisplayFromStr")]
    pub formed_at: u64,
    #[serde_as(as = "DisplayFromStr")]
    pub last_accessed: u64,
    pub context_hash: ContextHash,
    pub context: EventContext,
    pub outcome: String,
    pub memory_type: String,
}

#[derive(Debug, Serialize)]
pub struct StrategyResponse {
    pub id: u64,
    pub name: String,
    pub agent_id: AgentId,
    // ========== LLM-Retrievable Fields ==========
    pub summary: String,
    pub when_to_use: String,
    pub when_not_to_use: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub failure_modes: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub playbook: Vec<PlaybookStepResponse>,
    pub counterfactual: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub applicable_domains: Vec<String>,
    // ========== Core Fields ==========
    pub quality_score: f32,
    pub success_count: u32,
    pub failure_count: u32,
    pub reasoning_steps: Vec<ReasoningStepResponse>,
    pub strategy_type: String,
    pub support_count: u32,
    pub expected_success: f32,
    pub expected_cost: f32,
    pub expected_value: f32,
    pub confidence: f32,
    pub goal_bucket_id: u64,
    pub behavior_signature: String,
    pub precondition: String,
    pub action_hint: String,
}

#[derive(Debug, Serialize)]
pub struct SimilarStrategyResponse {
    pub score: f32,
    pub id: u64,
    pub name: String,
    pub agent_id: AgentId,
    // ========== LLM-Retrievable Fields ==========
    pub summary: String,
    pub when_to_use: String,
    pub when_not_to_use: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub failure_modes: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub playbook: Vec<PlaybookStepResponse>,
    pub counterfactual: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub applicable_domains: Vec<String>,
    // ========== Core Fields ==========
    pub quality_score: f32,
    pub success_count: u32,
    pub failure_count: u32,
    pub reasoning_steps: Vec<ReasoningStepResponse>,
    pub strategy_type: String,
    pub support_count: u32,
    pub expected_success: f32,
    pub expected_cost: f32,
    pub expected_value: f32,
    pub confidence: f32,
    pub goal_bucket_id: u64,
    pub behavior_signature: String,
    pub precondition: String,
    pub action_hint: String,
}

#[derive(Debug, Serialize)]
pub struct ReasoningStepResponse {
    pub description: String,
    pub sequence_order: usize,
}

#[derive(Debug, Serialize)]
pub struct PlaybookStepResponse {
    pub step: u32,
    pub action: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub condition: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub skip_if: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub recovery: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<PlaybookBranchResponse>,
}

#[derive(Debug, Serialize)]
pub struct PlaybookBranchResponse {
    pub condition: String,
    pub action: String,
}

#[derive(Debug, Serialize)]
pub struct ActionSuggestionResponse {
    pub action_name: String,
    pub success_probability: f32,
    pub evidence_count: u32,
    pub reasoning: String,
}

#[derive(Debug, Serialize)]
pub struct EpisodeResponse {
    pub id: u64,
    pub agent_id: AgentId,
    pub event_count: usize,
    pub significance: f32,
    pub outcome: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_events_processed: u64,
    pub total_nodes_created: u64,
    pub total_episodes_detected: u64,
    pub total_memories_formed: u64,
    pub total_strategies_extracted: u64,
    pub total_reinforcements_applied: u64,
    pub average_processing_time_ms: f64,
    /// Live aggregate counts from all stores
    pub stores: agent_db_graph::StoreMetrics,
}

#[derive(Debug, Serialize)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNodeResponse>,
    pub edges: Vec<GraphEdgeResponse>,
}

#[derive(Debug, Serialize)]
pub struct GraphPersistResponse {
    pub success: bool,
    pub nodes_persisted: usize,
    pub edges_persisted: usize,
}

#[serde_as]
#[derive(Debug, Serialize)]
pub struct GraphNodeResponse {
    pub id: u64,
    pub label: String,
    pub node_type: String,
    #[serde_as(as = "DisplayFromStr")]
    pub created_at: u64,
    pub properties: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct GraphEdgeResponse {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub weight: f32,
    pub confidence: f32,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub is_healthy: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub processing_rate: f64,
    pub write_lanes: WriteLanesHealth,
    pub read_gate: ReadGateHealth,
    pub sequence_tracker: SequenceTrackerHealth,
}

#[derive(Debug, Serialize)]
pub struct WriteLanesHealth {
    pub num_lanes: usize,
    pub lanes: Vec<LaneHealth>,
    pub total_submitted: u64,
    pub total_completed: u64,
    pub total_rejected: u64,
    pub write_p50_ms: f64,
    pub write_p95_ms: f64,
    pub write_p99_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct LaneHealth {
    pub lane_id: usize,
    pub depth: u64,
    pub in_flight: u64,
    pub completed: u64,
    pub rejected: u64,
}

#[derive(Debug, Serialize)]
pub struct ReadGateHealth {
    pub permits_total: usize,
    pub in_flight: u64,
    pub completed: u64,
    pub rejected: u64,
    pub read_p50_ms: f64,
    pub read_p95_ms: f64,
    pub read_p99_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct SequenceTrackerHealth {
    pub tracked_domains: usize,
}

#[derive(Debug, Serialize)]
pub struct AnalyticsResponse {
    pub node_count: usize,
    pub edge_count: usize,
    pub connected_components: usize,
    pub largest_component_size: usize,
    pub average_path_length: f32,
    pub diameter: u32,
    pub clustering_coefficient: f32,
    pub average_clustering: f32,
    pub modularity: f32,
    pub community_count: usize,
    pub learning_metrics: LearningMetricsResponse,
}

#[derive(Debug, Serialize)]
pub struct LearningMetricsResponse {
    pub total_events: usize,
    pub unique_contexts: usize,
    pub learned_patterns: usize,
    pub strong_memories: usize,
    pub overall_success_rate: f32,
    pub average_edge_weight: f32,
}

#[serde_as]
#[derive(Debug, Serialize)]
pub struct IndexStatsResponse {
    pub insert_count: u64,
    pub query_count: u64,
    pub range_query_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    #[serde_as(as = "DisplayFromStr")]
    pub last_accessed: u64,
}

#[derive(Debug, Serialize)]
pub struct CommunityResponse {
    pub community_id: u64,
    pub node_ids: Vec<u64>,
    pub size: usize,
}

#[derive(Debug, Serialize)]
pub struct CommunitiesResponse {
    pub communities: Vec<CommunityResponse>,
    pub modularity: f32,
    pub iterations: usize,
    pub community_count: usize,
    pub algorithm: String,
}

#[derive(Debug, Serialize)]
pub struct CentralityScoresResponse {
    pub node_id: u64,
    pub degree: f32,
    pub betweenness: f32,
    pub closeness: f32,
    pub eigenvector: f32,
    pub pagerank: f32,
    pub combined: f32,
}

// ============================================================================
// Graph Algorithm Query Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct PprQuery {
    pub source_node_id: u64,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ReachabilityQuery {
    pub source: u64,
    pub max_hops: Option<usize>,
    pub max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CausalPathQuery {
    pub source: u64,
    pub target: u64,
}

#[derive(Debug, Deserialize)]
pub struct CommunitiesQuery {
    pub algorithm: Option<String>,
}

// ============================================================================
// Graph Algorithm Response Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct PprResponse {
    pub source_node_id: u64,
    pub algorithm: String,
    pub scores: Vec<PprNodeScore>,
}

#[derive(Debug, Serialize)]
pub struct PprNodeScore {
    pub node_id: u64,
    pub score: f64,
}

#[serde_as]
#[derive(Debug, Serialize)]
pub struct ReachabilityResponse {
    pub source_node_id: u64,
    pub reachable_count: usize,
    pub max_depth: usize,
    pub edges_traversed: usize,
    pub reachable: Vec<ReachabilityNodeResponse>,
}

#[serde_as]
#[derive(Debug, Serialize)]
pub struct ReachabilityNodeResponse {
    pub node_id: u64,
    pub origin: u64,
    #[serde_as(as = "DisplayFromStr")]
    pub arrival_time: u64,
    pub hops: usize,
    pub predecessor: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct CausalPathResponse {
    pub source: u64,
    pub target: u64,
    pub found: bool,
    pub path: Option<Vec<u64>>,
}

#[serde_as]
#[derive(Debug, Serialize)]
pub struct ClaimResponse {
    pub claim_id: u64,
    pub claim_text: String,
    pub confidence: f32,
    #[serde_as(as = "DisplayFromStr")]
    pub source_event_id: u128,
    pub similarity: Option<f32>,
    pub evidence_spans: Vec<EvidenceSpanResponse>,
    pub support_count: u32,
    pub status: String,
    #[serde_as(as = "DisplayFromStr")]
    pub created_at: u64,
    #[serde_as(as = "DisplayFromStr")]
    pub last_accessed: u64,
    /// Claim type: Preference, Fact, Belief, Intention, Capability
    pub claim_type: String,
    /// Normalized primary entity this claim is about
    pub subject_entity: Option<String>,
    /// Explicit expiry timestamp (epoch seconds), if set
    pub expires_at: Option<u64>,
    /// Temporal freshness weight [0.0, 1.0]
    pub temporal_weight: f32,
    /// If superseded, the ID of the replacing claim
    pub superseded_by: Option<u64>,
    /// NER entities attached to this claim (text + label)
    pub entities: Vec<ClaimEntityResponse>,
}

/// A single NER entity attached to a claim.
#[derive(Debug, Serialize)]
pub struct ClaimEntityResponse {
    /// Entity text as it appears in the source
    pub text: String,
    /// NER label (PERSON, ORG, LOC, PRODUCT, DATE, EVENT, …)
    pub label: String,
}

#[derive(Debug, Serialize)]
pub struct ClaimGroup {
    pub subject: String,
    pub claims: Vec<ClaimResponse>,
}

#[derive(Debug, Serialize)]
pub struct GroupedClaimSearchResponse {
    pub groups: Vec<ClaimGroup>,
    pub ungrouped: Vec<ClaimResponse>,
    pub total_results: usize,
}

#[derive(Debug, Serialize)]
pub struct EvidenceSpanResponse {
    pub start_offset: usize,
    pub end_offset: usize,
    pub text_snippet: String,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingProcessResponse {
    pub claims_processed: usize,
    pub success: bool,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Ranked search results with scores
    pub results: Vec<SearchResultItem>,
    /// Search mode used
    pub mode: String,
    /// Total results found
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    /// Node ID
    pub node_id: u64,
    /// Relevance score
    pub score: f32,
    /// Node type
    pub node_type: String,
    /// Node properties (excerpt)
    pub properties: serde_json::Value,
}

// ============================================================================
// Default Functions
// ============================================================================

fn default_limit() -> usize {
    10
}

fn default_top_k() -> usize {
    10
}

fn default_min_similarity() -> f32 {
    0.7
}

/// Maximum allowed value for user-supplied `limit` / `top_k` parameters.
/// Prevents DoS via absurdly large result sets.
const MAX_LIMIT: usize = 1_000;

/// Deserializer that caps a `usize` limit to `MAX_LIMIT`.
fn deserialize_capped_limit<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = usize::deserialize(deserializer)?;
    Ok(value.min(MAX_LIMIT))
}

// ============================================================================
// Natural Language Query
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct NlqRequest {
    pub question: String,
    /// Partition key for multi-tenant scoping.
    /// When set, only nodes/edges with this group_id are searched.
    #[serde(default)]
    pub group_id: String,
    /// Optional pagination limit.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Optional pagination offset.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Optional session ID for conversational context.
    #[serde(default)]
    pub session_id: Option<String>,
    /// If true, include related memories and strategies in the response.
    #[serde(default)]
    pub include_context: bool,
    /// Arbitrary metadata for filtering/scoping queries.
    /// E.g., `{"user_id": "19039485485"}` to scope results to a specific user.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub federated_sources: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct NlqResponseBody {
    pub answer: String,
    pub intent: String,
    pub entities_resolved: Vec<NlqEntity>,
    pub confidence: f32,
    pub result_count: usize,
    pub execution_time_ms: u64,
    pub query_used: String,
    /// Step-by-step pipeline explanation.
    pub explanation: Vec<String>,
    /// Total result count before pagination.
    pub total_count: usize,
    /// Related memories (only populated when `include_context` is true).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub related_memories: Vec<agent_db_graph::MemorySummary>,
    /// Related strategies (only populated when `include_context` is true).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub related_strategies: Vec<agent_db_graph::StrategySummary>,
}

#[derive(Debug, Serialize)]
pub struct NlqEntity {
    pub text: String,
    pub node_id: u64,
    pub node_type: String,
    pub confidence: f32,
}

// ============================================================================
// Structured Memory
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct StructuredMemoryRequest {
    pub key: String,
    pub template: agent_db_graph::MemoryTemplate,
}

#[derive(Debug, Deserialize)]
pub struct StructuredMemoryKeyQuery {
    #[serde(default)]
    pub prefix: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LedgerAppendRequest {
    pub amount: f64,
    pub description: String,
    pub direction: agent_db_graph::LedgerDirection,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StateTransitionRequest {
    pub new_state: String,
    pub trigger: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PreferenceUpdateRequest {
    pub item: String,
    pub rank: usize,
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TreeAddChildRequest {
    pub parent: String,
    pub child: String,
}

// ============================================================================
// Code Intelligence Types
// ============================================================================

/// POST /api/events/code-review — code review event submission
#[serde_as]
#[derive(Debug, Deserialize)]
pub struct CodeReviewRequest {
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub agent_id: AgentId,
    pub agent_type: AgentType,
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub session_id: SessionId,
    pub review_id: String,
    /// "comment", "approve", "request_changes"
    pub action: String,
    pub body: String,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub line_range: Option<(usize, usize)>,
    pub repository: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub enable_semantic: bool,
}

/// POST /api/events/code-file — code file snapshot submission
#[serde_as]
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct CodeFileRequest {
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub agent_id: AgentId,
    pub agent_type: AgentType,
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub session_id: SessionId,
    pub file_path: String,
    pub content: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub repository: Option<String>,
    #[serde(default)]
    pub git_ref: Option<String>,
    #[serde(default)]
    pub enable_ast: bool,
    #[serde(default)]
    pub enable_semantic: bool,
}

/// POST /api/code/search — structural code search
#[derive(Debug, Deserialize)]
pub struct CodeSearchRequest {
    /// Glob pattern for entity name matching
    #[serde(default)]
    pub name_pattern: Option<String>,
    /// Filter by entity kind: "function", "struct", "enum", etc.
    #[serde(default)]
    pub kind: Option<String>,
    /// Filter by language
    #[serde(default)]
    pub language: Option<String>,
    /// Glob pattern for file path matching
    #[serde(default)]
    pub file_pattern: Option<String>,
    #[serde(
        default = "default_limit",
        deserialize_with = "deserialize_capped_limit"
    )]
    pub limit: usize,
}

/// Code search response
#[derive(Debug, Serialize)]
pub struct CodeSearchResponse {
    pub entities: Vec<CodeEntityResult>,
    pub total_matches: usize,
}

/// Single code entity result
#[derive(Debug, Serialize)]
pub struct CodeEntityResult {
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

// ============================================================================
// SDK Helper Event Types
// ============================================================================

/// POST /api/events/state-change — typed state-change event submission
#[serde_as]
#[derive(Debug, Deserialize)]
pub struct StateChangeEventRequest {
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub agent_id: AgentId,
    pub agent_type: AgentType,
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub session_id: SessionId,
    pub entity: String,
    pub new_state: String,
    #[serde(default)]
    pub old_state: Option<String>,
    #[serde(default)]
    pub trigger: Option<String>,
    #[serde(default)]
    pub extra_metadata: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub enable_semantic: bool,
    /// Partition key for multi-tenant isolation.
    #[serde(default)]
    pub group_id: String,
}

/// POST /api/events/transaction — typed transaction event submission
#[serde_as]
#[derive(Debug, Deserialize)]
pub struct TransactionEventRequest {
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub agent_id: AgentId,
    pub agent_type: AgentType,
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    pub session_id: SessionId,
    pub from: String,
    pub to: String,
    pub amount: f64,
    #[serde(default)]
    pub direction: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub extra_metadata: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub enable_semantic: bool,
    /// Partition key for multi-tenant isolation.
    #[serde(default)]
    pub group_id: String,
}
