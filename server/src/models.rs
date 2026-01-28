// Request and Response Models for EventGraphDB REST API

use agent_db_core::types::{AgentId, AgentType, ContextHash, SessionId};
use agent_db_events::core::EventContext;
use agent_db_events::Event;
use serde::{Deserialize, Serialize};

// ============================================================================
// Request Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessEventRequest {
    pub event: Event,
    /// Enable semantic memory processing (NER + claim extraction + embeddings)
    #[serde(default)]
    pub enable_semantic: bool,
}

#[derive(Debug, Deserialize)]
pub struct PaginationQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct ActionSuggestionsQuery {
    pub context_hash: ContextHash,
    #[serde(default)]
    pub last_action_node: Option<u64>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct GraphQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub session_id: Option<SessionId>,
    #[serde(default)]
    pub agent_type: Option<AgentType>,
}

#[derive(Debug, Deserialize)]
pub struct GraphContextQuery {
    pub context_hash: ContextHash,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub session_id: Option<SessionId>,
    #[serde(default)]
    pub agent_type: Option<AgentType>,
}

#[derive(Debug, Deserialize)]
pub struct ContextMemoriesRequest {
    pub context: EventContext,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub min_similarity: Option<f32>,
    #[serde(default)]
    pub agent_id: Option<AgentId>,
    #[serde(default)]
    pub session_id: Option<SessionId>,
}

#[derive(Debug, Deserialize)]
pub struct StrategySimilarityRequest {
    #[serde(default)]
    pub goal_ids: Vec<u64>,
    #[serde(default)]
    pub tool_names: Vec<String>,
    #[serde(default)]
    pub result_types: Vec<String>,
    #[serde(default)]
    pub context_hash: Option<ContextHash>,
    #[serde(default)]
    pub agent_id: Option<AgentId>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimSearchRequest {
    pub query_text: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
}

#[derive(Debug, Deserialize)]
pub struct ClaimListQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub event_id: Option<u128>,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct ProcessEventResponse {
    pub success: bool,
    pub nodes_created: usize,
    pub patterns_detected: usize,
    pub processing_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct MemoryResponse {
    pub id: u64,
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub strength: f32,
    pub relevance_score: f32,
    pub access_count: u32,
    pub formed_at: u64,
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
}

#[derive(Debug, Serialize)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNodeResponse>,
    pub edges: Vec<GraphEdgeResponse>,
}

#[derive(Debug, Serialize)]
pub struct GraphNodeResponse {
    pub id: u64,
    pub label: String,
    pub node_type: String,
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

#[derive(Debug, Serialize)]
pub struct IndexStatsResponse {
    pub insert_count: u64,
    pub query_count: u64,
    pub range_query_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
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

#[derive(Debug, Serialize)]
pub struct ClaimResponse {
    pub claim_id: u64,
    pub claim_text: String,
    pub confidence: f32,
    pub source_event_id: u128,
    pub similarity: Option<f32>,
    pub evidence_spans: Vec<EvidenceSpanResponse>,
    pub support_count: u32,
    pub status: String,
    pub created_at: u64,
    pub last_accessed: u64,
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
