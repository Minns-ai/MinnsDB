//! Shared types mirroring server models (client-side, Deserialize-focused).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Request Types (sent to server)
// ============================================================================

#[derive(Debug, Serialize)]
pub struct ConversationIngestRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    pub sessions: Vec<SessionInput>,
    pub include_assistant_facts: bool,
    pub group_id: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct SessionInput {
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    pub messages: Vec<MessageInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contains_fact: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_quote: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct NlqRequest {
    pub question: String,
    pub group_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub include_context: bool,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SimpleEventRequest {
    pub agent_id: u64,
    pub agent_type: String,
    pub session_id: u64,
    pub action: String,
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    #[serde(default)]
    pub enable_semantic: bool,
}

// ============================================================================
// Response Types (received from server)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct NlqResponse {
    pub answer: String,
    pub intent: String,
    pub entities_resolved: Vec<NlqEntity>,
    pub confidence: f32,
    pub result_count: usize,
    pub execution_time_ms: u64,
    pub query_used: String,
    pub explanation: Vec<String>,
    pub total_count: usize,
    #[serde(default)]
    pub related_memories: Vec<serde_json::Value>,
    #[serde(default)]
    pub related_strategies: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct NlqEntity {
    pub text: String,
    pub node_id: u64,
    pub node_type: String,
    pub confidence: f32,
}

#[derive(Debug, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub mode: String,
    pub total: usize,
}

#[derive(Debug, Deserialize)]
pub struct SearchResultItem {
    pub node_id: u64,
    pub score: f32,
    pub node_type: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct StatsResponse {
    pub total_events_processed: u64,
    pub total_nodes_created: u64,
    pub total_episodes_detected: u64,
    pub total_memories_formed: u64,
    pub total_strategies_extracted: u64,
    pub total_reinforcements_applied: u64,
    pub average_processing_time_ms: f64,
    #[serde(default)]
    pub stores: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub is_healthy: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub processing_rate: f64,
}

#[derive(Debug, Deserialize)]
pub struct ProcessEventResponse {
    pub success: bool,
    pub event_id: String,
    pub nodes_created: usize,
    pub patterns_detected: usize,
    pub processing_time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct StrategyResponse {
    pub id: u64,
    pub name: String,
    pub summary: String,
    pub when_to_use: String,
    pub when_not_to_use: String,
    pub quality_score: f32,
    pub confidence: f32,
    #[serde(default)]
    pub playbook: Vec<PlaybookStep>,
    #[serde(default)]
    pub applicable_domains: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct PlaybookStep {
    pub step: u32,
    pub action: String,
    #[serde(default)]
    pub condition: String,
}

#[derive(Debug, Deserialize)]
pub struct SimilarStrategyResponse {
    pub score: f32,
    pub id: u64,
    pub name: String,
    pub summary: String,
    pub when_to_use: String,
    pub quality_score: f32,
}

#[derive(Debug, Deserialize)]
pub struct CausalPathResponse {
    pub source: u64,
    pub target: u64,
    pub found: bool,
    pub path: Option<Vec<u64>>,
}

#[derive(Debug, Deserialize)]
pub struct CommunitiesResponse {
    pub communities: Vec<CommunityResponse>,
    pub modularity: f32,
    pub community_count: usize,
    pub algorithm: String,
}

#[derive(Debug, Deserialize)]
pub struct CommunityResponse {
    pub community_id: u64,
    pub node_ids: Vec<u64>,
    pub size: usize,
}

#[derive(Debug, Deserialize)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNodeResponse>,
    pub edges: Vec<GraphEdgeResponse>,
}

#[derive(Debug, Deserialize)]
pub struct GraphNodeResponse {
    pub id: u64,
    pub label: String,
    pub node_type: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct GraphEdgeResponse {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub weight: f32,
    pub confidence: f32,
}

#[derive(Debug, Deserialize)]
pub struct MemoryResponse {
    pub id: u64,
    pub summary: String,
    pub takeaway: String,
    pub tier: String,
    pub strength: f32,
    pub relevance_score: f32,
    pub memory_type: String,
    pub outcome: String,
}
