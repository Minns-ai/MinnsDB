//! HTTP client for EventGraphDB server.

use anyhow::{Context, Result};
use reqwest::Client;

use crate::types::*;

/// EventGraphDB API client.
pub struct EventGraphClient {
    base_url: String,
    group_id: String,
    client: Client,
}

impl EventGraphClient {
    pub fn new(base_url: String, group_id: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;
        Ok(Self {
            base_url,
            group_id,
            client,
        })
    }

    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    // ========================================================================
    // Health & Status
    // ========================================================================

    pub async fn health(&self) -> Result<HealthResponse> {
        let resp = self
            .client
            .get(format!("{}/api/health", self.base_url))
            .send()
            .await
            .context("Failed to connect to EventGraphDB")?;
        self.handle_response(resp).await
    }

    pub async fn stats(&self) -> Result<StatsResponse> {
        let resp = self
            .client
            .get(format!("{}/api/stats", self.base_url))
            .send()
            .await
            .context("Failed to connect to EventGraphDB")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // NLQ (Natural Language Query)
    // ========================================================================

    pub async fn nlq(
        &self,
        question: &str,
        include_context: bool,
        limit: Option<usize>,
    ) -> Result<NlqResponse> {
        let req = NlqRequest {
            question: question.to_string(),
            group_id: self.group_id.clone(),
            limit,
            offset: None,
            session_id: None,
            include_context,
            metadata: Default::default(),
        };
        let resp = self
            .client
            .post(format!("{}/api/nlq", self.base_url))
            .json(&req)
            .send()
            .await
            .context("NLQ request failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Search
    // ========================================================================

    pub async fn search(
        &self,
        query: &str,
        mode: Option<String>,
        limit: Option<usize>,
    ) -> Result<SearchResponse> {
        let req = SearchRequest {
            query: query.to_string(),
            mode,
            limit,
        };
        let resp = self
            .client
            .post(format!("{}/api/search", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Search request failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Conversation Ingest (used by learn command)
    // ========================================================================

    pub async fn ingest_conversation(
        &self,
        request: ConversationIngestRequest,
    ) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}/api/conversations/ingest", self.base_url))
            .json(&request)
            .send()
            .await
            .context("Conversation ingest failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Events (simple)
    // ========================================================================

    pub async fn submit_simple_event(
        &self,
        request: SimpleEventRequest,
    ) -> Result<ProcessEventResponse> {
        let resp = self
            .client
            .post(format!("{}/api/events/simple", self.base_url))
            .json(&request)
            .send()
            .await
            .context("Simple event submission failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Code Intelligence
    // ========================================================================

    pub async fn submit_code_file(
        &self,
        file_path: &str,
        content: &str,
        language: Option<&str>,
        repository: Option<&str>,
    ) -> Result<ProcessEventResponse> {
        let mut req = serde_json::json!({
            "agent_id": 1,
            "agent_type": "cli_scanner",
            "session_id": 1,
            "file_path": file_path,
            "content": content,
            "enable_ast": true,
            "enable_semantic": false,
        });
        if let Some(lang) = language {
            req.as_object_mut()
                .unwrap()
                .insert("language".into(), serde_json::json!(lang));
        }
        if let Some(repo) = repository {
            req.as_object_mut()
                .unwrap()
                .insert("repository".into(), serde_json::json!(repo));
        }
        let resp = self
            .client
            .post(format!("{}/api/events/code-file", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Code file submission failed")?;
        self.handle_response(resp).await
    }

    pub async fn search_code(
        &self,
        name_pattern: Option<&str>,
        kind: Option<&str>,
        language: Option<&str>,
        file_pattern: Option<&str>,
        limit: usize,
    ) -> Result<serde_json::Value> {
        let req = serde_json::json!({
            "name_pattern": name_pattern,
            "kind": kind,
            "language": language,
            "file_pattern": file_pattern,
            "limit": limit,
        });
        let resp = self
            .client
            .post(format!("{}/api/code/search", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Code search failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Strategies
    // ========================================================================

    pub async fn find_strategies(&self, query: &str) -> Result<Vec<SimilarStrategyResponse>> {
        let req = serde_json::json!({
            "context_hash": query,
            "limit": 10
        });
        let resp = self
            .client
            .post(format!("{}/api/strategies/similar", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Strategy search failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Graph Algorithms
    // ========================================================================

    pub async fn neighbors(&self, node_id: u64) -> Result<GraphResponse> {
        let resp = self
            .client
            .get(format!("{}/api/graph/context", self.base_url))
            .query(&[
                ("context_hash", &node_id.to_string()),
                ("limit", &"50".to_string()),
            ])
            .send()
            .await
            .context("Graph context query failed")?;
        self.handle_response(resp).await
    }

    pub async fn causal_path(&self, source: u64, target: u64) -> Result<CausalPathResponse> {
        let resp = self
            .client
            .get(format!("{}/api/causal-path", self.base_url))
            .query(&[
                ("source", &source.to_string()),
                ("target", &target.to_string()),
            ])
            .send()
            .await
            .context("Causal path query failed")?;
        self.handle_response(resp).await
    }

    pub async fn communities(&self) -> Result<CommunitiesResponse> {
        let resp = self
            .client
            .get(format!("{}/api/communities", self.base_url))
            .send()
            .await
            .context("Communities query failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Workflows
    // ========================================================================

    pub async fn create_workflow(
        &self,
        definition: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}/api/workflows", self.base_url))
            .json(&definition)
            .send()
            .await
            .context("Workflow creation failed")?;
        self.handle_response(resp).await
    }

    pub async fn list_workflows(&self) -> Result<serde_json::Value> {
        let resp = self
            .client
            .get(format!("{}/api/workflows", self.base_url))
            .send()
            .await
            .context("Workflow list failed")?;
        self.handle_response(resp).await
    }

    pub async fn get_workflow(&self, id: u64) -> Result<serde_json::Value> {
        let resp = self
            .client
            .get(format!("{}/api/workflows/{}", self.base_url, id))
            .send()
            .await
            .context("Workflow get failed")?;
        self.handle_response(resp).await
    }

    pub async fn update_workflow(
        &self,
        id: u64,
        definition: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let resp = self
            .client
            .put(format!("{}/api/workflows/{}", self.base_url, id))
            .json(&definition)
            .send()
            .await
            .context("Workflow update failed")?;
        self.handle_response(resp).await
    }

    pub async fn delete_workflow(&self, id: u64) -> Result<serde_json::Value> {
        let resp = self
            .client
            .delete(format!("{}/api/workflows/{}", self.base_url, id))
            .send()
            .await
            .context("Workflow delete failed")?;
        self.handle_response(resp).await
    }

    pub async fn workflow_step_transition(
        &self,
        workflow_id: u64,
        step_id: &str,
        state: &str,
        result: Option<&str>,
    ) -> Result<serde_json::Value> {
        let mut req = serde_json::json!({ "state": state });
        if let Some(r) = result {
            req.as_object_mut()
                .unwrap()
                .insert("result".into(), serde_json::json!(r));
        }
        let resp = self
            .client
            .post(format!(
                "{}/api/workflows/{}/steps/{}/transition",
                self.base_url, workflow_id, step_id
            ))
            .json(&req)
            .send()
            .await
            .context("Step transition failed")?;
        self.handle_response(resp).await
    }

    pub async fn workflow_feedback(
        &self,
        workflow_id: u64,
        feedback: &str,
        outcome: Option<&str>,
    ) -> Result<serde_json::Value> {
        let mut req = serde_json::json!({ "feedback": feedback });
        if let Some(o) = outcome {
            req.as_object_mut()
                .unwrap()
                .insert("outcome".into(), serde_json::json!(o));
        }
        let resp = self
            .client
            .post(format!(
                "{}/api/workflows/{}/feedback",
                self.base_url, workflow_id
            ))
            .json(&req)
            .send()
            .await
            .context("Workflow feedback failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Agent Registry
    // ========================================================================

    pub async fn register_agent(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let resp = self
            .client
            .post(format!("{}/api/agents/register", self.base_url))
            .json(&request)
            .send()
            .await
            .context("Agent registration failed")?;
        self.handle_response(resp).await
    }

    pub async fn list_agents(&self) -> Result<serde_json::Value> {
        let resp = self
            .client
            .get(format!("{}/api/agents", self.base_url))
            .query(&[("group_id", &self.group_id)])
            .send()
            .await
            .context("Agent list failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Planning
    // ========================================================================

    pub async fn plan(&self, goal: &str) -> Result<serde_json::Value> {
        let req = serde_json::json!({ "goal": goal });
        let resp = self
            .client
            .post(format!("{}/api/planning/plan", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Planning request failed")?;
        self.handle_response(resp).await
    }

    // ========================================================================
    // Internal
    // ========================================================================

    async fn handle_response<T: serde::de::DeserializeOwned>(
        &self,
        resp: reqwest::Response,
    ) -> Result<T> {
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Server returned {}: {}", status, body);
        }
        resp.json::<T>()
            .await
            .context("Failed to parse server response")
    }
}
