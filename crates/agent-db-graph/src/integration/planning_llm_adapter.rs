//! Concrete LLM client implementations for the planning engine.
//!
//! Adapts external LLM APIs (OpenAI, Anthropic) to the `PlanningLlmClient` trait
//! defined in `agent-db-planning`. These live in `agent-db-graph` because the
//! planning crate intentionally has no networking dependencies.

use agent_db_planning::llm_client::{CompletionRequest, CompletionResponse, PlanningLlmClient};
use agent_db_planning::PlanningError;

/// OpenAI-compatible planning LLM client.
///
/// Uses the Chat Completions API with `response_format: json_object`.
pub struct OpenAiPlanningClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiPlanningClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl PlanningLlmClient for OpenAiPlanningClient {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, PlanningError> {
        let model = request.model.as_deref().unwrap_or(&self.model);

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "response_format": {"type": "json_object"}
            }))
            .send()
            .await
            .map_err(|e| PlanningError::LlmError(format!("OpenAI request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(PlanningError::LlmError(format!(
                "OpenAI API error {}: {}",
                status, body
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| PlanningError::LlmError(format!("failed to parse response: {}", e)))?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                PlanningError::LlmError(format!(
                    "OpenAI response missing content field: {}",
                    serde_json::to_string(&json).unwrap_or_default()
                ))
            })?
            .to_string();

        let tokens_used = json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(CompletionResponse {
            content,
            tokens_used,
            model: model.to_string(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Anthropic Messages API planning LLM client.
pub struct AnthropicPlanningClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicPlanningClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl PlanningLlmClient for AnthropicPlanningClient {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, PlanningError> {
        let model = request.model.as_deref().unwrap_or(&self.model);

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": model,
                "max_tokens": request.max_tokens,
                "system": request.system_prompt,
                "messages": [
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
            }))
            .send()
            .await
            .map_err(|e| PlanningError::LlmError(format!("Anthropic request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(PlanningError::LlmError(format!(
                "Anthropic API error {}: {}",
                status, body
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| PlanningError::LlmError(format!("failed to parse response: {}", e)))?;

        // Anthropic returns content as an array of content blocks
        let content = json["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|b| {
                    if b["type"].as_str() == Some("text") {
                        b["text"].as_str().map(|s| s.to_string())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default();

        let tokens_used = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32
            + json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(CompletionResponse {
            content,
            tokens_used,
            model: model.to_string(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
