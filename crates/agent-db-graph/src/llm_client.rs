//! Unified LLM client abstraction.
//!
//! Provides a single `LlmClient` trait with OpenAI and Anthropic
//! implementations. Replaces the per-feature LLM clients scattered across
//! `nlq::llm_hint`, `claims`, etc.

use async_trait::async_trait;

/// Request to send to an LLM.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub json_mode: bool,
}

/// Response from an LLM.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub tokens_used: u32,
}

/// Unified LLM client trait.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a completion request and return the response.
    async fn complete(&self, request: LlmRequest) -> anyhow::Result<LlmResponse>;

    /// The model name this client is configured with.
    fn model_name(&self) -> &str;
}

// ────────── OpenAI client ──────────

pub struct OpenAiLlmClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl OpenAiLlmClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiLlmClient {
    async fn complete(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        let mut body = serde_json::json!({
            "model": self.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": [
                { "role": "system", "content": request.system_prompt },
                { "role": "user", "content": request.user_prompt }
            ]
        });

        if request.json_mode {
            body["response_format"] = serde_json::json!({ "type": "json_object" });
        }

        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let tokens_used = json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LlmResponse {
            content,
            tokens_used,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ────────── Anthropic client ──────────

pub struct AnthropicLlmClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl AnthropicLlmClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmClient for AnthropicLlmClient {
    async fn complete(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system_prompt,
            "messages": [
                { "role": "user", "content": request.user_prompt }
            ]
        });

        let resp = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;
        let content = json["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let tokens_used = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LlmResponse {
            content,
            tokens_used,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ────────── Utilities ──────────

/// Parse JSON from LLM response text.
///
/// Strips markdown fences if present and parses as `serde_json::Value`.
/// Returns `None` on any parse error.
pub fn parse_json_from_llm(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Strip markdown fences
    let json_str = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    serde_json::from_str(json_str).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_from_llm_plain() {
        let json = r#"{"category": "transaction", "amount": 50}"#;
        let result = parse_json_from_llm(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["category"], "transaction");
    }

    #[test]
    fn test_parse_json_from_llm_fenced() {
        let json = "```json\n{\"category\": \"state_change\"}\n```";
        let result = parse_json_from_llm(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["category"], "state_change");
    }

    #[test]
    fn test_parse_json_from_llm_bare_fences() {
        let json = "```\n{\"ok\": true}\n```";
        let result = parse_json_from_llm(json);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_json_from_llm_empty() {
        assert!(parse_json_from_llm("").is_none());
        assert!(parse_json_from_llm("  ").is_none());
    }

    #[test]
    fn test_parse_json_from_llm_invalid() {
        assert!(parse_json_from_llm("not json").is_none());
    }
}
