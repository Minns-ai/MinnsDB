//! LLM client abstraction for claim extraction

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// LLM extraction request
#[derive(Debug, Clone)]
pub struct LlmExtractionRequest {
    /// Raw text to extract claims from
    pub text: String,
    /// Entity mentions from NER (optional)
    pub entities: Vec<String>,
    /// Maximum number of claims to extract
    pub max_claims: usize,
}

/// LLM extraction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionResponse {
    /// Extracted claims with evidence
    pub claims: Vec<LlmClaim>,
    /// Tokens used (for cost tracking)
    pub tokens_used: u64,
}

/// Single claim from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmClaim {
    /// Atomic claim text
    pub claim_text: String,
    /// Evidence spans (byte offsets)
    pub evidence_spans: Vec<LlmEvidenceSpan>,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
}

/// Evidence span from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmEvidenceSpan {
    /// Start offset (UTF-8 bytes)
    pub start_offset: usize,
    /// End offset (UTF-8 bytes, exclusive)
    pub end_offset: usize,
}

/// Trait for LLM client implementations
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Extract claims from text
    async fn extract_claims(&self, request: LlmExtractionRequest) -> Result<LlmExtractionResponse>;

    /// Get model name for tracking
    fn model_name(&self) -> &str;
}

/// OpenAI client implementation
pub struct OpenAiClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiClient {
    /// Create a new OpenAI client
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }

    /// Build the system prompt
    fn system_prompt(max_claims: usize) -> String {
        format!(
            r#"You are a claim extractor. Extract atomic factual claims from the text provided.

Rules:
1. Extract at most {} claims
2. Each claim must be atomic (single fact/statement)
3. Each claim MUST have supporting evidence spans (byte offsets)
4. Only extract claims that are explicitly stated in the text
5. Do not infer or generate claims not present in the text

Output format: JSON with this structure:
{{
  "claims": [
    {{
      "claim_text": "John works at Google",
      "evidence_spans": [
        {{"start_offset": 0, "end_offset": 24}}
      ],
      "confidence": 0.95
    }}
  ]
}}"#,
            max_claims
        )
    }

    /// Build the user prompt
    fn user_prompt(text: &str, entities: &[String]) -> String {
        let entity_hint = if entities.is_empty() {
            String::new()
        } else {
            format!("\n\nEntities found: {}", entities.join(", "))
        };

        format!(
            "Text:\n\n{}{}\n\nExtract claims with evidence:",
            text, entity_hint
        )
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn extract_claims(&self, request: LlmExtractionRequest) -> Result<LlmExtractionResponse> {
        info!(
            "Extracting claims with OpenAI {} (max: {})",
            self.model, request.max_claims
        );

        let system_prompt = Self::system_prompt(request.max_claims);
        let user_prompt = Self::user_prompt(&request.text, &request.entities);

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error {}: {}", status, body));
        }

        let json: serde_json::Value = response.json().await?;

        // Extract content
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No content in OpenAI response"))?;

        // Parse claims
        let claims_json: serde_json::Value = serde_json::from_str(content)?;
        let claims: Vec<LlmClaim> = serde_json::from_value(claims_json["claims"].clone())?;

        // Extract token usage
        let tokens_used = json["usage"]["total_tokens"].as_u64().unwrap_or(0);

        info!(
            "Extracted {} claims, used {} tokens",
            claims.len(),
            tokens_used
        );

        Ok(LlmExtractionResponse {
            claims,
            tokens_used,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Anthropic Claude client implementation
pub struct AnthropicClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn extract_claims(&self, request: LlmExtractionRequest) -> Result<LlmExtractionResponse> {
        info!(
            "Extracting claims with Anthropic {} (max: {})",
            self.model, request.max_claims
        );

        let system_prompt = OpenAiClient::system_prompt(request.max_claims);
        let user_prompt = OpenAiClient::user_prompt(&request.text, &request.entities);

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error {}: {}", status, body));
        }

        let json: serde_json::Value = response.json().await?;

        // Extract content
        let content = json["content"][0]["text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No content in Anthropic response"))?;

        // Parse claims (expect JSON in response)
        let claims_json: serde_json::Value = serde_json::from_str(content)?;
        let claims: Vec<LlmClaim> = serde_json::from_value(claims_json["claims"].clone())?;

        // Extract token usage
        let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0);
        let tokens_used = input_tokens + output_tokens;

        info!(
            "Extracted {} claims, used {} tokens",
            claims.len(),
            tokens_used
        );

        Ok(LlmExtractionResponse {
            claims,
            tokens_used,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Mock client for testing (returns no claims)
pub struct MockClient {
    model_name: String,
}

impl MockClient {
    pub fn new() -> Self {
        Self {
            model_name: "mock-0.1".to_string(),
        }
    }
}

impl Default for MockClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for MockClient {
    async fn extract_claims(
        &self,
        _request: LlmExtractionRequest,
    ) -> Result<LlmExtractionResponse> {
        debug!("Mock client: returning no claims");
        Ok(LlmExtractionResponse {
            claims: vec![],
            tokens_used: 0,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client() {
        let client = MockClient::new();
        let request = LlmExtractionRequest {
            text: "Test text".to_string(),
            entities: vec![],
            max_claims: 5,
        };

        let result = client.extract_claims(request).await.unwrap();
        assert_eq!(result.claims.len(), 0);
        assert_eq!(result.tokens_used, 0);
    }

    #[test]
    fn test_system_prompt() {
        let prompt = OpenAiClient::system_prompt(5);
        assert!(prompt.contains("5 claims"));
        assert!(prompt.contains("atomic"));
        assert!(prompt.contains("evidence"));
    }

    #[test]
    fn test_user_prompt() {
        let prompt = OpenAiClient::user_prompt("Test text", &["PERSON".to_string()]);
        assert!(prompt.contains("Test text"));
        assert!(prompt.contains("PERSON"));
    }
}
