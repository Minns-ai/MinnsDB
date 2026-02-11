//! LLM client abstraction for claim extraction

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// A single labeled entity from NER, passed to the LLM for grounding.
#[derive(Debug, Clone)]
pub struct LabeledEntity {
    /// Entity text as it appears in the source (e.g. "John", "Google")
    pub text: String,
    /// NER label (e.g. "PERSON", "ORG", "LOC", "DATE", "PRODUCT", "EVENT")
    pub label: String,
}

/// LLM extraction request
#[derive(Debug, Clone)]
pub struct LlmExtractionRequest {
    /// Raw text to extract claims from
    pub text: String,
    /// Labeled entity mentions from NER (optional).
    /// Each entry carries the entity text **and** its NER label.
    pub entities: Vec<LabeledEntity>,
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
    /// Claim type: "preference", "fact", "belief", "intention", "capability"
    #[serde(default = "default_claim_type_str")]
    pub claim_type: String,
    /// Primary entity this claim is about (optional, LLM best-effort)
    #[serde(default)]
    pub subject_entity: Option<String>,
}

fn default_claim_type_str() -> String {
    "fact".to_string()
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
            r#"You are a claim extractor. You receive text **and** a list of named entities
pre-identified by an NER service (each with its label: PERSON, ORG, PRODUCT,
LOC, DATE, EVENT, etc.).  Use these entities to anchor your claims.

Rules:
1. Extract at most {max_claims} claims.
2. Each claim must be **atomic** (single fact/statement).
3. Each claim MUST have supporting evidence spans (UTF-8 byte offsets into the original text).
4. Only extract claims that are **explicitly** stated in the text — do not infer.
5. Classify each claim as one of:
   - "preference"  — personal taste or like/dislike ("I like X", "I prefer Y")
   - "fact"        — objective, verifiable statement ("X costs $10", "The API uses REST")
   - "belief"      — uncertain opinion ("I think X will work", "Maybe we should Y")
   - "intention"   — desired future action ("I want to do X", "I plan to Y")
   - "capability"  — system/agent ability ("The system supports X", "It can handle 1k RPS")
   Use the NER labels as a signal:
     • Claims about PERSON/ORG entities are often "fact".
     • Claims with sentiment words (like/dislike/prefer/hate) are "preference".
     • Claims about PRODUCT capabilities are "capability".
     • Claims with future-tense verbs (want/plan/will) are "intention".
     • Claims with hedging words (think/maybe/probably) are "belief".
6. Set "subject_entity" to the **primary NER entity** the claim is about.
   Prefer the exact entity text from the NER list (case-sensitive match).
   If no NER entity matches, use the most relevant noun phrase.

Output format — strict JSON:
{{
  "claims": [
    {{
      "claim_text": "John works at Google",
      "evidence_spans": [
        {{"start_offset": 0, "end_offset": 24}}
      ],
      "confidence": 0.95,
      "claim_type": "fact",
      "subject_entity": "John"
    }}
  ]
}}"#,
            max_claims = max_claims
        )
    }

    /// Build the user prompt, formatting NER entities with their labels.
    fn user_prompt(text: &str, entities: &[LabeledEntity]) -> String {
        let entity_hint = if entities.is_empty() {
            String::new()
        } else {
            // Group entities by label for a structured hint
            let formatted: Vec<String> = entities
                .iter()
                .map(|e| format!("{} [{}]", e.text, e.label))
                .collect();
            format!("\n\nNER entities detected:\n{}", formatted.join("\n"))
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
        assert!(prompt.contains("preference"));
        assert!(prompt.contains("NER"));
    }

    #[test]
    fn test_user_prompt_with_labeled_entities() {
        let entities = vec![
            LabeledEntity {
                text: "John".to_string(),
                label: "PERSON".to_string(),
            },
            LabeledEntity {
                text: "Google".to_string(),
                label: "ORG".to_string(),
            },
        ];
        let prompt = OpenAiClient::user_prompt("John works at Google", &entities);
        assert!(prompt.contains("John works at Google"));
        assert!(prompt.contains("John [PERSON]"));
        assert!(prompt.contains("Google [ORG]"));
        assert!(prompt.contains("NER entities detected"));
    }

    #[test]
    fn test_user_prompt_empty_entities() {
        let prompt = OpenAiClient::user_prompt("No entities here", &[]);
        assert!(prompt.contains("No entities here"));
        assert!(!prompt.contains("NER entities"));
    }
}
