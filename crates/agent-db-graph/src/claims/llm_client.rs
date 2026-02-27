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

/// A few-shot example for claim extraction.
#[derive(Debug, Clone)]
pub struct FewShotExample {
    /// Example input text
    pub text: String,
    /// Expected extracted claims (as JSON strings)
    pub claims: Vec<String>,
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
    /// Role of the content source for role-aware prompt selection.
    pub source_role: super::types::SourceRole,
    /// Custom natural-language instructions appended to the system prompt.
    pub custom_instructions: Option<String>,
    /// Types of facts to specifically look for (e.g., ["dietary preferences", "travel plans"]).
    pub extraction_includes: Vec<String>,
    /// Types of facts to specifically ignore (e.g., ["greetings", "small talk"]).
    pub extraction_excludes: Vec<String>,
    /// Few-shot examples to include in the prompt.
    pub few_shot_examples: Vec<FewShotExample>,
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
    /// Category tag for this claim (e.g., "personal", "preferences", "work")
    #[serde(default)]
    pub category: Option<String>,
    /// Temporal stability: "static", "dynamic", or "atemporal"
    #[serde(default = "default_temporal_type_str")]
    pub temporal_type: String,
}

fn default_temporal_type_str() -> String {
    "dynamic".to_string()
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

    /// Build the system prompt, with role-specific focus.
    ///
    /// Different roles emphasize different claim types to prevent cross-role
    /// pollution (e.g., agent instructions misclassified as user preferences).
    fn system_prompt(
        max_claims: usize,
        role: super::types::SourceRole,
        custom_instructions: Option<&str>,
        includes: &[String],
        excludes: &[String],
    ) -> String {
        let role_guidance = match role {
            crate::claims::types::SourceRole::User => {
                r#"You are extracting facts from USER-generated content.
Focus on:
- Personal preferences and opinions ("I like X", "I prefer Y")
- Personal facts about the user ("I work at X", "My name is Y")
- Intentions and plans ("I want to X", "I plan to Y")
- Beliefs and opinions ("I think X", "I believe Y")
Do NOT extract:
- System instructions or agent capabilities
- Generic knowledge that isn't personally attributed
- Repeated greetings or pleasantries"#
            },
            crate::claims::types::SourceRole::Assistant => {
                r#"You are extracting facts from AI ASSISTANT-generated content.
Focus on:
- Capabilities and features ("The system can X", "This supports Y")
- Tool usage patterns and recommendations
- Factual statements about systems or processes
- Corrections or clarifications the assistant made
Do NOT extract:
- The user's personal preferences (those belong to user extraction)
- Generic conversational filler or acknowledgments
- Hypothetical examples used for illustration"#
            },
            crate::claims::types::SourceRole::System => {
                r#"You are extracting facts from SYSTEM/CONTEXT content.
Focus on:
- Configuration values and constraints
- Environmental facts ("The database is PostgreSQL", "API rate limit is 100/min")
- Operational state ("Service is in maintenance mode")
- Rules and policies
Do NOT extract:
- User opinions or preferences
- Speculative or hypothetical statements"#
            },
        };

        let mut prompt = format!(
            r#"{role_guidance}

You also receive a list of named entities pre-identified by an NER service
(each with its label: PERSON, ORG, PRODUCT, LOC, DATE, EVENT, etc.).
Use these entities to anchor your claims.

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
7. Assign a "category" from: personal, preferences, work, health, education, finance,
   travel, food, hobbies, technology, relationships, goals, habits, lifestyle, opinions, other.
   Use the closest match. If none fits, use "other".
8. Classify temporal stability as "temporal_type":
   - "static"    — stable real-world facts unlikely to change ("Paris is the capital of France", "Python was created by Guido van Rossum")
   - "dynamic"   — facts that can change over time ("Alice lives in NYC", "The price is $10", "I prefer dark mode")
   - "atemporal" — mathematical, logical, or definitional truths ("2+2=4", "HTTP is a protocol")
   Default to "dynamic" if unsure.

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
      "subject_entity": "John",
      "category": "work",
      "temporal_type": "dynamic"
    }}
  ]
}}"#,
            role_guidance = role_guidance,
            max_claims = max_claims
        );

        if !includes.is_empty() {
            prompt.push_str("\n\nSPECIFICALLY LOOK FOR these types of information:\n- ");
            prompt.push_str(&includes.join("\n- "));
        }

        if !excludes.is_empty() {
            prompt.push_str("\n\nDO NOT EXTRACT these types of information:\n- ");
            prompt.push_str(&excludes.join("\n- "));
        }

        if let Some(instructions) = custom_instructions {
            prompt.push_str("\n\nADDITIONAL INSTRUCTIONS:\n");
            prompt.push_str(instructions);
        }

        prompt
    }

    /// Build the user prompt, formatting NER entities with their labels.
    fn user_prompt(text: &str, entities: &[LabeledEntity], examples: &[FewShotExample]) -> String {
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

        let examples_section = if examples.is_empty() {
            String::new()
        } else {
            let mut section = String::from("=== EXAMPLES ===\n");
            for (i, example) in examples.iter().enumerate() {
                section.push_str(&format!(
                    "\nExample {}:\nText: \"{}\"\nClaims:\n- {}\n",
                    i + 1,
                    example.text,
                    example.claims.join("\n- "),
                ));
            }
            section.push_str("\n=== END EXAMPLES ===\n\n");
            section
        };

        format!(
            "{}Text:\n\n{}{}\n\nExtract claims with evidence:",
            examples_section, text, entity_hint
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

        let system_prompt = Self::system_prompt(
            request.max_claims,
            request.source_role,
            request.custom_instructions.as_deref(),
            &request.extraction_includes,
            &request.extraction_excludes,
        );
        let user_prompt =
            Self::user_prompt(&request.text, &request.entities, &request.few_shot_examples);

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

        let system_prompt = OpenAiClient::system_prompt(
            request.max_claims,
            request.source_role,
            request.custom_instructions.as_deref(),
            &request.extraction_includes,
            &request.extraction_excludes,
        );
        let user_prompt =
            OpenAiClient::user_prompt(&request.text, &request.entities, &request.few_shot_examples);

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
            source_role: crate::claims::types::SourceRole::User,
            custom_instructions: None,
            extraction_includes: vec![],
            extraction_excludes: vec![],
            few_shot_examples: vec![],
        };

        let result = client.extract_claims(request).await.unwrap();
        assert_eq!(result.claims.len(), 0);
        assert_eq!(result.tokens_used, 0);
    }

    #[test]
    fn test_system_prompt_user_role() {
        let prompt =
            OpenAiClient::system_prompt(5, crate::claims::types::SourceRole::User, None, &[], &[]);
        assert!(prompt.contains("5 claims"));
        assert!(prompt.contains("atomic"));
        assert!(prompt.contains("evidence"));
        assert!(prompt.contains("preference"));
        assert!(prompt.contains("NER"));
        assert!(prompt.contains("USER-generated"));
        assert!(prompt.contains("Personal preferences"));
    }

    #[test]
    fn test_system_prompt_assistant_role() {
        let prompt = OpenAiClient::system_prompt(
            5,
            crate::claims::types::SourceRole::Assistant,
            None,
            &[],
            &[],
        );
        assert!(prompt.contains("AI ASSISTANT"));
        assert!(prompt.contains("Capabilities"));
        assert!(!prompt.contains("USER-generated"));
    }

    #[test]
    fn test_system_prompt_system_role() {
        let prompt = OpenAiClient::system_prompt(
            5,
            crate::claims::types::SourceRole::System,
            None,
            &[],
            &[],
        );
        assert!(prompt.contains("SYSTEM/CONTEXT"));
        assert!(prompt.contains("Configuration"));
    }

    #[test]
    fn test_system_prompt_includes_category() {
        let prompt =
            OpenAiClient::system_prompt(5, crate::claims::types::SourceRole::User, None, &[], &[]);
        assert!(prompt.contains("category"));
        assert!(prompt.contains("personal, preferences, work"));
        assert!(prompt.contains("\"category\": \"work\""));
    }

    #[test]
    fn test_system_prompt_with_custom_instructions() {
        let prompt = OpenAiClient::system_prompt(
            5,
            crate::claims::types::SourceRole::User,
            Some("Only extract dietary facts"),
            &[],
            &[],
        );
        assert!(prompt.contains("ADDITIONAL INSTRUCTIONS:"));
        assert!(prompt.contains("Only extract dietary facts"));
    }

    #[test]
    fn test_system_prompt_with_includes() {
        let includes = vec![
            "dietary preferences".to_string(),
            "travel plans".to_string(),
        ];
        let prompt = OpenAiClient::system_prompt(
            5,
            crate::claims::types::SourceRole::User,
            None,
            &includes,
            &[],
        );
        assert!(prompt.contains("SPECIFICALLY LOOK FOR"));
        assert!(prompt.contains("dietary preferences"));
        assert!(prompt.contains("travel plans"));
    }

    #[test]
    fn test_system_prompt_with_excludes() {
        let excludes = vec!["greetings".to_string(), "small talk".to_string()];
        let prompt = OpenAiClient::system_prompt(
            5,
            crate::claims::types::SourceRole::User,
            None,
            &[],
            &excludes,
        );
        assert!(prompt.contains("DO NOT EXTRACT"));
        assert!(prompt.contains("greetings"));
        assert!(prompt.contains("small talk"));
    }

    #[test]
    fn test_system_prompt_no_custom() {
        let prompt =
            OpenAiClient::system_prompt(5, crate::claims::types::SourceRole::User, None, &[], &[]);
        assert!(!prompt.contains("SPECIFICALLY LOOK FOR"));
        assert!(!prompt.contains("DO NOT EXTRACT"));
        assert!(!prompt.contains("ADDITIONAL INSTRUCTIONS:"));
    }

    #[test]
    fn test_llm_claim_category_deserialization() {
        // With category
        let json = r#"{"claim_text":"I like pizza","evidence_spans":[],"confidence":0.9,"claim_type":"preference","category":"food"}"#;
        let claim: LlmClaim = serde_json::from_str(json).unwrap();
        assert_eq!(claim.category.as_deref(), Some("food"));

        // Without category
        let json = r#"{"claim_text":"I like pizza","evidence_spans":[],"confidence":0.9}"#;
        let claim: LlmClaim = serde_json::from_str(json).unwrap();
        assert!(claim.category.is_none());
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
        let prompt = OpenAiClient::user_prompt("John works at Google", &entities, &[]);
        assert!(prompt.contains("John works at Google"));
        assert!(prompt.contains("John [PERSON]"));
        assert!(prompt.contains("Google [ORG]"));
        assert!(prompt.contains("NER entities detected"));
    }

    #[test]
    fn test_user_prompt_empty_entities() {
        let prompt = OpenAiClient::user_prompt("No entities here", &[], &[]);
        assert!(prompt.contains("No entities here"));
        assert!(!prompt.contains("NER entities"));
    }

    #[test]
    fn test_user_prompt_with_few_shot_examples() {
        let examples = vec![
            FewShotExample {
                text: "I work at Google".to_string(),
                claims: vec![r#"{"claim_text":"User works at Google"}"#.to_string()],
            },
            FewShotExample {
                text: "I love sushi and hate pizza".to_string(),
                claims: vec![
                    r#"{"claim_text":"User loves sushi"}"#.to_string(),
                    r#"{"claim_text":"User hates pizza"}"#.to_string(),
                ],
            },
        ];
        let prompt = OpenAiClient::user_prompt("Test text", &[], &examples);
        assert!(prompt.contains("=== EXAMPLES ==="));
        assert!(prompt.contains("Example 1:"));
        assert!(prompt.contains("Example 2:"));
        assert!(prompt.contains("I work at Google"));
        assert!(prompt.contains("User loves sushi"));
        assert!(prompt.contains("User hates pizza"));
        assert!(prompt.contains("=== END EXAMPLES ==="));
        assert!(prompt.contains("Test text"));
    }

    #[test]
    fn test_user_prompt_no_examples() {
        let prompt = OpenAiClient::user_prompt("Test text", &[], &[]);
        assert!(!prompt.contains("=== EXAMPLES ==="));
        assert!(prompt.contains("Test text"));
    }

    #[test]
    fn test_system_prompt_includes_temporal_type() {
        let prompt =
            OpenAiClient::system_prompt(5, crate::claims::types::SourceRole::User, None, &[], &[]);
        assert!(prompt.contains("temporal_type"));
        assert!(prompt.contains("\"static\""));
        assert!(prompt.contains("\"dynamic\""));
        assert!(prompt.contains("\"atemporal\""));
        assert!(prompt.contains("\"temporal_type\": \"dynamic\""));
    }

    #[test]
    fn test_llm_claim_temporal_type_deserialization() {
        // With temporal_type
        let json = r#"{"claim_text":"Paris is capital","evidence_spans":[],"confidence":0.95,"temporal_type":"static"}"#;
        let claim: LlmClaim = serde_json::from_str(json).unwrap();
        assert_eq!(claim.temporal_type, "static");

        // Without temporal_type (defaults to "dynamic")
        let json = r#"{"claim_text":"test","evidence_spans":[],"confidence":0.9}"#;
        let claim: LlmClaim = serde_json::from_str(json).unwrap();
        assert_eq!(claim.temporal_type, "dynamic");
    }

    #[test]
    fn test_few_shot_example_formatting() {
        let examples = vec![FewShotExample {
            text: "Sample text".to_string(),
            claims: vec![
                "claim A".to_string(),
                "claim B".to_string(),
                "claim C".to_string(),
            ],
        }];
        let prompt = OpenAiClient::user_prompt("Actual text", &[], &examples);
        assert!(prompt.contains("Text: \"Sample text\""));
        assert!(prompt.contains("- claim A\n- claim B\n- claim C"));
    }
}
