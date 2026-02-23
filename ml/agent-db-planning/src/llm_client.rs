//! LLM client abstraction for the planning engine.
//!
//! Defines a general-purpose completion interface separate from the claim
//! extraction LLM client. This keeps the planning crate standalone with no
//! dependency on `agent-db-graph`.

use crate::types::PlanningError;

/// A request for LLM completion.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub model: Option<String>,
}

/// A response from an LLM completion.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: String,
    pub tokens_used: u32,
    pub model: String,
}

/// General-purpose LLM client for the planning engine.
///
/// Separate from the claim extraction `LlmClient` because planning needs
/// generic completion (system + user prompts → structured JSON), not
/// domain-specific claim extraction.
#[async_trait::async_trait]
pub trait PlanningLlmClient: Send + Sync {
    /// Send a completion request and return the response.
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, PlanningError>;

    /// Return the model name this client is configured to use.
    fn model_name(&self) -> &str;
}

/// Mock LLM client for testing.
///
/// Can be configured to return a fixed response or to fail.
pub struct MockPlanningLlmClient {
    /// Fixed response content to return.
    pub response_content: String,
    /// If true, return an error instead of a response.
    pub should_fail: bool,
    /// Model name to report.
    pub model: String,
}

impl MockPlanningLlmClient {
    /// Create a mock that returns the given content.
    pub fn new(response_content: String) -> Self {
        Self {
            response_content,
            should_fail: false,
            model: "mock-model".to_string(),
        }
    }

    /// Create a mock that always fails.
    pub fn failing() -> Self {
        Self {
            response_content: String::new(),
            should_fail: true,
            model: "mock-model".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl PlanningLlmClient for MockPlanningLlmClient {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, PlanningError> {
        if self.should_fail {
            return Err(PlanningError::LlmError(
                "mock client configured to fail".to_string(),
            ));
        }
        Ok(CompletionResponse {
            content: self.response_content.clone(),
            tokens_used: 100,
            model: self.model.clone(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client_returns_content() {
        let client = MockPlanningLlmClient::new("hello world".to_string());
        let request = CompletionRequest {
            system_prompt: "You are helpful.".to_string(),
            user_prompt: "Say hello.".to_string(),
            temperature: 0.0,
            max_tokens: 100,
            model: None,
        };
        let response = client.complete(request).await.unwrap();
        assert_eq!(response.content, "hello world");
        assert_eq!(response.model, "mock-model");
    }

    #[tokio::test]
    async fn test_mock_client_fails() {
        let client = MockPlanningLlmClient::failing();
        let request = CompletionRequest {
            system_prompt: String::new(),
            user_prompt: String::new(),
            temperature: 0.0,
            max_tokens: 100,
            model: None,
        };
        let result = client.complete(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mock client"));
    }

    #[test]
    fn test_model_name() {
        let client = MockPlanningLlmClient::new("test".to_string());
        assert_eq!(client.model_name(), "mock-model");
    }
}
