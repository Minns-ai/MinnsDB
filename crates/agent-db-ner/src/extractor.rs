//! NER extraction trait and implementations

use agent_db_events::{EntitySpan, SentenceEntities};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Trait for NER extraction implementations
#[async_trait]
pub trait NerExtractor: Send + Sync {
    /// Extract entities from text
    async fn extract(&self, text: &str) -> Result<Vec<EntitySpan>>;

    /// Get model name/version for fingerprinting
    fn model_name(&self) -> &str;

    /// Extract entities from text, preserving sentence structure
    async fn extract_sentences(&self, text: &str) -> Result<Vec<SentenceEntities>> {
        let sentences = self.split_into_sentences(text);
        let mut results = Vec::new();

        for (sentence, start_offset) in sentences {
            let entities = self.extract(&sentence).await?;
            results.push(SentenceEntities {
                text: sentence,
                start_offset,
                entities,
            });
        }

        Ok(results)
    }

    /// Split text into sentences with start offsets
    /// Handles common abbreviations to avoid incorrect splits
    fn split_into_sentences(&self, text: &str) -> Vec<(String, usize)> {
        split_sentences_helper(text)
    }
}

/// Helper function to split text into sentences with start offsets
/// Handles common abbreviations to avoid incorrect splits
fn split_sentences_helper(text: &str) -> Vec<(String, usize)> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current_start = 0;

    let mut chars = text.char_indices().peekable();

    while let Some((byte_idx, c)) = chars.next() {
        if c == '.' || c == '!' || c == '?' {
            // Check if this is likely a sentence boundary
            if c == '.' && !is_likely_sentence_end(text, byte_idx) {
                continue;
            }

            // Include the punctuation in the sentence
            let end_byte = byte_idx + c.len_utf8();

            // Extract and trim the sentence
            if let Some(raw_sentence) = text.get(current_start..end_byte) {
                let trimmed = raw_sentence.trim_start();
                let start_trim_bytes = raw_sentence.len() - trimmed.len();
                let final_sentence = trimmed.trim_end();

                if !final_sentence.is_empty() {
                    sentences.push((final_sentence.to_string(), current_start + start_trim_bytes));
                }
            }

            current_start = end_byte;
        }
    }

    // Add remaining text if any
    if current_start < text.len() {
        if let Some(raw_sentence) = text.get(current_start..) {
            let trimmed = raw_sentence.trim_start();
            let start_trim_bytes = raw_sentence.len() - trimmed.len();
            let final_sentence = trimmed.trim_end();

            if !final_sentence.is_empty() {
                sentences.push((final_sentence.to_string(), current_start + start_trim_bytes));
            }
        }
    }

    // If no sentences found, return entire text as one sentence
    if sentences.is_empty() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let start_offset = text.find(trimmed).unwrap_or(0);
            sentences.push((trimmed.to_string(), start_offset));
        }
    }

    sentences
}

/// Check if a period is likely a sentence end (not an abbreviation)
fn is_likely_sentence_end(text: &str, period_idx: usize) -> bool {
    // Common abbreviations that should NOT end a sentence
    const ABBREVS: &[&str] = &["Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.", "St.", "vs.", "etc.", "e.g.", "i.e."];

    // Check the text before the period for common abbreviations
    for abbrev in ABBREVS {
        let abbrev_len = abbrev.len();
        if period_idx + 1 >= abbrev_len {
            let start = period_idx + 1 - abbrev_len;
            if let Some(slice) = text.get(start..=period_idx) {
                if slice == *abbrev {
                    return false;
                }
            }
        }
    }

    // Check for single capital letter abbreviations (initials like "J. Smith")
    if period_idx > 0 {
        if let Some(prev_char) = text[..period_idx].chars().last() {
            if prev_char.is_uppercase() {
                // Check if the character before is not alphanumeric (space or start)
                if period_idx >= 2 {
                    if let Some(before_prev) = text[..period_idx - 1].chars().last() {
                        if !before_prev.is_alphanumeric() {
                            return false;
                        }
                    }
                } else if period_idx == 1 {
                    // Single capital letter at start
                    return false;
                }
            }
        }
    }

    // Check what follows the period
    let after_period = &text[period_idx + 1..];
    if after_period.is_empty() {
        return true; // End of text
    }

    // Must have whitespace after period
    if let Some(first_char) = after_period.chars().next() {
        if !first_char.is_whitespace() {
            return false; // No space after period (e.g., "example.com")
        }
    }

    // Look for capital letter or number after whitespace
    for c in after_period.chars() {
        if !c.is_whitespace() {
            return c.is_uppercase() || c.is_numeric() || c == '"' || c == '\'';
        }
    }

    true
}

/// Configuration for external NER service
#[derive(Debug, Clone)]
pub struct NerServiceConfig {
    /// Base URL for the NER service endpoint
    pub base_url: String,
    /// HTTP request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Optional model name to request (service-specific)
    pub model: Option<String>,
    /// Number of retry attempts for failed requests (default: 3)
    pub max_retries: u32,
    /// Initial retry delay in milliseconds (default: 100)
    pub retry_delay_ms: u64,
}

impl Default for NerServiceConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8081/ner".to_string(),
            request_timeout_ms: 5_000,
            model: None,
            max_retries: 3,
            retry_delay_ms: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NerServiceExtractor {
    client: Client,
    config: NerServiceConfig,
    model_name: String,
}

#[derive(Debug, Serialize)]
struct NerServiceRequest {
    text: String,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NerServiceResponse {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    entities: Vec<NerServiceEntity>,
}

#[derive(Debug, Deserialize)]
struct NerServiceEntity {
    label: String,
    start_offset: usize,
    end_offset: usize,
    confidence: f32,
    #[serde(default)]
    text: Option<String>,
}

impl NerServiceExtractor {
    /// Create a new external NER service extractor
    pub fn new(config: NerServiceConfig) -> Result<Self> {
        if config.request_timeout_ms == 0 {
            return Err(anyhow!("request_timeout_ms must be greater than 0"));
        }

        let timeout = Duration::from_millis(config.request_timeout_ms);
        let client = Client::builder().timeout(timeout).build()?;

        let model_name = config
            .model
            .clone()
            .unwrap_or_else(|| "external-ner".to_string());

        info!(
            "Initialized external NER client: {} (retries: {}, timeout: {}ms)",
            config.base_url, config.max_retries, config.request_timeout_ms
        );

        Ok(Self {
            client,
            config,
            model_name,
        })
    }

    fn normalize_entity_text(
        source_text: &str,
        start_offset: usize,
        end_offset: usize,
        provided_text: Option<String>,
    ) -> Result<String> {
        let extracted = source_text
            .get(start_offset..end_offset)
            .ok_or_else(|| {
                anyhow!(
                    "Invalid UTF-8 byte offsets: {}..{} for text length {}",
                    start_offset,
                    end_offset,
                    source_text.len()
                )
            })?;

        match provided_text {
            Some(text) if text == extracted => Ok(text),
            Some(text) => {
                warn!(
                    "NER service returned mismatched entity text: '{}' vs extracted '{}'",
                    text, extracted
                );
                Ok(extracted.to_string())
            }
            None => Ok(extracted.to_string()),
        }
    }
}

#[async_trait]
impl NerExtractor for NerServiceExtractor {
    async fn extract(&self, text: &str) -> Result<Vec<EntitySpan>> {
        if text.is_empty() {
            debug!("Empty text provided to NER extractor");
            return Ok(Vec::new());
        }

        debug!("Extracting entities from text (length: {} bytes)", text.len());

        let request = NerServiceRequest {
            text: text.to_string(),
            model: self.config.model.clone(),
        };

        // Retry logic with exponential backoff
        let mut last_error: Option<anyhow::Error> = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(self.config.retry_delay_ms * 2_u64.pow(attempt - 1));
                debug!("Retry attempt {} after {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
            }

            let response = match self
                .client
                .post(&self.config.base_url)
                .json(&request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    warn!("NER service request failed (attempt {}): {}", attempt + 1, e);
                    last_error = Some(e.into());
                    continue;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();

                // Don't retry on client errors (4xx), only on server errors (5xx) or network issues
                if status.is_client_error() {
                    return Err(anyhow!(
                        "NER service client error (status {}): {}",
                        status,
                        body
                    ));
                }

                warn!("NER service server error (status {}, attempt {}): {}", status, attempt + 1, body);
                last_error = Some(anyhow!("Server error (status {}): {}", status, body));
                continue;
            }

            let response: NerServiceResponse = match response.json().await {
                Ok(resp) => resp,
                Err(e) => {
                    warn!("Failed to parse NER response (attempt {}): {}", attempt + 1, e);
                    last_error = Some(e.into());
                    continue;
                }
            };

            // Successfully received response, process entities
            let mut spans = Vec::new();

            for entity in response.entities {
                if entity.end_offset > text.len() || entity.start_offset >= entity.end_offset {
                    warn!(
                        "Skipping invalid entity span {}..{} for label {}",
                        entity.start_offset, entity.end_offset, entity.label
                    );
                    continue;
                }

                let span_text = match Self::normalize_entity_text(
                    text,
                    entity.start_offset,
                    entity.end_offset,
                    entity.text,
                ) {
                    Ok(text) => text,
                    Err(e) => {
                        warn!("Skipping entity due to invalid offsets: {}", e);
                        continue;
                    }
                };

                let span = EntitySpan::new(
                    entity.label,
                    entity.start_offset,
                    entity.end_offset,
                    entity.confidence,
                    span_text,
                );

                debug!(
                    "Found entity: {} ({}) at {}..{} with confidence {}",
                    span.text, span.label, span.start_offset, span.end_offset, span.confidence
                );

                spans.push(span);
            }

            info!("Extracted {} entities from text", spans.len());
            return Ok(spans);
        }

        // All retries exhausted
        Err(last_error.unwrap_or_else(|| anyhow!("NER service request failed after {} attempts", self.config.max_retries + 1)))
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_text() {
        let extractor = NerServiceExtractor::new(NerServiceConfig::default()).unwrap();
        let result = extractor.extract("").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_model_name() {
        let mut config = NerServiceConfig::default();
        config.model = Some("test-model".to_string());
        let extractor = NerServiceExtractor::new(config).unwrap();
        assert_eq!(extractor.model_name(), "test-model");
    }

    // Note: Full integration tests with external NER service require a running service
}
