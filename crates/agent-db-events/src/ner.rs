//! Named Entity Recognition (NER) types and structures

use agent_db_core::types::EventId;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Extracted NER features for an event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFeatures {
    /// Source event ID
    pub event_id: EventId,

    /// Entity spans with labels and offsets
    pub entity_spans: Vec<EntitySpan>,

    /// Fingerprint for idempotency (hash of source text + model version)
    pub feature_fingerprint: u64,

    /// Extraction timestamp
    pub extracted_at: u64,

    /// Model/version used for extraction
    pub ner_model: String,

    /// Sentence-level entities (optional, for granular validation)
    pub sentence_entities: Option<Vec<SentenceEntities>>,
}

/// Entities found within a single sentence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEntities {
    /// Sentence text
    pub text: String,
    /// Byte offset where sentence starts in original text
    pub start_offset: usize,
    /// Entities found in this sentence (with relative offsets)
    pub entities: Vec<EntitySpan>,
}

/// Single entity span with UTF-8 byte offsets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySpan {
    /// Entity label: PERSON, ORG, LOC, DATE, etc.
    pub label: String,

    /// Start offset (UTF-8 bytes)
    pub start_offset: usize,

    /// End offset (UTF-8 bytes, exclusive)
    pub end_offset: usize,

    /// Confidence score [0.0, 1.0]
    pub confidence: f32,

    /// Extracted text (for convenience and validation)
    pub text: String,
}

impl SentenceEntities {
    /// Split text into sentences with start offsets
    pub fn split_into_sentences(text: &str) -> Vec<Self> {
        let mut sentences = Vec::new();
        let mut start = 0;

        // Simple sentence splitting by common delimiters
        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let end = i + 1;
                let raw_sentence = &text[start..end];
                
                // Calculate absolute offset of the trimmed sentence
                let trimmed = raw_sentence.trim_start();
                let trim_offset = raw_sentence.len() - trimmed.len();
                let final_sentence = trimmed.trim_end();
                
                if !final_sentence.is_empty() {
                    sentences.push(Self {
                        text: final_sentence.to_string(),
                        start_offset: start + trim_offset,
                        entities: Vec::new(),
                    });
                }
                start = end;
            }
        }

        // Add remaining text if any
        if start < text.len() {
            let raw_sentence = &text[start..];
            let trimmed = raw_sentence.trim_start();
            let trim_offset = raw_sentence.len() - trimmed.len();
            let final_sentence = trimmed.trim_end();
            
            if !final_sentence.is_empty() {
                sentences.push(Self {
                    text: final_sentence.to_string(),
                    start_offset: start + trim_offset,
                    entities: Vec::new(),
                });
            }
        }

        sentences
    }
}

impl ExtractedFeatures {
    /// Compute fingerprint for deduplication and idempotency
    pub fn compute_fingerprint(text: &str, model: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        model.hash(&mut hasher);
        hasher.finish()
    }

    /// Create new ExtractedFeatures with computed fingerprint
    pub fn new(
        event_id: EventId,
        entity_spans: Vec<EntitySpan>,
        ner_model: String,
        source_text: &str,
        sentence_entities: Option<Vec<SentenceEntities>>,
    ) -> Self {
        let feature_fingerprint = Self::compute_fingerprint(source_text, &ner_model);
        let extracted_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            event_id,
            entity_spans,
            feature_fingerprint,
            extracted_at,
            ner_model,
            sentence_entities,
        }
    }

    /// Validate all entity spans against source text
    pub fn validate_spans(&self, source_text: &str) -> bool {
        for span in &self.entity_spans {
            if !span.validate(source_text) {
                return false;
            }
        }
        true
    }
}

impl EntitySpan {
    /// Create a new entity span with validation
    pub fn new(label: String, start: usize, end: usize, confidence: f32, text: String) -> Self {
        Self {
            label,
            start_offset: start,
            end_offset: end,
            confidence,
            text,
        }
    }

    /// Validate span against source text
    pub fn validate(&self, source_text: &str) -> bool {
        // Check bounds
        if self.end_offset > source_text.len() {
            return false;
        }

        if self.start_offset >= self.end_offset {
            return false;
        }

        // Extract and compare text
        let extracted = &source_text[self.start_offset..self.end_offset];
        extracted == self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_span_validation() {
        let text = "John works at Google in California.";

        let span = EntitySpan::new(
            "PERSON".to_string(),
            0,
            4,
            0.95,
            "John".to_string(),
        );

        assert!(span.validate(text));
    }

    #[test]
    fn test_entity_span_invalid_bounds() {
        let text = "Short text";

        let span = EntitySpan::new(
            "PERSON".to_string(),
            0,
            100, // Out of bounds
            0.95,
            "Invalid".to_string(),
        );

        assert!(!span.validate(text));
    }

    #[test]
    fn test_fingerprint_consistency() {
        let text = "Test text";
        let model = "test-model-v1";

        let fp1 = ExtractedFeatures::compute_fingerprint(text, model);
        let fp2 = ExtractedFeatures::compute_fingerprint(text, model);

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_inputs() {
        let text1 = "Test text 1";
        let text2 = "Test text 2";
        let model = "test-model-v1";

        let fp1 = ExtractedFeatures::compute_fingerprint(text1, model);
        let fp2 = ExtractedFeatures::compute_fingerprint(text2, model);

        assert_ne!(fp1, fp2);
    }
}
