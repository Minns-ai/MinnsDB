//! NER integration for GraphEngine
//!
//! This module provides semantic memory integration via NER extraction.

use crate::integration::GraphEngine;
use agent_db_events::{Event, EventType};
use std::sync::Arc;
use tracing::{debug, warn};

impl GraphEngine {
    /// Extract NER features from an event (async, non-blocking)
    ///
    /// This method:
    /// 1. Checks if event qualifies for NER extraction
    /// 2. Extracts text from Context events or large context payloads
    /// 3. Submits to NER queue for async processing
    /// 4. Stores results in NER feature store
    pub(crate) async fn extract_ner_async(&self, event: &Event) {
        // Check if NER queue is available
        let queue: Arc<agent_db_ner::NerExtractionQueue> = match &self.ner_queue {
            Some(q) => q.clone(),
            None => {
                debug!("NER queue not initialized, skipping extraction for event {}", event.id);
                return;
            }
        };

        // Determine if event qualifies for NER extraction
        let text_to_extract = self.get_text_for_ner(event);

        if let Some(text) = text_to_extract {
            debug!(
                "Submitting event {} for NER extraction ({} bytes)",
                event.id,
                text.len()
            );

            // Explicitly clone only the Arcs we need for the background task
            let event_id = event.id;
            let ner_store: Option<Arc<agent_db_ner::NerFeatureStore>> = self.ner_store.clone();

            // Spawn background task to handle extraction
            tokio::spawn(async move {
                match queue.extract(event_id, text.clone()).await {
                    Ok(features) => {
                        debug!(
                            "NER extraction complete for event {}: {} entities found",
                            event_id,
                            features.entity_spans.len()
                        );

                        // Store features if store is available
                        if let Some(store) = ner_store {
                            if let Err(e) = store.store(&features) {
                                warn!("Failed to store NER features for event {}: {}", event_id, e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("NER extraction failed for event {}: {}", event_id, e);
                    }
                }
            });
        }
    }

    /// Get text from event for NER extraction
    ///
    /// Returns Some(text) if event qualifies for extraction:
    /// - EventType::Context events always qualify
    /// - Other events qualify if context_size_bytes exceeds threshold
    fn get_text_for_ner(&self, event: &Event) -> Option<String> {
        match &event.event_type {
            // Context events always get NER extraction
            EventType::Context { text, .. } => {
                if text.is_empty() {
                    None
                } else {
                    Some(text.clone())
                }
            }
            // Other events: check context size threshold
            _ => {
                if event.context_size_bytes >= self.config.ner_promotion_threshold {
                    // For now, we don't have a way to extract context text from non-Context events
                    // This would require segment storage implementation
                    // TODO: Implement segment storage dereferencing
                    debug!(
                        "Event {} has large context ({} bytes) but segment storage not yet implemented",
                        event.id,
                        event.context_size_bytes
                    );
                    None
                } else {
                    None
                }
            }
        }
    }
}
