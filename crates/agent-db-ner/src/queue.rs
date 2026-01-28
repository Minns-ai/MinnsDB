//! Async NER extraction queue

use agent_db_core::types::EventId;
use agent_db_events::ExtractedFeatures;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::extractor::NerExtractor;

/// NER extraction job
struct NerJob {
    event_id: EventId,
    text: String,
    result_tx: tokio::sync::oneshot::Sender<Result<ExtractedFeatures>>,
}

/// Async queue for NER extraction
pub struct NerExtractionQueue {
    sender: mpsc::Sender<NerJob>,
}

impl NerExtractionQueue {
    /// Create a new extraction queue with worker pool
    ///
    /// # Arguments
    /// * `extractor` - The NER extractor implementation
    /// * `workers` - Number of worker threads
    pub fn new(extractor: Arc<dyn NerExtractor>, workers: usize) -> Self {
        Self::with_capacity(extractor, workers, 100)
    }

    /// Create a new extraction queue with specified capacity
    ///
    /// # Arguments
    /// * `extractor` - The NER extractor implementation
    /// * `workers` - Number of worker threads
    /// * `queue_size` - Maximum number of jobs that can be queued
    pub fn with_capacity(extractor: Arc<dyn NerExtractor>, workers: usize, queue_size: usize) -> Self {
        // Use bounded channel for backpressure
        let (tx, rx) = mpsc::channel(queue_size);

        info!(
            "Starting NER extraction queue with {} workers and queue size {}",
            workers, queue_size
        );

        // Share receiver among workers with Arc<Mutex>
        // Note: This creates some contention, but for I/O-bound NER work it's acceptable
        // For CPU-bound work, consider using crossbeam or flume for true MPMC
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        // Spawn worker pool
        for worker_id in 0..workers {
            let rx = Arc::clone(&rx);
            let extractor = Arc::clone(&extractor);

            tokio::spawn(async move {
                info!("NER worker {} started", worker_id);

                loop {
                    // Lock and receive job
                    // Keep critical section small - only recv() is inside the lock
                    let job = {
                        let mut rx_guard = rx.lock().await;
                        rx_guard.recv().await
                    };

                    match job {
                        Some(job) => {
                            if let Err(e) = Self::process_job(job, &*extractor).await {
                                error!("Worker {} failed to process job: {}", worker_id, e);
                            }
                        }
                        None => {
                            info!("NER worker {} channel closed, stopping", worker_id);
                            break;
                        }
                    }
                }

                info!("NER worker {} stopped", worker_id);
            });
        }

        Self { sender: tx }
    }

    /// Adjust entity span offsets from sentence-relative to document-absolute
    fn adjust_span_offset(
        span: &agent_db_events::EntitySpan,
        sentence_offset: usize,
    ) -> agent_db_events::EntitySpan {
        agent_db_events::EntitySpan {
            label: span.label.clone(),
            start_offset: span.start_offset + sentence_offset,
            end_offset: span.end_offset + sentence_offset,
            confidence: span.confidence,
            text: span.text.clone(),
        }
    }

    /// Process a single NER job
    async fn process_job(job: NerJob, extractor: &dyn NerExtractor) -> Result<()> {
        let event_id = job.event_id;

        let result = async {
            // Extract entities with sentence structure
            let sentence_entities = extractor.extract_sentences(&job.text).await?;

            // Flatten for top-level entity_spans, adjusting offsets from sentence-relative to document-absolute
            let mut all_spans = Vec::new();
            for sent in &sentence_entities {
                for span in &sent.entities {
                    let absolute_span = Self::adjust_span_offset(span, sent.start_offset);
                    all_spans.push(absolute_span);
                }
            }

            // Create ExtractedFeatures
            let features = ExtractedFeatures::new(
                event_id,
                all_spans,
                extractor.model_name().to_string(),
                &job.text,
                Some(sentence_entities),
            );

            Ok(features)
        }
        .await;

        // Send result back (ignore if receiver dropped)
        let _ = job.result_tx.send(result);

        Ok(())
    }

    /// Submit text for NER extraction
    ///
    /// This method will block if the queue is full until space is available.
    pub async fn extract(
        &self,
        event_id: EventId,
        text: String,
    ) -> Result<ExtractedFeatures> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender
            .send(NerJob {
                event_id,
                text,
                result_tx: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send NER job: {}", e))?;

        rx.await
            .map_err(|e| anyhow::anyhow!("Failed to receive NER result: {}", e))?
    }

    /// Try to submit text for NER extraction without blocking
    ///
    /// Returns None if the queue is full.
    pub fn try_extract(
        &self,
        event_id: EventId,
        text: String,
    ) -> Option<tokio::sync::oneshot::Receiver<Result<ExtractedFeatures>>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        match self.sender.try_send(NerJob {
            event_id,
            text,
            result_tx: tx,
        }) {
            Ok(_) => Some(rx),
            Err(e) => {
                warn!("Failed to send NER job (queue full or closed): {}", e);
                None
            }
        }
    }

    /// Submit text for NER extraction without waiting for result
    ///
    /// This method will block if the queue is full until space is available.
    /// Returns a receiver that can be awaited later to get the result.
    pub async fn extract_async(
        &self,
        event_id: EventId,
        text: String,
    ) -> Option<tokio::sync::oneshot::Receiver<Result<ExtractedFeatures>>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        match self.sender.send(NerJob {
            event_id,
            text,
            result_tx: tx,
        }).await {
            Ok(_) => Some(rx),
            Err(e) => {
                warn!("Failed to send async NER job: {}", e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::NerExtractor;
    use async_trait::async_trait;

    struct MockExtractor;

    #[async_trait]
    impl NerExtractor for MockExtractor {
        async fn extract(&self, _text: &str) -> Result<Vec<agent_db_events::EntitySpan>> {
            Ok(vec![])
        }

        fn model_name(&self) -> &str {
            "mock-0.1"
        }
    }

    #[tokio::test]
    async fn test_queue_creation() {
        let extractor = Arc::new(MockExtractor);
        let queue = NerExtractionQueue::new(extractor, 2);

        // Queue should be created successfully
        let result = queue.extract(1, "test text".to_string()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_extraction() {
        let extractor = Arc::new(MockExtractor);
        let queue = NerExtractionQueue::new(extractor, 1);

        let rx = queue.extract_async(2, "async test".to_string()).await;
        assert!(rx.is_some());

        let result = rx.unwrap().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_try_extract() {
        let extractor = Arc::new(MockExtractor);
        let queue = NerExtractionQueue::with_capacity(extractor, 1, 1);

        // First job should succeed
        let rx1 = queue.try_extract(1, "test 1".to_string());
        assert!(rx1.is_some());

        // Second job might fail if queue is full (racy but shows the API works)
        let _rx2 = queue.try_extract(2, "test 2".to_string());
        // Don't assert on _rx2 as it depends on timing
    }
}
