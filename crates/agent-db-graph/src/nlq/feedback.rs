//! Feedback logger for NLQ pipeline.
//!
//! Logs each NLQ query attempt for auditing and future improvement.

/// Feedback data from one NLQ pipeline execution.
#[derive(Debug, Clone)]
pub struct NlqFeedback {
    pub question: String,
    pub intent: String,
    pub entities_found: usize,
    pub template_used: Option<String>,
    pub query_built: bool,
    pub validation_result: String,
    pub execution_success: bool,
    pub result_count: usize,
    pub confidence: f32,
    pub execution_time_ms: u64,
}

/// Log NLQ feedback via tracing.
///
/// Uses structured logging so it can be captured by any tracing subscriber
/// (file, OpenTelemetry, etc.) without coupling to a specific storage type.
pub fn log_nlq_feedback(feedback: &NlqFeedback) {
    tracing::info!(
        question = %feedback.question,
        intent = %feedback.intent,
        entities_found = feedback.entities_found,
        template_used = ?feedback.template_used,
        query_built = feedback.query_built,
        validation_result = %feedback.validation_result,
        execution_success = feedback.execution_success,
        result_count = feedback.result_count,
        confidence = feedback.confidence,
        execution_time_ms = feedback.execution_time_ms,
        "NLQ query executed"
    );
}
