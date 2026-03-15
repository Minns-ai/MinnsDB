// crates/agent-db-graph/src/strategies/reasoning.rs
//
// Reasoning step extraction, generalization, and classification.

use crate::GraphResult;
use agent_db_events::core::{CognitiveType, Event, EventType};

use super::extractor::StrategyExtractor;
use super::types::ReasoningStep;

impl StrategyExtractor {
    /// Extract reasoning steps from cognitive events with generalization
    pub(crate) fn extract_reasoning_steps(&self, events: &[Event]) -> GraphResult<Vec<ReasoningStep>> {
        let mut raw_steps = Vec::new();
        let mut order = 0;

        // First pass: collect all reasoning traces
        for event in events {
            if let EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                reasoning_trace,
                ..
            } = &event.event_type
            {
                // Only extract from reasoning events
                for trace_step in reasoning_trace {
                    raw_steps.push((trace_step.clone(), order));
                    order += 1;
                }
            }
        }

        // Second pass: generalize the steps
        let mut generalized_steps = Vec::new();
        for (raw_step, seq_order) in raw_steps {
            let generalized = self.generalize_reasoning_step(&raw_step);

            // Determine applicability based on abstraction level
            let applicability = if self.is_highly_abstract(&generalized) {
                "general".to_string()
            } else if self.is_parameterized(&generalized) {
                "contextual".to_string()
            } else {
                "specific".to_string()
            };

            generalized_steps.push(ReasoningStep {
                description: generalized,
                applicability,
                expected_outcome: self.infer_expected_outcome(&raw_step),
                sequence_order: seq_order,
            });
        }

        Ok(generalized_steps)
    }

    /// Generalize a reasoning step by abstracting specific values
    pub(crate) fn generalize_reasoning_step(&self, step: &str) -> String {
        let mut generalized = step.to_string();

        // Pattern 1: Abstract file paths (simple pattern matching)
        // Look for common path patterns: /path, C:\path, ./path, ../path
        let path_patterns = ["/", "\\", "./", "../", "C:", "D:", "E:"];
        for pattern in &path_patterns {
            if generalized.contains(pattern) {
                // Simple heuristic: if contains path separator and looks like a path
                let words: Vec<&str> = generalized.split_whitespace().collect();
                let mut new_words = Vec::new();
                for word in words {
                    if word.contains('/')
                        || word.contains('\\')
                        || (word.starts_with('.') && word.len() > 1)
                    {
                        new_words.push("<file_path>");
                    } else {
                        new_words.push(word);
                    }
                }
                generalized = new_words.join(" ");
                break;
            }
        }

        // Pattern 2: Abstract error messages
        let error_keywords = [
            "error:",
            "exception:",
            "failure:",
            "error ",
            "exception ",
            "failed",
        ];
        for keyword in &error_keywords {
            if generalized.to_lowercase().contains(keyword) {
                // Replace error message with placeholder
                if let Some(pos) = generalized.to_lowercase().find(keyword) {
                    let before = &generalized[..pos];
                    let after_pos = pos + keyword.len();
                    // Find end of error message (next period, newline, or end)
                    let end_pos = generalized[after_pos..]
                        .find(['.', '\n', '\r'])
                        .map(|i| after_pos + i)
                        .unwrap_or(generalized.len());
                    generalized = format!(
                        "{}<error_type>: <error_message>{}",
                        before,
                        &generalized[end_pos..]
                    );
                }
                break;
            }
        }

        // Pattern 3: Abstract URLs
        if generalized.contains("http://") || generalized.contains("https://") {
            let words: Vec<&str> = generalized.split_whitespace().collect();
            let new_words: Vec<String> = words
                .iter()
                .map(|w| {
                    if w.starts_with("http://") || w.starts_with("https://") {
                        "<url>".to_string()
                    } else {
                        w.to_string()
                    }
                })
                .collect();
            generalized = new_words.join(" ");
        }

        // Pattern 4: Abstract large numbers (but keep small ones for structure)
        let words: Vec<&str> = generalized.split_whitespace().collect();
        let new_words: Vec<String> = words
            .iter()
            .map(|w| {
                if let Ok(num) = w.parse::<u64>() {
                    if num > 100 {
                        "<number>".to_string()
                    } else {
                        w.to_string()
                    }
                } else {
                    w.to_string()
                }
            })
            .collect();
        generalized = new_words.join(" ");

        // Pattern 5: Identify common reasoning structures
        generalized = self.identify_reasoning_structures(&generalized);

        generalized
    }

    /// Identify and label common reasoning structures
    pub(crate) fn identify_reasoning_structures(&self, step: &str) -> String {
        let step_lower = step.to_lowercase();

        // If-then structure
        if step_lower.contains("if")
            && (step_lower.contains("then") || step_lower.contains("check"))
        {
            return format!("[IF-THEN] {}", step);
        }

        // Try-catch/error handling
        if step_lower.contains("try")
            || step_lower.contains("catch")
            || step_lower.contains("handle error")
        {
            return format!("[ERROR-HANDLING] {}", step);
        }

        // Decomposition
        if step_lower.contains("break down")
            || step_lower.contains("decompose")
            || step_lower.contains("split")
        {
            return format!("[DECOMPOSE] {}", step);
        }

        // Verification/validation
        if step_lower.contains("verify")
            || step_lower.contains("validate")
            || step_lower.contains("check")
        {
            return format!("[VERIFY] {}", step);
        }

        // Search/lookup
        if step_lower.contains("search")
            || step_lower.contains("find")
            || step_lower.contains("lookup")
        {
            return format!("[SEARCH] {}", step);
        }

        step.to_string()
    }

    /// Check if a step is highly abstract (reusable across domains)
    pub(crate) fn is_highly_abstract(&self, step: &str) -> bool {
        let abstract_patterns = [
            "[IF-THEN]",
            "[ERROR-HANDLING]",
            "[DECOMPOSE]",
            "[VERIFY]",
            "[SEARCH]",
        ];

        abstract_patterns
            .iter()
            .any(|pattern| step.contains(pattern))
    }

    /// Check if a step is parameterized (has placeholders)
    pub(crate) fn is_parameterized(&self, step: &str) -> bool {
        step.contains("<") && step.contains(">")
    }

    /// Infer expected outcome from reasoning step
    pub(crate) fn infer_expected_outcome(&self, step: &str) -> Option<String> {
        let step_lower = step.to_lowercase();

        if step_lower.contains("solve") || step_lower.contains("fix") {
            Some("problem_resolved".to_string())
        } else if step_lower.contains("verify") || step_lower.contains("check") {
            Some("validation_result".to_string())
        } else if step_lower.contains("find") || step_lower.contains("search") {
            Some("item_found".to_string())
        } else if step_lower.contains("decompose") || step_lower.contains("break down") {
            Some("subproblems_identified".to_string())
        } else {
            None
        }
    }
}
