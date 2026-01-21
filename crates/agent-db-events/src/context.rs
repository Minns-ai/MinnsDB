//! Context processing and similarity matching

use crate::core::EventContext;
use agent_db_core::types::ContextHash;
use agent_db_core::utils::cosine_similarity;
use std::collections::HashMap;

/// Context matcher for finding similar contexts
pub struct ContextMatcher {
    /// Stored contexts with their hashes
    contexts: HashMap<ContextHash, EventContext>,

    /// Similarity threshold for matches
    similarity_threshold: f32,

    /// Maximum contexts to store
    max_contexts: usize,
}

impl ContextMatcher {
    /// Create new context matcher
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            similarity_threshold: 0.8,
            max_contexts: 10_000,
        }
    }

    /// Create matcher with custom settings
    pub fn with_settings(similarity_threshold: f32, max_contexts: usize) -> Self {
        Self {
            contexts: HashMap::new(),
            similarity_threshold,
            max_contexts,
        }
    }

    /// Add context to matcher
    pub fn add_context(&mut self, context: EventContext) {
        // If we're at capacity, remove random context (simple eviction)
        if self.contexts.len() >= self.max_contexts {
            if let Some(key) = self.contexts.keys().next().copied() {
                self.contexts.remove(&key);
            }
        }

        let hash = context.fingerprint;
        self.contexts.insert(hash, context);
    }

    /// Find similar contexts
    pub fn find_similar(&self, context: &EventContext) -> Vec<(ContextHash, f32)> {
        let mut similarities = Vec::new();

        for (hash, stored_context) in &self.contexts {
            let similarity = context.similarity(stored_context);
            if similarity >= self.similarity_threshold {
                similarities.push((*hash, similarity));
            }
        }

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities
    }

    /// Get exact context match
    pub fn get_exact_match(&self, hash: ContextHash) -> Option<&EventContext> {
        self.contexts.get(&hash)
    }

    /// Get number of stored contexts
    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    /// Check if matcher is empty
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    /// Clear all stored contexts
    pub fn clear(&mut self) {
        self.contexts.clear();
    }

    /// Set similarity threshold
    pub fn set_similarity_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get similarity threshold
    pub fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }
}

impl Default for ContextMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced context with additional computed features
#[derive(Debug, Clone)]
pub struct EnhancedContext {
    /// Base context
    pub context: EventContext,

    /// Computed features for matching
    pub features: ContextFeatures,

    /// Context classification
    pub classification: ContextClassification,
}

/// Extracted context features for similarity matching
#[derive(Debug, Clone)]
pub struct ContextFeatures {
    /// Environment feature vector
    pub environment_features: Vec<f32>,

    /// Goal-based features
    pub goal_features: Vec<f32>,

    /// Resource utilization features
    pub resource_features: Vec<f32>,

    /// Temporal features
    pub temporal_features: Vec<f32>,

    /// Combined feature vector
    pub combined_features: Vec<f32>,
}

/// Context classification for pattern recognition
#[derive(Debug, Clone)]
pub struct ContextClassification {
    /// Primary context type
    pub primary_type: ContextType,

    /// Secondary context types
    pub secondary_types: Vec<ContextType>,

    /// Confidence in classification
    pub confidence: f32,

    /// Context complexity score
    pub complexity: f32,
}

/// Types of contexts for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContextType {
    /// High-pressure, deadline-driven context
    HighPressure,

    /// Routine, normal operation context
    Routine,

    /// Resource-constrained context
    ResourceConstrained,

    /// Exploratory, learning context
    Exploratory,

    /// Emergency or failure context
    Emergency,

    /// Collaborative, multi-agent context
    Collaborative,

    /// Unknown or unclassified
    Unknown,
}

impl EnhancedContext {
    /// Create enhanced context from basic context
    pub fn from_context(context: EventContext) -> Self {
        let features = ContextFeatures::extract(&context);
        let classification = ContextClassification::classify(&context, &features);

        Self {
            context,
            features,
            classification,
        }
    }

    /// Calculate similarity with another enhanced context
    pub fn similarity(&self, other: &EnhancedContext) -> f32 {
        // Weighted combination of different similarity measures
        let base_similarity = self.context.similarity(&other.context);
        let feature_similarity = cosine_similarity(
            &self.features.combined_features,
            &other.features.combined_features,
        );

        // Weight base similarity more heavily
        0.7 * base_similarity + 0.3 * feature_similarity
    }
}

impl ContextFeatures {
    /// Extract features from context
    pub fn extract(context: &EventContext) -> Self {
        let environment_features = Self::extract_environment_features(context);
        let goal_features = Self::extract_goal_features(context);
        let resource_features = Self::extract_resource_features(context);
        let temporal_features = Self::extract_temporal_features(context);

        // Combine all features
        let mut combined_features = Vec::new();
        combined_features.extend_from_slice(&environment_features);
        combined_features.extend_from_slice(&goal_features);
        combined_features.extend_from_slice(&resource_features);
        combined_features.extend_from_slice(&temporal_features);

        Self {
            environment_features,
            goal_features,
            resource_features,
            temporal_features,
            combined_features,
        }
    }

    fn extract_environment_features(context: &EventContext) -> Vec<f32> {
        let mut features = Vec::new();

        // Number of environment variables
        features.push(context.environment.variables.len() as f32);

        // Spatial features
        if let Some(spatial) = &context.environment.spatial {
            features.extend_from_slice(&[
                spatial.location.0 as f32,
                spatial.location.1 as f32,
                spatial.location.2 as f32,
            ]);

            // Has bounds
            features.push(if spatial.bounds.is_some() { 1.0 } else { 0.0 });
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }

        // Temporal features
        if let Some(time) = &context.environment.temporal.time_of_day {
            features.push(time.hour as f32 / 24.0); // Normalize hour
            features.push(time.minute as f32 / 60.0); // Normalize minute
        } else {
            features.extend_from_slice(&[0.0, 0.0]);
        }

        features.push(context.environment.temporal.deadlines.len() as f32);

        features
    }

    fn extract_goal_features(context: &EventContext) -> Vec<f32> {
        let mut features = Vec::new();

        // Number of goals
        features.push(context.active_goals.len() as f32);

        // Goal statistics
        if !context.active_goals.is_empty() {
            let avg_priority: f32 = context.active_goals.iter().map(|g| g.priority).sum::<f32>()
                / context.active_goals.len() as f32;

            let avg_progress: f32 = context.active_goals.iter().map(|g| g.progress).sum::<f32>()
                / context.active_goals.len() as f32;

            let max_priority = context
                .active_goals
                .iter()
                .map(|g| g.priority)
                .fold(0.0f32, |a, b| a.max(b));

            features.extend_from_slice(&[avg_priority, avg_progress, max_priority]);

            // Goals with deadlines
            let goals_with_deadlines = context
                .active_goals
                .iter()
                .filter(|g| g.deadline.is_some())
                .count() as f32;
            features.push(goals_with_deadlines / context.active_goals.len() as f32);
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }

        features
    }

    fn extract_resource_features(context: &EventContext) -> Vec<f32> {
        let comp = &context.resources.computational;

        vec![
            comp.cpu_percent / 100.0,                               // Normalize CPU
            comp.memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0),  // GB
            comp.storage_bytes as f32 / (1024.0 * 1024.0 * 1024.0), // GB
            comp.network_bandwidth as f32 / 10000.0,                // Normalize to 10Gbps
            context.resources.external.len() as f32,                // External resource count
        ]
    }

    fn extract_temporal_features(context: &EventContext) -> Vec<f32> {
        vec![context.environment.temporal.patterns.len() as f32]
    }
}

impl ContextClassification {
    /// Classify context based on features
    pub fn classify(context: &EventContext, features: &ContextFeatures) -> Self {
        let mut scores = HashMap::new();

        // High pressure classification
        let high_pressure_score = Self::score_high_pressure(context, features);
        scores.insert(ContextType::HighPressure, high_pressure_score);

        // Resource constrained classification
        let resource_constrained_score = Self::score_resource_constrained(context, features);
        scores.insert(ContextType::ResourceConstrained, resource_constrained_score);

        // Routine classification
        let routine_score = Self::score_routine(context, features);
        scores.insert(ContextType::Routine, routine_score);

        // Emergency classification
        let emergency_score = Self::score_emergency(context, features);
        scores.insert(ContextType::Emergency, emergency_score);

        // Find primary type (highest score)
        let (primary_type, confidence) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(t, s)| (t.clone(), *s))
            .unwrap_or((ContextType::Unknown, 0.0));

        // Find secondary types (above threshold)
        let secondary_types: Vec<ContextType> = scores
            .iter()
            .filter(|(t, s)| **s > 0.3 && **t != primary_type)
            .map(|(t, _)| t.clone())
            .collect();

        // Calculate complexity
        let complexity = Self::calculate_complexity(context, features);

        Self {
            primary_type,
            secondary_types,
            confidence,
            complexity,
        }
    }

    fn score_high_pressure(context: &EventContext, _features: &ContextFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Many deadlines
        if !context.environment.temporal.deadlines.is_empty() {
            score += 0.4;

            // High priority deadlines
            let high_priority_deadlines = context
                .environment
                .temporal
                .deadlines
                .iter()
                .filter(|d| d.priority > 0.7)
                .count();
            if high_priority_deadlines > 0 {
                score += 0.3;
            }
        }

        // High priority goals
        let high_priority_goals = context
            .active_goals
            .iter()
            .filter(|g| g.priority > 0.8)
            .count();
        if high_priority_goals > 0 {
            score += 0.3;
        }

        score.min(1.0)
    }

    fn score_resource_constrained(_context: &EventContext, features: &ContextFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // High CPU usage
        if features.resource_features.len() > 0 && features.resource_features[0] > 0.8 {
            score += 0.4;
        }

        // Low available external resources
        // (This would need more sophisticated analysis of external resources)

        score.min(1.0)
    }

    fn score_routine(context: &EventContext, _features: &ContextFeatures) -> f32 {
        let mut score: f32 = 0.5; // Base routine score

        // Few deadlines
        if context.environment.temporal.deadlines.len() <= 1 {
            score += 0.2;
        }

        // Moderate priority goals
        let moderate_goals = context
            .active_goals
            .iter()
            .filter(|g| g.priority >= 0.3 && g.priority <= 0.7)
            .count();
        if moderate_goals > 0 {
            score += 0.3;
        }

        score.min(1.0)
    }

    fn score_emergency(context: &EventContext, features: &ContextFeatures) -> f32 {
        let mut score: f32 = 0.0;

        // Very high resource usage
        if features.resource_features.len() > 0 && features.resource_features[0] > 0.95 {
            score += 0.5;
        }

        // Multiple high-priority deadlines
        let urgent_deadlines = context
            .environment
            .temporal
            .deadlines
            .iter()
            .filter(|d| d.priority > 0.9)
            .count();
        if urgent_deadlines > 1 {
            score += 0.5;
        }

        score.min(1.0)
    }

    fn calculate_complexity(context: &EventContext, features: &ContextFeatures) -> f32 {
        let mut complexity = 0.0;

        // Number of active goals
        complexity += context.active_goals.len() as f32 * 0.1;

        // Number of environment variables
        complexity += context.environment.variables.len() as f32 * 0.05;

        // Number of external resources
        complexity += context.resources.external.len() as f32 * 0.1;

        // Feature vector dimensionality
        complexity += features.combined_features.len() as f32 * 0.01;

        complexity.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn create_test_context() -> EventContext {
        EventContext::new(
            EnvironmentState {
                variables: {
                    let mut vars = HashMap::new();
                    vars.insert("temp".to_string(), json!(23.0));
                    vars
                },
                spatial: Some(SpatialContext {
                    location: (1.0, 2.0, 3.0),
                    bounds: None,
                    reference_frame: "test".to_string(),
                }),
                temporal: TemporalContext {
                    time_of_day: Some(TimeOfDay {
                        hour: 14,
                        minute: 30,
                        timezone: "UTC".to_string(),
                    }),
                    deadlines: Vec::new(),
                    patterns: Vec::new(),
                },
            },
            vec![Goal {
                id: 1,
                description: "Test goal".to_string(),
                priority: 0.5,
                deadline: None,
                progress: 0.3,
                subgoals: Vec::new(),
            }],
            ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 60.0,
                    memory_bytes: 1024 * 1024,
                    storage_bytes: 1024 * 1024 * 1024,
                    network_bandwidth: 1000,
                },
                external: HashMap::new(),
            },
        )
    }

    #[test]
    fn test_context_matcher() {
        let mut matcher = ContextMatcher::new();
        let context1 = create_test_context();
        let context2 = create_test_context();

        matcher.add_context(context1.clone());
        matcher.add_context(context2);

        assert_eq!(matcher.len(), 2);

        let similar = matcher.find_similar(&context1);
        assert!(!similar.is_empty());

        // Should find itself
        let exact = matcher.get_exact_match(context1.fingerprint);
        assert!(exact.is_some());
    }

    #[test]
    fn test_enhanced_context() {
        let context = create_test_context();
        let enhanced = EnhancedContext::from_context(context);

        assert!(!enhanced.features.combined_features.is_empty());
        assert!(enhanced.classification.confidence >= 0.0);
        assert!(enhanced.classification.complexity >= 0.0);
    }

    #[test]
    fn test_context_features() {
        let context = create_test_context();
        let features = ContextFeatures::extract(&context);

        assert!(!features.environment_features.is_empty());
        assert!(!features.goal_features.is_empty());
        assert!(!features.resource_features.is_empty());
        assert!(!features.combined_features.is_empty());

        // Check that combined features include all component features
        let total_component_len = features.environment_features.len()
            + features.goal_features.len()
            + features.resource_features.len()
            + features.temporal_features.len();
        assert_eq!(features.combined_features.len(), total_component_len);
    }

    #[test]
    fn test_context_classification() {
        let context = create_test_context();
        let features = ContextFeatures::extract(&context);
        let classification = ContextClassification::classify(&context, &features);

        assert_ne!(classification.primary_type, ContextType::Unknown);
        assert!(classification.confidence >= 0.0 && classification.confidence <= 1.0);
        assert!(classification.complexity >= 0.0 && classification.complexity <= 1.0);
    }

    #[test]
    fn test_high_pressure_classification() {
        let mut context = create_test_context();

        // Add high priority deadlines
        context.environment.temporal.deadlines.push(Deadline {
            goal_id: 1,
            timestamp: 1000000,
            priority: 0.9,
        });

        // Add high priority goals
        context.active_goals[0].priority = 0.9;

        let features = ContextFeatures::extract(&context);
        let classification = ContextClassification::classify(&context, &features);

        // Should be classified as high pressure
        assert_eq!(classification.primary_type, ContextType::HighPressure);
        assert!(classification.confidence > 0.5);
    }
}
