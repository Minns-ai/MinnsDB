//! User-scoped graph operations within tenants
//!
//! Handles multiple users within a single tenant (e.g., all Cursor users)
//! while enabling controlled knowledge sharing and privacy protection.

use crate::{GraphResult, GraphError, GraphEngine, GraphEngineConfig, GraphQuery, QueryResult};
use agent_db_core::tenant::TenantId;
use agent_db_core::types::{AgentId, EventId, generate_event_id, current_timestamp};
use agent_db_events::Event;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// User identifier within a tenant
pub type UserId = String;  // e.g., "alice@company.com", "user_12345"

/// Privacy level for data sharing within a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingLevel {
    /// No sharing - user data stays completely isolated
    Private,
    
    /// Share anonymized patterns only (no raw data, no agent IDs)
    Anonymous {
        /// Allow pattern names to be shared (e.g., "code_review_pattern")
        share_pattern_names: bool,
        /// Allow aggregated statistics (e.g., "30% of users do X after Y")  
        share_statistics: bool,
    },
    
    /// Share within tenant but keep user boundaries
    TenantWide {
        /// Allow cross-user agent collaboration inference
        allow_cross_user_inference: bool,
        /// Share successful workflow patterns
        share_workflows: bool,
    },
    
    /// Completely open within tenant
    Open,
}

/// User-scoped event that includes privacy preferences
#[derive(Debug, Clone)]
pub struct UserScopedEvent {
    /// Which user this event belongs to
    pub user_id: UserId,
    
    /// User's workspace/project context (for additional scoping)
    pub workspace_id: Option<String>,
    
    /// The actual event
    pub event: Event,
    
    /// How this event can be shared/learned from
    pub sharing_level: SharingLevel,
    
    /// User-defined tags for grouping
    pub user_tags: Vec<String>,
}

/// Multi-user graph engine for a single tenant
pub struct UserScopedGraphEngine {
    /// Tenant this engine belongs to
    tenant_id: TenantId,
    
    /// Per-user isolated graphs
    user_graphs: Arc<RwLock<HashMap<UserId, Arc<GraphEngine>>>>,
    
    /// Shared knowledge layer (anonymized patterns)
    shared_knowledge: Arc<RwLock<SharedKnowledgeLayer>>,
    
    /// Configuration
    config: UserScopedConfig,
}

#[derive(Debug, Clone)]
pub struct UserScopedConfig {
    /// Default sharing level for new users
    pub default_sharing_level: SharingLevel,
    
    /// Enable cross-user pattern detection
    pub enable_cross_user_patterns: bool,
    
    /// Maximum users per tenant
    pub max_users: usize,
    
    /// Enable workspace-level isolation
    pub enable_workspace_isolation: bool,
}

impl UserScopedGraphEngine {
    pub async fn new(tenant_id: TenantId, config: UserScopedConfig) -> GraphResult<Self> {
        Ok(Self {
            tenant_id,
            user_graphs: Arc::new(RwLock::new(HashMap::new())),
            shared_knowledge: Arc::new(RwLock::new(SharedKnowledgeLayer::new())),
            config,
        })
    }
    
    /// Register a new user within the tenant
    pub async fn register_user(&self, user_id: UserId, user_config: Option<UserConfig>) -> GraphResult<()> {
        let mut graphs = self.user_graphs.write().await;
        
        if graphs.contains_key(&user_id) {
            return Err(GraphError::OperationError(format!("User {} already exists", user_id)));
        }
        
        // Create isolated graph for this user
        let graph_config = GraphEngineConfig {
            // Smaller limits for individual users
            batch_size: 50,
            max_graph_size: 10_000,
            auto_pattern_detection: true,
            ..GraphEngineConfig::default()
        };
        
        let user_graph = GraphEngine::with_config(graph_config).await?;
        graphs.insert(user_id.clone(), Arc::new(user_graph));
        
        // Initialize user in shared knowledge if they allow sharing
        if let Some(ref config) = user_config {
            if !matches!(config.sharing_level, SharingLevel::Private) {
                let mut shared = self.shared_knowledge.write().await;
                shared.register_user(&user_id, &config.sharing_level);
            }
        }
        
        Ok(())
    }
    
    /// Process an event for a specific user
    pub async fn process_user_event(&self, event: UserScopedEvent) -> GraphResult<UserEventResult> {
        // Get user's isolated graph
        let user_graph = {
            let graphs = self.user_graphs.read().await;
            graphs.get(&event.user_id)
                .ok_or_else(|| GraphError::OperationError(format!("User {} not found", event.user_id)))?
                .clone()
        };
        
        // Process event in user's isolated graph
        let result = user_graph.process_event(event.event.clone()).await?;
        
        // Extract patterns for shared learning (if allowed)
        let shared_patterns = self.extract_shareable_patterns(&event, &result).await?;
        
        // Update shared knowledge layer
        if !shared_patterns.is_empty() {
            let mut shared = self.shared_knowledge.write().await;
            shared.add_patterns(&event.user_id, shared_patterns);
        }
        
        // Get recommendations from shared knowledge
        let recommendations = {
            let shared = self.shared_knowledge.read().await;
            shared.get_recommendations(&event.user_id, &event.event)
        };
        
        Ok(UserEventResult {
            user_id: event.user_id,
            processing_result: result,
            shared_patterns_contributed: shared_patterns.len(),
            recommendations,
        })
    }
    
    /// Execute a query for a specific user
    pub async fn execute_user_query(&self, user_id: &UserId, query: UserScopedQuery) -> GraphResult<UserQueryResult> {
        match query {
            UserScopedQuery::UserOnly(graph_query) => {
                // Query only the user's isolated graph
                let user_graph = {
                    let graphs = self.user_graphs.read().await;
                    graphs.get(user_id)
                        .ok_or_else(|| GraphError::OperationError(format!("User {} not found", user_id)))?
                        .clone()
                };
                
                let result = user_graph.execute_query(graph_query).await?;
                Ok(UserQueryResult::UserOnly(result))
            }
            
            UserScopedQuery::WithSharedInsights(graph_query) => {
                // Query user's graph + get shared insights
                let user_result = {
                    let graphs = self.user_graphs.read().await;
                    let user_graph = graphs.get(user_id)
                        .ok_or_else(|| GraphError::OperationError(format!("User {} not found", user_id)))?;
                    user_graph.execute_query(graph_query).await?
                };
                
                let shared_insights = {
                    let shared = self.shared_knowledge.read().await;
                    shared.get_insights_for_user(user_id)
                };
                
                Ok(UserQueryResult::WithSharedInsights {
                    user_result,
                    shared_insights,
                })
            }
            
            UserScopedQuery::CrossUserPattern(pattern_query) => {
                // Query patterns across users (respecting privacy)
                let shared = self.shared_knowledge.read().await;
                let pattern_result = shared.query_cross_user_patterns(&pattern_query, user_id)?;
                Ok(UserQueryResult::CrossUserPatterns(pattern_result))
            }
        }
    }
    
    /// Get user-specific analytics while respecting privacy
    pub async fn get_user_analytics(&self, user_id: &UserId) -> GraphResult<UserAnalytics> {
        let user_graph = {
            let graphs = self.user_graphs.read().await;
            graphs.get(user_id)
                .ok_or_else(|| GraphError::OperationError(format!("User {} not found", user_id)))?
                .clone()
        };
        
        let user_stats = user_graph.get_graph_stats().await;
        let user_patterns = user_graph.get_patterns().await;
        
        let shared_insights = {
            let shared = self.shared_knowledge.read().await;
            shared.get_comparative_insights(user_id, &user_patterns)
        };
        
        Ok(UserAnalytics {
            user_id: user_id.clone(),
            personal_stats: user_stats,
            personal_patterns: user_patterns,
            comparative_insights: shared_insights,
        })
    }
    
    /// Extract patterns that can be shared based on privacy settings
    async fn extract_shareable_patterns(&self, event: &UserScopedEvent, result: &crate::GraphOperationResult) -> GraphResult<Vec<AnonymizedPattern>> {
        let mut shareable = Vec::new();
        
        match &event.sharing_level {
            SharingLevel::Private => {
                // No sharing
            }
            SharingLevel::Anonymous { share_pattern_names, share_statistics } => {
                for pattern in &result.patterns_detected {
                    if *share_pattern_names {
                        shareable.push(AnonymizedPattern {
                            pattern_type: pattern.clone(),
                            frequency: 1,
                            user_id: None, // Fully anonymous
                            context: None,
                        });
                    }
                }
            }
            SharingLevel::TenantWide { .. } | SharingLevel::Open => {
                for pattern in &result.patterns_detected {
                    shareable.push(AnonymizedPattern {
                        pattern_type: pattern.clone(),
                        frequency: 1,
                        user_id: Some(event.user_id.clone()),
                        context: event.workspace_id.clone(),
                    });
                }
            }
        }
        
        Ok(shareable)
    }
}

/// Shared knowledge layer for cross-user learning
#[derive(Debug)]
struct SharedKnowledgeLayer {
    /// Anonymized patterns from all users
    patterns: HashMap<String, Vec<AnonymizedPattern>>,
    
    /// User sharing preferences
    user_sharing_levels: HashMap<UserId, SharingLevel>,
    
    /// Cross-user statistics (anonymized)
    statistics: CrossUserStatistics,
}

impl SharedKnowledgeLayer {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            user_sharing_levels: HashMap::new(),
            statistics: CrossUserStatistics::default(),
        }
    }
    
    fn register_user(&mut self, user_id: &UserId, sharing_level: &SharingLevel) {
        self.user_sharing_levels.insert(user_id.clone(), sharing_level.clone());
    }
    
    fn add_patterns(&mut self, user_id: &UserId, patterns: Vec<AnonymizedPattern>) {
        for pattern in patterns {
            self.patterns
                .entry(pattern.pattern_type.clone())
                .or_insert_with(Vec::new)
                .push(pattern);
        }
        
        // Update statistics
        self.statistics.total_patterns += patterns.len();
        self.statistics.contributing_users.insert(user_id.clone());
    }
    
    fn get_recommendations(&self, user_id: &UserId, event: &Event) -> Vec<Recommendation> {
        // Generate recommendations based on what other users did in similar situations
        // This would use the anonymized patterns to suggest workflows
        vec![] // Placeholder
    }
    
    fn get_insights_for_user(&self, user_id: &UserId) -> SharedInsights {
        // Return insights that this user is allowed to see
        SharedInsights {
            popular_patterns: self.get_popular_patterns_for_user(user_id),
            success_rates: self.get_anonymized_success_rates(),
            workflow_suggestions: vec![], // Placeholder
        }
    }
    
    fn get_popular_patterns_for_user(&self, user_id: &UserId) -> Vec<PopularPattern> {
        let sharing_level = self.user_sharing_levels.get(user_id);
        
        match sharing_level {
            Some(SharingLevel::Private) => vec![], // No shared patterns
            Some(SharingLevel::Anonymous { .. }) => {
                // Only fully anonymous patterns
                self.patterns.iter()
                    .filter_map(|(name, patterns)| {
                        let anonymous_count = patterns.iter().filter(|p| p.user_id.is_none()).count();
                        if anonymous_count > 0 {
                            Some(PopularPattern {
                                pattern_name: name.clone(),
                                frequency: anonymous_count,
                                anonymized: true,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            _ => {
                // All patterns this user can see
                self.patterns.iter()
                    .map(|(name, patterns)| PopularPattern {
                        pattern_name: name.clone(),
                        frequency: patterns.len(),
                        anonymized: false,
                    })
                    .collect()
            }
        }
    }
    
    fn query_cross_user_patterns(&self, query: &PatternQuery, requesting_user: &UserId) -> GraphResult<CrossUserPatternResult> {
        // Implementation would respect privacy boundaries
        Ok(CrossUserPatternResult {
            matching_patterns: vec![],
            total_users_with_pattern: 0,
            anonymized: true,
        })
    }
    
    fn get_comparative_insights(&self, user_id: &UserId, user_patterns: &[String]) -> ComparativeInsights {
        // Compare user's patterns to anonymized tenant-wide patterns
        ComparativeInsights {
            unique_patterns: user_patterns.iter()
                .filter(|p| !self.patterns.contains_key(*p))
                .cloned()
                .collect(),
            common_patterns: user_patterns.iter()
                .filter(|p| self.patterns.contains_key(*p))
                .cloned()
                .collect(),
            percentile_ranking: 50.0, // Placeholder calculation
        }
    }
    
    fn get_anonymized_success_rates(&self) -> HashMap<String, f32> {
        // Return anonymized success rates for different patterns
        HashMap::new() // Placeholder
    }
}

// Supporting types for the user-scoped system

#[derive(Debug, Clone)]
pub struct UserConfig {
    pub sharing_level: SharingLevel,
    pub workspace_isolation: bool,
}

#[derive(Debug)]
pub struct UserEventResult {
    pub user_id: UserId,
    pub processing_result: crate::GraphOperationResult,
    pub shared_patterns_contributed: usize,
    pub recommendations: Vec<Recommendation>,
}

#[derive(Debug)]
pub enum UserScopedQuery {
    /// Query only the user's isolated graph
    UserOnly(crate::GraphQuery),
    /// Query user's graph and include shared insights
    WithSharedInsights(crate::GraphQuery),
    /// Query cross-user patterns (anonymized)
    CrossUserPattern(PatternQuery),
}

#[derive(Debug)]
pub enum UserQueryResult {
    UserOnly(QueryResult),
    WithSharedInsights {
        user_result: QueryResult,
        shared_insights: SharedInsights,
    },
    CrossUserPatterns(CrossUserPatternResult),
}

#[derive(Debug)]
pub struct UserAnalytics {
    pub user_id: UserId,
    pub personal_stats: crate::GraphStats,
    pub personal_patterns: Vec<String>,
    pub comparative_insights: ComparativeInsights,
}

#[derive(Debug, Clone)]
pub struct AnonymizedPattern {
    pub pattern_type: String,
    pub frequency: usize,
    pub user_id: Option<UserId>, // None for fully anonymous
    pub context: Option<String>, // Workspace/project context
}

#[derive(Debug, Default)]
struct CrossUserStatistics {
    total_patterns: usize,
    contributing_users: std::collections::HashSet<UserId>,
}

#[derive(Debug)]
pub struct SharedInsights {
    pub popular_patterns: Vec<PopularPattern>,
    pub success_rates: HashMap<String, f32>,
    pub workflow_suggestions: Vec<WorkflowSuggestion>,
}

#[derive(Debug)]
pub struct PopularPattern {
    pub pattern_name: String,
    pub frequency: usize,
    pub anonymized: bool,
}

#[derive(Debug)]
pub struct Recommendation {
    pub recommendation_type: String,
    pub description: String,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct PatternQuery {
    pub pattern_name: String,
    pub min_frequency: usize,
}

#[derive(Debug)]
pub struct CrossUserPatternResult {
    pub matching_patterns: Vec<AnonymizedPattern>,
    pub total_users_with_pattern: usize,
    pub anonymized: bool,
}

#[derive(Debug)]
pub struct ComparativeInsights {
    pub unique_patterns: Vec<String>,     // Patterns only this user has
    pub common_patterns: Vec<String>,     // Patterns shared with others
    pub percentile_ranking: f32,          // How this user compares to others
}

#[derive(Debug)]
pub struct WorkflowSuggestion {
    pub name: String,
    pub steps: Vec<String>,
    pub success_rate: f32,
}