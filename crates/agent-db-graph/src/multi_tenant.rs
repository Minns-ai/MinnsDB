//! Multi-tenant graph engine
//!
//! Provides isolated graph instances per tenant with shared infrastructure
//! and cross-tenant analytics capabilities.

use crate::{GraphResult, GraphError, GraphEngine, GraphEngineConfig, GraphQuery, QueryResult};
use agent_db_core::tenant::{TenantId, TenantManager, TenantConfig, Operation, UsageUpdate, TenantError};
use agent_db_events::Event;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Multi-tenant graph engine that manages isolated graph instances
pub struct MultiTenantGraphEngine {
    /// Tenant management and limits
    tenant_manager: Arc<RwLock<TenantManager>>,
    
    /// Per-tenant graph engines
    tenant_graphs: Arc<RwLock<HashMap<TenantId, Arc<GraphEngine>>>>,

    /// Default configuration for new tenants
    default_config: GraphEngineConfig,
    
    /// Cross-tenant analytics engine (opt-in)
    analytics: Option<Arc<CrossTenantAnalytics>>,
}

impl MultiTenantGraphEngine {
    /// Create a new multi-tenant graph engine
    pub async fn new() -> GraphResult<Self> {
        Ok(Self {
            tenant_manager: Arc::new(RwLock::new(TenantManager::new())),
            tenant_graphs: Arc::new(RwLock::new(HashMap::new())),
            default_config: GraphEngineConfig::default(),
            analytics: None,
        })
    }
    
    /// Register a new tenant
    pub async fn register_tenant(&self, config: TenantConfig) -> Result<(), TenantError> {
        // Register with tenant manager
        {
            let mut manager = self.tenant_manager.write().await;
            manager.register_tenant(config.clone())?;
        }
        
        // Create isolated graph engine for this tenant
        let graph_config = self.create_graph_config_for_tenant(&config);
        let graph_engine = GraphEngine::with_config(graph_config)
            .await
            .map_err(|_e| TenantError::InsufficientPermissions)?;
        
        // Store the tenant's graph engine
        {
            let mut graphs = self.tenant_graphs.write().await;
            graphs.insert(config.tenant_id.clone(), Arc::new(graph_engine));
        }
        
        Ok(())
    }
    
    /// Process an event for a specific tenant
    pub async fn process_event(&self, tenant_id: &TenantId, event: Event) -> Result<crate::GraphOperationResult, TenantError> {
        // Check tenant limits
        {
            let manager = self.tenant_manager.read().await;
            manager.check_limits(tenant_id, &Operation::ProcessEvent)?;
        }
        
        // Get tenant's graph engine
        let graph_engine = {
            let graphs = self.tenant_graphs.read().await;
            graphs.get(tenant_id)
                .ok_or_else(|| TenantError::TenantNotFound(tenant_id.clone()))?
                .clone()
        };
        
        // Process the event
        let result = graph_engine.process_event(event)
            .await
            .map_err(|_| TenantError::InsufficientPermissions)?;
        
        // Update usage statistics
        {
            let mut manager = self.tenant_manager.write().await;
            manager.update_usage(tenant_id, UsageUpdate::EventProcessed);
        }
        
        // Optional: Send to cross-tenant analytics
        if let Some(ref analytics) = self.analytics {
            let tenant_config = {
                let manager = self.tenant_manager.read().await;
                manager.get_tenant(tenant_id).cloned()
            };
            
            if let Some(config) = tenant_config {
                if config.features.cross_tenant_insights {
                    analytics.record_tenant_activity(tenant_id, &result).await;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Execute a query for a specific tenant
    pub async fn execute_query(&self, tenant_id: &TenantId, query: GraphQuery) -> Result<QueryResult, TenantError> {
        // Check tenant limits
        {
            let manager = self.tenant_manager.read().await;
            manager.check_limits(tenant_id, &Operation::ExecuteQuery)?;
        }
        
        // Update usage (query started)
        {
            let mut manager = self.tenant_manager.write().await;
            manager.update_usage(tenant_id, UsageUpdate::QueryStarted);
        }
        
        // Get tenant's graph engine
        let graph_engine = {
            let graphs = self.tenant_graphs.read().await;
            graphs.get(tenant_id)
                .ok_or_else(|| TenantError::TenantNotFound(tenant_id.clone()))?
                .clone()
        };
        
        // Execute the query
        let result = graph_engine.execute_query(query)
            .await
            .map_err(|_| TenantError::InsufficientPermissions);
        
        // Update usage (query completed)
        {
            let mut manager = self.tenant_manager.write().await;
            manager.update_usage(tenant_id, UsageUpdate::QueryCompleted);
        }
        
        result
    }
    
    /// Get tenant usage statistics
    pub async fn get_tenant_usage(&self, tenant_id: &TenantId) -> Option<agent_db_core::tenant::TenantUsage> {
        let manager = self.tenant_manager.read().await;
        manager.get_usage(tenant_id).cloned()
    }
    
    /// Get cross-tenant insights (requires permission)
    pub async fn get_cross_tenant_insights(&self, requesting_tenant: &TenantId) -> Result<CrossTenantInsights, TenantError> {
        // Check if tenant has permission for cross-tenant analytics
        let has_permission = {
            let manager = self.tenant_manager.read().await;
            if let Some(config) = manager.get_tenant(requesting_tenant) {
                config.features.cross_tenant_insights
            } else {
                return Err(TenantError::TenantNotFound(requesting_tenant.clone()));
            }
        };
        
        if !has_permission {
            return Err(TenantError::FeatureNotEnabled("cross_tenant_insights".to_string()));
        }
        
        // Generate anonymized insights
        if let Some(ref analytics) = self.analytics {
            Ok(analytics.generate_insights().await)
        } else {
            Ok(CrossTenantInsights::default())
        }
    }
    
    /// Perform maintenance tasks (cleanup, optimization)
    pub async fn maintenance(&self) -> GraphResult<MaintenanceReport> {
        let mut report = MaintenanceReport::default();
        
        // Get all tenants
        let tenant_ids = {
            let manager = self.tenant_manager.read().await;
            manager.tenants.keys().cloned().collect::<Vec<_>>()
        };
        
        // Perform maintenance on each tenant
        for tenant_id in tenant_ids {
            if let Some(graph_engine) = {
                let graphs = self.tenant_graphs.read().await;
                graphs.get(&tenant_id).cloned()
            } {
                // Cleanup old data
                graph_engine.cleanup().await?;
                report.tenants_processed += 1;
                
                // Check if tenant needs reset
                {
                    let mut manager = self.tenant_manager.write().await;
                    manager.maybe_reset_monthly_counters(&tenant_id, agent_db_core::types::current_timestamp());
                }
            }
        }
        
        Ok(report)
    }
    
    /// Create graph configuration tailored to tenant
    fn create_graph_config_for_tenant(&self, tenant_config: &TenantConfig) -> GraphEngineConfig {
        let mut config = self.default_config.clone();
        
        // Adjust configuration based on tenant tier and features
        match &tenant_config.tier {
            agent_db_core::tenant::TenantTier::Free { .. } => {
                config.batch_size = 50;
                config.max_graph_size = 5_000;
            }
            agent_db_core::tenant::TenantTier::Pro { .. } => {
                config.batch_size = 100;
                config.max_graph_size = 50_000;
            }
            agent_db_core::tenant::TenantTier::Enterprise { .. } => {
                config.batch_size = 200;
                config.max_graph_size = 1_000_000;
            }
        }
        
        // Enable/disable features based on tenant configuration
        config.auto_pattern_detection = tenant_config.features.advanced_patterns;
        config.enable_query_cache = true; // Always enabled for performance
        
        config
    }
}

/// Cross-tenant analytics for discovering shared patterns
#[derive(Debug)]
pub struct CrossTenantAnalytics {
    /// Anonymized pattern storage
    global_patterns: Arc<RwLock<HashMap<String, GlobalPattern>>>,
}

impl CrossTenantAnalytics {
    pub fn new() -> Self {
        Self {
            global_patterns: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Record tenant activity for global pattern analysis
    pub async fn record_tenant_activity(&self, tenant_id: &TenantId, result: &crate::GraphOperationResult) {
        // Anonymize and aggregate patterns across tenants
        let mut patterns = self.global_patterns.write().await;
        
        for pattern in &result.patterns_detected {
            let global_pattern = patterns.entry(pattern.clone()).or_insert_with(|| GlobalPattern {
                pattern_name: pattern.clone(),
                occurrence_count: 0,
                tenant_count: std::collections::HashSet::new(),
                confidence: 0.0,
            });
            
            global_pattern.occurrence_count += 1;
            global_pattern.tenant_count.insert(tenant_id.clone());
            global_pattern.confidence = global_pattern.occurrence_count as f32 / global_pattern.tenant_count.len() as f32;
        }
    }
    
    /// Generate cross-tenant insights
    pub async fn generate_insights(&self) -> CrossTenantInsights {
        let patterns = self.global_patterns.read().await;
        
        // Find patterns that appear across multiple tenants
        let common_patterns: Vec<_> = patterns
            .values()
            .filter(|p| p.tenant_count.len() > 1)
            .cloned()
            .collect();
        
        CrossTenantInsights {
            total_patterns: patterns.len(),
            cross_tenant_patterns: common_patterns,
            most_common_pattern: patterns
                .values()
                .max_by_key(|p| p.occurrence_count)
                .map(|p| p.pattern_name.clone()),
        }
    }
}

/// Global pattern across tenants
#[derive(Debug, Clone)]
pub struct GlobalPattern {
    pub pattern_name: String,
    pub occurrence_count: u64,
    pub tenant_count: std::collections::HashSet<TenantId>, // Anonymous tenant IDs
    pub confidence: f32,
}

/// Cross-tenant insights (anonymized)
#[derive(Debug, Default)]
pub struct CrossTenantInsights {
    pub total_patterns: usize,
    pub cross_tenant_patterns: Vec<GlobalPattern>,
    pub most_common_pattern: Option<String>,
}

/// Maintenance operation report
#[derive(Debug, Default)]
pub struct MaintenanceReport {
    pub tenants_processed: u64,
    pub data_cleaned_bytes: u64,
    pub patterns_archived: u64,
}

// Re-export for convenience
pub use agent_db_core::tenant::{TenantId, TenantConfig, TenantTier, TenantLimits, TenantFeatures, RetentionPolicy};