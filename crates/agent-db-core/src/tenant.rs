//! Multi-tenant support for the Agentic Database
//!
//! Provides tenant isolation, resource management, and cross-tenant analytics
//! while maintaining security and performance.

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique tenant identifier
pub type TenantId = String; // e.g., "cursor", "lovable", "claude-code"

/// Tenant-specific configuration and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Unique tenant identifier
    pub tenant_id: TenantId,
    
    /// Human-readable tenant name
    pub tenant_name: String,
    
    /// Subscription tier affecting resource limits
    pub tier: TenantTier,
    
    /// Resource limits for this tenant
    pub limits: TenantLimits,
    
    /// Feature flags enabled for this tenant
    pub features: TenantFeatures,
    
    /// Data retention policies
    pub retention: RetentionPolicy,
}

/// Tenant subscription tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TenantTier {
    Free {
        max_agents: u64,
        max_events_per_month: u64,
    },
    Pro {
        max_agents: u64,
        max_events_per_month: u64,
        max_storage_gb: u64,
    },
    Enterprise {
        max_agents: u64,
        max_events_per_month: u64,
        max_storage_gb: u64,
        dedicated_resources: bool,
    },
}

/// Resource limits per tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantLimits {
    /// Maximum number of agents
    pub max_agents: u64,
    
    /// Maximum events per minute (rate limiting)
    pub max_events_per_minute: u64,
    
    /// Maximum concurrent queries
    pub max_concurrent_queries: u64,
    
    /// Maximum graph size (nodes + edges)
    pub max_graph_size: u64,
    
    /// Maximum storage in bytes
    pub max_storage_bytes: u64,
    
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
}

/// Feature flags per tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantFeatures {
    /// Enable advanced pattern detection
    pub advanced_patterns: bool,
    
    /// Enable real-time inference
    pub realtime_inference: bool,
    
    /// Enable cross-agent analytics
    pub cross_agent_analytics: bool,
    
    /// Enable data export
    pub data_export: bool,
    
    /// Enable API access
    pub api_access: bool,
    
    /// Enable multi-tenant collaboration (shared patterns)
    pub cross_tenant_insights: bool,
}

/// Data retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// How long to keep raw events (in days)
    pub raw_events_retention_days: u32,
    
    /// How long to keep graph relationships (in days)  
    pub graph_retention_days: u32,
    
    /// How long to keep patterns (in days)
    pub patterns_retention_days: u32,
    
    /// Enable automatic data cleanup
    pub auto_cleanup: bool,
}

/// Tenant-scoped identifiers to ensure isolation
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct TenantScopedId<T> {
    pub tenant_id: TenantId,
    pub inner: T,
}

impl<T> TenantScopedId<T> {
    pub fn new(tenant_id: TenantId, inner: T) -> Self {
        Self { tenant_id, inner }
    }
    
    pub fn tenant(&self) -> &TenantId {
        &self.tenant_id
    }
    
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

/// Tenant-scoped agent identifier
pub type TenantAgentId = TenantScopedId<AgentId>;

/// Tenant-scoped event identifier  
pub type TenantEventId = TenantScopedId<EventId>;

/// Tenant-scoped node identifier
pub type TenantNodeId = TenantScopedId<NodeId>;

/// Multi-tenant database manager
#[derive(Debug)]
pub struct TenantManager {
    /// Tenant configurations
    tenants: HashMap<TenantId, TenantConfig>,
    
    /// Resource usage tracking per tenant
    usage_stats: HashMap<TenantId, TenantUsage>,
    
    /// Default configuration for new tenants
    default_config: TenantConfig,
}

/// Current resource usage for a tenant
#[derive(Debug, Clone, Default)]
pub struct TenantUsage {
    /// Current number of agents
    pub current_agents: u64,
    
    /// Events processed this month
    pub events_this_month: u64,
    
    /// Current storage usage in bytes
    pub storage_bytes: u64,
    
    /// Current active queries
    pub active_queries: u64,
    
    /// Current graph size
    pub graph_size: u64,
    
    /// Last reset timestamp (for monthly counters)
    pub last_reset: Timestamp,
}

impl TenantManager {
    /// Create a new tenant manager
    pub fn new() -> Self {
        Self {
            tenants: HashMap::new(),
            usage_stats: HashMap::new(),
            default_config: Self::default_tenant_config(),
        }
    }
    
    /// Register a new tenant
    pub fn register_tenant(&mut self, config: TenantConfig) -> Result<(), TenantError> {
        if self.tenants.contains_key(&config.tenant_id) {
            return Err(TenantError::TenantAlreadyExists(config.tenant_id));
        }
        
        self.usage_stats.insert(config.tenant_id.clone(), TenantUsage::default());
        self.tenants.insert(config.tenant_id.clone(), config);
        
        Ok(())
    }
    
    /// Get tenant configuration
    pub fn get_tenant(&self, tenant_id: &TenantId) -> Option<&TenantConfig> {
        self.tenants.get(tenant_id)
    }
    
    /// Check if tenant can perform an operation
    pub fn check_limits(&self, tenant_id: &TenantId, operation: &Operation) -> Result<(), TenantError> {
        let config = self.get_tenant(tenant_id)
            .ok_or_else(|| TenantError::TenantNotFound(tenant_id.clone()))?;
        
        let usage = self.usage_stats.get(tenant_id)
            .ok_or_else(|| TenantError::TenantNotFound(tenant_id.clone()))?;
        
        match operation {
            Operation::CreateAgent => {
                if usage.current_agents >= config.limits.max_agents {
                    return Err(TenantError::LimitExceeded("max_agents".to_string()));
                }
            }
            Operation::ProcessEvent => {
                // Check monthly event limit based on tier
                let monthly_limit = match &config.tier {
                    TenantTier::Free { max_events_per_month, .. } => *max_events_per_month,
                    TenantTier::Pro { max_events_per_month, .. } => *max_events_per_month,
                    TenantTier::Enterprise { max_events_per_month, .. } => *max_events_per_month,
                };
                
                if usage.events_this_month >= monthly_limit {
                    return Err(TenantError::LimitExceeded("monthly_events".to_string()));
                }
            }
            Operation::ExecuteQuery => {
                if usage.active_queries >= config.limits.max_concurrent_queries {
                    return Err(TenantError::LimitExceeded("concurrent_queries".to_string()));
                }
            }
            Operation::StoreData(bytes) => {
                if usage.storage_bytes + bytes > config.limits.max_storage_bytes {
                    return Err(TenantError::LimitExceeded("storage".to_string()));
                }
            }
        }
        
        Ok(())
    }
    
    /// Update tenant usage statistics
    pub fn update_usage(&mut self, tenant_id: &TenantId, update: UsageUpdate) {
        if let Some(usage) = self.usage_stats.get_mut(tenant_id) {
            match update {
                UsageUpdate::AgentCreated => usage.current_agents += 1,
                UsageUpdate::AgentRemoved => usage.current_agents = usage.current_agents.saturating_sub(1),
                UsageUpdate::EventProcessed => usage.events_this_month += 1,
                UsageUpdate::QueryStarted => usage.active_queries += 1,
                UsageUpdate::QueryCompleted => usage.active_queries = usage.active_queries.saturating_sub(1),
                UsageUpdate::StorageUsed(bytes) => usage.storage_bytes += bytes,
                UsageUpdate::StorageFreed(bytes) => usage.storage_bytes = usage.storage_bytes.saturating_sub(bytes),
                UsageUpdate::GraphGrew(nodes) => usage.graph_size += nodes,
            }
        }
    }
    
    /// Get current usage for a tenant
    pub fn get_usage(&self, tenant_id: &TenantId) -> Option<&TenantUsage> {
        self.usage_stats.get(tenant_id)
    }
    
    /// Reset monthly counters if needed
    pub fn maybe_reset_monthly_counters(&mut self, tenant_id: &TenantId, current_time: Timestamp) {
        if let Some(usage) = self.usage_stats.get_mut(tenant_id) {
            // Reset if it's been more than 30 days
            if current_time.saturating_sub(usage.last_reset) > 30 * 24 * 60 * 60 * 1_000_000_000 {
                usage.events_this_month = 0;
                usage.last_reset = current_time;
            }
        }
    }
    
    fn default_tenant_config() -> TenantConfig {
        TenantConfig {
            tenant_id: "default".to_string(),
            tenant_name: "Default Tenant".to_string(),
            tier: TenantTier::Free {
                max_agents: 100,
                max_events_per_month: 10_000,
            },
            limits: TenantLimits {
                max_agents: 100,
                max_events_per_minute: 100,
                max_concurrent_queries: 5,
                max_graph_size: 10_000,
                max_storage_bytes: 100 * 1024 * 1024, // 100MB
                query_timeout_ms: 5000,
            },
            features: TenantFeatures {
                advanced_patterns: false,
                realtime_inference: true,
                cross_agent_analytics: false,
                data_export: false,
                api_access: true,
                cross_tenant_insights: false,
            },
            retention: RetentionPolicy {
                raw_events_retention_days: 30,
                graph_retention_days: 90,
                patterns_retention_days: 365,
                auto_cleanup: true,
            },
        }
    }
}

/// Operations that can be performed by tenants
#[derive(Debug, Clone)]
pub enum Operation {
    CreateAgent,
    ProcessEvent,
    ExecuteQuery,
    StoreData(u64), // bytes
}

/// Usage updates to track
#[derive(Debug, Clone)]
pub enum UsageUpdate {
    AgentCreated,
    AgentRemoved,
    EventProcessed,
    QueryStarted,
    QueryCompleted,
    StorageUsed(u64),   // bytes added
    StorageFreed(u64),  // bytes removed
    GraphGrew(u64),     // nodes added
}

/// Tenant-related errors
#[derive(Debug, thiserror::Error)]
pub enum TenantError {
    #[error("Tenant not found: {0}")]
    TenantNotFound(TenantId),
    
    #[error("Tenant already exists: {0}")]
    TenantAlreadyExists(TenantId),
    
    #[error("Limit exceeded: {0}")]
    LimitExceeded(String),
    
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
    
    #[error("Insufficient permissions for operation")]
    InsufficientPermissions,
}