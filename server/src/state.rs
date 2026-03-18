// Application State

use crate::read_gate::ReadGate;
use crate::sequence::SequenceTracker;
use crate::write_lanes::WriteLanes;
use agent_db_graph::subscription::manager::SubscriptionManager;
use agent_db_graph::GraphEngine;
use agent_db_tables::catalog::TableCatalog;
use minns_wasm_runtime::registry::ModuleRegistry;
use minns_wasm_runtime::runtime::WasmRuntime;
use minns_wasm_runtime::scheduler::ScheduleRunner;
use minns_auth::store::KeyStore;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<GraphEngine>,
    pub write_lanes: Arc<WriteLanes>,
    pub read_gate: Arc<ReadGate>,
    pub seq_tracker: Arc<SequenceTracker>,
    /// Monotonic instant captured at server start (for uptime calculation)
    pub started_at: Instant,
    /// Subscription manager for MinnsQL live queries.
    pub subscription_manager: Arc<Mutex<SubscriptionManager>>,
    /// Maps subscription_id → original query text (for list endpoint).
    pub subscription_queries: Arc<Mutex<HashMap<u64, String>>>,
    /// Temporal table catalog.
    pub table_catalog: Arc<tokio::sync::RwLock<TableCatalog>>,
    /// WASM runtime engine.
    pub wasm_runtime: Arc<WasmRuntime>,
    /// WASM module registry.
    pub module_registry: Arc<tokio::sync::RwLock<ModuleRegistry>>,
    /// WASM schedule runner.
    pub schedule_runner: Arc<tokio::sync::RwLock<ScheduleRunner>>,
    /// API key store for authentication.
    pub key_store: Arc<tokio::sync::RwLock<KeyStore>>,
    /// Whether auth is enabled (can be disabled via MINNS_AUTH_DISABLED=true for development).
    pub auth_enabled: bool,
}
