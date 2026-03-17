// Application State

use crate::read_gate::ReadGate;
use crate::sequence::SequenceTracker;
use crate::write_lanes::WriteLanes;
use agent_db_graph::subscription::manager::SubscriptionManager;
use agent_db_graph::GraphEngine;
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
}
