// Application State

use agent_db_graph::GraphEngine;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<GraphEngine>,
    /// Monotonic instant captured at server start (for uptime calculation)
    pub started_at: Instant,
}
