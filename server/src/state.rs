// Application State

use crate::read_gate::ReadGate;
use crate::sequence::SequenceTracker;
use crate::write_lanes::WriteLanes;
use agent_db_graph::GraphEngine;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<GraphEngine>,
    pub write_lanes: Arc<WriteLanes>,
    pub read_gate: Arc<ReadGate>,
    pub seq_tracker: Arc<SequenceTracker>,
    /// Monotonic instant captured at server start (for uptime calculation)
    pub started_at: Instant,
}
