// Application State

use agent_db_graph::GraphEngine;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<GraphEngine>,
}
