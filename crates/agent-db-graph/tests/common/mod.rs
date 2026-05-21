//! Shared test fixtures for `agent-db-graph` integration tests.
//!
//! Every test that constructs a [`GraphEngine`](agent_db_graph::GraphEngine)
//! needs a reachable Qdrant; this module starts one container per test
//! binary and reuses it across tests.

pub mod qdrant_fixture;
