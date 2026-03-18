//! TriggerRouter: dispatch events to WASM modules based on their registered triggers.

use serde::Serialize;

use crate::abi;
use crate::error::WasmError;
use crate::module::ModuleInstance;
use crate::registry::ModuleRegistry;
use crate::runtime::WasmRuntime;

/// Event types that can trigger module execution.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum TriggerEvent {
    /// A row was inserted into a table.
    TableInsert {
        table: String,
        row_id: u64,
        version_id: u64,
    },
    /// A row was updated in a table.
    TableUpdate {
        table: String,
        row_id: u64,
        old_version_id: u64,
        new_version_id: u64,
    },
    /// A row was deleted from a table.
    TableDelete {
        table: String,
        row_id: u64,
        version_id: u64,
    },
    /// A graph edge was added.
    GraphEdgeAdded {
        source: u64,
        target: u64,
        edge_type: String,
    },
    /// A graph edge was removed.
    GraphEdgeRemoved {
        source: u64,
        target: u64,
        edge_type: String,
    },
    /// A graph node was added.
    GraphNodeAdded { node_id: u64, node_type: u8 },
}

impl TriggerEvent {
    /// Get the trigger type string for matching against module descriptors.
    fn trigger_type(&self) -> &str {
        match self {
            TriggerEvent::TableInsert { .. } => "table_insert",
            TriggerEvent::TableUpdate { .. } => "table_update",
            TriggerEvent::TableDelete { .. } => "table_delete",
            TriggerEvent::GraphEdgeAdded { .. } => "graph_edge_added",
            TriggerEvent::GraphEdgeRemoved { .. } => "graph_edge_removed",
            TriggerEvent::GraphNodeAdded { .. } => "graph_node_added",
        }
    }

    /// Get the table name if this is a table event.
    fn table_name(&self) -> Option<&str> {
        match self {
            TriggerEvent::TableInsert { table, .. }
            | TriggerEvent::TableUpdate { table, .. }
            | TriggerEvent::TableDelete { table, .. } => Some(table),
            _ => None,
        }
    }

    /// Get the edge type if this is a graph edge event.
    fn edge_type(&self) -> Option<&str> {
        match self {
            TriggerEvent::GraphEdgeAdded { edge_type, .. }
            | TriggerEvent::GraphEdgeRemoved { edge_type, .. } => Some(edge_type),
            _ => None,
        }
    }
}

/// Dispatch a trigger event to all matching modules.
/// Returns a list of (module_name, error) for any failures.
pub fn dispatch_trigger(
    runtime: &WasmRuntime,
    registry: &ModuleRegistry,
    event: &TriggerEvent,
) -> Vec<(String, WasmError)> {
    let event_bytes = match abi::to_msgpack(event) {
        Ok(b) => b,
        Err(e) => return vec![("*".into(), e)],
    };

    let event_type = event.trigger_type();
    let mut errors = Vec::new();

    for module_name in registry.list() {
        let instance = match registry.get(module_name) {
            Some(i) => i,
            None => continue,
        };

        if !instance.enabled {
            continue;
        }

        // Check if this module has a trigger matching this event
        for (trigger_idx, trigger) in instance.descriptor.triggers.iter().enumerate() {
            if !trigger_matches(trigger, event_type, event) {
                continue;
            }

            if let Err(e) = instance.call_trigger(runtime, trigger_idx as i32, &event_bytes) {
                tracing::warn!(
                    "trigger {} on module '{}' failed: {}",
                    trigger_idx,
                    module_name,
                    e
                );
                errors.push((module_name.to_string(), e));
            }
        }
    }

    errors
}

/// Check if a trigger definition matches an event.
fn trigger_matches(
    trigger: &crate::abi::TriggerDef,
    event_type: &str,
    event: &TriggerEvent,
) -> bool {
    if trigger.trigger_type != event_type {
        return false;
    }

    // For table triggers, check the table name matches.
    // If the trigger specifies a table, the event MUST have a matching table.
    if let Some(trigger_table) = &trigger.table {
        match event.table_name() {
            Some(event_table) if event_table == trigger_table => {},
            _ => return false, // event has no table or wrong table
        }
    }

    // For graph edge triggers, check the edge type matches.
    // If the trigger specifies an edge type, the event MUST have a matching edge.
    if let Some(trigger_edge) = &trigger.edge_type {
        match event.edge_type() {
            Some(event_edge) if event_edge == trigger_edge => {},
            _ => return false, // event has no edge type or wrong edge type
        }
    }

    true
}
