//! Causality tracking and analysis for events
//!
//! This module provides functionality for tracking and analyzing causal
//! relationships between events in the agent database.

use agent_db_core::error::{DatabaseError, DatabaseResult};
use agent_db_core::types::{EventId, Timestamp};
use std::collections::{HashMap, HashSet};

/// Tracks causal relationships between events
#[derive(Debug, Default)]
pub struct CausalityTracker {
    /// Map from event to its direct parents
    parents: HashMap<EventId, Vec<EventId>>,

    /// Map from event to its direct children
    children: HashMap<EventId, Vec<EventId>>,

    /// Timestamp tracking for temporal ordering
    timestamps: HashMap<EventId, Timestamp>,
}

impl CausalityTracker {
    /// Create a new causality tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a causal relationship
    pub fn add_causality(
        &mut self,
        child: EventId,
        parent: EventId,
        child_timestamp: Timestamp,
    ) -> DatabaseResult<()> {
        // Check temporal ordering if we have parent timestamp
        if let Some(&parent_timestamp) = self.timestamps.get(&parent) {
            if child_timestamp <= parent_timestamp {
                return Err(DatabaseError::validation(
                    "Child event cannot occur before or at same time as parent",
                ));
            }
        }

        // Add to parents map
        self.parents.entry(child).or_default().push(parent);

        // Add to children map
        self.children.entry(parent).or_default().push(child);

        // Record timestamp
        self.timestamps.insert(child, child_timestamp);

        Ok(())
    }

    /// Get direct parents of an event
    pub fn get_parents(&self, event_id: EventId) -> Option<&Vec<EventId>> {
        self.parents.get(&event_id)
    }

    /// Get direct children of an event
    pub fn get_children(&self, event_id: EventId) -> Option<&Vec<EventId>> {
        self.children.get(&event_id)
    }

    /// Check if event A is an ancestor of event B
    pub fn is_ancestor(&self, ancestor: EventId, descendant: EventId) -> bool {
        if let Some(parents) = self.get_parents(descendant) {
            for &parent in parents {
                if parent == ancestor || self.is_ancestor(ancestor, parent) {
                    return true;
                }
            }
        }
        false
    }

    /// Detect cycles in causality graph
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &event_id in self.timestamps.keys() {
            if self.has_cycle_util(event_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn has_cycle_util(
        &self,
        event_id: EventId,
        visited: &mut HashSet<EventId>,
        rec_stack: &mut HashSet<EventId>,
    ) -> bool {
        if rec_stack.contains(&event_id) {
            return true;
        }
        if visited.contains(&event_id) {
            return false;
        }

        visited.insert(event_id);
        rec_stack.insert(event_id);

        if let Some(children) = self.get_children(event_id) {
            for &child in children {
                if self.has_cycle_util(child, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(&event_id);
        false
    }

    /// Get all ancestors of an event
    pub fn get_ancestors(&self, event_id: EventId) -> HashSet<EventId> {
        let mut ancestors = HashSet::new();
        self.collect_ancestors(event_id, &mut ancestors);
        ancestors
    }

    fn collect_ancestors(&self, event_id: EventId, ancestors: &mut HashSet<EventId>) {
        if let Some(parents) = self.get_parents(event_id) {
            for &parent in parents {
                if ancestors.insert(parent) {
                    self.collect_ancestors(parent, ancestors);
                }
            }
        }
    }
}
