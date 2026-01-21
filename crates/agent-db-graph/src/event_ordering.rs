//! Event ordering and causal chain reconstruction
//!
//! Handles out-of-order event processing to ensure correct relationship inference
//! even when events arrive from multiple concurrent sources.

use crate::GraphResult;
use agent_db_core::types::{AgentId, EventId, Timestamp};
use agent_db_events::Event;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Event ordering engine that handles out-of-order events
pub struct EventOrderingEngine {
    /// Per-agent event buffers for ordering
    agent_buffers: Arc<RwLock<HashMap<AgentId, AgentEventBuffer>>>,

    /// Global event ordering buffer
    global_buffer: Arc<RwLock<GlobalEventBuffer>>,

    /// Configuration for ordering behavior
    config: OrderingConfig,
}

/// Configuration for event ordering
#[derive(Debug, Clone)]
pub struct OrderingConfig {
    /// How long to wait for late events (milliseconds)
    pub reorder_window_ms: u64,

    /// Maximum events to buffer per agent
    pub max_buffer_size: usize,

    /// How often to flush buffers (milliseconds)
    pub flush_interval_ms: u64,

    /// Watermark window for late event acceptance (milliseconds)
    pub watermark_window_ms: u64,

    /// Enable strict causality checking
    pub strict_causality: bool,

    /// Maximum clock skew tolerance (milliseconds)
    pub max_clock_skew_ms: u64,
}

impl Default for OrderingConfig {
    fn default() -> Self {
        Self {
            reorder_window_ms: 5000, // 5 second reorder window
            max_buffer_size: 1000,
            flush_interval_ms: 1000,   // Flush every second
            watermark_window_ms: 5000, // 5 second watermark window
            strict_causality: true,
            max_clock_skew_ms: 10000, // 10 second clock skew tolerance
        }
    }
}

/// Per-agent event buffer for ordering
#[derive(Debug)]
struct AgentEventBuffer {
    /// Agent this buffer belongs to
    #[allow(dead_code)]
    agent_id: AgentId,

    /// Events waiting to be ordered (timestamp -> event)
    pending_events: BTreeMap<Timestamp, Event>,

    /// Last processed event timestamp for this agent
    last_processed_timestamp: Option<Timestamp>,

    /// Causality chain state for this agent
    causality_state: CausalityState,

    /// Buffer statistics
    stats: BufferStats,
}

/// Global event buffer for cross-agent ordering
#[derive(Debug)]
struct GlobalEventBuffer {
    /// Events ordered by timestamp (global timeline)
    timeline: BTreeMap<Timestamp, Vec<Event>>,

    /// Latest processed timestamp
    last_processed_timestamp: Timestamp,

    /// Set of processed event IDs (idempotency)
    processed_event_ids: HashSet<EventId>,

    /// Pending events that might be out of order
    #[allow(dead_code)]
    pending_window: VecDeque<TimestampedEvent>,
}

/// Causality state tracking for an agent
#[derive(Debug, Clone)]
struct CausalityState {
    /// Expected next sequence number (for detecting gaps)
    #[allow(dead_code)]
    expected_sequence: u64,

    /// Events waiting for their dependencies
    #[allow(dead_code)]
    waiting_for_deps: HashMap<EventId, Event>,
}

/// Timestamped event with ordering metadata
#[derive(Debug, Clone)]
struct TimestampedEvent {
    #[allow(dead_code)]
    event: Event,
    #[allow(dead_code)]
    arrival_time: Timestamp,
    #[allow(dead_code)]
    processing_priority: f32,
}

/// Buffer statistics
#[derive(Debug, Default)]
struct BufferStats {
    events_buffered: u64,
    events_reordered: u64,
    events_dropped_late: u64,
    #[allow(dead_code)]
    max_buffer_size_reached: u64,
}

/// Result of event ordering operation
#[derive(Debug)]
pub struct OrderingResult {
    /// Events that are ready for processing (in correct order)
    pub ready_events: Vec<Event>,

    /// Events still waiting in buffers
    pub buffered_count: usize,

    /// Whether any reordering occurred
    pub reordering_occurred: bool,

    /// Detected ordering issues
    pub issues: Vec<OrderingIssue>,
}

/// Issues detected during event ordering
#[derive(Debug, Clone)]
pub enum OrderingIssue {
    /// Event arrived very late
    LateArrival {
        event_id: EventId,
        expected_time: Timestamp,
        actual_time: Timestamp,
        delay_ms: u64,
    },

    /// Missing event in causality chain
    MissingCausalEvent {
        waiting_event: EventId,
        missing_dependency: EventId,
    },

    /// Clock skew detected between agents
    ClockSkew { agent_id: AgentId, skew_ms: i64 },

    /// Buffer overflow - events dropped
    BufferOverflow {
        agent_id: AgentId,
        dropped_count: usize,
    },

    /// Duplicate event (already processed or buffered)
    DuplicateEvent { event_id: EventId },

    /// Late event dropped outside watermark window
    LateEventDropped {
        event_id: EventId,
        watermark_time: Timestamp,
        actual_time: Timestamp,
        delay_ms: u64,
    },
}

impl EventOrderingEngine {
    /// Create a new event ordering engine
    pub fn new(config: OrderingConfig) -> Self {
        Self {
            agent_buffers: Arc::new(RwLock::new(HashMap::new())),
            global_buffer: Arc::new(RwLock::new(GlobalEventBuffer::new())),
            config,
        }
    }

    /// Process an incoming event (potentially out of order)
    pub async fn process_event(&self, event: Event) -> GraphResult<OrderingResult> {
        let mut result = OrderingResult {
            ready_events: Vec::new(),
            buffered_count: 0,
            reordering_occurred: false,
            issues: Vec::new(),
        };

        // Idempotency: skip duplicates
        if self.is_duplicate_event(event.id).await {
            result
                .issues
                .push(OrderingIssue::DuplicateEvent { event_id: event.id });
            return Ok(result);
        }

        // Watermark: reject events that are too far in the past
        if self.is_late_beyond_watermark(&event).await? {
            let global = self.global_buffer.read().await;
            let watermark_time = global
                .last_processed_timestamp
                .saturating_sub(self.config.watermark_window_ms * 1_000_000);
            let delay = watermark_time.saturating_sub(event.timestamp);
            result.issues.push(OrderingIssue::LateEventDropped {
                event_id: event.id,
                watermark_time,
                actual_time: event.timestamp,
                delay_ms: delay / 1_000_000,
            });
            return Ok(result);
        }

        // Step 1: Add event to agent-specific buffer
        {
            let mut buffers = self.agent_buffers.write().await;
            let agent_buffer = buffers
                .entry(event.agent_id)
                .or_insert_with(|| AgentEventBuffer::new(event.agent_id));

            // Check for buffer overflow
            if agent_buffer.pending_events.len() >= self.config.max_buffer_size {
                result.issues.push(OrderingIssue::BufferOverflow {
                    agent_id: event.agent_id,
                    dropped_count: 1,
                });
                agent_buffer.stats.events_dropped_late += 1;
                return Ok(result);
            }

            // Check for late arrival
            if let Some(last_ts) = agent_buffer.last_processed_timestamp {
                if event.timestamp < last_ts {
                    let delay = last_ts - event.timestamp;
                    if delay > self.config.reorder_window_ms * 1_000_000 {
                        // Convert to nanoseconds
                        result.issues.push(OrderingIssue::LateArrival {
                            event_id: event.id,
                            expected_time: last_ts,
                            actual_time: event.timestamp,
                            delay_ms: delay / 1_000_000, // Convert to milliseconds
                        });
                    }
                    result.reordering_occurred = true;
                    agent_buffer.stats.events_reordered += 1;
                }
            }

            // Enforce causality: hold until dependencies are processed
            if self.config.strict_causality && !self.dependencies_satisfied(&event).await {
                for &dep_id in &event.causality_chain {
                    result.issues.push(OrderingIssue::MissingCausalEvent {
                        waiting_event: event.id,
                        missing_dependency: dep_id,
                    });
                }
                agent_buffer
                    .causality_state
                    .waiting_for_deps
                    .insert(event.id, event.clone());
            } else {
                // Add to buffer
                agent_buffer
                    .pending_events
                    .insert(event.timestamp, event.clone());
                agent_buffer.stats.events_buffered += 1;
            }
        }

        // Step 2: Check if we can process any buffered events
        let ready_from_agent = self.try_flush_agent_buffer(event.agent_id).await?;
        result.ready_events.extend(ready_from_agent);

        // Step 3: Update global ordering
        self.update_global_timeline(&result.ready_events).await;

        // Step 4: Check for causality violations
        if self.config.strict_causality {
            result
                .issues
                .extend(self.check_causality_violations(&event).await?);
        }

        // Step 5: Update buffer counts
        {
            let buffers = self.agent_buffers.read().await;
            result.buffered_count = buffers.values().map(|buf| buf.pending_events.len()).sum();
        }

        Ok(result)
    }

    async fn is_duplicate_event(&self, event_id: EventId) -> bool {
        let global = self.global_buffer.read().await;
        if global.processed_event_ids.contains(&event_id) {
            return true;
        }
        drop(global);

        let buffers = self.agent_buffers.read().await;
        for buffer in buffers.values() {
            if buffer.pending_events.values().any(|e| e.id == event_id) {
                return true;
            }
            if buffer
                .causality_state
                .waiting_for_deps
                .contains_key(&event_id)
            {
                return true;
            }
        }
        false
    }

    async fn is_late_beyond_watermark(&self, event: &Event) -> GraphResult<bool> {
        let global = self.global_buffer.read().await;
        if global.last_processed_timestamp == 0 {
            return Ok(false);
        }
        let watermark_time = global
            .last_processed_timestamp
            .saturating_sub(self.config.watermark_window_ms * 1_000_000);
        Ok(event.timestamp < watermark_time)
    }

    async fn dependencies_satisfied(&self, event: &Event) -> bool {
        if event.causality_chain.is_empty() {
            return true;
        }
        let global = self.global_buffer.read().await;
        for &dep_id in &event.causality_chain {
            if !global.processed_event_ids.contains(&dep_id) {
                return false;
            }
        }
        true
    }

    /// Try to flush events from an agent's buffer
    async fn try_flush_agent_buffer(&self, agent_id: AgentId) -> GraphResult<Vec<Event>> {
        let mut ready_events = Vec::new();
        let current_time = agent_db_core::types::current_timestamp();

        let mut buffers = self.agent_buffers.write().await;
        if let Some(agent_buffer) = buffers.get_mut(&agent_id) {
            // Move waiting events whose deps are now satisfied into pending
            if self.config.strict_causality
                && !agent_buffer.causality_state.waiting_for_deps.is_empty()
            {
                let waiting_ids: Vec<EventId> = agent_buffer
                    .causality_state
                    .waiting_for_deps
                    .keys()
                    .copied()
                    .collect();
                for event_id in waiting_ids {
                    if let Some(waiting_event) =
                        agent_buffer.causality_state.waiting_for_deps.get(&event_id)
                    {
                        if self.dependencies_satisfied(waiting_event).await {
                            let event = waiting_event.clone();
                            agent_buffer.pending_events.insert(event.timestamp, event);
                            agent_buffer
                                .causality_state
                                .waiting_for_deps
                                .remove(&event_id);
                        }
                    }
                }
            }

            let mut to_remove = Vec::new();

            // Find events that are ready to process (past reorder window)
            for (&timestamp, event) in &agent_buffer.pending_events {
                let age = current_time.saturating_sub(timestamp);
                let reorder_window_ns = self.config.reorder_window_ms * 1_000_000;

                // Event is ready if:
                // 1. It's older than reorder window, OR
                // 2. It's the next expected event in sequence
                let is_ready =
                    age >= reorder_window_ns || self.is_next_in_sequence(agent_buffer, event);

                if is_ready {
                    ready_events.push(event.clone());
                    to_remove.push(timestamp);
                    agent_buffer.last_processed_timestamp = Some(timestamp);
                }
            }

            // Remove processed events from buffer
            for timestamp in to_remove {
                agent_buffer.pending_events.remove(&timestamp);
            }
        }

        // Sort ready events by timestamp to maintain order
        ready_events.sort_by_key(|e| e.timestamp);

        Ok(ready_events)
    }

    /// Check if an event is the next expected in sequence
    fn is_next_in_sequence(&self, buffer: &AgentEventBuffer, event: &Event) -> bool {
        // Check timestamp ordering
        if let Some(last_ts) = buffer.last_processed_timestamp {
            event.timestamp >= last_ts
        } else {
            true // First event for this agent
        }
    }

    /// Update global event timeline
    async fn update_global_timeline(&self, events: &[Event]) {
        if events.is_empty() {
            return;
        }

        let mut global_buffer = self.global_buffer.write().await;

        for event in events {
            global_buffer
                .timeline
                .entry(event.timestamp)
                .or_insert_with(Vec::new)
                .push(event.clone());

            global_buffer.processed_event_ids.insert(event.id);

            global_buffer.last_processed_timestamp =
                global_buffer.last_processed_timestamp.max(event.timestamp);
        }
    }

    /// Check for causality violations
    async fn check_causality_violations(&self, event: &Event) -> GraphResult<Vec<OrderingIssue>> {
        let mut issues = Vec::new();

        // Check if causality chain references exist
        for &dep_id in &event.causality_chain {
            let dependency_exists = self.check_dependency_exists(dep_id).await;
            if !dependency_exists {
                issues.push(OrderingIssue::MissingCausalEvent {
                    waiting_event: event.id,
                    missing_dependency: dep_id,
                });
            }
        }

        Ok(issues)
    }

    /// Check if a dependency event has been processed
    async fn check_dependency_exists(&self, event_id: EventId) -> bool {
        let global_buffer = self.global_buffer.read().await;

        // Search through processed events
        for events_at_timestamp in global_buffer.timeline.values() {
            if events_at_timestamp.iter().any(|e| e.id == event_id) {
                return true;
            }
        }

        // Search through pending buffers
        let buffers = self.agent_buffers.read().await;
        for buffer in buffers.values() {
            if buffer.pending_events.values().any(|e| e.id == event_id) {
                return true;
            }
        }

        false
    }

    /// Force flush all buffers (useful for shutdown or testing)
    pub async fn flush_all_buffers(&self) -> GraphResult<Vec<Event>> {
        let mut all_events = Vec::new();

        let agent_ids: Vec<AgentId> = {
            let buffers = self.agent_buffers.read().await;
            buffers.keys().copied().collect()
        };

        for agent_id in agent_ids {
            let agent_events = self.force_flush_agent_buffer(agent_id).await?;
            all_events.extend(agent_events);
        }

        // Sort by timestamp
        all_events.sort_by_key(|e| e.timestamp);

        Ok(all_events)
    }

    /// Force flush a specific agent's buffer
    async fn force_flush_agent_buffer(&self, agent_id: AgentId) -> GraphResult<Vec<Event>> {
        let mut buffers = self.agent_buffers.write().await;
        if let Some(agent_buffer) = buffers.get_mut(&agent_id) {
            let events: Vec<Event> = agent_buffer.pending_events.values().cloned().collect();
            agent_buffer.pending_events.clear();
            Ok(events)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get ordering statistics
    pub async fn get_stats(&self) -> OrderingStats {
        let buffers = self.agent_buffers.read().await;
        let global_buffer = self.global_buffer.read().await;

        let total_buffered: usize = buffers.values().map(|buf| buf.pending_events.len()).sum();

        let total_processed: u64 = global_buffer
            .timeline
            .values()
            .map(|events| events.len() as u64)
            .sum();

        let total_reordered: u64 = buffers.values().map(|buf| buf.stats.events_reordered).sum();

        OrderingStats {
            total_agents: buffers.len(),
            total_buffered_events: total_buffered,
            total_processed_events: total_processed,
            total_reordered_events: total_reordered,
            oldest_buffered_event: self.get_oldest_buffered_timestamp(&buffers),
        }
    }

    fn get_oldest_buffered_timestamp(
        &self,
        buffers: &HashMap<AgentId, AgentEventBuffer>,
    ) -> Option<Timestamp> {
        buffers
            .values()
            .filter_map(|buf| buf.pending_events.keys().next().copied())
            .min()
    }
}

impl AgentEventBuffer {
    fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            pending_events: BTreeMap::new(),
            last_processed_timestamp: None,
            causality_state: CausalityState {
                expected_sequence: 0,
                waiting_for_deps: HashMap::new(),
            },
            stats: BufferStats::default(),
        }
    }
}

impl GlobalEventBuffer {
    fn new() -> Self {
        Self {
            timeline: BTreeMap::new(),
            last_processed_timestamp: 0,
            pending_window: VecDeque::new(),
            processed_event_ids: HashSet::new(),
        }
    }
}

/// Statistics about event ordering
#[derive(Debug)]
pub struct OrderingStats {
    pub total_agents: usize,
    pub total_buffered_events: usize,
    pub total_processed_events: u64,
    pub total_reordered_events: u64,
    pub oldest_buffered_event: Option<Timestamp>,
}
