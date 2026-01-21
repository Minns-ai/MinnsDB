//! Event buffering for high-throughput ingestion

use crate::core::Event;
use agent_db_core::error::{DatabaseError, DatabaseResult};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// High-performance event buffer for batching
pub struct EventBuffer {
    /// Buffered events
    events: VecDeque<Event>,
    
    /// Maximum buffer capacity
    capacity: usize,
    
    /// Buffer statistics
    stats: BufferStats,
    
    /// Configuration
    config: BufferConfig,
}

/// Buffer configuration
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Auto-flush when buffer reaches this size
    pub auto_flush_size: Option<usize>,
    
    /// Auto-flush after this duration
    pub auto_flush_interval: Option<Duration>,
    
    /// Drop oldest events when buffer is full (vs returning error)
    pub drop_on_full: bool,
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Total events added
    pub total_added: u64,
    
    /// Total events flushed
    pub total_flushed: u64,
    
    /// Total events dropped due to buffer full
    pub total_dropped: u64,
    
    /// Number of flush operations
    pub flush_count: u32,
    
    /// Last flush timestamp
    pub last_flush: Option<Instant>,
    
    /// Buffer creation time
    pub created_at: Instant,
}

impl Default for BufferStats {
    fn default() -> Self {
        Self {
            total_added: 0,
            total_flushed: 0,
            total_dropped: 0,
            flush_count: 0,
            last_flush: None,
            created_at: Instant::now(),
        }
    }
}

impl EventBuffer {
    /// Create new event buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            stats: BufferStats {
                created_at: Instant::now(),
                ..Default::default()
            },
            config: BufferConfig::default(),
        }
    }
    
    /// Create buffer with custom configuration
    pub fn with_config(capacity: usize, config: BufferConfig) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            stats: BufferStats {
                created_at: Instant::now(),
                ..Default::default()
            },
            config,
        }
    }
    
    /// Add event to buffer
    pub fn add(&mut self, event: Event) -> DatabaseResult<()> {
        // Check if we should auto-flush first
        if let Some(auto_size) = self.config.auto_flush_size {
            if self.events.len() >= auto_size {
                self.flush();
            }
        }
        
        // Check capacity
        if self.events.len() >= self.capacity {
            if self.config.drop_on_full {
                // Drop oldest event
                self.events.pop_front();
                self.stats.total_dropped += 1;
            } else {
                return Err(DatabaseError::ResourceExhausted(format!(
                    "Event buffer full (capacity: {})", 
                    self.capacity
                )));
            }
        }
        
        self.events.push_back(event);
        self.stats.total_added += 1;
        
        Ok(())
    }
    
    /// Add multiple events in batch
    pub fn add_batch(&mut self, events: Vec<Event>) -> DatabaseResult<()> {
        for event in events {
            self.add(event)?;
        }
        Ok(())
    }
    
    /// Flush all buffered events and return them
    pub fn flush(&mut self) -> Vec<Event> {
        let events: Vec<Event> = self.events.drain(..).collect();
        let count = events.len() as u64;
        
        self.stats.total_flushed += count;
        self.stats.flush_count += 1;
        self.stats.last_flush = Some(Instant::now());
        
        events
    }
    
    /// Drain specific number of events from buffer
    pub fn drain(&mut self, count: usize) -> Vec<Event> {
        let actual_count = count.min(self.events.len());
        let events: Vec<Event> = self.events.drain(..actual_count).collect();
        
        self.stats.total_flushed += events.len() as u64;
        if !events.is_empty() {
            self.stats.flush_count += 1;
            self.stats.last_flush = Some(Instant::now());
        }
        
        events
    }
    
    /// Get number of events currently in buffer
    pub fn len(&self) -> usize {
        self.events.len()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.events.len() >= self.capacity
    }
    
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get buffer utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        self.events.len() as f32 / self.capacity as f32
    }
    
    /// Get buffer statistics
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }
    
    /// Check if auto-flush should trigger based on time
    pub fn should_auto_flush_time(&self) -> bool {
        if let Some(interval) = self.config.auto_flush_interval {
            if let Some(last_flush) = self.stats.last_flush {
                return last_flush.elapsed() >= interval;
            } else {
                // No previous flush, check against creation time
                return self.stats.created_at.elapsed() >= interval;
            }
        }
        false
    }
    
    /// Check if auto-flush should trigger based on size
    pub fn should_auto_flush_size(&self) -> bool {
        if let Some(auto_size) = self.config.auto_flush_size {
            return self.events.len() >= auto_size;
        }
        false
    }
    
    /// Check if auto-flush should trigger (either time or size)
    pub fn should_auto_flush(&self) -> bool {
        self.should_auto_flush_time() || self.should_auto_flush_size()
    }
    
    /// Clear all events from buffer without returning them
    pub fn clear(&mut self) {
        let count = self.events.len() as u64;
        self.events.clear();
        self.stats.total_dropped += count;
    }
    
    /// Resize buffer capacity (may drop events if new capacity is smaller)
    pub fn resize(&mut self, new_capacity: usize) -> Vec<Event> {
        let mut dropped = Vec::new();
        
        if new_capacity < self.events.len() {
            // Need to drop some events
            let to_drop = self.events.len() - new_capacity;
            dropped = self.events.drain(..to_drop).collect();
            self.stats.total_dropped += dropped.len() as u64;
        }
        
        self.capacity = new_capacity;
        self.events.shrink_to_fit();
        
        dropped
    }
    
    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() 
            + self.events.iter().map(|e| e.size_bytes()).sum::<usize>()
            + self.events.capacity() * std::mem::size_of::<Event>()
    }
    
    /// Peek at oldest event without removing it
    pub fn peek_front(&self) -> Option<&Event> {
        self.events.front()
    }
    
    /// Peek at newest event without removing it
    pub fn peek_back(&self) -> Option<&Event> {
        self.events.back()
    }
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            auto_flush_size: Some(1000),                      // Auto-flush at 1k events
            auto_flush_interval: Some(Duration::from_millis(100)), // Auto-flush every 100ms
            drop_on_full: false,                              // Error on full by default
        }
    }
}

/// Ring buffer implementation for high-performance scenarios
pub struct RingEventBuffer {
    /// Fixed-size buffer
    buffer: Vec<Option<Event>>,
    
    /// Write position
    write_pos: usize,
    
    /// Read position  
    read_pos: usize,
    
    /// Number of events in buffer
    count: usize,
    
    /// Buffer capacity
    capacity: usize,
    
    /// Whether buffer has wrapped around
    wrapped: bool,
    
    /// Statistics
    stats: BufferStats,
}

impl RingEventBuffer {
    /// Create new ring buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        
        Self {
            buffer,
            write_pos: 0,
            read_pos: 0,
            count: 0,
            capacity,
            wrapped: false,
            stats: BufferStats {
                created_at: Instant::now(),
                ..Default::default()
            },
        }
    }
    
    /// Add event to ring buffer (overwrites oldest if full)
    pub fn add(&mut self, event: Event) -> Option<Event> {
        let old_event = self.buffer[self.write_pos].take();
        self.buffer[self.write_pos] = Some(event);
        
        // Update counters
        if old_event.is_some() {
            self.stats.total_dropped += 1;
        }
        
        self.stats.total_added += 1;
        
        // Advance write position
        self.write_pos = (self.write_pos + 1) % self.capacity;
        
        // Update count and wrapped flag
        if self.count < self.capacity {
            self.count += 1;
        } else {
            self.wrapped = true;
            // Move read position when we overwrite
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
        
        old_event
    }
    
    /// Read next event from ring buffer
    pub fn read(&mut self) -> Option<Event> {
        if self.count == 0 {
            return None;
        }
        
        let event = self.buffer[self.read_pos].take();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.count -= 1;
        
        if let Some(_) = &event {
            self.stats.total_flushed += 1;
        }
        
        event
    }
    
    /// Read up to `max_count` events
    pub fn read_batch(&mut self, max_count: usize) -> Vec<Event> {
        let mut events = Vec::new();
        
        for _ in 0..max_count {
            if let Some(event) = self.read() {
                events.push(event);
            } else {
                break;
            }
        }
        
        if !events.is_empty() {
            self.stats.flush_count += 1;
            self.stats.last_flush = Some(Instant::now());
        }
        
        events
    }
    
    /// Get number of events in buffer
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get statistics
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    fn create_test_event(agent_id: u64) -> Event {
        use crate::core::*;
        use std::collections::HashMap;
        
        Event::new(
            agent_id,
            "test_agent".to_string(), // agent_type
            1, // session_id
            EventType::Action {
                action_name: "test".to_string(),
                parameters: json!({}),
                outcome: ActionOutcome::Success { result: json!(true) },
                duration_ns: 1000,
            },
            EventContext::new(
                EnvironmentState {
                    variables: HashMap::new(),
                    spatial: None,
                    temporal: TemporalContext {
                        time_of_day: None,
                        deadlines: Vec::new(),
                        patterns: Vec::new(),
                    },
                },
                Vec::new(),
                ResourceState {
                    computational: ComputationalResources {
                        cpu_percent: 50.0,
                        memory_bytes: 1024,
                        storage_bytes: 1024,
                        network_bandwidth: 100,
                    },
                    external: HashMap::new(),
                },
            ),
        )
    }
    
    #[test]
    fn test_buffer_basic_operations() {
        let mut buffer = EventBuffer::new(10);
        
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        
        // Add some events
        for i in 0..5 {
            let event = create_test_event(i);
            assert!(buffer.add(event).is_ok());
        }
        
        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.utilization(), 0.5);
        
        // Flush events
        let events = buffer.flush();
        assert_eq!(events.len(), 5);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_buffer_overflow_error() {
        let mut buffer = EventBuffer::new(3);
        
        // Fill buffer
        for i in 0..3 {
            let event = create_test_event(i);
            assert!(buffer.add(event).is_ok());
        }
        
        assert!(buffer.is_full());
        
        // Try to add one more (should fail)
        let overflow_event = create_test_event(99);
        assert!(buffer.add(overflow_event).is_err());
        
        assert_eq!(buffer.len(), 3);
    }
    
    #[test]
    fn test_buffer_drop_on_full() {
        let config = BufferConfig {
            drop_on_full: true,
            auto_flush_size: None,
            auto_flush_interval: None,
        };
        let mut buffer = EventBuffer::with_config(3, config);
        
        // Fill buffer
        for i in 0..3 {
            let event = create_test_event(i);
            assert!(buffer.add(event).is_ok());
        }
        
        assert!(buffer.is_full());
        assert_eq!(buffer.stats().total_dropped, 0);
        
        // Add one more (should drop oldest)
        let overflow_event = create_test_event(99);
        assert!(buffer.add(overflow_event).is_ok());
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.stats().total_dropped, 1);
    }
    
    #[test]
    fn test_buffer_auto_flush_size() {
        let config = BufferConfig {
            auto_flush_size: Some(3),
            auto_flush_interval: None,
            drop_on_full: false,
        };
        let mut buffer = EventBuffer::with_config(10, config);
        
        // Add events up to auto-flush size
        for i in 0..2 {
            let event = create_test_event(i);
            buffer.add(event).unwrap();
        }
        
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.stats().flush_count, 0);
        
        // Adding third should trigger auto-flush
        let event = create_test_event(2);
        buffer.add(event).unwrap();
        
        // Buffer should be empty after auto-flush, then have 1 new event
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.stats().flush_count, 1);
    }
    
    #[test]
    fn test_buffer_drain() {
        let mut buffer = EventBuffer::new(10);
        
        // Add 5 events
        for i in 0..5 {
            let event = create_test_event(i);
            buffer.add(event).unwrap();
        }
        
        // Drain 3 events
        let drained = buffer.drain(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(buffer.len(), 2);
        
        // Drain more than available
        let drained = buffer.drain(10);
        assert_eq!(drained.len(), 2);
        assert_eq!(buffer.len(), 0);
    }
    
    #[test]
    fn test_ring_buffer() {
        let mut ring = RingEventBuffer::new(3);
        
        assert_eq!(ring.len(), 0);
        assert!(ring.is_empty());
        
        // Add events
        for i in 0..3 {
            let event = create_test_event(i);
            let dropped = ring.add(event);
            assert!(dropped.is_none()); // No events dropped yet
        }
        
        assert_eq!(ring.len(), 3);
        assert!(ring.is_full());
        
        // Add one more (should drop oldest)
        let event = create_test_event(99);
        let dropped = ring.add(event);
        assert!(dropped.is_some());
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.stats().total_dropped, 1);
        
        // Read events
        let events = ring.read_batch(5);
        assert_eq!(events.len(), 3);
        assert!(ring.is_empty());
    }
    
    #[test]
    fn test_buffer_statistics() {
        let mut buffer = EventBuffer::new(10);
        
        // Add some events
        for i in 0..5 {
            let event = create_test_event(i);
            buffer.add(event).unwrap();
        }
        
        assert_eq!(buffer.stats().total_added, 5);
        assert_eq!(buffer.stats().total_flushed, 0);
        
        // Flush
        let _ = buffer.flush();
        
        assert_eq!(buffer.stats().total_flushed, 5);
        assert_eq!(buffer.stats().flush_count, 1);
        assert!(buffer.stats().last_flush.is_some());
    }
    
    #[test]
    fn test_buffer_resize() {
        let mut buffer = EventBuffer::new(10);
        
        // Add 8 events
        for i in 0..8 {
            let event = create_test_event(i);
            buffer.add(event).unwrap();
        }
        
        assert_eq!(buffer.len(), 8);
        
        // Resize to smaller capacity
        let dropped = buffer.resize(5);
        assert_eq!(dropped.len(), 3); // 3 events should be dropped
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.capacity(), 5);
        assert_eq!(buffer.stats().total_dropped, 3);
    }
}