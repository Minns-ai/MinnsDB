//! Sequence Tracker — caller-provided `seq_no` validation per ordering domain.
//!
//! The server infers an ordering domain from each operation (e.g. `session:42`,
//! `ledger:alice_bob`) and validates the caller's `seq_no` against tracked
//! domain state. This detects duplicates, gaps, and out-of-order delivery.
//!
//! **Strict domains** (ledger, state, pref, tree) never silently skip — the
//! domain blocks until the gap is filled.
//!
//! **Permissive domains** (session) apply best-effort ordering; after a timeout,
//! held events are applied anyway (graph mutations are append-only and tolerant).
//!
//! NOTE: Most types/methods here are ready infrastructure that will be consumed
//! when API request schemas gain optional `seq_no` fields.

use dashmap::DashMap;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::time::Instant;

/// Result of validating a caller-provided seq_no.
#[allow(dead_code)]
#[derive(Debug)]
pub enum SeqCheckResult {
    /// seq_no matches `last_applied + 1` — ready to apply.
    Ready,
    /// seq_no was already applied.
    Duplicate,
    /// seq_no is older than last_applied (superseded).
    Stale,
    /// seq_no is ahead of expected — gap detected.
    Gap { expected: u64, received: u64 },
    /// Domain is in waiting_for_resend state (strict domains only).
    DomainWaiting { expected: u64, held_count: usize },
}

/// Strictness classification for an ordering domain.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum DomainStrictness {
    /// Accounting-sensitive: never skip, block on gap.
    Strict,
    /// Order-tolerant: apply best-effort, allow out-of-order after timeout.
    Permissive,
}

/// An event held due to a sequence gap.
#[allow(dead_code)]
pub struct HeldEvent {
    pub seq_no: u64,
    pub payload: HeldPayload,
    pub held_at: Instant,
}

/// Payload of a held event.
#[allow(dead_code)]
pub enum HeldPayload {
    ProcessEvent {
        event: agent_db_events::Event,
        enable_semantic: Option<bool>,
    },
}

/// State for a single ordering domain.
pub struct DomainState {
    /// Highest seq_no that has been fully applied.
    pub last_applied: u64,
    /// Events held due to gap (waiting for prior seq_no to arrive).
    #[allow(dead_code)]
    pub held: BTreeMap<u64, HeldEvent>,
    /// Recently applied seq_nos for duplicate detection (ring buffer, last 256).
    recent_applied: VecDeque<u64>,
    /// O(1) lookup companion for recent_applied.
    recent_applied_set: HashSet<u64>,
    /// Whether this domain is in "waiting" state (gap timeout exceeded).
    #[allow(dead_code)]
    pub waiting_for_resend: Option<WaitingState>,
    /// Last time any event was applied to this domain.
    pub last_activity: Instant,
}

/// Tracks the state when a strict domain is waiting for a missing seq_no.
#[allow(dead_code)]
pub struct WaitingState {
    /// The seq_no we are waiting for.
    pub expected_seq_no: u64,
    /// When the gap was first detected.
    pub gap_detected_at: Instant,
}

impl Default for DomainState {
    fn default() -> Self {
        Self {
            last_applied: 0,
            held: BTreeMap::new(),
            recent_applied: VecDeque::with_capacity(256),
            recent_applied_set: HashSet::with_capacity(256),
            waiting_for_resend: None,
            last_activity: Instant::now(),
        }
    }
}

/// Per-domain sequence validation and gap detection.
pub struct SequenceTracker {
    domain_state: DashMap<String, DomainState>,
}

impl SequenceTracker {
    pub fn new() -> Self {
        Self {
            domain_state: DashMap::new(),
        }
    }

    /// Infer domain strictness from the domain key prefix.
    #[allow(dead_code)]
    pub fn strictness(domain: &str) -> DomainStrictness {
        if domain.starts_with("ledger:")
            || domain.starts_with("state:")
            || domain.starts_with("pref:")
            || domain.starts_with("tree:")
        {
            DomainStrictness::Strict
        } else {
            DomainStrictness::Permissive
        }
    }

    /// Validate a caller-provided seq_no against the domain's tracked state.
    #[allow(dead_code)]
    pub fn check(&self, domain: &str, seq_no: u64) -> SeqCheckResult {
        // Fast path: try get_mut first to avoid String allocation for existing domains
        let mut state = if let Some(s) = self.domain_state.get_mut(domain) {
            s
        } else {
            self.domain_state.entry(domain.to_string()).or_default()
        };

        // If domain is in waiting_for_resend state (strict domains only)
        if let Some(ref waiting) = state.waiting_for_resend {
            if seq_no == waiting.expected_seq_no {
                // This IS the missing event — clear waiting state, return Ready
                state.waiting_for_resend = None;
                return SeqCheckResult::Ready;
            }
            return SeqCheckResult::DomainWaiting {
                expected: waiting.expected_seq_no,
                held_count: state.held.len(),
            };
        }

        // Duplicate: already applied (O(1) HashSet lookup)
        if state.recent_applied_set.contains(&seq_no) {
            return SeqCheckResult::Duplicate;
        }

        // Stale: older than last_applied
        if seq_no <= state.last_applied && state.last_applied > 0 {
            return SeqCheckResult::Stale;
        }

        // Expected: next in sequence
        if seq_no == state.last_applied + 1 || state.last_applied == 0 {
            return SeqCheckResult::Ready;
        }

        // Gap: seq_no > last_applied + 1
        SeqCheckResult::Gap {
            expected: state.last_applied + 1,
            received: seq_no,
        }
    }

    /// Mark a seq_no as applied for a domain.
    #[allow(dead_code)]
    pub fn mark_applied(&self, domain: &str, seq_no: u64) {
        // Fast path: try get_mut first to avoid String allocation for existing domains
        let mut state = if let Some(s) = self.domain_state.get_mut(domain) {
            s
        } else {
            self.domain_state.entry(domain.to_string()).or_default()
        };
        state.last_applied = seq_no;
        state.last_activity = Instant::now();

        // Add to recent_applied ring buffer + set
        if state.recent_applied.len() >= 256 {
            if let Some(old) = state.recent_applied.pop_front() {
                state.recent_applied_set.remove(&old);
            }
        }
        state.recent_applied.push_back(seq_no);
        state.recent_applied_set.insert(seq_no);

        // Clear waiting state if this was the missing event
        if let Some(ref waiting) = state.waiting_for_resend {
            if seq_no == waiting.expected_seq_no {
                state.waiting_for_resend = None;
            }
        }
    }

    /// Hold an event due to a sequence gap.
    #[allow(dead_code)]
    pub fn hold_event(&self, domain: &str, seq_no: u64, payload: HeldPayload) {
        let mut state = self.domain_state.entry(domain.to_string()).or_default();
        state.held.insert(
            seq_no,
            HeldEvent {
                seq_no,
                payload,
                held_at: Instant::now(),
            },
        );

        // For strict domains, enter waiting state if not already
        if Self::strictness(domain) == DomainStrictness::Strict
            && state.waiting_for_resend.is_none()
        {
            state.waiting_for_resend = Some(WaitingState {
                expected_seq_no: state.last_applied + 1,
                gap_detected_at: Instant::now(),
            });
        }
    }

    /// Flush held events that are now ready (seq_no == last_applied + 1).
    /// Returns the payloads in order for the caller to apply.
    #[allow(dead_code)]
    pub fn flush_held(&self, domain: &str) -> Vec<HeldPayload> {
        let mut state = match self.domain_state.get_mut(domain) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let mut flushed = Vec::new();

        loop {
            let next_expected = state.last_applied + 1;
            if let Some(held) = state.held.remove(&next_expected) {
                state.last_applied = next_expected;
                if state.recent_applied.len() >= 256 {
                    if let Some(old) = state.recent_applied.pop_front() {
                        state.recent_applied_set.remove(&old);
                    }
                }
                state.recent_applied.push_back(next_expected);
                state.recent_applied_set.insert(next_expected);
                state.last_activity = Instant::now();
                flushed.push(held.payload);
            } else {
                break;
            }
        }

        flushed
    }

    /// Evict idle domain state (for unbounded growth prevention).
    /// Returns the number of domains evicted.
    #[allow(dead_code)]
    pub fn evict_idle(&self, max_idle_secs: u64) -> usize {
        let now = Instant::now();
        let mut evicted = 0;
        self.domain_state.retain(|_key, state| {
            // Don't evict strict domains with waiting state
            if state.waiting_for_resend.is_some() {
                return true;
            }
            if now.duration_since(state.last_activity).as_secs() > max_idle_secs {
                evicted += 1;
                false
            } else {
                true
            }
        });
        evicted
    }

    /// Number of tracked domains.
    pub fn domain_count(&self) -> usize {
        self.domain_state.len()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_ready_first_event() {
        let tracker = SequenceTracker::new();
        match tracker.check("session:1", 1) {
            SeqCheckResult::Ready => {},
            other => panic!("Expected Ready, got {:?}", other),
        }
    }

    #[test]
    fn seq_duplicate_detection() {
        let tracker = SequenceTracker::new();
        tracker.mark_applied("session:1", 1);
        match tracker.check("session:1", 1) {
            SeqCheckResult::Duplicate => {},
            other => panic!("Expected Duplicate, got {:?}", other),
        }
    }

    #[test]
    fn seq_stale_detection() {
        let tracker = SequenceTracker::new();
        tracker.mark_applied("session:1", 1);
        tracker.mark_applied("session:1", 2);
        tracker.mark_applied("session:1", 3);
        match tracker.check("session:1", 1) {
            // seq_no 1 is in recent_applied_set, so Duplicate is returned
            SeqCheckResult::Duplicate => {},
            other => panic!("Expected Duplicate, got {:?}", other),
        }
    }

    #[test]
    fn seq_gap_detection() {
        let tracker = SequenceTracker::new();
        tracker.mark_applied("session:1", 1);
        match tracker.check("session:1", 5) {
            SeqCheckResult::Gap {
                expected: 2,
                received: 5,
            } => {},
            other => panic!("Expected Gap{{2, 5}}, got {:?}", other),
        }
    }

    fn make_test_event(id: u128) -> agent_db_events::Event {
        agent_db_events::Event {
            id,
            timestamp: 0,
            agent_id: 1,
            agent_type: String::new(),
            session_id: 1,
            event_type: agent_db_events::core::EventType::Observation {
                observation_type: "test".to_string(),
                data: serde_json::json!({}),
                source: "test".to_string(),
                confidence: 1.0,
            },
            causality_chain: vec![],
            context: agent_db_events::core::EventContext::default(),
            metadata: std::collections::HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        }
    }

    #[test]
    fn seq_flush_held() {
        let tracker = SequenceTracker::new();
        tracker.mark_applied("session:1", 1);

        // Hold events 3 and 2 (out of order)
        tracker.hold_event(
            "session:1",
            3,
            HeldPayload::ProcessEvent {
                event: make_test_event(3),
                enable_semantic: None,
            },
        );
        tracker.hold_event(
            "session:1",
            2,
            HeldPayload::ProcessEvent {
                event: make_test_event(2),
                enable_semantic: None,
            },
        );

        // Flush: should get events 2, 3 in order
        let flushed = tracker.flush_held("session:1");
        assert_eq!(flushed.len(), 2);

        // Verify last_applied is now 3
        let state = tracker.domain_state.get("session:1").unwrap();
        assert_eq!(state.last_applied, 3);
    }

    #[test]
    fn seq_domain_strictness() {
        assert_eq!(
            SequenceTracker::strictness("ledger:alice_bob"),
            DomainStrictness::Strict
        );
        assert_eq!(
            SequenceTracker::strictness("state:order"),
            DomainStrictness::Strict
        );
        assert_eq!(
            SequenceTracker::strictness("pref:alice"),
            DomainStrictness::Strict
        );
        assert_eq!(
            SequenceTracker::strictness("tree:org"),
            DomainStrictness::Strict
        );
        assert_eq!(
            SequenceTracker::strictness("session:42"),
            DomainStrictness::Permissive
        );
    }

    #[test]
    fn seq_evict_idle() {
        let tracker = SequenceTracker::new();
        tracker.mark_applied("session:1", 1);
        tracker.mark_applied("session:2", 1);

        // Domains are fresh (last_activity ~= now), so 0-sec idle won't evict
        // because duration_since().as_secs() == 0 and the check is > max_idle_secs.
        let evicted = tracker.evict_idle(0);
        assert_eq!(evicted, 0);
        assert_eq!(tracker.domain_count(), 2);

        // Manually backdate last_activity to trigger eviction
        for mut entry in tracker.domain_state.iter_mut() {
            entry.last_activity = Instant::now() - std::time::Duration::from_secs(10);
        }
        let evicted = tracker.evict_idle(5);
        assert_eq!(evicted, 2);
        assert_eq!(tracker.domain_count(), 0);
    }

    #[test]
    fn seq_duplicate_set_o1() {
        // Verify that HashSet provides O(1) duplicate check
        let tracker = SequenceTracker::new();
        for i in 1..=300u64 {
            tracker.mark_applied("session:1", i);
        }
        // Ring buffer capped at 256 — seq_no 1..44 should have been evicted
        let state = tracker.domain_state.get("session:1").unwrap();
        assert_eq!(state.recent_applied.len(), 256);
        assert_eq!(state.recent_applied_set.len(), 256);

        // seq_no 45 should still be in the set (first non-evicted)
        assert!(state.recent_applied_set.contains(&45));
        // seq_no 1 should have been evicted
        assert!(!state.recent_applied_set.contains(&1));
    }
}
