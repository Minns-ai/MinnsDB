// crates/agent-db-graph/src/structures/types.rs
//
// Type aliases, constants, and simple enums shared across the structures module.

use serde::{Deserialize, Serialize};

/// Truncate a string to `max_len` characters, appending "..." if truncated.
pub(crate) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Unique identifier for graph nodes
pub type NodeId = u64;

/// Unique identifier for graph edges
pub type EdgeId = u64;

/// Weight type for edges (can represent similarity, causality strength, etc.)
pub type EdgeWeight = f32;

/// Traversal direction for directed graph queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Follow outgoing edges only.
    Out,
    /// Follow incoming edges only.
    In,
    /// Follow both outgoing and incoming edges.
    Both,
}

/// Depth specification for traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Depth {
    /// Exactly N hops.
    Fixed(u32),
    /// Between min and max hops (inclusive).
    Range(u32, u32),
    /// No depth limit (bounded by budgets or graph size).
    Unbounded,
}

impl Depth {
    /// Maximum depth bound. `None` for `Unbounded`.
    pub fn max_depth(&self) -> Option<u32> {
        match self {
            Depth::Fixed(n) => Some(*n),
            Depth::Range(_, max) => Some(*max),
            Depth::Unbounded => None,
        }
    }

    /// Minimum depth bound. `0` for `Unbounded`.
    pub fn min_depth(&self) -> u32 {
        match self {
            Depth::Fixed(n) => *n,
            Depth::Range(min, _) => *min,
            Depth::Unbounded => 0,
        }
    }

    /// Validate depth specification. Returns error if Range has min > max.
    pub fn validate(&self) -> crate::GraphResult<()> {
        match self {
            Depth::Range(min, max) if min > max => Err(crate::GraphError::InvalidQuery(format!(
                "Depth range min ({}) > max ({})",
                min, max
            ))),
            _ => Ok(()),
        }
    }
}

/// Unique identifier for goal buckets (semantic partitions)
pub type GoalBucketId = u64;

/// Maximum number of shards. Agent/event IDs are modded by this value
/// so the shard count stays bounded regardless of entity cardinality.
pub const NUM_SHARDS: u64 = 256;
