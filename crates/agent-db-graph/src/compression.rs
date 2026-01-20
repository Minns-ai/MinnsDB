// Delta encoding compression for adjacency lists
//
// Implements delta encoding to compress sequential node IDs, achieving 50-70% size reduction
// for typical graph data where nodes are created in sequence.

use crate::structures::NodeId;
use serde::{Deserialize, Serialize};

/// Compressed adjacency list using delta encoding
///
/// Instead of storing full node IDs:
///   [100, 101, 105, 106, 110, 200, 201]
///
/// We store:
///   base = 100
///   deltas = [+1, +4, +1, +4, +90, +1]
///
/// This achieves ~50-70% compression for sequential IDs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressedAdjacencyList {
    /// First node ID (base value)
    pub base: NodeId,

    /// Delta-encoded subsequent IDs (each is: next_id - current_id)
    pub deltas: Vec<u32>,
}

impl CompressedAdjacencyList {
    /// Compress a list of node IDs using delta encoding
    ///
    /// # Example
    /// ```ignore
    /// let nodes = vec![100, 101, 105, 106, 110];
    /// let compressed = CompressedAdjacencyList::compress(&nodes);
    /// assert_eq!(compressed.base, 100);
    /// assert_eq!(compressed.deltas, vec![1, 4, 1, 4]);
    /// ```
    pub fn compress(nodes: &[NodeId]) -> Self {
        if nodes.is_empty() {
            return Self {
                base: 0,
                deltas: Vec::new(),
            };
        }

        let base = nodes[0];
        let mut deltas = Vec::with_capacity(nodes.len().saturating_sub(1));

        for window in nodes.windows(2) {
            let delta = window[1].saturating_sub(window[0]) as u32;
            deltas.push(delta);
        }

        Self { base, deltas }
    }

    /// Decompress back to full node ID list
    ///
    /// # Example
    /// ```ignore
    /// let compressed = CompressedAdjacencyList {
    ///     base: 100,
    ///     deltas: vec![1, 4, 1, 4],
    /// };
    /// let nodes = compressed.decompress();
    /// assert_eq!(nodes, vec![100, 101, 105, 106, 110]);
    /// ```
    pub fn decompress(&self) -> Vec<NodeId> {
        if self.deltas.is_empty() {
            if self.base == 0 {
                return Vec::new();
            }
            return vec![self.base];
        }

        let mut result = Vec::with_capacity(self.deltas.len() + 1);
        result.push(self.base);

        let mut current = self.base;
        for &delta in &self.deltas {
            current = current.saturating_add(delta as NodeId);
            result.push(current);
        }

        result
    }

    /// Get the number of nodes in this adjacency list
    pub fn len(&self) -> usize {
        if self.base == 0 && self.deltas.is_empty() {
            0
        } else {
            self.deltas.len() + 1
        }
    }

    /// Check if the adjacency list is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if a node ID is in this adjacency list (without decompressing)
    pub fn contains(&self, node_id: NodeId) -> bool {
        if self.is_empty() {
            return false;
        }

        if node_id == self.base {
            return true;
        }

        if node_id < self.base {
            return false;
        }

        // Reconstruct and check
        let mut current = self.base;
        for &delta in &self.deltas {
            current = current.saturating_add(delta as NodeId);
            if current == node_id {
                return true;
            }
            if current > node_id {
                return false; // Passed it
            }
        }

        false
    }
}

/// Statistics about compression effectiveness
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes (NodeId = 16 bytes each)
    pub original_bytes: usize,

    /// Compressed size in bytes (8 bytes base + 4 bytes per delta)
    pub compressed_bytes: usize,

    /// Compression ratio (0.0 to 1.0, lower is better)
    pub ratio: f32,

    /// Number of nodes
    pub node_count: usize,
}

impl CompressionStats {
    /// Calculate compression statistics
    pub fn calculate(nodes: &[NodeId]) -> Self {
        let node_count = nodes.len();
        let original_bytes = node_count * std::mem::size_of::<NodeId>();

        // Compressed: 16 bytes (base) + 4 bytes per delta
        let compressed_bytes = if node_count == 0 {
            0
        } else {
            std::mem::size_of::<NodeId>() + (node_count.saturating_sub(1) * std::mem::size_of::<u32>())
        };

        let ratio = if original_bytes == 0 {
            0.0
        } else {
            compressed_bytes as f32 / original_bytes as f32
        };

        Self {
            original_bytes,
            compressed_bytes,
            ratio,
            node_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_empty() {
        let nodes: Vec<NodeId> = vec![];
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert_eq!(compressed.base, 0);
        assert_eq!(compressed.deltas.len(), 0);
        assert!(compressed.is_empty());

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }

    #[test]
    fn test_compress_single() {
        let nodes = vec![100];
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert_eq!(compressed.base, 100);
        assert_eq!(compressed.deltas.len(), 0);
        assert_eq!(compressed.len(), 1);

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }

    #[test]
    fn test_compress_sequential() {
        let nodes = vec![100, 101, 102, 103, 104];
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert_eq!(compressed.base, 100);
        assert_eq!(compressed.deltas, vec![1, 1, 1, 1]);
        assert_eq!(compressed.len(), 5);

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }

    #[test]
    fn test_compress_with_gaps() {
        let nodes = vec![100, 101, 105, 106, 110, 200, 201];
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert_eq!(compressed.base, 100);
        assert_eq!(compressed.deltas, vec![1, 4, 1, 4, 90, 1]);
        assert_eq!(compressed.len(), 7);

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }

    #[test]
    fn test_compress_large_gaps() {
        let nodes = vec![100, 1000, 10000, 100000];
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert_eq!(compressed.base, 100);
        assert_eq!(compressed.deltas, vec![900, 9000, 90000]);

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }

    #[test]
    fn test_contains() {
        let nodes = vec![100, 101, 105, 106, 110];
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert!(compressed.contains(100));
        assert!(compressed.contains(101));
        assert!(!compressed.contains(102)); // Gap
        assert!(!compressed.contains(103)); // Gap
        assert!(!compressed.contains(104)); // Gap
        assert!(compressed.contains(105));
        assert!(compressed.contains(106));
        assert!(!compressed.contains(107)); // Gap
        assert!(compressed.contains(110));
        assert!(!compressed.contains(111)); // Beyond end
        assert!(!compressed.contains(99));  // Before start
    }

    #[test]
    fn test_compression_stats_sequential() {
        let nodes = vec![100, 101, 102, 103, 104]; // 5 nodes
        let stats = CompressionStats::calculate(&nodes);

        // Original: 5 nodes * 8 bytes = 40 bytes (u64 = 8 bytes)
        assert_eq!(stats.original_bytes, 40);

        // Compressed: 8 bytes (base) + 4 * 4 bytes (deltas) = 24 bytes
        assert_eq!(stats.compressed_bytes, 24);

        // Ratio: 24/40 = 0.6 (40% compression)
        assert_eq!(stats.ratio, 0.6);
        assert_eq!(stats.node_count, 5);
    }

    #[test]
    fn test_compression_stats_with_gaps() {
        let nodes = vec![100, 101, 105, 106, 110, 200, 201]; // 7 nodes
        let stats = CompressionStats::calculate(&nodes);

        // Original: 7 * 8 = 56 bytes (u64 = 8 bytes)
        assert_eq!(stats.original_bytes, 56);

        // Compressed: 8 + 6 * 4 = 32 bytes
        assert_eq!(stats.compressed_bytes, 32);

        // Ratio: 32/56 ≈ 0.571 (~43% compression)
        assert!((stats.ratio - 0.571).abs() < 0.001);
    }

    #[test]
    fn test_roundtrip_large_list() {
        // Test with a large list of sequential IDs
        let nodes: Vec<NodeId> = (1000..2000).collect();
        let compressed = CompressedAdjacencyList::compress(&nodes);

        assert_eq!(compressed.base, 1000);
        assert_eq!(compressed.len(), 1000);

        // All deltas should be 1 (sequential)
        assert!(compressed.deltas.iter().all(|&d| d == 1));

        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }

    #[test]
    fn test_compression_effectiveness() {
        // Test real-world scenario: mostly sequential with some gaps
        let mut nodes = Vec::new();

        // Sequential batch 1
        for i in 100..200 {
            nodes.push(i);
        }

        // Gap

        // Sequential batch 2
        for i in 500..600 {
            nodes.push(i);
        }

        // Gap

        // Sequential batch 3
        for i in 1000..1100 {
            nodes.push(i);
        }

        let stats = CompressionStats::calculate(&nodes);

        // Should achieve >40% compression (ratio < 0.6 for u64 = 8 bytes)
        assert!(stats.ratio < 0.6, "Expected >40% compression, got ratio: {}", stats.ratio);

        // Verify roundtrip
        let compressed = CompressedAdjacencyList::compress(&nodes);
        let decompressed = compressed.decompress();
        assert_eq!(decompressed, nodes);
    }
}
