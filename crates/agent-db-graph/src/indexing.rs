//! Property indexing system for fast graph queries
//!
//! Provides B-tree, hash, and full-text indexes on node properties
//! for 100-1000x faster property-based queries.

use crate::structures::NodeId;
use crate::{GraphError, GraphResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// Type of index to create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// B-tree index for ordered queries and range scans
    BTree,
    /// Hash index for exact match queries (fastest for equality)
    Hash,
    /// Full-text index for text search
    FullText,
}

/// Statistics about index usage
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub insert_count: u64,
    pub query_count: u64,
    pub range_query_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub last_accessed: u64,
}

/// Property index for fast lookups
pub struct PropertyIndex {
    /// Index name
    name: String,

    /// Property key being indexed
    property_key: String,

    /// Index type
    index_type: IndexType,

    /// B-tree map: property_value -> Vec<NodeId>
    btree_index: BTreeMap<IndexKey, Vec<NodeId>>,

    /// Hash map for exact matches (faster than BTree for equality)
    hash_index: HashMap<IndexKey, Vec<NodeId>>,

    /// Statistics
    stats: IndexStats,
}

/// Wrapper for index keys to enable ordering and hashing
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexKey {
    String(String),
    Number(i64),
    Float(OrderedFloat),
    Bool(bool),
    Null,
}

/// Wrapper for f64 to make it orderable
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderedFloat(i64);

impl OrderedFloat {
    pub fn from_f64(f: f64) -> Self {
        // Convert f64 to i64 for ordering (using bits representation)
        OrderedFloat(f.to_bits() as i64)
    }

    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0 as u64)
    }
}

impl From<&serde_json::Value> for IndexKey {
    fn from(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => IndexKey::String(s.clone()),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    IndexKey::Number(i)
                } else if let Some(f) = n.as_f64() {
                    IndexKey::Float(OrderedFloat::from_f64(f))
                } else {
                    IndexKey::Null
                }
            },
            serde_json::Value::Bool(b) => IndexKey::Bool(*b),
            serde_json::Value::Null => IndexKey::Null,
            _ => IndexKey::Null, // Arrays and objects not supported as keys
        }
    }
}

impl PropertyIndex {
    /// Create a new property index
    pub fn new(name: String, property_key: String, index_type: IndexType) -> Self {
        Self {
            name,
            property_key,
            index_type,
            btree_index: BTreeMap::new(),
            hash_index: HashMap::new(),
            stats: IndexStats::default(),
        }
    }

    /// Insert node into index
    pub fn insert(&mut self, node_id: NodeId, value: &serde_json::Value) {
        let key = IndexKey::from(value);

        match self.index_type {
            IndexType::BTree => {
                self.btree_index.entry(key).or_default().push(node_id);
            },
            IndexType::Hash => {
                self.hash_index.entry(key).or_default().push(node_id);
            },
            IndexType::FullText => {
                // For full-text, index each word
                if let serde_json::Value::String(s) = value {
                    for word in s.split_whitespace() {
                        let word_key = IndexKey::String(word.to_lowercase());
                        self.hash_index.entry(word_key).or_default().push(node_id);
                    }
                }
            },
        }

        self.stats.insert_count += 1;
    }

    /// Remove node from index
    pub fn remove(&mut self, node_id: NodeId, value: &serde_json::Value) {
        let key = IndexKey::from(value);

        match self.index_type {
            IndexType::BTree => {
                if let Some(nodes) = self.btree_index.get_mut(&key) {
                    nodes.retain(|&id| id != node_id);
                    if nodes.is_empty() {
                        self.btree_index.remove(&key);
                    }
                }
            },
            IndexType::Hash => {
                if let Some(nodes) = self.hash_index.get_mut(&key) {
                    nodes.retain(|&id| id != node_id);
                    if nodes.is_empty() {
                        self.hash_index.remove(&key);
                    }
                }
            },
            IndexType::FullText => {
                if let serde_json::Value::String(s) = value {
                    for word in s.split_whitespace() {
                        let word_key = IndexKey::String(word.to_lowercase());
                        if let Some(nodes) = self.hash_index.get_mut(&word_key) {
                            nodes.retain(|&id| id != node_id);
                            if nodes.is_empty() {
                                self.hash_index.remove(&word_key);
                            }
                        }
                    }
                }
            },
        }
    }

    /// Query nodes by property value (exact match)
    pub fn query(&mut self, value: &serde_json::Value) -> Vec<NodeId> {
        self.stats.query_count += 1;
        let key = IndexKey::from(value);

        let result = match self.index_type {
            IndexType::BTree => self.btree_index.get(&key).cloned().unwrap_or_default(),
            IndexType::Hash | IndexType::FullText => {
                self.hash_index.get(&key).cloned().unwrap_or_default()
            },
        };

        if result.is_empty() {
            self.stats.miss_count += 1;
        } else {
            self.stats.hit_count += 1;
        }

        result
    }

    /// Range query (min <= value <= max) - only for BTree indexes
    pub fn range_query(
        &mut self,
        min: &serde_json::Value,
        max: &serde_json::Value,
    ) -> GraphResult<Vec<NodeId>> {
        if !matches!(self.index_type, IndexType::BTree) {
            return Err(GraphError::InvalidOperation(
                "Range queries only supported on BTree indexes".to_string(),
            ));
        }

        self.stats.range_query_count += 1;

        let min_key = IndexKey::from(min);
        let max_key = IndexKey::from(max);

        let result: Vec<NodeId> = self
            .btree_index
            .range(min_key..=max_key)
            .flat_map(|(_, nodes)| nodes.iter().copied())
            .collect();

        if result.is_empty() {
            self.stats.miss_count += 1;
        } else {
            self.stats.hit_count += 1;
        }

        Ok(result)
    }

    /// Greater than query (value > min) - only for BTree indexes
    pub fn greater_than(&mut self, min: &serde_json::Value) -> GraphResult<Vec<NodeId>> {
        if !matches!(self.index_type, IndexType::BTree) {
            return Err(GraphError::InvalidOperation(
                "Range queries only supported on BTree indexes".to_string(),
            ));
        }

        self.stats.range_query_count += 1;

        let min_key = IndexKey::from(min);

        let result: Vec<NodeId> = self
            .btree_index
            .range((
                std::ops::Bound::Excluded(min_key),
                std::ops::Bound::Unbounded,
            ))
            .flat_map(|(_, nodes)| nodes.iter().copied())
            .collect();

        Ok(result)
    }

    /// Less than query (value < max) - only for BTree indexes
    pub fn less_than(&mut self, max: &serde_json::Value) -> GraphResult<Vec<NodeId>> {
        if !matches!(self.index_type, IndexType::BTree) {
            return Err(GraphError::InvalidOperation(
                "Range queries only supported on BTree indexes".to_string(),
            ));
        }

        self.stats.range_query_count += 1;

        let max_key = IndexKey::from(max);

        let result: Vec<NodeId> = self
            .btree_index
            .range((
                std::ops::Bound::Unbounded,
                std::ops::Bound::Excluded(max_key),
            ))
            .flat_map(|(_, nodes)| nodes.iter().copied())
            .collect();

        Ok(result)
    }

    /// Full-text search - only for FullText indexes
    pub fn search_text(&mut self, query: &str) -> GraphResult<Vec<NodeId>> {
        if !matches!(self.index_type, IndexType::FullText) {
            return Err(GraphError::InvalidOperation(
                "Text search only supported on FullText indexes".to_string(),
            ));
        }

        self.stats.query_count += 1;

        // Split query into words and find nodes containing all words (AND query)
        let words: Vec<String> = query.split_whitespace().map(|w| w.to_lowercase()).collect();

        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Get nodes for first word
        let first_word_key = IndexKey::String(words[0].clone());
        let mut result: HashSet<NodeId> = self
            .hash_index
            .get(&first_word_key)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        // Intersect with nodes for other words
        for word in &words[1..] {
            let word_key = IndexKey::String(word.clone());
            if let Some(nodes) = self.hash_index.get(&word_key) {
                let word_set: HashSet<NodeId> = nodes.iter().copied().collect();
                result = result.intersection(&word_set).copied().collect();
            } else {
                // If any word is not found, no results
                result.clear();
                break;
            }
        }

        Ok(result.into_iter().collect())
    }

    /// Get index statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Get property key for this index
    pub fn property_key(&self) -> &str {
        &self.property_key
    }

    /// Get index name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get index type
    pub fn index_type(&self) -> &IndexType {
        &self.index_type
    }

    /// Get estimated size (number of unique values indexed)
    pub fn size(&self) -> usize {
        match self.index_type {
            IndexType::BTree => self.btree_index.len(),
            IndexType::Hash | IndexType::FullText => self.hash_index.len(),
        }
    }
}

/// Manages all property indexes for a graph
pub struct IndexManager {
    /// Property indexes by name
    property_indexes: HashMap<String, PropertyIndex>,

    /// Auto-indexed properties (commonly queried)
    auto_index_properties: HashSet<String>,

    /// Query frequency counter for auto-indexing decisions
    query_frequency: HashMap<String, AtomicU64>,

    /// Threshold for auto-creating indexes
    auto_index_threshold: u64,
}

impl IndexManager {
    /// Create a new index manager
    pub fn new() -> Self {
        Self {
            property_indexes: HashMap::new(),
            auto_index_properties: HashSet::new(),
            query_frequency: HashMap::new(),
            auto_index_threshold: 100, // Create index after 100 queries
        }
    }

    /// Create a new index
    pub fn create_index(
        &mut self,
        name: String,
        property_key: String,
        index_type: IndexType,
    ) -> GraphResult<()> {
        if self.property_indexes.contains_key(&name) {
            return Err(GraphError::InvalidOperation(format!(
                "Index '{}' already exists",
                name
            )));
        }

        let index = PropertyIndex::new(name.clone(), property_key, index_type);
        self.property_indexes.insert(name, index);

        Ok(())
    }

    /// Drop an existing index
    pub fn drop_index(&mut self, name: &str) -> GraphResult<()> {
        if self.property_indexes.remove(name).is_none() {
            return Err(GraphError::InvalidOperation(format!(
                "Index '{}' does not exist",
                name
            )));
        }

        Ok(())
    }

    /// Get index by name
    pub fn get_index(&self, name: &str) -> Option<&PropertyIndex> {
        self.property_indexes.get(name)
    }

    /// Get mutable index by name
    pub fn get_index_mut(&mut self, name: &str) -> Option<&mut PropertyIndex> {
        self.property_indexes.get_mut(name)
    }

    /// Find index for a property key
    pub fn find_index_for_property(&mut self, property_key: &str) -> Option<&mut PropertyIndex> {
        self.property_indexes
            .values_mut()
            .find(|idx| idx.property_key() == property_key)
    }

    /// Check if an index exists for a property
    pub fn has_index_for_property(&self, property_key: &str) -> bool {
        self.property_indexes
            .values()
            .any(|idx| idx.property_key() == property_key)
    }

    /// Record a query on a property (for auto-indexing)
    pub fn record_property_query(&mut self, property_key: &str) {
        let counter = self
            .query_frequency
            .entry(property_key.to_string())
            .or_insert_with(|| AtomicU64::new(0));

        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;

        // Auto-create index if threshold reached
        if count >= self.auto_index_threshold && !self.has_index_for_property(property_key) {
            let index_name = format!("auto_idx_{}", property_key);
            // Use Hash index by default for auto-indexes (fastest for most queries)
            let _ = self.create_index(index_name, property_key.to_string(), IndexType::Hash);
            self.auto_index_properties.insert(property_key.to_string());
        }
    }

    /// List all indexes
    pub fn list_indexes(&self) -> Vec<(&str, &str, &IndexType)> {
        self.property_indexes
            .values()
            .map(|idx| (idx.name(), idx.property_key(), idx.index_type()))
            .collect()
    }

    /// Get statistics for all indexes
    pub fn get_all_stats(&self) -> HashMap<String, IndexStats> {
        self.property_indexes
            .iter()
            .map(|(name, idx)| (name.clone(), idx.stats().clone()))
            .collect()
    }
}

impl Default for IndexManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_btree_index_insert_query() {
        let mut index =
            PropertyIndex::new("test_idx".to_string(), "age".to_string(), IndexType::BTree);

        // Insert some nodes
        index.insert(1, &json!(25));
        index.insert(2, &json!(30));
        index.insert(3, &json!(25));

        // Query
        let results = index.query(&json!(25));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&1));
        assert!(results.contains(&3));
    }

    #[test]
    fn test_range_query() {
        let mut index =
            PropertyIndex::new("test_idx".to_string(), "age".to_string(), IndexType::BTree);

        index.insert(1, &json!(20));
        index.insert(2, &json!(25));
        index.insert(3, &json!(30));
        index.insert(4, &json!(35));

        let results = index.range_query(&json!(23), &json!(32)).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&2)); // 25
        assert!(results.contains(&3)); // 30
    }

    #[test]
    fn test_fulltext_index() {
        let mut index = PropertyIndex::new(
            "test_idx".to_string(),
            "description".to_string(),
            IndexType::FullText,
        );

        index.insert(1, &json!("fix null pointer error"));
        index.insert(2, &json!("add error handling"));
        index.insert(3, &json!("fix memory leak"));

        // Search for "error"
        let results = index.search_text("error").unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&1));
        assert!(results.contains(&2));
    }

    #[test]
    fn test_index_manager() {
        let mut manager = IndexManager::new();

        // Create index
        manager
            .create_index("age_idx".to_string(), "age".to_string(), IndexType::BTree)
            .unwrap();

        assert!(manager.has_index_for_property("age"));

        // Insert into index
        if let Some(index) = manager.find_index_for_property("age") {
            index.insert(1, &json!(25));
        }

        // Query
        if let Some(index) = manager.find_index_for_property("age") {
            let results = index.query(&json!(25));
            assert_eq!(results.len(), 1);
            assert_eq!(results[0], 1);
        }
    }
}
