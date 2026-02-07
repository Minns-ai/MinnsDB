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

// ============================================================================
// BM25 Full-Text Search Implementation
// ============================================================================

/// BM25 parameters for tuning ranking behavior
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term saturation parameter (typically 1.2-2.0)
    /// Controls how quickly term frequency saturates
    pub k1: f32,

    /// Length normalization parameter (typically 0.75)
    /// Controls document length normalization strength
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.5, // Standard BM25 parameter
            b: 0.75, // Standard length normalization
        }
    }
}

/// Document statistics for BM25 scoring
#[derive(Debug, Clone)]
struct DocumentStats {
    /// Document length (number of terms)
    length: usize,
    /// Term frequencies in this document
    term_frequencies: HashMap<String, usize>,
}

/// Corpus-level statistics for BM25
#[derive(Debug)]
struct CorpusStats {
    /// Total number of documents
    total_docs: usize,
    /// Average document length
    avg_doc_length: f32,
    /// Document frequency per term (how many docs contain term)
    doc_frequencies: HashMap<String, usize>,
}

/// BM25 (Best Matching 25) ranking index
///
/// Provides superior keyword-based ranking compared to simple term frequency.
/// BM25 is a probabilistic retrieval model that ranks documents based on:
/// - Term frequency (TF): How often terms appear in document
/// - Inverse document frequency (IDF): Rarity of terms across corpus
/// - Document length normalization: Prevents bias toward long documents
#[derive(Debug)]
pub struct Bm25Index {
    config: Bm25Config,
    corpus_stats: CorpusStats,
    documents: HashMap<NodeId, DocumentStats>,
}

impl Bm25Index {
    /// Create new BM25 index with default configuration
    pub fn new() -> Self {
        Self::with_config(Bm25Config::default())
    }

    /// Create new BM25 index with custom configuration
    pub fn with_config(config: Bm25Config) -> Self {
        Self {
            config,
            corpus_stats: CorpusStats {
                total_docs: 0,
                avg_doc_length: 0.0,
                doc_frequencies: HashMap::new(),
            },
            documents: HashMap::new(),
        }
    }

    /// Index a document for BM25 search
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for the document
    /// * `text` - Text content to index
    pub fn index_document(&mut self, node_id: NodeId, text: &str) {
        let terms = Self::tokenize(text);
        let length = terms.len();

        // Calculate term frequencies
        let mut term_frequencies = HashMap::new();
        for term in &terms {
            *term_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        // Update corpus statistics
        self.corpus_stats.total_docs += 1;

        // Update document frequencies (only count each term once per document)
        for term in term_frequencies.keys() {
            *self
                .corpus_stats
                .doc_frequencies
                .entry(term.clone())
                .or_insert(0) += 1;
        }

        // Recalculate average document length
        let total_length: usize = self.documents.values().map(|d| d.length).sum::<usize>() + length;
        self.corpus_stats.avg_doc_length =
            total_length as f32 / self.corpus_stats.total_docs as f32;

        // Store document stats
        self.documents.insert(
            node_id,
            DocumentStats {
                length,
                term_frequencies,
            },
        );
    }

    /// Remove a document from the index
    pub fn remove_document(&mut self, node_id: NodeId) {
        if let Some(doc_stats) = self.documents.remove(&node_id) {
            self.corpus_stats.total_docs -= 1;

            // Update document frequencies
            for term in doc_stats.term_frequencies.keys() {
                if let Some(count) = self.corpus_stats.doc_frequencies.get_mut(term) {
                    *count -= 1;
                    if *count == 0 {
                        self.corpus_stats.doc_frequencies.remove(term);
                    }
                }
            }

            // Recalculate average document length
            if self.corpus_stats.total_docs > 0 {
                let total_length: usize = self.documents.values().map(|d| d.length).sum();
                self.corpus_stats.avg_doc_length =
                    total_length as f32 / self.corpus_stats.total_docs as f32;
            } else {
                self.corpus_stats.avg_doc_length = 0.0;
            }
        }
    }

    /// Search using BM25 scoring
    ///
    /// Returns results sorted by relevance score (highest first)
    ///
    /// # Arguments
    /// * `query` - Search query text
    /// * `limit` - Maximum number of results to return
    pub fn search(&self, query: &str, limit: usize) -> Vec<(NodeId, f32)> {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Calculate BM25 score for each document
        for (&node_id, doc_stats) in &self.documents {
            let mut score = 0.0;

            for term in &query_terms {
                score += self.bm25_term_score(term, doc_stats);
            }

            if score > 0.0 {
                scores.insert(node_id, score);
            }
        }

        // Sort by score descending
        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        results
    }

    /// Calculate BM25 score for a single term in a document
    fn bm25_term_score(&self, term: &str, doc_stats: &DocumentStats) -> f32 {
        // Get term frequency in document
        let tf = *doc_stats.term_frequencies.get(term).unwrap_or(&0) as f32;
        if tf == 0.0 {
            return 0.0;
        }

        // Calculate IDF (Inverse Document Frequency)
        let df = *self.corpus_stats.doc_frequencies.get(term).unwrap_or(&0) as f32;
        let n = self.corpus_stats.total_docs as f32;

        // Standard BM25 IDF formula
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        // Calculate normalized term frequency
        let doc_length = doc_stats.length as f32;
        let k1 = self.config.k1;
        let b = self.config.b;
        let avg_length = self.corpus_stats.avg_doc_length;

        let normalized_tf =
            (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_length / avg_length)));

        idf * normalized_tf
    }

    /// Tokenize text into terms
    ///
    /// Simple tokenization: lowercase, split on whitespace, filter short words
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|s| s.len() > 2) // Filter out very short words
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Get index statistics
    pub fn stats(&self) -> Bm25Stats {
        Bm25Stats {
            total_docs: self.corpus_stats.total_docs,
            unique_terms: self.corpus_stats.doc_frequencies.len(),
            avg_doc_length: self.corpus_stats.avg_doc_length,
        }
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about BM25 index
#[derive(Debug, Clone)]
pub struct Bm25Stats {
    pub total_docs: usize,
    pub unique_terms: usize,
    pub avg_doc_length: f32,
}

// ============================================================================
// Fusion Strategies for Hybrid Search
// ============================================================================

/// Search mode for hybrid retrieval
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// BM25 keyword search only
    Keyword,
    /// Semantic embedding search only
    Semantic,
    /// Hybrid: combine both with fusion
    Hybrid,
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Hybrid
    }
}

/// Fusion strategy for combining search results
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (RRF)
    /// More robust to score scale differences
    ReciprocalRank { k: f32 },

    /// Weighted average of scores
    Weighted {
        keyword_weight: f32,
        semantic_weight: f32,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        // RRF is more robust to score scale differences
        Self::ReciprocalRank { k: 60.0 }
    }
}

impl FusionStrategy {
    /// Fuse two ranked result lists
    pub fn fuse(
        &self,
        keyword_results: Vec<(NodeId, f32)>,
        semantic_results: Vec<(NodeId, f32)>,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::ReciprocalRank { k } => {
                Self::reciprocal_rank_fusion(keyword_results, semantic_results, *k)
            },
            Self::Weighted {
                keyword_weight,
                semantic_weight,
            } => Self::weighted_fusion(
                keyword_results,
                semantic_results,
                *keyword_weight,
                *semantic_weight,
            ),
        }
    }

    /// Reciprocal Rank Fusion
    ///
    /// Formula: RRF_score(d) = Σ 1/(k + rank(d))
    /// where rank(d) is the rank of document d in each result list
    fn reciprocal_rank_fusion(
        keyword_results: Vec<(NodeId, f32)>,
        semantic_results: Vec<(NodeId, f32)>,
        k: f32,
    ) -> Vec<(NodeId, f32)> {
        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Add keyword RRF scores
        for (rank, (node_id, _)) in keyword_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank + 1) as f32);
            *scores.entry(*node_id).or_insert(0.0) += rrf_score;
        }

        // Add semantic RRF scores
        for (rank, (node_id, _)) in semantic_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank + 1) as f32);
            *scores.entry(*node_id).or_insert(0.0) += rrf_score;
        }

        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Weighted fusion of raw scores
    fn weighted_fusion(
        keyword_results: Vec<(NodeId, f32)>,
        semantic_results: Vec<(NodeId, f32)>,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> Vec<(NodeId, f32)> {
        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Add weighted keyword scores
        for (node_id, score) in keyword_results {
            *scores.entry(node_id).or_insert(0.0) += score * keyword_weight;
        }

        // Add weighted semantic scores
        for (node_id, score) in semantic_results {
            *scores.entry(node_id).or_insert(0.0) += score * semantic_weight;
        }

        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
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

    #[test]
    fn test_bm25_basic_ranking() {
        let mut index = Bm25Index::new();

        index.index_document(1, "the quick brown fox jumps over the lazy dog");
        index.index_document(2, "the quick brown fox");
        index.index_document(3, "the lazy dog sleeps");

        let results = index.search("quick fox", 10);

        // Doc 2 should rank highest (contains both terms, shorter document)
        assert_eq!(results[0].0, 2);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_bm25_idf_weighting() {
        let mut index = Bm25Index::new();

        // "the" appears in all docs (low IDF)
        // "authentication" appears in one (high IDF)
        index.index_document(1, "the user logged in successfully");
        index.index_document(2, "the authentication failed");
        index.index_document(3, "the system started");

        let results = index.search("authentication", 10);

        // Only doc 2 should match
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_bm25_remove_document() {
        let mut index = Bm25Index::new();

        index.index_document(1, "test document one");
        index.index_document(2, "test document two");

        assert_eq!(index.stats().total_docs, 2);

        index.remove_document(1);

        assert_eq!(index.stats().total_docs, 1);
        let results = index.search("test", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let keyword_results = vec![(1, 10.0), (2, 8.0), (3, 5.0)];
        let semantic_results = vec![(3, 0.95), (1, 0.90), (4, 0.85)];

        let strategy = FusionStrategy::ReciprocalRank { k: 60.0 };
        let fused = strategy.fuse(keyword_results, semantic_results);

        // Node 1 and 3 appear in both, should rank high
        assert!(fused[0].0 == 1 || fused[0].0 == 3);
        assert!(fused.len() == 4); // 4 unique nodes
    }

    #[test]
    fn test_weighted_fusion() {
        let keyword_results = vec![(1, 10.0), (2, 5.0)];
        let semantic_results = vec![(2, 0.9), (3, 0.8)];

        let strategy = FusionStrategy::Weighted {
            keyword_weight: 0.7,
            semantic_weight: 0.3,
        };
        let fused = strategy.fuse(keyword_results, semantic_results);

        // Node 1 should win (10.0 * 0.7 = 7.0)
        // Node 2 gets (5.0 * 0.7 + 0.9 * 0.3 = 3.5 + 0.27 = 3.77)
        assert_eq!(fused[0].0, 1);
        assert!(fused[0].1 > 6.0);
    }
}
