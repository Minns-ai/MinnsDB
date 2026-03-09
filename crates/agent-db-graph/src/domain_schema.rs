//! Canonical domain registry for temporal state predicates.
//!
//! Thin wrapper around OntologyRegistry for backward compatibility.
//! All behavioral decisions are driven by the ontology — no hardcoded
//! category names in this module.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// How many values can be simultaneously active for a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Cardinality {
    /// One value at a time. Newest supersedes all previous. (location, employer)
    Single,
    /// Multiple values simultaneously active. Sub-key disambiguates slots. (routines, relationships)
    Multi,
    /// Append-only, never supersede. (financial transactions)
    Append,
}

// ---------------------------------------------------------------------------
// Dynamic domain registry — thin wrapper around OntologyRegistry
// ---------------------------------------------------------------------------

/// A learned domain slot discovered at runtime from LLM extraction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LearnedSlot {
    pub category: String,
    pub canonical_predicate: String,
    pub cardinality: Cardinality,
    pub location_dependent: bool,
    pub discovered_at: u64,
    pub usage_count: u64,
}

/// Unified resolved slot — wraps ontology property data.
pub struct ResolvedSlot {
    cardinality: Cardinality,
    canonical_predicate: String,
    location_dependent: bool,
}

impl ResolvedSlot {
    pub fn cardinality(&self) -> Cardinality {
        self.cardinality
    }

    pub fn canonical_predicate(&self) -> &str {
        &self.canonical_predicate
    }

    pub fn location_dependent(&self) -> bool {
        self.location_dependent
    }
}

/// Dynamic domain registry that delegates to the OWL/RDFS ontology.
///
/// All behavioral queries (single-valued, append-only, symmetric, etc.)
/// are resolved via the ontology — no hardcoded category names.
pub struct DomainRegistry {
    ontology: Arc<crate::ontology::OntologyRegistry>,
}

impl DomainRegistry {
    pub fn new(ontology: Arc<crate::ontology::OntologyRegistry>) -> Self {
        Self { ontology }
    }

    /// Unified lookup: resolves a category via ontology properties.
    pub fn resolve(&self, category: &str) -> Option<ResolvedSlot> {
        let cat_lower = category.to_lowercase();
        let desc = self.ontology.resolve(&cat_lower)?;

        let cardinality = if desc.append_only {
            Cardinality::Append
        } else if self.ontology.is_single_valued(&cat_lower) {
            Cardinality::Single
        } else {
            Cardinality::Multi
        };

        Some(ResolvedSlot {
            cardinality,
            canonical_predicate: desc.canonical_predicate.clone(),
            location_dependent: desc.cascade_dependent,
        })
    }

    /// Returns true if the given category is single-valued.
    pub fn is_single_valued(&self, category: &str) -> bool {
        self.ontology.is_single_valued(category)
    }

    /// Returns true if the given category is location-dependent (cascade-dependent).
    pub fn is_location_dependent(&self, category: &str) -> bool {
        self.ontology.is_cascade_dependent(category)
    }

    /// Synchronous predicate canonicalization via ontology.
    pub fn canonicalize_predicate<'a>(
        &self,
        category: &str,
        raw_predicate: &'a str,
    ) -> Cow<'a, str> {
        self.ontology
            .canonicalize_predicate(category, raw_predicate)
    }

    /// Register a new learned category. Returns true if newly registered.
    pub fn register_category(
        &self,
        category: &str,
        cardinality: Cardinality,
        canonical_predicate: &str,
        location_dependent: bool,
    ) -> bool {
        self.ontology.register_category(
            category,
            cardinality,
            canonical_predicate,
            location_dependent,
        )
    }

    /// Bump usage_count for a learned slot (no-op — ontology tracks properties, not usage).
    pub fn record_usage(&self, category: &str) {
        self.ontology.record_usage(category);
    }

    /// Formatted category block for LLM prompt injection.
    pub fn prompt_category_block(&self) -> String {
        self.ontology.prompt_category_block()
    }

    /// Pipe-separated category list for LLM enum constraint.
    pub fn prompt_category_enum(&self) -> String {
        self.ontology.prompt_category_enum()
    }

    /// Return all learned slots for persistence.
    pub fn learned_slots(&self) -> Vec<LearnedSlot> {
        self.ontology.learned_slots()
    }

    /// Bulk restore learned slots (e.g., from persisted concept nodes).
    pub fn load_learned(&self, slots: Vec<LearnedSlot>) {
        self.ontology.load_learned(slots);
    }
}

/// Persist a learned domain slot as a concept node in the graph.
///
/// Creates a concept node named `__domain_slot:{category}` with domain metadata
/// stored as node properties.
pub fn persist_domain_slot(graph: &mut crate::structures::Graph, slot: &LearnedSlot) {
    use crate::structures::{ConceptType, GraphNode, NodeType};

    let concept_name = format!("__domain_slot:{}", slot.category);
    // Check if already exists
    if graph.get_concept_node(&concept_name).is_some() {
        // Update properties on existing node
        if let Some(&nid) = graph.concept_index.get(concept_name.as_str()) {
            if let Some(node) = graph.nodes.get_mut(nid) {
                node.properties.insert(
                    "domain_cardinality".to_string(),
                    serde_json::json!(format!("{:?}", slot.cardinality)),
                );
                node.properties.insert(
                    "domain_usage_count".to_string(),
                    serde_json::json!(slot.usage_count),
                );
            }
        }
        return;
    }
    let mut node = GraphNode::new(NodeType::Concept {
        concept_name: concept_name.clone(),
        concept_type: ConceptType::NamedEntity,
        confidence: 1.0,
    });
    node.properties.insert(
        "domain_cardinality".to_string(),
        serde_json::json!(format!("{:?}", slot.cardinality)),
    );
    node.properties.insert(
        "domain_canonical_predicate".to_string(),
        serde_json::json!(slot.canonical_predicate),
    );
    node.properties.insert(
        "domain_location_dependent".to_string(),
        serde_json::json!(slot.location_dependent),
    );
    node.properties.insert(
        "domain_discovered_at".to_string(),
        serde_json::json!(slot.discovered_at),
    );
    node.properties.insert(
        "domain_usage_count".to_string(),
        serde_json::json!(slot.usage_count),
    );
    if let Err(e) = graph.add_node(node) {
        tracing::warn!("Failed to persist domain slot '{}': {}", concept_name, e);
    }
}

/// Restore learned slots from `__domain_slot:*` concept nodes in the graph.
pub fn restore_learned_slots_from_graph(graph: &crate::structures::Graph) -> Vec<LearnedSlot> {
    let mut slots = Vec::new();
    for (name, &nid) in &graph.concept_index {
        if let Some(rest) = name.strip_prefix("__domain_slot:") {
            if let Some(node) = graph.nodes.get(nid) {
                let cardinality = node
                    .properties
                    .get("domain_cardinality")
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "Single" => Cardinality::Single,
                        "Append" => Cardinality::Append,
                        _ => Cardinality::Multi,
                    })
                    .unwrap_or(Cardinality::Multi);
                let canonical = node
                    .properties
                    .get("domain_canonical_predicate")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let loc_dep = node
                    .properties
                    .get("domain_location_dependent")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let discovered_at = node
                    .properties
                    .get("domain_discovered_at")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let usage_count = node
                    .properties
                    .get("domain_usage_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                slots.push(LearnedSlot {
                    category: rest.to_string(),
                    canonical_predicate: canonical,
                    cardinality,
                    location_dependent: loc_dep,
                    discovered_at,
                    usage_count,
                });
            }
        }
    }
    slots
}

// ---------------------------------------------------------------------------
// Embedding-based predicate canonicalizer
// ---------------------------------------------------------------------------

/// Cosine similarity threshold for matching a raw predicate to a canonical one.
const SIMILARITY_THRESHOLD: f32 = 0.70;

/// Embedding-based predicate canonicalizer.
///
/// Pre-embeds canonical predicates for single-valued domains, then uses cosine
/// similarity to match unknown predicates without hardcoded alias lists.
///
/// Thread-safe: uses interior mutability for the resolution cache.
pub struct PredicateCanonicalizer {
    /// Embedding client for generating vectors.
    embedding_client: Arc<dyn crate::claims::EmbeddingClient>,
    /// Ontology registry for property lookup (replaces DOMAIN_SLOTS).
    ontology: Arc<crate::ontology::OntologyRegistry>,
    /// Pre-computed embeddings for canonical predicates.
    /// Key: "category:canonical_predicate" (e.g., "location:lives_in")
    canonical_embeddings: RwLock<HashMap<String, Vec<f32>>>,
    /// Cache of resolved raw→canonical mappings.
    /// Key: "category:raw_predicate", Value: canonical_predicate (or raw if no match).
    resolution_cache: RwLock<HashMap<String, String>>,
    /// Whether canonical embeddings have been initialized.
    initialized: RwLock<bool>,
}

impl PredicateCanonicalizer {
    /// Create a new canonicalizer with the given embedding client and ontology.
    pub fn new(
        embedding_client: Arc<dyn crate::claims::EmbeddingClient>,
        ontology: Arc<crate::ontology::OntologyRegistry>,
    ) -> Self {
        Self {
            embedding_client,
            ontology,
            canonical_embeddings: RwLock::new(HashMap::new()),
            resolution_cache: RwLock::new(HashMap::new()),
            initialized: RwLock::new(false),
        }
    }

    /// Lazily initialize canonical embeddings on first use.
    async fn ensure_initialized(&self) {
        {
            let init = self.initialized.read().await;
            if *init {
                return;
            }
        }

        let mut init = self.initialized.write().await;
        if *init {
            return; // double-check after acquiring write lock
        }

        let mut embeddings = self.canonical_embeddings.write().await;

        // Collect canonical predicates from ontology (all single-valued with non-empty canonical)
        let to_embed = self.ontology.single_valued_canonicals();

        // Embed each canonical predicate with category context
        for (category, predicate) in &to_embed {
            let text = format!("{} {}", category, predicate.replace('_', " "));
            let request = crate::claims::EmbeddingRequest {
                text,
                context: Some(format!("predicate for {} domain", category)),
            };
            match self.embedding_client.embed(request).await {
                Ok(resp) => {
                    let key = format!("{}:{}", category, predicate);
                    embeddings.insert(key, resp.embedding);
                },
                Err(e) => {
                    tracing::warn!(
                        "Failed to embed canonical predicate '{}:{}': {}",
                        category,
                        predicate,
                        e
                    );
                },
            }
        }

        tracing::info!(
            "PredicateCanonicalizer initialized with {} canonical embeddings",
            embeddings.len()
        );

        *init = true;
    }

    /// Async predicate canonicalization using embedding similarity.
    ///
    /// For Single-valued domains:
    /// 1. Check resolution cache → return if hit
    /// 2. Embed the raw predicate
    /// 3. Cosine similarity against canonical embeddings in the same category
    /// 4. If above threshold → return canonical; otherwise return raw
    /// 5. Cache the result
    ///
    /// For Multi/Append or unknown domains: returns raw predicate unchanged.
    pub async fn canonicalize(&self, category: &str, raw_predicate: &str) -> String {
        // Check ontology for property info
        let desc = match self.ontology.resolve(category) {
            Some(d) => d,
            None => return raw_predicate.to_string(),
        };

        // Only canonicalize single-valued domains
        if desc.canonical_predicate.is_empty() || !self.ontology.is_single_valued(category) {
            return raw_predicate.to_string();
        }

        // Exact match — already canonical
        if raw_predicate == desc.canonical_predicate {
            return raw_predicate.to_string();
        }

        // Check resolution cache
        let cache_key = format!("{}:{}", category, raw_predicate.to_lowercase());
        {
            let cache = self.resolution_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        // Ensure canonical embeddings are ready
        self.ensure_initialized().await;

        // Embed the raw predicate
        let text = format!("{} {}", category, raw_predicate.replace('_', " "));
        let request = crate::claims::EmbeddingRequest {
            text,
            context: Some(format!("predicate for {} domain", category)),
        };
        let raw_embedding = match self.embedding_client.embed(request).await {
            Ok(resp) => resp.embedding,
            Err(e) => {
                tracing::debug!(
                    "Failed to embed predicate '{}:{}': {} — falling back to canonical",
                    category,
                    raw_predicate,
                    e
                );
                // Fallback: for single-valued, canonical is the safe default
                return desc.canonical_predicate.clone();
            },
        };

        // Find best match among canonical embeddings in the same category
        let canonical_embeddings = self.canonical_embeddings.read().await;
        let canonical_key = format!("{}:{}", category, desc.canonical_predicate);

        let result = if let Some(canonical_vec) = canonical_embeddings.get(&canonical_key) {
            let sim = agent_db_core::utils::cosine_similarity(&raw_embedding, canonical_vec);
            tracing::debug!(
                "Predicate similarity '{}:{}' vs canonical '{}': {:.3}",
                category,
                raw_predicate,
                desc.canonical_predicate,
                sim
            );
            if sim >= SIMILARITY_THRESHOLD {
                desc.canonical_predicate.clone()
            } else {
                raw_predicate.to_string()
            }
        } else {
            // No canonical embedding available — fall back to canonical for single-valued
            desc.canonical_predicate.clone()
        };

        // Cache the result
        {
            let mut cache = self.resolution_cache.write().await;
            cache.insert(cache_key, result.clone());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ontology() -> Arc<crate::ontology::OntologyRegistry> {
        // Tests run from crate directory (crates/agent-db-graph/), so use relative path
        let path = std::path::PathBuf::from("../../data/ontology");
        Arc::new(
            crate::ontology::OntologyRegistry::load_from_directory(&path).unwrap_or_else(|_| {
                // Fallback: try workspace root path
                let alt = std::path::PathBuf::from("data/ontology");
                crate::ontology::OntologyRegistry::load_from_directory(&alt)
                    .unwrap_or_else(|_| crate::ontology::OntologyRegistry::new())
            }),
        )
    }

    #[test]
    fn test_domain_registry_static_resolve() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        let slot = reg.resolve("location").unwrap();
        assert_eq!(slot.cardinality(), Cardinality::Single);
        assert_eq!(slot.canonical_predicate(), "lives_in");
        assert!(!slot.location_dependent());
    }

    #[test]
    fn test_domain_registry_register_and_resolve() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        assert!(reg.register_category("pets", Cardinality::Multi, "", false));
        // Second register is a no-op
        assert!(!reg.register_category("pets", Cardinality::Single, "", false));
        let slot = reg.resolve("pets").unwrap();
        assert_eq!(slot.cardinality(), Cardinality::Multi);
    }

    #[test]
    fn test_domain_registry_no_override_static() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        // Cannot register a static category
        assert!(!reg.register_category("location", Cardinality::Multi, "", false));
        // Still resolves to single-valued from ontology
        assert!(reg.is_single_valued("location"));
    }

    #[test]
    fn test_domain_registry_prompt_category_enum() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        reg.register_category("pets", Cardinality::Multi, "", false);
        let enum_str = reg.prompt_category_enum();
        // Should contain ontology-loaded categories
        assert!(enum_str.contains("other"));
    }

    #[test]
    fn test_domain_registry_load_learned() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        let slots = vec![LearnedSlot {
            category: "transportation".to_string(),
            canonical_predicate: "drives".to_string(),
            cardinality: Cardinality::Single,
            location_dependent: false,
            discovered_at: 100,
            usage_count: 5,
        }];
        reg.load_learned(slots);
        assert!(reg.is_single_valued("transportation"));
        assert_eq!(
            reg.canonicalize_predicate("transportation", "commutes_by")
                .as_ref(),
            "drives"
        );
    }

    #[test]
    fn test_canonicalize_single_valued_returns_canonical() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        // Any predicate in a single-valued domain returns the canonical
        let result = reg.canonicalize_predicate("location", "resides");
        assert_eq!(result.as_ref(), "lives_in");

        let result = reg.canonicalize_predicate("location", "moved_to");
        assert_eq!(result.as_ref(), "lives_in");
    }

    #[test]
    fn test_canonicalize_multi_valued_passthrough() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        let result = reg.canonicalize_predicate("routine", "morning_yoga");
        assert_eq!(result.as_ref(), "morning_yoga");
    }

    #[test]
    fn test_canonicalize_unknown_category_passthrough() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        let result = reg.canonicalize_predicate("unknown_cat", "some_pred");
        assert_eq!(result.as_ref(), "some_pred");
    }

    #[test]
    fn test_is_location_dependent() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        assert!(reg.is_location_dependent("routine"));
        assert!(!reg.is_location_dependent("preference"));
        assert!(!reg.is_location_dependent("location"));
        assert!(!reg.is_location_dependent("financial"));
    }

    #[test]
    fn test_is_single_valued() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        assert!(reg.is_single_valued("location"));
        assert!(reg.is_single_valued("work"));
        assert!(reg.is_single_valued("education"));
        assert!(!reg.is_single_valued("routine"));
        assert!(!reg.is_single_valued("preference"));
        assert!(!reg.is_single_valued("financial"));
    }

    #[test]
    fn test_decode_legacy_state_assoc() {
        let onto = test_ontology();
        let (cat, pred) = onto.decode_legacy_state_assoc("state:location").unwrap();
        assert_eq!(cat, "location");
        assert_eq!(pred, "lives_in");

        let (cat, pred) = onto.decode_legacy_state_assoc("state:routine").unwrap();
        assert_eq!(cat, "routine");
        assert_eq!(pred, "");

        let (cat, pred) = onto.decode_legacy_state_assoc("state:financial").unwrap();
        assert_eq!(cat, "financial");
        assert_eq!(pred, "payment");
    }

    #[test]
    fn test_decode_legacy_state_assoc_legacy_names() {
        let onto = test_ontology();
        let (cat, _) = onto.decode_legacy_state_assoc("state:landmark").unwrap();
        assert_eq!(cat, "routine");

        let (cat, _) = onto
            .decode_legacy_state_assoc("state:saturday_routine")
            .unwrap();
        assert_eq!(cat, "routine");

        let (cat, _) = onto.decode_legacy_state_assoc("state:breakfast").unwrap();
        assert_eq!(cat, "routine");
    }

    #[test]
    fn test_decode_legacy_state_assoc_non_state_returns_none() {
        let onto = test_ontology();
        assert!(onto
            .decode_legacy_state_assoc("location:lives_in")
            .is_none());
        assert!(onto.decode_legacy_state_assoc("routine:morning").is_none());
    }

    #[test]
    fn test_work_canonicalization() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        let result = reg.canonicalize_predicate("work", "employed_by");
        assert_eq!(result.as_ref(), "works_at");

        let result = reg.canonicalize_predicate("work", "works_for");
        assert_eq!(result.as_ref(), "works_at");
    }

    #[test]
    fn test_domain_registry_record_usage() {
        let onto = test_ontology();
        let reg = DomainRegistry::new(onto);
        reg.register_category("pets", Cardinality::Multi, "", false);
        // record_usage is a no-op in ontology mode
        reg.record_usage("pets");
    }
}
