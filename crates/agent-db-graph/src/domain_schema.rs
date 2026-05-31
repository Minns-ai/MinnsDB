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

/// Cosine similarity threshold for overriding the LLM's category pick with
/// the embedding classifier's pick. Higher than the predicate threshold
/// because category is upstream of every supersession/contradiction
/// decision — we want strong confidence before overriding the model.
pub(crate) const CATEGORY_OVERRIDE_THRESHOLD: f32 = 0.75;

/// Lower-bound similarity below which the embedding classifier is treated
/// as having no useful signal. Used when the LLM picked a category the
/// ontology doesn't recognise — we'd rather pass the LLM's pick through
/// than guess wildly from a near-zero similarity.
pub(crate) const CATEGORY_MIN_SIGNAL: f32 = 0.40;

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
    /// Pre-computed embeddings for entire categories — one vector per
    /// top-level property, derived from its label, comment, and
    /// canonical predicate. Used by `classify_category` to map an
    /// arbitrary predicate+statement to the closest registered category.
    /// Key: property id (e.g., "location").
    category_embeddings: RwLock<HashMap<String, Vec<f32>>>,
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
            category_embeddings: RwLock::new(HashMap::new()),
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

        // Embed every canonical predicate in ONE batched HTTP call. The
        // previous loop fired one round-trip per predicate (~10-20
        // predicates × ~100ms each = 1-2 s) on every cold init of the
        // canonicalizer, which lazily happens during the first ingest.
        // Batching collapses that into one call.
        let predicate_requests: Vec<crate::claims::EmbeddingRequest> = to_embed
            .iter()
            .map(|(category, predicate)| crate::claims::EmbeddingRequest {
                text: format!("{} {}", category, predicate.replace('_', " ")),
                context: Some(format!("predicate for {} domain", category)),
            })
            .collect();
        match self.embedding_client.embed_batch(predicate_requests).await {
            Ok(responses) => {
                for ((category, predicate), resp) in to_embed.iter().zip(responses.into_iter()) {
                    let key = format!("{}:{}", category, predicate);
                    embeddings.insert(key, resp.embedding);
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to embed {} canonical predicates in batch: {}",
                    to_embed.len(),
                    e
                );
            },
        }

        // Embed each top-level category as a rich description: label,
        // comment, and canonical predicate together. Comments in the TTL
        // include synonym lists (e.g. "where someone lives, moved to,
        // resides") which give the embedding model enough surface area
        // to map arbitrary raw predicates from any language onto the
        // right category by cosine similarity alone.
        let mut category_embeddings = self.category_embeddings.write().await;
        let descriptors = self.ontology.top_level_property_descriptors();
        // Same batching reasoning as the canonical-predicates loop above:
        // ~10-15 round-trips per init become one HTTP call.
        //
        // Skip empty descriptors before the batch — a property with no
        // label, comment, or canonical predicate has nothing to anchor
        // an embedding against and would just add noise.
        let category_inputs: Vec<&(String, String, String, String)> = descriptors
            .iter()
            .filter(|(_id, label, comment, canonical)| {
                !(label.is_empty() && comment.is_empty() && canonical.is_empty())
            })
            .collect();
        let category_requests: Vec<crate::claims::EmbeddingRequest> = category_inputs
            .iter()
            .map(
                |(id, label, comment, canonical)| crate::claims::EmbeddingRequest {
                    text: format!("{} {} {}", label, comment, canonical.replace('_', " "))
                        .trim()
                        .to_string(),
                    context: Some(format!("ontology category {}", id)),
                },
            )
            .collect();
        match self.embedding_client.embed_batch(category_requests).await {
            Ok(responses) => {
                for ((id, _, _, _), resp) in category_inputs.iter().zip(responses.into_iter()) {
                    category_embeddings.insert((*id).clone(), resp.embedding);
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to embed {} ontology categories in batch: {}",
                    category_inputs.len(),
                    e
                );
            },
        }

        tracing::info!(
            "PredicateCanonicalizer initialized with {} canonical and {} category embeddings",
            embeddings.len(),
            category_embeddings.len(),
        );

        *init = true;
    }

    /// Classify the most likely ontology category for an LLM-emitted
    /// `(raw_predicate, statement)` pair by cosine similarity against the
    /// pre-embedded category descriptors.
    ///
    /// Returns `(property_id, similarity)` for the best match, or `None`
    /// when no category has been embedded (no embedding client at boot,
    /// initialisation failed for every property, etc).
    ///
    /// The caller decides whether to act on the similarity score —
    /// `CATEGORY_OVERRIDE_THRESHOLD` is the default cutoff for overriding
    /// a disagreeing LLM, `CATEGORY_MIN_SIGNAL` is the floor below which
    /// the embedding signal should be discarded entirely.
    pub async fn classify_category(
        &self,
        raw_predicate: &str,
        statement: &str,
    ) -> Option<(String, f32)> {
        self.ensure_initialized().await;

        // Embed the raw predicate together with the source statement so
        // sparse predicates ("relocated to") get disambiguated by the
        // surrounding context, and so the model has a multilingual
        // signal even when the predicate alone is too short to localise.
        let text = format!("{} {}", raw_predicate.replace('_', " "), statement);
        let request = crate::claims::EmbeddingRequest {
            text,
            context: Some("classify against ontology categories".to_string()),
        };
        let query_embedding = match self.embedding_client.embed(request).await {
            Ok(resp) => resp.embedding,
            Err(e) => {
                tracing::debug!(
                    "classify_category embed failed for predicate '{}': {}",
                    raw_predicate,
                    e
                );
                return None;
            },
        };

        let category_embeddings = self.category_embeddings.read().await;
        if category_embeddings.is_empty() {
            return None;
        }

        let mut best: Option<(String, f32)> = None;
        for (id, vec) in category_embeddings.iter() {
            let sim = agent_db_core::utils::cosine_similarity(&query_embedding, vec);
            match &best {
                Some((_, best_sim)) if sim <= *best_sim => {},
                _ => best = Some((id.clone(), sim)),
            }
        }
        best
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
        // Should contain the dynamically-registered category and no
        // "other" escape hatch.
        assert!(enum_str.contains("pets"));
        assert!(!enum_str.contains("other"));
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

    // ────────── Embedding category classifier tests ──────────
    //
    // The classifier is exercised against a deterministic stub embedding
    // client. Each registered category gets a unique unit vector; the
    // query text is mapped to whichever category's keywords it contains.
    // This isolates the cosine-similarity wiring from the embedding
    // model itself — the tests assert the classifier picks the right
    // category, not that any particular model agrees with us.

    use crate::claims::embeddings::{
        Embedding, EmbeddingClient, EmbeddingRequest, EmbeddingResponse,
    };

    /// Deterministic stub. Inspects the request text for category keywords
    /// and returns a one-hot vector in the corresponding slot. Cosine
    /// similarity then picks out the right category exactly.
    struct StubEmbeddingClient {
        dim: usize,
    }

    #[async_trait::async_trait]
    impl EmbeddingClient for StubEmbeddingClient {
        async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
            let lower = request.text.to_lowercase();
            // Map keywords to dimension indices. Each category gets a
            // unique slot so the embedding for "location stuff" and the
            // embedding for "relocated to NYC" both light up slot 0 and
            // cosine to ~1.0.
            let mut v: Embedding = vec![0.0; self.dim];
            if lower.contains("location")
                || lower.contains("lives")
                || lower.contains("relocated")
                || lower.contains("moved to")
                || lower.contains("resides")
            {
                v[0] = 1.0;
            } else if lower.contains("financial")
                || lower.contains("payment")
                || lower.contains("paid")
            {
                v[1] = 1.0;
            } else if lower.contains("preference")
                || lower.contains("likes")
                || lower.contains("loves")
            {
                v[2] = 1.0;
            } else {
                // Unknown text gets no signal in any keyword slot. The
                // stub doesn't try to model "semantic dissimilarity →
                // low cosine" because that's a real-embedding-model
                // property, not something a fixed-slot stub can fake
                // without large dimensions to avoid hash collisions.
            }
            Ok(EmbeddingResponse {
                embedding: v,
                model: "stub".to_string(),
                tokens_used: 0,
            })
        }

        fn dimensions(&self) -> usize {
            self.dim
        }

        fn model_name(&self) -> &str {
            "stub"
        }
    }

    fn stub_canonicalizer() -> PredicateCanonicalizer {
        PredicateCanonicalizer::new(Arc::new(StubEmbeddingClient { dim: 8 }), test_ontology())
    }

    #[tokio::test]
    async fn classify_category_picks_location_for_moved_to() {
        let pc = stub_canonicalizer();
        let pick = pc
            .classify_category("relocated to", "User relocated to New York")
            .await;
        let (id, score) = pick.expect("classifier returned None");
        assert_eq!(id, "location");
        assert!(score > 0.99, "expected ~1.0 cosine, got {}", score);
    }

    #[tokio::test]
    async fn classify_category_picks_financial_for_payment() {
        let pc = stub_canonicalizer();
        let (id, score) = pc
            .classify_category("paid", "User paid 50 dollars for dinner")
            .await
            .expect("classifier returned None");
        assert_eq!(id, "financial");
        assert!(score > 0.99);
    }

    #[tokio::test]
    async fn classify_category_cached_init_runs_once() {
        let pc = stub_canonicalizer();
        // Two back-to-back calls should reuse the already-initialised
        // category_embeddings map (no panic, both return Some).
        let a = pc
            .classify_category("relocated to", "User relocated to Berlin")
            .await
            .expect("first call");
        let b = pc
            .classify_category("relocated to", "User relocated to Berlin")
            .await
            .expect("second call");
        assert_eq!(a.0, b.0);
    }
}
