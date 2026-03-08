//! Canonical domain registry for temporal state predicates.
//!
//! Maps arbitrary LLM-generated predicates to canonical domain slots with
//! cardinality rules. Predicate canonicalization uses embedding similarity
//! (cosine distance) to match unknown predicates to canonical forms without
//! hardcoded alias lists.

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

/// A canonical domain slot describing a predicate category.
pub struct DomainSlot {
    pub category: &'static str,
    pub canonical_predicate: &'static str,
    pub cardinality: Cardinality,
    pub location_dependent: bool,
}

/// Static registry of known domain slots.
static DOMAIN_SLOTS: &[DomainSlot] = &[
    DomainSlot {
        category: "location",
        canonical_predicate: "lives_in",
        cardinality: Cardinality::Single,
        location_dependent: false,
    },
    DomainSlot {
        category: "work",
        canonical_predicate: "works_at",
        cardinality: Cardinality::Single,
        location_dependent: false,
    },
    DomainSlot {
        category: "education",
        canonical_predicate: "studies_at",
        cardinality: Cardinality::Single,
        location_dependent: false,
    },
    DomainSlot {
        category: "routine",
        canonical_predicate: "",
        cardinality: Cardinality::Multi,
        location_dependent: true,
    },
    DomainSlot {
        category: "preference",
        canonical_predicate: "",
        cardinality: Cardinality::Multi,
        location_dependent: false,
    },
    DomainSlot {
        category: "relationship",
        canonical_predicate: "",
        cardinality: Cardinality::Multi,
        location_dependent: false,
    },
    DomainSlot {
        category: "health",
        canonical_predicate: "",
        cardinality: Cardinality::Multi,
        location_dependent: false,
    },
    DomainSlot {
        category: "financial",
        canonical_predicate: "payment",
        cardinality: Cardinality::Append,
        location_dependent: false,
    },
];

// ---------------------------------------------------------------------------
// Dynamic domain registry
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

/// Unified resolved slot — either a static bootstrap slot or a learned one.
pub enum ResolvedSlot {
    Static(&'static DomainSlot),
    Learned(LearnedSlot),
}

impl ResolvedSlot {
    pub fn cardinality(&self) -> Cardinality {
        match self {
            ResolvedSlot::Static(s) => s.cardinality,
            ResolvedSlot::Learned(l) => l.cardinality,
        }
    }

    pub fn canonical_predicate(&self) -> &str {
        match self {
            ResolvedSlot::Static(s) => s.canonical_predicate,
            ResolvedSlot::Learned(l) => &l.canonical_predicate,
        }
    }

    pub fn location_dependent(&self) -> bool {
        match self {
            ResolvedSlot::Static(s) => s.location_dependent,
            ResolvedSlot::Learned(l) => l.location_dependent,
        }
    }
}

/// Dynamic domain registry that combines static bootstrap slots with learned ones.
///
/// Uses `std::sync::RwLock` (not tokio) since reads happen inside tokio RwLock
/// write guards in `write_fact_to_graph`.
pub struct DomainRegistry {
    learned: std::sync::RwLock<HashMap<String, LearnedSlot>>,
}

impl DomainRegistry {
    pub fn new() -> Self {
        Self {
            learned: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Unified lookup: checks learned slots first, then static `DOMAIN_SLOTS`.
    pub fn resolve(&self, category: &str) -> Option<ResolvedSlot> {
        let cat_lower = category.to_lowercase();
        // Check learned first
        if let Ok(learned) = self.learned.read() {
            if let Some(slot) = learned.get(&cat_lower) {
                return Some(ResolvedSlot::Learned(slot.clone()));
            }
        }
        // Fall back to static
        DOMAIN_SLOTS
            .iter()
            .find(|slot| slot.category == cat_lower)
            .map(ResolvedSlot::Static)
    }

    /// Returns true if the given category is single-valued.
    pub fn is_single_valued(&self, category: &str) -> bool {
        self.resolve(category)
            .map(|s| s.cardinality() == Cardinality::Single)
            .unwrap_or(false)
    }

    /// Returns true if the given category is location-dependent.
    pub fn is_location_dependent(&self, category: &str) -> bool {
        self.resolve(category)
            .map(|s| s.location_dependent())
            .unwrap_or(false)
    }

    /// Synchronous predicate canonicalization via registry.
    pub fn canonicalize_predicate<'a>(
        &self,
        category: &str,
        raw_predicate: &'a str,
    ) -> Cow<'a, str> {
        if let Some(slot) = self.resolve(category) {
            let cp = slot.canonical_predicate();
            if !cp.is_empty() && slot.cardinality() == Cardinality::Single {
                // For learned slots we must return an owned string since the data is not 'static
                return match slot {
                    ResolvedSlot::Static(s) => Cow::Borrowed(s.canonical_predicate),
                    ResolvedSlot::Learned(_) => Cow::Owned(cp.to_string()),
                };
            }
        }
        Cow::Borrowed(raw_predicate)
    }

    /// Register a new learned category. Returns true if newly registered, false if already exists.
    pub fn register_category(
        &self,
        category: &str,
        cardinality: Cardinality,
        canonical_predicate: &str,
        location_dependent: bool,
    ) -> bool {
        let cat_lower = category.to_lowercase();
        // Skip if it's a static category
        if DOMAIN_SLOTS.iter().any(|s| s.category == cat_lower) {
            return false;
        }
        let mut learned = self.learned.write().unwrap();
        if learned.contains_key(&cat_lower) {
            return false;
        }
        learned.insert(
            cat_lower.clone(),
            LearnedSlot {
                category: cat_lower,
                canonical_predicate: canonical_predicate.to_string(),
                cardinality,
                location_dependent,
                discovered_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                usage_count: 1,
            },
        );
        true
    }

    /// Bump usage_count for a learned slot.
    pub fn record_usage(&self, category: &str) {
        let cat_lower = category.to_lowercase();
        if let Ok(mut learned) = self.learned.write() {
            if let Some(slot) = learned.get_mut(&cat_lower) {
                slot.usage_count += 1;
            }
        }
    }

    /// Formatted category block for LLM prompt injection.
    pub fn prompt_category_block(&self) -> String {
        let mut lines = Vec::new();
        for slot in DOMAIN_SLOTS {
            let card = match slot.cardinality {
                Cardinality::Single => "single-valued, newest supersedes all",
                Cardinality::Multi => "multi-valued, multiple active",
                Cardinality::Append => "append-only, never supersede",
            };
            lines.push(format!(
                "- \"{}\": {} ({})",
                slot.category,
                slot_description(slot.category),
                card,
            ));
        }
        if let Ok(learned) = self.learned.read() {
            for slot in learned.values() {
                let card = match slot.cardinality {
                    Cardinality::Single => "single-valued, newest supersedes all",
                    Cardinality::Multi => "multi-valued, multiple active",
                    Cardinality::Append => "append-only, never supersede",
                };
                lines.push(format!(
                    "- \"{}\": learned category ({})",
                    slot.category, card
                ));
            }
        }
        lines.join("\n")
    }

    /// Pipe-separated category list for LLM enum constraint.
    pub fn prompt_category_enum(&self) -> String {
        let mut cats: Vec<String> = DOMAIN_SLOTS
            .iter()
            .map(|s| s.category.to_string())
            .collect();
        cats.push("other".to_string());
        if let Ok(learned) = self.learned.read() {
            for cat in learned.keys() {
                cats.push(cat.clone());
            }
        }
        cats.join("|")
    }

    /// Return all learned slots for persistence.
    pub fn learned_slots(&self) -> Vec<LearnedSlot> {
        self.learned.read().unwrap().values().cloned().collect()
    }

    /// Bulk restore learned slots (e.g., from persisted concept nodes).
    pub fn load_learned(&self, slots: Vec<LearnedSlot>) {
        let mut learned = self.learned.write().unwrap();
        for slot in slots {
            let cat = slot.category.to_lowercase();
            // Don't overwrite static categories
            if DOMAIN_SLOTS.iter().any(|s| s.category == cat) {
                continue;
            }
            learned.insert(cat, slot);
        }
    }
}

/// Human-readable description for static categories (used in prompt generation).
fn slot_description(category: &str) -> &'static str {
    match category {
        "location" => "where someone lives, moved to",
        "work" => "job, employer, role",
        "education" => "school, courses, learning",
        "routine" => "daily habits, regular activities",
        "preference" => "likes, dislikes, favorites",
        "relationship" => "connections between people",
        "health" => "medical, fitness, diet",
        "financial" => "payments, debts, expenses",
        _ => "other",
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

/// Find the domain slot for a category.
pub fn resolve_slot(category: &str, _predicate: &str) -> Option<&'static DomainSlot> {
    let cat_lower = category.to_lowercase();
    DOMAIN_SLOTS.iter().find(|slot| slot.category == cat_lower)
}

/// Synchronous predicate canonicalization (cache-only fast path).
///
/// For Single-valued domains: returns the canonical predicate unconditionally
/// (e.g., any location predicate → "lives_in"). This is the correct fallback
/// when no embedding client is available — all predicates within a single-valued
/// category are semantically equivalent to the canonical.
///
/// For Multi/Append or unknown domains: returns the raw predicate unchanged.
pub fn canonicalize_predicate<'a>(category: &str, raw_predicate: &'a str) -> Cow<'a, str> {
    if let Some(slot) = resolve_slot(category, raw_predicate) {
        if !slot.canonical_predicate.is_empty() && slot.cardinality == Cardinality::Single {
            return Cow::Borrowed(slot.canonical_predicate);
        }
    }
    Cow::Borrowed(raw_predicate)
}

/// Returns true if the given category is location-dependent (cascade-supersede on location change).
pub fn is_location_dependent(category: &str) -> bool {
    let cat_lower = category.to_lowercase();
    DOMAIN_SLOTS
        .iter()
        .any(|slot| slot.category == cat_lower && slot.location_dependent)
}

/// Returns true if the given category is single-valued (newest supersedes all previous).
pub fn is_single_valued(category: &str) -> bool {
    let cat_lower = category.to_lowercase();
    DOMAIN_SLOTS
        .iter()
        .any(|slot| slot.category == cat_lower && slot.cardinality == Cardinality::Single)
}

/// Decode a legacy `"state:*"` association type into `(category, canonical_predicate)`.
///
/// Maps e.g. `"state:location"` → `("location", "lives_in")`,
/// `"state:routine"` → `("routine", "")`, etc.
pub fn decode_legacy_state_assoc(assoc_type: &str) -> Option<(&'static str, &'static str)> {
    let rest = assoc_type.strip_prefix("state:")?;
    let rest_lower = rest.to_lowercase();

    // Direct category match
    for slot in DOMAIN_SLOTS {
        if slot.category == rest_lower {
            return Some((slot.category, slot.canonical_predicate));
        }
    }

    // Legacy names that map to known categories
    match rest_lower.as_str() {
        "landmark" | "activity" | "saturday_routine" | "morning_routine" | "weekend"
        | "breakfast" => Some(("routine", "")),
        _ => None,
    }
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
    /// Create a new canonicalizer with the given embedding client.
    pub fn new(embedding_client: Arc<dyn crate::claims::EmbeddingClient>) -> Self {
        Self {
            embedding_client,
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

        // Collect canonical predicates for single-valued domains (static + learned)
        let to_embed: Vec<(String, String)> = DOMAIN_SLOTS
            .iter()
            .filter(|s| s.cardinality == Cardinality::Single && !s.canonical_predicate.is_empty())
            .map(|s| (s.category.to_string(), s.canonical_predicate.to_string()))
            .collect();

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

        // Note: learned slots with single-valued cardinality and non-empty canonical
        // predicates should also be embedded here when a DomainRegistry reference is
        // available. For now, the PredicateCanonicalizer handles static slots only;
        // learned slots use the synchronous DomainRegistry::canonicalize_predicate path.

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
        let slot = match resolve_slot(category, raw_predicate) {
            Some(s) => s,
            None => return raw_predicate.to_string(),
        };

        // Only canonicalize single-valued domains
        if slot.canonical_predicate.is_empty() || slot.cardinality != Cardinality::Single {
            return raw_predicate.to_string();
        }

        // Exact match — already canonical
        if raw_predicate == slot.canonical_predicate {
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
                return slot.canonical_predicate.to_string();
            },
        };

        // Find best match among canonical embeddings in the same category
        let canonical_embeddings = self.canonical_embeddings.read().await;
        let canonical_key = format!("{}:{}", category, slot.canonical_predicate);

        let result = if let Some(canonical_vec) = canonical_embeddings.get(&canonical_key) {
            let sim = agent_db_core::utils::cosine_similarity(&raw_embedding, canonical_vec);
            tracing::debug!(
                "Predicate similarity '{}:{}' vs canonical '{}': {:.3}",
                category,
                raw_predicate,
                slot.canonical_predicate,
                sim
            );
            if sim >= SIMILARITY_THRESHOLD {
                slot.canonical_predicate.to_string()
            } else {
                raw_predicate.to_string()
            }
        } else {
            // No canonical embedding available — fall back to canonical for single-valued
            slot.canonical_predicate.to_string()
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

    #[test]
    fn test_resolve_slot_location() {
        let slot = resolve_slot("location", "anything").unwrap();
        assert_eq!(slot.category, "location");
        assert_eq!(slot.canonical_predicate, "lives_in");
        assert_eq!(slot.cardinality, Cardinality::Single);
    }

    #[test]
    fn test_canonicalize_single_valued_returns_canonical() {
        // Any predicate in a single-valued domain returns the canonical
        let result = canonicalize_predicate("location", "resides");
        assert_eq!(result.as_ref(), "lives_in");

        let result = canonicalize_predicate("location", "moved_to");
        assert_eq!(result.as_ref(), "lives_in");

        let result = canonicalize_predicate("location", "some_random_verb");
        assert_eq!(result.as_ref(), "lives_in");
    }

    #[test]
    fn test_canonicalize_lives_in_passthrough() {
        let result = canonicalize_predicate("location", "lives_in");
        assert_eq!(result.as_ref(), "lives_in");
    }

    #[test]
    fn test_canonicalize_multi_valued_passthrough() {
        let result = canonicalize_predicate("routine", "morning_yoga");
        assert_eq!(result.as_ref(), "morning_yoga");
    }

    #[test]
    fn test_canonicalize_unknown_category_passthrough() {
        let result = canonicalize_predicate("unknown_cat", "some_pred");
        assert_eq!(result.as_ref(), "some_pred");
    }

    #[test]
    fn test_is_location_dependent() {
        assert!(is_location_dependent("routine"));
        assert!(!is_location_dependent("preference"));
        assert!(!is_location_dependent("location"));
        assert!(!is_location_dependent("financial"));
    }

    #[test]
    fn test_is_single_valued() {
        assert!(is_single_valued("location"));
        assert!(is_single_valued("work"));
        assert!(is_single_valued("education"));
        assert!(!is_single_valued("routine"));
        assert!(!is_single_valued("preference"));
        assert!(!is_single_valued("financial"));
    }

    #[test]
    fn test_decode_legacy_state_assoc() {
        let (cat, pred) = decode_legacy_state_assoc("state:location").unwrap();
        assert_eq!(cat, "location");
        assert_eq!(pred, "lives_in");

        let (cat, pred) = decode_legacy_state_assoc("state:routine").unwrap();
        assert_eq!(cat, "routine");
        assert_eq!(pred, "");

        let (cat, pred) = decode_legacy_state_assoc("state:financial").unwrap();
        assert_eq!(cat, "financial");
        assert_eq!(pred, "payment");
    }

    #[test]
    fn test_decode_legacy_state_assoc_legacy_names() {
        let (cat, _) = decode_legacy_state_assoc("state:landmark").unwrap();
        assert_eq!(cat, "routine");

        let (cat, _) = decode_legacy_state_assoc("state:saturday_routine").unwrap();
        assert_eq!(cat, "routine");

        let (cat, _) = decode_legacy_state_assoc("state:breakfast").unwrap();
        assert_eq!(cat, "routine");
    }

    #[test]
    fn test_decode_legacy_state_assoc_non_state_returns_none() {
        assert!(decode_legacy_state_assoc("location:lives_in").is_none());
        assert!(decode_legacy_state_assoc("routine:morning").is_none());
    }

    #[test]
    fn test_work_canonicalization() {
        let result = canonicalize_predicate("work", "employed_by");
        assert_eq!(result.as_ref(), "works_at");

        let result = canonicalize_predicate("work", "works_for");
        assert_eq!(result.as_ref(), "works_at");
    }

    // Async tests for PredicateCanonicalizer require a mock EmbeddingClient.
    // The mock is defined in crate::claims::MockClient but is test-only.
    // Integration tests for embedding-based canonicalization should use
    // the actual embedding client with an API key.

    #[test]
    fn test_domain_registry_static_resolve() {
        let reg = DomainRegistry::new();
        let slot = reg.resolve("location").unwrap();
        assert_eq!(slot.cardinality(), Cardinality::Single);
        assert_eq!(slot.canonical_predicate(), "lives_in");
        assert!(!slot.location_dependent());
    }

    #[test]
    fn test_domain_registry_register_and_resolve() {
        let reg = DomainRegistry::new();
        assert!(reg.register_category("pets", Cardinality::Multi, "", false));
        // Second register is a no-op
        assert!(!reg.register_category("pets", Cardinality::Single, "", false));
        let slot = reg.resolve("pets").unwrap();
        assert_eq!(slot.cardinality(), Cardinality::Multi);
    }

    #[test]
    fn test_domain_registry_no_override_static() {
        let reg = DomainRegistry::new();
        // Cannot register a static category
        assert!(!reg.register_category("location", Cardinality::Multi, "", false));
        // Still resolves to static
        assert!(reg.is_single_valued("location"));
    }

    #[test]
    fn test_domain_registry_prompt_category_enum() {
        let reg = DomainRegistry::new();
        reg.register_category("pets", Cardinality::Multi, "", false);
        let enum_str = reg.prompt_category_enum();
        assert!(enum_str.contains("location"));
        assert!(enum_str.contains("pets"));
        assert!(enum_str.contains("other"));
    }

    #[test]
    fn test_domain_registry_load_learned() {
        let reg = DomainRegistry::new();
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
    fn test_domain_registry_record_usage() {
        let reg = DomainRegistry::new();
        reg.register_category("pets", Cardinality::Multi, "", false);
        reg.record_usage("pets");
        reg.record_usage("pets");
        let slots = reg.learned_slots();
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].usage_count, 3); // 1 initial + 2 bumps
    }
}
