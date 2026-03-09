//! OWL/RDFS-aligned ontology layer for edge/property behaviors.
//!
//! Replaces ad-hoc `DomainSlot` system with declarative property descriptors
//! loaded from Turtle (.ttl) files. All behavioral decisions (symmetry,
//! cardinality, append-only, cascade) are expressed as metadata — no hardcoded
//! category names in behavioral code.

use crate::structures::ConceptType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// OWL Property Characteristics
// ---------------------------------------------------------------------------

/// OWL-aligned property characteristics.
/// Each flag corresponds to an OWL property axiom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyCharacteristic {
    /// owl:SymmetricProperty — edge implies reverse edge (A→B ⟹ B→A)
    Symmetric,
    /// owl:TransitiveProperty — chains collapse (A→B→C ⟹ A→C)
    Transitive,
    /// owl:FunctionalProperty — at most one value per subject (latest supersedes)
    Functional,
    /// owl:InverseFunctionalProperty — at most one subject per value
    InverseFunctional,
}

// ---------------------------------------------------------------------------
// Property Descriptor (replaces DomainSlot)
// ---------------------------------------------------------------------------

/// OWL/RDFS-aligned property (edge type) descriptor.
///
/// Every behavioral decision previously hardcoded as `if category == "X"`
/// is now expressed as metadata on this descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDescriptor {
    // ── Identity (RDFS) ──
    /// Unique property identifier (e.g., "location", "relationship", "financial")
    pub id: String,
    /// rdfs:label — human-readable name
    pub label: String,
    /// rdfs:comment — description for LLM prompt injection
    pub comment: String,

    // ── Type constraints (RDFS) ──
    /// rdfs:domain — valid ConceptTypes for the source node (empty = any)
    pub domain: Vec<String>,
    /// rdfs:range — valid ConceptTypes for the target node (empty = any)
    pub range: Vec<String>,

    // ── OWL characteristics ──
    /// Set of OWL property characteristics
    pub characteristics: Vec<PropertyCharacteristic>,

    // ── Cardinality (OWL Restriction) ──
    /// owl:maxCardinality per subject (1 = functional/single-valued, None = unlimited)
    pub max_cardinality: Option<u32>,

    // ── Hierarchy (RDFS) ──
    /// rdfs:subPropertyOf — parent property ID (enables query expansion)
    pub sub_property_of: Option<String>,
    /// owl:inverseOf — inverse property ID (e.g., "employs" ↔ "works_at")
    pub inverse_of: Option<String>,

    // ── Behavioral extensions (custom, not standard OWL) ──
    /// Append-only: never supersede existing edges. (financial transactions)
    pub append_only: bool,
    /// When this property's value changes, supersede edges of cascade_dependents.
    pub cascade_dependents: Vec<String>,
    /// This property is superseded when a cascade_trigger property changes.
    pub cascade_dependent: bool,
    /// Skip LLM-based conflict detection for this property.
    pub skip_conflict_detection: bool,
    /// Canonical predicate for edge type construction (e.g., "lives_in" for location)
    pub canonical_predicate: String,
}

impl PropertyDescriptor {
    /// Create a minimal default descriptor for an unknown property.
    pub fn default_for(id: &str) -> Self {
        Self {
            id: id.to_string(),
            label: id.to_string(),
            comment: String::new(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: Vec::new(),
            max_cardinality: None,
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: String::new(),
        }
    }

    pub fn has_characteristic(&self, c: PropertyCharacteristic) -> bool {
        self.characteristics.contains(&c)
    }
}

// ---------------------------------------------------------------------------
// RDFS Class Hierarchy
// ---------------------------------------------------------------------------

/// RDFS-aligned class descriptor for concept type hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDescriptor {
    /// Class identifier (maps to ConceptType variant name)
    pub id: String,
    /// rdfs:label
    pub label: String,
    /// rdfs:subClassOf — parent class IDs
    pub sub_class_of: Vec<String>,
}

/// Class hierarchy for rdfs:domain/range validation.
pub struct ClassHierarchy {
    classes: HashMap<String, ClassDescriptor>,
}

impl ClassHierarchy {
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
        }
    }

    pub fn register(&mut self, desc: ClassDescriptor) {
        self.classes.insert(desc.id.clone(), desc);
    }

    /// Check if `child` is a subclass of `parent` (transitive).
    pub fn is_subclass_of(&self, child: &str, parent: &str) -> bool {
        if child == parent {
            return true;
        }
        let mut current = child;
        let mut visited = std::collections::HashSet::new();
        while visited.insert(current.to_string()) {
            if let Some(desc) = self.classes.get(current) {
                for sup in &desc.sub_class_of {
                    if sup == parent {
                        return true;
                    }
                    // Follow the first parent chain (single inheritance for simplicity)
                    current = sup;
                }
            } else {
                break;
            }
        }
        false
    }

    /// Check if a ConceptType satisfies a domain/range constraint.
    pub fn concept_type_satisfies(&self, actual: &ConceptType, required: &str) -> bool {
        let actual_name = concept_type_to_class_id(actual);
        self.is_subclass_of(&actual_name, required)
    }
}

/// Map ConceptType enum variant to ontology class ID string.
fn concept_type_to_class_id(ct: &ConceptType) -> String {
    match ct {
        ConceptType::Person => "Person".to_string(),
        ConceptType::Organization => "Organization".to_string(),
        ConceptType::Location => "Location".to_string(),
        ConceptType::Product => "Product".to_string(),
        ConceptType::DateTime => "DateTime".to_string(),
        ConceptType::Event => "Event".to_string(),
        ConceptType::NamedEntity => "NamedEntity".to_string(),
        ConceptType::BehaviorPattern => "BehaviorPattern".to_string(),
        ConceptType::CausalPattern => "CausalPattern".to_string(),
        ConceptType::TemporalPattern => "TemporalPattern".to_string(),
        ConceptType::Function => "Function".to_string(),
        ConceptType::Class => "Class".to_string(),
        ConceptType::Module => "Module".to_string(),
        ConceptType::Variable => "Variable".to_string(),
        ConceptType::Interface => "Interface".to_string(),
        ConceptType::Enum => "Enum".to_string(),
        ConceptType::TypeAlias => "TypeAlias".to_string(),
        _ => "Thing".to_string(),
    }
}

/// Map a class ID string back to ConceptType.
fn class_id_to_concept_type(id: &str) -> ConceptType {
    match id {
        "Person" => ConceptType::Person,
        "Organization" => ConceptType::Organization,
        "Location" => ConceptType::Location,
        "Product" => ConceptType::Product,
        "DateTime" => ConceptType::DateTime,
        "Event" => ConceptType::Event,
        "NamedEntity" => ConceptType::NamedEntity,
        "BehaviorPattern" => ConceptType::BehaviorPattern,
        "CausalPattern" => ConceptType::CausalPattern,
        "TemporalPattern" => ConceptType::TemporalPattern,
        "Function" => ConceptType::Function,
        "Class" => ConceptType::Class,
        "Module" => ConceptType::Module,
        "Variable" => ConceptType::Variable,
        "Interface" => ConceptType::Interface,
        "Enum" => ConceptType::Enum,
        "TypeAlias" => ConceptType::TypeAlias,
        _ => ConceptType::NamedEntity,
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Result of domain/range validation.
#[derive(Debug)]
pub enum ValidationResult {
    Valid,
    DomainMismatch {
        expected: Vec<String>,
        actual: String,
    },
    RangeMismatch {
        expected: Vec<String>,
        actual: String,
    },
}

// ---------------------------------------------------------------------------
// Ontology Registry (replaces DomainRegistry)
// ---------------------------------------------------------------------------

/// OWL/RDFS ontology registry. Replaces the ad-hoc DomainRegistry.
///
/// All behavioral queries that previously relied on hardcoded if-chains
/// (`if category == "financial"`) now consult this registry.
pub struct OntologyRegistry {
    /// Property descriptors keyed by ID
    properties: std::sync::RwLock<HashMap<String, PropertyDescriptor>>,
    /// Class hierarchy
    class_hierarchy: ClassHierarchy,
    /// Reverse index: property ID → sub-properties (for query expansion)
    sub_property_index: std::sync::RwLock<HashMap<String, Vec<String>>>,
}

impl OntologyRegistry {
    pub fn new() -> Self {
        Self {
            properties: std::sync::RwLock::new(HashMap::new()),
            class_hierarchy: ClassHierarchy::new(),
            sub_property_index: std::sync::RwLock::new(HashMap::new()),
        }
    }

    // ── Behavioral queries (replace all hardcoded if-chains) ──

    pub fn is_symmetric(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| p.has_characteristic(PropertyCharacteristic::Symmetric))
            .unwrap_or(false)
    }

    pub fn is_transitive(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| p.has_characteristic(PropertyCharacteristic::Transitive))
            .unwrap_or(false)
    }

    pub fn is_functional(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| {
                p.has_characteristic(PropertyCharacteristic::Functional)
                    || p.max_cardinality == Some(1)
            })
            .unwrap_or(false)
    }

    pub fn is_append_only(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| p.append_only)
            .unwrap_or(false)
    }

    pub fn skip_conflict_detection(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| p.skip_conflict_detection)
            .unwrap_or(false)
    }

    pub fn triggers_cascade(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| !p.cascade_dependents.is_empty())
            .unwrap_or(false)
    }

    pub fn cascade_dependents(&self, property_id: &str) -> Vec<String> {
        self.resolve(property_id)
            .map(|p| p.cascade_dependents.clone())
            .unwrap_or_default()
    }

    pub fn is_cascade_dependent(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| p.cascade_dependent)
            .unwrap_or(false)
    }

    pub fn range_types(&self, property_id: &str) -> Vec<String> {
        self.resolve(property_id)
            .map(|p| p.range.clone())
            .unwrap_or_default()
    }

    /// Return the default ConceptType for the target node of this property.
    /// Uses the first range type if available, otherwise NamedEntity.
    pub fn default_target_concept_type(&self, property_id: &str) -> ConceptType {
        let ranges = self.range_types(property_id);
        if let Some(first) = ranges.first() {
            class_id_to_concept_type(first)
        } else {
            ConceptType::NamedEntity
        }
    }

    /// Returns true if this property is single-valued (functional or max_cardinality == 1).
    pub fn is_single_valued(&self, property_id: &str) -> bool {
        self.resolve(property_id)
            .map(|p| {
                p.has_characteristic(PropertyCharacteristic::Functional)
                    || p.max_cardinality == Some(1)
            })
            .unwrap_or(false)
    }

    /// Returns true if this property is location-dependent (cascade-dependent on location).
    /// Backward compat with DomainRegistry::is_location_dependent.
    pub fn is_location_dependent(&self, property_id: &str) -> bool {
        self.is_cascade_dependent(property_id)
    }

    /// Returns the canonical predicate for a property.
    pub fn canonical_predicate(&self, property_id: &str) -> String {
        self.resolve(property_id)
            .map(|p| {
                if p.canonical_predicate.is_empty() {
                    String::new()
                } else {
                    p.canonical_predicate.clone()
                }
            })
            .unwrap_or_default()
    }

    // ── Edge type utilities ──

    /// Build an edge type string from property ID and predicate.
    /// E.g., build_edge_type("financial", "payment") → "financial:payment"
    pub fn build_edge_type(&self, property_id: &str, predicate: &str) -> String {
        format!("{}:{}", property_id, predicate)
    }

    /// Parse the category prefix from an association type string.
    /// E.g., "financial:payment" → Some("financial")
    pub fn parse_edge_category(assoc_type: &str) -> Option<&str> {
        assoc_type.split(':').next()
    }

    // ── Query expansion (rdfs:subPropertyOf) ──

    /// Returns property_id + all sub-properties (transitive).
    /// E.g., "relationship" → ["relationship", "colleague", "friend", ...]
    pub fn expand_property(&self, property_id: &str) -> Vec<String> {
        let mut result = vec![property_id.to_string()];
        if let Ok(index) = self.sub_property_index.read() {
            if let Some(subs) = index.get(property_id) {
                for sub in subs {
                    // Recursively expand
                    let sub_expanded = self.expand_property(sub);
                    for s in sub_expanded {
                        if !result.contains(&s) {
                            result.push(s);
                        }
                    }
                }
            }
        }
        result
    }

    // ── Inverse resolution (owl:inverseOf) ──

    pub fn inverse_of(&self, property_id: &str) -> Option<String> {
        self.resolve(property_id).and_then(|p| p.inverse_of.clone())
    }

    // ── Domain/range validation ──

    pub fn validate_edge(
        &self,
        source_type: &ConceptType,
        target_type: &ConceptType,
        property_id: &str,
    ) -> ValidationResult {
        if let Some(prop) = self.resolve(property_id) {
            // Check domain
            if !prop.domain.is_empty() {
                let source_class = concept_type_to_class_id(source_type);
                let satisfies = prop
                    .domain
                    .iter()
                    .any(|d| self.class_hierarchy.is_subclass_of(&source_class, d));
                if !satisfies {
                    return ValidationResult::DomainMismatch {
                        expected: prop.domain.clone(),
                        actual: source_class,
                    };
                }
            }
            // Check range
            if !prop.range.is_empty() {
                let target_class = concept_type_to_class_id(target_type);
                let satisfies = prop
                    .range
                    .iter()
                    .any(|r| self.class_hierarchy.is_subclass_of(&target_class, r));
                if !satisfies {
                    return ValidationResult::RangeMismatch {
                        expected: prop.range.clone(),
                        actual: target_class,
                    };
                }
            }
        }
        ValidationResult::Valid
    }

    // ── Cardinality ──

    pub fn max_cardinality(&self, property_id: &str) -> Option<u32> {
        self.resolve(property_id).and_then(|p| p.max_cardinality)
    }

    // ── LLM prompt generation ──

    /// Formatted category block for LLM prompt injection.
    pub fn prompt_category_block(&self) -> String {
        let props = self.properties.read().unwrap();
        let mut lines = Vec::new();
        for prop in props.values() {
            let card = if prop.append_only {
                "append-only, never supersede"
            } else if prop.has_characteristic(PropertyCharacteristic::Functional)
                || prop.max_cardinality == Some(1)
            {
                "single-valued, newest supersedes all"
            } else {
                "multi-valued, multiple active"
            };
            let desc = if prop.comment.is_empty() {
                &prop.label
            } else {
                &prop.comment
            };
            lines.push(format!("- \"{}\": {} ({})", prop.id, desc, card));
        }
        lines.sort(); // deterministic output
        lines.join("\n")
    }

    /// Pipe-separated category list for LLM enum constraint.
    pub fn prompt_category_enum(&self) -> String {
        let props = self.properties.read().unwrap();
        let mut cats: Vec<String> = props.keys().cloned().collect();
        cats.sort();
        cats.push("other".to_string());
        cats.join("|")
    }

    // ── Runtime registration ──

    /// Register a property descriptor. Returns true if newly registered.
    pub fn register_property(&self, descriptor: PropertyDescriptor) -> bool {
        let id = descriptor.id.clone();
        let sub_prop = descriptor.sub_property_of.clone();

        let mut props = self.properties.write().unwrap();
        if props.contains_key(&id) {
            return false;
        }
        props.insert(id.clone(), descriptor);
        drop(props);

        // Update sub-property index
        if let Some(parent) = sub_prop {
            let mut index = self.sub_property_index.write().unwrap();
            index.entry(parent).or_default().push(id);
        }
        true
    }

    /// Resolve a property descriptor by ID.
    pub fn resolve(&self, property_id: &str) -> Option<PropertyDescriptor> {
        let props = self.properties.read().unwrap();
        props.get(property_id).cloned()
    }

    /// Return all property IDs that have a given characteristic.
    pub fn properties_with_characteristic(&self, c: PropertyCharacteristic) -> Vec<String> {
        let props = self.properties.read().unwrap();
        props
            .values()
            .filter(|p| p.has_characteristic(c))
            .map(|p| p.id.clone())
            .collect()
    }

    /// Return all registered property IDs.
    pub fn all_property_ids(&self) -> Vec<String> {
        let props = self.properties.read().unwrap();
        props.keys().cloned().collect()
    }

    /// Build indices after loading (sub-property reverse index).
    fn build_indices(&self) {
        let props = self.properties.read().unwrap();
        let mut index = self.sub_property_index.write().unwrap();
        index.clear();
        for prop in props.values() {
            if let Some(ref parent) = prop.sub_property_of {
                index
                    .entry(parent.clone())
                    .or_default()
                    .push(prop.id.clone());
            }
        }
    }

    // ── Backward compatibility with DomainRegistry ──

    /// Synchronous predicate canonicalization.
    /// For single-valued properties: returns the canonical predicate.
    /// For multi-valued or unknown: returns the raw predicate unchanged.
    pub fn canonicalize_predicate<'a>(
        &self,
        property_id: &str,
        raw_predicate: &'a str,
    ) -> std::borrow::Cow<'a, str> {
        if let Some(prop) = self.resolve(property_id) {
            if !prop.canonical_predicate.is_empty() && self.is_single_valued(property_id) {
                return std::borrow::Cow::Owned(prop.canonical_predicate);
            }
        }
        std::borrow::Cow::Borrowed(raw_predicate)
    }

    /// Register a category with DomainRegistry-style parameters.
    /// Returns true if newly registered.
    pub fn register_category(
        &self,
        category: &str,
        cardinality: crate::domain_schema::Cardinality,
        canonical_predicate: &str,
        location_dependent: bool,
    ) -> bool {
        let cat_lower = category.to_lowercase();
        // Check if already exists
        if self.resolve(&cat_lower).is_some() {
            return false;
        }

        let mut chars = Vec::new();
        let max_card = match cardinality {
            crate::domain_schema::Cardinality::Single => {
                chars.push(PropertyCharacteristic::Functional);
                Some(1)
            },
            crate::domain_schema::Cardinality::Multi => None,
            crate::domain_schema::Cardinality::Append => None,
        };

        let descriptor = PropertyDescriptor {
            id: cat_lower,
            label: category.to_string(),
            comment: String::new(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: chars,
            max_cardinality: max_card,
            sub_property_of: None,
            inverse_of: None,
            append_only: matches!(cardinality, crate::domain_schema::Cardinality::Append),
            cascade_dependents: Vec::new(),
            cascade_dependent: location_dependent,
            skip_conflict_detection: matches!(
                cardinality,
                crate::domain_schema::Cardinality::Append
            ),
            canonical_predicate: canonical_predicate.to_string(),
        };
        self.register_property(descriptor)
    }

    /// Record usage (no-op for ontology — kept for DomainRegistry compat).
    pub fn record_usage(&self, _category: &str) {}

    /// Returns (property_id, canonical_predicate) pairs for all single-valued
    /// properties with non-empty canonical predicates.
    /// Used by PredicateCanonicalizer to embed canonical predicates.
    pub fn single_valued_canonicals(&self) -> Vec<(String, String)> {
        let props = self.properties.read().unwrap();
        props
            .values()
            .filter(|p| self.is_single_valued(&p.id) && !p.canonical_predicate.is_empty())
            .map(|p| (p.id.clone(), p.canonical_predicate.clone()))
            .collect()
    }

    /// Decode a legacy `"state:*"` association type into `(category, canonical_predicate)`.
    ///
    /// Maps e.g. `"state:location"` → `("location", "lives_in")` using ontology data.
    /// Falls back to hardcoded legacy name mapping for backward compatibility with
    /// pre-ontology stored edges.
    pub fn decode_legacy_state_assoc(&self, assoc_type: &str) -> Option<(String, String)> {
        let rest = assoc_type.strip_prefix("state:")?;
        let rest_lower = rest.to_lowercase();

        // Try ontology lookup first
        if let Some(desc) = self.resolve(&rest_lower) {
            return Some((rest_lower, desc.canonical_predicate.clone()));
        }

        // Legacy names that map to known categories (backward compat for stored data)
        match rest_lower.as_str() {
            "landmark" | "activity" | "saturday_routine" | "morning_routine" | "weekend"
            | "breakfast" => Some(("routine".to_string(), String::new())),
            _ => None,
        }
    }

    /// Return learned slots for persistence (backward compat).
    pub fn learned_slots(&self) -> Vec<crate::domain_schema::LearnedSlot> {
        // In the ontology model, all properties are equal. We return
        // non-bootstrap properties as "learned" for backward compat.
        let props = self.properties.read().unwrap();
        let bootstrap = [
            "location",
            "work",
            "education",
            "routine",
            "preference",
            "relationship",
            "health",
            "financial",
        ];
        props
            .values()
            .filter(|p| !bootstrap.contains(&p.id.as_str()))
            .map(|p| {
                let cardinality = if p.append_only {
                    crate::domain_schema::Cardinality::Append
                } else if self.is_single_valued(&p.id) {
                    crate::domain_schema::Cardinality::Single
                } else {
                    crate::domain_schema::Cardinality::Multi
                };
                crate::domain_schema::LearnedSlot {
                    category: p.id.clone(),
                    canonical_predicate: p.canonical_predicate.clone(),
                    cardinality,
                    location_dependent: p.cascade_dependent,
                    discovered_at: 0,
                    usage_count: 0,
                }
            })
            .collect()
    }

    /// Bulk load learned slots (backward compat with DomainRegistry::load_learned).
    pub fn load_learned(&self, slots: Vec<crate::domain_schema::LearnedSlot>) {
        for slot in slots {
            self.register_category(
                &slot.category,
                slot.cardinality,
                &slot.canonical_predicate,
                slot.location_dependent,
            );
        }
    }

    // ── Turtle file loading ──

    /// Load all .ttl files from the given directory.
    pub fn load_from_directory(path: &Path) -> Result<Self, OntologyError> {
        let mut registry = Self::new();
        registry.load_bootstrap_classes();

        if !path.exists() {
            tracing::info!(
                "Ontology directory {:?} not found, starting with empty ontology",
                path
            );
            return Ok(registry);
        }

        let entries = std::fs::read_dir(path).map_err(OntologyError::Io)?;
        for entry in entries {
            let entry = entry.map_err(OntologyError::Io)?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("ttl") {
                let content = std::fs::read_to_string(entry.path()).map_err(OntologyError::Io)?;
                if let Err(e) = registry.load_turtle(&content) {
                    tracing::warn!("Failed to parse {:?}: {}", entry.path(), e);
                }
            }
        }
        registry.build_indices();
        Ok(registry)
    }

    /// Load bootstrap RDFS class hierarchy.
    fn load_bootstrap_classes(&mut self) {
        let classes = vec![
            ("Thing", "", "Thing"),
            ("NamedEntity", "Thing", "Named Entity"),
            ("Person", "NamedEntity", "Person"),
            ("Organization", "NamedEntity", "Organization"),
            ("Location", "NamedEntity", "Location"),
            ("Product", "NamedEntity", "Product"),
            ("Event", "NamedEntity", "Event"),
            ("DateTime", "NamedEntity", "Date/Time"),
            ("Pattern", "Thing", "Pattern"),
            ("BehaviorPattern", "Pattern", "Behavior Pattern"),
            ("CausalPattern", "Pattern", "Causal Pattern"),
            ("TemporalPattern", "Pattern", "Temporal Pattern"),
            ("CodeEntity", "Thing", "Code Entity"),
            ("Function", "CodeEntity", "Function"),
            ("Class", "CodeEntity", "Class"),
            ("Module", "CodeEntity", "Module"),
            ("Variable", "CodeEntity", "Variable"),
            ("Interface", "CodeEntity", "Interface"),
            ("Enum", "CodeEntity", "Enum"),
            ("TypeAlias", "CodeEntity", "Type Alias"),
        ];
        for (id, parent, label) in classes {
            self.class_hierarchy.register(ClassDescriptor {
                id: id.to_string(),
                label: label.to_string(),
                sub_class_of: if parent.is_empty() {
                    Vec::new()
                } else {
                    vec![parent.to_string()]
                },
            });
        }
    }

    /// Parse a Turtle string and extract PropertyDescriptors and ClassDescriptors.
    fn load_turtle(&mut self, ttl: &str) -> Result<(), OntologyError> {
        use rio_api::model::{Literal, Term};
        use rio_api::parser::TriplesParser;
        use rio_turtle::TurtleParser;

        let eg = "http://eventgraphdb.local/ontology#";
        let owl = "http://www.w3.org/2002/07/owl#";
        let rdfs = "http://www.w3.org/2000/01/rdf-schema#";
        let rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

        // Collect all triples first
        struct Triple {
            subject: String,
            predicate: String,
            object: String,
        }
        let mut triples = Vec::new();

        let base_iri = oxiri::Iri::parse("http://eventgraphdb.local/ontology".to_string())
            .map_err(|e| OntologyError::Parse(format!("Invalid base IRI: {}", e)))?;
        let mut parser = TurtleParser::new(ttl.as_bytes(), Some(base_iri));

        parser
            .parse_all(&mut |t| {
                let subject = t.subject.to_string();
                let predicate = t.predicate.iri.to_string();
                let object = match t.object {
                    Term::NamedNode(n) => n.iri.to_string(),
                    Term::Literal(Literal::Simple { value }) => value.to_string(),
                    Term::Literal(Literal::LanguageTaggedString { value, .. }) => value.to_string(),
                    Term::Literal(Literal::Typed { value, .. }) => value.to_string(),
                    Term::BlankNode(b) => b.id.to_string(),
                    _ => String::new(),
                };
                triples.push(Triple {
                    subject,
                    predicate,
                    object,
                });
                Ok(()) as Result<(), rio_turtle::TurtleError>
            })
            .map_err(|e| OntologyError::Parse(format!("{}", e)))?;

        // Group triples by subject
        let mut subjects: HashMap<String, Vec<(&str, &str)>> = HashMap::new();
        for t in &triples {
            subjects
                .entry(t.subject.clone())
                .or_default()
                .push((&t.predicate, &t.object));
        }

        let rdf_type = format!("{}type", rdf);
        let owl_obj_prop = format!("{}ObjectProperty", owl);
        let owl_sym = format!("{}SymmetricProperty", owl);
        let owl_trans = format!("{}TransitiveProperty", owl);
        let owl_func = format!("{}FunctionalProperty", owl);
        let owl_inv_func = format!("{}InverseFunctionalProperty", owl);
        let rdfs_class = format!("{}Class", rdfs);
        let rdfs_label = format!("{}label", rdfs);
        let rdfs_comment = format!("{}comment", rdfs);
        let rdfs_domain = format!("{}domain", rdfs);
        let rdfs_range = format!("{}range", rdfs);
        let rdfs_sub_prop = format!("{}subPropertyOf", rdfs);
        let rdfs_sub_class = format!("{}subClassOf", rdfs);
        let owl_inverse = format!("{}inverseOf", owl);
        let owl_max_card = format!("{}maxCardinality", owl);
        let eg_canonical = format!("{}canonicalPredicate", eg);
        let eg_append = format!("{}appendOnly", eg);
        let eg_skip_conflict = format!("{}skipConflictDetection", eg);
        let eg_cascade_deps = format!("{}cascadeDependents", eg);
        let eg_cascade_dep = format!("{}cascadeDependent", eg);

        for (subject, predicates) in &subjects {
            // Extract local name from URI
            let local_name = extract_local_name(subject);
            if local_name.is_empty() {
                continue;
            }

            let types: Vec<&str> = predicates
                .iter()
                .filter(|(p, _)| *p == rdf_type.as_str())
                .map(|(_, o)| *o)
                .collect();

            let is_property = types.iter().any(|t| *t == owl_obj_prop);
            let is_class = types.iter().any(|t| *t == rdfs_class);

            if is_property {
                let mut desc = PropertyDescriptor::default_for(&local_name);

                // OWL characteristics from types
                if types.iter().any(|t| *t == owl_sym) {
                    desc.characteristics.push(PropertyCharacteristic::Symmetric);
                }
                if types.iter().any(|t| *t == owl_trans) {
                    desc.characteristics
                        .push(PropertyCharacteristic::Transitive);
                }
                if types.iter().any(|t| *t == owl_func) {
                    desc.characteristics
                        .push(PropertyCharacteristic::Functional);
                    desc.max_cardinality = Some(1);
                }
                if types.iter().any(|t| *t == owl_inv_func) {
                    desc.characteristics
                        .push(PropertyCharacteristic::InverseFunctional);
                }

                // RDFS predicates
                for (pred, obj) in predicates {
                    if *pred == rdfs_label {
                        desc.label = obj.to_string();
                    } else if *pred == rdfs_comment {
                        desc.comment = obj.to_string();
                    } else if *pred == rdfs_domain {
                        desc.domain.push(extract_local_name(obj));
                    } else if *pred == rdfs_range {
                        desc.range.push(extract_local_name(obj));
                    } else if *pred == rdfs_sub_prop {
                        desc.sub_property_of = Some(extract_local_name(obj));
                    } else if *pred == owl_inverse {
                        desc.inverse_of = Some(extract_local_name(obj));
                    } else if *pred == owl_max_card {
                        desc.max_cardinality = obj.parse().ok();
                    } else if *pred == eg_canonical {
                        desc.canonical_predicate = obj.to_string();
                    } else if *pred == eg_append {
                        desc.append_only = *obj == "true";
                    } else if *pred == eg_skip_conflict {
                        desc.skip_conflict_detection = *obj == "true";
                    } else if *pred == eg_cascade_dep {
                        desc.cascade_dependent = *obj == "true";
                    } else if *pred == eg_cascade_deps {
                        // Cascade dependents — the object is a local name
                        let dep = extract_local_name(obj);
                        if !dep.is_empty() {
                            desc.cascade_dependents.push(dep);
                        }
                    }
                }

                self.register_property(desc);
            } else if is_class {
                let mut class = ClassDescriptor {
                    id: local_name,
                    label: String::new(),
                    sub_class_of: Vec::new(),
                };
                for (pred, obj) in predicates {
                    if *pred == rdfs_label {
                        class.label = obj.to_string();
                    } else if *pred == rdfs_sub_class {
                        class.sub_class_of.push(extract_local_name(obj));
                    }
                }
                self.class_hierarchy.register(class);
            }
        }

        Ok(())
    }
}

/// Extract the local name from a URI or prefixed name.
/// E.g., "<http://eventgraphdb.local/ontology#location>" → "location"
/// E.g., "http://eventgraphdb.local/ontology#location" → "location"
fn extract_local_name(uri: &str) -> String {
    // Strip angle brackets if present
    let uri = uri.trim_start_matches('<').trim_end_matches('>');
    // Try fragment (#)
    if let Some(pos) = uri.rfind('#') {
        return uri[pos + 1..].to_string();
    }
    // Try last path segment
    if let Some(pos) = uri.rfind('/') {
        return uri[pos + 1..].to_string();
    }
    uri.to_string()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum OntologyError {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for OntologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OntologyError::Io(e) => write!(f, "IO error: {}", e),
            OntologyError::Parse(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for OntologyError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> OntologyRegistry {
        let reg = OntologyRegistry::new();

        // location: functional, cascades routine
        reg.register_property(PropertyDescriptor {
            id: "location".into(),
            label: "Location".into(),
            comment: "where someone lives, moved to".into(),
            domain: vec!["Person".into(), "NamedEntity".into()],
            range: vec!["Location".into()],
            characteristics: vec![PropertyCharacteristic::Functional],
            max_cardinality: Some(1),
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: vec!["routine".into()],
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: "lives_in".into(),
        });

        // relationship: symmetric
        reg.register_property(PropertyDescriptor {
            id: "relationship".into(),
            label: "Relationship".into(),
            comment: "connections between people".into(),
            domain: vec!["Person".into(), "NamedEntity".into()],
            range: vec!["Person".into(), "NamedEntity".into()],
            characteristics: vec![PropertyCharacteristic::Symmetric],
            max_cardinality: None,
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: String::new(),
        });

        // colleague: symmetric, sub-property of relationship
        reg.register_property(PropertyDescriptor {
            id: "colleague".into(),
            label: "Colleague".into(),
            comment: "workplace connections".into(),
            domain: vec!["Person".into()],
            range: vec!["Person".into()],
            characteristics: vec![PropertyCharacteristic::Symmetric],
            max_cardinality: None,
            sub_property_of: Some("relationship".into()),
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: String::new(),
        });

        // financial: append-only, skip conflict detection
        reg.register_property(PropertyDescriptor {
            id: "financial".into(),
            label: "Financial".into(),
            comment: "payments, debts, expenses".into(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: Vec::new(),
            max_cardinality: None,
            sub_property_of: None,
            inverse_of: None,
            append_only: true,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: true,
            canonical_predicate: "payment".into(),
        });

        // routine: cascade-dependent
        reg.register_property(PropertyDescriptor {
            id: "routine".into(),
            label: "Routine".into(),
            comment: "daily habits, regular activities".into(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: Vec::new(),
            max_cardinality: None,
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: true,
            skip_conflict_detection: false,
            canonical_predicate: String::new(),
        });

        // work: functional, symmetric
        reg.register_property(PropertyDescriptor {
            id: "work".into(),
            label: "Work".into(),
            comment: "job, employer, role".into(),
            domain: vec!["Person".into()],
            range: vec!["Organization".into(), "NamedEntity".into()],
            characteristics: vec![
                PropertyCharacteristic::Functional,
                PropertyCharacteristic::Symmetric,
            ],
            max_cardinality: Some(1),
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: "works_at".into(),
        });

        reg
    }

    #[test]
    fn test_symmetric_property() {
        let reg = test_registry();
        assert!(reg.is_symmetric("relationship"));
        assert!(reg.is_symmetric("colleague"));
        assert!(reg.is_symmetric("work"));
        assert!(!reg.is_symmetric("location"));
        assert!(!reg.is_symmetric("financial"));
        assert!(!reg.is_symmetric("unknown"));
    }

    #[test]
    fn test_functional_property() {
        let reg = test_registry();
        assert!(reg.is_functional("location"));
        assert!(reg.is_functional("work"));
        assert!(!reg.is_functional("relationship"));
        assert!(!reg.is_functional("financial"));
    }

    #[test]
    fn test_append_only_property() {
        let reg = test_registry();
        assert!(reg.is_append_only("financial"));
        assert!(!reg.is_append_only("location"));
        assert!(!reg.is_append_only("relationship"));
    }

    #[test]
    fn test_skip_conflict_detection() {
        let reg = test_registry();
        assert!(reg.skip_conflict_detection("financial"));
        assert!(!reg.skip_conflict_detection("location"));
    }

    #[test]
    fn test_cascade_triggers() {
        let reg = test_registry();
        assert!(reg.triggers_cascade("location"));
        assert!(!reg.triggers_cascade("financial"));
        assert_eq!(reg.cascade_dependents("location"), vec!["routine"]);
    }

    #[test]
    fn test_cascade_dependent() {
        let reg = test_registry();
        assert!(reg.is_cascade_dependent("routine"));
        assert!(!reg.is_cascade_dependent("location"));
    }

    #[test]
    fn test_single_valued() {
        let reg = test_registry();
        assert!(reg.is_single_valued("location"));
        assert!(reg.is_single_valued("work"));
        assert!(!reg.is_single_valued("routine"));
        assert!(!reg.is_single_valued("financial"));
    }

    #[test]
    fn test_default_target_concept_type() {
        let reg = test_registry();
        assert!(matches!(
            reg.default_target_concept_type("location"),
            ConceptType::Location
        ));
        assert!(matches!(
            reg.default_target_concept_type("relationship"),
            ConceptType::Person
        ));
        assert!(matches!(
            reg.default_target_concept_type("financial"),
            ConceptType::NamedEntity
        ));
    }

    #[test]
    fn test_sub_property_expansion() {
        let reg = test_registry();
        // Must rebuild indices after registration
        reg.build_indices();
        let expanded = reg.expand_property("relationship");
        assert!(expanded.contains(&"relationship".to_string()));
        assert!(expanded.contains(&"colleague".to_string()));
    }

    #[test]
    fn test_build_edge_type() {
        let reg = test_registry();
        assert_eq!(
            reg.build_edge_type("financial", "payment"),
            "financial:payment"
        );
        assert_eq!(
            reg.build_edge_type("relationship", "friend"),
            "relationship:friend"
        );
    }

    #[test]
    fn test_parse_edge_category() {
        assert_eq!(
            OntologyRegistry::parse_edge_category("financial:payment"),
            Some("financial")
        );
        assert_eq!(
            OntologyRegistry::parse_edge_category("relationship:friend"),
            Some("relationship")
        );
    }

    #[test]
    fn test_unknown_property_safe_defaults() {
        let reg = test_registry();
        assert!(!reg.is_symmetric("nonexistent"));
        assert!(!reg.is_functional("nonexistent"));
        assert!(!reg.is_append_only("nonexistent"));
        assert!(!reg.skip_conflict_detection("nonexistent"));
        assert!(!reg.triggers_cascade("nonexistent"));
        assert!(matches!(
            reg.default_target_concept_type("nonexistent"),
            ConceptType::NamedEntity
        ));
    }

    #[test]
    fn test_canonicalize_predicate() {
        let reg = test_registry();
        // Single-valued → returns canonical
        assert_eq!(
            reg.canonicalize_predicate("location", "moved_to").as_ref(),
            "lives_in"
        );
        // Multi-valued → returns raw
        assert_eq!(
            reg.canonicalize_predicate("routine", "morning_yoga")
                .as_ref(),
            "morning_yoga"
        );
        // Unknown → returns raw
        assert_eq!(reg.canonicalize_predicate("unknown", "foo").as_ref(), "foo");
    }

    #[test]
    fn test_register_category_compat() {
        let reg = OntologyRegistry::new();
        assert!(reg.register_category("pets", crate::domain_schema::Cardinality::Multi, "", false));
        // Already exists
        assert!(!reg.register_category(
            "pets",
            crate::domain_schema::Cardinality::Single,
            "",
            false
        ));
        assert!(!reg.is_single_valued("pets"));
    }

    #[test]
    fn test_validate_edge_valid() {
        let mut reg = OntologyRegistry::new();
        reg.load_bootstrap_classes();
        reg.register_property(PropertyDescriptor {
            id: "location".into(),
            label: "Location".into(),
            comment: String::new(),
            domain: vec!["Person".into()],
            range: vec!["Location".into()],
            characteristics: vec![],
            max_cardinality: Some(1),
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: "lives_in".into(),
        });

        let result = reg.validate_edge(&ConceptType::Person, &ConceptType::Location, "location");
        assert!(matches!(result, ValidationResult::Valid));
    }

    #[test]
    fn test_validate_edge_domain_mismatch() {
        let mut reg = OntologyRegistry::new();
        reg.load_bootstrap_classes();
        reg.register_property(PropertyDescriptor {
            id: "location".into(),
            label: "Location".into(),
            comment: String::new(),
            domain: vec!["Person".into()],
            range: vec!["Location".into()],
            characteristics: vec![],
            max_cardinality: Some(1),
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: "lives_in".into(),
        });

        // Location is not a Person
        let result = reg.validate_edge(&ConceptType::Location, &ConceptType::Location, "location");
        assert!(matches!(result, ValidationResult::DomainMismatch { .. }));
    }

    #[test]
    fn test_class_hierarchy_subclass() {
        let mut reg = OntologyRegistry::new();
        reg.load_bootstrap_classes();

        assert!(reg.class_hierarchy.is_subclass_of("Person", "NamedEntity"));
        assert!(reg.class_hierarchy.is_subclass_of("Person", "Thing"));
        assert!(reg
            .class_hierarchy
            .is_subclass_of("BehaviorPattern", "Pattern"));
        assert!(reg
            .class_hierarchy
            .is_subclass_of("BehaviorPattern", "Thing"));
        assert!(!reg.class_hierarchy.is_subclass_of("Person", "Location"));
    }

    #[test]
    fn test_prompt_category_enum() {
        let reg = test_registry();
        let enum_str = reg.prompt_category_enum();
        assert!(enum_str.contains("location"));
        assert!(enum_str.contains("financial"));
        assert!(enum_str.contains("other"));
    }

    #[test]
    fn test_inverse_of() {
        let reg = OntologyRegistry::new();
        reg.register_property(PropertyDescriptor {
            id: "employs".into(),
            label: "Employs".into(),
            comment: String::new(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: Vec::new(),
            max_cardinality: None,
            sub_property_of: None,
            inverse_of: Some("works_at".into()),
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: String::new(),
        });
        assert_eq!(reg.inverse_of("employs"), Some("works_at".to_string()));
        assert_eq!(reg.inverse_of("unknown"), None);
    }

    #[test]
    fn test_turtle_loading() {
        let ttl = r#"
@prefix eg:   <http://eventgraphdb.local/ontology#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

eg:testprop a owl:ObjectProperty ;
    rdfs:label "Test Property" ;
    rdfs:comment "a test property" ;
    rdfs:domain eg:Person ;
    rdfs:range eg:Location ;
    a owl:FunctionalProperty ;
    a owl:SymmetricProperty ;
    eg:canonicalPredicate "test_pred" ;
    eg:appendOnly "true" ;
    eg:skipConflictDetection "true" .
"#;
        let mut reg = OntologyRegistry::new();
        reg.load_bootstrap_classes();
        reg.load_turtle(ttl).expect("TTL parse failed");

        let prop = reg.resolve("testprop").expect("Property not found");
        assert_eq!(prop.label, "Test Property");
        assert_eq!(prop.comment, "a test property");
        assert!(prop.has_characteristic(PropertyCharacteristic::Functional));
        assert!(prop.has_characteristic(PropertyCharacteristic::Symmetric));
        assert_eq!(prop.max_cardinality, Some(1));
        assert_eq!(prop.canonical_predicate, "test_pred");
        assert!(prop.append_only);
        assert!(prop.skip_conflict_detection);
        assert_eq!(prop.domain, vec!["Person".to_string()]);
        assert_eq!(prop.range, vec!["Location".to_string()]);
    }

    #[test]
    fn test_turtle_sub_property() {
        let ttl = r#"
@prefix eg:   <http://eventgraphdb.local/ontology#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

eg:parent a owl:ObjectProperty ;
    rdfs:label "Parent" ;
    a owl:SymmetricProperty .

eg:child a owl:ObjectProperty ;
    rdfs:label "Child" ;
    rdfs:subPropertyOf eg:parent ;
    a owl:SymmetricProperty .
"#;
        let mut reg = OntologyRegistry::new();
        reg.load_turtle(ttl).expect("TTL parse failed");
        reg.build_indices();

        let expanded = reg.expand_property("parent");
        assert!(expanded.contains(&"parent".to_string()));
        assert!(expanded.contains(&"child".to_string()));

        // Child inherits symmetric from its own declaration
        assert!(reg.is_symmetric("child"));
    }

    #[test]
    fn test_load_from_directory() {
        let reg =
            OntologyRegistry::load_from_directory(std::path::Path::new("../../data/ontology"))
                .expect("Failed to load ontology from directory");

        // Check that bootstrap properties were loaded
        assert!(
            reg.resolve("location").is_some(),
            "location property missing"
        );
        assert!(
            reg.resolve("financial").is_some(),
            "financial property missing"
        );
        assert!(
            reg.resolve("relationship").is_some(),
            "relationship property missing"
        );

        // Verify behaviors from TTL files
        assert!(
            reg.is_single_valued("location"),
            "location should be single-valued"
        );
        assert!(
            reg.is_append_only("financial"),
            "financial should be append-only"
        );
        assert!(
            reg.is_symmetric("relationship"),
            "relationship should be symmetric"
        );
        assert!(
            reg.skip_conflict_detection("financial"),
            "financial should skip conflict detection"
        );
        assert!(
            reg.triggers_cascade("location"),
            "location should trigger cascade"
        );

        // Sub-property expansion
        let expanded = reg.expand_property("relationship");
        assert!(
            expanded.contains(&"colleague".to_string()),
            "colleague should be sub-property of relationship"
        );

        // Financial sub-properties
        let fin_expanded = reg.expand_property("financial");
        assert!(
            fin_expanded.contains(&"transaction".to_string()),
            "transaction should be sub-property of financial"
        );
        assert!(
            fin_expanded.contains(&"expense".to_string()),
            "expense should be sub-property of financial"
        );
    }
}
