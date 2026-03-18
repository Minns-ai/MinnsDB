//! Ontology Evolution Engine — self-expanding knowledge schema.
//!
//! Discovers new ontology properties from data patterns, infers their OWL
//! characteristics (symmetric, functional, etc.), groups related predicates
//! into hierarchies via LLM, and hot-reloads the ontology without restart.
//!
//! # Architecture
//!
//! 1. **Observation**: Every edge creation records `(category, predicate, domain, range)`.
//! 2. **Inference**: Periodically scans observations + graph edges to infer behaviors.
//! 3. **Discovery**: LLM clusters related predicates into parent-child hierarchies.
//! 4. **Proposal**: Changes are staged as proposals for review/approval.
//! 5. **Application**: Approved proposals generate TTL files and hot-reload the registry.

use crate::ontology::{OntologyRegistry, PropertyCharacteristic, PropertyDescriptor};
use crate::structures::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the ontology evolution engine.
#[derive(Debug, Clone)]
pub struct OntologyEvolutionConfig {
    /// Minimum edge count before a predicate is eligible for proposal.
    pub min_observations: u64,
    /// If single-value ratio exceeds this, infer FunctionalProperty.
    pub functional_ratio_threshold: f64,
    /// If symmetric co-occurrence ratio exceeds this, infer SymmetricProperty.
    pub symmetric_ratio_threshold: f64,
    /// If supersession ratio exceeds this, infer supersession cascade.
    pub supersession_ratio_threshold: f64,
    /// Directory to write generated TTL files.
    pub generated_ttl_dir: PathBuf,
    /// Whether to enable evolution tracking. Default: true.
    pub enabled: bool,
    /// Auto-apply proposals with confidence above this threshold.
    /// Set to `None` to require manual approval (API-only).
    /// Default: Some(0.85) — high-confidence proposals are applied automatically.
    pub auto_apply_threshold: Option<f64>,
}

impl Default for OntologyEvolutionConfig {
    fn default() -> Self {
        Self {
            min_observations: 5,
            functional_ratio_threshold: 0.90,
            symmetric_ratio_threshold: 0.85,
            supersession_ratio_threshold: 0.70,
            generated_ttl_dir: PathBuf::from("data/ontology"),
            enabled: true,
            auto_apply_threshold: Some(0.85),
        }
    }
}

// ---------------------------------------------------------------------------
// Observation tracking
// ---------------------------------------------------------------------------

/// Accumulated statistics for a single predicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateObservation {
    /// Raw predicate string (e.g., "plays_tennis").
    pub predicate: String,
    /// Category it appeared under (e.g., "hobby").
    pub category: String,
    /// Total edge creations observed.
    pub edge_count: u64,
    /// Distinct subject entity names seen.
    pub subjects: HashMap<String, u64>,
    /// Distinct object entity names seen.
    pub objects: HashMap<String, u64>,
    /// Domain ConceptType names observed (counts).
    pub domain_types: HashMap<String, u64>,
    /// Range ConceptType names observed (counts).
    pub range_types: HashMap<String, u64>,
    /// First seen timestamp (epoch nanos).
    pub first_seen: u64,
    /// Last seen timestamp (epoch nanos).
    pub last_seen: u64,
}

impl PredicateObservation {
    fn new(category: &str, predicate: &str, timestamp: u64) -> Self {
        Self {
            predicate: predicate.to_string(),
            category: category.to_string(),
            edge_count: 0,
            subjects: HashMap::new(),
            objects: HashMap::new(),
            domain_types: HashMap::new(),
            range_types: HashMap::new(),
            first_seen: timestamp,
            last_seen: timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Inferred behavior
// ---------------------------------------------------------------------------

/// Inferred OWL characteristics for a predicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredBehavior {
    pub predicate: String,
    pub category: String,
    pub is_symmetric: bool,
    pub is_functional: bool,
    pub is_append_only: bool,
    pub has_supersession: bool,
    /// Overall confidence in the inference (0.0..1.0).
    pub confidence: f64,
    /// Number of edges analyzed.
    pub sample_size: u64,
}

// ---------------------------------------------------------------------------
// Proposals
// ---------------------------------------------------------------------------

/// A proposed ontology expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyProposal {
    pub id: u64,
    pub status: ProposalStatus,
    pub created_at: u64,
    /// Properties being proposed.
    pub properties: Vec<ProposedProperty>,
    /// If creating a hierarchy, the parent property.
    pub parent_property: Option<String>,
    /// LLM-generated rationale.
    pub rationale: String,
    /// Generated TTL content (preview).
    pub ttl_preview: String,
}

/// Status of a proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    Applied,
}

/// A single property within a proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedProperty {
    pub id: String,
    pub label: String,
    pub comment: String,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub characteristics: Vec<PropertyCharacteristic>,
    pub max_cardinality: Option<u32>,
    pub sub_property_of: Option<String>,
    pub append_only: bool,
    pub canonical_predicate: String,
    /// Properties whose edges should be superseded when this property changes.
    pub cascade_dependents: Vec<String>,
    /// Whether this property's edges are affected by cascades from other properties.
    pub cascade_dependent: bool,
}

// ---------------------------------------------------------------------------
// Evolution Engine
// ---------------------------------------------------------------------------

/// Tracks unknown predicates, infers behaviors, and manages proposals.
pub struct OntologyEvolutionEngine {
    config: OntologyEvolutionConfig,
    /// Observations keyed by "category:predicate".
    observations: HashMap<String, PredicateObservation>,
    /// Active proposals.
    proposals: Vec<OntologyProposal>,
    /// Monotonic proposal ID counter.
    next_proposal_id: u64,
}

impl OntologyEvolutionEngine {
    pub fn new(config: OntologyEvolutionConfig) -> Self {
        Self {
            config,
            observations: HashMap::new(),
            proposals: Vec::new(),
            next_proposal_id: 1,
        }
    }

    // ── Observation Recording ──

    /// Record an edge creation. Called from the pipeline after edge insertion.
    ///
    /// Uses simple key tracking — no lock contention on the graph.
    #[allow(clippy::too_many_arguments)]
    pub fn record_observation(
        &mut self,
        category: &str,
        predicate: &str,
        subject_name: &str,
        object_name: &str,
        subject_type: &str,
        object_type: &str,
        timestamp: u64,
    ) {
        let key = format!("{}:{}", category, predicate);
        let obs = self
            .observations
            .entry(key)
            .or_insert_with(|| PredicateObservation::new(category, predicate, timestamp));

        obs.edge_count += 1;
        obs.last_seen = timestamp;
        *obs.subjects.entry(subject_name.to_string()).or_insert(0) += 1;
        *obs.objects.entry(object_name.to_string()).or_insert(0) += 1;
        if !subject_type.is_empty() {
            *obs.domain_types
                .entry(subject_type.to_string())
                .or_insert(0) += 1;
        }
        if !object_type.is_empty() {
            *obs.range_types.entry(object_type.to_string()).or_insert(0) += 1;
        }
    }

    /// Return all observations (for API exposure).
    pub fn observations(&self) -> &HashMap<String, PredicateObservation> {
        &self.observations
    }

    /// Return observations for predicates not yet in the ontology.
    pub fn unknown_observations(&self, ontology: &OntologyRegistry) -> Vec<&PredicateObservation> {
        self.observations
            .values()
            .filter(|obs| {
                obs.edge_count >= self.config.min_observations
                    && ontology.resolve(&obs.predicate).is_none()
            })
            .collect()
    }

    // ── Behavior Inference ──

    /// Analyze graph edges to infer OWL characteristics for tracked predicates.
    ///
    /// Scans the graph for each observed predicate and checks:
    /// - Symmetry: for each A→B edge, does B→A exist?
    /// - Functionality: does each subject have at most one value?
    /// - Supersession: what fraction of edges have `valid_until` set?
    /// - Append-only: are edges never superseded?
    pub fn infer_behaviors(
        &self,
        graph: &Graph,
        ontology: &OntologyRegistry,
    ) -> Vec<InferredBehavior> {
        let mut results = Vec::new();

        for obs in self.observations.values() {
            if obs.edge_count < self.config.min_observations {
                continue;
            }
            // Skip predicates already fully defined in the ontology
            if ontology.resolve(&obs.predicate).is_some() {
                continue;
            }

            let assoc_type = format!("{}:{}", obs.category, obs.predicate);
            let (forward, reverse, superseded, total, subjects_single, subjects_total) =
                self.scan_edges(graph, &assoc_type);

            if total == 0 {
                continue;
            }

            let is_symmetric = forward > 0
                && (reverse as f64 / forward as f64) >= self.config.symmetric_ratio_threshold;
            let is_functional = subjects_total >= 3
                && (subjects_single as f64 / subjects_total as f64)
                    >= self.config.functional_ratio_threshold;
            let has_supersession = total >= 3
                && (superseded as f64 / total as f64) >= self.config.supersession_ratio_threshold;
            let is_append_only = total >= self.config.min_observations && superseded == 0;

            // Confidence based on sample size
            let confidence = (total as f64 / (total as f64 + 10.0)).min(0.95);

            results.push(InferredBehavior {
                predicate: obs.predicate.clone(),
                category: obs.category.clone(),
                is_symmetric,
                is_functional,
                is_append_only,
                has_supersession,
                confidence,
                sample_size: total,
            });
        }

        results
    }

    /// Scan graph edges for a given association type, returning statistics.
    ///
    /// Returns `(forward_count, reverse_count, superseded, total, single_value_subjects, total_subjects)`.
    fn scan_edges(&self, graph: &Graph, assoc_type: &str) -> (u64, u64, u64, u64, u64, u64) {
        use crate::structures::EdgeType;

        let mut forward = 0u64;
        let mut reverse = 0u64;
        let mut superseded = 0u64;
        let mut total = 0u64;
        // subject_id → count of active values
        let mut subject_values: HashMap<u64, u64> = HashMap::new();

        for edge in graph.edges.values() {
            if let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            {
                if association_type != assoc_type {
                    continue;
                }
                total += 1;
                forward += 1;

                if edge.valid_until.is_some() {
                    superseded += 1;
                } else {
                    *subject_values.entry(edge.source).or_insert(0) += 1;
                }

                // Check if reverse edge exists
                let has_reverse = graph.get_edges_from(edge.target).iter().any(|rev| {
                    if let EdgeType::Association {
                        association_type: ref at,
                        ..
                    } = rev.edge_type
                    {
                        at == assoc_type && rev.target == edge.source
                    } else {
                        false
                    }
                });
                if has_reverse {
                    reverse += 1;
                }
            }
        }

        let subjects_total = subject_values.len() as u64;
        let subjects_single = subject_values.values().filter(|&&v| v == 1).count() as u64;

        (
            forward,
            reverse,
            superseded,
            total,
            subjects_single,
            subjects_total,
        )
    }

    // ── Proposal Creation ──

    /// Create a proposal from inferred behaviors.
    ///
    /// Groups related predicates by category, generates TTL, and stages a proposal.
    pub fn create_proposals_from_inferred(
        &mut self,
        behaviors: &[InferredBehavior],
        ontology: &OntologyRegistry,
    ) -> Vec<u64> {
        // Group by category
        let mut by_category: HashMap<String, Vec<&InferredBehavior>> = HashMap::new();
        for b in behaviors {
            by_category.entry(b.category.clone()).or_default().push(b);
        }

        let mut proposal_ids = Vec::new();

        for (category, predicates) in &by_category {
            let mut properties = Vec::new();

            for b in predicates {
                let mut chars = Vec::new();
                if b.is_symmetric {
                    chars.push(PropertyCharacteristic::Symmetric);
                }
                if b.is_functional {
                    chars.push(PropertyCharacteristic::Functional);
                }

                // Determine domain/range from observation data
                let obs_key = format!("{}:{}", b.category, b.predicate);
                let (domain, range) = if let Some(obs) = self.observations.get(&obs_key) {
                    let dom = most_common_type(&obs.domain_types);
                    let rng = most_common_type(&obs.range_types);
                    (
                        dom.map(|d| vec![d]).unwrap_or_default(),
                        rng.map(|r| vec![r]).unwrap_or_default(),
                    )
                } else {
                    (Vec::new(), Vec::new())
                };

                // Check if the category exists as a parent property
                let sub_property_of = if ontology.resolve(category).is_some() {
                    Some(category.clone())
                } else {
                    None
                };

                properties.push(ProposedProperty {
                    id: b.predicate.clone(),
                    label: predicate_to_label(&b.predicate),
                    comment: format!(
                        "Auto-discovered property ({} observations, confidence {:.0}%)",
                        b.sample_size,
                        b.confidence * 100.0
                    ),
                    domain,
                    range,
                    characteristics: chars,
                    max_cardinality: if b.is_functional { Some(1) } else { None },
                    sub_property_of,
                    append_only: b.is_append_only,
                    cascade_dependents: Vec::new(),
                    cascade_dependent: false,
                    canonical_predicate: b.predicate.clone(),
                });
            }

            if properties.is_empty() {
                continue;
            }

            let ttl_preview = generate_ttl_for_properties(&properties);
            let rationale = format!(
                "Discovered {} new properties under category '{}' from observed edge patterns.",
                properties.len(),
                category
            );

            let parent_in_ontology = ontology.resolve(category).is_some();

            let proposal = OntologyProposal {
                id: self.next_proposal_id,
                status: ProposalStatus::Pending,
                created_at: now_nanos(),
                properties,
                parent_property: if parent_in_ontology {
                    Some(category.clone())
                } else {
                    None
                },
                rationale,
                ttl_preview,
            };

            proposal_ids.push(self.next_proposal_id);
            self.proposals.push(proposal);
            self.next_proposal_id += 1;
        }

        proposal_ids
    }

    // ── LLM-Assisted Hierarchy Discovery ──

    /// Use the LLM to cluster unknown predicates and propose parent hierarchies.
    ///
    /// Returns the system prompt + user prompt for the LLM call. The caller
    /// is responsible for invoking the LLM and passing the response to
    /// `parse_hierarchy_response`.
    pub fn build_hierarchy_discovery_prompt(
        &self,
        ontology: &OntologyRegistry,
    ) -> Option<(String, String)> {
        let unknowns = self.unknown_observations(ontology);
        if unknowns.is_empty() {
            return None;
        }

        let existing_parents: Vec<String> = ontology.all_property_ids();

        let system = concat!(
            "You are an ontology engineer for a personal knowledge graph.\n",
            "Given observed predicates with their frequency and domain/range types, ",
            "group related predicates under existing or new parent properties.\n\n",
            "Output strict JSON:\n",
            "{\n",
            "  \"groups\": [\n",
            "    {\n",
            "      \"parent\": \"existing_or_new_parent_id\",\n",
            "      \"parent_label\": \"Human Readable Label\",\n",
            "      \"is_new_parent\": false,\n",
            "      \"children\": [\"predicate_a\", \"predicate_b\"],\n",
            "      \"rationale\": \"Why these belong together\"\n",
            "    }\n",
            "  ]\n",
            "}\n\n",
            "Rules:\n",
            "- Prefer grouping under existing parents when semantically appropriate\n",
            "- Only create new parents when no existing one fits\n",
            "- Use snake_case for IDs, Title Case for labels\n",
            "- No markdown fences, no explanation outside JSON",
        )
        .to_string();

        let mut user = String::from("Observed predicates:\n");
        for obs in &unknowns {
            let dom = most_common_type(&obs.domain_types).unwrap_or_else(|| "unknown".to_string());
            let rng = most_common_type(&obs.range_types).unwrap_or_else(|| "unknown".to_string());
            user.push_str(&format!(
                "- {} ({} edges, domain: {}, range: {})\n",
                obs.predicate, obs.edge_count, dom, rng
            ));
        }
        user.push_str(&format!(
            "\nExisting parent properties: {}\n",
            existing_parents.join(", ")
        ));

        Some((system, user))
    }

    /// Parse the LLM's hierarchy discovery response and create proposals.
    pub fn parse_hierarchy_response(
        &mut self,
        response: &str,
        ontology: &OntologyRegistry,
    ) -> Vec<u64> {
        let parsed: serde_json::Value = match crate::llm_client::parse_json_from_llm(response) {
            Some(v) => v,
            None => {
                tracing::warn!("Failed to parse hierarchy discovery LLM response");
                return Vec::new();
            },
        };

        let groups = match parsed["groups"].as_array() {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut proposal_ids = Vec::new();

        for group in groups {
            let parent = group["parent"].as_str().unwrap_or("").to_string();
            let parent_label = group["parent_label"]
                .as_str()
                .unwrap_or(&parent)
                .to_string();
            let is_new_parent = group["is_new_parent"].as_bool().unwrap_or(false);
            let rationale = group["rationale"]
                .as_str()
                .unwrap_or("LLM-suggested grouping")
                .to_string();

            let children: Vec<String> = group["children"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            if parent.is_empty() || children.is_empty() {
                continue;
            }

            let mut properties = Vec::new();

            // If parent is new, include it in the proposal
            if is_new_parent && ontology.resolve(&parent).is_none() {
                properties.push(ProposedProperty {
                    id: parent.clone(),
                    label: parent_label.clone(),
                    comment: format!("Auto-discovered parent property: {}", rationale),
                    domain: Vec::new(),
                    range: Vec::new(),
                    characteristics: Vec::new(),
                    max_cardinality: None,
                    sub_property_of: None,
                    append_only: false,
                    cascade_dependents: Vec::new(),
                    cascade_dependent: false,
                    canonical_predicate: String::new(),
                });
            }

            // Add children as sub-properties
            for child_id in &children {
                // Look up observation data for domain/range
                let obs_key_match = self
                    .observations
                    .values()
                    .find(|o| o.predicate == *child_id);
                let (domain, range) = if let Some(obs) = obs_key_match {
                    (
                        most_common_type(&obs.domain_types)
                            .map(|d| vec![d])
                            .unwrap_or_default(),
                        most_common_type(&obs.range_types)
                            .map(|r| vec![r])
                            .unwrap_or_default(),
                    )
                } else {
                    (Vec::new(), Vec::new())
                };

                properties.push(ProposedProperty {
                    id: child_id.clone(),
                    label: predicate_to_label(child_id),
                    comment: format!("Sub-property of {}: {}", parent, rationale),
                    domain,
                    range,
                    characteristics: Vec::new(), // Will be enriched by behavior inference
                    max_cardinality: None,
                    sub_property_of: Some(parent.clone()),
                    append_only: false,
                    cascade_dependents: Vec::new(),
                    cascade_dependent: false,
                    canonical_predicate: child_id.clone(),
                });
            }

            let ttl_preview = generate_ttl_for_properties(&properties);

            let proposal = OntologyProposal {
                id: self.next_proposal_id,
                status: ProposalStatus::Pending,
                created_at: now_nanos(),
                properties,
                parent_property: Some(parent),
                rationale,
                ttl_preview,
            };

            proposal_ids.push(self.next_proposal_id);
            self.proposals.push(proposal);
            self.next_proposal_id += 1;
        }

        proposal_ids
    }

    // ── LLM-Assisted Cascade & Temporal Dependency Inference ──

    /// Build a prompt for the LLM to infer cascade dependencies and temporal
    /// groupings across all registered ontology properties.
    ///
    /// This should be called after new properties are added (via hierarchy
    /// discovery or TTL upload) to automatically determine which properties
    /// are temporally dependent on others.
    ///
    /// Returns `(system_prompt, user_prompt)` for the LLM call. The caller
    /// passes the response to `parse_cascade_inference_response`.
    pub fn build_cascade_inference_prompt(ontology: &OntologyRegistry) -> (String, String) {
        let system = concat!(
            "You are an ontology engineer analyzing temporal dependencies in a personal knowledge graph.\n\n",
            "When a person's single-valued property changes (e.g., they move cities), some other properties\n",
            "become stale and should be superseded. For example:\n",
            "- When someone moves cities, their neighbor relationship is no longer valid\n",
            "- When someone changes jobs, their colleague relationships may change\n",
            "- When someone moves, their daily routine (gym, commute) likely changes\n\n",
            "Your job: given the full list of ontology properties, determine:\n",
            "1. Which single-valued (Functional) properties are \"triggers\" — when they change, other properties become stale\n",
            "2. Which properties are \"cascade dependents\" — their edges become stale when a trigger changes\n\n",
            "Output strict JSON:\n",
            "{\n",
            "  \"cascade_rules\": [\n",
            "    {\n",
            "      \"trigger\": \"property_id\",\n",
            "      \"dependents\": [\"dependent_property_id_1\", \"dependent_property_id_2\"],\n",
            "      \"rationale\": \"Why these depend on the trigger\"\n",
            "    }\n",
            "  ]\n",
            "}\n\n",
            "Rules:\n",
            "- Only single-valued (Functional) properties can be triggers\n",
            "- Append-only properties (like financial transactions) are NEVER dependents\n",
            "- Family relationships (parent, child, sibling, spouse) are NOT location-dependent\n",
            "- Only include dependencies where a real-world change in the trigger would genuinely\n",
            "  invalidate the dependent (e.g., moving cities invalidates neighbor, but NOT friend)\n",
            "- Be conservative — false positives cause data loss. Only flag clear temporal dependencies\n",
            "- Use only property IDs from the provided list\n",
            "- No markdown fences, no explanation outside JSON",
        )
        .to_string();

        let all_ids = ontology.all_property_ids();
        let mut user = String::from("Ontology properties:\n\n");
        for pid in &all_ids {
            if let Some(desc) = ontology.resolve(pid) {
                let chars: Vec<&str> = desc
                    .characteristics
                    .iter()
                    .map(|c| match c {
                        PropertyCharacteristic::Functional => "Functional",
                        PropertyCharacteristic::Symmetric => "Symmetric",
                        PropertyCharacteristic::Transitive => "Transitive",
                        PropertyCharacteristic::InverseFunctional => "InverseFunctional",
                    })
                    .collect();
                let parent = desc.sub_property_of.as_deref().unwrap_or("(root)");
                user.push_str(&format!(
                    "- {} (label: \"{}\", comment: \"{}\", parent: {}, characteristics: [{}], append_only: {})\n",
                    pid, desc.label, desc.comment, parent, chars.join(", "), desc.append_only
                ));
            }
        }

        (system, user)
    }

    /// Parse the LLM's cascade inference response and update the ontology registry
    /// with inferred cascade_dependents and cascade_dependent flags.
    ///
    /// Returns the number of properties updated.
    pub fn parse_cascade_inference_response(response: &str, ontology: &OntologyRegistry) -> usize {
        let parsed: serde_json::Value = match crate::llm_client::parse_json_from_llm(response) {
            Some(v) => v,
            None => {
                tracing::warn!("Failed to parse cascade inference LLM response");
                return 0;
            },
        };

        let rules = match parsed["cascade_rules"].as_array() {
            Some(r) => r,
            None => return 0,
        };

        let mut updated = 0usize;

        for rule in rules {
            let trigger = match rule["trigger"].as_str() {
                Some(t) => t,
                None => continue,
            };

            let dependents: Vec<String> = rule["dependents"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            if dependents.is_empty() {
                continue;
            }

            // Validate: trigger must exist and be functional
            let trigger_desc = match ontology.resolve(trigger) {
                Some(d) => d,
                None => {
                    tracing::warn!(
                        "Cascade inference: trigger '{}' not found in ontology",
                        trigger
                    );
                    continue;
                },
            };
            if !trigger_desc
                .characteristics
                .contains(&PropertyCharacteristic::Functional)
            {
                tracing::warn!(
                    "Cascade inference: trigger '{}' is not Functional, skipping",
                    trigger
                );
                continue;
            }

            // Filter dependents to only valid property IDs, exclude append-only
            let valid_dependents: Vec<String> = dependents
                .into_iter()
                .filter(|dep| {
                    if let Some(d) = ontology.resolve(dep) {
                        if d.append_only {
                            tracing::debug!(
                                "Cascade inference: skipping append-only dependent '{}'",
                                dep
                            );
                            return false;
                        }
                        true
                    } else {
                        tracing::warn!(
                            "Cascade inference: dependent '{}' not found in ontology",
                            dep
                        );
                        false
                    }
                })
                .collect();

            if valid_dependents.is_empty() {
                continue;
            }

            // Update the trigger property's cascade_dependents
            if ontology.update_cascade_dependents(trigger, &valid_dependents) {
                updated += 1;
                tracing::info!(
                    "Cascade inference: {} triggers cascade on [{}]",
                    trigger,
                    valid_dependents.join(", ")
                );
            }

            // Mark each dependent as cascade_dependent
            for dep in &valid_dependents {
                if ontology.mark_cascade_dependent(dep) {
                    updated += 1;
                }
            }
        }

        updated
    }

    // ── Proposal Management ──

    /// List all proposals.
    pub fn proposals(&self) -> &[OntologyProposal] {
        &self.proposals
    }

    /// Get a proposal by ID.
    pub fn get_proposal(&self, id: u64) -> Option<&OntologyProposal> {
        self.proposals.iter().find(|p| p.id == id)
    }

    /// Approve a proposal.
    pub fn approve_proposal(&mut self, id: u64) -> bool {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            if p.status == ProposalStatus::Pending {
                p.status = ProposalStatus::Approved;
                return true;
            }
        }
        false
    }

    /// Reject a proposal.
    pub fn reject_proposal(&mut self, id: u64) -> bool {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            if p.status == ProposalStatus::Pending {
                p.status = ProposalStatus::Rejected;
                return true;
            }
        }
        false
    }

    /// Apply an approved proposal: write TTL, register properties, update status.
    ///
    /// Returns the number of new properties registered.
    pub fn apply_proposal(
        &mut self,
        id: u64,
        ontology: &OntologyRegistry,
    ) -> Result<usize, OntologyEvolutionError> {
        let proposal = self
            .proposals
            .iter()
            .find(|p| p.id == id)
            .ok_or(OntologyEvolutionError::ProposalNotFound(id))?;

        if proposal.status != ProposalStatus::Approved {
            return Err(OntologyEvolutionError::InvalidStatus(format!(
                "Proposal {} is {:?}, not Approved",
                id, proposal.status
            )));
        }

        let ttl_content = &proposal.ttl_preview;

        // Write TTL file
        let ttl_path = self
            .config
            .generated_ttl_dir
            .join(format!("evolved_{:04}.ttl", id));
        std::fs::write(&ttl_path, ttl_content).map_err(OntologyEvolutionError::Io)?;
        tracing::info!("Wrote evolved ontology to {:?}", ttl_path);

        // Register properties directly into the ontology registry (hot-reload)
        let mut registered = 0;
        // We need to clone the properties to avoid borrow issues
        let properties = proposal.properties.clone();
        for prop in &properties {
            let descriptor = PropertyDescriptor {
                id: prop.id.clone(),
                label: prop.label.clone(),
                comment: prop.comment.clone(),
                domain: prop.domain.clone(),
                range: prop.range.clone(),
                characteristics: prop.characteristics.clone(),
                max_cardinality: prop.max_cardinality,
                sub_property_of: prop.sub_property_of.clone(),
                inverse_of: None,
                append_only: prop.append_only,
                cascade_dependents: prop.cascade_dependents.clone(),
                cascade_dependent: prop.cascade_dependent,
                skip_conflict_detection: prop.append_only,
                canonical_predicate: prop.canonical_predicate.clone(),
            };
            if ontology.register_property(descriptor) {
                registered += 1;
            }
        }

        // Update proposal status
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            p.status = ProposalStatus::Applied;
        }

        tracing::info!(
            "Applied proposal {}: registered {} new properties",
            id,
            registered
        );
        Ok(registered)
    }

    // ── Full Discovery Pass ──

    /// Run a complete discovery pass: infer behaviors → create proposals.
    ///
    /// Returns IDs of newly created proposals.
    pub fn run_inference_pass(&mut self, graph: &Graph, ontology: &OntologyRegistry) -> Vec<u64> {
        let behaviors = self.infer_behaviors(graph, ontology);
        if behaviors.is_empty() {
            return Vec::new();
        }
        tracing::info!(
            "Ontology evolution: inferred behaviors for {} predicates",
            behaviors.len()
        );
        self.create_proposals_from_inferred(&behaviors, ontology)
    }

    /// Run a complete discovery pass with automatic application of
    /// high-confidence proposals (if `auto_apply_threshold` is set).
    ///
    /// Returns (proposal_ids_created, proposal_ids_auto_applied).
    pub fn run_inference_pass_auto(
        &mut self,
        graph: &Graph,
        ontology: &OntologyRegistry,
    ) -> (Vec<u64>, Vec<u64>) {
        let behaviors = self.infer_behaviors(graph, ontology);
        if behaviors.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Build a confidence map: predicate → confidence
        let confidence_map: HashMap<String, f64> = behaviors
            .iter()
            .map(|b| (b.predicate.clone(), b.confidence))
            .collect();

        tracing::info!(
            "Ontology evolution: inferred behaviors for {} predicates",
            behaviors.len()
        );
        let proposal_ids = self.create_proposals_from_inferred(&behaviors, ontology);

        let threshold = match self.config.auto_apply_threshold {
            Some(t) => t,
            None => return (proposal_ids, Vec::new()),
        };

        // Auto-apply proposals where ALL properties exceed the threshold
        let mut auto_applied = Vec::new();
        for &pid in &proposal_ids {
            let all_above = self
                .proposals
                .iter()
                .find(|p| p.id == pid)
                .map(|p| {
                    p.properties.iter().all(|prop| {
                        confidence_map.get(&prop.id).copied().unwrap_or(0.0) >= threshold
                    })
                })
                .unwrap_or(false);

            if all_above {
                self.approve_proposal(pid);
                match self.apply_proposal(pid, ontology) {
                    Ok(n) => {
                        tracing::info!(
                            "Ontology evolution: auto-applied proposal {} ({} properties)",
                            pid,
                            n
                        );
                        auto_applied.push(pid);
                    },
                    Err(e) => {
                        tracing::warn!(
                            "Ontology evolution: auto-apply failed for proposal {}: {}",
                            pid,
                            e
                        );
                    },
                }
            }
        }

        (proposal_ids, auto_applied)
    }

    /// Return summary statistics for the evolution engine.
    pub fn stats(&self) -> EvolutionStats {
        EvolutionStats {
            total_observations: self.observations.len(),
            total_edge_count: self.observations.values().map(|o| o.edge_count).sum(),
            pending_proposals: self
                .proposals
                .iter()
                .filter(|p| p.status == ProposalStatus::Pending)
                .count(),
            applied_proposals: self
                .proposals
                .iter()
                .filter(|p| p.status == ProposalStatus::Applied)
                .count(),
        }
    }
}

/// Summary statistics for the evolution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStats {
    pub total_observations: usize,
    pub total_edge_count: u64,
    pub pending_proposals: usize,
    pub applied_proposals: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum OntologyEvolutionError {
    ProposalNotFound(u64),
    InvalidStatus(String),
    Io(std::io::Error),
}

impl std::fmt::Display for OntologyEvolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProposalNotFound(id) => write!(f, "Proposal {} not found", id),
            Self::InvalidStatus(msg) => write!(f, "Invalid proposal status: {}", msg),
            Self::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for OntologyEvolutionError {}

// ---------------------------------------------------------------------------
// TTL generation
// ---------------------------------------------------------------------------

/// Generate Turtle content for a set of proposed properties.
fn generate_ttl_for_properties(properties: &[ProposedProperty]) -> String {
    let mut ttl = String::new();
    ttl.push_str("## ── Auto-evolved ontology ──\n");
    ttl.push_str("## Generated by OntologyEvolutionEngine\n\n");
    ttl.push_str("@prefix eg:   <http://minnsdb.local/ontology#> .\n");
    ttl.push_str("@prefix owl:  <http://www.w3.org/2002/07/owl#> .\n");
    ttl.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n");

    for prop in properties {
        ttl.push_str(&format!("eg:{} a owl:ObjectProperty ;\n", prop.id));
        ttl.push_str(&format!("    rdfs:label \"{}\" ;\n", prop.label));
        if !prop.comment.is_empty() {
            ttl.push_str(&format!("    rdfs:comment \"{}\" ;\n", prop.comment));
        }
        if !prop.domain.is_empty() {
            let domains: Vec<String> = prop.domain.iter().map(|d| format!("eg:{}", d)).collect();
            ttl.push_str(&format!("    rdfs:domain {} ;\n", domains.join(", ")));
        }
        if !prop.range.is_empty() {
            let ranges: Vec<String> = prop.range.iter().map(|r| format!("eg:{}", r)).collect();
            ttl.push_str(&format!("    rdfs:range {} ;\n", ranges.join(", ")));
        }
        if let Some(ref parent) = prop.sub_property_of {
            ttl.push_str(&format!("    rdfs:subPropertyOf eg:{} ;\n", parent));
        }
        for ch in &prop.characteristics {
            match ch {
                PropertyCharacteristic::Symmetric => {
                    ttl.push_str("    a owl:SymmetricProperty ;\n");
                },
                PropertyCharacteristic::Functional => {
                    ttl.push_str("    a owl:FunctionalProperty ;\n");
                },
                PropertyCharacteristic::Transitive => {
                    ttl.push_str("    a owl:TransitiveProperty ;\n");
                },
                PropertyCharacteristic::InverseFunctional => {
                    ttl.push_str("    a owl:InverseFunctionalProperty ;\n");
                },
            }
        }
        if prop.append_only {
            ttl.push_str("    eg:appendOnly true ;\n");
        }
        if !prop.cascade_dependents.is_empty() {
            let deps: Vec<String> = prop
                .cascade_dependents
                .iter()
                .map(|d| format!("eg:{}", d))
                .collect();
            ttl.push_str(&format!("    eg:cascadeDependents {} ;\n", deps.join(", ")));
        }
        if prop.cascade_dependent {
            ttl.push_str("    eg:cascadeDependent \"true\" ;\n");
        }
        if !prop.canonical_predicate.is_empty() {
            ttl.push_str(&format!(
                "    eg:canonicalPredicate \"{}\" ;\n",
                prop.canonical_predicate
            ));
        }
        // Replace trailing " ;\n" with " .\n"
        if ttl.ends_with(" ;\n") {
            ttl.truncate(ttl.len() - 3);
            ttl.push_str(" .\n");
        }
        ttl.push('\n');
    }

    ttl
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert "plays_tennis" → "Plays Tennis".
fn predicate_to_label(predicate: &str) -> String {
    predicate
        .split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Return the most common type from a frequency map.
fn most_common_type(types: &HashMap<String, u64>) -> Option<String> {
    types
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(k, _)| k.clone())
}

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_observation() {
        let config = OntologyEvolutionConfig::default();
        let mut engine = OntologyEvolutionEngine::new(config);

        engine.record_observation(
            "hobby",
            "plays_tennis",
            "Alice",
            "Tennis",
            "Person",
            "NamedEntity",
            1000,
        );
        engine.record_observation(
            "hobby",
            "plays_tennis",
            "Bob",
            "Tennis",
            "Person",
            "NamedEntity",
            2000,
        );
        engine.record_observation(
            "hobby",
            "plays_tennis",
            "Alice",
            "Tennis",
            "Person",
            "NamedEntity",
            3000,
        );

        let obs = engine.observations.get("hobby:plays_tennis").unwrap();
        assert_eq!(obs.edge_count, 3);
        assert_eq!(obs.subjects.len(), 2); // Alice, Bob
        assert_eq!(obs.objects.len(), 1); // Tennis
        assert_eq!(obs.first_seen, 1000);
        assert_eq!(obs.last_seen, 3000);
    }

    #[test]
    fn test_predicate_to_label() {
        assert_eq!(predicate_to_label("plays_tennis"), "Plays Tennis");
        assert_eq!(predicate_to_label("friend"), "Friend");
        assert_eq!(predicate_to_label("lives_in"), "Lives In");
    }

    #[test]
    fn test_generate_ttl() {
        let props = vec![ProposedProperty {
            id: "plays_tennis".to_string(),
            label: "Plays Tennis".to_string(),
            comment: "Tennis player".to_string(),
            domain: vec!["Person".to_string()],
            range: vec!["NamedEntity".to_string()],
            characteristics: vec![PropertyCharacteristic::Symmetric],
            max_cardinality: None,
            sub_property_of: Some("hobby".to_string()),
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            canonical_predicate: "plays_tennis".to_string(),
        }];

        let ttl = generate_ttl_for_properties(&props);
        assert!(ttl.contains("eg:plays_tennis"));
        assert!(ttl.contains("rdfs:subPropertyOf eg:hobby"));
        assert!(ttl.contains("owl:SymmetricProperty"));
        assert!(ttl.contains("rdfs:domain eg:Person"));
    }

    #[test]
    fn test_proposal_lifecycle() {
        let config = OntologyEvolutionConfig::default();
        let mut engine = OntologyEvolutionEngine::new(config);

        // Create a manual proposal
        let behaviors = vec![InferredBehavior {
            predicate: "plays_chess".to_string(),
            category: "hobby".to_string(),
            is_symmetric: false,
            is_functional: false,
            is_append_only: true,
            has_supersession: false,
            confidence: 0.8,
            sample_size: 10,
        }];

        let ontology = OntologyRegistry::new();
        let ids = engine.create_proposals_from_inferred(&behaviors, &ontology);
        assert_eq!(ids.len(), 1);

        let proposal = engine.get_proposal(ids[0]).unwrap();
        assert_eq!(proposal.status, ProposalStatus::Pending);

        // Approve
        assert!(engine.approve_proposal(ids[0]));
        let proposal = engine.get_proposal(ids[0]).unwrap();
        assert_eq!(proposal.status, ProposalStatus::Approved);

        // Can't approve again
        assert!(!engine.approve_proposal(ids[0]));
    }

    #[test]
    fn test_unknown_observations() {
        let config = OntologyEvolutionConfig {
            min_observations: 2,
            ..OntologyEvolutionConfig::default()
        };
        let mut engine = OntologyEvolutionEngine::new(config);

        // Add observations
        for i in 0..5 {
            engine.record_observation(
                "hobby",
                "plays_tennis",
                "Alice",
                "Tennis",
                "Person",
                "NamedEntity",
                i,
            );
        }
        engine.record_observation(
            "hobby",
            "plays_chess",
            "Bob",
            "Chess",
            "Person",
            "NamedEntity",
            0,
        );

        let ontology = OntologyRegistry::new();
        let unknowns = engine.unknown_observations(&ontology);
        // plays_tennis has 5 obs (>= 2), plays_chess has 1 (< 2)
        assert_eq!(unknowns.len(), 1);
        assert_eq!(unknowns[0].predicate, "plays_tennis");
    }

    #[test]
    fn test_evolution_stats() {
        let config = OntologyEvolutionConfig::default();
        let mut engine = OntologyEvolutionEngine::new(config);
        engine.record_observation(
            "hobby",
            "plays_tennis",
            "Alice",
            "Tennis",
            "Person",
            "NamedEntity",
            0,
        );

        let stats = engine.stats();
        assert_eq!(stats.total_observations, 1);
        assert_eq!(stats.total_edge_count, 1);
        assert_eq!(stats.pending_proposals, 0);
    }

    #[test]
    fn test_cascade_inference_prompt_includes_all_properties() {
        let ontology = crate::ontology::tests::test_registry();
        let (system, user) = OntologyEvolutionEngine::build_cascade_inference_prompt(&ontology);

        assert!(system.contains("cascade"));
        assert!(system.contains("temporal dependencies"));
        // Should include all registered properties
        assert!(user.contains("location"));
        assert!(user.contains("relationship"));
        assert!(user.contains("financial"));
        assert!(user.contains("routine"));
        // Should show characteristics
        assert!(user.contains("Functional"));
        assert!(user.contains("Symmetric"));
    }

    #[test]
    fn test_parse_cascade_inference_response() {
        let ontology = crate::ontology::tests::test_registry();

        // Clear existing cascade_dependents on location to test inference
        // (test_registry already sets them, so we test with a fresh ontology)
        let fresh_ontology = crate::ontology::OntologyRegistry::new();
        fresh_ontology.register_property(crate::ontology::PropertyDescriptor {
            id: "location".into(),
            label: "Location".into(),
            comment: "where someone lives".into(),
            domain: vec!["Person".into()],
            range: vec!["Location".into()],
            characteristics: vec![PropertyCharacteristic::Functional],
            max_cardinality: Some(1),
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: "lives_in".into(),
        });
        fresh_ontology.register_property(crate::ontology::PropertyDescriptor {
            id: "routine".into(),
            label: "Routine".into(),
            comment: "daily habits".into(),
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
        });
        fresh_ontology.register_property(crate::ontology::PropertyDescriptor {
            id: "financial".into(),
            label: "Financial".into(),
            comment: "payments".into(),
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
            canonical_predicate: String::new(),
        });

        let llm_response = r#"{
            "cascade_rules": [
                {
                    "trigger": "location",
                    "dependents": ["routine", "financial"],
                    "rationale": "Moving invalidates daily routines"
                }
            ]
        }"#;

        let updated = OntologyEvolutionEngine::parse_cascade_inference_response(
            llm_response,
            &fresh_ontology,
        );

        // Should have updated location's cascade_dependents (added routine)
        // and marked routine as cascade_dependent.
        // financial should be excluded (append-only).
        assert!(updated > 0);
        let loc = fresh_ontology.resolve("location").unwrap();
        assert_eq!(loc.cascade_dependents, vec!["routine"]);
        assert!(fresh_ontology.is_cascade_dependent("routine"));
        // financial is append-only, should not be marked
        assert!(!fresh_ontology.is_cascade_dependent("financial"));
    }

    #[test]
    fn test_parse_cascade_inference_rejects_non_functional_trigger() {
        let ontology = crate::ontology::OntologyRegistry::new();
        ontology.register_property(crate::ontology::PropertyDescriptor {
            id: "relationship".into(),
            label: "Relationship".into(),
            comment: "connections".into(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: vec![PropertyCharacteristic::Symmetric], // NOT Functional
            max_cardinality: None,
            sub_property_of: None,
            inverse_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
            skip_conflict_detection: false,
            canonical_predicate: String::new(),
        });
        ontology.register_property(crate::ontology::PropertyDescriptor {
            id: "routine".into(),
            label: "Routine".into(),
            comment: "habits".into(),
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
        });

        let llm_response = r#"{
            "cascade_rules": [
                {
                    "trigger": "relationship",
                    "dependents": ["routine"],
                    "rationale": "Bad inference"
                }
            ]
        }"#;

        let updated =
            OntologyEvolutionEngine::parse_cascade_inference_response(llm_response, &ontology);

        // Should reject: relationship is not Functional
        assert_eq!(updated, 0);
    }

    #[test]
    fn test_proposed_property_cascade_fields_in_ttl() {
        let props = vec![ProposedProperty {
            id: "location".to_string(),
            label: "Location".to_string(),
            comment: "where someone lives".to_string(),
            domain: vec!["Person".to_string()],
            range: vec!["Location".to_string()],
            characteristics: vec![PropertyCharacteristic::Functional],
            max_cardinality: Some(1),
            sub_property_of: None,
            append_only: false,
            cascade_dependents: vec!["routine".to_string(), "relationship".to_string()],
            cascade_dependent: false,
            canonical_predicate: "lives_in".to_string(),
        }];

        let ttl = generate_ttl_for_properties(&props);
        assert!(ttl.contains("eg:cascadeDependents eg:routine, eg:relationship"));
        assert!(!ttl.contains("eg:cascadeDependent \"true\""));

        // Test cascade_dependent property
        let props2 = vec![ProposedProperty {
            id: "routine".to_string(),
            label: "Routine".to_string(),
            comment: "daily habits".to_string(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: Vec::new(),
            max_cardinality: None,
            sub_property_of: None,
            append_only: false,
            cascade_dependents: Vec::new(),
            cascade_dependent: true,
            canonical_predicate: String::new(),
        }];

        let ttl2 = generate_ttl_for_properties(&props2);
        assert!(ttl2.contains("eg:cascadeDependent \"true\""));
    }
}
