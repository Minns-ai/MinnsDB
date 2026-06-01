//! Ontology evolution engine — graph-scan-based predicate discovery.
//!
//! Discovers new predicates from the live graph, infers OWL characteristics
//! (Functional, Symmetric, append-only, supersession-cascade), groups related
//! predicates into hierarchies via LLM, and stages proposals for human
//! approval before any schema change is applied.
//!
//! # Design
//!
//! - **No per-edge observation state.** Stats are derived from a single
//!   graph scan at `/discover` time. The graph is the source of truth.
//!   Memory is bounded by the number of distinct predicates seen, not by
//!   total edge count or by distinct entity names.
//! - **No auto-apply.** Every proposal requires explicit approval. There
//!   is no confidence threshold that triggers automatic schema change.
//! - **Single-pass O(edges) scan** for snapshot stats; the previous design
//!   ran a full edge sweep per observed predicate (O(predicates * edges)).
//!
//! Proposals live in-memory on the engine instance. Persistence to redb is
//! a planned follow-up and is the only known restart-safety gap.

use crate::ontology::{OntologyRegistry, PropertyCharacteristic, PropertyDescriptor};
use crate::structures::{EdgeType, Graph, NodeType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the ontology evolution engine.
///
/// All thresholds gate which predicates produce a *proposal* — they never
/// trigger automatic application. A human always approves before the
/// ontology changes.
#[derive(Debug, Clone)]
pub struct OntologyEvolutionConfig {
    /// Minimum edge count before a predicate is eligible for proposal.
    pub min_observations: u64,
    /// Single-value ratio above which a predicate is flagged Functional.
    pub functional_ratio_threshold: f64,
    /// Symmetric pair ratio above which a predicate is flagged Symmetric.
    pub symmetric_ratio_threshold: f64,
    /// Supersession ratio above which a predicate is flagged as having
    /// cascade-supersession semantics.
    pub supersession_ratio_threshold: f64,
    /// Directory to write generated TTL files for applied proposals.
    pub generated_ttl_dir: PathBuf,
}

impl Default for OntologyEvolutionConfig {
    fn default() -> Self {
        Self {
            min_observations: 5,
            functional_ratio_threshold: 0.90,
            symmetric_ratio_threshold: 0.85,
            supersession_ratio_threshold: 0.70,
            generated_ttl_dir: PathBuf::from("data/ontology"),
        }
    }
}

// ---------------------------------------------------------------------------
// PredicateStats — produced by a single graph scan, bounded memory
// ---------------------------------------------------------------------------

/// Per-predicate statistics derived from a single graph scan.
///
/// All counts are `u64`. The two type lists are capped at the top 5 entries
/// each so memory is bounded by `distinct_predicates * (constant)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateStats {
    pub category: String,
    pub predicate: String,
    /// Total edges seen (active + superseded).
    pub edge_count: u64,
    /// Active edges (valid_until is None).
    pub forward_active: u64,
    /// Active edges where the reverse direction also exists. Excludes
    /// self-loops. Used to detect symmetric properties.
    pub reverse_active: u64,
    /// Edges marked as superseded (valid_until is Some). Disjoint from
    /// `forward_active` so `edge_count == forward_active + superseded`.
    pub superseded: u64,
    /// Distinct subject NodeIds with at least one active edge.
    pub distinct_subjects: u64,
    /// Subjects with exactly one active edge — basis for the functional /
    /// single-valued ratio.
    pub single_value_subjects: u64,
    /// Top 5 source node-type labels by count.
    pub domain_types: Vec<(String, u64)>,
    /// Top 5 target node-type labels by count.
    pub range_types: Vec<(String, u64)>,
    /// Earliest valid_from across all edges (for diagnostics).
    pub first_valid_from: Option<u64>,
    /// Latest valid_from across all edges (for diagnostics).
    pub last_valid_from: Option<u64>,
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
    /// Overall confidence in the inference (0.0..1.0). Surfaced to the
    /// human reviewer; never used for any automated decision.
    pub confidence: f64,
    /// Number of edges analysed.
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

/// Manages proposals and runs discovery passes.
///
/// Holds no per-edge state — stats are derived from the graph on demand.
pub struct OntologyEvolutionEngine {
    config: OntologyEvolutionConfig,
    /// Active proposals. Bounded by human approval rate; in-memory until
    /// persistence to redb lands as a follow-up.
    proposals: Vec<OntologyProposal>,
    /// Monotonic proposal ID counter.
    next_proposal_id: u64,
}

impl OntologyEvolutionEngine {
    pub fn new(config: OntologyEvolutionConfig) -> Self {
        Self {
            config,
            proposals: Vec::new(),
            next_proposal_id: 1,
        }
    }

    // ── Graph scan ────────────────────────────────────────────────────────

    /// Walk every Association edge in the graph exactly once and produce a
    /// `PredicateStats` per distinct `(category, predicate)` pair.
    ///
    /// O(edges) time. Memory bounded by `distinct_predicates * (5 type
    /// entries + per-subject HashSet + per-pair HashSet)`. Transient
    /// HashSets are dropped at the end of the function.
    pub fn snapshot_predicate_stats(graph: &Graph) -> Vec<PredicateStats> {
        struct Raw {
            edge_count: u64,
            forward_active: u64,
            superseded: u64,
            /// subject_id → count of active edges from that subject
            subject_active_counts: HashMap<u64, u64>,
            /// type label → count (bounded by ~11 NodeType variants + ConceptType subvariants)
            domain_type_counts: HashMap<String, u64>,
            range_type_counts: HashMap<String, u64>,
            first_valid_from: Option<u64>,
            last_valid_from: Option<u64>,
            /// Active (source, target) pairs, used for symmetry detection.
            /// One entry per active edge, dropped at end of scan.
            active_pairs: HashSet<(u64, u64)>,
        }
        impl Raw {
            fn new() -> Self {
                Self {
                    edge_count: 0,
                    forward_active: 0,
                    superseded: 0,
                    subject_active_counts: HashMap::new(),
                    domain_type_counts: HashMap::new(),
                    range_type_counts: HashMap::new(),
                    first_valid_from: None,
                    last_valid_from: None,
                    active_pairs: HashSet::new(),
                }
            }
        }

        let mut by_assoc: HashMap<String, Raw> = HashMap::new();

        // Single pass.
        for edge in graph.edges.values() {
            let EdgeType::Association {
                association_type, ..
            } = &edge.edge_type
            else {
                continue;
            };
            let raw = by_assoc
                .entry(association_type.clone())
                .or_insert_with(Raw::new);

            raw.edge_count += 1;
            if edge.valid_until.is_some() {
                raw.superseded += 1;
            } else {
                raw.forward_active += 1;
                *raw.subject_active_counts.entry(edge.source).or_insert(0) += 1;
                // Skip self-loops in the symmetry set so they don't
                // self-match below.
                if edge.source != edge.target {
                    raw.active_pairs.insert((edge.source, edge.target));
                }
            }

            if let Some(vf) = edge.valid_from {
                raw.first_valid_from = Some(raw.first_valid_from.map_or(vf, |x| x.min(vf)));
                raw.last_valid_from = Some(raw.last_valid_from.map_or(vf, |x| x.max(vf)));
            }

            if let Some(src_node) = graph.get_node(edge.source) {
                let label = node_type_label(&src_node.node_type);
                *raw.domain_type_counts.entry(label).or_insert(0) += 1;
            }
            if let Some(tgt_node) = graph.get_node(edge.target) {
                let label = node_type_label(&tgt_node.node_type);
                *raw.range_type_counts.entry(label).or_insert(0) += 1;
            }
        }

        // Finalise each bucket. The transient HashMaps/HashSets are
        // consumed here and freed when `by_assoc` is dropped.
        let mut out = Vec::with_capacity(by_assoc.len());
        for (assoc_type, raw) in by_assoc {
            // Symmetry: count active edges whose reverse is also present.
            // Each symmetric pair (A→B, B→A) contributes 2 to reverse_active
            // (both directions match), giving forward/reverse parity for
            // perfectly symmetric data — same semantics as the prior
            // per-predicate scan.
            let reverse_active = raw
                .active_pairs
                .iter()
                .filter(|(s, t)| raw.active_pairs.contains(&(*t, *s)))
                .count() as u64;

            let distinct_subjects = raw.subject_active_counts.len() as u64;
            let single_value_subjects = raw
                .subject_active_counts
                .values()
                .filter(|&&c| c == 1)
                .count() as u64;

            let domain_types = top_n(raw.domain_type_counts, 5);
            let range_types = top_n(raw.range_type_counts, 5);

            // `association_type` format is `{category}:{predicate}`. Split at
            // the first colon; predicates that happen to contain ':' keep
            // the remainder.
            let (category, predicate) = assoc_type
                .split_once(':')
                .map(|(c, p)| (c.to_string(), p.to_string()))
                .unwrap_or_else(|| (assoc_type.clone(), String::new()));

            out.push(PredicateStats {
                category,
                predicate,
                edge_count: raw.edge_count,
                forward_active: raw.forward_active,
                reverse_active,
                superseded: raw.superseded,
                distinct_subjects,
                single_value_subjects,
                domain_types,
                range_types,
                first_valid_from: raw.first_valid_from,
                last_valid_from: raw.last_valid_from,
            });
        }

        out
    }

    // ── Behaviour Inference ──────────────────────────────────────────────

    /// Infer OWL characteristics for each predicate not already in the
    /// ontology, gated by `min_observations`. Pure function over `&[PredicateStats]`.
    pub fn infer_behaviours(
        &self,
        stats: &[PredicateStats],
        ontology: &OntologyRegistry,
    ) -> Vec<InferredBehavior> {
        let mut results = Vec::new();
        for s in stats {
            if s.edge_count < self.config.min_observations {
                continue;
            }
            // Skip predicates already fully defined in the ontology.
            if ontology.resolve(&s.predicate).is_some() {
                continue;
            }
            let total = s.edge_count;
            let forward = s.forward_active;
            let is_symmetric = forward > 0
                && (s.reverse_active as f64 / forward as f64)
                    >= self.config.symmetric_ratio_threshold;
            let is_functional = s.distinct_subjects >= 3
                && (s.single_value_subjects as f64 / s.distinct_subjects as f64)
                    >= self.config.functional_ratio_threshold;
            let has_supersession = total >= 3
                && (s.superseded as f64 / total as f64) >= self.config.supersession_ratio_threshold;
            let is_append_only = total >= self.config.min_observations && s.superseded == 0;

            let confidence = (total as f64 / (total as f64 + 10.0)).min(0.95);

            results.push(InferredBehavior {
                predicate: s.predicate.clone(),
                category: s.category.clone(),
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

    // ── Proposal Creation ────────────────────────────────────────────────

    /// Turn inferred behaviours into proposals. Domain/range are looked up
    /// from `stats` (caller passes the same vector used by `infer_behaviours`).
    pub fn create_proposals_from_inferred(
        &mut self,
        behaviours: &[InferredBehavior],
        ontology: &OntologyRegistry,
        stats: &[PredicateStats],
    ) -> Vec<u64> {
        // Index stats by predicate for O(1) domain/range lookup.
        let stats_by_pred: HashMap<&str, &PredicateStats> =
            stats.iter().map(|s| (s.predicate.as_str(), s)).collect();

        // Group behaviours by category.
        let mut by_category: HashMap<String, Vec<&InferredBehavior>> = HashMap::new();
        for b in behaviours {
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

                let (domain, range) = match stats_by_pred.get(b.predicate.as_str()) {
                    Some(s) => (
                        s.domain_types
                            .first()
                            .map(|(t, _)| vec![t.clone()])
                            .unwrap_or_default(),
                        s.range_types
                            .first()
                            .map(|(t, _)| vec![t.clone()])
                            .unwrap_or_default(),
                    ),
                    None => (Vec::new(), Vec::new()),
                };

                let sub_property_of = if ontology.resolve(category).is_some() {
                    Some(category.clone())
                } else {
                    None
                };

                properties.push(ProposedProperty {
                    id: b.predicate.clone(),
                    label: predicate_to_label(&b.predicate),
                    comment: format!(
                        "Auto-discovered property ({} edges, confidence {:.0}%)",
                        b.sample_size,
                        b.confidence * 100.0
                    ),
                    domain,
                    range,
                    characteristics: chars,
                    max_cardinality: if b.is_functional { Some(1) } else { None },
                    sub_property_of,
                    append_only: b.is_append_only,
                    canonical_predicate: b.predicate.clone(),
                    cascade_dependents: Vec::new(),
                    cascade_dependent: false,
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

    // ── LLM-Assisted Hierarchy Discovery ─────────────────────────────────

    /// Build a hierarchy-discovery prompt from current predicate stats.
    /// Returns `None` if there are no unknown predicates above `min_observations`.
    ///
    /// Static method so callers can drive prompt + LLM + parse without
    /// holding a write lock on the engine during the (slow) LLM call.
    pub fn build_hierarchy_discovery_prompt(
        stats: &[PredicateStats],
        ontology: &OntologyRegistry,
        min_observations: u64,
    ) -> Option<(String, String)> {
        let unknowns: Vec<&PredicateStats> = stats
            .iter()
            .filter(|s| {
                s.edge_count >= min_observations && ontology.resolve(&s.predicate).is_none()
            })
            .collect();
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
        for s in &unknowns {
            let dom = s
                .domain_types
                .first()
                .map(|(t, _)| t.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let rng = s
                .range_types
                .first()
                .map(|(t, _)| t.clone())
                .unwrap_or_else(|| "unknown".to_string());
            user.push_str(&format!(
                "- {} ({} edges, domain: {}, range: {})\n",
                s.predicate, s.edge_count, dom, rng
            ));
        }
        user.push_str(&format!(
            "\nExisting parent properties: {}\n",
            existing_parents.join(", ")
        ));
        Some((system, user))
    }

    /// Parse the LLM's hierarchy-discovery response and create proposals.
    /// Stats are needed for domain/range lookup on each child predicate.
    pub fn parse_hierarchy_response(
        &mut self,
        response: &str,
        ontology: &OntologyRegistry,
        stats: &[PredicateStats],
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

        let stats_by_pred: HashMap<&str, &PredicateStats> =
            stats.iter().map(|s| (s.predicate.as_str(), s)).collect();

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
                    canonical_predicate: String::new(),
                    cascade_dependents: Vec::new(),
                    cascade_dependent: false,
                });
            }
            for child_id in &children {
                let (domain, range) = match stats_by_pred.get(child_id.as_str()) {
                    Some(s) => (
                        s.domain_types
                            .first()
                            .map(|(t, _)| vec![t.clone()])
                            .unwrap_or_default(),
                        s.range_types
                            .first()
                            .map(|(t, _)| vec![t.clone()])
                            .unwrap_or_default(),
                    ),
                    None => (Vec::new(), Vec::new()),
                };
                properties.push(ProposedProperty {
                    id: child_id.clone(),
                    label: predicate_to_label(child_id),
                    comment: format!("Sub-property of {}: {}", parent, rationale),
                    domain,
                    range,
                    characteristics: Vec::new(),
                    max_cardinality: None,
                    sub_property_of: Some(parent.clone()),
                    append_only: false,
                    canonical_predicate: child_id.clone(),
                    cascade_dependents: Vec::new(),
                    cascade_dependent: false,
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

    // ── LLM-Assisted Cascade & Temporal Dependency Inference ─────────────
    // Unchanged from the prior implementation. Kept verbatim so existing
    // /api/ontology/cascade-inference and /api/ontology/upload continue
    // working.

    /// Build the prompt for cascade-dependency inference.
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

    /// Parse the cascade-inference response and update the ontology registry
    /// with inferred cascade_dependents / cascade_dependent flags. Returns the
    /// number of properties updated.
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
            if ontology.update_cascade_dependents(trigger, &valid_dependents) {
                updated += 1;
                tracing::info!(
                    "Cascade inference: {} triggers cascade on [{}]",
                    trigger,
                    valid_dependents.join(", ")
                );
            }
            for dep in &valid_dependents {
                if ontology.mark_cascade_dependent(dep) {
                    updated += 1;
                }
            }
        }
        updated
    }

    // ── Proposal Lifecycle ───────────────────────────────────────────────

    pub fn proposals(&self) -> &[OntologyProposal] {
        &self.proposals
    }

    pub fn get_proposal(&self, id: u64) -> Option<&OntologyProposal> {
        self.proposals.iter().find(|p| p.id == id)
    }

    pub fn approve_proposal(&mut self, id: u64) -> bool {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            if p.status == ProposalStatus::Pending {
                p.status = ProposalStatus::Approved;
                return true;
            }
        }
        false
    }

    pub fn reject_proposal(&mut self, id: u64) -> bool {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            if p.status == ProposalStatus::Pending {
                p.status = ProposalStatus::Rejected;
                return true;
            }
        }
        false
    }

    /// Apply an approved proposal: write the TTL file and register the
    /// properties into the ontology (hot-reload). Marks the proposal Applied.
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

        let ttl_path = self
            .config
            .generated_ttl_dir
            .join(format!("evolved_{:04}.ttl", id));
        std::fs::write(&ttl_path, &proposal.ttl_preview).map_err(OntologyEvolutionError::Io)?;
        tracing::info!("Wrote evolved ontology to {:?}", ttl_path);

        let properties = proposal.properties.clone();
        let mut registered = 0;
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

    // ── Full Discovery Pass (graph scan + behaviour proposals) ───────────

    /// Run a complete behaviour-inference pass: scan the graph, infer
    /// behaviours, create proposals. Returns IDs of newly created proposals.
    ///
    /// The LLM-assisted hierarchy discovery is a separate operation —
    /// callers that want it should also invoke `build_hierarchy_discovery_prompt`,
    /// run the LLM, and pass the response to `parse_hierarchy_response`.
    /// This split keeps the engine lock-free during the LLM call.
    pub fn run_inference_pass(&mut self, graph: &Graph, ontology: &OntologyRegistry) -> Vec<u64> {
        let stats = Self::snapshot_predicate_stats(graph);
        let behaviours = self.infer_behaviours(&stats, ontology);
        if behaviours.is_empty() {
            return Vec::new();
        }
        tracing::info!(
            "Ontology evolution: inferred behaviours for {} predicates from {} stats",
            behaviours.len(),
            stats.len()
        );
        self.create_proposals_from_inferred(&behaviours, ontology, &stats)
    }

    /// Per-predicate stats threshold for callers that build LLM prompts.
    pub fn min_observations(&self) -> u64 {
        self.config.min_observations
    }

    /// Summary statistics. Predicate counts come from a fresh graph scan
    /// when callers ask; this method only reports proposal-side counts.
    pub fn stats(&self) -> EvolutionStats {
        EvolutionStats {
            pending_proposals: self
                .proposals
                .iter()
                .filter(|p| p.status == ProposalStatus::Pending)
                .count(),
            approved_proposals: self
                .proposals
                .iter()
                .filter(|p| p.status == ProposalStatus::Approved)
                .count(),
            rejected_proposals: self
                .proposals
                .iter()
                .filter(|p| p.status == ProposalStatus::Rejected)
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
    pub pending_proposals: usize,
    pub approved_proposals: usize,
    pub rejected_proposals: usize,
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

/// Stable string label for a `NodeType` variant. Bounded set, used to
/// populate `domain_types` / `range_types` in `PredicateStats`.
fn node_type_label(nt: &NodeType) -> String {
    match nt {
        NodeType::Concept { concept_type, .. } => format!("Concept:{:?}", concept_type),
        NodeType::Agent { .. } => "Agent".to_string(),
        NodeType::Event { .. } => "Event".to_string(),
        NodeType::Context { .. } => "Context".to_string(),
        NodeType::Goal { .. } => "Goal".to_string(),
        NodeType::Episode { .. } => "Episode".to_string(),
        NodeType::Memory { .. } => "Memory".to_string(),
        NodeType::Strategy { .. } => "Strategy".to_string(),
        NodeType::Tool { .. } => "Tool".to_string(),
        NodeType::Result { .. } => "Result".to_string(),
        NodeType::Claim { .. } => "Claim".to_string(),
    }
}

/// Sort a count HashMap descending and keep the top `n` entries.
fn top_n(counts: HashMap<String, u64>, n: usize) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.truncate(n);
    v
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

    fn make_stats(predicate: &str, edges: u64, forward: u64, reverse: u64) -> PredicateStats {
        PredicateStats {
            category: "hobby".to_string(),
            predicate: predicate.to_string(),
            edge_count: edges,
            forward_active: forward,
            reverse_active: reverse,
            superseded: edges - forward,
            distinct_subjects: 0,
            single_value_subjects: 0,
            domain_types: vec![("Concept:Person".to_string(), forward)],
            range_types: vec![("Concept:NamedEntity".to_string(), forward)],
            first_valid_from: Some(1000),
            last_valid_from: Some(2000),
        }
    }

    #[test]
    fn test_predicate_to_label() {
        assert_eq!(predicate_to_label("plays_tennis"), "Plays Tennis");
        assert_eq!(predicate_to_label("friend"), "Friend");
        assert_eq!(predicate_to_label("lives_in"), "Lives In");
    }

    #[test]
    fn test_top_n_takes_largest_first() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 1);
        m.insert("b".to_string(), 3);
        m.insert("c".to_string(), 2);
        let out = top_n(m, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], ("b".to_string(), 3));
        assert_eq!(out[1], ("c".to_string(), 2));
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
            canonical_predicate: "plays_tennis".to_string(),
            cascade_dependents: Vec::new(),
            cascade_dependent: false,
        }];

        let ttl = generate_ttl_for_properties(&props);
        assert!(ttl.contains("eg:plays_tennis"));
        assert!(ttl.contains("rdfs:subPropertyOf eg:hobby"));
        assert!(ttl.contains("owl:SymmetricProperty"));
        assert!(ttl.contains("rdfs:domain eg:Person"));
    }

    #[test]
    fn test_infer_behaviours_flags_symmetric_above_threshold() {
        let engine = OntologyEvolutionEngine::new(OntologyEvolutionConfig {
            min_observations: 2,
            symmetric_ratio_threshold: 0.85,
            ..OntologyEvolutionConfig::default()
        });
        let ontology = OntologyRegistry::new();
        // 10 forward, 9 reverse → ratio 0.9 → symmetric
        let stats = vec![make_stats("friend_of", 10, 10, 9)];
        let behaviours = engine.infer_behaviours(&stats, &ontology);
        assert_eq!(behaviours.len(), 1);
        assert!(behaviours[0].is_symmetric);
    }

    #[test]
    fn test_infer_behaviours_respects_min_observations() {
        let engine = OntologyEvolutionEngine::new(OntologyEvolutionConfig {
            min_observations: 5,
            ..OntologyEvolutionConfig::default()
        });
        let ontology = OntologyRegistry::new();
        let stats = vec![make_stats("rare_pred", 3, 3, 3)];
        let behaviours = engine.infer_behaviours(&stats, &ontology);
        assert!(behaviours.is_empty());
    }

    #[test]
    fn test_proposal_lifecycle() {
        let mut engine = OntologyEvolutionEngine::new(OntologyEvolutionConfig::default());
        let behaviours = vec![InferredBehavior {
            predicate: "plays_chess".to_string(),
            category: "hobby".to_string(),
            is_symmetric: false,
            is_functional: false,
            is_append_only: true,
            has_supersession: false,
            confidence: 0.8,
            sample_size: 10,
        }];
        let stats = vec![make_stats("plays_chess", 10, 10, 0)];
        let ontology = OntologyRegistry::new();
        let ids = engine.create_proposals_from_inferred(&behaviours, &ontology, &stats);
        assert_eq!(ids.len(), 1);
        let p = engine.get_proposal(ids[0]).unwrap();
        assert_eq!(p.status, ProposalStatus::Pending);

        assert!(engine.approve_proposal(ids[0]));
        let p = engine.get_proposal(ids[0]).unwrap();
        assert_eq!(p.status, ProposalStatus::Approved);
        // can't approve again
        assert!(!engine.approve_proposal(ids[0]));
    }

    #[test]
    fn test_reject_proposal() {
        let mut engine = OntologyEvolutionEngine::new(OntologyEvolutionConfig::default());
        let behaviours = vec![InferredBehavior {
            predicate: "p".to_string(),
            category: "c".to_string(),
            is_symmetric: false,
            is_functional: false,
            is_append_only: false,
            has_supersession: false,
            confidence: 0.5,
            sample_size: 5,
        }];
        let stats = vec![make_stats("p", 5, 5, 0)];
        let ontology = OntologyRegistry::new();
        let ids = engine.create_proposals_from_inferred(&behaviours, &ontology, &stats);
        assert!(engine.reject_proposal(ids[0]));
        assert_eq!(
            engine.get_proposal(ids[0]).unwrap().status,
            ProposalStatus::Rejected
        );
        // Cannot approve a rejected proposal
        assert!(!engine.approve_proposal(ids[0]));
    }

    #[test]
    fn test_hierarchy_prompt_returns_none_when_no_unknowns() {
        let ontology = OntologyRegistry::new();
        let stats: Vec<PredicateStats> = Vec::new();
        let prompt =
            OntologyEvolutionEngine::build_hierarchy_discovery_prompt(&stats, &ontology, 5);
        assert!(prompt.is_none());
    }

    #[test]
    fn test_evolution_stats() {
        let mut engine = OntologyEvolutionEngine::new(OntologyEvolutionConfig::default());
        let behaviours = vec![InferredBehavior {
            predicate: "p".to_string(),
            category: "c".to_string(),
            is_symmetric: false,
            is_functional: false,
            is_append_only: false,
            has_supersession: false,
            confidence: 0.5,
            sample_size: 5,
        }];
        let stats = vec![make_stats("p", 5, 5, 0)];
        let ontology = OntologyRegistry::new();
        engine.create_proposals_from_inferred(&behaviours, &ontology, &stats);
        let s = engine.stats();
        assert_eq!(s.pending_proposals, 1);
        assert_eq!(s.approved_proposals, 0);
        assert_eq!(s.applied_proposals, 0);
    }

    #[test]
    fn test_cascade_inference_prompt_includes_all_properties() {
        let ontology = crate::ontology::tests::test_registry();
        let (system, user) = OntologyEvolutionEngine::build_cascade_inference_prompt(&ontology);
        assert!(system.contains("cascade"));
        assert!(system.contains("temporal dependencies"));
        assert!(user.contains("location"));
        assert!(user.contains("relationship"));
        assert!(user.contains("financial"));
        assert!(user.contains("routine"));
        assert!(user.contains("Functional"));
        assert!(user.contains("Symmetric"));
    }

    #[test]
    fn test_parse_cascade_inference_response() {
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

        let response = r#"{
            "cascade_rules": [
                {
                    "trigger": "location",
                    "dependents": ["routine"],
                    "rationale": "Moving cities invalidates daily routine"
                }
            ]
        }"#;
        let updated =
            OntologyEvolutionEngine::parse_cascade_inference_response(response, &fresh_ontology);
        assert!(updated >= 2);
        let location = fresh_ontology.resolve("location").unwrap();
        assert!(location.cascade_dependents.contains(&"routine".to_string()));
        let routine = fresh_ontology.resolve("routine").unwrap();
        assert!(routine.cascade_dependent);
    }

    #[test]
    fn test_parse_cascade_inference_rejects_non_functional_trigger() {
        let ontology = OntologyRegistry::new();
        ontology.register_property(crate::ontology::PropertyDescriptor {
            id: "not_functional".into(),
            label: "Not Functional".into(),
            comment: "".into(),
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
        let response = r#"{"cascade_rules":[{"trigger":"not_functional","dependents":["foo"],"rationale":"x"}]}"#;
        let updated =
            OntologyEvolutionEngine::parse_cascade_inference_response(response, &ontology);
        assert_eq!(updated, 0);
    }

    #[test]
    fn test_proposed_property_cascade_fields_in_ttl() {
        let props = vec![ProposedProperty {
            id: "moves_to".into(),
            label: "Moves To".into(),
            comment: String::new(),
            domain: Vec::new(),
            range: Vec::new(),
            characteristics: vec![PropertyCharacteristic::Functional],
            max_cardinality: Some(1),
            sub_property_of: None,
            append_only: false,
            canonical_predicate: "moves_to".into(),
            cascade_dependents: vec!["neighbor".into(), "routine".into()],
            cascade_dependent: false,
        }];
        let ttl = generate_ttl_for_properties(&props);
        assert!(ttl.contains("eg:cascadeDependents eg:neighbor, eg:routine"));
    }
}
