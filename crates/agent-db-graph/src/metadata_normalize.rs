//! Resilient metadata normalization for structured memory auto-detection.
//!
//! Provides a multi-tier alias resolution system that maps arbitrary metadata
//! key names to canonical roles (Entity, NewState, From, To, Amount, etc.).
//! An optional LLM fallback handles keys that resist rule-based resolution.

use agent_db_events::core::MetadataValue;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ────────── Core types ──────────

/// Canonical role a metadata key can play in structured memory detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataRole {
    Entity,
    NewState,
    OldState,
    From,
    To,
    Amount,
    Direction,
    Description,
}

impl MetadataRole {
    /// Parse a canonical role name (as returned by the LLM).
    pub fn parse_role(s: &str) -> Option<Self> {
        match s {
            "entity" => Some(Self::Entity),
            "new_state" => Some(Self::NewState),
            "old_state" => Some(Self::OldState),
            "from" => Some(Self::From),
            "to" => Some(Self::To),
            "amount" => Some(Self::Amount),
            "direction" => Some(Self::Direction),
            "description" => Some(Self::Description),
            _ => None,
        }
    }

    /// All roles in enum order (used for disambiguation tie-breaking).
    const ALL: &'static [MetadataRole] = &[
        Self::Entity,
        Self::NewState,
        Self::OldState,
        Self::From,
        Self::To,
        Self::Amount,
        Self::Direction,
        Self::Description,
    ];

    /// Whether this role expects a numeric value.
    fn expects_numeric(self) -> bool {
        matches!(self, Self::Amount)
    }
}

/// How a key was resolved to its role.
#[derive(Debug, Clone)]
pub enum ResolutionMethod {
    ExactCanonical,
    ExactCustom,
    StemMatch,
    BigramMatch(f32),
    LlmNormalized,
}

/// A key resolved to a canonical role.
#[derive(Debug)]
pub struct ResolvedKey {
    pub original_key: String,
    pub method: ResolutionMethod,
}

/// Result of normalizing an event's metadata keys.
#[derive(Debug, Default)]
pub struct NormalizedMetadata {
    pub roles: HashMap<MetadataRole, ResolvedKey>,
}

impl NormalizedMetadata {
    /// Get a string value for a resolved role.
    pub fn get_str(
        &self,
        role: MetadataRole,
        md: &HashMap<String, MetadataValue>,
    ) -> Option<String> {
        let resolved = self.roles.get(&role)?;
        let val = md.get(&resolved.original_key)?;
        metadata_as_str(val)
    }

    /// Get a float value for a resolved role.
    pub fn get_f64(&self, role: MetadataRole, md: &HashMap<String, MetadataValue>) -> Option<f64> {
        let resolved = self.roles.get(&role)?;
        let val = md.get(&resolved.original_key)?;
        metadata_as_f64(val)
    }

    /// Check if a role was resolved.
    pub fn has_role(&self, role: MetadataRole) -> bool {
        self.roles.contains_key(&role)
    }
}

// ────────── Alias configuration ──────────

/// User-provided custom key mappings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AliasConfig {
    /// Custom mappings: `"absender" -> "from"`, `"betrag" -> "amount"`, etc.
    pub custom_mappings: HashMap<String, String>,
}

// ────────── Role specs (stems + type info) ──────────

struct RoleSpec {
    stems: &'static [&'static str],
    #[allow(dead_code)]
    expects_numeric: bool,
}

fn role_specs() -> Vec<(MetadataRole, RoleSpec)> {
    vec![
        (
            MetadataRole::Entity,
            RoleSpec {
                stems: &["entit", "actor", "subject"],
                expects_numeric: false,
            },
        ),
        (
            MetadataRole::NewState,
            RoleSpec {
                stems: &["state", "status"],
                expects_numeric: false,
            },
        ),
        (
            MetadataRole::OldState,
            RoleSpec {
                stems: &["old_state", "prev"],
                expects_numeric: false,
            },
        ),
        (
            MetadataRole::From,
            RoleSpec {
                stems: &["sender", "source", "payer", "origin"],
                expects_numeric: false,
            },
        ),
        (
            MetadataRole::To,
            RoleSpec {
                stems: &["recip", "dest", "payee", "receiver"],
                expects_numeric: false,
            },
        ),
        (
            MetadataRole::Amount,
            RoleSpec {
                stems: &["amount", "value", "total", "quant", "price", "cost"],
                expects_numeric: true,
            },
        ),
        (
            MetadataRole::Direction,
            RoleSpec {
                stems: &["direction"],
                expects_numeric: false,
            },
        ),
        (
            MetadataRole::Description,
            RoleSpec {
                stems: &["descr", "memo", "note"],
                expects_numeric: false,
            },
        ),
    ]
}

// ────────── Built-in exact aliases ──────────

fn built_in_aliases() -> Vec<(&'static str, MetadataRole)> {
    vec![
        // Entity
        ("entity", MetadataRole::Entity),
        ("entity_name", MetadataRole::Entity),
        ("subject", MetadataRole::Entity),
        ("actor", MetadataRole::Entity),
        ("target_entity", MetadataRole::Entity),
        ("object", MetadataRole::Entity),
        ("item", MetadataRole::Entity),
        ("node", MetadataRole::Entity),
        // NewState
        ("new_state", MetadataRole::NewState),
        // NOTE: "entity_state" is a boolean flag ("true"/"false"), NOT the actual state value.
        // Do NOT alias it to NewState — the actual value is in "new_state".
        ("state", MetadataRole::NewState),
        ("status", MetadataRole::NewState),
        ("new_status", MetadataRole::NewState),
        ("current_state", MetadataRole::NewState),
        // OldState
        ("old_state", MetadataRole::OldState),
        ("previous_state", MetadataRole::OldState),
        ("old_status", MetadataRole::OldState),
        ("prev_state", MetadataRole::OldState),
        ("prior_state", MetadataRole::OldState),
        // From
        ("from", MetadataRole::From),
        ("from_entity", MetadataRole::From),
        ("sender", MetadataRole::From),
        ("source", MetadataRole::From),
        ("payer", MetadataRole::From),
        ("origin", MetadataRole::From),
        ("debtor", MetadataRole::From),
        // To
        ("to", MetadataRole::To),
        ("to_entity", MetadataRole::To),
        ("recipient", MetadataRole::To),
        ("target", MetadataRole::To),
        ("dest", MetadataRole::To),
        ("destination", MetadataRole::To),
        ("payee", MetadataRole::To),
        ("receiver", MetadataRole::To),
        ("creditor", MetadataRole::To),
        // Amount
        ("amount", MetadataRole::Amount),
        ("value", MetadataRole::Amount),
        ("sum", MetadataRole::Amount),
        ("total", MetadataRole::Amount),
        ("quantity", MetadataRole::Amount),
        ("price", MetadataRole::Amount),
        ("cost", MetadataRole::Amount),
        // Direction
        ("direction", MetadataRole::Direction),
        ("txn_type", MetadataRole::Direction),
        ("transaction_type", MetadataRole::Direction),
        // Description
        ("description", MetadataRole::Description),
        ("note", MetadataRole::Description),
        ("memo", MetadataRole::Description),
        ("reason", MetadataRole::Description),
        ("label", MetadataRole::Description),
        ("comment", MetadataRole::Description),
        ("narrative", MetadataRole::Description),
    ]
}

// ────────── MetadataNormalizer ──────────

/// Multi-tier metadata key normalizer.
pub struct MetadataNormalizer {
    /// O(1) lookup: lowercased key -> role (built-in + custom).
    exact_lookup: HashMap<String, MetadataRole>,
    /// Per-role stem specs for tier 2.
    role_specs: Vec<(MetadataRole, RoleSpec)>,
}

impl MetadataNormalizer {
    /// Create a new normalizer with optional custom aliases.
    pub fn new(config: &AliasConfig) -> Self {
        let mut exact_lookup = HashMap::new();

        // Insert built-in aliases
        for (key, role) in built_in_aliases() {
            exact_lookup.insert(key.to_string(), role);
        }

        // Insert custom mappings (may override built-ins)
        for (custom_key, canonical_name) in &config.custom_mappings {
            if let Some(role) = MetadataRole::parse_role(canonical_name) {
                exact_lookup.insert(custom_key.to_lowercase(), role);
            }
        }

        Self {
            exact_lookup,
            role_specs: role_specs(),
        }
    }

    /// Normalize metadata keys to canonical roles (sync, no LLM).
    pub fn normalize(&self, md: &HashMap<String, MetadataValue>) -> NormalizedMetadata {
        let mut result = NormalizedMetadata::default();

        // Collect all candidate matches with scores for disambiguation
        struct Candidate {
            key: String,
            role: MetadataRole,
            tier: u8, // 4=exact_canonical, 3=exact_custom, 2=stem, 1=bigram
            method: ResolutionMethod,
        }

        let mut candidates: Vec<Candidate> = Vec::new();

        for (key, value) in md {
            let lower = key.to_lowercase();

            // Tier 1: exact match
            if let Some(&role) = self.exact_lookup.get(&lower) {
                // Type guard
                if role.expects_numeric() && !is_numeric_value(value) {
                    continue;
                }
                // Determine if built-in or custom
                let is_built_in = built_in_aliases().iter().any(|(k, _)| *k == lower.as_str());
                let tier = if is_built_in { 4 } else { 3 };
                let method = if is_built_in {
                    ResolutionMethod::ExactCanonical
                } else {
                    ResolutionMethod::ExactCustom
                };
                candidates.push(Candidate {
                    key: key.clone(),
                    role,
                    tier,
                    method,
                });
                continue;
            }

            // Tier 2: token-aware stem match (exact token == stem)
            let tokens = tokenize_key(&lower);
            let mut stem_matched = false;
            for (role, spec) in &self.role_specs {
                if role.expects_numeric() && !is_numeric_value(value) {
                    continue;
                }
                for stem in spec.stems {
                    // Token must exactly equal the stem
                    if tokens.iter().any(|t| t == stem) {
                        candidates.push(Candidate {
                            key: key.clone(),
                            role: *role,
                            tier: 2,
                            method: ResolutionMethod::StemMatch,
                        });
                        stem_matched = true;
                        break;
                    }
                }
            }
            if stem_matched {
                continue;
            }

            // Tier 3: bigram similarity (only for keys >= 4 chars)
            if lower.len() >= 4 {
                let mut best_bigram: Option<(MetadataRole, f32)> = None;
                for (alias, role) in built_in_aliases() {
                    if alias.len() < 4 {
                        continue;
                    }
                    if role.expects_numeric() && !is_numeric_value(value) {
                        continue;
                    }
                    let sim = ngram_similarity(&lower, alias, 2);
                    if sim > 0.5 && (best_bigram.is_none() || sim > best_bigram.unwrap().1) {
                        best_bigram = Some((role, sim));
                    }
                }
                if let Some((role, sim)) = best_bigram {
                    candidates.push(Candidate {
                        key: key.clone(),
                        role,
                        tier: 1,
                        method: ResolutionMethod::BigramMatch(sim),
                    });
                }
            }
        }

        // Sort candidates by tier descending (highest first), then by enum order for ties
        candidates.sort_by(|a, b| {
            b.tier.cmp(&a.tier).then_with(|| {
                let a_idx = MetadataRole::ALL
                    .iter()
                    .position(|r| *r == a.role)
                    .unwrap_or(usize::MAX);
                let b_idx = MetadataRole::ALL
                    .iter()
                    .position(|r| *r == b.role)
                    .unwrap_or(usize::MAX);
                a_idx.cmp(&b_idx)
            })
        });

        // Assign: one key per role, one role per key
        let mut assigned_roles: HashSet<MetadataRole> = HashSet::new();
        let mut assigned_keys: HashSet<String> = HashSet::new();

        for c in candidates {
            if assigned_roles.contains(&c.role) || assigned_keys.contains(&c.key) {
                continue;
            }
            assigned_roles.insert(c.role);
            assigned_keys.insert(c.key.clone());
            result.roles.insert(
                c.role,
                ResolvedKey {
                    original_key: c.key,
                    method: c.method,
                },
            );
        }

        result
    }

    /// Apply LLM-returned mappings on top of alias-resolved results.
    ///
    /// Rules:
    /// - Ignore unknown canonical role strings
    /// - Apply type guards before accepting
    /// - One key per role, one role per key (first wins on collision)
    /// - LLM returning "unknown" preserves alias result
    /// - Roles already resolved by alias are NOT overwritten
    pub fn apply_llm_mappings(
        &self,
        llm_mappings: &HashMap<String, String>,
        md: &HashMap<String, MetadataValue>,
        alias_result: &NormalizedMetadata,
    ) -> NormalizedMetadata {
        let mut result = NormalizedMetadata::default();

        // Copy alias results
        let mut assigned_roles: HashSet<MetadataRole> = HashSet::new();
        let mut assigned_keys: HashSet<String> = HashSet::new();

        for (role, resolved) in &alias_result.roles {
            result.roles.insert(
                *role,
                ResolvedKey {
                    original_key: resolved.original_key.clone(),
                    method: match &resolved.method {
                        ResolutionMethod::ExactCanonical => ResolutionMethod::ExactCanonical,
                        ResolutionMethod::ExactCustom => ResolutionMethod::ExactCustom,
                        ResolutionMethod::StemMatch => ResolutionMethod::StemMatch,
                        ResolutionMethod::BigramMatch(s) => ResolutionMethod::BigramMatch(*s),
                        ResolutionMethod::LlmNormalized => ResolutionMethod::LlmNormalized,
                    },
                },
            );
            assigned_roles.insert(*role);
            assigned_keys.insert(resolved.original_key.clone());
        }

        // Apply LLM mappings for unresolved keys/roles
        for (original_key, canonical_name) in llm_mappings {
            if canonical_name == "unknown" {
                continue;
            }
            let Some(role) = MetadataRole::parse_role(canonical_name) else {
                continue;
            };
            if assigned_roles.contains(&role) || assigned_keys.contains(original_key) {
                continue;
            }
            // Type guard
            if let Some(value) = md.get(original_key) {
                if role.expects_numeric() && !is_numeric_value(value) {
                    continue;
                }
            } else {
                continue;
            }
            assigned_roles.insert(role);
            assigned_keys.insert(original_key.clone());
            result.roles.insert(
                role,
                ResolvedKey {
                    original_key: original_key.clone(),
                    method: ResolutionMethod::LlmNormalized,
                },
            );
        }

        result
    }
}

// ────────── Helper functions ──────────

/// Extract a string from a MetadataValue.
pub fn metadata_as_str(v: &MetadataValue) -> Option<String> {
    match v {
        MetadataValue::String(s) => Some(s.clone()),
        MetadataValue::Json(serde_json::Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Extract a float from a MetadataValue.
pub fn metadata_as_f64(v: &MetadataValue) -> Option<f64> {
    match v {
        MetadataValue::Float(f) => Some(*f),
        MetadataValue::Integer(i) => Some(*i as f64),
        MetadataValue::Json(serde_json::Value::Number(n)) => n.as_f64(),
        _ => None,
    }
}

/// Check if a MetadataValue is numeric.
fn is_numeric_value(v: &MetadataValue) -> bool {
    matches!(
        v,
        MetadataValue::Float(_)
            | MetadataValue::Integer(_)
            | MetadataValue::Json(serde_json::Value::Number(_))
    )
}

/// Produce a short preview of a MetadataValue for the LLM.
pub fn metadata_value_preview(v: &MetadataValue) -> String {
    match v {
        MetadataValue::String(s) => {
            if s.len() > 80 {
                let mut end = 80;
                while end > 0 && !s.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &s[..end])
            } else {
                s.clone()
            }
        },
        MetadataValue::Integer(i) => i.to_string(),
        MetadataValue::Float(f) => f.to_string(),
        MetadataValue::Boolean(b) => b.to_string(),
        MetadataValue::Json(v) => {
            let s = v.to_string();
            if s.len() > 80 {
                let mut end = 80;
                while end > 0 && !s.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &s[..end])
            } else {
                s
            }
        },
    }
}

/// Split a key into tokens on `_`, `-`, camelCase boundaries, and digits.
pub fn tokenize_key(key: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in key.chars() {
        if ch == '_' || ch == '-' {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else if ch.is_ascii_digit() {
            if !current.is_empty() && !current.chars().last().is_some_and(|c| c.is_ascii_digit()) {
                tokens.push(std::mem::take(&mut current));
            }
            current.push(ch);
        } else if ch.is_uppercase() {
            // camelCase boundary
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            current.push(ch.to_lowercase().next().unwrap_or(ch));
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Character n-gram Jaccard similarity.
fn ngram_similarity(a: &str, b: &str, n: usize) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() < n || b.len() < n {
        return 0.0;
    }
    let ngrams_a: HashSet<&[u8]> = a.as_bytes().windows(n).collect();
    let ngrams_b: HashSet<&[u8]> = b.as_bytes().windows(n).collect();
    let intersection = ngrams_a.intersection(&ngrams_b).count();
    let union = ngrams_a.union(&ngrams_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

// ────────── LLM Metadata Normalizer ──────────

const LLM_SYSTEM_PROMPT: &str = concat!(
    "You normalize metadata key names for a structured memory system.\n",
    "Given a list of key-value pairs, map each key to one of these canonical roles:\n",
    "\"entity\", \"new_state\", \"old_state\", \"from\", \"to\", \"amount\", ",
    "\"direction\", \"description\", \"unknown\"\n",
    "Output strict JSON: {\"mappings\": {\"original_key\": \"canonical_role\", ...}}\n",
    "No markdown fences, no explanation.",
);

/// Async trait for LLM-based metadata normalization.
#[async_trait]
pub trait MetadataLlmNormalizer: Send + Sync {
    async fn normalize_keys(
        &self,
        keys_with_samples: &[(String, String)],
    ) -> anyhow::Result<Option<HashMap<String, String>>>;
}

#[derive(Debug, Deserialize)]
struct LlmMappingResponse {
    mappings: HashMap<String, String>,
}

fn parse_mapping_response(text: &str) -> Option<HashMap<String, String>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let json_str = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };
    serde_json::from_str::<LlmMappingResponse>(json_str)
        .ok()
        .map(|r| r.mappings)
}

fn format_llm_input(keys_with_samples: &[(String, String)]) -> String {
    let mut s = String::from("Key-value pairs:\n");
    for (key, sample) in keys_with_samples {
        s.push_str(&format!("- \"{}\": \"{}\"\n", key, sample));
    }
    s
}

// ────────── OpenAI client ──────────

pub struct OpenAiMetadataNormalizer {
    api_key: String,
    model: String,
    http: reqwest::Client,
    cache: tokio::sync::Mutex<NormCache>,
}

impl OpenAiMetadataNormalizer {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
            cache: tokio::sync::Mutex::new(NormCache::new(500)),
        }
    }
}

#[async_trait]
impl MetadataLlmNormalizer for OpenAiMetadataNormalizer {
    async fn normalize_keys(
        &self,
        keys_with_samples: &[(String, String)],
    ) -> anyhow::Result<Option<HashMap<String, String>>> {
        let cache_key = make_cache_key(keys_with_samples);
        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(Some(cached.clone()));
            }
        }

        let user_msg = format_llm_input(keys_with_samples);
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 256,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": LLM_SYSTEM_PROMPT },
                { "role": "user", "content": user_msg }
            ]
        });

        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        let result = parse_mapping_response(text);
        if let Some(ref mappings) = result {
            let mut cache = self.cache.lock().await;
            cache.insert(cache_key, mappings.clone());
        }
        Ok(result)
    }
}

// ────────── Anthropic client ──────────

pub struct AnthropicMetadataNormalizer {
    api_key: String,
    model: String,
    http: reqwest::Client,
    cache: tokio::sync::Mutex<NormCache>,
}

impl AnthropicMetadataNormalizer {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
            cache: tokio::sync::Mutex::new(NormCache::new(500)),
        }
    }
}

#[async_trait]
impl MetadataLlmNormalizer for AnthropicMetadataNormalizer {
    async fn normalize_keys(
        &self,
        keys_with_samples: &[(String, String)],
    ) -> anyhow::Result<Option<HashMap<String, String>>> {
        let cache_key = make_cache_key(keys_with_samples);
        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(Some(cached.clone()));
            }
        }

        let user_msg = format_llm_input(keys_with_samples);
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 256,
            "system": LLM_SYSTEM_PROMPT,
            "messages": [
                { "role": "user", "content": user_msg }
            ]
        });

        let resp = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;
        let text = json["content"][0]["text"].as_str().unwrap_or("");

        let result = parse_mapping_response(text);
        if let Some(ref mappings) = result {
            let mut cache = self.cache.lock().await;
            cache.insert(cache_key, mappings.clone());
        }
        Ok(result)
    }
}

// ────────── Cache ──────────

struct NormCache {
    order: VecDeque<Vec<String>>,
    entries: HashMap<Vec<String>, HashMap<String, String>>,
    max_size: usize,
}

impl NormCache {
    fn new(max_size: usize) -> Self {
        Self {
            order: VecDeque::new(),
            entries: HashMap::new(),
            max_size,
        }
    }

    fn get(&self, key: &[String]) -> Option<&HashMap<String, String>> {
        self.entries.get(key)
    }

    fn insert(&mut self, key: Vec<String>, value: HashMap<String, String>) {
        if self.entries.contains_key(&key) {
            return;
        }
        if self.entries.len() >= self.max_size {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }
}

fn make_cache_key(keys_with_samples: &[(String, String)]) -> Vec<String> {
    let mut keys: Vec<String> = keys_with_samples
        .iter()
        .map(|(k, _)| k.to_lowercase())
        .collect();
    keys.sort();
    keys.dedup();
    keys
}

// ────────── Tests ──────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_db_events::core::MetadataValue;

    fn normalizer() -> MetadataNormalizer {
        MetadataNormalizer::new(&AliasConfig::default())
    }

    fn md(pairs: Vec<(&str, MetadataValue)>) -> HashMap<String, MetadataValue> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn s(val: &str) -> MetadataValue {
        MetadataValue::String(val.to_string())
    }

    fn f(val: f64) -> MetadataValue {
        MetadataValue::Float(val)
    }

    #[test]
    fn test_exact_canonical_keys() {
        let n = normalizer();
        let metadata = md(vec![
            ("entity", s("OrderA")),
            ("new_state", s("shipped")),
            ("amount", f(42.0)),
        ]);
        let result = n.normalize(&metadata);
        assert!(result.has_role(MetadataRole::Entity));
        assert!(result.has_role(MetadataRole::NewState));
        assert!(result.has_role(MetadataRole::Amount));
        assert_eq!(
            result.get_str(MetadataRole::Entity, &metadata),
            Some("OrderA".to_string())
        );
        assert_eq!(result.get_f64(MetadataRole::Amount, &metadata), Some(42.0));
    }

    #[test]
    fn test_custom_mapping_overrides() {
        let config = AliasConfig {
            custom_mappings: {
                let mut m = HashMap::new();
                m.insert("absender".to_string(), "from".to_string());
                m
            },
        };
        let n = MetadataNormalizer::new(&config);
        let metadata = md(vec![("absender", s("Alice")), ("amount", f(100.0))]);
        let result = n.normalize(&metadata);
        assert!(result.has_role(MetadataRole::From));
        assert_eq!(
            result.get_str(MetadataRole::From, &metadata),
            Some("Alice".to_string())
        );
    }

    #[test]
    fn test_token_stem_order_status() {
        let n = normalizer();
        let metadata = md(vec![
            ("order_status", s("shipped")),
            ("entity", s("Order1")),
        ]);
        let result = n.normalize(&metadata);
        // "order_status" -> tokens ["order", "status"] -> "status" == NewState stem
        assert!(result.has_role(MetadataRole::NewState));
        assert_eq!(
            result
                .roles
                .get(&MetadataRole::NewState)
                .unwrap()
                .original_key,
            "order_status"
        );
    }

    #[test]
    fn test_token_stem_destroyed_not_dest() {
        // "destroyed" -> tokens ["destroyed"]. No stem exactly equals "destroyed".
        let n = normalizer();
        let metadata = md(vec![("destroyed", s("value"))]);
        let result = n.normalize(&metadata);
        assert!(
            !result.has_role(MetadataRole::To),
            "destroyed should NOT match To"
        );
    }

    #[test]
    fn test_token_stem_costume_not_amount() {
        let n = normalizer();
        let metadata = md(vec![("costume", s("pirate"))]);
        let result = n.normalize(&metadata);
        assert!(
            !result.has_role(MetadataRole::Amount),
            "costume should NOT match Amount"
        );
    }

    #[test]
    fn test_bigram_recipent_typo() {
        let n = normalizer();
        let metadata = md(vec![("recipent", s("Bob"))]);
        let result = n.normalize(&metadata);
        assert!(result.has_role(MetadataRole::To));
        assert_eq!(
            result.roles.get(&MetadataRole::To).unwrap().original_key,
            "recipent"
        );
    }

    #[test]
    fn test_bigram_short_key_rejected() {
        let n = normalizer();
        // "tx" is too short (< 4 chars) for bigram matching
        let metadata = md(vec![("tx", s("some_val"))]);
        let result = n.normalize(&metadata);
        assert!(!result.has_role(MetadataRole::To));
    }

    #[test]
    fn test_type_guard_amount_string_rejected() {
        let n = normalizer();
        let metadata = md(vec![("amount", s("ten"))]);
        let result = n.normalize(&metadata);
        assert!(
            !result.has_role(MetadataRole::Amount),
            "String value should be rejected for Amount role"
        );
    }

    #[test]
    fn test_disambiguation_target_with_amount() {
        let n = normalizer();
        let metadata = md(vec![
            ("target", s("Bob")),
            ("amount", f(50.0)),
            ("entity", s("OrderX")),
        ]);
        let result = n.normalize(&metadata);
        assert!(result.has_role(MetadataRole::To));
        assert_eq!(
            result.roles.get(&MetadataRole::To).unwrap().original_key,
            "target"
        );
        assert!(result.has_role(MetadataRole::Entity));
        assert!(result.has_role(MetadataRole::Amount));
    }

    #[test]
    fn test_balance_alone_not_transaction() {
        let n = normalizer();
        let metadata = md(vec![("balance", f(100.0))]);
        let result = n.normalize(&metadata);
        assert!(
            !result.has_role(MetadataRole::Amount),
            "balance should not map to Amount"
        );
    }

    #[test]
    fn test_partial_resolution() {
        let n = normalizer();
        let metadata = md(vec![
            ("sender", s("Alice")),
            ("recipient", s("Bob")),
            ("sum_total", f(99.0)),
        ]);
        let result = n.normalize(&metadata);
        assert!(result.has_role(MetadataRole::From));
        assert!(result.has_role(MetadataRole::To));
        // "sum_total" -> tokens ["sum", "total"]. "total" exactly matches Amount stem.
        assert!(result.has_role(MetadataRole::Amount));
    }

    // ── LLM mapping tests ──

    #[test]
    fn test_llm_unknown_role_ignored() {
        let n = normalizer();
        let metadata = md(vec![("weird_key", s("val"))]);
        let alias_result = n.normalize(&metadata);

        let mut llm_mappings = HashMap::new();
        llm_mappings.insert("weird_key".to_string(), "nonexistent_role".to_string());

        let result = n.apply_llm_mappings(&llm_mappings, &metadata, &alias_result);
        assert!(result.roles.is_empty());
    }

    #[test]
    fn test_llm_duplicate_role_first_wins() {
        let n = normalizer();
        let metadata = md(vec![("key_a", s("Alice")), ("key_b", s("Bob"))]);
        let alias_result = n.normalize(&metadata);

        let mut llm_mappings = HashMap::new();
        llm_mappings.insert("key_a".to_string(), "from".to_string());
        llm_mappings.insert("key_b".to_string(), "from".to_string());

        let result = n.apply_llm_mappings(&llm_mappings, &metadata, &alias_result);
        assert!(result.has_role(MetadataRole::From));
        let count = result
            .roles
            .values()
            .filter(|r| r.original_key == "key_a" || r.original_key == "key_b")
            .count();
        assert_eq!(count, 1, "Only one key should be assigned to From");
    }

    #[test]
    fn test_llm_cache_hit() {
        let cache_key = make_cache_key(&[
            ("sender".to_string(), "Alice".to_string()),
            ("amount".to_string(), "100".to_string()),
        ]);
        let mut cache = NormCache::new(10);
        let mut mappings = HashMap::new();
        mappings.insert("sender".to_string(), "from".to_string());
        cache.insert(cache_key.clone(), mappings);

        let cached = cache.get(&cache_key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().get("sender"), Some(&"from".to_string()));
    }

    #[test]
    fn test_llm_unknown_preserves_alias() {
        let n = normalizer();
        let metadata = md(vec![("sender", s("Alice")), ("custom_field", s("xyz"))]);
        let alias_result = n.normalize(&metadata);
        assert!(alias_result.has_role(MetadataRole::From));

        let mut llm_mappings = HashMap::new();
        llm_mappings.insert("sender".to_string(), "unknown".to_string());
        llm_mappings.insert("custom_field".to_string(), "entity".to_string());

        let result = n.apply_llm_mappings(&llm_mappings, &metadata, &alias_result);
        assert!(result.has_role(MetadataRole::From));
        assert_eq!(
            result.roles.get(&MetadataRole::From).unwrap().original_key,
            "sender"
        );
        assert!(result.has_role(MetadataRole::Entity));
    }

    #[test]
    fn test_llm_type_guard_applied() {
        let n = normalizer();
        let metadata = md(vec![("weird_val", s("not a number"))]);
        let alias_result = n.normalize(&metadata);

        let mut llm_mappings = HashMap::new();
        llm_mappings.insert("weird_val".to_string(), "amount".to_string());

        let result = n.apply_llm_mappings(&llm_mappings, &metadata, &alias_result);
        assert!(
            !result.has_role(MetadataRole::Amount),
            "String value should be rejected for Amount even via LLM"
        );
    }

    // ── Tokenizer tests ──

    #[test]
    fn test_tokenize_camel_case() {
        assert_eq!(tokenize_key("orderStatus"), vec!["order", "status"]);
    }

    #[test]
    fn test_tokenize_snake_case() {
        assert_eq!(tokenize_key("from_entity"), vec!["from", "entity"]);
    }

    #[test]
    fn test_tokenize_mixed() {
        assert_eq!(tokenize_key("txnType123"), vec!["txn", "type", "123"]);
    }

    #[test]
    fn test_parse_mapping_response_valid() {
        let text = r#"{"mappings": {"sender": "from", "betrag": "amount"}}"#;
        let result = parse_mapping_response(text);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.get("sender"), Some(&"from".to_string()));
        assert_eq!(m.get("betrag"), Some(&"amount".to_string()));
    }

    #[test]
    fn test_parse_mapping_response_fenced() {
        let text = "```json\n{\"mappings\": {\"x\": \"entity\"}}\n```";
        let result = parse_mapping_response(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_mapping_response_invalid() {
        assert!(parse_mapping_response("not json").is_none());
        assert!(parse_mapping_response("").is_none());
    }
}
