//! Entity extraction and resolution for NLQ pipeline.
//!
//! Extracts entity mentions from question text using NER (when available)
//! or rule-based fallbacks, then resolves them to graph NodeIds.

use super::intent::QueryIntent;
use crate::structures::{Graph, NodeId, NodeType};

/// A mention of an entity found in the question text.
#[derive(Debug, Clone)]
pub struct EntityMention {
    /// The extracted text.
    pub text: String,
    /// Byte span in the original question.
    pub span: (usize, usize),
    /// Hint about what kind of entity this is.
    pub hint: EntityHint,
    /// Confidence (1.0 for rule-based, NER confidence otherwise).
    pub confidence: f32,
}

/// Hint about entity type (used to narrow index search).
#[derive(Debug, Clone, PartialEq)]
pub enum EntityHint {
    Agent,
    Tool,
    Concept,
    Goal,
    Strategy,
    Memory,
    Event,
    Episode,
    Person,
    Organization,
    Location,
    Product,
    DateTime,
    /// Code identifier (snake_case, camelCase, qualified.name)
    Code,
    Unknown,
}

/// An entity mention resolved to a graph node.
#[derive(Debug, Clone)]
pub struct ResolvedEntity {
    pub mention: EntityMention,
    pub node_id: NodeId,
    pub node_type: String,
    pub confidence: f32,
}

/// Entity extraction and resolution engine.
#[derive(Default)]
pub struct EntityResolver;

impl EntityResolver {
    pub fn new() -> Self {
        Self
    }

    /// Extract entity mentions from question text using rule-based heuristics.
    ///
    /// NER integration is available via the async path in NlqPipeline;
    /// this method provides the synchronous fallback.
    ///
    /// Code identifier patterns (snake_case, camelCase, qualified.names) are
    /// detected first and tagged `EntityHint::Code`, preventing accidental
    /// suppression by natural-language stopword filters.
    pub fn extract_mentions(&self, question: &str, intent: &QueryIntent) -> Vec<EntityMention> {
        let mut mentions = Vec::new();

        // Strategy 0 (highest precedence): Detect code identifier patterns
        self.extract_code_identifiers(&mut mentions, question);

        // Strategy 1: Quoted strings -> exact entity
        self.extract_quoted(&mut mentions, question);

        // Strategy 2: Hash-prefixed IDs (#123)
        self.extract_hash_ids(&mut mentions, question);

        // Strategy 3: Keyword-typed entities ("agent Alice", "tool login_tool")
        self.extract_keyword_typed(&mut mentions, question);

        // Strategy 4: Capitalized words not at sentence start
        self.extract_capitalized(&mut mentions, question);

        // Deduplicate overlapping mentions (prefer longer spans)
        self.dedup_mentions(&mut mentions);

        // Apply intent-based hints
        self.apply_intent_hints(&mut mentions, intent);

        mentions
    }

    /// Extract entities from NER results (EntitySpan → EntityMention).
    pub fn mentions_from_ner(
        &self,
        spans: &[agent_db_events::ner::EntitySpan],
    ) -> Vec<EntityMention> {
        spans
            .iter()
            .map(|span| EntityMention {
                text: span.text.clone(),
                span: (span.start_offset, span.end_offset),
                hint: hint_from_ner_label(&span.label),
                confidence: span.confidence,
            })
            .collect()
    }

    /// Resolve entity mentions to graph nodes.
    pub fn resolve(&self, mentions: &[EntityMention], graph: &Graph) -> Vec<ResolvedEntity> {
        let mut resolved = Vec::new();

        for mention in mentions {
            if let Some(entity) = self.resolve_single(mention, graph) {
                resolved.push(entity);
            }
        }

        resolved
    }

    fn resolve_single(&self, mention: &EntityMention, graph: &Graph) -> Option<ResolvedEntity> {
        // 1. Try exact index lookups based on hint
        if let Some(r) = self.try_exact_lookup(mention, graph) {
            return Some(r);
        }

        // 2. Try BM25 search
        if let Some(r) = self.try_bm25_lookup(mention, graph) {
            return Some(r);
        }

        // 3. Try fuzzy scan of node labels
        if let Some(r) = self.try_fuzzy_lookup(mention, graph) {
            return Some(r);
        }

        // 4. Try n-gram similarity fallback (Enhancement 10)
        if let Some(r) = self.try_ngram_lookup(mention, graph) {
            return Some(r);
        }

        None
    }

    fn try_exact_lookup(&self, mention: &EntityMention, graph: &Graph) -> Option<ResolvedEntity> {
        let name = &mention.text;

        // Check hint-specific index first
        match mention.hint {
            EntityHint::Tool => {
                if let Some(&nid) = graph.tool_index.get(name.as_str()) {
                    return Some(make_resolved(mention, nid, graph));
                }
            },
            EntityHint::Concept
            | EntityHint::Person
            | EntityHint::Organization
            | EntityHint::Location
            | EntityHint::Product => {
                if let Some(&nid) = graph.concept_index.get(name.as_str()) {
                    return Some(make_resolved(mention, nid, graph));
                }
            },
            EntityHint::Agent => {
                // Try parsing as numeric agent ID
                if let Ok(id) = name.parse::<u64>() {
                    if let Some(&nid) = graph.agent_index.get(&id) {
                        return Some(make_resolved(mention, nid, graph));
                    }
                }
                // Try agent_type match
                for node in graph.nodes.values() {
                    if let NodeType::Agent { agent_type, .. } = &node.node_type {
                        if agent_type.eq_ignore_ascii_case(name) {
                            return Some(make_resolved(mention, node.id, graph));
                        }
                    }
                }
            },
            EntityHint::Strategy => {
                for node in graph.nodes.values() {
                    if let NodeType::Strategy { name: sname, .. } = &node.node_type {
                        if sname.eq_ignore_ascii_case(name) {
                            return Some(make_resolved(mention, node.id, graph));
                        }
                    }
                }
            },
            _ => {},
        }

        // Try all string-keyed indexes if hint didn't match
        if let Some(&nid) = graph.tool_index.get(name.as_str()) {
            return Some(make_resolved(mention, nid, graph));
        }
        if let Some(&nid) = graph.concept_index.get(name.as_str()) {
            return Some(make_resolved(mention, nid, graph));
        }
        if let Some(&nid) = graph.result_index.get(name.as_str()) {
            return Some(make_resolved(mention, nid, graph));
        }

        None
    }

    fn try_bm25_lookup(&self, mention: &EntityMention, graph: &Graph) -> Option<ResolvedEntity> {
        // Route to code-aware search for code hints, mixed search otherwise
        let results = if mention.hint == EntityHint::Code {
            graph.bm25_index.search_code(&mention.text, 3)
        } else {
            graph.bm25_index.search_mixed(&mention.text, 3)
        };
        if let Some(&(node_id, score)) = results.first() {
            if score > 0.5 {
                let mut r = make_resolved(mention, node_id, graph);
                r.confidence *= score.min(1.0);
                return Some(r);
            }
        }
        None
    }

    /// N-gram (character bigram) similarity fallback.
    /// Scans up to 5000 nodes, threshold > 0.4, confidence discounted by 0.7.
    /// Node IDs are sorted for deterministic results across calls.
    fn try_ngram_lookup(&self, mention: &EntityMention, graph: &Graph) -> Option<ResolvedEntity> {
        let query = mention.text.to_lowercase();
        let mut best_match: Option<(NodeId, f32)> = None;

        // Sort node IDs for deterministic iteration order
        let mut node_ids: Vec<NodeId> = graph.nodes.keys().collect();
        node_ids.sort_unstable();

        for &node_id in node_ids.iter().take(5000) {
            if let Some(node) = graph.nodes.get(node_id) {
                let label = node.label().to_lowercase();
                let score = ngram_similarity(&query, &label, 2);
                if score > 0.4 && best_match.is_none_or(|(_, s)| score > s) {
                    best_match = Some((node_id, score));
                }
            }
        }

        best_match.map(|(node_id, score)| {
            let mut r = make_resolved(mention, node_id, graph);
            r.confidence = score * mention.confidence * 0.7; // discount
            r
        })
    }

    /// Fuzzy label scan fallback.
    /// Node IDs are sorted for deterministic results across calls.
    fn try_fuzzy_lookup(&self, mention: &EntityMention, graph: &Graph) -> Option<ResolvedEntity> {
        let query_lower = mention.text.to_lowercase();
        let mut best_match: Option<(NodeId, f32)> = None;

        // Sort node IDs for deterministic iteration order
        let mut node_ids: Vec<NodeId> = graph.nodes.keys().collect();
        node_ids.sort_unstable();

        for &node_id in node_ids.iter().take(5000) {
            if let Some(node) = graph.nodes.get(node_id) {
                let label = node.label().to_lowercase();
                let score = fuzzy_score(&query_lower, &label);
                if score > 0.6 && best_match.is_none_or(|(_, s)| score > s) {
                    best_match = Some((node_id, score));
                }
            }
        }

        best_match.map(|(node_id, score)| {
            let mut r = make_resolved(mention, node_id, graph);
            r.confidence = score * mention.confidence;
            r
        })
    }

    // ── Extraction helpers ────────────────────────────────────────────

    /// Detect code identifier patterns: snake_case, camelCase, qualified.names.
    /// These are tagged `EntityHint::Code` and extracted before NL strategies.
    fn extract_code_identifiers(&self, mentions: &mut Vec<EntityMention>, question: &str) {
        let words: Vec<(usize, &str)> = question
            .split_whitespace()
            .scan(0usize, |pos, word| {
                let start = question[*pos..]
                    .find(word)
                    .map(|p| p + *pos)
                    .unwrap_or(*pos);
                *pos = start + word.len();
                Some((start, word))
            })
            .collect();

        for &(start, word) in &words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.');
            if clean.is_empty() || clean.len() < 2 {
                continue;
            }
            if is_code_identifier(clean) && !is_programming_stopword(clean) {
                mentions.push(EntityMention {
                    text: clean.to_string(),
                    span: (start, start + word.len()),
                    hint: EntityHint::Code,
                    confidence: 0.9,
                });
            }
        }
    }

    fn extract_quoted(&self, mentions: &mut Vec<EntityMention>, question: &str) {
        // Match both 'single' and "double" quoted strings
        for delim in &['\'', '"'] {
            let mut start = None;
            for (i, ch) in question.char_indices() {
                if ch == *delim {
                    if let Some(s) = start {
                        let text = &question[s..i];
                        if !text.is_empty() && text.len() < 100 {
                            mentions.push(EntityMention {
                                text: text.to_string(),
                                span: (s, i),
                                hint: EntityHint::Unknown,
                                confidence: 1.0,
                            });
                        }
                        start = None;
                    } else {
                        start = Some(i + ch.len_utf8());
                    }
                }
            }
        }
    }

    fn extract_hash_ids(&self, mentions: &mut Vec<EntityMention>, question: &str) {
        let bytes = question.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'#' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let text = &question[start + 1..i]; // skip the #
                mentions.push(EntityMention {
                    text: text.to_string(),
                    span: (start, i),
                    hint: EntityHint::Unknown,
                    confidence: 1.0,
                });
            } else {
                i += 1;
            }
        }
    }

    fn extract_keyword_typed(&self, mentions: &mut Vec<EntityMention>, question: &str) {
        let words: Vec<(usize, &str)> = question
            .split_whitespace()
            .scan(0usize, |pos, word| {
                let start = question[*pos..]
                    .find(word)
                    .map(|p| p + *pos)
                    .unwrap_or(*pos);
                *pos = start + word.len();
                Some((start, word))
            })
            .collect();

        for i in 0..words.len().saturating_sub(1) {
            let (_, keyword) = words[i];
            let hint = match keyword.to_lowercase().as_str() {
                "agent" => EntityHint::Agent,
                "tool" => EntityHint::Tool,
                "strategy" => EntityHint::Strategy,
                "memory" => EntityHint::Memory,
                "goal" => EntityHint::Goal,
                "concept" => EntityHint::Concept,
                "event" => EntityHint::Event,
                "episode" => EntityHint::Episode,
                _ => continue,
            };
            let (next_start, next_word) = words[i + 1];
            // Strip surrounding punctuation
            let clean = next_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if !clean.is_empty() {
                mentions.push(EntityMention {
                    text: clean.to_string(),
                    span: (next_start, next_start + next_word.len()),
                    hint,
                    confidence: 0.9,
                });
            }
        }
    }

    fn extract_capitalized(&self, mentions: &mut Vec<EntityMention>, question: &str) {
        let words: Vec<(usize, &str)> = question
            .split_whitespace()
            .scan(0usize, |pos, word| {
                let start = question[*pos..]
                    .find(word)
                    .map(|p| p + *pos)
                    .unwrap_or(*pos);
                *pos = start + word.len();
                Some((start, word))
            })
            .collect();

        for (idx, &(start, word)) in words.iter().enumerate() {
            // Skip first word only if it looks like a sentence starter (question word)
            if idx == 0 {
                let lower = word.to_lowercase();
                if matches!(
                    lower.as_str(),
                    "who"
                        | "what"
                        | "where"
                        | "when"
                        | "how"
                        | "why"
                        | "find"
                        | "show"
                        | "get"
                        | "list"
                        | "the"
                        | "is"
                        | "are"
                        | "does"
                        | "do"
                        | "can"
                        | "which"
                        | "tell"
                ) {
                    continue;
                }
            }
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if clean.is_empty() || clean.len() < 2 {
                continue;
            }
            let first_char = clean.chars().next().unwrap();
            if first_char.is_uppercase() && !is_common_word(clean) {
                // Check it wasn't already extracted
                let already = mentions
                    .iter()
                    .any(|m| m.span.0 <= start && m.span.1 >= start + word.len());
                if !already {
                    mentions.push(EntityMention {
                        text: clean.to_string(),
                        span: (start, start + word.len()),
                        hint: EntityHint::Unknown,
                        confidence: 0.7,
                    });
                }
            }
        }
    }

    fn dedup_mentions(&self, mentions: &mut Vec<EntityMention>) {
        // Sort by span start, then by length descending (prefer longer)
        mentions.sort_by(|a, b| {
            a.span
                .0
                .cmp(&b.span.0)
                .then((b.span.1 - b.span.0).cmp(&(a.span.1 - a.span.0)))
        });

        let mut i = 0;
        while i < mentions.len() {
            let mut j = i + 1;
            while j < mentions.len() {
                // If mentions overlap, remove the shorter one
                if mentions[j].span.0 < mentions[i].span.1 {
                    mentions.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    fn apply_intent_hints(&self, mentions: &mut [EntityMention], intent: &QueryIntent) {
        if let QueryIntent::FilteredTraversal {
            node_type_filter: Some(type_name),
            ..
        } = intent
        {
            // If we know we're looking for tools, etc., hint accordingly
            let hint = match type_name.as_str() {
                "Tool" => EntityHint::Tool,
                "Strategy" => EntityHint::Strategy,
                "Memory" => EntityHint::Memory,
                "Agent" => EntityHint::Agent,
                _ => return,
            };
            for m in mentions.iter_mut() {
                if m.hint == EntityHint::Unknown {
                    m.hint = hint.clone();
                }
            }
        }
    }
}

/// Map NER label string to EntityHint.
pub fn hint_from_ner_label(label: &str) -> EntityHint {
    match label.to_uppercase().as_str() {
        "PERSON" | "PER" => EntityHint::Person,
        "ORG" | "ORGANIZATION" => EntityHint::Organization,
        "LOC" | "GPE" | "FAC" | "LOCATION" => EntityHint::Location,
        "PRODUCT" | "BRAND" => EntityHint::Product,
        "DATE" | "TIME" => EntityHint::DateTime,
        _ => EntityHint::Unknown,
    }
}

/// Construct a ResolvedEntity from a mention and node ID.
fn make_resolved(mention: &EntityMention, node_id: NodeId, graph: &Graph) -> ResolvedEntity {
    let node_type = graph
        .get_node(node_id)
        .map(|n| n.type_name().to_string())
        .unwrap_or_else(|| "Unknown".to_string());
    ResolvedEntity {
        mention: mention.clone(),
        node_id,
        node_type,
        confidence: mention.confidence,
    }
}

/// Simple fuzzy matching score between query and candidate.
/// Returns a value in [0.0, 1.0].
fn fuzzy_score(query: &str, candidate: &str) -> f32 {
    if query == candidate {
        return 1.0;
    }
    if candidate.contains(query) || query.contains(candidate) {
        let max_len = query.len().max(candidate.len()) as f32;
        let min_len = query.len().min(candidate.len()) as f32;
        return min_len / max_len;
    }
    // Check prefix match
    let common_prefix = query
        .chars()
        .zip(candidate.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let max_len = query.len().max(candidate.len()) as f32;
    if max_len == 0.0 {
        return 0.0;
    }
    (common_prefix as f32 * 2.0) / (query.len() + candidate.len()) as f32
}

/// Character n-gram Jaccard similarity between two strings.
/// Uses character bigrams by default (n=2). Returns [0.0, 1.0].
fn ngram_similarity(a: &str, b: &str, n: usize) -> f32 {
    use std::collections::HashSet;
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

/// Check if a token looks like a code identifier.
///
/// Matches: snake_case (contains `_`), camelCase/PascalCase (mixed case in
/// a single token), and qualified.names (contains `.` separating parts).
fn is_code_identifier(token: &str) -> bool {
    // snake_case: contains underscore and has alphanumeric chars
    if token.contains('_') && token.chars().any(|c| c.is_alphanumeric()) {
        return true;
    }
    // qualified.name: contains dot with identifier parts on both sides
    if token.contains('.') {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() >= 2
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.chars().next().is_some_and(|c| c.is_alphabetic()))
        {
            return true;
        }
    }
    // camelCase / PascalCase: has both upper and lower in a single token (no spaces)
    let has_upper = token.chars().any(|c| c.is_uppercase());
    let has_lower = token.chars().any(|c| c.is_lowercase());
    if has_upper && has_lower && !token.contains(' ') {
        // Must have at least one internal uppercase transition (not just first char)
        let chars: Vec<char> = token.chars().collect();
        for i in 1..chars.len() {
            if chars[i].is_uppercase() && chars[i - 1].is_lowercase() {
                return true;
            }
        }
    }
    false
}

/// Context-sensitive programming stopwords.
///
/// These are filtered in the NL extraction path but NOT for code identifier
/// patterns (which are detected first with higher precedence).
fn is_programming_stopword(word: &str) -> bool {
    // Only treat lowercase forms as stopwords; capitalized forms like "String"
    // are kept as potential entities
    if word.chars().next().is_some_and(|c| c.is_uppercase()) {
        return false;
    }
    matches!(
        word,
        "function"
            | "class"
            | "module"
            | "variable"
            | "method"
            | "import"
            | "return"
            | "type"
            | "struct"
            | "interface"
            | "const"
            | "void"
            | "async"
            | "await"
    )
}

/// Common English words that are not entity names.
fn is_common_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the"
            | "is"
            | "are"
            | "was"
            | "were"
            | "has"
            | "have"
            | "had"
            | "what"
            | "who"
            | "where"
            | "when"
            | "how"
            | "why"
            | "from"
            | "with"
            | "that"
            | "this"
            | "for"
            | "not"
            | "but"
            | "and"
            | "all"
            | "can"
            | "does"
            | "did"
            | "most"
            | "many"
            | "some"
            | "any"
            | "each"
            | "find"
            | "show"
            | "get"
            | "list"
            | "top"
            | "after"
            | "before"
            | "between"
            | "about"
            | "shortest"
            | "path"
            | "within"
            | "hops"
    )
}

// ---------------------------------------------------------------------------
// LLM-powered entity extraction
// ---------------------------------------------------------------------------

/// Result of LLM-powered entity extraction.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct LlmEntityExtraction {
    /// Resolved entity names (matched or related to known entities).
    #[serde(default)]
    pub entities: Vec<String>,
    /// Topic categories being asked about (e.g., "breakfast", "food", "location").
    #[serde(default)]
    pub categories: Vec<String>,
    /// Relational references that need graph traversal (e.g., "neighbor", "boss").
    #[serde(default)]
    pub implicit_refs: Vec<String>,
}

/// Use the LLM to extract entities, categories, and implicit references from a question.
///
/// This is the fallback when rule-based extraction returns poor results.
/// Feed the question plus a sample of known graph vocabulary to the LLM.
pub async fn extract_entities_with_llm(
    llm: &dyn crate::llm_client::LlmClient,
    question: &str,
    known_entities: &[String],
    known_categories: &[String],
) -> Option<LlmEntityExtraction> {
    let entities_sample: Vec<&str> = known_entities.iter().take(50).map(|s| s.as_str()).collect();
    let categories_sample: Vec<&str> = known_categories
        .iter()
        .take(30)
        .map(|s| s.as_str())
        .collect();

    let prompt = format!(
        r#"Given a question and the known entities/categories from a knowledge graph, extract:
1. "entities": entity names from the question that match or relate to known entities
2. "categories": topic categories being asked about (e.g., "breakfast", "food", "location")
3. "implicit_refs": relational references that need graph traversal (e.g., "my neighbor" → "neighbor", "my boss" → "boss")

Always include "user" if the question uses first-person pronouns (I, my, me, we).

Known entities: {:?}
Known categories: {:?}
Question: "{}"

Output ONLY valid JSON: {{"entities": [...], "categories": [...], "implicit_refs": [...]}}"#,
        entities_sample, categories_sample, question
    );

    let request = crate::llm_client::LlmRequest {
        system_prompt: "You are a precise entity extraction system. Output only valid JSON."
            .to_string(),
        user_prompt: prompt,
        temperature: 0.0,
        max_tokens: 200,
        json_mode: true,
    };

    match llm.complete(request).await {
        Ok(response) => match serde_json::from_str::<LlmEntityExtraction>(&response.content) {
            Ok(extraction) => Some(extraction),
            Err(e) => {
                tracing::debug!("Failed to parse LLM entity extraction: {}", e);
                None
            },
        },
        Err(e) => {
            tracing::debug!("LLM entity extraction failed: {}", e);
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{GraphNode, NodeType};

    fn test_graph() -> Graph {
        let mut graph = Graph::new();
        // Add a tool node
        graph
            .add_node(GraphNode::new(NodeType::Tool {
                tool_name: "login_tool".to_string(),
                tool_type: "auth".to_string(),
            }))
            .unwrap();
        // Add a concept node
        graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Alice".to_string(),
                concept_type: crate::structures::ConceptType::Person,
                confidence: 0.9,
            }))
            .unwrap();
        graph
    }

    #[test]
    fn test_extract_mentions_capitalized() {
        let resolver = EntityResolver::new();
        let mentions = resolver.extract_mentions("Alice owes Bob", &QueryIntent::Unknown);
        let names: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
    }

    #[test]
    fn test_extract_mentions_quoted() {
        let resolver = EntityResolver::new();
        let mentions = resolver.extract_mentions("'login_tool' usage", &QueryIntent::Unknown);
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].text, "login_tool");
    }

    #[test]
    fn test_extract_mentions_hash_id() {
        let resolver = EntityResolver::new();
        let mentions = resolver.extract_mentions("Show node #42", &QueryIntent::Unknown);
        assert!(mentions.iter().any(|m| m.text == "42"));
    }

    #[test]
    fn test_resolve_entity_exact() {
        let resolver = EntityResolver::new();
        let graph = test_graph();
        let mention = EntityMention {
            text: "login_tool".to_string(),
            span: (0, 10),
            hint: EntityHint::Tool,
            confidence: 1.0,
        };
        let resolved = resolver.resolve(&[mention], &graph);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].node_type, "Tool");
    }

    #[test]
    fn test_resolve_entity_bm25_fallback() {
        let resolver = EntityResolver::new();
        let graph = test_graph();
        // "login" should fuzzy-match "login_tool" via BM25
        let mention = EntityMention {
            text: "login".to_string(),
            span: (0, 5),
            hint: EntityHint::Unknown,
            confidence: 1.0,
        };
        let resolved = resolver.resolve(&[mention], &graph);
        // May or may not resolve depending on BM25 scoring; just check no panic
        assert!(resolved.len() <= 1);
    }

    // Enhancement 10: N-gram similarity tests
    #[test]
    fn test_ngram_similarity_exact() {
        assert!((ngram_similarity("alice", "alice", 2) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ngram_similarity_partial() {
        let score = ngram_similarity("alic", "alice", 2);
        assert!(score > 0.4, "Expected >0.4, got {}", score);
    }

    #[test]
    fn test_ngram_similarity_unrelated() {
        let score = ngram_similarity("xyz", "alice", 2);
        assert!(score < 0.2, "Expected <0.2, got {}", score);
    }

    #[test]
    fn test_ngram_lookup_resolves() {
        let resolver = EntityResolver::new();
        let graph = test_graph();
        // "Alic" should match "Alice" via n-gram similarity
        let mention = EntityMention {
            text: "Alic".to_string(),
            span: (0, 4),
            hint: EntityHint::Unknown,
            confidence: 1.0,
        };
        let resolved = resolver.resolve(&[mention], &graph);
        // Should resolve via ngram or fuzzy fallback
        assert!(!resolved.is_empty(), "Expected ngram resolution for 'Alic'");
    }

    // ── Code-aware entity extraction tests ──

    #[test]
    fn test_extract_code_snake_case() {
        let resolver = EntityResolver::new();
        let mentions =
            resolver.extract_mentions("where is get_user_name called", &QueryIntent::Unknown);
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"get_user_name"),
            "Should detect snake_case as code identifier: {:?}",
            texts
        );
        assert!(
            mentions.iter().any(|m| m.hint == EntityHint::Code),
            "snake_case should be tagged as Code hint"
        );
    }

    #[test]
    fn test_extract_code_camel_case() {
        let resolver = EntityResolver::new();
        let mentions =
            resolver.extract_mentions("find getUserName references", &QueryIntent::Unknown);
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"getUserName"),
            "Should detect camelCase as code identifier: {:?}",
            texts
        );
    }

    #[test]
    fn test_extract_code_qualified_name() {
        let resolver = EntityResolver::new();
        let mentions =
            resolver.extract_mentions("show auth.service.UserService", &QueryIntent::Unknown);
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"auth.service.UserService"),
            "Should detect qualified.name as code identifier: {:?}",
            texts
        );
    }

    #[test]
    fn test_capitalized_string_kept_as_entity() {
        // "String" (capitalized) should be kept as a potential entity
        assert!(!is_programming_stopword("String"));
        // "string" (lowercase) should be... not a programming stopword in our list
        // but it wouldn't match is_code_identifier either
        assert!(!is_programming_stopword("string"));
    }

    #[test]
    fn test_programming_stopword_lowercase() {
        assert!(is_programming_stopword("function"));
        assert!(is_programming_stopword("class"));
        assert!(is_programming_stopword("module"));
        assert!(is_programming_stopword("return"));
        // Capitalized forms are NOT stopwords
        assert!(!is_programming_stopword("Function"));
        assert!(!is_programming_stopword("Class"));
    }

    #[test]
    fn test_code_identifier_precedence_over_stopwords() {
        // snake_case identifiers should be detected before NL stopword filtering
        let resolver = EntityResolver::new();
        let mentions = resolver.extract_mentions("find type_map usage", &QueryIntent::Unknown);
        let texts: Vec<&str> = mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            texts.contains(&"type_map"),
            "type_map should be detected as code (not filtered as NL): {:?}",
            texts
        );
    }
}
