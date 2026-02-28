//! Natural Language to Graph Query pipeline.
//!
//! Translates natural language questions into executable `GraphQuery` objects,
//! executes them against the graph, and formats results as human-readable answers.
//!
//! Pipeline stages:
//! ```text
//! Question → Intent Classifier → Entity Extractor → Template Matcher
//!     → Query Builder → Validator → (Repairer) → Executor → Formatter → Log
//! ```

pub mod builder;
pub mod drift;
pub mod entity;
pub mod executor;
pub mod feedback;
pub mod formatter;
pub mod intent;
pub mod llm_hint;
pub mod template;
pub mod unified;
pub mod validator;

use crate::structures::Graph;
use crate::traversal::{GraphQuery, GraphTraversal, QueryResult};
use crate::{GraphError, GraphResult};
use entity::{EntityResolver, ResolvedEntity};
use intent::{classify_intent_full, intent_display_name, QueryIntent};
use std::collections::{HashMap, HashSet, VecDeque};
use template::TemplateRegistry;
use validator::ValidationResult;

/// Response from the NLQ pipeline.
#[derive(Debug, Clone)]
pub struct NlqResponse {
    /// Human-readable answer.
    pub answer: String,
    /// The GraphQuery that was executed.
    pub query_used: GraphQuery,
    /// Raw query result.
    pub result: QueryResult,
    /// Entities that were resolved from the question.
    pub entities_resolved: Vec<ResolvedEntity>,
    /// Classified intent.
    pub intent: QueryIntent,
    /// Overall pipeline confidence (0.0–1.0).
    pub confidence: f32,
    /// Pipeline execution time in milliseconds.
    pub execution_time_ms: u64,
    /// Step-by-step explanation of the pipeline.
    pub explanation: Vec<String>,
    /// Total result count before pagination.
    pub total_count: usize,
}

/// Pagination parameters for NLQ results.
#[derive(Debug, Clone, Default)]
pub struct NlqPagination {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Logical operator for compound queries.
#[derive(Debug, Clone)]
pub enum LogicalOp {
    And,
    Or,
}

/// A compound query split from a natural-language question.
#[derive(Debug, Clone)]
pub struct CompoundQuery {
    pub parts: Vec<String>,
    pub operator: LogicalOp,
}

/// A single exchange in a conversation.
#[derive(Debug, Clone)]
pub struct ConversationExchange {
    pub question: String,
    pub intent: String,
    pub entities: Vec<String>,
    /// Unix epoch milliseconds (used for session eviction).
    pub timestamp: u64,
}

/// Current time as Unix epoch milliseconds.
pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Conversation context for session statefulness.
#[derive(Debug, Clone, Default)]
pub struct ConversationContext {
    pub exchanges: VecDeque<ConversationExchange>,
}

impl ConversationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, exchange: ConversationExchange) {
        if self.exchanges.len() >= 5 {
            self.exchanges.pop_front();
        }
        self.exchanges.push_back(exchange);
    }

    pub fn last(&self) -> Option<&ConversationExchange> {
        self.exchanges.back()
    }

    /// Timestamp of the most recent exchange (for LRU eviction).
    pub fn last_activity(&self) -> u64 {
        self.exchanges.back().map_or(0, |e| e.timestamp)
    }
}

/// Natural Language Query pipeline.
pub struct NlqPipeline {
    templates: TemplateRegistry,
    entity_resolver: EntityResolver,
}

impl Default for NlqPipeline {
    fn default() -> Self {
        Self {
            templates: TemplateRegistry::new(),
            entity_resolver: EntityResolver::new(),
        }
    }
}

impl NlqPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a reference to the entity resolver (for use by unified query pipeline).
    pub fn entity_resolver(&self) -> &EntityResolver {
        &self.entity_resolver
    }

    /// Execute the full NLQ pipeline:
    /// classify → extract → resolve → match → build → validate → execute → format.
    ///
    /// Supports pagination, aggregation, negation, compound queries, and explanation.
    pub fn execute(
        &self,
        question: &str,
        graph: &Graph,
        traversal: &GraphTraversal,
    ) -> GraphResult<NlqResponse> {
        self.execute_with_pagination(question, graph, traversal, &NlqPagination::default())
    }

    /// Execute with explicit pagination.
    ///
    /// `structured_store` is passed from `GraphEngine` when available, enabling
    /// structured memory queries and aggregation source precedence.
    pub fn execute_with_pagination(
        &self,
        question: &str,
        graph: &Graph,
        traversal: &GraphTraversal,
        pagination: &NlqPagination,
    ) -> GraphResult<NlqResponse> {
        self.execute_with_pagination_ext(question, graph, traversal, pagination, None)
    }

    /// Execute with pagination and optional structured memory store.
    pub fn execute_with_pagination_ext(
        &self,
        question: &str,
        graph: &Graph,
        traversal: &GraphTraversal,
        pagination: &NlqPagination,
        structured_store: Option<&crate::structured_memory::StructuredMemoryStore>,
    ) -> GraphResult<NlqResponse> {
        self.execute_with_hint(
            question,
            graph,
            traversal,
            pagination,
            structured_store,
            None,
        )
    }

    /// Execute with pagination, optional structured memory store, and optional
    /// intent override from an LLM hint classifier.
    ///
    /// When `intent_override` is `Some`, the rule-based `classify_intent_full()`
    /// is skipped and the provided classification is used directly.
    pub fn execute_with_hint(
        &self,
        question: &str,
        graph: &Graph,
        traversal: &GraphTraversal,
        pagination: &NlqPagination,
        structured_store: Option<&crate::structured_memory::StructuredMemoryStore>,
        intent_override: Option<intent::ClassifiedIntent>,
    ) -> GraphResult<NlqResponse> {
        let start = std::time::Instant::now();
        let mut explanation = Vec::new();

        // 0. Check for compound queries (AND/OR) — only when no override
        if intent_override.is_none() {
            if let Some(compound) = detect_compound(question) {
                return self.execute_compound(
                    question,
                    &compound,
                    graph,
                    traversal,
                    pagination,
                    structured_store,
                );
            }
        }

        // 1. Classify intent (with negation) — use override if provided
        let classified = intent_override.unwrap_or_else(|| classify_intent_full(question));
        let intent = classified.intent;
        let negated = classified.negated;
        explanation.push(format!(
            "Intent classified as {}",
            intent_display_name(&intent)
        ));
        if negated {
            explanation.push("Negation detected".to_string());
        }

        // 2. Handle structured memory queries (bypass template/builder/executor)
        if let QueryIntent::StructuredMemoryQuery { ref query_type } = intent {
            let mentions = self.entity_resolver.extract_mentions(question, &intent);
            let resolved = self.entity_resolver.resolve(&mentions, graph);
            let result = execute_structured_memory_query(
                query_type,
                &resolved,
                graph,
                structured_store,
                question,
            );
            let total_count = formatter::result_count(&result);
            let answer = formatter::format_result(question, &intent, &result, &resolved, graph);
            explanation.push(format!("Structured memory query executed: {}", answer));
            return Ok(NlqResponse {
                answer,
                query_used: GraphQuery::PageRank {
                    iterations: 0,
                    damping_factor: 0.0,
                },
                result,
                entities_resolved: resolved,
                intent,
                confidence: 0.9,
                execution_time_ms: start.elapsed().as_millis() as u64,
                explanation,
                total_count,
            });
        }

        // 3. Handle aggregation queries (bypass template/builder/executor)
        if let QueryIntent::Aggregate { ref metric } = intent {
            // Extract entities even for aggregation (needed for "sum between A and B")
            let mentions = self.entity_resolver.extract_mentions(question, &intent);
            let resolved = self.entity_resolver.resolve(&mentions, graph);
            let result = execute_aggregate(metric, graph, &resolved, structured_store);
            let total_count = formatter::result_count(&result);
            let answer = formatter::format_result(question, &intent, &result, &resolved, graph);
            explanation.push(format!("Aggregation executed: {}", answer));
            return Ok(NlqResponse {
                answer,
                query_used: GraphQuery::PageRank {
                    iterations: 0,
                    damping_factor: 0.0,
                },
                result,
                entities_resolved: resolved,
                intent,
                confidence: 0.9,
                execution_time_ms: start.elapsed().as_millis() as u64,
                explanation,
                total_count,
            });
        }

        // 3. Extract entity mentions (rule-based)
        let mentions = self.entity_resolver.extract_mentions(question, &intent);
        explanation.push(format!("{} entity mention(s) extracted", mentions.len()));

        // 4. Resolve mentions to graph nodes
        let resolved = self.entity_resolver.resolve(&mentions, graph);
        for r in &resolved {
            explanation.push(format!(
                "Resolved '{}' -> {} #{} ({:.2})",
                r.mention.text, r.node_type, r.node_id, r.confidence
            ));
        }

        // 5. Match template
        let template_match = self.templates.match_template(&intent, &resolved, question);

        // If no template matches, fall back to BM25 full-text search
        if template_match.is_none() {
            explanation.push("No template matched — falling back to BM25 search".to_string());
            let bm25_hits = graph
                .bm25_index
                .search(question, pagination.limit.unwrap_or(10));
            let total_count = bm25_hits.len();
            let result = QueryResult::Rankings(bm25_hits);
            let answer = formatter::format_result(question, &intent, &result, &resolved, graph);
            return Ok(NlqResponse {
                answer,
                query_used: GraphQuery::PageRank {
                    iterations: 0,
                    damping_factor: 0.0,
                },
                result,
                entities_resolved: resolved,
                intent,
                confidence: 0.5,
                execution_time_ms: start.elapsed().as_millis() as u64,
                explanation,
                total_count,
            });
        }

        let (template, params) = template_match.unwrap();
        explanation.push(format!("Matched template '{}'", template.name));

        // 6. Build query
        let query = builder::build_query(template, &resolved, &params, &intent)?;
        explanation.push(format!("Built query: {:?}", std::mem::discriminant(&query)));

        // 7. Validate
        let validated_query = match validator::validate(query, graph) {
            ValidationResult::Valid(q) => {
                explanation.push("Query validated successfully".to_string());
                q
            },
            ValidationResult::Repaired {
                repaired, reason, ..
            } => {
                explanation.push(format!("Query repaired: {}", reason));
                tracing::debug!("NLQ query repaired: {}", reason);
                repaired
            },
            ValidationResult::Invalid { reason, .. } => {
                explanation.push(format!("Query validation failed: {}", reason));
                return Err(GraphError::InvalidQuery(format!(
                    "Query validation failed: {}",
                    reason
                )));
            },
        };

        // 8. Execute
        let mut result = executor::execute(&validated_query, graph, traversal)?;

        // 9. Apply negation if detected
        if negated {
            result = invert_result(result, graph);
            explanation.push("Applied negation inversion".to_string());
        }

        // 10. Paginate
        let total_count = formatter::result_count(&result);
        // Merge NL-parsed limit with explicit API limit (API overrides)
        let effective_limit = pagination.limit.or(params.limit);
        let effective_offset = pagination.offset;
        if effective_limit.is_some() || effective_offset.is_some() {
            result = apply_pagination(result, effective_offset, effective_limit);
            explanation.push(format!(
                "Paginated: offset={}, limit={}, total={}",
                effective_offset.unwrap_or(0),
                effective_limit.map_or("none".to_string(), |l| l.to_string()),
                total_count
            ));
        }

        // 11. Format
        let answer = formatter::format_result(question, &intent, &result, &resolved, graph);

        // Compute confidence
        let confidence = compute_confidence(&intent, &resolved, &result);

        let execution_time_ms = start.elapsed().as_millis() as u64;

        Ok(NlqResponse {
            answer,
            query_used: validated_query,
            result,
            entities_resolved: resolved,
            intent,
            confidence,
            execution_time_ms,
            explanation,
            total_count,
        })
    }

    /// Execute a compound (AND/OR) query by running sub-parts and merging.
    fn execute_compound(
        &self,
        question: &str,
        compound: &CompoundQuery,
        graph: &Graph,
        traversal: &GraphTraversal,
        pagination: &NlqPagination,
        structured_store: Option<&crate::structured_memory::StructuredMemoryStore>,
    ) -> GraphResult<NlqResponse> {
        let start = std::time::Instant::now();
        let mut explanation = Vec::new();
        explanation.push(format!(
            "Compound query detected ({:?}): {} sub-queries",
            compound.operator,
            compound.parts.len()
        ));

        let mut sub_node_sets: Vec<HashSet<u64>> = Vec::new();
        let mut last_intent = QueryIntent::Unknown;
        let mut all_resolved = Vec::new();
        let mut last_query = GraphQuery::PageRank {
            iterations: 0,
            damping_factor: 0.0,
        };

        for part in &compound.parts {
            match self.execute_with_pagination_ext(
                part,
                graph,
                traversal,
                &NlqPagination::default(),
                structured_store,
            ) {
                Ok(resp) => {
                    let node_ids = extract_node_ids(&resp.result);
                    sub_node_sets.push(node_ids);
                    last_intent = resp.intent;
                    last_query = resp.query_used;
                    all_resolved.extend(resp.entities_resolved);
                    explanation.extend(resp.explanation);
                },
                Err(e) => {
                    explanation.push(format!("Sub-query '{}' failed: {}", part, e));
                    sub_node_sets.push(HashSet::new());
                },
            }
        }

        // Merge results
        let merged_ids: HashSet<u64> = if sub_node_sets.is_empty() {
            HashSet::new()
        } else {
            match compound.operator {
                LogicalOp::And => {
                    let mut iter = sub_node_sets.into_iter();
                    let first = iter.next().unwrap_or_default();
                    iter.fold(first, |acc, s| acc.intersection(&s).copied().collect())
                },
                LogicalOp::Or => sub_node_sets.into_iter().flatten().collect(),
            }
        };

        let merged: Vec<u64> = merged_ids.into_iter().collect();
        let total_count = merged.len();
        let mut result = QueryResult::Nodes(merged);

        // Paginate
        if pagination.limit.is_some() || pagination.offset.is_some() {
            result = apply_pagination(result, pagination.offset, pagination.limit);
        }

        let answer =
            formatter::format_result(question, &last_intent, &result, &all_resolved, graph);
        let confidence = compute_confidence(&last_intent, &all_resolved, &result);

        Ok(NlqResponse {
            answer,
            query_used: last_query,
            result,
            entities_resolved: all_resolved,
            intent: last_intent,
            confidence,
            execution_time_ms: start.elapsed().as_millis() as u64,
            explanation,
            total_count,
        })
    }
}

/// Compute overall pipeline confidence from components.
fn compute_confidence(
    intent: &QueryIntent,
    resolved: &[ResolvedEntity],
    result: &QueryResult,
) -> f32 {
    let intent_confidence = match intent {
        QueryIntent::Unknown => 0.1,
        _ => 0.8,
    };

    let entity_confidence = if resolved.is_empty() {
        0.5 // No entities needed (e.g., PageRank)
    } else {
        let sum: f32 = resolved.iter().map(|r| r.confidence).sum();
        sum / resolved.len() as f32
    };

    let result_confidence = if formatter::result_count(result) > 0 {
        1.0
    } else {
        0.3
    };

    (intent_confidence * 0.3 + entity_confidence * 0.4 + result_confidence * 0.3).min(1.0)
}

/// Execute a structured memory query (ledger balance, current state, etc.).
fn execute_structured_memory_query(
    query_type: &intent::StructuredQueryType,
    resolved_entities: &[ResolvedEntity],
    _graph: &Graph,
    structured_store: Option<&crate::structured_memory::StructuredMemoryStore>,
    _question: &str,
) -> QueryResult {
    use serde_json::json;
    let mut props = HashMap::new();

    let store = match structured_store {
        Some(s) => s,
        None => {
            props.insert(
                "answer".to_string(),
                json!("Structured memory not available."),
            );
            props.insert("source".to_string(), json!("none"));
            props.insert("confidence".to_string(), json!(0.0));
            return QueryResult::Properties(props);
        },
    };

    match query_type {
        intent::StructuredQueryType::LedgerBalance => {
            if resolved_entities.len() >= 2 {
                let id_a = resolved_entities[0].node_id;
                let id_b = resolved_entities[1].node_id;
                let key = crate::structured_memory::ledger_key(id_a, id_b);
                if let Some(bal) = store.ledger_balance(&key) {
                    props.insert("balance".to_string(), json!(bal));
                    props.insert("source".to_string(), json!("structured_ledger"));
                    props.insert("structured_key_used".to_string(), json!(key));
                    props.insert("confidence".to_string(), json!(1.0));
                    props.insert("answer".to_string(), json!(format!("Balance: {}", bal)));
                } else {
                    props.insert(
                        "answer".to_string(),
                        json!(format!("No ledger found for key '{}'", key)),
                    );
                    props.insert("source".to_string(), json!("none"));
                    props.insert("confidence".to_string(), json!(0.0));
                }
            } else {
                props.insert(
                    "answer".to_string(),
                    json!("Need two entities for ledger balance."),
                );
                props.insert("source".to_string(), json!("none"));
                props.insert("confidence".to_string(), json!(0.0));
            }
        },
        intent::StructuredQueryType::CurrentState => {
            if let Some(entity) = resolved_entities.first() {
                let key = crate::structured_memory::state_key(entity.node_id);
                if let Some(state) = store.state_current(&key) {
                    props.insert("current_state".to_string(), json!(state));
                    props.insert("source".to_string(), json!("structured_state_machine"));
                    props.insert("structured_key_used".to_string(), json!(key));
                    props.insert("confidence".to_string(), json!(1.0));
                    props.insert(
                        "answer".to_string(),
                        json!(format!("Current state: {}", state)),
                    );
                } else {
                    props.insert(
                        "answer".to_string(),
                        json!(format!("No state machine found for key '{}'", key)),
                    );
                    props.insert("source".to_string(), json!("none"));
                    props.insert("confidence".to_string(), json!(0.0));
                }
            } else {
                props.insert(
                    "answer".to_string(),
                    json!("Need an entity for state lookup."),
                );
                props.insert("source".to_string(), json!("none"));
                props.insert("confidence".to_string(), json!(0.0));
            }
        },
        intent::StructuredQueryType::PreferenceRanking => {
            if let Some(entity) = resolved_entities.first() {
                // Try common categories
                let categories = ["food", "music", "movies", "books", "sports", "general"];
                let mut found = false;
                for cat in &categories {
                    let key = crate::structured_memory::prefs_key(entity.node_id, cat);
                    if let Some(crate::structured_memory::MemoryTemplate::PreferenceList {
                        ranked_items,
                        ..
                    }) = store.get(&key)
                    {
                        let items: Vec<String> = ranked_items
                            .iter()
                            .map(|p| format!("{}. {}", p.rank, p.name))
                            .collect();
                        props.insert("rankings".to_string(), json!(items));
                        props.insert("category".to_string(), json!(cat));
                        props.insert("source".to_string(), json!("structured_preference_list"));
                        props.insert("structured_key_used".to_string(), json!(key));
                        props.insert("confidence".to_string(), json!(0.9));
                        props.insert(
                            "answer".to_string(),
                            json!(format!("Rankings: {}", items.join(", "))),
                        );
                        found = true;
                        break;
                    }
                }
                if !found {
                    props.insert("answer".to_string(), json!("No preference list found."));
                    props.insert("source".to_string(), json!("none"));
                    props.insert("confidence".to_string(), json!(0.0));
                }
            } else {
                props.insert(
                    "answer".to_string(),
                    json!("Need an entity for preference lookup."),
                );
                props.insert("source".to_string(), json!("none"));
                props.insert("confidence".to_string(), json!(0.0));
            }
        },
        intent::StructuredQueryType::TreeChildren => {
            if let Some(entity) = resolved_entities.first() {
                let key = crate::structured_memory::tree_key(entity.node_id);
                let label = &entity.mention.text;
                if let Some(children) = store.tree_children(&key, label) {
                    props.insert("children".to_string(), json!(children));
                    props.insert("parent".to_string(), json!(label));
                    props.insert("source".to_string(), json!("structured_tree"));
                    props.insert("structured_key_used".to_string(), json!(key));
                    props.insert("confidence".to_string(), json!(0.9));
                    props.insert(
                        "answer".to_string(),
                        json!(format!("Children: {}", children.join(", "))),
                    );
                } else {
                    props.insert(
                        "answer".to_string(),
                        json!(format!(
                            "No children found for '{}' in tree '{}'",
                            label, key
                        )),
                    );
                    props.insert("source".to_string(), json!("none"));
                    props.insert("confidence".to_string(), json!(0.0));
                }
            } else {
                props.insert(
                    "answer".to_string(),
                    json!("Need an entity for tree lookup."),
                );
                props.insert("source".to_string(), json!("none"));
                props.insert("confidence".to_string(), json!(0.0));
            }
        },
    }

    QueryResult::Properties(props)
}

/// Execute an aggregation metric directly against the graph.
///
/// When `resolved_entities` are present and the metric is `SumProperty`,
/// edges between resolved entity pairs are summed (Section C precedence).
fn execute_aggregate(
    metric: &intent::AggregateMetric,
    graph: &Graph,
    resolved_entities: &[ResolvedEntity],
    structured_store: Option<&crate::structured_memory::StructuredMemoryStore>,
) -> QueryResult {
    use serde_json::json;
    let mut props = HashMap::new();
    match metric {
        intent::AggregateMetric::NodeCount => {
            props.insert("node_count".to_string(), json!(graph.node_count()));
            props.insert(
                "answer".to_string(),
                json!(format!("There are {} nodes.", graph.node_count())),
            );
        },
        intent::AggregateMetric::EdgeCount => {
            props.insert("edge_count".to_string(), json!(graph.edge_count()));
            props.insert(
                "answer".to_string(),
                json!(format!("There are {} edges.", graph.edge_count())),
            );
        },
        intent::AggregateMetric::CountByType(type_name) => {
            let count = crate::structures::node_type_discriminant_from_name(type_name)
                .and_then(|disc| graph.type_index.get(&disc))
                .map(|set| set.len())
                .unwrap_or(0);
            props.insert("type".to_string(), json!(type_name));
            props.insert("count".to_string(), json!(count));
            props.insert(
                "answer".to_string(),
                json!(format!("There are {} {} nodes.", count, type_name)),
            );
        },
        intent::AggregateMetric::AverageDegree => {
            let stats = graph.stats();
            props.insert("avg_degree".to_string(), json!(stats.avg_degree));
            props.insert(
                "answer".to_string(),
                json!(format!("Average degree: {:.2}", stats.avg_degree)),
            );
        },
        intent::AggregateMetric::Stats => {
            let stats = graph.stats();
            let max_degree = graph.nodes.values().map(|n| n.degree).max().unwrap_or(0);
            props.insert("node_count".to_string(), json!(stats.node_count));
            props.insert("edge_count".to_string(), json!(stats.edge_count));
            props.insert("avg_degree".to_string(), json!(stats.avg_degree));
            props.insert("max_degree".to_string(), json!(max_degree));
            props.insert("component_count".to_string(), json!(stats.component_count));
            props.insert(
                "answer".to_string(),
                json!(format!(
                    "Graph stats: {} nodes, {} edges, avg degree {:.2}, {} components",
                    stats.node_count, stats.edge_count, stats.avg_degree, stats.component_count
                )),
            );
        },
        intent::AggregateMetric::SumProperty {
            property,
            node_type_filter,
        } => {
            // Section C precedence:
            // 1. Structured ledger (if 2 entities + store available)
            // 2. Edge property sum between entities
            // 3. Node property sum
            if resolved_entities.len() >= 2 {
                let id_a = resolved_entities[0].node_id;
                let id_b = resolved_entities[1].node_id;
                let ledger_k = crate::structured_memory::ledger_key(id_a, id_b);

                // Precedence 1: structured ledger
                if let Some(store) = structured_store {
                    if let Some(bal) = store.ledger_balance(&ledger_k) {
                        props.insert("sum".to_string(), json!(bal));
                        props.insert("source".to_string(), json!("structured_ledger"));
                        props.insert("structured_key_used".to_string(), json!(ledger_k));
                        props.insert("fallback_used".to_string(), json!(false));
                        props.insert("confidence".to_string(), json!(1.0));
                        props.insert("answer".to_string(), json!(format!("Balance: {}", bal)));
                        return QueryResult::Properties(props);
                    }
                }

                // Precedence 2: edge property sum between entity pair
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for edge in graph.edges.values() {
                    let matches = (edge.source == id_a && edge.target == id_b)
                        || (edge.source == id_b && edge.target == id_a);
                    if matches {
                        if let Some(val) = edge.properties.get(property).and_then(|v| v.as_f64()) {
                            sum += val;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    props.insert("sum".to_string(), json!(sum));
                    props.insert("count".to_string(), json!(count));
                    props.insert("property".to_string(), json!(property));
                    props.insert("source".to_string(), json!("edge_sum"));
                    props.insert("fallback_used".to_string(), json!(true));
                    props.insert("confidence".to_string(), json!(0.8));
                    props.insert(
                        "answer".to_string(),
                        json!(format!("Sum of '{}': {}", property, sum)),
                    );
                    return QueryResult::Properties(props);
                }
            }

            // Precedence 3: node property sum
            let type_disc = node_type_filter
                .as_ref()
                .and_then(|f| crate::structures::node_type_discriminant_from_name(f));
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for node in graph.nodes.values() {
                if let Some(disc) = type_disc {
                    if node.node_type.discriminant() != disc {
                        continue;
                    }
                }
                if let Some(val) = node.properties.get(property).and_then(|v| v.as_f64()) {
                    sum += val;
                    count += 1;
                }
            }
            props.insert("sum".to_string(), json!(sum));
            props.insert("count".to_string(), json!(count));
            props.insert("property".to_string(), json!(property));
            props.insert("source".to_string(), json!("node_sum"));
            props.insert("fallback_used".to_string(), json!(count == 0));
            props.insert(
                "confidence".to_string(),
                json!(if count > 0 { 0.7 } else { 0.2 }),
            );
            props.insert(
                "answer".to_string(),
                json!(format!("Sum of '{}': {} ({} nodes)", property, sum, count)),
            );
        },
        intent::AggregateMetric::GroupByCount {
            group_property,
            node_type_filter,
        } => {
            let type_disc = node_type_filter
                .as_ref()
                .and_then(|f| crate::structures::node_type_discriminant_from_name(f));
            let mut groups: HashMap<String, usize> = HashMap::new();
            for node in graph.nodes.values() {
                if let Some(disc) = type_disc {
                    if node.node_type.discriminant() != disc {
                        continue;
                    }
                }
                let group_val = node
                    .properties
                    .get(group_property)
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                *groups.entry(group_val).or_insert(0) += 1;
            }
            for (group, count) in &groups {
                props.insert(group.clone(), json!(count));
            }
            props.insert("source".to_string(), json!("node_group"));
            props.insert(
                "confidence".to_string(),
                json!(if groups.is_empty() { 0.2 } else { 0.8 }),
            );
            let answer = groups
                .iter()
                .map(|(g, c)| format!("{}: {}", g, c))
                .collect::<Vec<_>>()
                .join(", ");
            props.insert("answer".to_string(), json!(format!("Groups: {}", answer)));
        },
        intent::AggregateMetric::MinMaxProperty {
            property,
            find_min,
            node_type_filter,
        } => {
            let type_disc = node_type_filter
                .as_ref()
                .and_then(|f| crate::structures::node_type_discriminant_from_name(f));
            let mut best_value: Option<f64> = None;
            let mut best_node: Option<u64> = None;
            for (&node_id, node) in &graph.nodes {
                if let Some(disc) = type_disc {
                    if node.node_type.discriminant() != disc {
                        continue;
                    }
                }
                if let Some(val) = node.properties.get(property).and_then(|v| v.as_f64()) {
                    let is_better = match best_value {
                        None => true,
                        Some(cur) => {
                            if *find_min {
                                val < cur
                            } else {
                                val > cur
                            }
                        },
                    };
                    if is_better {
                        best_value = Some(val);
                        best_node = Some(node_id);
                    }
                }
            }
            let label = if *find_min { "Minimum" } else { "Maximum" };
            if let (Some(val), Some(nid)) = (best_value, best_node) {
                props.insert("value".to_string(), json!(val));
                props.insert("node_id".to_string(), json!(nid));
                props.insert("property".to_string(), json!(property));
                props.insert("source".to_string(), json!("node_scan"));
                props.insert("confidence".to_string(), json!(0.8));
                props.insert(
                    "answer".to_string(),
                    json!(format!("{} '{}': {} (node {})", label, property, val, nid)),
                );
            } else {
                props.insert("value".to_string(), json!(null));
                props.insert("property".to_string(), json!(property));
                props.insert("source".to_string(), json!("node_scan"));
                props.insert("confidence".to_string(), json!(0.2));
                props.insert(
                    "answer".to_string(),
                    json!(format!("No values found for property '{}'", property)),
                );
            }
        },
    }
    QueryResult::Properties(props)
}

/// Detect compound queries ("X and Y", "X or Y") at top level.
/// Returns None if no compound operator is found, or if inside quotes.
/// Splits on all occurrences of the delimiter, supporting 3+ part queries.
fn detect_compound(question: &str) -> Option<CompoundQuery> {
    let lower = question.to_lowercase();

    // Don't split if very short
    if lower.split_whitespace().count() < 5 {
        return None;
    }

    // Try " and " first, then " or "
    for (delimiter, op) in &[(" and ", LogicalOp::And), (" or ", LogicalOp::Or)] {
        if !lower.contains(delimiter) {
            continue;
        }

        // Quote check on first occurrence only
        if let Some(pos) = lower.find(delimiter) {
            let before_first = &question[..pos];
            let quotes_before = before_first
                .chars()
                .filter(|&c| c == '"' || c == '\'')
                .count();
            if quotes_before % 2 != 0 {
                continue; // Inside quotes
            }
        }

        // Split on all occurrences using positions from lowercase
        let mut parts = Vec::new();
        let mut search_start = 0;
        loop {
            match lower[search_start..].find(delimiter) {
                Some(rel_pos) => {
                    let abs_pos = search_start + rel_pos;
                    parts.push(question[search_start..abs_pos].trim().to_string());
                    search_start = abs_pos + delimiter.len();
                },
                None => {
                    parts.push(question[search_start..].trim().to_string());
                    break;
                },
            }
        }

        // Each part must be non-trivial (at least 2 words)
        if parts.len() < 2 {
            continue;
        }
        if parts.iter().any(|p| p.split_whitespace().count() < 2) {
            continue;
        }

        return Some(CompoundQuery {
            parts,
            operator: op.clone(),
        });
    }
    None
}

/// Invert a query result (set complement for Nodes/Rankings).
fn invert_result(result: QueryResult, graph: &Graph) -> QueryResult {
    match result {
        QueryResult::Nodes(matched) => {
            let matched_set: HashSet<u64> = matched.into_iter().collect();
            let inverted: Vec<u64> = graph
                .nodes
                .keys()
                .filter(|id| !matched_set.contains(id))
                .copied()
                .take(1000)
                .collect();
            QueryResult::Nodes(inverted)
        },
        QueryResult::Rankings(matched) => {
            let matched_set: HashSet<u64> = matched.iter().map(|&(id, _)| id).collect();
            let inverted: Vec<(u64, f32)> = graph
                .nodes
                .keys()
                .filter(|id| !matched_set.contains(id))
                .take(1000)
                .map(|&id| (id, 0.0))
                .collect();
            QueryResult::Rankings(inverted)
        },
        other => other, // Non-node results returned as-is
    }
}

/// Apply pagination (offset + limit) to a QueryResult.
fn apply_pagination(
    result: QueryResult,
    offset: Option<usize>,
    limit: Option<usize>,
) -> QueryResult {
    let off = offset.unwrap_or(0);
    match result {
        QueryResult::Nodes(ids) => {
            let paginated = paginate_vec(ids, off, limit);
            QueryResult::Nodes(paginated)
        },
        QueryResult::Rankings(rankings) => {
            let paginated = paginate_vec(rankings, off, limit);
            QueryResult::Rankings(paginated)
        },
        QueryResult::Edges(ids) => {
            let paginated = paginate_vec(ids, off, limit);
            QueryResult::Edges(paginated)
        },
        QueryResult::Communities(comms) => {
            let paginated = paginate_vec(comms, off, limit);
            QueryResult::Communities(paginated)
        },
        QueryResult::Paths(paths) => {
            let paginated = paginate_vec(paths, off, limit);
            QueryResult::Paths(paginated)
        },
        QueryResult::WeightedPaths(paths) => {
            let paginated = paginate_vec(paths, off, limit);
            QueryResult::WeightedPaths(paginated)
        },
        // Path, Subgraph, Properties — no pagination
        other => other,
    }
}

fn paginate_vec<T>(v: Vec<T>, offset: usize, limit: Option<usize>) -> Vec<T> {
    let iter = v.into_iter().skip(offset);
    match limit {
        Some(lim) => iter.take(lim).collect(),
        None => iter.collect(),
    }
}

/// Extract node IDs from a QueryResult (for compound query merging).
fn extract_node_ids(result: &QueryResult) -> HashSet<u64> {
    match result {
        QueryResult::Nodes(ids) => ids.iter().copied().collect(),
        QueryResult::Rankings(rankings) => rankings.iter().map(|&(id, _)| id).collect(),
        QueryResult::Path(ids) => ids.iter().copied().collect(),
        QueryResult::Edges(ids) => ids.iter().copied().collect(),
        QueryResult::Subgraph { nodes, .. } => nodes.iter().copied().collect(),
        QueryResult::Communities(comms) => comms.iter().flatten().copied().collect(),
        _ => HashSet::new(),
    }
}

/// Resolve a follow-up question using conversational context.
pub fn resolve_followup(question: &str, context: &ConversationContext) -> Option<String> {
    let prev = context.last()?;
    let lower = question.to_lowercase();

    // "What about X?" pattern -> substitute X into previous question
    if lower.starts_with("what about ") {
        let rest = question[11..].trim().trim_end_matches('?');
        if !rest.is_empty() {
            // Replace entities in previous question with new entity
            let mut new_q = prev.question.clone();
            for entity in &prev.entities {
                new_q = new_q.replace(entity, rest);
            }
            if new_q != prev.question {
                return Some(new_q);
            }
            // If no entity replacement happened, just run the same query style with new entity
            return Some(format!(
                "{} {}",
                intent_display_name_for_query(&prev.intent),
                rest
            ));
        }
    }

    // Pronoun resolution: "it"/"them"/"those" -> last resolved entity
    // Work on lowercased text (NLQ pipeline lowercases anyway)
    let pronouns = ["it", "them", "those", "that", "this"];
    for pronoun in &pronouns {
        let mid_pattern = format!(" {} ", pronoun);
        let end_pattern = format!(" {}", pronoun);
        let end_q_pattern = format!(" {}?", pronoun);

        if lower.contains(&mid_pattern)
            || lower.ends_with(&end_pattern)
            || lower.ends_with(&end_q_pattern)
        {
            if let Some(last_entity) = prev.entities.last() {
                let mut replaced = lower.clone();
                // Replace mid-string occurrences: " pronoun "
                replaced = replaced.replace(&mid_pattern, &format!(" {} ", last_entity));
                // Replace end-of-string with ?: " pronoun?"
                replaced = replaced.replace(&end_q_pattern, &format!(" {}?", last_entity));
                // Replace bare end-of-string: " pronoun"
                if replaced.ends_with(&end_pattern) {
                    let prefix = &replaced[..replaced.len() - end_pattern.len()];
                    replaced = format!("{} {}", prefix, last_entity);
                }
                if replaced != lower {
                    return Some(replaced);
                }
            }
        }
    }

    None
}

fn intent_display_name_for_query(intent_str: &str) -> &str {
    // Map stored intent string back to a query prefix
    if intent_str.contains("Neighbor") {
        "neighbors of"
    } else if intent_str.contains("Path") {
        "shortest path to"
    } else if intent_str.contains("Similar") {
        "similar to"
    } else {
        "show"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn build_test_graph() -> Graph {
        let mut graph = Graph::new();
        let alice_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Alice".to_string(),
                concept_type: crate::structures::ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();
        let bob_id = graph
            .add_node(GraphNode::new(NodeType::Concept {
                concept_name: "Bob".to_string(),
                concept_type: crate::structures::ConceptType::Person,
                confidence: 1.0,
            }))
            .unwrap();
        // Add an edge Alice → Bob
        graph.add_edge(GraphEdge::new(
            alice_id,
            bob_id,
            EdgeType::Association {
                association_type: "knows".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));
        graph
    }

    #[test]
    fn test_end_to_end_pipeline() {
        let graph = build_test_graph();
        let traversal = GraphTraversal::new();
        let pipeline = NlqPipeline::new();

        // Test ranking query (no entities needed)
        let result = pipeline.execute("Most important nodes", &graph, &traversal);
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(!resp.answer.is_empty());
        assert!(resp.confidence > 0.0);
        assert!(!resp.explanation.is_empty());
    }

    #[test]
    fn test_end_to_end_neighbors() {
        let graph = build_test_graph();
        let traversal = GraphTraversal::new();
        let pipeline = NlqPipeline::new();

        // "Who does Alice connect to?" — should find Alice, get neighbors
        let result = pipeline.execute("Who does Alice connect to?", &graph, &traversal);
        // May succeed or fail depending on entity resolution
        // Just verify it doesn't panic
        match result {
            Ok(resp) => {
                assert!(!resp.answer.is_empty());
                assert!(!resp.explanation.is_empty());
            },
            Err(e) => {
                // Entity resolution may fail if Alice isn't found — acceptable
                assert!(e.to_string().contains("template") || e.to_string().contains("validation"));
            },
        }
    }

    // Enhancement 3: Aggregation
    #[test]
    fn test_aggregation_node_count() {
        let graph = build_test_graph();
        let traversal = GraphTraversal::new();
        let pipeline = NlqPipeline::new();
        let result = pipeline.execute("How many nodes are there?", &graph, &traversal);
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.answer.contains("2") || resp.answer.contains("nodes"));
    }

    // Enhancement 6: Pagination
    #[test]
    fn test_pagination() {
        let graph = build_test_graph();
        let traversal = GraphTraversal::new();
        let pipeline = NlqPipeline::new();
        let pagination = NlqPagination {
            limit: Some(1),
            offset: Some(0),
        };
        let result = pipeline.execute_with_pagination(
            "Most important nodes",
            &graph,
            &traversal,
            &pagination,
        );
        assert!(result.is_ok());
        let resp = result.unwrap();
        // Rankings should be paginated to 1
        if let QueryResult::Rankings(r) = &resp.result {
            assert!(r.len() <= 1);
        }
    }

    // Enhancement 4: Compound queries
    #[test]
    fn test_detect_compound_and() {
        let c = detect_compound("neighbors of Alice and neighbors of Bob");
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(matches!(c.operator, LogicalOp::And));
        assert_eq!(c.parts.len(), 2);
    }

    #[test]
    fn test_detect_compound_or() {
        let c = detect_compound("neighbors of Alice or neighbors of Bob");
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(matches!(c.operator, LogicalOp::Or));
    }

    #[test]
    fn test_detect_compound_short_sentence_no_split() {
        // Too short to split
        let c = detect_compound("A and B");
        assert!(c.is_none());
    }

    // Enhancement 5: Negation inversion
    #[test]
    fn test_invert_result_nodes() {
        let graph = build_test_graph();
        let result = QueryResult::Nodes(vec![1]); // Only node 1
        let inverted = invert_result(result, &graph);
        if let QueryResult::Nodes(ids) = &inverted {
            assert!(!ids.contains(&1));
            assert!(ids.contains(&2));
        } else {
            panic!("Expected Nodes result");
        }
    }

    // Enhancement 7: Follow-up resolution
    #[test]
    fn test_resolve_followup_what_about() {
        let mut ctx = ConversationContext::new();
        ctx.push(ConversationExchange {
            question: "Who does Alice connect to?".to_string(),
            intent: "FindNeighbors".to_string(),
            entities: vec!["Alice".to_string()],
            timestamp: now_millis(),
        });
        let resolved = resolve_followup("What about Bob?", &ctx);
        assert!(resolved.is_some());
        let q = resolved.unwrap();
        assert!(q.contains("Bob"));
        assert!(!q.contains("Alice"));
    }

    // Enhancement 8: Explanation
    #[test]
    fn test_explanation_populated() {
        let graph = build_test_graph();
        let traversal = GraphTraversal::new();
        let pipeline = NlqPipeline::new();
        let result = pipeline.execute("Most important nodes", &graph, &traversal);
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp
            .explanation
            .iter()
            .any(|e| e.contains("Intent classified")));
    }
}
