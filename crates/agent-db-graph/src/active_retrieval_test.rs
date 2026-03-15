//! Active Retrieval Testing (ART) — validates that embedded nodes are actually retrievable.
//!
//! After nodes are embedded, ART generates test questions via LLM, searches the graph indexes,
//! and enhances nodes that fail to appear in search results. This closes the loop between
//! embedding quality and retrieval accuracy.
//!
//! ## Flow per node
//! ```text
//! Node added → LLM generates test questions → Embed questions → Search graph indexes
//! → If hit rate < threshold: LLM generates enhancement → Update properties + re-index + re-embed
//! ```

use crate::claims::embeddings::{EmbeddingClient, EmbeddingRequest};
use crate::inference::GraphInference;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::structures::NodeId;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Truncate a string to at most `max_bytes` bytes, ensuring the cut
/// falls on a valid UTF-8 character boundary.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Configuration for Active Retrieval Testing.
#[derive(Debug, Clone)]
pub struct ArtConfig {
    /// Whether ART is enabled. Default: false.
    pub enabled: bool,
    /// Number of test questions to generate per node. Default: 5.
    pub test_questions: usize,
    /// Top-K results to check in each index. Default: 10.
    pub top_k: usize,
    /// Minimum fraction of test questions that must find the node. Default: 0.6.
    pub min_hit_rate: f32,
    /// Maximum nodes to process per ART pass. Default: 20.
    pub max_nodes_per_pass: usize,
}

impl Default for ArtConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            test_questions: 5,
            top_k: 10,
            min_hit_rate: 0.6,
            max_nodes_per_pass: 20,
        }
    }
}

/// Result of a single ART pass.
#[derive(Debug, Clone, Default)]
pub struct ArtPassResult {
    pub nodes_tested: usize,
    pub nodes_enhanced: usize,
    pub total_hits: usize,
    pub total_misses: usize,
}

/// LLM response for test question generation.
#[derive(Debug, Deserialize)]
struct TestQuestionsResponse {
    questions: Vec<String>,
}

/// LLM response for node enhancement.
#[derive(Debug, Deserialize)]
struct EnhancementResponse {
    enhanced_description: String,
    keywords: Vec<String>,
}

/// Generate diverse test search queries that should find a node's content.
///
/// Single LLM call producing `config.test_questions` queries in JSON mode.
async fn generate_test_questions(
    client: &dyn LlmClient,
    node_label: &str,
    node_properties: &str,
    num_questions: usize,
) -> Vec<String> {
    let system = format!(
        "Generate exactly {} diverse search queries that a user might type to find the following \
         piece of information. Queries should vary in style: some natural language questions, \
         some keyword-based. Output strict JSON: {{\"questions\": [...]}}. No markdown fences.",
        num_questions
    );

    let user = format!(
        "Node content:\n{}\n\nProperties:\n{}",
        node_label, node_properties
    );

    let request = LlmRequest {
        system_prompt: system,
        user_prompt: user,
        temperature: 0.7,
        max_tokens: 256,
        json_mode: true,
    };

    match tokio::time::timeout(std::time::Duration::from_secs(10), client.complete(request)).await {
        Ok(Ok(response)) => {
            if let Some(value) = parse_json_from_llm(&response.content) {
                if let Ok(parsed) = serde_json::from_value::<TestQuestionsResponse>(value) {
                    return parsed
                        .questions
                        .into_iter()
                        .filter(|q| !q.trim().is_empty())
                        .take(num_questions)
                        .collect();
                }
            }
            vec![]
        },
        _ => vec![],
    }
}

/// Test whether a node appears in search results for a set of embedded questions.
///
/// For each question embedding, searches both node_vector_index and bm25_index.
/// Returns (hits, misses).
fn test_retrieval(
    graph: &crate::structures::Graph,
    target_node_id: NodeId,
    question_embeddings: &[Vec<f32>],
    question_texts: &[String],
    top_k: usize,
) -> (usize, usize) {
    let mut hits = 0usize;
    let mut misses = 0usize;

    for (i, embedding) in question_embeddings.iter().enumerate() {
        let mut found = false;

        // Search vector index
        let vector_results = graph.node_vector_index.search(embedding, top_k, 0.0);
        if vector_results.iter().any(|(nid, _)| *nid == target_node_id) {
            found = true;
        }

        // Search BM25 index
        if !found {
            if let Some(text) = question_texts.get(i) {
                let bm25_results = graph.bm25_index.search(text, top_k);
                if bm25_results.iter().any(|(nid, _)| *nid == target_node_id) {
                    found = true;
                }
            }
        }

        if found {
            hits += 1;
        } else {
            misses += 1;
        }
    }

    (hits, misses)
}

/// Generate an enhanced description and keywords for a node that fails retrieval.
///
/// Single LLM call that produces richer text and search terms.
async fn generate_enhancement(
    client: &dyn LlmClient,
    node_label: &str,
    node_properties: &str,
    failing_queries: &[String],
) -> Option<EnhancementResponse> {
    let system = "A graph node is not being found by the search queries listed below. \
                  Generate an enhanced description (1-3 sentences, information-rich) and \
                  5-10 search keywords that would help this node be found. \
                  Output strict JSON: {\"enhanced_description\": \"...\", \"keywords\": [...]}. \
                  No markdown fences.";

    let user = format!(
        "Node content: {}\nProperties: {}\n\nFailing search queries:\n{}",
        node_label,
        node_properties,
        failing_queries
            .iter()
            .enumerate()
            .map(|(i, q)| format!("{}. {}", i + 1, q))
            .collect::<Vec<_>>()
            .join("\n")
    );

    let request = LlmRequest {
        system_prompt: system.to_string(),
        user_prompt: user,
        temperature: 0.3,
        max_tokens: 256,
        json_mode: true,
    };

    match tokio::time::timeout(std::time::Duration::from_secs(10), client.complete(request)).await {
        Ok(Ok(response)) => {
            if let Some(value) = parse_json_from_llm(&response.content) {
                serde_json::from_value::<EnhancementResponse>(value).ok()
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Apply enhancement to a node: update properties, re-index BM25, re-embed.
async fn apply_enhancement_to_graph(
    inference: &Arc<RwLock<GraphInference>>,
    embedding_client: &dyn EmbeddingClient,
    node_id: NodeId,
    enhancement: &EnhancementResponse,
) {
    // Build combined text for re-embedding
    let combined_text = format!(
        "{} {}",
        enhancement.enhanced_description,
        enhancement.keywords.join(" ")
    );

    // Generate new embedding
    let new_embedding = match embedding_client
        .embed(EmbeddingRequest {
            text: combined_text.clone(),
            context: None,
        })
        .await
    {
        Ok(resp) if !resp.embedding.is_empty() => resp.embedding,
        _ => return,
    };

    // Apply changes to graph
    let mut inf = inference.write().await;
    let graph = inf.graph_mut();

    if let Some(node) = graph.get_node_mut(node_id) {
        // Store enhancement in properties
        node.properties.insert(
            "art_enhanced_description".to_string(),
            serde_json::Value::String(enhancement.enhanced_description.clone()),
        );
        node.properties.insert(
            "art_keywords".to_string(),
            serde_json::Value::Array(
                enhancement
                    .keywords
                    .iter()
                    .map(|k| serde_json::Value::String(k.clone()))
                    .collect(),
            ),
        );
        node.embedding = new_embedding.clone();
    }

    // Re-index BM25: remove old, add combined text
    graph.bm25_index.remove_document(node_id);
    let label = graph
        .get_node(node_id)
        .map(|n| n.label())
        .unwrap_or_default();
    let bm25_text = format!("{} {}", label, combined_text);
    graph.bm25_index.index_document(node_id, &bm25_text);

    // Update vector index
    graph.node_vector_index.insert(node_id, new_embedding);
}

/// Run a single ART pass over candidate nodes.
///
/// Processes nodes sequentially up to `config.max_nodes_per_pass`. For each node:
/// 1. Generate test questions (1 LLM call)
/// 2. Embed test questions (batch)
/// 3. Test retrieval (pure graph, no LLM)
/// 4. If hit rate < threshold: generate enhancement (1 LLM call) and apply
pub async fn run_art_pass(
    candidate_node_ids: Vec<NodeId>,
    inference: &Arc<RwLock<GraphInference>>,
    llm: &dyn LlmClient,
    embedding_client: &dyn EmbeddingClient,
    config: &ArtConfig,
) -> ArtPassResult {
    let mut result = ArtPassResult::default();

    let nodes_to_process: Vec<NodeId> = candidate_node_ids
        .into_iter()
        .take(config.max_nodes_per_pass)
        .collect();

    for &node_id in &nodes_to_process {
        // Collect node info under read lock
        let (node_label, node_props) = {
            let inf = inference.read().await;
            let graph = inf.graph();
            match graph.get_node(node_id) {
                Some(node) => {
                    let label = node.label();
                    let props = node
                        .properties
                        .iter()
                        .filter(|(k, _)| !k.starts_with("art_"))
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    (label, props)
                },
                None => continue,
            }
        };

        result.nodes_tested += 1;

        // Step 1: Generate test questions
        let questions =
            generate_test_questions(llm, &node_label, &node_props, config.test_questions).await;
        if questions.is_empty() {
            continue;
        }

        // Step 2: Embed test questions (batch)
        let embed_requests: Vec<EmbeddingRequest> = questions
            .iter()
            .map(|q| EmbeddingRequest {
                text: q.clone(),
                context: None,
            })
            .collect();

        let embeddings = match embedding_client.embed_batch(embed_requests).await {
            Ok(resps) => resps.into_iter().map(|r| r.embedding).collect::<Vec<_>>(),
            Err(_) => continue,
        };

        // Step 3: Test retrieval
        let (hits, misses) = {
            let inf = inference.read().await;
            let graph = inf.graph();
            test_retrieval(graph, node_id, &embeddings, &questions, config.top_k)
        };

        result.total_hits += hits;
        result.total_misses += misses;

        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };

        if hit_rate >= config.min_hit_rate {
            tracing::debug!(
                "ART: node {} '{}' passed ({}/{} hit rate {:.0}%)",
                node_id,
                safe_truncate(&node_label, 40),
                hits,
                total,
                hit_rate * 100.0
            );
            continue;
        }

        // Step 4: Generate enhancement for failing node
        tracing::info!(
            "ART: node {} '{}' failed ({}/{} hit rate {:.0}%), enhancing",
            node_id,
            safe_truncate(&node_label, 40),
            hits,
            total,
            hit_rate * 100.0
        );

        // Collect the queries that missed (check vector index under read lock)
        let failing_queries: Vec<String> = {
            let inf = inference.read().await;
            let graph = inf.graph();
            questions
                .iter()
                .zip(embeddings.iter())
                .filter(|(_, emb)| {
                    let vec_results = graph.node_vector_index.search(emb, config.top_k, 0.0);
                    !vec_results.iter().any(|(nid, _)| *nid == node_id)
                })
                .map(|(q, _)| q.clone())
                .collect()
        };

        if let Some(enhancement) =
            generate_enhancement(llm, &node_label, &node_props, &failing_queries).await
        {
            apply_enhancement_to_graph(inference, embedding_client, node_id, &enhancement).await;
            result.nodes_enhanced += 1;
        }
    }

    if result.nodes_tested > 0 {
        tracing::info!(
            "ART pass complete: {}/{} nodes tested, {} enhanced, {}/{} total hits",
            result.nodes_tested,
            nodes_to_process.len(),
            result.nodes_enhanced,
            result.total_hits,
            result.total_hits + result.total_misses,
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_art_config_defaults() {
        let config = ArtConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.test_questions, 5);
        assert_eq!(config.top_k, 10);
        assert!((config.min_hit_rate - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.max_nodes_per_pass, 20);
    }

    #[test]
    fn test_art_pass_result_default() {
        let result = ArtPassResult::default();
        assert_eq!(result.nodes_tested, 0);
        assert_eq!(result.nodes_enhanced, 0);
        assert_eq!(result.total_hits, 0);
        assert_eq!(result.total_misses, 0);
    }

    #[test]
    fn test_retrieval_empty_graph() {
        let graph = crate::structures::Graph::new();
        let (hits, misses) =
            test_retrieval(&graph, 0, &[vec![0.1, 0.2, 0.3]], &["test".to_string()], 10);
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
    }
}
