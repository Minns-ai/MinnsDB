// crates/agent-db-graph/src/refinement.rs
//
// LLM Refinement Pipeline
//
// After a Memory or Strategy is formed with a template summary, this module:
//   1. Sends the template to the LLM to produce a refined natural-language summary,
//      takeaway, and causal note.
//   2. Embeds the refined summary for semantic retrieval.
//   3. Updates the Memory/Strategy in-place.
//
// Runs asynchronously — fire-and-forget after formation.

use crate::claims::EmbeddingClient;
use crate::memory::{Memory, MemoryId};
use crate::stores::MemoryStore;
use crate::strategies::{PlaybookStep, Strategy};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ============================================================================
// LLM Refinement Prompts
// ============================================================================

const MEMORY_REFINEMENT_SYSTEM_PROMPT: &str = r#"You are an expert at distilling AI agent experiences into concise, actionable knowledge.
Given raw event data from an agent's episode, produce a JSON response with:
1. "summary": A clear 2-3 sentence narrative of what happened (no jargon, no IDs)
2. "takeaway": The single most important lesson from this experience (1 sentence)
3. "causal_note": Why the outcome was what it was — identify the key causal factors (1-2 sentences)

If broader knowledge context is provided, use it to place the memory within the agent's
overall knowledge graph. Reference relevant communities or patterns when they add context.

Be specific, not generic. Name the actual actions, tools, and outcomes.
Output ONLY valid JSON, no markdown fences."#;

const STRATEGY_REFINEMENT_SYSTEM_PROMPT: &str = r#"You are an expert at codifying AI agent strategies into reusable, generalized playbooks.
Given raw strategy data, produce a JSON response with:
1. "summary": A clear 2-3 sentence description of this strategy, generalized for reuse
2. "when_to_use": Specific conditions where this strategy applies (1-2 sentences)
3. "when_not_to_use": Conditions where this strategy should NOT be used (1-2 sentences)
4. "failure_modes": Array of known failure modes, each as a short sentence (max 3)
5. "counterfactual": What would have happened differently with an alternative approach (1 sentence)
6. "playbook": Array of generalized step objects: [{"step": 1, "action": "...", "condition": "...", "recovery": "..."}]

CRITICAL: Generalize ALL instance-specific values into reusable descriptions:
- Replace specific IDs with their role: "ORD-1001" → "the target order ID", "user_42" → "the requesting user"
- Replace specific names: "John Smith" → "the customer", "alice@corp.com" → "the user's email"
- Replace specific URLs/paths with descriptions: "https://api.example.com/v2/orders" → "the orders API endpoint"
- Keep action/tool names (e.g., "lookup_order", "send_email") — those are reusable patterns
- The playbook should work for ANY similar situation, not just the specific instance

If broader knowledge context or similar strategies are provided, reference them to identify
patterns, differentiate from existing strategies, and note complements or conflicts.

Output ONLY valid JSON, no markdown fences."#;

/// Refined memory fields from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedMemoryFields {
    pub summary: String,
    pub takeaway: String,
    pub causal_note: String,
}

/// Refined strategy fields from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedStrategyFields {
    pub summary: String,
    pub when_to_use: String,
    pub when_not_to_use: String,
    #[serde(default)]
    pub failure_modes: Vec<String>,
    #[serde(default)]
    pub counterfactual: String,
    #[serde(default)]
    pub playbook: Vec<RefinedPlaybookStep>,
}

/// A single generalized playbook step from LLM refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedPlaybookStep {
    pub step: u32,
    pub action: String,
    #[serde(default)]
    pub condition: String,
    #[serde(default)]
    pub recovery: String,
}

// ============================================================================
// Refinement Engine
// ============================================================================

/// Configuration for the refinement pipeline
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Whether LLM refinement is enabled (requires API key)
    pub enable_llm_refinement: bool,
    /// Whether to embed summaries for semantic retrieval
    pub enable_summary_embedding: bool,
    /// Model to use for refinement (e.g. "gpt-4o-mini")
    pub model: String,
    /// Maximum tokens for refinement responses
    pub max_tokens: u32,
    /// Temperature for refinement
    pub temperature: f32,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            enable_llm_refinement: false,
            enable_summary_embedding: true,
            model: "gpt-4o-mini".to_string(),
            max_tokens: 512,
            temperature: 0.3,
        }
    }
}

/// The refinement engine that processes memories and strategies post-formation.
pub struct RefinementEngine {
    config: RefinementConfig,
    api_key: Option<String>,
    http_client: reqwest::Client,
}

impl RefinementEngine {
    pub fn new(config: RefinementConfig, api_key: Option<String>) -> Self {
        Self {
            config,
            api_key,
            http_client: reqwest::Client::new(),
        }
    }

    /// Check if LLM refinement is available (has API key + enabled)
    pub fn is_llm_available(&self) -> bool {
        self.config.enable_llm_refinement && self.api_key.is_some()
    }

    // ========== Memory Refinement ==========

    /// Refine a memory's summary, takeaway, and causal_note using LLM.
    /// When `event_data` is provided, it contains a rich narrative of the raw events
    /// that the LLM can use to produce a much better refinement than template text alone.
    pub async fn refine_memory(
        &self,
        memory: &Memory,
        event_data: Option<&str>,
        community_context: Option<&str>,
    ) -> Result<RefinedMemoryFields> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No API key for refinement"))?;

        let event_section = match event_data {
            Some(narrative) => format!("\n\nRaw event narrative:\n{}", narrative),
            None => String::new(),
        };

        let context_section = match community_context {
            Some(ctx) if !ctx.is_empty() => format!("\n\nBroader knowledge context:\n{}", ctx),
            _ => String::new(),
        };

        let user_prompt = format!(
            "Template summary: {}\nTakeaway: {}\nCausal note: {}{}{}\n\nOutcome: {:?}, Strength: {:.2}, Relevance: {:.2}",
            memory.summary,
            memory.takeaway,
            memory.causal_note,
            event_section,
            context_section,
            memory.outcome,
            memory.strength,
            memory.relevance_score
        );

        let response = self
            .call_llm(api_key, MEMORY_REFINEMENT_SYSTEM_PROMPT, &user_prompt)
            .await?;

        let refined: RefinedMemoryFields = serde_json::from_str(&response).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse memory refinement response: {} — raw: {}",
                e,
                response
            )
        })?;

        Ok(refined)
    }

    /// Refine a strategy's summary, when_to_use, when_not_to_use, etc. using LLM.
    /// When `event_data` is provided, it contains a rich narrative of the raw events.
    pub async fn refine_strategy(
        &self,
        strategy: &Strategy,
        event_data: Option<&str>,
        community_context: Option<&str>,
        similar_strategies: Option<&str>,
    ) -> Result<RefinedStrategyFields> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No API key for refinement"))?;

        let event_section = match event_data {
            Some(narrative) => format!("\n\nRaw event narrative:\n{}", narrative),
            None => String::new(),
        };

        let context_section = match community_context {
            Some(ctx) if !ctx.is_empty() => format!("\n\nBroader knowledge context:\n{}", ctx),
            _ => String::new(),
        };

        let strategy_section = match similar_strategies {
            Some(s) if !s.is_empty() => format!("\n\n{}", s),
            _ => String::new(),
        };

        let playbook_text = if strategy.playbook.is_empty() {
            "No playbook steps".to_string()
        } else {
            strategy
                .playbook
                .iter()
                .map(|s| format!("  {}. {}", s.step, s.action))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let user_prompt = format!(
            "Raw strategy data:\n\nSummary: {}\nWhen to use: {}\nWhen not to use: {}\nAction hint: {}\nPrecondition: {}\nSuccess rate: {:.0}%\nQuality: {:.2}\nFailure patterns: {:?}\nPlaybook steps:\n{}{}{}{}",
            strategy.summary,
            strategy.when_to_use,
            strategy.when_not_to_use,
            strategy.action_hint,
            strategy.precondition,
            strategy.expected_success * 100.0,
            strategy.quality_score,
            strategy.failure_patterns,
            playbook_text,
            event_section,
            context_section,
            strategy_section,
        );

        let response = self
            .call_llm(api_key, STRATEGY_REFINEMENT_SYSTEM_PROMPT, &user_prompt)
            .await?;

        let refined: RefinedStrategyFields = serde_json::from_str(&response).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse strategy refinement response: {} — raw: {}",
                e,
                response
            )
        })?;

        Ok(refined)
    }

    /// Embed a summary text and return the embedding vector.
    pub async fn embed_summary(
        &self,
        text: &str,
        embedding_client: &dyn EmbeddingClient,
    ) -> Result<Vec<f32>> {
        let request = crate::claims::EmbeddingRequest {
            text: text.to_string(),
            context: None,
        };
        let response = embedding_client.embed(request).await?;
        Ok(response.embedding)
    }

    // ========== Full Refinement Pipeline ==========

    /// Run the full refinement pipeline for a newly formed memory:
    /// 1. LLM refine summary/takeaway/causal_note (if enabled)
    /// 2. Embed the summary (if enabled)
    /// 3. Update the memory in the store
    pub async fn refine_and_embed_memory(
        &self,
        memory_id: MemoryId,
        store: &Arc<RwLock<Box<dyn MemoryStore>>>,
        embedding_client: Option<&Arc<dyn EmbeddingClient>>,
        event_data: Option<String>,
        community_context: Option<String>,
    ) -> Result<()> {
        // Get current memory
        let memory = {
            let s = store.read().await;
            s.get_memory(memory_id)
                .ok_or_else(|| anyhow::anyhow!("Memory {} not found", memory_id))?
        };

        let mut updated_memory = memory.clone();

        // Step 1: LLM refinement
        if self.is_llm_available() {
            match self
                .refine_memory(&memory, event_data.as_deref(), community_context.as_deref())
                .await
            {
                Ok(refined) => {
                    info!(
                        "Refined memory {} summary: {} -> {}",
                        memory_id,
                        &memory.summary[..memory.summary.len().min(50)],
                        &refined.summary[..refined.summary.len().min(50)]
                    );
                    updated_memory.summary = refined.summary;
                    updated_memory.takeaway = refined.takeaway;
                    updated_memory.causal_note = refined.causal_note;
                },
                Err(e) => {
                    warn!("LLM refinement failed for memory {}: {}", memory_id, e);
                    // Keep template summary
                },
            }
        }

        // Step 2: Embed the summary
        if self.config.enable_summary_embedding {
            if let Some(client) = embedding_client {
                match self
                    .embed_summary(&updated_memory.summary, client.as_ref())
                    .await
                {
                    Ok(embedding) => {
                        debug!(
                            "Embedded memory {} summary ({} dims)",
                            memory_id,
                            embedding.len()
                        );
                        updated_memory.summary_embedding = embedding;
                    },
                    Err(e) => {
                        warn!("Embedding failed for memory {}: {}", memory_id, e);
                    },
                }
            }
        }

        // Step 3: Update in store
        {
            let mut s = store.write().await;
            s.store_consolidated_memory(updated_memory);
        }

        Ok(())
    }

    /// Run the full refinement pipeline for a strategy (LLM + embed).
    /// Returns the refined strategy (caller is responsible for persisting).
    pub async fn refine_and_embed_strategy(
        &self,
        strategy: &Strategy,
        embedding_client: Option<&Arc<dyn EmbeddingClient>>,
        event_data: Option<String>,
        community_context: Option<String>,
        similar_strategies: Option<String>,
    ) -> Result<Strategy> {
        let mut updated = strategy.clone();

        // Step 1: LLM refinement
        if self.is_llm_available() {
            match self
                .refine_strategy(
                    strategy,
                    event_data.as_deref(),
                    community_context.as_deref(),
                    similar_strategies.as_deref(),
                )
                .await
            {
                Ok(refined) => {
                    info!(
                        "Refined strategy {} summary: {} -> {}",
                        strategy.id,
                        &strategy.summary[..strategy.summary.len().min(50)],
                        &refined.summary[..refined.summary.len().min(50)]
                    );
                    updated.summary = refined.summary;
                    updated.when_to_use = refined.when_to_use;
                    updated.when_not_to_use = refined.when_not_to_use;
                    if !refined.failure_modes.is_empty() {
                        updated.failure_modes = refined.failure_modes;
                    }
                    if !refined.counterfactual.is_empty() {
                        updated.counterfactual = refined.counterfactual;
                    }
                    if !refined.playbook.is_empty() {
                        updated.playbook = refined
                            .playbook
                            .iter()
                            .map(|rs| PlaybookStep {
                                step: rs.step,
                                action: rs.action.clone(),
                                condition: rs.condition.clone(),
                                recovery: rs.recovery.clone(),
                                ..Default::default()
                            })
                            .collect();
                    }
                },
                Err(e) => {
                    warn!("LLM refinement failed for strategy {}: {}", strategy.id, e);
                },
            }
        }

        // Step 2: Embed the summary
        if self.config.enable_summary_embedding {
            if let Some(client) = embedding_client {
                match self.embed_summary(&updated.summary, client.as_ref()).await {
                    Ok(embedding) => {
                        debug!(
                            "Embedded strategy {} summary ({} dims)",
                            strategy.id,
                            embedding.len()
                        );
                        updated.summary_embedding = embedding;
                    },
                    Err(e) => {
                        warn!("Embedding failed for strategy {}: {}", strategy.id, e);
                    },
                }
            }
        }

        Ok(updated)
    }

    // ========== Internal ==========

    async fn call_llm(
        &self,
        api_key: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
        let response = self
            .http_client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"}
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow::anyhow!("LLM API error {}: {}", status, body));
        }

        let json: serde_json::Value = response.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No content in LLM response"))?;

        Ok(content.to_string())
    }
}
