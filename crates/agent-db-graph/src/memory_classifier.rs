//! LLM-driven memory update classification.
//!
//! When new facts/memories arrive, this module:
//! 1. Searches for similar existing memories (by embedding or BM25)
//! 2. Maps real memory IDs to small integers (prevents LLM hallucination)
//! 3. Asks the LLM to classify each new fact as ADD/UPDATE/DELETE/NONE
//! 4. Executes the classified operations
//!
//! Inspired by prior work's memory update classification pipeline.

use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::memory::{Memory, MemoryId};
use serde::{Deserialize, Serialize};

/// Classification decision for a new fact against existing memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MemoryAction {
    /// Create a new memory — no relevant existing memory found.
    Add,
    /// Update an existing memory — new fact refines/extends it.
    Update,
    /// Delete an existing memory — it's been contradicted or superseded.
    Delete,
    /// Do nothing — the fact is already captured or irrelevant.
    None,
}

/// A single classified operation from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedOperation {
    /// The action to take.
    pub action: MemoryAction,
    /// For UPDATE/DELETE: the integer index of the existing memory to act on.
    /// Uses integer indices (not real IDs) to prevent LLM hallucination.
    pub target_index: Option<usize>,
    /// For ADD/UPDATE: the new/updated text for the memory.
    pub new_text: Option<String>,
    /// The original fact text that triggered this classification.
    pub fact_text: String,
}

/// Result of the full classification pipeline.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Operations to execute, in order.
    pub operations: Vec<ClassifiedOperation>,
    /// Number of tokens used by the LLM call.
    pub tokens_used: u32,
}

/// Maps real MemoryIds to small integers for the LLM prompt.
#[derive(Debug)]
struct IdMapper {
    /// index → real MemoryId
    id_to_real: Vec<MemoryId>,
}

impl IdMapper {
    fn new(memories: &[&Memory]) -> Self {
        Self {
            id_to_real: memories.iter().map(|m| m.id).collect(),
        }
    }

    /// Convert LLM integer index back to real MemoryId.
    fn to_real(&self, index: usize) -> Option<MemoryId> {
        self.id_to_real.get(index).copied()
    }

    /// Build the existing memories section for the prompt.
    fn format_existing(&self, memories: &[&Memory]) -> String {
        if memories.is_empty() {
            return "No existing memories found.".to_string();
        }
        memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "  [{}]: {} (takeaway: {})",
                    i,
                    truncate(&m.summary, 200),
                    truncate(&m.takeaway, 100),
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// System prompt for the memory update classifier LLM call.
const SYSTEM_PROMPT: &str = r#"You are a memory management assistant. Your task is to decide how new facts should be integrated with existing memories.

For each new fact, decide ONE action:
- ADD: The fact is genuinely new information not captured by any existing memory. Create a new memory.
- UPDATE: The fact refines, extends, or corrects an existing memory. Specify which memory (by index number) to update and provide the COMPLETE updated text that merges the old and new information (not just the delta — the old memory will be replaced entirely).
- DELETE: The fact directly contradicts an existing memory, making it false. Only use DELETE when the old fact is clearly wrong, not merely outdated. Prefer UPDATE when the fact is a refinement or correction.
- NONE: The fact is already fully captured by existing memories, or is too trivial to store.

Rules:
- Be conservative with DELETE — only delete when clearly contradicted (e.g., "User is vegetarian" contradicts "User loves steak").
- Prefer UPDATE over ADD when a fact extends or refines existing knowledge about the same subject.
- Prefer UPDATE over DELETE when the new fact corrects but doesn't negate the old one.
- Use NONE for redundant or trivial information.
- Reference existing memories ONLY by their integer index [0], [1], etc.
- For UPDATE, the "new_text" MUST contain the complete merged text. The old memory will be replaced entirely with this text.
- If knowledge graph context is provided, use it to better determine if a new fact overlaps
  with existing knowledge. Prefer UPDATE over ADD when the same entity appears across communities.

Examples:

1. New fact: "User's favorite color is blue"
   Existing: (none)
   → {"action": "ADD", "target_index": null, "new_text": "User's favorite color is blue", "fact": "User's favorite color is blue"}

2. New fact: "User now lives in San Francisco"
   Existing: [0]: "User lives in New York and works at Google"
   → {"action": "UPDATE", "target_index": 0, "new_text": "User lives in San Francisco and works at Google", "fact": "User now lives in San Francisco"}

3. New fact: "User is vegetarian"
   Existing: [0]: "User loves eating steak every weekend"
   → {"action": "DELETE", "target_index": 0, "new_text": null, "fact": "User is vegetarian"}
   followed by: {"action": "ADD", "target_index": null, "new_text": "User is vegetarian", "fact": "User is vegetarian"}

Respond with a JSON array. Each element has:
  {"action": "ADD"|"UPDATE"|"DELETE"|"NONE", "target_index": <int or null>, "new_text": <string or null>, "fact": <original fact text>}
"#;

/// Build the user prompt with existing memories and new facts.
fn build_user_prompt(existing_section: &str, facts: &[&str]) -> String {
    let facts_section = facts
        .iter()
        .enumerate()
        .map(|(i, f)| format!("  {}. {}", i + 1, f))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Existing memories:\n{}\n\nNew facts to classify:\n{}\n\nRespond with the JSON array.",
        existing_section, facts_section
    )
}

/// Parse the LLM response into classified operations.
fn parse_classification_response(
    response: &str,
    facts: &[&str],
    mapper: &IdMapper,
) -> Vec<ClassifiedOperation> {
    let json = match parse_json_from_llm(response) {
        Some(v) => v,
        None => return default_add_all(facts),
    };

    let arr = match json.as_array() {
        Some(a) => a,
        None => return default_add_all(facts),
    };

    let mut ops = Vec::new();
    for (i, item) in arr.iter().enumerate() {
        let fact_text = item["fact"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| facts.get(i).unwrap_or(&"").to_string());

        let action_str = item["action"].as_str().unwrap_or("ADD").to_uppercase();
        let action = match action_str.as_str() {
            "UPDATE" => MemoryAction::Update,
            "DELETE" => MemoryAction::Delete,
            "NONE" => MemoryAction::None,
            _ => MemoryAction::Add,
        };

        let target_index = item["target_index"].as_u64().map(|v| v as usize);

        // Validate target_index exists in mapper
        let valid_target = match (action, target_index) {
            (MemoryAction::Update | MemoryAction::Delete, Some(idx)) => {
                if mapper.to_real(idx).is_some() {
                    Some(idx)
                } else {
                    // Invalid index — fall back to ADD
                    None
                }
            },
            _ => target_index,
        };

        // If UPDATE/DELETE had an invalid index, downgrade to ADD
        let final_action = match (action, valid_target) {
            (MemoryAction::Update, None) => MemoryAction::Add,
            (MemoryAction::Delete, None) => MemoryAction::None,
            (a, _) => a,
        };

        let new_text = item["new_text"].as_str().map(|s| s.to_string());

        ops.push(ClassifiedOperation {
            action: final_action,
            target_index: valid_target,
            new_text,
            fact_text,
        });
    }

    ops
}

/// Fallback: if LLM response is unparseable, ADD all facts.
fn default_add_all(facts: &[&str]) -> Vec<ClassifiedOperation> {
    facts
        .iter()
        .map(|f| ClassifiedOperation {
            action: MemoryAction::Add,
            target_index: None,
            new_text: Some(f.to_string()),
            fact_text: f.to_string(),
        })
        .collect()
}

/// Classify new facts against existing similar memories using an LLM.
///
/// # Arguments
/// * `llm` — The LLM client to use for classification.
/// * `new_facts` — New fact strings to classify.
/// * `similar_memories` — Existing memories that are semantically similar to the facts.
/// * `community_context` — Optional community context to improve classification.
///
/// # Returns
/// A `ClassificationResult` with operations to execute and token usage.
pub async fn classify_memory_updates(
    llm: &dyn LlmClient,
    new_facts: &[&str],
    similar_memories: &[&Memory],
    community_context: Option<&str>,
) -> anyhow::Result<ClassificationResult> {
    if new_facts.is_empty() {
        return Ok(ClassificationResult {
            operations: Vec::new(),
            tokens_used: 0,
        });
    }

    let mapper = IdMapper::new(similar_memories);
    let existing_section = mapper.format_existing(similar_memories);
    let mut user_prompt = build_user_prompt(&existing_section, new_facts);

    if let Some(ctx) = community_context {
        if !ctx.is_empty() {
            user_prompt.push_str(&format!("\n\nKnowledge graph context:\n{}", ctx));
        }
    }

    let request = LlmRequest {
        system_prompt: SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 1024,
        json_mode: true,
    };

    let response = llm.complete(request).await?;
    let operations = parse_classification_response(&response.content, new_facts, &mapper);

    Ok(ClassificationResult {
        operations,
        tokens_used: response.tokens_used,
    })
}

/// Resolve a classified operation's target_index to a real MemoryId.
pub fn resolve_target(op: &ClassifiedOperation, similar_memories: &[&Memory]) -> Option<MemoryId> {
    op.target_index
        .and_then(|idx| similar_memories.get(idx).map(|m| m.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_classification_add() {
        let mapper = IdMapper { id_to_real: vec![] };
        let response = r#"[{"action": "ADD", "target_index": null, "new_text": "User likes pizza", "fact": "User likes pizza"}]"#;
        let ops = parse_classification_response(response, &["User likes pizza"], &mapper);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Add);
        assert_eq!(ops[0].new_text.as_deref(), Some("User likes pizza"));
    }

    #[test]
    fn test_parse_classification_update() {
        let mapper = IdMapper {
            id_to_real: vec![100, 200],
        };
        let response = r#"[{"action": "UPDATE", "target_index": 0, "new_text": "User prefers pepperoni pizza (updated from generic pizza)", "fact": "User now prefers pepperoni"}]"#;
        let ops = parse_classification_response(response, &["User now prefers pepperoni"], &mapper);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Update);
        assert_eq!(ops[0].target_index, Some(0));
        assert!(ops[0].new_text.as_deref().unwrap().contains("pepperoni"));
    }

    #[test]
    fn test_parse_classification_delete() {
        let mapper = IdMapper {
            id_to_real: vec![100],
        };
        let response = r#"[{"action": "DELETE", "target_index": 0, "new_text": null, "fact": "User is actually vegetarian"}]"#;
        let ops =
            parse_classification_response(response, &["User is actually vegetarian"], &mapper);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Delete);
        assert_eq!(ops[0].target_index, Some(0));
    }

    #[test]
    fn test_parse_classification_none() {
        let mapper = IdMapper {
            id_to_real: vec![100],
        };
        let response = r#"[{"action": "NONE", "target_index": null, "new_text": null, "fact": "The sky is blue"}]"#;
        let ops = parse_classification_response(response, &["The sky is blue"], &mapper);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::None);
    }

    #[test]
    fn test_parse_invalid_target_index_downgrades() {
        // target_index=5 but only 2 memories → UPDATE downgrades to ADD
        let mapper = IdMapper {
            id_to_real: vec![100, 200],
        };
        let response = r#"[{"action": "UPDATE", "target_index": 5, "new_text": "updated", "fact": "some fact"}]"#;
        let ops = parse_classification_response(response, &["some fact"], &mapper);
        assert_eq!(ops[0].action, MemoryAction::Add);
    }

    #[test]
    fn test_parse_invalid_json_falls_back_to_add_all() {
        let mapper = IdMapper { id_to_real: vec![] };
        let facts = vec!["fact one", "fact two"];
        let ops = parse_classification_response("not json at all", &facts, &mapper);
        assert_eq!(ops.len(), 2);
        assert!(ops.iter().all(|o| o.action == MemoryAction::Add));
    }

    #[test]
    fn test_parse_fenced_json() {
        let mapper = IdMapper {
            id_to_real: vec![100],
        };
        let response = "```json\n[{\"action\": \"NONE\", \"target_index\": null, \"new_text\": null, \"fact\": \"hi\"}]\n```";
        let ops = parse_classification_response(response, &["hi"], &mapper);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::None);
    }

    #[test]
    fn test_resolve_target() {
        use crate::episodes::EpisodeOutcome;
        use crate::memory::MemoryType;
        use agent_db_events::core::EventContext;

        let mem = Memory {
            id: 42,
            agent_id: 1,
            session_id: 1,
            episode_id: 1,
            summary: "test".to_string(),
            takeaway: "test".to_string(),
            causal_note: String::new(),
            summary_embedding: Vec::new(),
            tier: crate::memory::MemoryTier::Episodic,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: crate::memory::ConsolidationStatus::Active,
            context: EventContext::default(),
            key_events: Vec::new(),
            strength: 0.7,
            relevance_score: 0.5,
            formed_at: 0,
            last_accessed: 0,
            access_count: 0,
            outcome: EpisodeOutcome::Success,
            memory_type: MemoryType::Episodic { significance: 0.5 },
            metadata: std::collections::HashMap::new(),
            expires_at: None,
        };

        let similar = vec![&mem];
        let op = ClassifiedOperation {
            action: MemoryAction::Update,
            target_index: Some(0),
            new_text: Some("updated".to_string()),
            fact_text: "fact".to_string(),
        };

        assert_eq!(resolve_target(&op, &similar), Some(42));

        let op_none = ClassifiedOperation {
            action: MemoryAction::Add,
            target_index: None,
            new_text: Some("new".to_string()),
            fact_text: "fact".to_string(),
        };
        assert_eq!(resolve_target(&op_none, &similar), None);
    }

    #[test]
    fn test_id_mapper_format() {
        use crate::episodes::EpisodeOutcome;
        use crate::memory::MemoryType;
        use agent_db_events::core::EventContext;

        let mem = Memory {
            id: 42,
            agent_id: 1,
            session_id: 1,
            episode_id: 1,
            summary: "Agent logged in successfully".to_string(),
            takeaway: "Login works".to_string(),
            causal_note: String::new(),
            summary_embedding: Vec::new(),
            tier: crate::memory::MemoryTier::Episodic,
            consolidated_from: Vec::new(),
            schema_id: None,
            consolidation_status: crate::memory::ConsolidationStatus::Active,
            context: EventContext::default(),
            key_events: Vec::new(),
            strength: 0.7,
            relevance_score: 0.5,
            formed_at: 0,
            last_accessed: 0,
            access_count: 0,
            outcome: EpisodeOutcome::Success,
            memory_type: MemoryType::Episodic { significance: 0.5 },
            metadata: std::collections::HashMap::new(),
            expires_at: None,
        };

        let mems = vec![&mem];
        let mapper = IdMapper::new(&mems);
        let formatted = mapper.format_existing(&mems);
        assert!(formatted.contains("[0]:"));
        assert!(formatted.contains("Agent logged in"));
        assert!(formatted.contains("Login works"));
    }
}
