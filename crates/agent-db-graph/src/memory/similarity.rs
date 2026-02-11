// crates/agent-db-graph/src/memory/similarity.rs
//
// Shared context similarity logic used by both in-memory and persistent stores.

use agent_db_core::utils::cosine_similarity;
use agent_db_events::core::EventContext;
use std::collections::HashSet;

/// Canonical context similarity used across the crate.
///
/// Prefers embedding cosine similarity when both sides have embeddings.
/// Falls back to structured similarity based on environment/goals/resources.
pub(crate) fn calculate_context_similarity(a: &EventContext, b: &EventContext) -> f32 {
    if let (Some(embed_a), Some(embed_b)) = (&a.embeddings, &b.embeddings) {
        return cosine_similarity(embed_a, embed_b);
    }

    fallback_context_similarity(a, b)
}

fn fallback_context_similarity(a: &EventContext, b: &EventContext) -> f32 {
    let env_similarity = key_overlap_ratio(
        a.environment.variables.keys(),
        b.environment.variables.keys(),
    );
    let goals_similarity = id_overlap_ratio(
        a.active_goals.iter().map(|goal| goal.id),
        b.active_goals.iter().map(|goal| goal.id),
    );
    let resources_similarity = resource_similarity(a, b);

    let mut total = 0.0;
    let mut weight = 0.0;

    if env_similarity >= 0.0 {
        total += env_similarity * 0.4;
        weight += 0.4;
    }

    if goals_similarity >= 0.0 {
        total += goals_similarity * 0.3;
        weight += 0.3;
    }

    total += resources_similarity * 0.3;
    weight += 0.3;

    if weight == 0.0 {
        0.0
    } else {
        (total / weight).clamp(0.0, 1.0)
    }
}

fn key_overlap_ratio<'a, I, J>(a: I, b: J) -> f32
where
    I: Iterator<Item = &'a String>,
    J: Iterator<Item = &'a String>,
{
    let set_a: HashSet<&String> = a.collect();
    let set_b: HashSet<&String> = b.collect();
    if set_a.is_empty() && set_b.is_empty() {
        return -1.0;
    }

    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;

    if union == 0.0 {
        -1.0
    } else {
        intersection / union
    }
}

fn id_overlap_ratio<I, J>(a: I, b: J) -> f32
where
    I: Iterator<Item = u64>,
    J: Iterator<Item = u64>,
{
    let set_a: HashSet<u64> = a.collect();
    let set_b: HashSet<u64> = b.collect();
    if set_a.is_empty() && set_b.is_empty() {
        return -1.0;
    }

    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;

    if union == 0.0 {
        -1.0
    } else {
        intersection / union
    }
}

fn resource_similarity(a: &EventContext, b: &EventContext) -> f32 {
    let cpu_a = a.resources.computational.cpu_percent;
    let cpu_b = b.resources.computational.cpu_percent;
    let cpu_max = cpu_a.max(cpu_b).max(1.0);
    let cpu_sim = 1.0 - ((cpu_a - cpu_b).abs() / cpu_max).min(1.0);

    let mem_a = a.resources.computational.memory_bytes as f32;
    let mem_b = b.resources.computational.memory_bytes as f32;
    let mem_max = mem_a.max(mem_b);
    let mem_sim = if mem_max == 0.0 {
        1.0
    } else {
        1.0 - ((mem_a - mem_b).abs() / mem_max).min(1.0)
    };

    ((cpu_sim + mem_sim) / 2.0).clamp(0.0, 1.0)
}
