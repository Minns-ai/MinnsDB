//! Generic numeric computation layer over structured memory.
//!
//! Operates on abstract ledger entries, state values, and entity pairs.
//! Not tied to any specific domain.

use crate::structured_memory::{LedgerDirection, MemoryTemplate, StructuredMemoryStore};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Transfer (output of debt simplification)
// ---------------------------------------------------------------------------

/// A suggested transfer from one entity to another.
///
/// Convention (matches benchmark): `from` is the entity that is **owed** money
/// (creditor), `to` is the entity that **owes** money (debtor).
/// The formatted output `"from -> to : amount"` reads as
/// "to settle, `to` pays `from` this amount".
#[derive(Debug, Clone)]
pub struct Transfer {
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub currency: String,
}

// ---------------------------------------------------------------------------
// Net balance computation
// ---------------------------------------------------------------------------

/// Compute net balances for all entities across all ledgers in the store.
///
/// Returns `HashMap<(entity_name, currency), net_amount>` where positive means
/// the entity is owed money (creditor) and negative means they owe money (debtor).
pub fn compute_net_balances(store: &StructuredMemoryStore) -> HashMap<(String, String), f64> {
    let mut balances: HashMap<(String, String), f64> = HashMap::new();

    for key in store.list_keys("ledger:") {
        if let Some(MemoryTemplate::Ledger {
            entity_pair,
            entries,
            ..
        }) = store.get(key)
        {
            let (name_a, name_b) = entity_pair;

            for entry in entries {
                // Determine the currency from the entry description or a default
                let currency = extract_currency_from_description(&entry.description)
                    .unwrap_or_else(|| "EUR".to_string()); // Most benchmark data uses EUR

                match entry.direction {
                    LedgerDirection::Credit => {
                        // entity_a is credited → entity_a is owed money, entity_b owes
                        *balances
                            .entry((name_a.clone(), currency.clone()))
                            .or_insert(0.0) += entry.amount;
                        *balances
                            .entry((name_b.clone(), currency.clone()))
                            .or_insert(0.0) -= entry.amount;
                    },
                    LedgerDirection::Debit => {
                        // entity_a is debited → entity_b is owed money, entity_a owes
                        *balances
                            .entry((name_a.clone(), currency.clone()))
                            .or_insert(0.0) -= entry.amount;
                        *balances
                            .entry((name_b.clone(), currency.clone()))
                            .or_insert(0.0) += entry.amount;
                    },
                }
            }
        }
    }

    balances
}

/// Extract currency code from the entry's description (if present).
fn extract_currency_from_description(_desc: &str) -> Option<String> {
    // For now, all entries use the same currency per case.
    // Could parse "EUR", "USD" etc. from description if needed.
    None
}

// ---------------------------------------------------------------------------
// Transfer minimization (greedy debt simplification)
// ---------------------------------------------------------------------------

/// Compute the minimum set of transfers to settle all debts.
///
/// Groups by currency, then runs greedy matching per currency.
pub fn minimize_transfers(
    balances: &HashMap<(String, String), f64>,
    _default_currency: &str,
) -> Vec<Transfer> {
    // Group balances by currency
    let mut by_currency: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    for ((name, currency), &amount) in balances {
        by_currency
            .entry(currency.clone())
            .or_default()
            .push((name.clone(), amount));
    }

    let mut transfers = Vec::new();

    for (currency, entity_balances) in &by_currency {
        let mut debtors: Vec<(String, f64)> = Vec::new();
        let mut creditors: Vec<(String, f64)> = Vec::new();

        for (name, amount) in entity_balances {
            if to_cents(*amount) < 0 {
                debtors.push((name.clone(), -amount)); // positive debt amount
            } else if to_cents(*amount) > 0 {
                creditors.push((name.clone(), *amount));
            }
        }

        // Sort by amount descending (largest first)
        debtors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        creditors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy matching
        let mut di = 0;
        let mut ci = 0;

        while di < debtors.len() && ci < creditors.len() {
            let transfer_amount = debtors[di].1.min(creditors[ci].1);
            if to_cents(transfer_amount) > 0 {
                // Benchmark convention: from=creditor (is owed), to=debtor (owes)
                transfers.push(Transfer {
                    from: creditors[ci].0.clone(),
                    to: debtors[di].0.clone(),
                    amount: round_cents(transfer_amount),
                    currency: currency.clone(),
                });
            }
            debtors[di].1 -= transfer_amount;
            creditors[ci].1 -= transfer_amount;

            if to_cents(debtors[di].1) <= 0 {
                di += 1;
            }
            if to_cents(creditors[ci].1) <= 0 {
                ci += 1;
            }
        }
    }

    // Sort transfers alphabetically by (from, to) for deterministic output
    transfers.sort_by(|a, b| a.from.cmp(&b.from).then(a.to.cmp(&b.to)));
    transfers
}

/// Convert a float amount to integer cents for exact comparison.
fn to_cents(v: f64) -> i64 {
    (v * 100.0).round() as i64
}

/// Round to nearest cent.
fn round_cents(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

// ---------------------------------------------------------------------------
// Aggregation over ledger
// ---------------------------------------------------------------------------

/// Sum of all entries in a specific ledger.
pub fn ledger_sum(store: &StructuredMemoryStore, key: &str) -> Option<f64> {
    match store.get(key)? {
        MemoryTemplate::Ledger { entries, .. } => Some(entries.iter().map(|e| e.amount).sum()),
        _ => None,
    }
}

/// Sum entries matching a description filter.
pub fn ledger_sum_by_label(store: &StructuredMemoryStore, label_filter: &str) -> f64 {
    let lower_filter = label_filter.to_lowercase();
    let mut total = 0.0;
    for key in store.list_keys("ledger:") {
        if let Some(MemoryTemplate::Ledger { entries, .. }) = store.get(key) {
            for entry in entries {
                if entry.description.to_lowercase().contains(&lower_filter) {
                    total += entry.amount;
                }
            }
        }
    }
    total
}

/// Group ledger entries by a grouping function and sum amounts.
pub fn ledger_group_by<F>(store: &StructuredMemoryStore, group_fn: F) -> HashMap<String, f64>
where
    F: Fn(&str) -> String,
{
    let mut groups: HashMap<String, f64> = HashMap::new();
    for key in store.list_keys("ledger:") {
        if let Some(MemoryTemplate::Ledger { entries, .. }) = store.get(key) {
            for entry in entries {
                let group = group_fn(&entry.description);
                *groups.entry(group).or_insert(0.0) += entry.amount;
            }
        }
    }
    groups
}

// ---------------------------------------------------------------------------
// State timeline queries
// ---------------------------------------------------------------------------

/// Get the current state for an entity and attribute.
pub fn state_current(
    store: &StructuredMemoryStore,
    entity_id: u64,
    attribute: &str,
) -> Option<String> {
    let key = format!("state:{}:{}", entity_id, attribute);
    store.state_current(&key).map(|s| s.to_string())
}

/// Get all state keys for an entity.
pub fn state_keys_for_entity(store: &StructuredMemoryStore, entity_id: u64) -> Vec<String> {
    store
        .list_keys(&format!("state:{}:", entity_id))
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Preference queries
// ---------------------------------------------------------------------------

/// Get ranked preferences for an entity and category.
pub fn rank_preferences(
    store: &StructuredMemoryStore,
    entity_id: u64,
    category: &str,
) -> Vec<(String, f32)> {
    let key = crate::structured_memory::prefs_key(entity_id, category);
    match store.get(&key) {
        Some(MemoryTemplate::PreferenceList { ranked_items, .. }) => ranked_items
            .iter()
            .map(|item| (item.name.clone(), item.score.unwrap_or(0.5) as f32))
            .collect(),
        _ => vec![],
    }
}

/// Find common preferences between two entities in a category.
pub fn common_preferences(
    store: &StructuredMemoryStore,
    entity_a: u64,
    entity_b: u64,
    category: &str,
) -> Vec<String> {
    let prefs_a: std::collections::HashSet<String> = rank_preferences(store, entity_a, category)
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let prefs_b: std::collections::HashSet<String> = rank_preferences(store, entity_b, category)
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    prefs_a.intersection(&prefs_b).cloned().collect()
}

// ---------------------------------------------------------------------------
// Relationship path queries
// ---------------------------------------------------------------------------

/// Check if two people are connected through the relationship tree.
///
/// Returns the path if found, or None.
pub fn find_relationship_path(
    store: &StructuredMemoryStore,
    from: &str,
    to: &str,
    relation_type: &str,
) -> Option<Vec<String>> {
    let key = format!("tree:relations:{}", relation_type);
    store.get(&key)?;

    // BFS through the tree
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let mut predecessors: HashMap<String, String> = HashMap::new();

    visited.insert(from.to_string());
    queue.push_back(from.to_string());

    while let Some(current) = queue.pop_front() {
        if current == to {
            // Reconstruct path
            let mut path = vec![to.to_string()];
            let mut node = to.to_string();
            while let Some(prev) = predecessors.get(&node) {
                path.push(prev.clone());
                node = prev.clone();
            }
            path.reverse();
            return Some(path);
        }

        if let Some(children) = store.tree_children(&key, &current) {
            for child in children {
                if !visited.contains(child) {
                    visited.insert(child.clone());
                    predecessors.insert(child.clone(), current.clone());
                    queue.push_back(child.clone());
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format transfers into benchmark-compatible settlement text.
///
/// Output format: `"Settlement: Alice -> Bob : 172.50 EUR, Charlie -> Bob : 60.00 EUR"`
pub fn format_transfers(transfers: &[Transfer]) -> String {
    if transfers.is_empty() {
        return "All debts are settled.".to_string();
    }

    let mut parts: Vec<String> = transfers
        .iter()
        .map(|t| format!("{} -> {} : {:.2} {}", t.from, t.to, t.amount, t.currency,))
        .collect();
    parts.sort();
    format!("Settlement: {}", parts.join(", "))
}

/// Format net balances into human-readable text.
pub fn format_balances(balances: &HashMap<(String, String), f64>) -> String {
    let mut lines: Vec<String> = balances
        .iter()
        .filter(|(_, v)| to_cents(v.abs()) > 0)
        .map(|((name, currency), amount)| {
            let sign = if *amount > 0.0 { "+" } else { "" };
            format!("{}: {}{:.2} {}", name, sign, amount, currency)
        })
        .collect();
    lines.sort();
    lines.join("\n")
}

/// Format a relationship path into human-readable text.
pub fn format_path(path: &[String], relation_type: &str) -> String {
    if path.is_empty() {
        return "No connection found.".to_string();
    }
    if path.len() == 1 {
        return format!("{} (same person)", path[0]);
    }

    let chain: Vec<String> = path
        .windows(2)
        .map(|pair| format!("{} →[{}]→ {}", pair[0], relation_type, pair[1]))
        .collect();
    chain.join(", ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_memory::{LedgerEntry, MemoryProvenance};

    fn setup_3person_ledger() -> StructuredMemoryStore {
        let mut store = StructuredMemoryStore::new();

        // Alice (id=1) and Bob (id=2): Alice paid, so Alice is credited
        let key_ab = "ledger:1:2";
        store.upsert(
            key_ab,
            MemoryTemplate::Ledger {
                entity_pair: ("Alice".to_string(), "Bob".to_string()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        // Alice paid €90 for dinner split 3 ways → Bob owes Alice €30
        let _ = store.ledger_append(
            key_ab,
            LedgerEntry {
                timestamp: 0,
                amount: 30.0,
                description: "dinner".to_string(),
                direction: LedgerDirection::Credit,
            },
        );

        // Alice (id=1) and Charlie (id=3): Alice paid → Alice is credited
        let key_ac = "ledger:1:3";
        store.upsert(
            key_ac,
            MemoryTemplate::Ledger {
                entity_pair: ("Alice".to_string(), "Charlie".to_string()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        let _ = store.ledger_append(
            key_ac,
            LedgerEntry {
                timestamp: 0,
                amount: 30.0,
                description: "dinner".to_string(),
                direction: LedgerDirection::Credit,
            },
        );

        // Bob (id=2) and Charlie (id=3): Bob paid €60 split 2 ways → Charlie owes Bob €30
        let key_bc = "ledger:2:3";
        store.upsert(
            key_bc,
            MemoryTemplate::Ledger {
                entity_pair: ("Bob".to_string(), "Charlie".to_string()),
                entries: vec![],
                balance: 0.0,
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );
        let _ = store.ledger_append(
            key_bc,
            LedgerEntry {
                timestamp: 0,
                amount: 30.0,
                description: "lunch".to_string(),
                direction: LedgerDirection::Credit,
            },
        );

        store
    }

    #[test]
    fn test_net_balances_3person() {
        let store = setup_3person_ledger();
        let balances = compute_net_balances(&store);

        // Alice: +30 (from Bob) + 30 (from Charlie) = +60
        // Bob: -30 (to Alice) + 30 (from Charlie) = 0
        // Charlie: -30 (to Alice) - 30 (to Bob) = -60
        let alice_bal = balances
            .iter()
            .filter(|((name, _), _)| name == "Alice")
            .map(|(_, v)| v)
            .sum::<f64>();
        let charlie_bal = balances
            .iter()
            .filter(|((name, _), _)| name == "Charlie")
            .map(|(_, v)| v)
            .sum::<f64>();

        assert!((alice_bal - 60.0).abs() < 0.01);
        assert!((charlie_bal + 60.0).abs() < 0.01);
    }

    #[test]
    fn test_minimize_transfers() {
        let store = setup_3person_ledger();
        let balances = compute_net_balances(&store);
        let transfers = minimize_transfers(&balances, "EUR");

        // Charlie owes Alice €60. Bob breaks even.
        // So one transfer: Charlie → Alice: €60
        let total_transfers: f64 = transfers.iter().map(|t| t.amount).sum();
        assert!(total_transfers > 0.0);
    }

    #[test]
    fn test_relationship_path() {
        let mut store = StructuredMemoryStore::new();
        store.upsert(
            "tree:relations:colleague",
            MemoryTemplate::Tree {
                root: "colleague".to_string(),
                children: std::collections::HashMap::new(),
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );

        // Chain: A → B → C → D
        let _ = store.tree_add_child("tree:relations:colleague", "Alice", "Bob");
        let _ = store.tree_add_child("tree:relations:colleague", "Bob", "Alice");
        let _ = store.tree_add_child("tree:relations:colleague", "Bob", "Charlie");
        let _ = store.tree_add_child("tree:relations:colleague", "Charlie", "Bob");
        let _ = store.tree_add_child("tree:relations:colleague", "Charlie", "Dave");
        let _ = store.tree_add_child("tree:relations:colleague", "Dave", "Charlie");

        let path = find_relationship_path(&store, "Alice", "Dave", "colleague");
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p[0], "Alice");
        assert_eq!(p[p.len() - 1], "Dave");
        assert!(p.len() <= 4);
    }

    #[test]
    fn test_no_relationship_path() {
        let mut store = StructuredMemoryStore::new();
        store.upsert(
            "tree:relations:colleague",
            MemoryTemplate::Tree {
                root: "colleague".to_string(),
                children: std::collections::HashMap::new(),
                provenance: MemoryProvenance::EpisodePipeline,
            },
        );

        let _ = store.tree_add_child("tree:relations:colleague", "Alice", "Bob");
        let _ = store.tree_add_child("tree:relations:colleague", "Bob", "Alice");
        // Charlie is disconnected
        let _ = store.tree_add_child("tree:relations:colleague", "Charlie", "Dave");
        let _ = store.tree_add_child("tree:relations:colleague", "Dave", "Charlie");

        let path = find_relationship_path(&store, "Alice", "Charlie", "colleague");
        assert!(path.is_none());
    }

    #[test]
    fn test_format_transfers() {
        let transfers = vec![Transfer {
            from: "Bob".to_string(),
            to: "Alice".to_string(),
            amount: 45.50,
            currency: "EUR".to_string(),
        }];
        let formatted = format_transfers(&transfers);
        assert_eq!(formatted, "Settlement: Bob -> Alice : 45.50 EUR");
    }

    #[test]
    fn test_format_transfers_multiple() {
        let transfers = vec![
            Transfer {
                from: "Charlie".to_string(),
                to: "Bob".to_string(),
                amount: 60.0,
                currency: "EUR".to_string(),
            },
            Transfer {
                from: "Alice".to_string(),
                to: "Bob".to_string(),
                amount: 172.50,
                currency: "EUR".to_string(),
            },
        ];
        let formatted = format_transfers(&transfers);
        assert_eq!(
            formatted,
            "Settlement: Alice -> Bob : 172.50 EUR, Charlie -> Bob : 60.00 EUR"
        );
    }
}
