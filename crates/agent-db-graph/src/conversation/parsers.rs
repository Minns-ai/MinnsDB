//! Category-specific parsers that extract structured data from classified messages.
//!
//! Each parser takes the raw content string and returns the appropriate parsed data.

use super::types::*;

// ---------------------------------------------------------------------------
// Transaction parser
// ---------------------------------------------------------------------------

/// Parse a transaction message.
///
/// Handles formats:
/// - "Alice: Paid €179 for museum - split with Bob"
/// - "Alice: Paid €146 for groceries - split among all"
/// - "Bob: Refund €27 each for all"
/// - "Charlie tipped $10 at the restaurant"
pub fn parse_transaction(
    content: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let lower = content.to_lowercase();

    // Try "Name: Paid/Refund €/$/£Amount ..." format first (benchmark primary terse)
    if let Some(data) = try_colon_paid_format(content, &lower, known_participants) {
        return Some(data);
    }

    // Try verbose "Name: I paid/covered/bought/got ..." format
    if let Some(data) = try_verbose_speaker_format(content, &lower, known_participants) {
        return Some(data);
    }

    // Try "Name: Description was/cost/were €Amount" format
    if let Some(data) = try_amount_was_cost_format(content, &lower, known_participants) {
        return Some(data);
    }

    // Try refund format: "Name: ... refund ..."
    if let Some(data) = try_got_refund_format(content, &lower, known_participants) {
        return Some(data);
    }

    // Try cancellation format: "Name: I need to cancel..."
    if let Some(data) = try_cancellation_format(content, &lower, known_participants) {
        return Some(data);
    }

    // Try "Name paid $Amount for ..." format (no colon)
    if let Some(data) = try_name_paid_format(content, &lower, known_participants) {
        return Some(data);
    }

    // Try "Name tipped $Amount" format
    if let Some(data) = try_tipped_format(content, &lower) {
        return Some(data);
    }

    // Last resort: any message with a colon-separated speaker + currency amount
    if let Some(data) = try_generic_speaker_amount(content, &lower, known_participants) {
        return Some(data);
    }

    None
}

/// Parse "Name: Paid €X for Y - split with/among Z" format.
fn try_colon_paid_format(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let colon_pos = lower.find(':')?;
    let payer = original[..colon_pos].trim().to_string();
    let rest = original[colon_pos + 1..].trim();
    let rest_lower = rest.to_lowercase();

    // Detect kind
    let is_refund = rest_lower.starts_with("refund");

    if !rest_lower.starts_with("paid") && !is_refund {
        return None;
    }

    // Extract amount and currency
    let (amount, currency) = extract_currency_amount(rest)?;

    // Extract description: text between "for" and "- split"
    let description = extract_description(rest);

    // Detect split mode and beneficiaries
    let (beneficiaries, split_mode, scope) =
        extract_split_info(rest, &payer, known_participants, is_refund);

    // For "each" refunds, the stated amount is per-person. Multiply by
    // beneficiary count so that the bridge's division (amount / N) yields
    // the correct per-person share.
    let final_amount = if is_refund && rest.to_lowercase().contains("each") {
        amount * beneficiaries.len().max(1) as f64
    } else {
        amount
    };

    let kind = if is_refund {
        TransactionKind::Reimbursement
    } else {
        TransactionKind::Payment
    };

    Some(TransactionData {
        payer,
        beneficiaries,
        amount: final_amount,
        currency,
        description,
        split_mode,
        kind,
        participants_scope: scope,
    })
}

/// Parse "Name paid $Amount for Y" format (no colon prefix).
fn try_name_paid_format(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let paid_pos = lower.find(" paid ")?;
    let payer = original[..paid_pos].trim().to_string();

    // Verify payer looks like a name (starts with uppercase)
    if !payer.starts_with(|c: char| c.is_uppercase()) {
        return None;
    }

    let rest = &original[paid_pos + 6..]; // skip " paid "
    let (amount, currency) = extract_currency_amount(rest)?;
    let description = extract_description(rest);
    let (beneficiaries, split_mode, scope) =
        extract_split_info(rest, &payer, known_participants, false);

    Some(TransactionData {
        payer,
        beneficiaries,
        amount,
        currency,
        description,
        split_mode,
        kind: TransactionKind::Payment,
        participants_scope: scope,
    })
}

/// Parse "Name tipped $Amount" format.
fn try_tipped_format(original: &str, lower: &str) -> Option<TransactionData> {
    let tip_pos = lower.find(" tipped ")?;
    let payer = original[..tip_pos].trim().to_string();
    let rest = &original[tip_pos + 8..];
    let (amount, currency) = extract_currency_amount(rest)?;

    let description = if let Some(at_pos) = lower.find(" at ") {
        original[at_pos + 4..]
            .trim_end_matches('.')
            .trim()
            .to_string()
    } else {
        "tip".to_string()
    };

    Some(TransactionData {
        payer: payer.clone(),
        beneficiaries: vec![payer],
        amount,
        currency,
        description,
        split_mode: SplitMode::SoleBeneficiary,
        kind: TransactionKind::Tip,
        participants_scope: ParticipantsScope::Explicit,
    })
}

/// Parse verbose "Name: I paid/covered/bought/got for X: €Amount [total]. Split/Shared ..."
fn try_verbose_speaker_format(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let colon_pos = lower.find(':')?;
    let payer = original[..colon_pos].trim().to_string();
    if !payer.starts_with(|c: char| c.is_uppercase()) || payer.split_whitespace().count() > 2 {
        return None;
    }
    let rest = original[colon_pos + 1..].trim();
    let rest_lower = rest.to_lowercase();

    // Must contain a payment verb
    let verb_patterns = ["i paid", "i covered", "i bought", "i purchased", "i got"];
    let has_verb = verb_patterns.iter().any(|p| rest_lower.starts_with(p));
    if !has_verb {
        return None;
    }

    // Extract amount and currency
    let (amount, currency) = extract_currency_amount(rest)?;

    // Extract description
    let description = extract_verbose_description(rest);

    // Check for "owe me" pattern to extract explicit beneficiaries and amounts
    if let Some(owe_data) = try_owe_me_in_text(rest, &payer, known_participants) {
        return Some(owe_data);
    }

    // Detect split mode and beneficiaries
    let (beneficiaries, split_mode, scope) =
        extract_split_info_verbose(rest, &payer, known_participants);

    Some(TransactionData {
        payer,
        beneficiaries,
        amount,
        currency,
        description,
        split_mode,
        kind: TransactionKind::Payment,
        participants_scope: scope,
    })
}

/// Parse "Name: Description was/cost/were €Amount [total]. [Split clause]"
fn try_amount_was_cost_format(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let colon_pos = lower.find(':')?;
    let payer = original[..colon_pos].trim().to_string();
    if !payer.starts_with(|c: char| c.is_uppercase()) || payer.split_whitespace().count() > 2 {
        return None;
    }
    let rest = original[colon_pos + 1..].trim();
    let rest_lower = rest.to_lowercase();

    // Must have "was €X" or "cost €X" or "cost me €X" or "were €X"
    let has_was_cost = rest_lower.contains(" was ")
        || rest_lower.contains(" cost ")
        || rest_lower.contains(" were ");
    if !has_was_cost {
        return None;
    }

    let (amount, currency) = extract_currency_amount(rest)?;
    let description = extract_verbose_description(rest);

    // Check for "owe me" pattern
    if let Some(owe_data) = try_owe_me_in_text(rest, &payer, known_participants) {
        return Some(owe_data);
    }

    let (beneficiaries, split_mode, scope) =
        extract_split_info_verbose(rest, &payer, known_participants);

    Some(TransactionData {
        payer,
        beneficiaries,
        amount,
        currency,
        description,
        split_mode,
        kind: TransactionKind::Payment,
        participants_scope: scope,
    })
}

/// Parse "Name: I got a €X refund ..." / "Name: ... gave us a €X refund ..."
fn try_got_refund_format(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    if !lower.contains("refund") {
        return None;
    }
    let colon_pos = lower.find(':')?;
    let payer = original[..colon_pos].trim().to_string();
    if !payer.starts_with(|c: char| c.is_uppercase()) || payer.split_whitespace().count() > 2 {
        return None;
    }
    let rest = original[colon_pos + 1..].trim();

    let (amount, currency) = extract_currency_amount(rest)?;
    let description = extract_verbose_description(rest);

    let (beneficiaries, _split_mode, scope) =
        extract_split_info_verbose(rest, &payer, known_participants);

    // Keep payer in beneficiaries — the bridge skips payer during
    // entry creation and needs the full count for correct division.
    Some(TransactionData {
        payer,
        beneficiaries,
        amount,
        currency,
        description,
        split_mode: SplitMode::Equal,
        kind: TransactionKind::Reimbursement,
        participants_scope: scope,
    })
}

/// Parse "Name: I need to cancel the X payment of €Y for Z"
fn try_cancellation_format(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    if !lower.contains("cancel") {
        return None;
    }
    let colon_pos = lower.find(':')?;
    let payer = original[..colon_pos].trim().to_string();
    if !payer.starts_with(|c: char| c.is_uppercase()) || payer.split_whitespace().count() > 2 {
        return None;
    }
    let rest = original[colon_pos + 1..].trim();

    let (amount, currency) = extract_currency_amount(rest)?;
    let description = extract_verbose_description(rest);

    let (beneficiaries, _split_mode, scope) =
        extract_split_info_verbose(rest, &payer, known_participants);

    Some(TransactionData {
        payer,
        beneficiaries,
        amount,
        currency,
        description,
        split_mode: SplitMode::Equal,
        kind: TransactionKind::Reimbursement,
        participants_scope: scope,
    })
}

/// Generic fallback: any "Name: ... €Amount ... split/shared/for all ..." pattern
fn try_generic_speaker_amount(
    original: &str,
    lower: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let colon_pos = lower.find(':')?;
    let payer = original[..colon_pos].trim().to_string();
    if !payer.starts_with(|c: char| c.is_uppercase()) || payer.split_whitespace().count() > 2 {
        return None;
    }
    let rest = original[colon_pos + 1..].trim();
    let rest_lower = rest.to_lowercase();

    let (amount, currency) = extract_currency_amount(rest)?;

    // Must have a strong split/shared signal — require explicit split/shared language
    let has_split_signal = rest_lower.contains("split")
        || rest_lower.contains("shared")
        || rest_lower.contains("owe me")
        || rest_lower.contains("owes me")
        || rest_lower.contains("three ways")
        || rest_lower.contains("equally");
    if !has_split_signal {
        return None;
    }

    // Check for "owe me" pattern
    if let Some(owe_data) = try_owe_me_in_text(rest, &payer, known_participants) {
        return Some(owe_data);
    }

    let description = extract_verbose_description(rest);
    let (beneficiaries, split_mode, scope) =
        extract_split_info_verbose(rest, &payer, known_participants);

    Some(TransactionData {
        payer,
        beneficiaries,
        amount,
        currency,
        description,
        split_mode,
        kind: TransactionKind::Payment,
        participants_scope: scope,
    })
}

/// Extract "X and Y owe me €Z each" from text, returning a TransactionData.
fn try_owe_me_in_text(
    text: &str,
    payer: &str,
    known_participants: &std::collections::HashSet<String>,
) -> Option<TransactionData> {
    let lower = text.to_lowercase();
    let owe_pos = lower.find("owe me").or_else(|| lower.find("owes me"))?;

    // Try to extract amount from after "owe me" first (more specific)
    let after_owe = &text[owe_pos..];
    let after_owe_lower = &lower[owe_pos..];
    let is_each = after_owe_lower.contains("each");

    // First try amount from the "owe me" clause specifically
    let (amount, currency) =
        extract_currency_amount(after_owe).or_else(|| extract_currency_amount(text))?;

    // Extract names immediately before "owe me" in the same sentence.
    // Find the start of the sentence containing "owe me"
    let sentence_start = text[..owe_pos].rfind(". ").map(|p| p + 2).unwrap_or(0);
    let before_owe_sentence = text[sentence_start..owe_pos].trim();

    // First, try to find known participants in the sentence before "owe me"
    let mut beneficiaries: Vec<String> = known_participants
        .iter()
        .filter(|p| {
            *p != payer
                && before_owe_sentence
                    .to_lowercase()
                    .contains(&p.to_lowercase())
        })
        .cloned()
        .collect();

    // If no known participants found, extract capitalized names
    if beneficiaries.is_empty() {
        beneficiaries = extract_names_from_text(before_owe_sentence)
            .into_iter()
            .filter(|n| known_participants.contains(n))
            .collect();
    }

    // Fallback: check the whole text for known participants
    if beneficiaries.is_empty() {
        for p in known_participants {
            if p != payer && lower.contains(&p.to_lowercase()) {
                beneficiaries.push(p.clone());
            }
        }
    }

    if beneficiaries.is_empty() {
        return None;
    }

    let description = extract_verbose_description(text);

    if is_each {
        // "each" means the amount is per person
        // "Alice and Charlie owe me €18 each" means payer paid €54 total (3 people × €18)
        let num = beneficiaries.len() as f64;
        let total_amount = amount * (num + 1.0);
        let mut all_beneficiaries = beneficiaries.clone();
        all_beneficiaries.push(payer.to_string());
        all_beneficiaries.sort();
        return Some(TransactionData {
            payer: payer.to_string(),
            beneficiaries: all_beneficiaries,
            amount: total_amount,
            currency,
            description,
            split_mode: SplitMode::Equal,
            kind: TransactionKind::Payment,
            participants_scope: ParticipantsScope::Explicit,
        });
    }

    // No "each" — this is "X owes me €Z" for total, split among named
    let mut all_beneficiaries = beneficiaries;
    all_beneficiaries.push(payer.to_string());
    all_beneficiaries.sort();
    Some(TransactionData {
        payer: payer.to_string(),
        beneficiaries: all_beneficiaries,
        amount,
        currency,
        description,
        split_mode: SplitMode::Equal,
        kind: TransactionKind::Payment,
        participants_scope: ParticipantsScope::Explicit,
    })
}

/// Extract split info for verbose message formats.
/// Handles: "split equally", "shared between", "split three ways", "for all three of us",
/// "for our group", "for everyone", "Shared equally", "total for all", etc.
fn extract_split_info_verbose(
    text: &str,
    payer: &str,
    known_participants: &std::collections::HashSet<String>,
) -> (Vec<String>, SplitMode, ParticipantsScope) {
    let lower = text.to_lowercase();

    // "for all three of us" / "for our group of three" / "for our group" / "for everyone"
    // / "among all three" / "among the three of us" / "for all" / "split three ways"
    // / "shared between all three" / "three seats" / "all three" / "for 3 nights"
    // / "for weekly passes for all" / "daily passes for all"
    if lower.contains("for all three")
        || lower.contains("for our group")
        || lower.contains("for everyone")
        || lower.contains("among all three")
        || lower.contains("among the three")
        || lower.contains("split three ways")
        || lower.contains("shared between all three")
        || lower.contains("three seats")
        || lower.contains("for all of us")
        || lower.contains("equally among the three")
        || lower.contains("for our")
        || lower.contains("passes for all")
        || lower.contains("among all")
        || lower.contains("for all")
    {
        let mut all: Vec<String> = known_participants.iter().cloned().collect();
        if !all.contains(&payer.to_string()) {
            all.push(payer.to_string());
        }
        all.sort();
        return (all, SplitMode::Equal, ParticipantsScope::EveryoneKnown);
    }

    // "Shared equally" / "Split equally" without explicit names → everyone
    if lower.contains("shared equally") || lower.contains("split equally") {
        let mut all: Vec<String> = known_participants.iter().cloned().collect();
        if !all.contains(&payer.to_string()) {
            all.push(payer.to_string());
        }
        all.sort();
        return (all, SplitMode::Equal, ParticipantsScope::EveryoneKnown);
    }

    // "Shared between X and me" / "Shared between X and Y"
    if let Some(shared_pos) = lower.find("shared between") {
        let after = &text[shared_pos + "shared between".len()..];
        let mut names = extract_names_from_text(after);
        let after_lower = after.to_lowercase();
        if (after_lower.contains(" me") || after_lower.contains(" and me"))
            && !names.contains(&payer.to_string())
        {
            names.push(payer.to_string());
        }
        if !names.contains(&payer.to_string()) {
            names.push(payer.to_string());
        }
        names.sort();
        names.dedup();
        return (names, SplitMode::Equal, ParticipantsScope::Explicit);
    }

    // "split with X" / "split between X and Y"
    if let Some(split_pos) = lower.find("split with").or(lower.find("split between")) {
        let after = &text[split_pos..];
        let mut names = extract_names_from_text(after);
        if !names.contains(&payer.to_string()) {
            names.push(payer.to_string());
        }
        names.sort();
        return (names, SplitMode::Equal, ParticipantsScope::Explicit);
    }

    // "X and Y owe me their shares" — names before "owe me"
    if let Some(owe_pos) = lower.find("owe me") {
        let before = &text[..owe_pos];
        let mut names = extract_names_from_text(before);
        if !names.contains(&payer.to_string()) {
            names.push(payer.to_string());
        }
        names.sort();
        return (names, SplitMode::Equal, ParticipantsScope::Explicit);
    }

    // "for Name and Name" / "for Name, Name and Name" — explicit named beneficiaries
    if let Some(for_pos) = lower.find(" for ") {
        let after_for = &text[for_pos + 5..];
        let names: Vec<String> = extract_names_from_text(after_for)
            .into_iter()
            .filter(|n| known_participants.contains(n))
            .collect();
        if !names.is_empty() {
            let mut result = names;
            if !result.contains(&payer.to_string()) {
                result.push(payer.to_string());
            }
            result.sort();
            result.dedup();
            return (result, SplitMode::Equal, ParticipantsScope::Explicit);
        }
    }

    // Default for verbose: if we found amount but no split clause, assume everyone
    // (Most verbose benchmark messages imply "for the group")
    let mut all: Vec<String> = known_participants.iter().cloned().collect();
    if !all.contains(&payer.to_string()) {
        all.push(payer.to_string());
    }
    all.sort();
    if all.len() > 1 {
        (all, SplitMode::Equal, ParticipantsScope::EveryoneKnown)
    } else {
        (
            vec![payer.to_string()],
            SplitMode::SoleBeneficiary,
            ParticipantsScope::Explicit,
        )
    }
}

/// Extract description from verbose payment text.
fn extract_verbose_description(text: &str) -> String {
    let lower = text.to_lowercase();

    // Try "for X:" or "for the X" patterns
    if let Some(for_pos) = lower.find(" for ") {
        let after_for = &text[for_pos + 5..];
        // Truncate at split clause, amount, or sentence end
        let terminators = [
            " - split", " split", ". split", " shared", ". shared", " owe me", " among", " equally",
        ];
        let mut end = after_for.len();
        for t in &terminators {
            if let Some(p) = after_for.to_lowercase().find(t) {
                end = end.min(p);
            }
        }
        // Also truncate at currency symbols
        for sym in &['€', '$', '£', '¥'] {
            if let Some(p) = after_for.find(*sym) {
                end = end.min(p);
            }
        }
        let desc = after_for[..end]
            .trim()
            .trim_end_matches(':')
            .trim_end_matches('.')
            .trim_end_matches(',')
            .trim();
        if !desc.is_empty() {
            return desc.to_string();
        }
    }

    // Try extracting from "Description was/cost/were €X"
    for marker in &[" was ", " cost ", " were "] {
        if let Some(pos) = lower.find(marker) {
            let before = &text[..pos].trim();
            // Remove "The" prefix and "I" prefix
            let cleaned = before
                .trim_start_matches("The ")
                .trim_start_matches("the ")
                .trim_start_matches("I ")
                .trim();
            if !cleaned.is_empty() && cleaned.len() > 1 {
                return cleaned.to_string();
            }
        }
    }

    extract_description(text)
}

/// Extract currency symbol and numeric amount from text.
fn extract_currency_amount(text: &str) -> Option<(f64, String)> {
    let symbols = [('€', "EUR"), ('$', "USD"), ('£', "GBP"), ('¥', "JPY")];

    for (sym, curr) in &symbols {
        if let Some(pos) = text.find(*sym) {
            let after = &text[pos + sym.len_utf8()..];
            if let Some(amount) = parse_leading_number(after) {
                return Some((amount, curr.to_string()));
            }
        }
    }

    // Try plain number pattern with currency word
    let words: Vec<&str> = text.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if let Ok(amount) = word.replace(',', "").parse::<f64>() {
            // Check next word for currency
            let currency = if i + 1 < words.len() {
                match words[i + 1].to_lowercase().as_str() {
                    "usd" | "dollars" => "USD",
                    "eur" | "euros" => "EUR",
                    "gbp" | "pounds" => "GBP",
                    _ => "USD",
                }
            } else {
                "USD"
            };
            return Some((amount, currency.to_string()));
        }
    }

    None
}

/// Parse a leading number from text (e.g., "179 for museum" → 179.0).
fn parse_leading_number(text: &str) -> Option<f64> {
    let num_str: String = text
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect();
    if num_str.is_empty() {
        return None;
    }
    num_str.replace(',', "").parse::<f64>().ok()
}

/// Extract description from payment text.
fn extract_description(text: &str) -> String {
    let lower = text.to_lowercase();

    // Look for "for <desc>" before any split clause
    if let Some(for_pos) = lower.find(" for ") {
        let after_for = &text[for_pos + 5..];
        // Truncate at " - split" or end
        let end = after_for
            .to_lowercase()
            .find(" - split")
            .or_else(|| after_for.to_lowercase().find(" split"))
            .unwrap_or(after_for.len());
        let desc = after_for[..end].trim();
        if !desc.is_empty() {
            return desc.trim_end_matches('.').to_string();
        }
    }

    // Fallback: the word after the amount
    "expense".to_string()
}

/// Extract split info: beneficiaries, split mode, and participant scope.
fn extract_split_info(
    text: &str,
    payer: &str,
    known_participants: &std::collections::HashSet<String>,
    _is_refund: bool,
) -> (Vec<String>, SplitMode, ParticipantsScope) {
    let lower = text.to_lowercase();

    // "each for all" / "among all" / "for everyone" / "for all" → all known participants
    if lower.contains("among all")
        || lower.contains("for all")
        || lower.contains("for everyone")
        || lower.contains("each for all")
    {
        let mut all: Vec<String> = known_participants.iter().cloned().collect();
        if !all.contains(&payer.to_string()) {
            all.push(payer.to_string());
        }
        all.sort();

        // Include payer in beneficiaries — the bridge skips payer during
        // ledger entry creation. Keeping payer ensures correct per-person
        // division (e.g. "split among all" = N people, not N-1).
        return (all, SplitMode::Equal, ParticipantsScope::EveryoneKnown);
    }

    // "split with/between X and Y" or "split with X"
    if let Some(split_pos) = lower.find("split with").or(lower.find("split between")) {
        let after_split = &text[split_pos..];
        let names = extract_names_from_text(after_split);
        let mut beneficiaries = names;
        // Add payer if not in beneficiaries (they share the split)
        if !beneficiaries.contains(&payer.to_string()) {
            beneficiaries.push(payer.to_string());
        }
        beneficiaries.sort();
        return (beneficiaries, SplitMode::Equal, ParticipantsScope::Explicit);
    }

    // "for Name's thing" (possessive = sole beneficiary)
    if let Some(for_pos) = lower.find(" for ") {
        let after_for = &text[for_pos + 5..];
        if after_for.contains("'s ") || after_for.contains("\u{2019}s ") {
            let names = extract_names_from_text(after_for);
            if let Some(beneficiary) = names.into_iter().next() {
                return (
                    vec![beneficiary],
                    SplitMode::SoleBeneficiary,
                    ParticipantsScope::Explicit,
                );
            }
        }
    }

    // "for Name and Name" / "for Name, Name and Name" — explicit named beneficiaries
    if let Some(for_pos) = lower.find(" for ") {
        let after_for = &text[for_pos + 5..];
        let names: Vec<String> = extract_names_from_text(after_for)
            .into_iter()
            .filter(|n| known_participants.contains(n))
            .collect();
        if !names.is_empty() {
            let mut beneficiaries = names;
            if !beneficiaries.contains(&payer.to_string()) {
                beneficiaries.push(payer.to_string());
            }
            beneficiaries.sort();
            return (beneficiaries, SplitMode::Equal, ParticipantsScope::Explicit);
        }
    }

    // Default: payer is sole beneficiary
    (
        vec![payer.to_string()],
        SplitMode::SoleBeneficiary,
        ParticipantsScope::Explicit,
    )
}

/// Extract capitalized names from text.
pub fn extract_names_from_text(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    let skip_words = [
        "Paid", "Refund", "The", "This", "That", "I", "My", "It", "A", "An", "For", "With", "And",
        "Or", "But", "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Do", "Does", "Did", "In",
        "On", "At", "To", "From", "By", "Of", "About", "Split", "Each", "All", "Between", "Among",
    ];

    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
        if word.is_empty() || !word.starts_with(|c: char| c.is_uppercase()) {
            i += 1;
            continue;
        }
        if skip_words.contains(&word) {
            i += 1;
            continue;
        }

        // Try two-word name first
        if i + 1 < words.len() {
            let next = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric());
            if !next.is_empty()
                && next.starts_with(|c: char| c.is_uppercase())
                && !skip_words.contains(&next)
            {
                names.push(format!("{} {}", word, next));
                i += 2;
                continue;
            }
        }

        // Single-word name
        names.push(word.to_string());
        i += 1;
    }

    names
}

// ---------------------------------------------------------------------------
// State change parser
// ---------------------------------------------------------------------------

/// Parse a state change / location / routine message.
pub fn parse_state_change(content: &str) -> Vec<StateChangeData> {
    let mut changes = Vec::new();
    let lower = content.to_lowercase();

    // "I live in X" / "I live near X"
    if lower.contains("i live in") {
        if let Some(location) = extract_location_after(content, "live in") {
            changes.push(StateChangeData {
                entity: "user".to_string(),
                attribute: "location".to_string(),
                new_value: location,
                old_value: None,
            });
        }
    }

    // "I'm moving to X" / "moving to X"
    if lower.contains("moving to") || lower.contains("relocated to") {
        let pattern = if lower.contains("moving to") {
            "moving to"
        } else {
            "relocated to"
        };
        if let Some(location) = extract_location_after(content, pattern) {
            changes.push(StateChangeData {
                entity: "user".to_string(),
                attribute: "location".to_string(),
                new_value: location,
                old_value: None,
            });
        }
    }

    // "I live near X" — landmark
    if lower.contains("near ") || lower.contains("close to ") {
        let pattern = if lower.contains("near ") {
            "near "
        } else {
            "close to "
        };
        if let Some(pos) = lower.find(pattern) {
            let after = &content[pos + pattern.len()..];
            let landmark = extract_proper_noun_phrase(after);
            if !landmark.is_empty() {
                changes.push(StateChangeData {
                    entity: "user".to_string(),
                    attribute: "landmark".to_string(),
                    new_value: landmark,
                    old_value: None,
                });
            }
        }
    }

    // Routine: "every morning/evening/day/week"
    if lower.contains("every morning")
        || lower.contains("every evening")
        || lower.contains("every day")
        || lower.contains("every week")
    {
        let slot = if lower.contains("every morning") {
            "morning"
        } else if lower.contains("every evening") {
            "evening"
        } else if lower.contains("every day") {
            "daily"
        } else {
            "weekly"
        };
        changes.push(StateChangeData {
            entity: "user".to_string(),
            attribute: format!("routine:{}", slot),
            new_value: content.to_string(),
            old_value: None,
        });
    }

    // "I found an apartment near/in X"
    if lower.contains("found an apartment") || lower.contains("found a place") {
        if let Some(location) = extract_location_after(content, "near")
            .or_else(|| extract_location_after(content, "in"))
        {
            changes.push(StateChangeData {
                entity: "user".to_string(),
                attribute: "landmark".to_string(),
                new_value: location,
                old_value: None,
            });
        }
    }

    // "I enjoy/love <activity> from/at <place>"
    if (lower.contains("i enjoy") || lower.contains("i love"))
        && (lower.contains("from ") || lower.contains("at "))
    {
        changes.push(StateChangeData {
            entity: "user".to_string(),
            attribute: "activity".to_string(),
            new_value: content.to_string(),
            old_value: None,
        });
    }

    // "enjoy coffee at X"
    if lower.contains("enjoy coffee at") || lower.contains("enjoy tea at") {
        changes.push(StateChangeData {
            entity: "user".to_string(),
            attribute: "activity".to_string(),
            new_value: content.to_string(),
            old_value: None,
        });
    }

    // "Saturday morning walks to X"
    if lower.contains("saturday") || lower.contains("sunday") {
        let day = if lower.contains("saturday") {
            "saturday"
        } else {
            "sunday"
        };
        changes.push(StateChangeData {
            entity: "user".to_string(),
            attribute: format!("routine:{}", day),
            new_value: content.to_string(),
            old_value: None,
        });
    }

    changes
}

/// Extract a location string after a pattern like "live in" or "moving to".
fn extract_location_after(content: &str, pattern: &str) -> Option<String> {
    let lower = content.to_lowercase();
    let pos = lower.find(&pattern.to_lowercase())?;
    let after = &content[pos + pattern.len()..].trim_start();

    // Take text until a comma, period, dash, "for", or end
    let end_markers = [" for ", " - ", ". ", ", and", " and I"];
    let mut end = after.len();
    for marker in &end_markers {
        if let Some(p) = after.find(marker) {
            end = end.min(p);
        }
    }

    let location = after[..end]
        .trim()
        .trim_end_matches('.')
        .trim_end_matches(',');
    if location.is_empty() {
        return None;
    }

    Some(location.to_string())
}

/// Extract a proper noun phrase (capitalized words).
fn extract_proper_noun_phrase(text: &str) -> String {
    let text = text.trim_start_matches(|c: char| !c.is_alphanumeric());
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut phrase = Vec::new();

    for word in &words {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        if clean.is_empty() {
            break;
        }
        // Allow "de", "da", "do", "of", "the" as connectors in place names
        let connectors = ["de", "da", "do", "of", "the", "del", "di"];
        if connectors.contains(&clean.to_lowercase().as_str()) {
            phrase.push(clean);
            continue;
        }
        if clean.starts_with(|c: char| c.is_uppercase()) {
            phrase.push(clean);
        } else if !phrase.is_empty() {
            break;
        }
    }

    phrase.join(" ")
}

// ---------------------------------------------------------------------------
// Relationship parser
// ---------------------------------------------------------------------------

/// Parse a relationship message.
pub fn parse_relationship(content: &str) -> Option<RelationshipData> {
    let lower = content.to_lowercase();

    // "X works with Y"
    if let Some(pos) = lower.find("works with") {
        let subject = extract_name_before(content, pos);
        let object = extract_name_after(content, pos + "works with".len());
        if !subject.is_empty() && !object.is_empty() {
            return Some(RelationshipData {
                subject,
                object,
                relation_type: "colleague".to_string(),
            });
        }
    }

    // "X is a colleague of Y"
    if let Some(pos) = lower.find("colleague of") {
        // Find the subject: everything before "is a colleague of"
        let is_pos = lower[..pos]
            .rfind("is a ")
            .or_else(|| lower[..pos].rfind("is "))?;
        let subject = extract_name_before(content, is_pos);
        let object = extract_name_after(content, pos + "colleague of".len());
        if !subject.is_empty() && !object.is_empty() {
            return Some(RelationshipData {
                subject,
                object,
                relation_type: "colleague".to_string(),
            });
        }
    }

    // "X collaborates with Y"
    if let Some(pos) = lower.find("collaborates with") {
        let subject = extract_name_before(content, pos);
        let object = extract_name_after(content, pos + "collaborates with".len());
        if !subject.is_empty() && !object.is_empty() {
            return Some(RelationshipData {
                subject,
                object,
                relation_type: "colleague".to_string(),
            });
        }
    }

    // "X and Y are colleagues/friends"
    if lower.contains(" are colleagues")
        || lower.contains(" are coworkers")
        || lower.contains(" are friends")
    {
        let are_pos = lower
            .find(" are colleagues")
            .or_else(|| lower.find(" are coworkers"))
            .or_else(|| lower.find(" are friends"))?;
        let before = &content[..are_pos];
        let names = extract_names_from_text(before);
        if names.len() >= 2 {
            let relation = if lower.contains("friends") {
                "friend"
            } else {
                "colleague"
            };
            return Some(RelationshipData {
                subject: names[0].clone(),
                object: names[1].clone(),
                relation_type: relation.to_string(),
            });
        }
    }

    None
}

/// Extract a name (1-2 capitalized words) from text before a given position.
fn extract_name_before(text: &str, pos: usize) -> String {
    let before = text[..pos].trim();
    let words: Vec<&str> = before.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }

    // Take last 1-2 capitalized words
    let mut name_words = Vec::new();
    for word in words.iter().rev() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.starts_with(|c: char| c.is_uppercase()) && clean.len() > 1 {
            name_words.push(clean);
            if name_words.len() >= 2 {
                break;
            }
        } else {
            break;
        }
    }
    name_words.reverse();
    name_words.join(" ")
}

/// Extract a name (1-2 capitalized words) from text after a given position.
fn extract_name_after(text: &str, pos: usize) -> String {
    let after = text[pos..].trim();
    let words: Vec<&str> = after.split_whitespace().collect();
    let mut name_words = Vec::new();

    for word in &words {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.is_empty() {
            continue;
        }
        if clean.starts_with(|c: char| c.is_uppercase()) && clean.len() > 1 {
            name_words.push(clean.to_string());
            if name_words.len() >= 2 {
                break;
            }
        } else if !name_words.is_empty() {
            break;
        }
    }

    name_words.join(" ")
}

// ---------------------------------------------------------------------------
// Preference parser
// ---------------------------------------------------------------------------

/// Parse a preference message.
pub fn parse_preference(content: &str, session_topic: Option<&str>) -> Option<PreferenceData> {
    let lower = content.to_lowercase();

    // Determine sentiment from explicit expressions
    let (sentiment, item) = if let Some(rest) = strip_prefix_ci(&lower, "i love ") {
        (1.0, extract_item_from_rest(content, "i love ", rest.len()))
    } else if let Some(rest) = strip_prefix_ci(&lower, "i like ") {
        (0.8, extract_item_from_rest(content, "i like ", rest.len()))
    } else if let Some(rest) = strip_prefix_ci(&lower, "i enjoy ") {
        (0.7, extract_item_from_rest(content, "i enjoy ", rest.len()))
    } else if let Some(rest) = strip_prefix_ci(&lower, "i prefer ") {
        (
            0.8,
            extract_item_from_rest(content, "i prefer ", rest.len()),
        )
    } else if let Some(rest) = strip_prefix_ci(&lower, "i hate ") {
        (0.0, extract_item_from_rest(content, "i hate ", rest.len()))
    } else if let Some(rest) = strip_prefix_ci(&lower, "i dislike ") {
        (
            0.2,
            extract_item_from_rest(content, "i dislike ", rest.len()),
        )
    } else if lower.starts_with("saw ") || lower.contains("visited ") || lower.contains("went to ")
    {
        // Implicit positive from visiting
        let item = extract_visited_item(content);
        (0.6, item)
    } else if session_topic.is_some() {
        // Use session topic as implicit preference with moderate positive
        let topic = session_topic.unwrap();
        // The fact_quote from the session contains the core opinion
        (0.7, topic.to_string())
    } else {
        return None;
    };

    if item.is_empty() {
        return None;
    }

    // Infer category from session topic or item content
    let category = infer_preference_category(&item, session_topic);

    Some(PreferenceData {
        entity: "user".to_string(),
        item,
        category,
        sentiment,
    })
}

/// Strip a case-insensitive prefix and return the rest.
fn strip_prefix_ci<'a>(lower: &'a str, prefix: &str) -> Option<&'a str> {
    // Search within the string (not just at start)
    if let Some(pos) = lower.find(prefix) {
        Some(&lower[pos + prefix.len()..])
    } else {
        None
    }
}

/// Extract item text from original content given a prefix pattern.
fn extract_item_from_rest(original: &str, _prefix: &str, _rest_len: usize) -> String {
    let lower = original.to_lowercase();
    let prefix_lower = _prefix;
    if let Some(pos) = lower.find(prefix_lower) {
        let after = &original[pos + prefix_lower.len()..];
        // Take until period, comma, or dash
        let end = after
            .find('.')
            .or_else(|| after.find(','))
            .or_else(|| after.find(" - "))
            .unwrap_or(after.len());
        after[..end].trim().to_string()
    } else {
        String::new()
    }
}

/// Extract the item being visited/seen.
fn extract_visited_item(content: &str) -> String {
    let lower = content.to_lowercase();
    let patterns = ["saw ", "visited ", "went to "];

    for pat in &patterns {
        if let Some(pos) = lower.find(pat) {
            let after = &content[pos + pat.len()..];
            // Take the proper noun phrase
            let phrase = extract_proper_noun_phrase(after);
            if !phrase.is_empty() {
                return phrase;
            }
            // Fallback: take until period
            let end = after.find('.').unwrap_or(after.len());
            return after[..end].trim().to_string();
        }
    }
    String::new()
}

/// Infer the preference category from item content or session topic.
pub fn infer_preference_category(item: &str, session_topic: Option<&str>) -> String {
    let lower = item.to_lowercase();
    let topic_lower = session_topic.map(|t| t.to_lowercase()).unwrap_or_default();

    // Art-related
    let art_signals = [
        "monet",
        "rodin",
        "michelangelo",
        "cezanne",
        "cézanne",
        "van gogh",
        "picasso",
        "david",
        "water lilies",
        "starry night",
        "sculpture",
        "painting",
        "gallery",
        "museum",
        "canvas",
        "exhibition",
        "fresco",
        "art",
    ];
    if art_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "art".to_string();
    }

    // Music
    let music_signals = [
        "album", "song", "band", "concert", "music", "symphony", "jazz",
    ];
    if music_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "music".to_string();
    }

    // Movies
    let movie_signals = ["movie", "film", "cinema", "director", "actor"];
    if movie_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "movies".to_string();
    }

    // Books
    let book_signals = ["book", "novel", "author", "read", "chapter"];
    if book_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "books".to_string();
    }

    // Food
    let food_signals = [
        "food",
        "restaurant",
        "cafe",
        "bakery",
        "dish",
        "cuisine",
        "coffee",
    ];
    if food_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "food".to_string();
    }

    // Sports
    let sports_signals = [
        "sport",
        "game",
        "team",
        "play",
        "match",
        "board game",
        "boardgame",
    ];
    if sports_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "sports".to_string();
    }

    // TV Series
    let series_signals = ["series", "show", "episode", "season", "tv"];
    if series_signals
        .iter()
        .any(|s| lower.contains(s) || topic_lower.contains(s))
    {
        return "series".to_string();
    }

    "general".to_string()
}

// ---------------------------------------------------------------------------
// Top-level classify+parse entry point
// ---------------------------------------------------------------------------

/// Classify and parse a single message, returning the parsed envelope.
///
/// Pipeline:
/// 1. Run keyword-based classification
/// 2. Attempt keyword-based extraction
pub fn classify_and_parse(
    ctx: &ConversationContext,
    state: &ConversationState,
    content: &str,
    role: &str,
    session_topic: Option<&str>,
) -> ParsedMessage {
    let result = super::classifier::classify(ctx, state, content, role);

    let parsed = match result.category {
        MessageCategory::Transaction => parse_transaction(content, &state.known_participants)
            .map(ParsedPayload::Transaction)
            .unwrap_or_else(|| ParsedPayload::Chitchat(content.to_string())),
        MessageCategory::StateChange => parse_state_change(content)
            .into_iter()
            .next()
            .map(ParsedPayload::StateChange)
            .unwrap_or_else(|| ParsedPayload::Chitchat(content.to_string())),
        MessageCategory::Relationship => parse_relationship(content)
            .map(ParsedPayload::Relationship)
            .unwrap_or_else(|| ParsedPayload::Chitchat(content.to_string())),
        MessageCategory::Preference => parse_preference(content, session_topic)
            .map(ParsedPayload::Preference)
            .unwrap_or_else(|| ParsedPayload::Chitchat(content.to_string())),
        MessageCategory::Chitchat => ParsedPayload::Chitchat(content.to_string()),
    };

    // If parse failed, downgrade category to Chitchat
    let final_category = if matches!(parsed, ParsedPayload::Chitchat(_))
        && result.category != MessageCategory::Chitchat
    {
        MessageCategory::Chitchat
    } else {
        result.category
    };

    ParsedMessage {
        category: final_category,
        original_content: content.to_string(),
        session_id: ctx.session_id.clone(),
        message_index: ctx.message_index,
        confidence: result.confidence,
        parsed,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn participants_abc() -> HashSet<String> {
        let mut set = HashSet::new();
        set.insert("Alice".to_string());
        set.insert("Bob".to_string());
        set.insert("Charlie".to_string());
        set
    }

    // ===== Transaction parser =====

    #[test]
    fn parse_tx_paid_split_with() {
        let known = participants_abc();
        let tx = parse_transaction("Alice: Paid €179 for museum - split with Bob", &known).unwrap();
        assert_eq!(tx.payer, "Alice");
        assert!((tx.amount - 179.0).abs() < 0.01);
        assert_eq!(tx.currency, "EUR");
        assert!(tx.beneficiaries.contains(&"Bob".to_string()));
        assert!(tx.beneficiaries.contains(&"Alice".to_string()));
        assert_eq!(tx.split_mode, SplitMode::Equal);
    }

    #[test]
    fn parse_tx_paid_split_among_all() {
        let known = participants_abc();
        let tx =
            parse_transaction("Alice: Paid €146 for groceries - split among all", &known).unwrap();
        assert_eq!(tx.payer, "Alice");
        assert!((tx.amount - 146.0).abs() < 0.01);
        assert_eq!(tx.participants_scope, ParticipantsScope::EveryoneKnown);
        assert_eq!(tx.beneficiaries.len(), 3);
    }

    #[test]
    fn parse_tx_refund() {
        let known = participants_abc();
        let tx = parse_transaction("Bob: Refund €27 each for all", &known).unwrap();
        assert_eq!(tx.payer, "Bob");
        // "each" → amount is multiplied by 3 (all participants) = 81
        assert!((tx.amount - 81.0).abs() < 0.01);
        assert_eq!(tx.kind, TransactionKind::Reimbursement);
        // All participants included — bridge handles payer-skipping
        assert_eq!(tx.beneficiaries.len(), 3);
    }

    #[test]
    fn parse_tx_split_with_named() {
        let known = participants_abc();
        let tx =
            parse_transaction("Bob: Paid €87 for snacks - split with Charlie", &known).unwrap();
        assert_eq!(tx.payer, "Bob");
        assert!(tx.beneficiaries.contains(&"Charlie".to_string()));
        assert!(tx.beneficiaries.contains(&"Bob".to_string()));
    }

    // ===== Relationship parser =====

    #[test]
    fn parse_rel_works_with() {
        let rel = parse_relationship("Johnny Fisher works with Christopher Peterson.").unwrap();
        assert_eq!(rel.subject, "Johnny Fisher");
        assert_eq!(rel.object, "Christopher Peterson");
        assert_eq!(rel.relation_type, "colleague");
    }

    #[test]
    fn parse_rel_colleague_of() {
        let rel =
            parse_relationship("Christopher Peterson is a colleague of Kathleen Herrera.").unwrap();
        assert_eq!(rel.subject, "Christopher Peterson");
        assert_eq!(rel.object, "Kathleen Herrera");
        assert_eq!(rel.relation_type, "colleague");
    }

    // ===== State change parser =====

    #[test]
    fn parse_state_live_in() {
        let changes = parse_state_change("I live in Alfama, Lisbon, and I start my day with a pastel de nata from the corner bakery every morning.");
        assert!(changes.iter().any(|c| c.attribute == "location"));
        assert!(changes.iter().any(|c| c.attribute.starts_with("routine:")));
    }

    #[test]
    fn parse_state_moving() {
        let changes =
            parse_state_change("I'm moving to Lower Manhattan, NYC for a 6-month work assignment.");
        assert!(!changes.is_empty());
        let loc = changes.iter().find(|c| c.attribute == "location").unwrap();
        assert!(loc.new_value.contains("Lower Manhattan"));
    }

    // ===== Preference parser =====

    #[test]
    fn parse_pref_explicit_love() {
        let pref = parse_preference(
            "I love watching the sunset from Miradouro da Senhora do Monte.",
            None,
        );
        assert!(pref.is_some());
        let p = pref.unwrap();
        assert!((p.sentiment - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_pref_from_topic() {
        let pref = parse_preference(
            "Monet's Water Lilies series captures light in a way that feels alive.",
            Some("Monet's Water Lilies"),
        );
        assert!(pref.is_some());
        let p = pref.unwrap();
        assert_eq!(p.category, "art");
    }
}
