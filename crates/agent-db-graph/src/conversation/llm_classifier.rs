//! LLM-based conversation message classifier.
//!
//! Uses a unified `LlmClient` to classify messages and extract structured data
//! as JSON. Rust validates numbers/currencies via `nlp::numbers` and
//! `nlp::gazetteer`. Falls back to keyword path when LLM is unavailable.

use super::types::*;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use std::collections::HashSet;
use std::sync::Arc;

/// LLM-based classifier + extractor for conversation messages.
pub struct ConversationLlmClassifier {
    client: Arc<dyn LlmClient>,
}

const SYSTEM_PROMPT: &str = concat!(
    "You classify conversation messages and extract structured data. ",
    "Output strict JSON with these fields:\n",
    "- \"category\": one of \"transaction\", \"state_change\", \"relationship\", \"preference\", \"chitchat\"\n",
    "- \"data\": category-specific extraction (see below)\n\n",
    "For \"transaction\":\n",
    "  {\"payer\": \"Name\", \"amount\": 50.0, \"currency\": \"EUR\", \"description\": \"dinner\", ",
    "\"beneficiaries\": [\"Name1\", \"Name2\"], \"kind\": \"payment|reimbursement|tip\", ",
    "\"split\": \"equal|sole|everyone\"}\n\n",
    "For \"state_change\":\n",
    "  {\"entity\": \"Name or user\", \"attribute\": \"location|routine|landmark|activity\", ",
    "\"new_value\": \"the value\"}\n\n",
    "For \"relationship\":\n",
    "  {\"subject\": \"Name\", \"object\": \"Name\", \"relation_type\": \"colleague|friend|neighbor\"}\n\n",
    "For \"preference\":\n",
    "  {\"entity\": \"Name or user\", \"item\": \"what they like\", ",
    "\"category\": \"art|music|food|general\", \"sentiment\": 0.8}\n\n",
    "For \"chitchat\": {}\n\n",
    "No markdown fences, no explanation, no other fields.",
);

impl ConversationLlmClassifier {
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self { client }
    }

    /// Classify and extract structured data from a message.
    ///
    /// Returns `None` if the LLM fails or returns unparseable output.
    pub async fn classify_and_extract(
        &self,
        content: &str,
        known_participants: &HashSet<String>,
        session_topic: Option<&str>,
    ) -> Option<ParsedMessage> {
        let mut user_prompt = format!("Message: {}", content);
        if !known_participants.is_empty() {
            let names: Vec<&str> = known_participants.iter().map(|s| s.as_str()).collect();
            user_prompt.push_str(&format!("\nKnown participants: {}", names.join(", ")));
        }
        if let Some(topic) = session_topic {
            user_prompt.push_str(&format!("\nSession topic: {}", topic));
        }

        let request = LlmRequest {
            system_prompt: SYSTEM_PROMPT.to_string(),
            user_prompt,
            temperature: 0.0,
            max_tokens: 256,
            json_mode: true,
        };

        let response = self.client.complete(request).await.ok()?;
        let json = parse_json_from_llm(&response.content)?;

        let category_str = json["category"].as_str()?;
        let data = &json["data"];

        match category_str {
            "transaction" => parse_llm_transaction(data, known_participants),
            "state_change" => parse_llm_state_change(data),
            "relationship" => parse_llm_relationship(data),
            "preference" => parse_llm_preference(data, session_topic),
            "chitchat" => Some(ParsedMessage {
                category: MessageCategory::Chitchat,
                original_content: content.to_string(),
                session_id: String::new(),
                message_index: 0,
                confidence: 1.0,
                parsed: ParsedPayload::Chitchat(content.to_string()),
            }),
            _ => None,
        }
    }
}

/// Parse and validate a transaction from LLM JSON output.
fn parse_llm_transaction(
    data: &serde_json::Value,
    known_participants: &HashSet<String>,
) -> Option<ParsedMessage> {
    let payer = data["payer"].as_str()?.to_string();

    // Validate amount using nlp::numbers
    let amount = if let Some(amt) = data["amount"].as_f64() {
        amt
    } else if let Some(amt_str) = data["amount"].as_str() {
        crate::nlp::numbers::parse_numeric_token(amt_str)?
    } else {
        return None;
    };

    // Validate currency
    let raw_currency = data["currency"].as_str().unwrap_or("USD");
    let currency = crate::nlp::gazetteer::is_currency(raw_currency)
        .unwrap_or(raw_currency)
        .to_string();

    let description = data["description"]
        .as_str()
        .unwrap_or("expense")
        .to_string();

    let beneficiaries: Vec<String> = data["beneficiaries"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let kind_str = data["kind"].as_str().unwrap_or("payment");
    let kind = match kind_str {
        "reimbursement" | "refund" => TransactionKind::Reimbursement,
        "tip" => TransactionKind::Tip,
        "transfer" => TransactionKind::Transfer,
        "debt" => TransactionKind::DebtStatement,
        _ => TransactionKind::Payment,
    };

    let split_str = data["split"].as_str().unwrap_or("equal");
    let (split_mode, scope) = match split_str {
        "everyone" | "all" => {
            let mut all: Vec<String> = known_participants.iter().cloned().collect();
            if !all.contains(&payer) {
                all.push(payer.clone());
            }
            all.sort();
            (SplitMode::Equal, ParticipantsScope::EveryoneKnown)
        },
        "sole" => (SplitMode::SoleBeneficiary, ParticipantsScope::Explicit),
        _ => (SplitMode::Equal, ParticipantsScope::Explicit),
    };

    let final_beneficiaries = if scope == ParticipantsScope::EveryoneKnown {
        let mut all: Vec<String> = known_participants.iter().cloned().collect();
        if !all.contains(&payer) {
            all.push(payer.clone());
        }
        all.sort();
        all
    } else if beneficiaries.is_empty() {
        vec![payer.clone()]
    } else {
        let mut b = beneficiaries;
        if !b.contains(&payer) {
            b.push(payer.clone());
        }
        b.sort();
        b
    };

    let tx = TransactionData {
        payer,
        beneficiaries: final_beneficiaries,
        amount,
        currency,
        description,
        split_mode,
        kind,
        participants_scope: scope,
    };

    Some(ParsedMessage {
        category: MessageCategory::Transaction,
        original_content: String::new(),
        session_id: String::new(),
        message_index: 0,
        confidence: 1.0,
        parsed: ParsedPayload::Transaction(tx),
    })
}

fn parse_llm_state_change(data: &serde_json::Value) -> Option<ParsedMessage> {
    let entity = data["entity"].as_str().unwrap_or("user").to_string();
    let attribute = data["attribute"].as_str()?.to_string();
    let new_value = data["new_value"].as_str()?.to_string();

    let sc = StateChangeData {
        entity,
        attribute,
        new_value,
        old_value: None,
    };

    Some(ParsedMessage {
        category: MessageCategory::StateChange,
        original_content: String::new(),
        session_id: String::new(),
        message_index: 0,
        confidence: 1.0,
        parsed: ParsedPayload::StateChange(sc),
    })
}

fn parse_llm_relationship(data: &serde_json::Value) -> Option<ParsedMessage> {
    let subject = data["subject"].as_str()?.to_string();
    let object = data["object"].as_str()?.to_string();
    let relation_type = data["relation_type"]
        .as_str()
        .unwrap_or("colleague")
        .to_string();

    let rel = RelationshipData {
        subject,
        object,
        relation_type,
    };

    Some(ParsedMessage {
        category: MessageCategory::Relationship,
        original_content: String::new(),
        session_id: String::new(),
        message_index: 0,
        confidence: 1.0,
        parsed: ParsedPayload::Relationship(rel),
    })
}

fn parse_llm_preference(
    data: &serde_json::Value,
    session_topic: Option<&str>,
) -> Option<ParsedMessage> {
    let entity = data["entity"].as_str().unwrap_or("user").to_string();
    let item = data["item"].as_str()?.to_string();
    let category = data["category"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| super::parsers::infer_preference_category(&item, session_topic));
    let sentiment = data["sentiment"].as_f64().unwrap_or(0.7) as f32;

    let pref = PreferenceData {
        entity,
        item,
        category,
        sentiment,
    };

    Some(ParsedMessage {
        category: MessageCategory::Preference,
        original_content: String::new(),
        session_id: String::new(),
        message_index: 0,
        confidence: 1.0,
        parsed: ParsedPayload::Preference(pref),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::MockLlmClient;

    #[tokio::test]
    async fn test_llm_classify_transaction() {
        let mock = MockLlmClient::new(vec![
            r#"{"category": "transaction", "data": {"payer": "Alice", "amount": 50.0, "currency": "EUR", "description": "dinner", "beneficiaries": ["Bob"], "kind": "payment", "split": "equal"}}"#.to_string(),
        ]);
        let classifier = ConversationLlmClassifier::new(Arc::new(mock));

        let mut participants = HashSet::new();
        participants.insert("Alice".to_string());
        participants.insert("Bob".to_string());

        let result = classifier
            .classify_and_extract(
                "Alice: Paid €50 for dinner - split with Bob",
                &participants,
                None,
            )
            .await;

        assert!(result.is_some());
        let msg = result.unwrap();
        assert_eq!(msg.category, MessageCategory::Transaction);
        if let ParsedPayload::Transaction(tx) = &msg.parsed {
            assert_eq!(tx.payer, "Alice");
            assert!((tx.amount - 50.0).abs() < 0.01);
            assert_eq!(tx.currency, "EUR");
        } else {
            panic!("Expected Transaction");
        }
    }

    #[tokio::test]
    async fn test_llm_classify_state_change() {
        let mock = MockLlmClient::new(vec![
            r#"{"category": "state_change", "data": {"entity": "user", "attribute": "location", "new_value": "Lisbon"}}"#.to_string(),
        ]);
        let classifier = ConversationLlmClassifier::new(Arc::new(mock));

        let result = classifier
            .classify_and_extract("I live in Lisbon.", &HashSet::new(), None)
            .await;

        assert!(result.is_some());
        let msg = result.unwrap();
        assert_eq!(msg.category, MessageCategory::StateChange);
    }

    #[tokio::test]
    async fn test_llm_classify_chitchat() {
        let mock = MockLlmClient::new(vec![r#"{"category": "chitchat", "data": {}}"#.to_string()]);
        let classifier = ConversationLlmClassifier::new(Arc::new(mock));

        let result = classifier
            .classify_and_extract("The music makes this trip special!", &HashSet::new(), None)
            .await;

        assert!(result.is_some());
        assert_eq!(result.unwrap().category, MessageCategory::Chitchat);
    }

    #[tokio::test]
    async fn test_llm_classify_fallback_on_bad_json() {
        let mock = MockLlmClient::new(vec!["not valid json".to_string()]);
        let classifier = ConversationLlmClassifier::new(Arc::new(mock));

        let result = classifier
            .classify_and_extract("test", &HashSet::new(), None)
            .await;

        assert!(result.is_none());
    }
}
