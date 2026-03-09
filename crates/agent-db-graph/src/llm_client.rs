//! Unified LLM client abstraction.
//!
//! Provides a single `LlmClient` trait with OpenAI and Anthropic
//! implementations. Replaces the per-feature LLM clients scattered across
//! `nlq::llm_hint`, `claims`, etc.

use async_trait::async_trait;

/// Request to send to an LLM.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub json_mode: bool,
}

/// Response from an LLM.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub tokens_used: u32,
}

/// Unified LLM client trait.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a completion request and return the response.
    async fn complete(&self, request: LlmRequest) -> anyhow::Result<LlmResponse>;

    /// The model name this client is configured with.
    fn model_name(&self) -> &str;
}

// ────────── OpenAI client ──────────

pub struct OpenAiLlmClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl OpenAiLlmClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiLlmClient {
    async fn complete(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        let mut body = serde_json::json!({
            "model": self.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": [
                { "role": "system", "content": request.system_prompt },
                { "role": "user", "content": request.user_prompt }
            ]
        });

        if request.json_mode {
            body["response_format"] = serde_json::json!({ "type": "json_object" });
        }

        // Retry with exponential backoff for rate limits (429) and server errors (5xx)
        let mut last_err = String::new();
        for attempt in 0..4u32 {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tracing::info!("OpenAI retry attempt {} after {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
            }

            let resp = match self
                .http
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("HTTP send error: {}", e);
                    continue;
                },
            };

            let status = resp.status();
            let json: serde_json::Value = resp.json().await?;

            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                last_err = format!(
                    "429 rate limited: {}",
                    json.to_string().chars().take(200).collect::<String>()
                );
                tracing::warn!("OpenAI 429 rate limit on attempt {}", attempt);
                continue;
            }
            if status.is_server_error() {
                last_err = format!(
                    "{} server error: {}",
                    status,
                    json.to_string().chars().take(200).collect::<String>()
                );
                tracing::warn!("OpenAI {} on attempt {}", status, attempt);
                continue;
            }
            if !status.is_success() {
                let err_msg = json["error"]["message"].as_str().unwrap_or("unknown error");
                return Err(anyhow::anyhow!("OpenAI API error {}: {}", status, err_msg));
            }

            let content = json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let tokens_used = json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;

            return Ok(LlmResponse {
                content,
                tokens_used,
            });
        }

        Err(anyhow::anyhow!(
            "OpenAI API failed after 4 attempts: {}",
            last_err
        ))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ────────── Anthropic client ──────────

pub struct AnthropicLlmClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl AnthropicLlmClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmClient for AnthropicLlmClient {
    async fn complete(&self, request: LlmRequest) -> anyhow::Result<LlmResponse> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system_prompt,
            "messages": [
                { "role": "user", "content": request.user_prompt }
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
        let content = json["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let tokens_used = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LlmResponse {
            content,
            tokens_used,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ────────── Utilities ──────────

/// Parse JSON from LLM response text with robust recovery.
///
/// Handles common LLM output quirks:
/// - Markdown code fences (`\`\`\`json ... \`\`\``)
/// - Preamble/trailing prose around the JSON
/// - Trailing commas in objects and arrays
/// - Single quotes instead of double quotes (simple cases)
/// - Unescaped control characters in strings
///
/// Returns `None` only when no valid JSON can be recovered.
pub fn parse_json_from_llm(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // 1. Strip markdown fences (```json ... ``` or ``` ... ```)
    let defenced = strip_markdown_fences(trimmed);

    // 2. Try direct parse first (fast path)
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(defenced) {
        return Some(v);
    }

    // 3. Extract JSON substring: find the first `{` or `[` and match its closing bracket
    if let Some(extracted) = extract_json_substring(defenced) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(extracted) {
            return Some(v);
        }

        // 4. Try fixing trailing commas
        let fixed = fix_trailing_commas(extracted);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&fixed) {
            return Some(v);
        }

        // 5. Try fixing single quotes → double quotes (only outside existing double-quoted strings)
        let requoted = single_to_double_quotes(&fixed);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&requoted) {
            return Some(v);
        }

        // 6. Sanitize control characters in string values and retry
        let sanitized = sanitize_control_chars(&fixed);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&sanitized) {
            return Some(v);
        }
    }

    None
}

/// Strip markdown code fences from LLM output.
fn strip_markdown_fences(text: &str) -> &str {
    // Handle ```json\n...\n``` or ```\n...\n```
    if let Some(rest) = text.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.strip_prefix("JSON").unwrap_or(rest);
        let rest = rest.trim_start_matches([' ', '\t']);
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        let rest = rest.trim_end();
        rest.strip_suffix("```").unwrap_or(rest).trim()
    } else {
        text
    }
}

/// Find the first `{` or `[` and return the balanced substring through
/// its matching close bracket. Skips over string literals.
fn extract_json_substring(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut start = None;

    // Find first { or [
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'{' || b == b'[' {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let open_char = bytes[start];
    let close_char = if open_char == b'{' { b'}' } else { b']' };

    // Walk forward, tracking depth and skipping strings
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, &b) in bytes[start..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == open_char {
            depth += 1;
        } else if b == close_char {
            depth -= 1;
            if depth == 0 {
                return Some(&text[start..start + i + 1]);
            }
        }
    }
    // Unbalanced — return from start to end as best effort
    Some(&text[start..])
}

/// Remove trailing commas before `}` or `]`.
/// E.g. `{"a": 1, "b": 2,}` → `{"a": 1, "b": 2}`
fn fix_trailing_commas(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escape_next = false;
    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        if escape_next {
            escape_next = false;
            result.push(c);
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            result.push(c);
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            result.push(c);
            continue;
        }
        if in_string {
            result.push(c);
            continue;
        }
        if c == ',' {
            // Look ahead past whitespace for } or ]
            let rest = &chars[i + 1..];
            let next_non_ws = rest.iter().find(|&&ch| !ch.is_whitespace());
            if matches!(next_non_ws, Some('}') | Some(']')) {
                continue; // skip the trailing comma
            }
        }
        result.push(c);
    }
    result
}

/// Convert single-quoted strings to double-quoted (simple heuristic).
/// Only converts quotes that appear to be JSON string delimiters,
/// not apostrophes within double-quoted strings.
fn single_to_double_quotes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_double = false;
    let mut in_single = false;
    let mut escape_next = false;

    for c in text.chars() {
        if escape_next {
            escape_next = false;
            result.push(c);
            continue;
        }
        if c == '\\' && (in_double || in_single) {
            escape_next = true;
            result.push(c);
            continue;
        }
        if c == '"' && !in_single {
            in_double = !in_double;
            result.push(c);
            continue;
        }
        if c == '\'' && !in_double {
            in_single = !in_single;
            result.push('"'); // convert to double quote
            continue;
        }
        result.push(c);
    }
    result
}

/// Replace unescaped control characters (0x00-0x1F except \t \n \r)
/// inside JSON strings with spaces.
fn sanitize_control_chars(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escape_next = false;

    for c in text.chars() {
        if escape_next {
            escape_next = false;
            result.push(c);
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            result.push(c);
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            result.push(c);
            continue;
        }
        if in_string && c.is_control() && c != '\t' && c != '\n' && c != '\r' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_from_llm_plain() {
        let json = r#"{"category": "transaction", "amount": 50}"#;
        let result = parse_json_from_llm(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["category"], "transaction");
    }

    #[test]
    fn test_parse_json_from_llm_fenced() {
        let json = "```json\n{\"category\": \"state_change\"}\n```";
        let result = parse_json_from_llm(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["category"], "state_change");
    }

    #[test]
    fn test_parse_json_from_llm_bare_fences() {
        let json = "```\n{\"ok\": true}\n```";
        let result = parse_json_from_llm(json);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_json_from_llm_empty() {
        assert!(parse_json_from_llm("").is_none());
        assert!(parse_json_from_llm("  ").is_none());
    }

    #[test]
    fn test_parse_json_from_llm_invalid() {
        assert!(parse_json_from_llm("not json at all").is_none());
    }

    #[test]
    fn test_parse_json_preamble_text() {
        let text = "Here is the extracted JSON:\n{\"facts\": [{\"subject\": \"Alice\"}]}";
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["facts"][0]["subject"], "Alice");
    }

    #[test]
    fn test_parse_json_trailing_text() {
        let text = "{\"answer\": \"Tokyo\"}\n\nLet me know if you need anything else!";
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["answer"], "Tokyo");
    }

    #[test]
    fn test_parse_json_preamble_and_trailing() {
        let text = "Sure! Here's the result:\n\n{\"status\": \"ok\"}\n\nHope this helps.";
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["status"], "ok");
    }

    #[test]
    fn test_parse_json_trailing_commas() {
        let text = r#"{"a": 1, "b": [1, 2, 3,], "c": "hello",}"#;
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["a"], 1);
        assert_eq!(result["b"].as_array().unwrap().len(), 3);
        assert_eq!(result["c"], "hello");
    }

    #[test]
    fn test_parse_json_single_quotes() {
        let text = "{'category': 'state_change', 'confidence': 0.9}";
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["category"], "state_change");
    }

    #[test]
    fn test_parse_json_nested_trailing_commas() {
        let text = r#"{"facts": [{"s": "A", "p": "B",}, {"s": "C", "p": "D",},],}"#;
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["facts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_parse_json_fenced_with_preamble() {
        let text = "Here you go:\n```json\n{\"ok\": true}\n```\nDone!";
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn test_parse_json_array_response() {
        let text = "[{\"action\": \"add\"}, {\"action\": \"skip\"}]";
        let result = parse_json_from_llm(text).unwrap();
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_parse_json_array_with_preamble() {
        let text = "The classification results:\n[{\"action\": \"add\"}]";
        let result = parse_json_from_llm(text).unwrap();
        assert!(result.is_array());
    }

    #[test]
    fn test_strip_markdown_fences_json_tag() {
        assert_eq!(strip_markdown_fences("```json\n{}\n```"), "{}");
    }

    #[test]
    fn test_strip_markdown_fences_bare() {
        assert_eq!(strip_markdown_fences("```\n{}\n```"), "{}");
    }

    #[test]
    fn test_strip_markdown_fences_none() {
        assert_eq!(strip_markdown_fences("{\"a\": 1}"), "{\"a\": 1}");
    }

    #[test]
    fn test_fix_trailing_commas_in_string_preserved() {
        // Commas inside strings should NOT be removed
        let text = r#"{"msg": "hello, world,"}"#;
        let fixed = fix_trailing_commas(text);
        let v: serde_json::Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(v["msg"], "hello, world,");
    }

    #[test]
    fn test_extract_json_substring_with_prose() {
        let text = "The answer is: {\"x\": 1} and that's it.";
        let extracted = extract_json_substring(text).unwrap();
        assert_eq!(extracted, "{\"x\": 1}");
    }

    #[test]
    fn test_extract_json_substring_nested() {
        let text = "Result: {\"a\": {\"b\": 2}} done";
        let extracted = extract_json_substring(text).unwrap();
        assert_eq!(extracted, "{\"a\": {\"b\": 2}}");
    }

    #[test]
    fn test_extract_json_with_braces_in_strings() {
        let text = r#"{"msg": "use {braces} here", "ok": true}"#;
        let extracted = extract_json_substring(text).unwrap();
        assert_eq!(extracted, text);
        let v: serde_json::Value = serde_json::from_str(extracted).unwrap();
        assert_eq!(v["ok"], true);
    }
}
