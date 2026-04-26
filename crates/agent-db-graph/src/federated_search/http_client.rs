use super::{
    FederatedError, FederatedSearchProvider, FederatedSearchRequest, FederatedSearchResponse,
};
use async_trait::async_trait;
use std::time::Duration;

pub struct HttpFederatedClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    timeout: Duration,
}

impl HttpFederatedClient {
    pub fn new(base_url: String, api_key: String, timeout_ms: u64) -> Self {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            timeout: Duration::from_millis(timeout_ms),
        }
    }
}

#[async_trait]
impl FederatedSearchProvider for HttpFederatedClient {
    async fn search(
        &self,
        request: &FederatedSearchRequest,
    ) -> Result<FederatedSearchResponse, FederatedError> {
        let url = format!("{}/federated/search", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .json(request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    FederatedError::Timeout(self.timeout.as_millis() as u64)
                } else {
                    FederatedError::Http(e.to_string())
                }
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            let body_truncated = if body.len() > 512 {
                &body[..512]
            } else {
                &body
            };
            return Err(FederatedError::Http(format!(
                "{}: {}",
                status, body_truncated
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| FederatedError::Http(e.to_string()))?;

        const MAX_BODY: usize = 2 * 1024 * 1024; // 2 MiB
        if bytes.len() > MAX_BODY {
            return Err(FederatedError::Http(format!(
                "response too large: {} bytes (max {})",
                bytes.len(),
                MAX_BODY
            )));
        }

        serde_json::from_slice::<FederatedSearchResponse>(&bytes)
            .map_err(|e| FederatedError::Deserialize(e.to_string()))
    }
}
