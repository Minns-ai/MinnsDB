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
        Self {
            client: reqwest::Client::new(),
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

        let response = tokio::time::timeout(self.timeout, async {
            self.client
                .post(&url)
                .header("X-API-Key", &self.api_key)
                .json(request)
                .send()
                .await
                .map_err(|e| FederatedError::Http(e.to_string()))
        })
        .await
        .map_err(|_| FederatedError::Timeout(self.timeout.as_millis() as u64))??;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(FederatedError::Http(format!("{}: {}", status, body)));
        }

        response
            .json::<FederatedSearchResponse>()
            .await
            .map_err(|e| FederatedError::Deserialize(e.to_string()))
    }
}
