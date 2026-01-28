// Configuration for EventGraphDB Server

use agent_db_graph::{GraphEngineConfig, integration::StorageBackend};
use std::env;
use std::path::PathBuf;
use tracing::info;

pub fn create_engine_config() -> anyhow::Result<GraphEngineConfig> {
    let mut config = GraphEngineConfig::default();

    // Configure persistent storage backend
    config.storage_backend = StorageBackend::Persistent;
    config.redb_path = PathBuf::from("./data/eventgraph.redb");
    config.redb_cache_size_mb = 128; // 128MB redb cache
    config.memory_cache_size = 10_000; // 10K memories in RAM (~20MB)
    config.strategy_cache_size = 5_000; // 5K strategies in RAM (~15MB)
    config.enable_louvain = env::var("ENABLE_LOUVAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(false);
    config.louvain_interval = env::var("LOUVAIN_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);

    // Configure semantic memory (always available)
    config.enable_semantic_memory = true;
    config.ner_workers = 2;
    config.ner_service_url =
        env::var("NER_SERVICE_URL").unwrap_or_else(|_| "http://localhost:8081/ner".to_string());
    config.ner_request_timeout_ms = env::var("NER_REQUEST_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(5_000);
    config.claim_workers = 4;
    config.embedding_workers = 2;
    config.ner_storage_path = Some(PathBuf::from("./data/ner_features.redb"));
    config.claim_storage_path = Some(PathBuf::from("./data/claims.redb"));
    config.openai_api_key = env::var("OPENAI_API_KEY").ok();
    config.llm_model = env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    config.claim_min_confidence = 0.7;
    config.claim_max_per_input = 10;
    config.enable_embedding_generation = true;

    // Create data directory if it doesn't exist
    if let Some(parent) = config.redb_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    info!("✓ Semantic memory enabled");
    info!("  NER workers: {}", config.ner_workers);
    info!("  Claim workers: {}", config.claim_workers);
    info!("  Embedding workers: {}", config.embedding_workers);

    Ok(config)
}
