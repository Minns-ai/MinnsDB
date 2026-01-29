// Configuration for EventGraphDB Server

use agent_db_graph::{integration::StorageBackend, GraphEngineConfig};
use std::env;
use std::path::PathBuf;
use tracing::info;

pub fn create_engine_config() -> anyhow::Result<GraphEngineConfig> {
    let mut config = GraphEngineConfig::default();

    // Check if running in free profile mode
    let service_profile = env::var("SERVICE_PROFILE").unwrap_or_else(|_| "normal".to_string());
    let is_free = service_profile.to_lowercase() == "free";

    if is_free {
        info!("🆓 Running in FREE profile mode - limited features");
    } else {
        info!("✨ Running in NORMAL profile mode - full features");
    }

    // Configure persistent storage backend
    config.storage_backend = StorageBackend::Persistent;
    config.redb_path = PathBuf::from("./data/eventgraph.redb");

    // Apply profile-specific limits
    if is_free {
        // FREE PROFILE: Reduced limits
        config.redb_cache_size_mb = 64;      // 64MB redb cache (vs 256MB normal)
        config.memory_cache_size = 1_000;    // 1K memories in RAM (vs 10K normal)
        config.strategy_cache_size = 500;    // 500 strategies in RAM (vs 5K normal)
        config.max_graph_size = 50_000;      // Max 50K nodes (vs 1M normal)
        config.enable_louvain = false;       // Louvain disabled in free
        info!("  Cache limits: 64MB redb / 1K memories / 500 strategies");
        info!("  Max graph size: 50,000 nodes");
        info!("  Louvain: DISABLED");
    } else {
        // NORMAL PROFILE: Full capacity
        config.redb_cache_size_mb = 256;     // 256MB redb cache
        config.memory_cache_size = 10_000;   // 10K memories in RAM
        config.strategy_cache_size = 5_000;  // 5K strategies in RAM
        config.max_graph_size = 1_000_000;   // Max 1M nodes
        config.enable_louvain = env::var("ENABLE_LOUVAIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(true);                // Louvain enabled by default
        info!("  Cache limits: 256MB redb / 10K memories / 5K strategies");
        info!("  Max graph size: 1,000,000 nodes");
        info!("  Louvain: {}", if config.enable_louvain { "ENABLED" } else { "DISABLED" });
    }

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
